// Light tree traversal: a stochastic walk down a 4-wide tree with one light sample per query, and the PMF walk back
// up for MIS. Each node holds the cluster of "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"
// (Tokuyoshi, Ikeda, Kulkarni and Harada 2024, Section 3.1): flux, mean, a vMF fit of the emission and the spatial
// deviation. A node's importance takes one of two forms:
// - by default, not their SG lighting integral but a cheap bound over the cone to its bounding sphere, oriented by
//   the normal, except for the glossy lobe, which weighs the node as their SG light, widened by its spatial deviation;
// - under SG_LIGHT_TREE, their SG lighting (Sections 3 to 6, SGLighting.slang): the cluster as an SG light, the
//   diffuse importance by interpolation (Eq. 7) and the glossy one by filtered anisotropic GGX (Eq. 12), with their
//   point light forms for small, distant clusters wholly above the horizon. More accurate, far more code.
// LightTreeBuild.slang builds the tree on the GPU every frame, Morton sorted after Lin and Yuksel 2020.

#include "LightTreeInterop.h"

#ifdef SG_LIGHT_TREE
import SGLighting;
#endif

static const float LIGHT_TREE_INV_PI = 0.318309886183791;
static const float LIGHT_TREE_TWO_PI = 6.28318530717959;

// Accessors for the packed node (see LightTreeInterop.h).

uint LightTreeChildOrLight(GpuLightTreeNode n)
{
    return n.packed.x;
}

float LightTreeRadius(GpuLightTreeNode n)
{
    return asfloat(n.packed.w);
}

// vMF sharpness of the node's emission (0 for isotropic emitters).
float LightTreeSharpness(GpuLightTreeNode n)
{
    return f16tof32(n.packed.z & 0xFFFF);
}

// Flux weighted standard deviation of the cluster's light positions.
float LightTreeStdDev(GpuLightTreeNode n)
{
    return f16tof32(n.packed.z >> 16);
}

// Octahedral snorm16x2 to a unit vector, x in the low half.
float3 LightTreeAxis(GpuLightTreeNode n)
{
    uint bits = n.packed.y;
    float2 e = max(float2(int2(int(bits << 16) >> 16, int(bits) >> 16)) * (1.0 / 32767.0), -1.0);
    float3 v = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-v.z);
    v.x += v.x >= 0.0 ? -t : t;
    v.y += v.y >= 0.0 ? -t : t;
    return normalize(v);
}

struct LightTreeShadingContext
{
    float3 position;
    float3 normal;
    float diffuseWeight;   // reflectance of the diffuse lobe (e.g. luminance of the albedo)
    float specularWeight;  // reflectance of the glossy lobe (e.g. luminance of the Fresnel term)
    // A transmissive surface's diffuse-like lobes on the far side of normal, as a diffuse lobe facing -normal.
    // 0 for a surface that only reflects, which leaves out every node wholly below the tangent plane.
    float backDiffuseWeight;
#ifdef SG_LIGHT_TREE
    GlossyContext glossy;
    // SG light sharpness from which a node wholly above the horizon takes the far-field (point light)
    // form of the diffuse and of the glossy importance.
    float diffuseFarFieldSharpness;
    float glossyFarFieldSharpness;
#else
    float3 reflectDir;     // the glossy lobe's axis, the view direction mirrored about normal
    float glossySharpness; // the glossy lobe as a vMF around reflectDir
#endif
};

// T: the tangent along which alphaX lies, perpendicular to N, the bitangent being cross(N, T); only SG_LIGHT_TREE's
// anisotropic glossy importance reads it.
// diffuseOnlyAlpha: GGX alpha (roughness squared) on both axes at or above which the glossy lobe counts as
// diffuse, wide enough that the diffuse term ranks clusters about as well.
// farFieldSharpness (SG_LIGHT_TREE only): SG light sharpness (squared distance over spatial variance) from which a
// cluster is small enough, seen from here, to be treated as a point light. FLT_MAX turns the far field off.
LightTreeShadingContext MakeLightTreeShadingContext(float3 position, float3 T, float3 N, float3 V, float diffuseWeight,
                                                    float specularWeight, float backDiffuseWeight, float alphaX,
                                                    float alphaY, float diffuseOnlyAlpha, float farFieldSharpness)
{
    LightTreeShadingContext ctx;
    ctx.position = position;
    ctx.normal = N;
    ctx.backDiffuseWeight = backDiffuseWeight;
    float specular = dot(V, N) > 0.0 ? specularWeight : 0.0;
    // Depends only on the shading point, so SampleLightTree and LightTreePmf still agree.
    bool roughLobe = min(alphaX, alphaY) >= diffuseOnlyAlpha;
    ctx.diffuseWeight = roughLobe ? diffuseWeight + specular : diffuseWeight;
    ctx.specularWeight = roughLobe ? 0.0 : specular;

#ifdef SG_LIGHT_TREE
    ctx.glossy = MakeGlossyContext(T, cross(N, T), N, V, alphaX, alphaY);
    float jjtMax = max(ctx.glossy.jjt.x, ctx.glossy.jjt.z);
    float alpha2Min = min(ctx.glossy.alpha2.x, ctx.glossy.alpha2.y);
    ctx.diffuseFarFieldSharpness = farFieldSharpness;
    ctx.glossyFarFieldSharpness = max(farFieldSharpness, 20.0 * jjtMax / alpha2Min);
#else
    // GGX's reflected directions as a vMF, of the rougher axis so that it bounds the lobe: the NDF's sharpness
    // 2 / alpha^2 over half vectors halves for the reflected directions.
    ctx.reflectDir = 2.0 * dot(V, N) * N - V;
    float alpha = clamp(max(alphaX, alphaY), 1e-3, 1.0);
    float alpha2 = alpha * alpha;
    ctx.glossySharpness = (1.0 - alpha2) / (2.0 * alpha2);
#endif
    return ctx;
}

#ifdef SG_LIGHT_TREE

// Section 3.2: incoming radiance from the cluster as an SG light,
// L(x, o) ~ Phi g(o; -nu, lambda) g(o; dir, d^2 / sigma^2) / (2 pi sigma^2 int g(w; -nu, lambda) dw).
// toLight is the vector from the shading point to the cluster mean.
SGLight MakeSGLight(GpuLightTreeNode node, float3 toLight, float3 n)
{
    float d2 = dot(toLight, toLight);
    // One rsqrt rather than a sqrt and three divisions.
    float3 dir = d2 > 0.0 ? toLight * rsqrt(d2) : n;

    // Hybrid variance: blend towards the conservative bound when the cluster centre is behind the surface.
    float radius = LightTreeRadius(node);
    float stdDev = LightTreeStdDev(node);
    float c = max(dot(n, -dir), 0.0);
    float sigma2 = lerp(stdDev * stdDev, 0.5 * radius * radius, c);
    // A single point light has no spatial deviation: the floor makes it a very sharp SG, whose amplitude times
    // integral still comes to the flux over d^2.
    sigma2 = max(sigma2, 1e-8 * d2 + 1e-12);

    float invSigma2 = 1.0 / sigma2;
    float s = d2 * invSigma2;

    // The node amplitude already holds Phi / (2 pi int g(w; -nu, lambda) dw).
    SGLight sg;
    float logScale;
    SGProduct(-LightTreeAxis(node), LightTreeSharpness(node), dir, s, sg.axis, sg.sharpness, logScale);
    sg.amplitude = node.meanAmplitude.w * exp(logScale) * invSigma2;
    return sg;
}

// Eq. 6 with the diffuse (Eq. 7) and glossy (Eq. 12) approximations, weighted by lobe reflectance.
float LightTreeNodeImportance(GpuLightTreeNode node, LightTreeShadingContext ctx)
{
    // A bounding sphere entirely below the tangent plane has a true integral of zero, unless the surface
    // transmits. Either test rejects the empty nodes that pad a group, whose radius is -FLT_MAX.
    float3 toMean = node.meanAmplitude.xyz - ctx.position;
    float height = dot(toMean, ctx.normal);
    float radius = LightTreeRadius(node);
    bool front = height >= -radius;
    bool back = ctx.backDiffuseWeight > 0.0 && height <= radius;
    if (!front && !back)
        return 0.0;

    // A node behind a transmissive surface takes the conservative variance as seen from that side.
    SGLight sg = MakeSGLight(node, toMean, back && height < 0.0 ? -ctx.normal : ctx.normal);

    bool aboveHorizon = height > radius;

    float importance = 0.0;
    if (front && ctx.diffuseWeight > 0.0)
    {
        if (aboveHorizon && sg.sharpness >= ctx.diffuseFarFieldSharpness)
            importance += ctx.diffuseWeight * SGDiffuseImportanceFarField(sg, ctx.normal);
        else
            importance += ctx.diffuseWeight * SGDiffuseImportance(sg, ctx.normal);
    }
    if (front && ctx.specularWeight > 0.0)
    {
        if (aboveHorizon && sg.sharpness >= ctx.glossyFarFieldSharpness)
            importance += ctx.specularWeight * SGGlossyImportanceFarField(sg, ctx.glossy);
        else
            importance += ctx.specularWeight * SGGlossyImportance(sg, ctx.glossy);
    }
    if (back)
        importance += ctx.backDiffuseWeight * SGDiffuseImportance(sg, -ctx.normal);

    // Guard against float underflow so that Eq. 1 (I > 0 whenever the integral is > 0) still holds.
    return max(importance, 1e-30);
}

#else

// The largest cosine between an axis and the directions of a cone, given the cosine to the cone's centre and the
// cone's half angle as sine and cosine: 1 if the axis is inside the cone.
float LightTreeConeCosine(float cosAxis, float sinCone, float cosCone)
{
    float sinAxis = sqrt(saturate(1.0 - cosAxis * cosAxis));
    return cosAxis >= cosCone ? 1.0 : cosAxis * cosCone + sinAxis * sinCone;
}

// The light the node's cluster sends to the shading point's lobes: the node's vMF emission lobe at its strongest over
// the cone of directions to its bounding sphere, over the squared distance (at least the radius), times the diffuse
// lobes as cosine / pi at their strongest over the cone, and the glossy lobe weighed by the cluster as an SG light.
// The emission lobe is left as it is: a one-sided emitter's sharpness is at most about 1.8, already broad.
float LightTreeNodeImportance(GpuLightTreeNode node, LightTreeShadingContext ctx)
{
    // A bounding sphere entirely below the tangent plane gets nothing, unless the surface transmits. Either test
    // rejects the empty nodes that pad a group, whose radius is -FLT_MAX.
    float3 toMean = node.meanAmplitude.xyz - ctx.position;
    float height = dot(toMean, ctx.normal);
    float radius = LightTreeRadius(node);
    bool front = height >= -radius;
    bool back = ctx.backDiffuseWeight > 0.0 && height <= radius;
    if (!front && !back)
        return 0.0;

    // The cone of directions to the sphere; from inside it, every direction.
    float d2 = dot(toMean, toMean);
    float invD = rsqrt(max(d2, 1e-20));
    float3 dir = toMean * invD;
    float sinCone = min(radius * invD, 1.0);
    float cosCone = radius * radius < d2 ? sqrt(1.0 - sinCone * sinCone) : -1.0;

    // The amplitude holds the flux over the emission vMF's integral, so this is the intensity towards the
    // point over 2 pi, the same scale for every node.
    float emission = exp(LightTreeSharpness(node) * (LightTreeConeCosine(-dot(LightTreeAxis(node), dir), sinCone, cosCone) - 1.0));
    float intensity = node.meanAmplitude.w * emission / max(d2, radius * radius);

    float cosNormal = dot(ctx.normal, dir);
    float lobes = 0.0;
    if (front)
    {
        lobes += ctx.diffuseWeight * LIGHT_TREE_INV_PI * max(LightTreeConeCosine(cosNormal, sinCone, cosCone), 0.0);
        if (ctx.specularWeight > 0.0)
        {
            // Seen from the point, the cluster spreads over directions like an SG of sharpness d^2 / sigma^2 around
            // the one to its mean, which Tokuyoshi et al. use for the SG light. Convolved with the glossy vMF the
            // sharpnesses combine harmonically, k' = k d^2 / (d^2 + k sigma^2), so a cluster that is wide for its
            // distance widens the lobe it is weighed with, and one around the point flattens it. The normalised
            // vMF's k' / (2 pi (1 - exp(-2 k'))) is approximated by sqrt(k'^2 + 1/4) / (2 pi), within 11%.
            float sigma = LightTreeStdDev(node);
            float distance2 = max(d2, 1e-20);
            float k = ctx.glossySharpness * distance2 / (distance2 + ctx.glossySharpness * sigma * sigma);
            lobes += ctx.specularWeight * (sqrt(k * k + 0.25) / LIGHT_TREE_TWO_PI) *
                     exp(k * (dot(ctx.reflectDir, dir) - 1.0));
        }
    }
    if (back)
        lobes += ctx.backDiffuseWeight * LIGHT_TREE_INV_PI * max(LightTreeConeCosine(-cosNormal, sinCone, cosCone), 0.0);

    // Guard against float underflow so that a node that may contribute keeps a non-zero probability.
    return max(intensity * lobes, 1e-30);
}

#endif // SG_LIGHT_TREE

float4 LightTreeGroupImportance(StructuredBuffer<GpuLightTreeNode> nodes, uint first, LightTreeShadingContext ctx,
                                out uint4 childOrLight)
{
    GpuLightTreeNode group[LIGHT_TREE_WIDTH];
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_WIDTH; ++i)
        group[i] = nodes[first + i];

    [unroll]
    for (uint i = 0; i < LIGHT_TREE_WIDTH; ++i)
        childOrLight[i] = LightTreeChildOrLight(group[i]);
    float4 importance = float4(0.0, 0.0, 0.0, 0.0);
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_WIDTH; ++i)
    {
        GpuLightTreeNode node = i == 0 ? group[0] : i == 1 ? group[1] : i == 2 ? group[2] : group[3];
        float nodeImportance = LightTreeNodeImportance(node, ctx);
        importance = float4(i == 0 ? nodeImportance : importance.x, i == 1 ? nodeImportance : importance.y,
                            i == 2 ? nodeImportance : importance.z, i == 3 ? nodeImportance : importance.w);
    }
    return importance;
}

float LightTreeImportanceSum(float4 importance)
{
    return (importance.x + importance.y) + (importance.z + importance.w);
}

static const float LIGHT_TREE_ONE_MINUS_EPSILON = 0.99999994;

// Picks child i of a group with probability importance[i] / sum, and rescales u to [0, 1) within its
// interval. Children with zero importance (empty ones among them) are never picked, also when
// rounding puts u * sum past the last interval.
uint LightTreePickChild(float4 importance, float sum, inout float u)
{
    float target = u * sum;
    float before = 0.0;
    uint chosen = 0;
    float chosenBefore = 0.0;
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_WIDTH; ++i)
    {
        if (importance[i] > 0.0 && before <= target)
        {
            chosen = i;
            chosenBefore = before;
        }
        before += importance[i];
    }
    u = min((target - chosenBefore) / importance[chosen], LIGHT_TREE_ONE_MINUS_EPSILON);
    return chosen;
}

// Stochastic traversal with one light sample per query, then a pick within the leaf in proportion to
// flux. u is a uniform random number in [0, 1) which is rescaled at each level.
// Returns false when no light can contribute.
bool SampleLightTree(StructuredBuffer<GpuLightTreeNode> nodes, StructuredBuffer<GpuLightTreeLeafEntry> leafLights,
                     LightTreeShadingContext ctx, float u, out uint lightIndex, out float pmf)
{
    pmf = 1.0;
    lightIndex = 0;

    uint first = 0;
    uint leaf;
    [loop]
    while (true)
    {
        uint4 childOrLight;
        float4 importance = LightTreeGroupImportance(nodes, first, ctx, childOrLight);
        float sum = LightTreeImportanceSum(importance);
        if (!(sum > 0.0))
            return false;

        uint chosen = LightTreePickChild(importance, sum, u);
        pmf *= importance[chosen] / sum;
        uint next = childOrLight[chosen];
        if ((next & LIGHT_TREE_LEAF_BIT) != 0)
        {
            leaf = next;
            break;
        }
        first = next;
    }

    // The leaf's entries are loaded together; past its count they repeat its last one.
    uint begin = leaf & LIGHT_TREE_LEAF_FIRST_MASK;
    uint last = (leaf >> LIGHT_TREE_LEAF_COUNT_SHIFT) & (LIGHT_TREE_MAX_LEAF_LIGHTS - 1);
    GpuLightTreeLeafEntry entries[LIGHT_TREE_MAX_LEAF_LIGHTS];
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_MAX_LEAF_LIGHTS; ++i)
        entries[i] = leafLights[begin + min(i, last)];

    uint chosen = 0;
    float before = 0.0;
    [unroll]
    for (uint i = 1; i < LIGHT_TREE_MAX_LEAF_LIGHTS; ++i)
    {
        before += entries[i - 1].pmf;
        if (i <= last && u >= before)
            chosen = i;
    }
    lightIndex = entries[chosen].index;
    pmf *= entries[chosen].pmf;
    return true;
}

// Probability of SampleLightTree selecting lightIndex. Needed for MIS when BRDF sampling hits an emitter.
// Walks from the light's leaf up to the root using the parent link of each group.
float LightTreePmf(StructuredBuffer<GpuLightTreeNode> nodes, StructuredBuffer<uint> parents,
                   StructuredBuffer<GpuLightTreeLeafEntry> leafOfLight, LightTreeShadingContext ctx, uint lightIndex)
{
    GpuLightTreeLeafEntry leaf = leafOfLight[lightIndex];
    uint index = leaf.index;
    float pmf = leaf.pmf;
    [loop]
    while (true)
    {
        // The group is known from the node's index alone, so its loads need not wait for the parent.
        uint first = index & ~(LIGHT_TREE_WIDTH - 1);
        uint parent = parents[first / LIGHT_TREE_WIDTH];
        uint4 childOrLight;
        float4 importance = LightTreeGroupImportance(nodes, first, ctx, childOrLight);
        float sum = LightTreeImportanceSum(importance);
        if (!(sum > 0.0))
            return 0.0;
        pmf *= importance[index - first] / sum;
        if (first == 0)
            return pmf;
        index = parent;
    }
    return pmf;
}
