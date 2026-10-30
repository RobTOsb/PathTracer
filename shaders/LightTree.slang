// Light tree traversal after "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting" (Tokuyoshi,
// Ikeda, Kulkarni and Harada 2024): their clusters (Section 3.1), but a cheap bound on each node's contribution in
// place of their SG lighting integrals, which keeps the walk small in code and registers. The tree is built on the
// CPU by LightTreeBuilder.

#include "LightTreeInterop.h"

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
    float3 reflectDir;     // the glossy lobe's axis, the view direction mirrored about normal
    float glossySharpness; // the glossy lobe as a vMF around reflectDir
    float glossyNorm;      // that vMF's normalisation, so that it integrates to 1 over the sphere
};

// diffuseOnlyAlpha: GGX alpha (roughness squared) on both axes at or above which the glossy lobe counts as
// diffuse, wide enough that the diffuse term ranks clusters about as well.
LightTreeShadingContext MakeLightTreeShadingContext(float3 position, float3 N, float3 V, float diffuseWeight,
                                                    float specularWeight, float backDiffuseWeight, float alphaX,
                                                    float alphaY, float diffuseOnlyAlpha)
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

    // GGX's reflected directions as a vMF, of the rougher axis so that it bounds the lobe: the NDF's sharpness
    // 2 / alpha^2 over half vectors halves for the reflected directions.
    ctx.reflectDir = 2.0 * dot(V, N) * N - V;
    float alpha = clamp(max(alphaX, alphaY), 1e-3, 1.0);
    float alpha2 = alpha * alpha;
    float k = (1.0 - alpha2) / (2.0 * alpha2);
    ctx.glossySharpness = k;
    ctx.glossyNorm = k < 1e-3 ? 0.5 / LIGHT_TREE_TWO_PI : k / (LIGHT_TREE_TWO_PI * (1.0 - exp(-2.0 * k)));
    return ctx;
}

// The largest cosine between an axis and the directions of a cone, given the cosine to the cone's centre and the
// cone's half angle as sine and cosine: 1 if the axis is inside the cone.
float LightTreeConeCosine(float cosAxis, float sinCone, float cosCone)
{
    float sinAxis = sqrt(saturate(1.0 - cosAxis * cosAxis));
    return cosAxis >= cosCone ? 1.0 : cosAxis * cosCone + sinAxis * sinCone;
}

// A bound on the light the node's cluster sends to the shading point's lobes, from the cone of directions to its
// bounding sphere: the node's vMF emission lobe at its strongest over the cone, over the squared distance (at
// least the radius), times the receiving lobes at their strongest over the cone, the diffuse ones as cosine / pi
// and the glossy one as its vMF. Two exps, no SG integrals, a fraction of the code of the paper's Eq. 6.
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
            lobes += ctx.specularWeight * ctx.glossyNorm *
                     exp(ctx.glossySharpness * (LightTreeConeCosine(dot(ctx.reflectDir, dir), sinCone, cosCone) - 1.0));
    }
    if (back)
        lobes += ctx.backDiffuseWeight * LIGHT_TREE_INV_PI * max(LightTreeConeCosine(-cosNormal, sinCone, cosCone), 0.0);

    // Guard against float underflow so that a node that may contribute keeps a non-zero probability.
    return max(intensity * lobes, 1e-30);
}

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
