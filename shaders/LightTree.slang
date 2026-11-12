// Light tree traversal: a stochastic walk down a 4-wide tree with one light sample per query, and for MIS the PMF of a
// known light, by the same walk down following its path (WalkLightTree). Each node holds the cluster of "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"
// (Tokuyoshi, Ikeda, Kulkarni and Harada 2024, Section 3.1): flux, mean, a vMF fit of the emission and the spatial
// deviation. A node's importance takes one of two forms:
// - by default, not their SG lighting integral but a cheap bound over the cone to its bounding sphere, oriented by
//   the normal, except for the glossy lobe, which weighs the node as their SG light, widened by its spatial deviation;
// - under SG_LIGHT_TREE, their SG lighting (Sections 3 to 6, SGLighting.slang): the cluster as an SG light, the
//   diffuse importance by interpolation (Eq. 7) and the glossy one by filtered anisotropic GGX (Eq. 12), with their
//   point light forms for small, distant clusters wholly above the horizon. More accurate, far more code.
// LightTreeBuild.slang builds the tree on the GPU every frame, Morton sorted after Lin and Yuksel 2020. Its links are
// implicit (LightTreeShape) and its nodes 16 bytes (GpuLightTreeNode), a group of four half a cache line, so that
// the walks, which take a line per level in every lane, move as little as they can through the L2.

#include "LightTreeInterop.h"

#ifdef SG_LIGHT_TREE
import SGLighting;
#endif

static const float LIGHT_TREE_INV_PI = 0.318309886183791;
static const float LIGHT_TREE_TWO_PI = 6.28318530717959;

// A node unpacked for one shading point (see GpuLightTreeNode), the mean already relative to it.
struct LightTreeNode
{
    float3 toMean; // from the shading point to the flux weighted mean of the light positions
    float amplitude; // the node constant part of the SG light amplitude; 0 for an empty node
    float radius; // bounding sphere radius around the mean
    float sharpness; // vMF sharpness of the emission (0 for isotropic emitters)
    float stdDev; // flux weighted standard deviation of the light positions
    uint axisBits; // vMF axis, octahedral snorm8x2 (see LightTreeAxis)
};

// Octahedral snorm8x2 to a unit vector, x in the low byte.
float3 LightTreeAxis(LightTreeNode n)
{
    uint bits = n.axisBits;
    float2 e = max(float2(int2(int(bits << 24) >> 24, int(bits << 16) >> 24)) * (1.0 / 127.0), -1.0);
    float3 v = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-v.z);
    v.x += v.x >= 0.0 ? -t : t;
    v.y += v.y >= 0.0 ? -t : t;
    return normalize(v);
}

struct LightTreeShadingContext
{
    float3 position; // less the tree's origin (FrameConstants::lightTreeOrigin), which the node means count from
    float nodeStep; // of the node means' quantisation (FrameConstants::lightTreeStep)
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
// origin and step: the tree's node mean quantisation (FrameConstants::lightTreeOrigin and lightTreeStep).
LightTreeShadingContext MakeLightTreeShadingContext(float3 position, float3 T, float3 N, float3 V, float diffuseWeight,
                                                    float specularWeight, float backDiffuseWeight, float alphaX,
                                                    float alphaY, float diffuseOnlyAlpha, float farFieldSharpness,
                                                    float3 origin, float step)
{
    LightTreeShadingContext ctx;
    ctx.position = position - origin;
    ctx.nodeStep = step;
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

// The node as the shading point of ctx sees it.
LightTreeNode UnpackLightTreeNode(uint4 p, LightTreeShadingContext ctx)
{
    LightTreeNode n;
    n.toMean = float3(p.x & 0xFFFF, p.x >> 16, p.y & 0xFFFF) * ctx.nodeStep - ctx.position;
    n.radius = f16tof32(p.y >> 16);
    n.amplitude = asfloat(p.z << 16);
    n.axisBits = p.z >> 16;
    n.sharpness = f16tof32(p.w & 0xFFFF);
    n.stdDev = f16tof32(p.w >> 16);
    return n;
}

#ifdef SG_LIGHT_TREE

// Section 3.2: incoming radiance from the cluster as an SG light,
// L(x, o) ~ Phi g(o; -nu, lambda) g(o; dir, d^2 / sigma^2) / (2 pi sigma^2 int g(w; -nu, lambda) dw).
// toLight is the vector from the shading point to the cluster mean.
SGLight MakeSGLight(LightTreeNode node, float3 toLight, float3 n)
{
    float d2 = dot(toLight, toLight);
    // One rsqrt rather than a sqrt and three divisions.
    float3 dir = d2 > 0.0 ? toLight * rsqrt(d2) : n;

    // Hybrid variance: blend towards the conservative bound when the cluster centre is behind the surface.
    float radius = node.radius;
    float stdDev = node.stdDev;
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
    SGProduct(-LightTreeAxis(node), node.sharpness, dir, s, sg.axis, sg.sharpness, logScale);
    sg.amplitude = node.amplitude * exp(logScale) * invSigma2;
    return sg;
}

// Eq. 6 with the diffuse (Eq. 7) and glossy (Eq. 12) approximations, weighted by lobe reflectance.
float LightTreeNodeImportance(LightTreeNode node, LightTreeShadingContext ctx)
{
    // Nothing for the empty nodes that pad a group, and for a bounding sphere entirely below the tangent plane,
    // whose true integral is zero, unless the surface transmits.
    if (!(node.amplitude > 0.0))
        return 0.0;
    float3 toMean = node.toMean;
    float height = dot(toMean, ctx.normal);
    float radius = node.radius;
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
float LightTreeNodeImportance(LightTreeNode node, LightTreeShadingContext ctx)
{
    // Nothing for the empty nodes that pad a group, and for a bounding sphere entirely below the tangent plane,
    // unless the surface transmits.
    if (!(node.amplitude > 0.0))
        return 0.0;
    float3 toMean = node.toMean;
    float height = dot(toMean, ctx.normal);
    float radius = node.radius;
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
    float emission = exp(node.sharpness * (LightTreeConeCosine(-dot(LightTreeAxis(node), dir), sinCone, cosCone) - 1.0));
    float intensity = node.amplitude * emission / max(d2, radius * radius);

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
            float sigma = node.stdDev;
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

// A loop, not unrolled: the node importance is the largest code in the path tracer, and four copies of it per
// walk, in every walk the shader inlines, overflow the instruction cache (Nsight's "No Instructions" stall). The
// group's four nodes share a 64 byte half line, so after the first load the others hit in L1.
float4 LightTreeGroupImportance(StructuredBuffer<GpuLightTreeNode> nodes, uint first, LightTreeShadingContext ctx)
{
    float4 importance = float4(0.0, 0.0, 0.0, 0.0);
    [loop]
    for (uint i = 0; i < LIGHT_TREE_WIDTH; ++i)
    {
        float nodeImportance = LightTreeNodeImportance(UnpackLightTreeNode(nodes[first + i].packed, ctx), ctx);
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

// The tree's shape, which is all its links are made of (LightTreeLayout in LightTreeBuild.slang): the leaves are the
// lights in Morton order, leafSize at a time, all on level leafLevel; each level above has a quarter as many nodes,
// rounded up, and every level is padded to whole groups and stored from the root group down, so node j of a level
// has its children at 4j of the level below.
struct LightTreeShape
{
    uint leafLevel;
    uint lightCount;
    uint leafSize;

    uint LeafCount()
    {
        return (lightCount + leafSize - 1) / leafSize;
    }
};

// Where a light sits in the tree, for following it down: its leaf's position along the leaf level and its PMF within
// the leaf. The position's base-4 digits, from the top, are the child taken in each group on the way down. A light
// without a leaf has position 0 and a PMF of 0, so its path is meaningless but its PMF comes out 0.
struct LightTreePath
{
    uint leafPosition;
    float leafPmf;
};

LightTreePath LightTreeFindLight(StructuredBuffer<GpuLightTreeLeafEntry> leafOfLight, uint lightIndex)
{
    GpuLightTreeLeafEntry leaf = leafOfLight[lightIndex];
    LightTreePath path;
    path.leafPosition = leaf.index;
    path.leafPmf = leaf.pmf;
    return path;
}

// One walk from the root down to a leaf, both for drawing a sample and for the PMF of a known light, so that a wave
// whose lanes need either runs the node importance once per level rather than in two walks. follow: take the child on
// the way to leafPosition; otherwise pick each child in proportion to its importance with u, which is rescaled at each
// level. leaf is the leaf's position along its level, pmf the product of the taken children's shares of their group's
// importance. False, with a pmf of 0, where the lane's child has no importance: an empty group when sampling, a light
// the shading point cannot see the cluster of when following.
bool WalkLightTree(StructuredBuffer<GpuLightTreeNode> nodes, LightTreeShadingContext ctx, LightTreeShape shape, bool follow,
                   uint leafPosition, inout float u, out uint leaf, out float pmf)
{
    pmf = 1.0;
    leaf = 0;
    uint leafCount = shape.LeafCount();
    uint shift = 2 * shape.leafLevel; // of the digit of the child taken at this level, 2 less each level down
    uint levelFirst = 0; // the level's first node
    uint group = 0; // the group's first node along the level
    [loop]
    while (true)
    {
        float4 importance = LightTreeGroupImportance(nodes, levelFirst + group, ctx);
        float sum = LightTreeImportanceSum(importance);
        // A sampled child never lacks importance when the group has some (LightTreePickChild).
        uint chosen = follow ? (leafPosition >> shift) & (LIGHT_TREE_WIDTH - 1) : LightTreePickChild(importance, sum, u);
        float share = importance[chosen];
        if (!(share > 0.0 && sum > 0.0))
        {
            pmf = 0.0;
            return false;
        }
        pmf *= share / sum;
        uint position = group + chosen;
        if (shift == 0)
        {
            leaf = position;
            return true;
        }
        // Past this level's nodes, padded to whole groups, to the chosen node's children.
        uint width = (leafCount + (1u << shift) - 1) >> shift;
        levelFirst += (width + LIGHT_TREE_WIDTH - 1) & ~(LIGHT_TREE_WIDTH - 1);
        group = position * LIGHT_TREE_WIDTH;
        shift -= 2;
    }
    return false;
}

// A light of leaf (its position along the leaf level) in proportion to its share of the leaf's flux, picked with u,
// and that share.
void PickLeafLight(StructuredBuffer<GpuLightTreeLeafEntry> leafLights, LightTreeShape shape, uint leaf, float u,
                   out uint lightIndex, out float pmf)
{
    // The leaf's entries are loaded together; past its count they repeat its last one. Lights without flux, sorted
    // to the end, keep a slot with a PMF of 0, which is never picked.
    uint begin = leaf * shape.leafSize;
    uint last = min(shape.leafSize, shape.lightCount - begin) - 1;
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
        if (i <= last && entries[i].pmf > 0.0 && u >= before)
            chosen = i;
    }
    lightIndex = entries[chosen].index;
    pmf = entries[chosen].pmf;
}

// Stochastic traversal with one light sample per query, then a pick within the leaf in proportion to
// flux. u is a uniform random number in [0, 1) which is rescaled at each level.
// Returns false when no light can contribute.
bool SampleLightTree(StructuredBuffer<GpuLightTreeNode> nodes, StructuredBuffer<GpuLightTreeLeafEntry> leafLights,
                     LightTreeShadingContext ctx, LightTreeShape shape, float u, out uint lightIndex, out float pmf)
{
    lightIndex = 0;
    uint leaf;
    if (!WalkLightTree(nodes, ctx, shape, false, 0, u, leaf, pmf))
        return false;
    float leafPmf;
    PickLeafLight(leafLights, shape, leaf, u, lightIndex, leafPmf);
    pmf *= leafPmf;
    return true;
}

// Probability of SampleLightTree selecting lightIndex. Needed for MIS when BRDF sampling hits an emitter.
float LightTreePmf(StructuredBuffer<GpuLightTreeNode> nodes, StructuredBuffer<GpuLightTreeLeafEntry> leafOfLight,
                   LightTreeShadingContext ctx, LightTreeShape shape, uint lightIndex)
{
    LightTreePath path = LightTreeFindLight(leafOfLight, lightIndex);
    float u = 0.0;
    uint leaf;
    float pmf;
    WalkLightTree(nodes, ctx, shape, true, path.leafPosition, u, leaf, pmf);
    return pmf * path.leafPmf;
}
