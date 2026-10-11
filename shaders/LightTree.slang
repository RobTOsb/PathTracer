// SG light tree traversal (Sections 3 and 6 of "Hierarchical Light Sampling with Accurate Spherical
// Gaussian Lighting"). The tree is built on the CPU by LightTreeBuilder.

#include "LightTreeInterop.h"

import SGLighting;

// Accessors for the packed node (see LightTreeInterop.h).

uint LightTreeChildOrLight(GpuLightTreeNode n)
{
    return n.packed.x;
}

float LightTreeRadius(GpuLightTreeNode n)
{
    return asfloat(n.packed.w);
}

// (sharpness, spatial standard deviation).
float2 LightTreeSharpnessStdDev(GpuLightTreeNode n)
{
    uint bits = n.packed.z;
    return float2(f16tof32(bits & 0xFFFF), f16tof32(bits >> 16));
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
    GlossyContext glossy;
    // SG light sharpness from which a node wholly above the horizon takes the far-field (point light)
    // form of the diffuse and of the glossy importance.
    float diffuseFarFieldSharpness;
    float glossyFarFieldSharpness;
};

// diffuseOnlyAlpha: GGX alpha (roughness squared) on both axes at or above which the glossy importance
// is replaced by the diffuse one. Such lobes are wide enough that the diffuse estimate ranks clusters
// about as well, and the glossy evaluation is the expensive half of LightTreeNodeImportance.
// farFieldSharpness: SG light sharpness (squared distance over spatial variance) from which a cluster is
// small enough, seen from here, to be treated as a point light. FLT_MAX turns the far field off.
LightTreeShadingContext MakeLightTreeShadingContext(float3 position, float3 T, float3 B, float3 N, float3 V,
                                                    float diffuseWeight, float specularWeight,
                                                    float alphaX, float alphaY, float diffuseOnlyAlpha,
                                                    float farFieldSharpness)
{
    LightTreeShadingContext ctx;
    ctx.position = position;
    ctx.normal = N;
    float specular = dot(V, N) > 0.0 ? specularWeight : 0.0;
    // Depends only on the shading point, so SampleLightTree and LightTreePmf still agree.
    bool roughLobe = min(alphaX, alphaY) >= diffuseOnlyAlpha;
    ctx.diffuseWeight = roughLobe ? diffuseWeight + specular : diffuseWeight;
    ctx.specularWeight = roughLobe ? 0.0 : specular;
    ctx.glossy = MakeGlossyContext(T, B, N, V, alphaX, alphaY);

    float jjtMax = max(ctx.glossy.jjt.x, ctx.glossy.jjt.z);
    float alpha2Min = min(ctx.glossy.alpha2.x, ctx.glossy.alpha2.y);
    ctx.diffuseFarFieldSharpness = farFieldSharpness;
    ctx.glossyFarFieldSharpness = max(farFieldSharpness, 20.0 * jjtMax / alpha2Min);
    return ctx;
}

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
    float2 sharpnessStdDev = LightTreeSharpnessStdDev(node);
    float c = max(dot(n, -dir), 0.0);
    float sigma2 = lerp(sharpnessStdDev.y * sharpnessStdDev.y, 0.5 * radius * radius, c);
    sigma2 = max(sigma2, 1e-8 * d2 + 1e-12);

    float invSigma2 = 1.0 / sigma2;
    float s = d2 * invSigma2;

    // The node amplitude already holds Phi / (2 pi int g(w; -nu, lambda) dw).
    SGLight sg;
    float logScale;
    SGProduct(-LightTreeAxis(node), sharpnessStdDev.x, dir, s, sg.axis, sg.sharpness, logScale);
    sg.amplitude = node.meanAmplitude.w * exp(logScale) * invSigma2;
    return sg;
}

// Eq. 6 with the diffuse (Eq. 7) and glossy (Eq. 12) approximations, weighted by lobe reflectance.
float LightTreeNodeImportance(GpuLightTreeNode node, LightTreeShadingContext ctx)
{
    // The bounding sphere lies entirely below the tangent plane: the true integral is zero. Also
    // rejects the empty nodes that pad a group, whose radius is -FLT_MAX.
    float3 toMean = node.meanAmplitude.xyz - ctx.position;
    float height = dot(toMean, ctx.normal);
    float radius = LightTreeRadius(node);
    if (height < -radius)
        return 0.0;

    SGLight sg = MakeSGLight(node, toMean, ctx.normal);

    bool aboveHorizon = height > radius;

    float importance = 0.0;
    if (ctx.diffuseWeight > 0.0)
    {
        if (aboveHorizon && sg.sharpness >= ctx.diffuseFarFieldSharpness)
            importance += ctx.diffuseWeight * SGDiffuseImportanceFarField(sg, ctx.normal);
        else
            importance += ctx.diffuseWeight * SGDiffuseImportance(sg, ctx.normal);
    }
    if (ctx.specularWeight > 0.0)
    {
        if (aboveHorizon && sg.sharpness >= ctx.glossyFarFieldSharpness)
            importance += ctx.specularWeight * SGGlossyImportanceFarField(sg, ctx.glossy);
        else
            importance += ctx.specularWeight * SGGlossyImportance(sg, ctx.glossy);
    }

    // Guard against float underflow so that Eq. 1 (I > 0 whenever the integral is > 0) still holds.
    return max(importance, 1e-30);
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
    [loop]
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
