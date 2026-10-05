// SG light tree traversal (Sections 3 and 6 of "Hierarchical Light Sampling with Accurate Spherical
// Gaussian Lighting"). The tree is built on the CPU by LightTreeBuilder.

#include "LightTreeInterop.h"

import SGLighting;

bool LightTreeIsLeaf(GpuLightTreeNode n)
{
    return (n.childOrLight & LIGHT_TREE_LEAF_BIT) != 0;
}

uint LightTreeLeafLight(GpuLightTreeNode n)
{
    return n.childOrLight & ~LIGHT_TREE_LEAF_BIT;
}

struct LightTreeShadingContext
{
    float3 position;
    float3 normal;
    float diffuseWeight;   // reflectance of the diffuse lobe (e.g. luminance of the albedo)
    float specularWeight;  // reflectance of the glossy lobe (e.g. luminance of the Fresnel term)
    GlossyContext glossy;
};

LightTreeShadingContext MakeLightTreeShadingContext(float3 position, float3 T, float3 B, float3 N, float3 V,
                                                    float diffuseWeight, float specularWeight,
                                                    float alphaX, float alphaY)
{
    LightTreeShadingContext ctx;
    ctx.position = position;
    ctx.normal = N;
    ctx.diffuseWeight = diffuseWeight;
    ctx.specularWeight = dot(V, N) > 0.0 ? specularWeight : 0.0;
    ctx.glossy = MakeGlossyContext(T, B, N, V, alphaX, alphaY);
    return ctx;
}

// Section 3.2: incoming radiance from the cluster as an SG light,
// L(x, o) ~ Phi g(o; -nu, lambda) g(o; dir, d^2 / sigma^2) / (2 pi sigma^2 int g(w; -nu, lambda) dw).
SGLight MakeSGLight(GpuLightTreeNode node, float3 x, float3 n)
{
    float3 toLight = node.mean - x;
    float d2 = dot(toLight, toLight);
    float d = sqrt(d2);
    float3 dir = d > 0.0 ? toLight / d : n;

    // Hybrid variance: blend towards the conservative bound when the cluster centre is behind the surface.
    float c = max(dot(n, -dir), 0.0);
    float sigma2 = lerp(node.variance, 0.5 * node.radius * node.radius, c);
    sigma2 = max(sigma2, 1e-8 * d2 + 1e-12);

    float invSigma2 = 1.0 / sigma2;
    float s = d2 * invSigma2;

    SGLight sg;
    float logScale;
    SGProduct(-node.axis, node.sharpness, dir, s, sg.axis, sg.sharpness, logScale);
    sg.amplitude = node.flux * exp(logScale) * invSigma2 / (SG_TWO_PI * SGIntegral(node.sharpness));
    return sg;
}

// Eq. 6 with the diffuse (Eq. 7) and glossy (Eq. 12) approximations, weighted by lobe reflectance.
float LightTreeNodeImportance(GpuLightTreeNode node, LightTreeShadingContext ctx)
{
    // The bounding sphere lies entirely below the tangent plane: the true integral is zero.
    float3 toMean = node.mean - ctx.position;
    if (dot(toMean, ctx.normal) < -node.radius)
        return 0.0;

    SGLight sg = MakeSGLight(node, ctx.position, ctx.normal);

    float importance = 0.0;
    if (ctx.diffuseWeight > 0.0)
        importance += ctx.diffuseWeight * SGDiffuseImportance(sg, ctx.normal);
    if (ctx.specularWeight > 0.0)
        importance += ctx.specularWeight * SGGlossyImportance(sg, ctx.glossy);

    // Guard against float underflow so that Eq. 1 (I > 0 whenever the integral is > 0) still holds.
    return max(importance, 1e-30);
}

// Stochastic traversal with one light sample per query.
// u is a uniform random number in [0, 1) which is rescaled at each level.
// Returns false when no light can contribute.
bool SampleLightTree(StructuredBuffer<GpuLightTreeNode> nodes, LightTreeShadingContext ctx, float u,
                     out uint lightIndex, out float pmf)
{
    const float ONE_MINUS_EPSILON = 0.99999994;
    pmf = 1.0;
    lightIndex = 0;

    GpuLightTreeNode node = nodes[0];
    [loop]
    while (!LightTreeIsLeaf(node))
    {
        uint left = node.childOrLight;
        GpuLightTreeNode leftNode = nodes[left];
        GpuLightTreeNode rightNode = nodes[left + 1];
        float iL = LightTreeNodeImportance(leftNode, ctx);
        float iR = LightTreeNodeImportance(rightNode, ctx);
        float sum = iL + iR;
        if (!(sum > 0.0))
            return false;

        // pR is computed directly rather than as 1 - pL: it keeps precision for rarely chosen children
        // and matches LightTreePmf exactly, which MIS relies on.
        float pL = iL / sum;
        float pR = iR / sum;
        if (u < pL)
        {
            u = min(u / pL, ONE_MINUS_EPSILON);
            pmf *= pL;
            node = leftNode;
        }
        else
        {
            u = min((u - pL) / pR, ONE_MINUS_EPSILON);
            pmf *= pR;
            node = rightNode;
        }
    }

    lightIndex = LightTreeLeafLight(node);
    return true;
}

// Probability of SampleLightTree selecting lightIndex. Needed for MIS when BRDF sampling hits an emitter.
// Walks from the leaf to the root using parent links.
float LightTreePmf(StructuredBuffer<GpuLightTreeNode> nodes, StructuredBuffer<uint> leafOfLight,
                   LightTreeShadingContext ctx, uint lightIndex)
{
    uint index = leafOfLight[lightIndex];
    float pmf = 1.0;
    [loop]
    while (index != 0)
    {
        uint parentIndex = nodes[index].parent;
        uint left = nodes[parentIndex].childOrLight;
        float iL = LightTreeNodeImportance(nodes[left], ctx);
        float iR = LightTreeNodeImportance(nodes[left + 1], ctx);
        float sum = iL + iR;
        if (!(sum > 0.0))
            return 0.0;
        pmf *= (index == left ? iL : iR) / sum;
        index = parentIndex;
    }
    return pmf;
}
