#ifndef LIGHT_TREE_INTEROP_H
#define LIGHT_TREE_INTEROP_H

#ifdef __cplusplus
#include "Maths.h"
#endif


#define LIGHT_TREE_WIDTH 4u

#define LIGHT_TREE_LEAF_BIT 0x80000000u
#define LIGHT_TREE_LEAF_COUNT_SHIFT 29u
#define LIGHT_TREE_LEAF_FIRST_MASK 0x1FFFFFFFu
#define LIGHT_TREE_MAX_LEAF_LIGHTS 4u

// One node of the SG light tree from "Hierarchical Light Sampling with Accurate Spherical Gaussian
// Lighting" (Tokuyoshi, Ikeda, Kulkarni and Harada 2024): the paper's cluster (Section 3.1) and the
// child link, packed into 32 bytes so that the shader fetches it with two 16 byte loads. The parent
// links, which only the PMF walk needs, live in a buffer of their own (one per group).
//
// Empty nodes, which pad a group, have a radius of -FLT_MAX: LightTreeNodeImportance then rejects them
// as lying below the horizon, so they get zero importance without a test of their own.
struct GpuLightTreeNode
{
#ifdef __cplusplus
	float3 mean; // mu, flux weighted mean of the light positions
	float amplitude; // Phi / (2 pi int g(w; -nu, lambda) dw), the node constant part of the SG light amplitude
	uint32_t childOrLight; // interior: first node of the child group; leaf: see LIGHT_TREE_LEAF_BIT
	uint32_t axis; // nu, vMF axis of the radiant intensity, octahedral snorm16x2 (x in the low half)
	uint32_t sharpnessStdDev; // half2: lambda (vMF sharpness, 0 for isotropic emitters) low, sigma_s high
	float radius; // bounding sphere radius around the mean
#else
	float4 meanAmplitude;
	uint4 packed; // childOrLight, axis, sharpnessStdDev, asuint(radius)
#endif
};

// Pairs an index with the probability of picking a light within its leaf, which is its share of the
// leaf's flux. The leaf light buffer holds one per leaf slot, with index the light; the light leaf
// buffer holds one per light, with index its leaf node, for the MIS PMF.
struct GpuLightTreeLeafEntry
{
#ifdef __cplusplus
	uint32_t index;
#else
	uint index;
#endif
	float pmf;
};

#ifdef __cplusplus
static_assert(sizeof(GpuLightTreeNode) == 32, "GpuLightTreeNode must match the structured buffer stride");
static_assert(sizeof(GpuLightTreeLeafEntry) == 8, "GpuLightTreeLeafEntry must match the structured buffer stride");
#endif

#endif // LIGHT_TREE_INTEROP_H
