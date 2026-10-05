#ifndef LIGHT_TREE_INTEROP_H
#define LIGHT_TREE_INTEROP_H

#ifdef __cplusplus
#include "Maths.h"
#endif

// Set in GpuLightTreeNode::childOrLight for leaves, the remaining bits are the light index.
#define LIGHT_TREE_LEAF_BIT 0x80000000u

// One node of the SG light tree from "Hierarchical Light Sampling with Accurate Spherical Gaussian
// Lighting" (Tokuyoshi, Ikeda, Kulkarni and Harada 2024). The ten floats of the paper's cluster
// (Section 3.1) plus the child and parent links.
struct GpuLightTreeNode
{
	float3 mean; // mu, flux weighted mean of the light positions
	float variance; // sigma_s^2, spatial variance of the light positions
	float3 axis; // nu, vMF axis of the radiant intensity
	float sharpness; // lambda, vMF sharpness, 0 for isotropic emitters
	float radius; // bounding sphere radius around the mean
	float flux; // Phi, radiant flux (luminance)
	uint childOrLight; // interior: left child, the right child is next to it; leaf: light index | LIGHT_TREE_LEAF_BIT
	uint parent; // parent node, the root points to itself
};

#ifdef __cplusplus
static_assert(sizeof(GpuLightTreeNode) == 48, "GpuLightTreeNode must match the structured buffer stride");
#endif

#endif // LIGHT_TREE_INTEROP_H
