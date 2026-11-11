#ifndef LIGHT_TREE_INTEROP_H
#define LIGHT_TREE_INTEROP_H

#ifdef __cplusplus
#include "Maths.h"
#endif


#define LIGHT_TREE_WIDTH 4u

#define LIGHT_TREE_LEAF_BIT 0x80000000u
#define LIGHT_TREE_LEAF_COUNT_SHIFT 28u // bits 28 to 30 hold the leaf's light count less one
#define LIGHT_TREE_LEAF_FIRST_MASK 0x0FFFFFFFu
#define LIGHT_TREE_MAX_LEAF_LIGHTS 8u

// One node of the light tree: the cluster of "Hierarchical Light Sampling with Accurate Spherical Gaussian
// Lighting" (Tokuyoshi, Ikeda, Kulkarni and Harada 2024, Section 3.1) and the child link, packed into 32
// bytes so that the shader fetches it with two 16 byte loads. The parent links, which only the PMF walk
// needs, live in a buffer of their own (one per group).
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

// Power light selection: light i owns [cdfEnd of light i - 1, cdfEnd) of a CDF over the lights' flux in light
// order, quantised to integers below 2^31 so that it is exact and monotonic however the GPU summed it, and pmf is
// the width of that interval over 2^31. guide (entries below the guide table's size, the largest power of two not
// above the light count) is the light that owns the start of bucket i, the first light worth searching from.
// The intervals cover a little less than [0, 2^31); a draw past the last one finds no light.
struct GpuLightPowerEntry
{
#ifdef __cplusplus
	uint32_t cdfEnd;
	float pmf;
	uint32_t guide;
	uint32_t pad;
#else
	uint cdfEnd;
	float pmf;
	uint guide;
	uint pad;
#endif
};

// A light's or a node's cluster in full precision (Eqs. 2 to 5 of Tokuyoshi et al.), the build's scratch: LightTreeBuild.slang
// packs it into a GpuLightTreeNode. Zero flux marks a light without a leaf, or an empty node.
struct GpuLightTreeCluster
{
	float3 mean;
	float flux;
	float3 averageDirection; // nu bar in Eq. 2, 0 for an isotropic emitter
	float variance;
	float radius;
};

#define LIGHT_TREE_BUILD_GROUP_SIZE 256u
#define LIGHT_TREE_RADIX_BITS 4u
#define LIGHT_TREE_RADIX_PASSES 8u // of 32 bit keys
#define LIGHT_TREE_BUILD_BLOCK_SIZE 1024u // keys per radix sort or scan block, four per thread
// Interior levels of at most this many nodes are built together, by one group.
#define LIGHT_TREE_BUILD_TOP_NODES 256u

// Root constants of the light tree build kernels (LightTreeBuild.slang), at most 16 dwords.
struct LightTreeBuildConstants
{
	float3 boundsMin; // of the light centroids, which the Morton keys quantise
	float boundsScale; // 1024 over the bounds' largest extent, so the cells are cubes
	uint lightCount;
	uint leafSize; // lights per leaf, 1 to LIGHT_TREE_MAX_LEAF_LIGHTS
	uint pass; // per kernel: the radix sort pass, the tree level, or the first of the levels built together
	uint lightsHandle; // RWStructuredBuffer<GpuLight>, only read: the light buffer has a UAV descriptor alone
	uint emissionScalesHandle; // StructuredBuffer<float>, per emissive instance
};

// One dynamic instance's world transform for TransformDynamicLights (DynamicLights.slang), in the CPU's float3x4
// layout: p' = p.x rows[0].xyz + p.y rows[1].xyz + p.z rows[2].xyz + (rows[0].w, rows[1].w, rows[2].w).
struct DynamicLightTransform
{
#ifdef __cplusplus
	float m[12];
	uint32_t swapCorners; // the transform mirrors where the rest one did not, or the other way round
	uint32_t pad[3];
#else
	float4 rows[3];
	uint swapCorners;
	uint3 pad;
#endif
};

#define DYNAMIC_LIGHTS_GROUP_SIZE 256u

// Root constants of TransformDynamicLights. Thread i writes light staticLightCount + firstThread + i: the dynamic
// triangle lights moved by their instance's transform, then the point lights copied from this frame's upload.
struct DynamicLightConstants
{
	uint firstThread; // 0 to move the dynamic lights too, dynamicLightCount to write only the point lights
	uint dynamicLightCount;
	uint pointLightCount;
	uint staticLightCount; // where the dynamic lights start in the light buffer
	uint objectLightsHandle; // StructuredBuffer<GpuLight>, per dynamic light, its triangle in object space
	uint lightInstancesHandle; // StructuredBuffer<uint>, per dynamic light, its dynamic instance
	uint transformsHandle; // StructuredBuffer<DynamicLightTransform>, per dynamic instance
	uint pointLightsHandle; // StructuredBuffer<GpuLight>
	uint lightsHandle; // RWStructuredBuffer<GpuLight>, the light buffer
};

// Descriptor heap indices of the build's buffers, which only change when the light count does; a constant
// buffer, as they do not fit the root constants. RW buffers unless noted.
struct LightTreeBuildResources
{
	uint fluxScales; // StructuredBuffer<float>, per light: its emissive texture's average
	uint lightEmitters; // StructuredBuffer<uint>, per light: its emissive instance
	uint lightClusters; // GpuLightTreeCluster per light
	uint nodeClusters; // GpuLightTreeCluster per node
	uint keysA; // the radix sort's (key, light) pairs, sorted into the A pair at the end
	uint valuesA;
	uint keysB; // after the sort, each light's quantised flux for the power CDF
	uint valuesB;
	uint radixCounts; // per radix digit and block, digit major
	uint blockSums; // per scan block of the power CDF
	uint nodes; // GpuLightTreeNode
	uint parents; // uint per group
	uint leafLights; // GpuLightTreeLeafEntry per leaf slot
	uint leafOfLight; // GpuLightTreeLeafEntry per light
	uint power; // GpuLightPowerEntry per light
};

#ifdef __cplusplus
static_assert(sizeof(GpuLightTreeNode) == 32, "GpuLightTreeNode must match the structured buffer stride");
static_assert(sizeof(GpuLightTreeLeafEntry) == 8, "GpuLightTreeLeafEntry must match the structured buffer stride");
static_assert(sizeof(GpuLightPowerEntry) == 16, "GpuLightPowerEntry must match the structured buffer stride");
static_assert(sizeof(GpuLightTreeCluster) == 36, "GpuLightTreeCluster must match the structured buffer stride");
static_assert(sizeof(LightTreeBuildConstants) <= 16 * 4, "LightTreeBuildConstants must fit the root constants");
static_assert(sizeof(DynamicLightTransform) == 64, "DynamicLightTransform must match the structured buffer stride");
static_assert(sizeof(DynamicLightConstants) <= 16 * 4, "DynamicLightConstants must fit the root constants");
#endif

#endif // LIGHT_TREE_INTEROP_H
