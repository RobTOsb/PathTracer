#ifndef LIGHT_TREE_INTEROP_H
#define LIGHT_TREE_INTEROP_H

#ifdef __cplusplus
#include "Maths.h"
#endif


#define LIGHT_TREE_WIDTH 4u

#define LIGHT_TREE_MAX_LEAF_LIGHTS 8u

// One node of the light tree: the cluster of "Hierarchical Light Sampling with Accurate Spherical Gaussian
// Lighting" (Tokuyoshi, Ikeda, Kulkarni and Harada 2024, Section 3.1), quantised to 16 bytes, so a group of four is
// half a cache line. The links are implicit (LightTreeShape in LightTree.slang). LightTree.slang's
// UnpackLightTreeNode decodes it:
//   x  mean.x | mean.y << 16    the flux weighted mean of the light positions, in 16-bit steps of
//   y  mean.z | radius << 16    lightTreeStep from lightTreeOrigin (FrameConstants), and a half bounding sphere
//                               radius around that quantised mean, rounded up
//   z  amplitude | axis << 16   bfloat16 Phi / (2 pi int g(w; -nu, lambda) dw), the node constant part of the SG light
//                               amplitude, and nu, the vMF axis of the radiant intensity, octahedral snorm8x2
//   w  sharpness | stdDev << 16 halves: lambda (vMF sharpness, 0 for isotropic emitters) and sigma_s
// An empty node, which pads a group, is all zero: its amplitude of 0 gets it no importance.
struct GpuLightTreeNode
{
#ifdef __cplusplus
	uint32_t packed[4];
#else
	uint4 packed;
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

// Two-level light hierarchy after "Dynamic Many-Light Sampling for Real-Time Ray Tracing" (Moreau, Pharr and Clarberg
// 2019): the light buffer is cut into contiguous ranges, each a bottom-level tree (BLAS) of its own, and a top-level
// tree (TLAS) over the BLASes' root clusters picks the BLAS. A BLAS's topology (which lights share a leaf, in Morton
// order within the BLAS's bounds) is built once on the GPU and refit every frame, bottom up, as its lights move and
// flash; the TLAS is rebuilt on the GPU every frame. Every node is a GpuLightTreeNode, so either importance ranks
// both levels.
//
// One BLAS: its nodes are an implicit 4-wide tree laid out as the single-level tree (LightTreeLayout in
// LightTreeBuild.slang) from nodeOffset, its leaf slots the lights' sorted order from firstLight on.
struct GpuLightBlas
{
#ifdef __cplusplus
	uint32_t nodeOffset; // first node of its root group in the node buffer
	uint32_t firstLight; // its lights are [firstLight, firstLight + lightCount) of the light buffer, and so its leaf slots
	uint32_t lightCount;
	uint32_t leafLevel; // its depth less one, with the frame's leaf size
#else
	uint nodeOffset;
	uint firstLight;
	uint lightCount;
	uint leafLevel;
#endif
};

// One node the refit computes (RefitLeaves, RefitInterior), from count children: lights in the leaf slots from first
// for a leaf, nodes from first (always 4, or 0 for the empty nodes that pad a level) for an interior node.
struct GpuLightRefitEntry
{
#ifdef __cplusplus
	uint32_t node;
	uint32_t first;
	uint32_t count;
#else
	uint node;
	uint first;
	uint count;
#endif
};

#define LIGHT_TREE_BUILD_GROUP_SIZE 256u
#define LIGHT_TREE_RADIX_BITS 4u
#define LIGHT_TREE_RADIX_PASSES 8u // of 32 bit keys
#define LIGHT_TREE_BUILD_BLOCK_SIZE 1024u // keys per radix sort or scan block, four per thread
// Interior levels of at most this many nodes are built together, by one group.
#define LIGHT_TREE_BUILD_TOP_NODES 256u
// The most BLASes: the topology's sort key holds the BLAS index above the Morton code.
#define LIGHT_TREE_MAX_BLAS_BITS 20u

// LightTreeBuildConstants::flags
#define LIGHT_TREE_BUILD_TLAS 1u // the tree kernels build the TLAS: its "lights" are the BLASes' clusters

// Root constants of the light tree build kernels (LightTreeBuild.slang), at most 16 dwords.
struct LightTreeBuildConstants
{
	float3 boundsMin; // of the light centroids, which the Morton keys quantise
	float boundsScale; // 1024 over the bounds' largest extent, so the cells are cubes
	uint lightCount; // of the tree the kernel builds: the lights, or for the TLAS the BLASes
	uint leafSize; // lights per leaf, 1 to LIGHT_TREE_MAX_LEAF_LIGHTS (1 for the TLAS)
	uint pass; // per kernel: the radix sort pass, the tree level, or the first of the levels built together
	uint lightsHandle; // RWStructuredBuffer<GpuLight>, only read: the light buffer has a UAV descriptor alone
	uint emissionScalesHandle; // StructuredBuffer<float>, per emissive instance
	float quantStep; // the node means' 16-bit steps from boundsMin (LightTreeBuilder::GetStep)
	uint entryOffset; // of the tree's entries in the leaf buffers: 0, or for the TLAS the light count
	uint flags; // LIGHT_TREE_BUILD_*
	uint first; // refit: the first entry of the height's list; topology: the BLAS count
	uint count; // refit: entries of the height's list to compute
	uint mortonBits; // topology: bits of the sort key below the BLAS index, at most 30
	uint pad;
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
// triangle lights moved by their instance's transform, then the punctual lights copied from this frame's upload.
struct DynamicLightConstants
{
	uint firstThread; // 0 to move the dynamic lights too, dynamicLightCount to write only the punctual lights
	uint dynamicLightCount;
	uint punctualLightCount;
	uint staticLightCount; // where the dynamic lights start in the light buffer
	uint objectLightsHandle; // StructuredBuffer<GpuLight>, per dynamic light, its triangle in object space
	uint lightInstancesHandle; // StructuredBuffer<uint>, per dynamic light, its dynamic instance
	uint transformsHandle; // StructuredBuffer<DynamicLightTransform>, per dynamic instance
	uint punctualLightsHandle; // StructuredBuffer<GpuLight>
	uint lightsHandle; // RWStructuredBuffer<GpuLight>, the light buffer
	// The dynamic lights from skinnedLightStart on are skinned triangles, whose object-space corners follow the pose:
	// they come from skinnedLightsHandle (StructuredBuffer<GpuLight>, from that light on) instead of objectLightsHandle.
	uint skinnedLightStart; // dynamicLightCount without skinned lights
	uint skinnedLightsHandle;
};

#define LIGHT_SWARM_GROUP_SIZE 256u

// Root constants of AnimateLightSwarms (DynamicLights.slang): swarms of point lights, lightsPerSwarm each, written to
// the light buffer from firstLight on. Each swarm's centre wanders the region on a path of its own, its lights orbit
// home positions within radius of it, and each light flashes on a clock of its own.
struct LightSwarmConstants
{
	float3 regionMin; // where the swarms' centres go
	float time; // seconds
	float3 regionSize;
	float radius; // of a swarm, scene units
	uint firstLight;
	uint lightsPerSwarm;
	uint lightCount; // of every swarm
	uint lightsHandle; // RWStructuredBuffer<GpuLight>
	float intensity; // radiant intensity of a light at the peak of a flash, W/sr
	float flashRate; // flashes per second, on average
	float speed; // how fast the swarms wander and their lights orbit, 1 = as designed
	uint seed;
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
	uint leafLights; // GpuLightTreeLeafEntry per leaf slot
	uint leafOfLight; // GpuLightTreeLeafEntry per light
	uint power; // GpuLightPowerEntry per light
	// Two-level only.
	uint blasClusters; // GpuLightTreeCluster per BLAS, of its root group: the TLAS's "light" clusters
	uint blasBounds; // uint per BLAS times 6: its lights' bounds as ordered floats, min then max, for the topology
	uint blases; // StructuredBuffer<GpuLightBlas>
	uint lightBlas; // StructuredBuffer<uint>, per light, its BLAS
	uint refitEntries; // StructuredBuffer<GpuLightRefitEntry>, the nodes per height, leaves first
};

#ifdef __cplusplus
static_assert(sizeof(GpuLightTreeNode) == 16, "GpuLightTreeNode must match the structured buffer stride");
static_assert(sizeof(GpuLightTreeLeafEntry) == 8, "GpuLightTreeLeafEntry must match the structured buffer stride");
static_assert(sizeof(GpuLightPowerEntry) == 16, "GpuLightPowerEntry must match the structured buffer stride");
static_assert(sizeof(GpuLightTreeCluster) == 36, "GpuLightTreeCluster must match the structured buffer stride");
static_assert(sizeof(GpuLightBlas) == 16, "GpuLightBlas must match the structured buffer stride");
static_assert(sizeof(GpuLightRefitEntry) == 12, "GpuLightRefitEntry must match the structured buffer stride");
static_assert(sizeof(LightSwarmConstants) <= 16 * 4, "LightSwarmConstants must fit the root constants");
static_assert(sizeof(LightTreeBuildConstants) <= 16 * 4, "LightTreeBuildConstants must fit the root constants");
static_assert(sizeof(DynamicLightTransform) == 64, "DynamicLightTransform must match the structured buffer stride");
static_assert(sizeof(DynamicLightConstants) <= 16 * 4, "DynamicLightConstants must fit the root constants");
#endif

#endif // LIGHT_TREE_INTEROP_H
