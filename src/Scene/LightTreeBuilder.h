#pragma once

#include "LightTreeInterop.h"
#include "ShaderInterop.h"

#include <cstdint>
#include <span>
#include <vector>

// CPU construction of the SG light tree from "Hierarchical Light Sampling with Accurate Spherical
// Gaussian Lighting" (Tokuyoshi, Ikeda, Kulkarni and Harada 2024), Section 3.1. Traversal lives in
// shaders/LightTree.slang.

// Where a tree goes in the buffers it may share with another tree: the static emitters' tree is built
// once with a slot of its root group left free, and the dynamic emitters' tree goes after it, under a
// node in that slot. Every index the tree stores is a buffer index, with these offsets applied.
struct LightTreeLayout
{
	uint32_t firstNode = 0; // of the root group, a multiple of LIGHT_TREE_WIDTH; 0 is the root of all
	uint32_t firstLight = 0; // buffer index of the first input light, and the tree's first leaf slot
	uint32_t rootWidth = LIGHT_TREE_WIDTH; // members the root group may have, the rest stay empty
	uint32_t rootParent = 0; // node above the root group, for the PMF walk; unused at firstNode 0
};

struct LightTree
{
	// Groups of LIGHT_TREE_WIDTH nodes, the first one the root group, starting at layout.firstNode.
	// The root group is there even without a light, all of it empty then.
	std::vector<GpuLightTreeNode> nodes;
	std::vector<uint32_t> parents; // parent node of each group, layout.rootParent for the root group
	// Per leaf slot from layout.firstLight on: light index and its PMF within the leaf. Only lights with
	// a non-zero flux get one.
	std::vector<GpuLightTreeLeafEntry> leafLights;
	// Per input light: its leaf node and PMF within it (LightTreePmf). A light without a leaf has a PMF
	// of 0 at node 0, which the walk turns into 0 after one group.
	std::vector<GpuLightTreeLeafEntry> leafOfLight;
	// Per input light: power selection by flux within this tree, see GpuLightAliasEntry. Aliases are
	// buffer indices.
	std::vector<GpuLightAliasEntry> powerTable;
	// The node a tree it is embedded in links to: the whole tree as one cluster, with the root group as
	// its child. Empty when flux is 0.
	GpuLightTreeNode root = {};
	double flux = 0.0; // of every light in the tree
	uint32_t treeLights = 0; // lights with a leaf
	uint32_t maxDepth = 0; // levels of groups the deepest light is under
	float averageDepth = 0.0f; // the same averaged over the lights
};

struct LightTreeBuildSettings
{
	uint32_t binCount = 12; // clamped to [2, 32]
	uint32_t maxLeafLights = 4; // clamped to [1, LIGHT_TREE_MAX_LEAF_LIGHTS]
	// Split cost also weighs how far each side's normals spread (the orientation measure of SAOH,
	// fitted to the vMF the nodes store). Off: flux times surface area only.
	bool orientationSplits = true;

	bool operator==(const LightTreeBuildSettings &) const = default;
};

// Builds a 4-wide tree over area lights (one-sided Lambertian triangles); other light types, and lights
// that emit nothing, get no leaf but keep their index. A binary tree is built top down first, with a
// binned split costing flux times surface area (and the orientation measure with orientationSplits)
// that stops at maxLeafLights lights, and is then collapsed into groups of four. Cluster parameters
// follow Eqs. 2 to 5. fluxScales, if not empty, holds a factor per light that its flux is multiplied
// by, such as the average of a textured area light's radiance map.
LightTree BuildLightTree(std::span<const GpuLight> lights,
						 std::span<const float> fluxScales = {},
						 const LightTreeBuildSettings &settings = {},
						 const LightTreeLayout &layout = {});

// The flux the tree gives a light, 0 for one that BuildLightTree gives no leaf.
float LightTreeFlux(const GpuLight &light, float fluxScale = 1.0f);

// A node that LightTreeNodeImportance rejects, for the free slots of a group.
GpuLightTreeNode EmptyLightTreeNode();

