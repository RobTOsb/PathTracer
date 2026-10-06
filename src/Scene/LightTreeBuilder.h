#pragma once

#include "LightTreeInterop.h"
#include "ShaderInterop.h"

#include <cstdint>
#include <span>
#include <vector>

// CPU construction of the SG light tree from "Hierarchical Light Sampling with Accurate Spherical
// Gaussian Lighting" (Tokuyoshi, Ikeda, Kulkarni and Harada 2024), Section 3.1. Traversal lives in
// shaders/LightTree.slang.

struct LightTree
{
	// Groups of LIGHT_TREE_WIDTH nodes, group 0 holds the children of the implicit root.
	std::vector<GpuLightTreeNode> nodes;
	std::vector<uint32_t> parents; // parent node of each group, 0 for group 0
	std::vector<GpuLightTreeLeafEntry> leafLights; // per leaf slot: light index and its PMF within the leaf
	std::vector<GpuLight> lights; // the lights the leaves index, only those with non-zero flux
	std::vector<GpuLightTreeLeafEntry> leafOfLight; // per light: its leaf node and PMF within it (LightTreePmf)
	std::vector<uint32_t> sourceIndex; // index into the input lights for each light
	uint32_t maxDepth = 0; // levels of groups the deepest light is under
	float averageDepth = 0.0f; // the same averaged over the lights
};

struct LightTreeBuildSettings
{
	uint32_t binCount = 12; // clamped to [2, 32]
	uint32_t maxLeafLights = 4; // clamped to [1, LIGHT_TREE_MAX_LEAF_LIGHTS]
};

// Builds a 4-wide tree over point lights (isotropic) and area lights (one-sided Lambertian
// triangles); other light types are skipped. A binary tree is built top down first, with a flux
// weighted binned SAH split (replace with SAOH [Conty and Kulla 2018] for best quality) that stops at
// maxLeafLights lights, and is then collapsed into groups of four. Cluster parameters follow Eqs. 2
// to 5. fluxScales, if not empty, holds a factor per light that its flux is multiplied by, such as the
// average of a textured area light's radiance map.
LightTree BuildLightTree(std::span<const GpuLight> lights,
						 std::span<const float> fluxScales = {},
						 const LightTreeBuildSettings &settings = {});

// The flux the tree gives a light, 0 for one that BuildLightTree would skip. The tree keeps the
// input order of the lights it does not skip, so lights with a non-zero flux here keep their index
// when every light before them has one too.
float LightTreeFlux(const GpuLight &light, float fluxScale = 1.0f);

constexpr uint32_t kInvalidLightIndex = INVALID_LIGHT_INDEX;

// Maps each input light to its index in LightTree::lights, or kInvalidLightIndex if it was
// skipped. Lets the path tracer find the light hit by a BSDF sampled ray.
std::vector<uint32_t> BuildInputToLightMap(const LightTree &tree, size_t inputLightCount);
