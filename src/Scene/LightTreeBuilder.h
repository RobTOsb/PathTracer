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
	std::vector<GpuLightTreeNode> nodes; // nodes[0] is the root
	std::vector<GpuLight> lights; // the lights the leaves index, only those with non-zero flux
	std::vector<uint32_t> leafOfLight; // leaf node of each light, for the MIS PMF (LightTreePmf)
	std::vector<uint32_t> sourceIndex; // index into the input lights for each light
};

struct LightTreeBuildSettings
{
	uint32_t binCount = 12; // clamped to [2, 32]
};

// Builds a binary tree with one light per leaf over point lights (isotropic) and area lights
// (one-sided Lambertian triangles); other light types are skipped. Topology uses a flux weighted
// binned SAH split, replace with SAOH [Conty and Kulla 2018] for best quality. Cluster parameters
// follow Eqs. 2 to 5.
LightTree BuildLightTree(std::span<const GpuLight> lights, const LightTreeBuildSettings &settings = {});

constexpr uint32_t kInvalidLightIndex = 0xFFFFFFFFu;

// Maps each input light to its index in LightTree::lights, or kInvalidLightIndex if it was
// skipped. Lets the path tracer find the light hit by a BSDF sampled ray.
std::vector<uint32_t> BuildInputToLightMap(const LightTree &tree, size_t inputLightCount);
