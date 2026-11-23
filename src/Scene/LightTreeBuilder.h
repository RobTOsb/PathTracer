#pragma once

#include "LightTreeInterop.h"
#include "RenderContext.h"
#include "RenderGraph.h"
#include "ShaderInterop.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

class FrameGraph;

float LightTreeFlux(const GpuLight &light, float fluxScale = 1.0f);

class LightTreeBuilder
{
public:
	explicit LightTreeBuilder(RenderContext *renderContext);
	~LightTreeBuilder();

	LightTreeBuilder(const LightTreeBuilder &) = delete;
	LightTreeBuilder &operator=(const LightTreeBuilder &) = delete;

	void Reset(std::span<const float> fluxScales, ResourceHandle lightEmitters);

	struct Inputs
	{
		ResourceHandle lights = InvalidResourceHandle; // StructuredBuffer<GpuLight>, the lights Reset was given
		ResourceHandle emissionScales = InvalidResourceHandle; // StructuredBuffer<float>, per emissive instance
		// Of the light centroids, for the Morton codes; looser bounds only cost quality.
		float3 boundsMin = {};
		float3 boundsMax = {};
		uint32_t maxLeafLights = 4; // clamped to [1, LIGHT_TREE_MAX_LEAF_LIGHTS]
	};

	// Adds the build to the graph in four passes, each a few dispatches with global UAV barriers between them:
	// LightTree/Keys (the lights' clusters and Morton codes), LightTree/Sort (the radix sort of the codes),
	// LightTree/Nodes (the leaves, then the levels up to the root) and LightTree/Power (the power selection CDF).
	// Everything is a graph transient, the outputs too, since each frame builds them anew. With no lights it records
	// nothing and the outputs are imports of one-element placeholders, so the bindings stay valid.
	void AddPasses(FrameGraph &frame, const Inputs &inputs);
	// Forgets the outputs, for a frame that does not call AddPasses.
	void ClearOutputs()
	{
		outputs_ = {};
	}

	// This frame's outputs, as graph resources, once AddPasses has run; invalid before Reset.
	rg::ResourceId GetNodes() const
	{
		return outputs_[kNodes];
	}

	// GpuLightTreeLeafEntry per leaf slot: the light and its PMF within the leaf.
	rg::ResourceId GetLeafLights() const
	{
		return outputs_[kLeafLights];
	}

	// GpuLightTreeLeafEntry per light: its leaf's position along the leaf level and its PMF within it.
	rg::ResourceId GetLeafOfLight() const
	{
		return outputs_[kLeafOfLight];
	}

	// GpuLightPowerEntry per light, for power selection.
	rg::ResourceId GetPower() const
	{
		return outputs_[kPower];
	}

	// Levels of groups every light is under, in the last build.
	uint32_t GetDepth() const
	{
		return depth_;
	}

	// The lights the tree holds and the lights per leaf in the last build, which with the depth make its shape
	// (LightTreeShape in LightTree.slang).
	uint32_t GetLightCount() const
	{
		return lightCount_;
	}

	uint32_t GetLeafSize() const
	{
		return leafSize_;
	}

	// The node means' quantisation in the last build: 16-bit steps of GetStep() from GetOrigin().
	float3 GetOrigin() const
	{
		return origin_;
	}

	float GetStep() const
	{
		return step_;
	}

private:
	void DestroyBuffers();

	RenderContext *renderContext_ = nullptr;

	PipelineHandle keysPipeline_ = InvalidPipelineHandle;
	PipelineHandle radixCountPipeline_ = InvalidPipelineHandle;
	PipelineHandle radixScanPipeline_ = InvalidPipelineHandle;
	PipelineHandle radixScatterPipeline_ = InvalidPipelineHandle;
	PipelineHandle leavesPipeline_ = InvalidPipelineHandle;
	PipelineHandle levelPipeline_ = InvalidPipelineHandle;
	PipelineHandle topLevelsPipeline_ = InvalidPipelineHandle;
	PipelineHandle powerReducePipeline_ = InvalidPipelineHandle;
	PipelineHandle powerScanPipeline_ = InvalidPipelineHandle;
	PipelineHandle powerApplyPipeline_ = InvalidPipelineHandle;
	PipelineHandle powerGuidePipeline_ = InvalidPipelineHandle;

	uint32_t lightCount_ = 0;
	// LightTreeBuildResources, one per frame in flight: the transients' descriptors change when the graph places
	// them again.
	std::vector<ResourceHandle> resources_;
	ResourceHandle fluxScales_ = InvalidResourceHandle;
	ResourceHandle lightEmitters_ = InvalidResourceHandle;
	// The transients' sizes, in elements: per light, per node (for one light per leaf) and per block.
	uint32_t lightElements_ = 0;
	uint32_t nodeElements_ = 0;
	uint32_t blockElements_ = 0;
	enum Output
	{
		kNodes,
		kLeafLights,
		kLeafOfLight,
		kPower,
		kOutputCount
	};
	// Without lights, one-element stand-ins for the outputs.
	std::array<ResourceHandle, kOutputCount> placeholders_ = { InvalidResourceHandle, InvalidResourceHandle,
															   InvalidResourceHandle, InvalidResourceHandle };
	std::array<rg::ResourceId, kOutputCount> outputs_ = {};
	uint32_t depth_ = 0;
	uint32_t leafSize_ = 1;
	float3 origin_ = { 0.0f, 0.0f, 0.0f };
	float step_ = 0.0f;
};
