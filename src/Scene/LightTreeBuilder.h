#pragma once

#include "LightTreeInterop.h"
#include "RenderContext.h"
#include "ShaderInterop.h"

#include <cstdint>
#include <span>

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

	// Records the build on the current graphics command list, and leaves the outputs readable by any shader.
	void Build(const Inputs &inputs);

	ResourceHandle GetNodes() const
	{
		return nodes_;
	}

	// Parent node of each group (uint), for the PMF walk.
	ResourceHandle GetParents() const
	{
		return parents_;
	}

	// GpuLightTreeLeafEntry per leaf slot: the light and its PMF within the leaf.
	ResourceHandle GetLeafLights() const
	{
		return leafLights_;
	}

	// GpuLightTreeLeafEntry per light: its leaf node and its PMF within it.
	ResourceHandle GetLeafOfLight() const
	{
		return leafOfLight_;
	}

	// GpuLightPowerEntry per light, for power selection.
	ResourceHandle GetPower() const
	{
		return power_;
	}

	// Levels of groups every light is under, in the last build.
	uint32_t GetDepth() const
	{
		return depth_;
	}

	// The leaf level's first node in the last build: a light's leaf less this is its position along the level.
	uint32_t GetLeafOffset() const
	{
		return leafOffset_;
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
	ResourceHandle resources_ = InvalidResourceHandle; // LightTreeBuildResources
	ResourceHandle fluxScales_ = InvalidResourceHandle;
	ResourceHandle lightClusters_ = InvalidResourceHandle;
	ResourceHandle nodeClusters_ = InvalidResourceHandle;
	ResourceHandle keys_[2] = { InvalidResourceHandle, InvalidResourceHandle };
	ResourceHandle values_[2] = { InvalidResourceHandle, InvalidResourceHandle };
	ResourceHandle radixCounts_ = InvalidResourceHandle;
	ResourceHandle blockSums_ = InvalidResourceHandle;
	ResourceHandle nodes_ = InvalidResourceHandle;
	ResourceHandle parents_ = InvalidResourceHandle;
	ResourceHandle leafLights_ = InvalidResourceHandle;
	ResourceHandle leafOfLight_ = InvalidResourceHandle;
	ResourceHandle power_ = InvalidResourceHandle;
	uint32_t depth_ = 0;
	uint32_t leafOffset_ = 0;
};
