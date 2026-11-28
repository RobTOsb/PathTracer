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

// A bottom-level light tree's lights: a contiguous range of the light buffer, in light order. Dynamic ones are refit
// every frame; static ones only when the topology is built and when RefitStatic asks.
struct LightBlasDesc
{
	uint32_t firstLight = 0;
	uint32_t lightCount = 0;
	bool dynamic = false;
};

// Builds the light tree on the GPU (LightTreeBuild.slang), in one of two forms:
// - single-level: one tree over every light, Morton sorted and built from scratch every frame (Lin and Yuksel 2020);
// - two-level (Moreau, Pharr and Clarberg 2019): a BLAS per range of lights (LightBlasDesc), whose topology is built
//   once, when the layout changes, and refit every frame, bottom up, one dispatch per height for every BLAS at once,
//   and a TLAS over the BLASes' root clusters, rebuilt every frame.
// Both rank nodes the same way, so the cone-bound and the SG importance work with either.
class LightTreeBuilder
{
public:
	explicit LightTreeBuilder(RenderContext *renderContext);
	~LightTreeBuilder();

	LightTreeBuilder(const LightTreeBuilder &) = delete;
	LightTreeBuilder &operator=(const LightTreeBuilder &) = delete;

	// New lights: fluxScales and lightEmitters per light, blases covering them in order (none for one BLAS over all).
	// Waits for the GPU.
	void Reset(std::span<const float> fluxScales, ResourceHandle lightEmitters, std::span<const LightBlasDesc> blases);

	// Chooses the form and the leaf size for the next AddPasses. Call before the frame's command list: when the
	// two-level layout changes it uploads the BLAS tables, which waits for the GPU, and the next AddPasses builds the
	// topology.
	void Prepare(bool twoLevel, uint32_t maxLeafLights);

	struct Inputs
	{
		ResourceHandle lights = InvalidResourceHandle; // StructuredBuffer<GpuLight>, the lights Reset was given
		ResourceHandle emissionScales = InvalidResourceHandle; // StructuredBuffer<float>, per emissive instance
		// Of the light centroids, for the Morton codes and the node means' quantisation; looser bounds only cost quality.
		float3 boundsMin = {};
		float3 boundsMax = {};
		// Two-level: refit the static BLASes too, as their lights' emission changed.
		bool refitStatic = false;
	};

	// Adds the build to the graph: LightTree/Keys, /Sort and /Nodes for the single-level tree; LightTree/Topology
	// (when the layout changed), /Refit and /TLAS for the two-level one; then LightTree/Power, the power selection CDF.
	// Each pass is a few dispatches with global UAV barriers between them. The outputs stay from frame to frame, as the
	// BLASes that are not refit keep theirs. With no lights it records nothing.
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

	// GpuLightTreeLeafEntry per leaf slot: the light and its PMF within the leaf. Two-level: then per TLAS leaf, its
	// BLAS (from GetTlasEntryOffset()).
	rg::ResourceId GetLeafLights() const
	{
		return outputs_[kLeafLights];
	}

	// GpuLightTreeLeafEntry per light: its leaf's position along its tree's leaf level and its PMF within it. Two-level:
	// then per BLAS, its TLAS leaf.
	rg::ResourceId GetLeafOfLight() const
	{
		return outputs_[kLeafOfLight];
	}

	// GpuLightPowerEntry per light, for power selection.
	rg::ResourceId GetPower() const
	{
		return outputs_[kPower];
	}

	// Levels of groups every leaf of the top tree (the single tree, or the TLAS) is under.
	uint32_t GetDepth() const
	{
		return depth_;
	}

	// The top tree's "lights" (the lights, or the BLASes) and lights per leaf (1 for the TLAS), which with the depth
	// make its shape (LightTreeShape in LightTree.slang).
	uint32_t GetLightCount() const
	{
		return twoLevel_ ? static_cast<uint32_t>(blasDescs_.size()) : lightCount_;
	}

	uint32_t GetLeafSize() const
	{
		return twoLevel_ ? 1u : leafSize_;
	}

	bool IsTwoLevel() const
	{
		return twoLevel_ && lightCount_ > 0;
	}

	// Two-level: the BLASes' lights per leaf, 0 for the single-level tree.
	uint32_t GetBlasLeafSize() const
	{
		return IsTwoLevel() ? leafSize_ : 0u;
	}

	uint32_t GetBlasCount() const
	{
		return static_cast<uint32_t>(blasDescs_.size());
	}

	// The deepest BLAS's levels of groups.
	uint32_t GetMaxBlasDepth() const
	{
		return static_cast<uint32_t>(heights_.size());
	}

	// Where the TLAS's entries start in the leaf buffers: after the lights'.
	uint32_t GetTlasEntryOffset() const
	{
		return lightCount_;
	}

	// GpuLightBlas per BLAS and the BLAS of each light, SRVs the path tracer reads bindless; one-element stand-ins
	// without them.
	ResourceHandle GetBlasBuffer() const
	{
		return blases_ != InvalidResourceHandle ? blases_ : placeholderBlases_;
	}

	ResourceHandle GetLightBlasBuffer() const
	{
		return lightBlas_;
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
	void DestroyLayout();

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
	PipelineHandle clearBoundsPipeline_ = InvalidPipelineHandle;
	PipelineHandle boundsPipeline_ = InvalidPipelineHandle;
	PipelineHandle topologyKeysPipeline_ = InvalidPipelineHandle;
	PipelineHandle topologySlotsPipeline_ = InvalidPipelineHandle;
	PipelineHandle refitLeavesPipeline_ = InvalidPipelineHandle;
	PipelineHandle refitInteriorPipeline_ = InvalidPipelineHandle;
	PipelineHandle blasClustersPipeline_ = InvalidPipelineHandle;

	uint32_t lightCount_ = 0;
	// LightTreeBuildResources, one per frame in flight: the transients' descriptors change when the graph places
	// them again.
	std::vector<ResourceHandle> resources_;
	ResourceHandle lightEmitters_ = InvalidResourceHandle;

	// Kept from frame to frame: the static BLASes are not rebuilt. Per light unless noted.
	ResourceHandle fluxScales_ = InvalidResourceHandle; // float, SRV
	ResourceHandle lightBlas_ = InvalidResourceHandle; // uint, SRV
	ResourceHandle lightClusters_ = InvalidResourceHandle; // GpuLightTreeCluster
	ResourceHandle nodeClusters_ = InvalidResourceHandle; // GpuLightTreeCluster per node
	ResourceHandle blasClusters_ = InvalidResourceHandle; // GpuLightTreeCluster per BLAS
	ResourceHandle blasBounds_ = InvalidResourceHandle; // 6 uints per BLAS
	ResourceHandle placeholderBlases_ = InvalidResourceHandle;
	enum Output
	{
		kNodes,
		kLeafLights,
		kLeafOfLight,
		kPower,
		kOutputCount
	};
	std::array<ResourceHandle, kOutputCount> outputBuffers_ = { InvalidResourceHandle, InvalidResourceHandle,
																InvalidResourceHandle, InvalidResourceHandle };
	std::array<rg::ResourceId, kOutputCount> outputs_ = {};
	// Transient sizes, in elements: per light and per block.
	uint32_t lightElements_ = 0;
	uint32_t blockElements_ = 0;

	// The two-level layout, for leafSize_: the BLASes and, per height above the leaves, the nodes to refit, the dynamic
	// BLASes' first.
	std::vector<LightBlasDesc> blasDescs_;
	ResourceHandle blases_ = InvalidResourceHandle; // GpuLightBlas, SRV
	ResourceHandle refitEntries_ = InvalidResourceHandle; // GpuLightRefitEntry, SRV
	struct HeightRange
	{
		uint32_t first;
		uint32_t dynamicCount;
		uint32_t totalCount;
	};
	std::vector<HeightRange> heights_;
	bool layoutValid_ = false;
	bool topologyPending_ = false; // the next AddPasses builds the topology and refits every BLAS

	bool twoLevel_ = false;
	uint32_t depth_ = 0;
	uint32_t leafSize_ = 4;
	float3 origin_ = { 0.0f, 0.0f, 0.0f };
	float step_ = 0.0f;
};
