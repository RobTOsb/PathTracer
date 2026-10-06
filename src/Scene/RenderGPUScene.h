#pragma once

#include "GPUScene.h"
#include "RenderContext.h"

#include "Maths.h"
#include "ShaderInterop.h"

#include <span>
#include <vector>

namespace Scene
{
	class Texture;
}

class RenderGPUScene final : public Scene::GPUScene
{
public:
	explicit RenderGPUScene(RenderContext *renderContext);
	~RenderGPUScene() override;

	RenderGPUScene(const RenderGPUScene &) = delete;
	RenderGPUScene &operator=(const RenderGPUScene &) = delete;

	void BuildGPUScene(const Scene::GLTFScene &scene) override;

	// Refreshes the per-instance transforms in the instance table. The TLAS is built once at start
	// up and is not refitted here yet.
	void UpdateInstances(const Scene::GLTFScene &scene) override;

	void BuildAccelerationStructures();

	// Rebuilds the SG light tree over point and area lights and writes it with the light buffer
	// into the current frame's slot. Call once per frame after BeginFrame, whose fence wait frees
	// that slot; the buffer getters then return this frame's copies.
	void UpdateLights(std::span<const GpuLight> lights);

	bool HasGeometry() const
	{
		return tlas_ != InvalidAccelerationStructureHandle;
	}

	ResourceHandle GetTLASSRV() const
	{
		return tlasSRV_;
	}

	// Bindless indices of the material table (GpuMaterial) and primitive table (PrimitiveGPUDesc).
	uint32_t GetMaterialBufferIndex() const
	{
		return materialBufferIndex_;
	}

	uint32_t GetPrimitiveBufferIndex() const
	{
		return primitiveBufferIndex_;
	}

	// Resource handles of the same buffers, for binding them directly (e.g. as root SRVs).
	ResourceHandle GetStreamBuffer(GPUStream stream) const
	{
		return streamBuffers_[(uint32_t) stream];
	}

	ResourceHandle GetMaterialBuffer() const
	{
		return materialBuffer_;
	}

	// Rows in the material buffer, including the default material at the end.
	uint32_t GetMaterialCount() const
	{
		return static_cast<uint32_t>(materialAlphaModes_.size());
	}

	ResourceHandle GetPrimitiveBuffer() const
	{
		return primitiveBuffer_;
	}

	// The lights the light tree indexes (GpuLight) and its nodes (GpuLightTreeNode), as written by
	// the last UpdateLights. Both always hold at least one element so they can stay bound,
	// GetLightCount is 0 without lights.
	ResourceHandle GetLightBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].lights;
	}

	ResourceHandle GetLightTreeNodeBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].nodes;
	}

	uint32_t GetLightCount() const
	{
		return lightCount_;
	}

	// CPU time of the last light tree build.
	float GetLightTreeBuildMs() const
	{
		return lightTreeBuildMs_;
	}

	// World-space bounds over every instanced primitive, for framing the camera.
	float3 GetBoundsMin() const
	{
		return boundsMin_;
	}

	float3 GetBoundsMax() const
	{
		return boundsMax_;
	}

private:
	void UploadStreams(UploadBatchHandle batch);
	void UploadMaterials(const Scene::GLTFScene &scene, UploadBatchHandle &batch);
	uint32_t UploadTexture(const Scene::Texture &texture,
						   Format format,
						   const char *debugName,
						   UploadBatchHandle &batch);
	ResourceHandle UploadStream(
			const void *data, uint64_t sizeInBytes, uint32_t stride, const char *debugName, UploadBatchHandle batch);
	void ComputeBounds(const Scene::GLTFScene &scene);
	void FillInstances(const Scene::GLTFScene &scene);
	void DestroyLights();

	RenderContext *renderContext_ = nullptr;

	// CPU-side staging for the shared streams, released once BuildGPUScene has uploaded them.
	std::vector<float> positions_;
	std::vector<float> texCoords_;
	std::vector<float> normals_;
	std::vector<float> tangents_;
	std::vector<uint32_t> indices_;
	std::vector<uint32_t> meshletData_; // Scene::Meshlet rows, raw
	std::vector<uint32_t> meshletVertices_;
	std::vector<uint32_t> meshletTriangles_;

	ResourceHandle streamBuffers_[(uint32_t) GPUStream::Count] = {};
	ResourceHandle primitiveBuffer_ = InvalidResourceHandle;
	ResourceHandle materialBuffer_ = InvalidResourceHandle;
	ResourceHandle instanceBuffer_ = InvalidResourceHandle;
	uint32_t primitiveBufferIndex_ = InvalidResourceDescriptorIndex;
	uint32_t materialBufferIndex_ = InvalidResourceDescriptorIndex;

	// One set per frame in flight. GPU upload heap buffers, so the CPU writes them in place and the
	// GPU reads them from video memory.
	struct LightFrameBuffers
	{
		ResourceHandle lights = InvalidResourceHandle;
		ResourceHandle nodes = InvalidResourceHandle;
		size_t lightCapacity = 0;
		size_t nodeCapacity = 0;
	};

	std::vector<LightFrameBuffers> lightFrames_;
	uint32_t lightFrame_ = 0;
	uint32_t lightCount_ = 0;
	float lightTreeBuildMs_ = 0.0f;

	std::vector<ResourceHandle> textures_;
	std::vector<uint32_t> materialAlphaModes_; // GpuAlphaMode per GpuMaterial row; decides each BLAS's geometry flags
	uint64_t pendingUploadBytes_ = 0;

	std::vector<AccelerationStructureHandle> blases_; // one per primitive desc, invalid for skipped primitives
	AccelerationStructureHandle tlas_ = InvalidAccelerationStructureHandle;
	ResourceHandle tlasSRV_ = InvalidResourceHandle;

	float3 boundsMin_ = { 0.0f, 0.0f, 0.0f };
	float3 boundsMax_ = { 0.0f, 0.0f, 0.0f };
};
