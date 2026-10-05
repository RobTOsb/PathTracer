#pragma once

#include "GPUScene.h"
#include "RenderContext.h"

#include "Maths.h"

#include <vector>

namespace Scene
{
	class Texture;
}

// GPUScene backed by the RenderContext abstraction. Packs every primitive into the shared stream
// buffers described by GPUScene, uploads materials/textures as bindless resources, and builds one
// BLAS per primitive plus a single TLAS over every (node, primitive) pair.
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

	// Records texture transitions and the BLAS/TLAS builds on the current graphics command list.
	// Call once after BuildGPUScene, between BeginGraphicsCommandList and EndGraphicsCommandList.
	void BuildAccelerationStructures();

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

	ResourceHandle GetPrimitiveBuffer() const
	{
		return primitiveBuffer_;
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
	uint32_t UploadTexture(const Scene::Texture &texture, Format format, const char *debugName, UploadBatchHandle &batch);
	ResourceHandle UploadStream(const void *data, uint64_t sizeInBytes, uint32_t stride, const char *debugName, UploadBatchHandle batch);
	void ComputeBounds(const Scene::GLTFScene &scene);
	void FillInstances(const Scene::GLTFScene &scene);

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

	std::vector<ResourceHandle> textures_;
	std::vector<bool> materialIsOpaque_; // per GpuMaterial row; decides each BLAS's opaque flag
	uint64_t pendingUploadBytes_ = 0;

	std::vector<AccelerationStructureHandle> blases_; // one per primitive desc, invalid for skipped primitives
	AccelerationStructureHandle tlas_ = InvalidAccelerationStructureHandle;
	ResourceHandle tlasSRV_ = InvalidResourceHandle;

	float3 boundsMin_ = { 0.0f, 0.0f, 0.0f };
	float3 boundsMax_ = { 0.0f, 0.0f, 0.0f };
};
