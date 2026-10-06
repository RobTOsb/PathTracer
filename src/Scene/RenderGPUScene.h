#pragma once

#include "GPUScene.h"
#include "RenderContext.h"

#include "Maths.h"
#include "PathTracerInterop.h"
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

	void UpdateLights(std::span<const GpuLight> lights, uint32_t maxLeafLights);

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

	ResourceHandle GetLightBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].lights;
	}

	ResourceHandle GetLightTreeNodeBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].nodes;
	}

	ResourceHandle GetLightLeafBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].leaves;
	}

	// Parent node of each light tree group (uint), for the shader's PMF walk.
	ResourceHandle GetLightTreeParentBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].parents;
	}

	ResourceHandle GetLightTreeLeafLightBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].leafLights;
	}

	// Maps a ray hit to the light of the emissive triangle it hit (uint). The first entry per TLAS
	// instance (indexed by InstanceIndex()) is the offset of that instance's triangle table in the
	// same buffer, or INVALID_LIGHT_INDEX if it does not emit; the table then holds the light index
	// of each triangle (indexed by PrimitiveIndex()), INVALID_LIGHT_INDEX for those that emit nothing.
	ResourceHandle GetEmitterBuffer() const
	{
		return emitterBuffer_;
	}

	// Emissive triangles at the front of the light buffer.
	uint32_t GetEmissiveTriangleCount() const
	{
		return static_cast<uint32_t>(emissiveLights_.size());
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

	// Levels of node groups above the deepest light and above the average light in the last tree.
	uint32_t GetLightTreeMaxDepth() const
	{
		return lightTreeMaxDepth_;
	}

	float GetLightTreeAverageDepth() const
	{
		return lightTreeAverageDepth_;
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
	// One TLAS instance: a primitive of a scene instance's mesh.
	struct TLASInstanceRef
	{
		uint32_t instance; // into instances
		uint32_t meshPrimitive; // primitive within the instance's mesh
		uint32_t primitive; // into primitiveDescs
	};

	// The TLAS instances in the order BuildAccelerationStructures adds them, which is their InstanceIndex().
	std::vector<TLASInstanceRef> EnumerateTLASInstances() const;

	void UploadStreams(UploadBatchHandle batch);
	void UploadMaterials(const Scene::GLTFScene &scene, UploadBatchHandle &batch, std::vector<GpuMaterial> &materials);
	void CollectEmitters(const Scene::GLTFScene &scene,
						 const std::vector<GpuMaterial> &materials,
						 UploadBatchHandle batch);
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
	ResourceHandle emitterBuffer_ = InvalidResourceHandle;
	uint32_t primitiveBufferIndex_ = InvalidResourceDescriptorIndex;
	uint32_t materialBufferIndex_ = InvalidResourceDescriptorIndex;

	// One set per frame in flight. GPU upload heap buffers, so the CPU writes them in place and the
	// GPU reads them from video memory.
	struct LightFrameBuffers
	{
		ResourceHandle lights = InvalidResourceHandle;
		ResourceHandle nodes = InvalidResourceHandle;
		ResourceHandle leaves = InvalidResourceHandle;
		ResourceHandle parents = InvalidResourceHandle;
		ResourceHandle leafLights = InvalidResourceHandle;
		size_t lightCapacity = 0;
		size_t nodeCapacity = 0;
		size_t leafCapacity = 0;
		size_t parentCapacity = 0;
		size_t leafLightCapacity = 0;
	};

	// The light tree input: the static emissive triangles (one-sided area lights, with the average
	// of their emissive texture as the flux scale) followed by the lights passed to UpdateLights.
	std::vector<GpuLight> emissiveLights_;
	std::vector<float> emissiveFluxScales_;
	std::vector<GpuLight> lightInput_;
	std::vector<float> lightFluxScales_;

	std::vector<LightFrameBuffers> lightFrames_;
	uint32_t lightFrame_ = 0;
	uint32_t lightCount_ = 0;
	float lightTreeBuildMs_ = 0.0f;
	uint32_t lightTreeMaxDepth_ = 0;
	float lightTreeAverageDepth_ = 0.0f;

	std::vector<ResourceHandle> textures_;
	std::vector<uint32_t> materialAlphaModes_; // GpuAlphaMode per GpuMaterial row; decides each BLAS's geometry flags
	uint64_t pendingUploadBytes_ = 0;

	std::vector<AccelerationStructureHandle> blases_; // one per primitive desc, invalid for skipped primitives
	AccelerationStructureHandle tlas_ = InvalidAccelerationStructureHandle;
	ResourceHandle tlasSRV_ = InvalidResourceHandle;

	float3 boundsMin_ = { 0.0f, 0.0f, 0.0f };
	float3 boundsMax_ = { 0.0f, 0.0f, 0.0f };
};
