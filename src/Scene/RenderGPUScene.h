#pragma once

#include "GPUScene.h"
#include "RenderContext.h"

#include "LightTreeBuilder.h"
#include "Maths.h"
#include "PathTracerInterop.h"
#include "ShaderInterop.h"

#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace Scene
{
	class Texture;
}

class SceneCacheWriter;

struct MaterialGeometryFlags
{
	explicit MaterialGeometryFlags(const MaterialData &material)
	{
		const bool passThrough = (material.transmissionFlags & TRANSMISSION_PASS_THROUGH) != 0;
		isOpaque = material.alphaMode == ALPHA_MODE_OPAQUE && !passThrough;
		// Shadow rays multiply in what these let through, once per triangle.
		noDuplicateAnyHit = passThrough || material.alphaMode == ALPHA_MODE_BLEND;
		// A mask's unknown micro-triangles may be taken as opaque or transparent. A blended surface's are
		// partly there, which only its any-hit shaders get right for camera and bounce rays (shadow rays force
		// 2-state on every micromap, see TraceShadowRay).
		opacityMicromap2State = material.alphaMode == ALPHA_MODE_MASK;
	}

	bool isOpaque;
	bool noDuplicateAnyHit;
	bool opacityMicromap2State;
};

class RenderGPUScene final : public Scene::GPUScene
{
public:
	explicit RenderGPUScene(RenderContext *renderContext);
	~RenderGPUScene() override;

	RenderGPUScene(const RenderGPUScene &) = delete;
	RenderGPUScene &operator=(const RenderGPUScene &) = delete;

	void BuildGPUScene(const Scene::GLTFScene &scene) override;
	void BuildGPUScene(const Scene::GLTFScene &scene, const std::filesystem::path &cachePath);
	bool LoadGPUScene(const std::filesystem::path &cachePath);

	// Refreshes the per-instance transforms in the instance table. The TLAS is built once at start
	// up and is not refitted here yet.
	void UpdateInstances(const Scene::GLTFScene &scene) override;

	void BuildAccelerationStructures();

	// Picks the emissive scene instances that move, counted in instance order among those with an
	// emissive primitive (the load logs each one). Their lights go at the end of the light buffer, in a
	// light tree of their own under the root group's last slot, so moving them rebuilds only that tree.
	// Call before BuildGPUScene or LoadGPUScene.
	void SetDynamicEmitters(std::vector<uint32_t> emissiveInstances)
	{
		dynamicEmitterSelection_ = std::move(emissiveInstances);
	}

	// The scene instances SetDynamicEmitters picked.
	uint32_t GetDynamicInstanceCount() const
	{
		return static_cast<uint32_t>(dynamicInstances_.size());
	}

	// A dynamic instance's world transform at load, a float3x4 in the InstanceGPU layout.
	const float *GetDynamicInstanceRestTransform(uint32_t dynamicInstance) const
	{
		return dynamicRestTransforms_[dynamicInstance].data();
	}

	// Moves a dynamic instance: its TLAS instances follow at the next UpdateAccelerationStructures and
	// its lights at the next UpdateLights.
	void SetDynamicInstanceTransform(uint32_t dynamicInstance, const float (&worldMatrix)[12]);

	// The scene instances with an emissive primitive, numbered as SetDynamicEmitters counts them.
	uint32_t GetEmissiveInstanceCount() const
	{
		return static_cast<uint32_t>(emissiveInstances_.size());
	}

	// The dynamic instance an emissive instance is, or UINT32_MAX for a static one.
	uint32_t GetDynamicInstanceOfEmitter(uint32_t emissiveInstance) const
	{
		return emissiveInstance < emissiveInstances_.size() ? dynamicOfInstance_[emissiveInstances_[emissiveInstance]]
															: UINT32_MAX;
	}

	void SetEmissionScale(uint32_t emissiveInstance, float scale);

	// Rebuilds the TLASes if a dynamic instance has moved since the last call. Records on the current
	// frame's graphics command list, so call it after BeginGraphicsCommandList and before any tracing.
	void UpdateAccelerationStructures();

	// Builds the static emitters' light tree on the first call and again whenever the settings change,
	// and the dynamic emitters' tree then too and whenever one of them has moved; otherwise it does
	// nothing. Call after BeginFrame, as a build writes the current frame's light buffers.
	void UpdateLights(const LightTreeBuildSettings &settings);

	bool HasGeometry() const
	{
		return tlas_ != InvalidAccelerationStructureHandle;
	}

	ResourceHandle GetTLASSRV(bool opacityMicromaps = false) const
	{
		return opacityMicromaps && HasOpacityMicromaps() ? ommTlasSRV_ : tlasSRV_;
	}

	// Some primitive has opacity micromaps and the device can trace them.
	bool HasOpacityMicromaps() const
	{
		return ommTlas_ != InvalidAccelerationStructureHandle;
	}

	// Bindless index of the micromap TLAS whose alpha-tested instances read their micromaps in 2-state mode,
	// for the rays past RenderSettings::opacityMicromap2StateBounce. Only with HasOpacityMicromaps.
	uint32_t GetOpacityMicromap2StateTLASIndex() const
	{
		const ResourceHandle srv = omm2StateTlasSRV_ != InvalidResourceHandle ? omm2StateTlasSRV_ : ommTlasSRV_;
		return srv != InvalidResourceHandle ? renderContext_->GetResourceViewInfo(srv).descriptorIndex : 0;
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

	ResourceHandle GetStreamBuffer(GPUStream stream) const
	{
		return streamBuffers_[(uint32_t) stream];
	}

	// Every primitive's vertices as GpuVertex rows, indexed by PrimitiveGPUDesc::vertexOffset.
	ResourceHandle GetVertexBuffer() const
	{
		return vertexBuffer_;
	}

	ResourceHandle GetMaterialBuffer() const
	{
		return materialBuffer_;
	}

	// Rows in the material buffer, including the default material at the end.
	uint32_t GetMaterialCount() const
	{
		return static_cast<uint32_t>(materialGeometry_.size());
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

	// GpuLightAliasEntry per light, for power light selection.
	ResourceHandle GetLightPowerBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].power;
	}

	// Maps a ray hit to the light of the emissive triangle it hit (uint). The first entry per TLAS
	// instance (indexed by InstanceIndex()) is the offset of that instance's triangle table in the
	// same buffer, or INVALID_LIGHT_INDEX if it does not emit; the table then holds the light index
	// of each triangle (indexed by PrimitiveIndex()), INVALID_LIGHT_INDEX for those that emit nothing.
	ResourceHandle GetEmitterBuffer() const
	{
		return emitterBuffer_;
	}

	// Per light, its emissive instance (uint): where its scale is in the emission scale buffer.
	ResourceHandle GetLightEmitterBuffer() const
	{
		return lightEmitterBuffer_;
	}

	// Per emissive instance, the factor SetEmissionScale gave its emission (float).
	ResourceHandle GetEmissionScaleBuffer() const
	{
		return lightFrames_.empty() ? InvalidResourceHandle : lightFrames_[lightFrame_].emissionScales;
	}

	uint32_t GetLightCount() const
	{
		return lightCount_;
	}

	// Lights before this index in the light buffer are static emitters, those from it on dynamic ones.
	uint32_t GetStaticLightCount() const
	{
		return staticLightCount_;
	}

	// The static emitters' share of the flux, for power selection.
	float GetStaticLightPowerShare() const
	{
		return staticLightPowerShare_;
	}

	// CPU time of the last build of the static and of the dynamic emitters' light tree.
	float GetStaticLightTreeBuildMs() const
	{
		return staticLightTreeBuildMs_;
	}

	float GetDynamicLightTreeBuildMs() const
	{
		return dynamicLightTreeBuildMs_;
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
	void BuildMaterials(const Scene::GLTFScene &scene, UploadBatchHandle &batch, std::vector<GpuMaterial> &materials);
	void UploadMaterials(const std::vector<GpuMaterial> &materials, UploadBatchHandle batch);
	// Fills the emissive lights and returns the emitter buffer's contents.
	std::vector<uint32_t> CollectEmitters(const Scene::GLTFScene &scene, const std::vector<GpuMaterial> &materials);
	// Moves the lights of the instances dynamicEmitterSelection_ picks to the end of the light buffer,
	// keeping the order within each part, and renumbers them in the emitter buffer's contents. Also numbers
	// the emissive instances and resets their emission scales.
	void PartitionDynamicLights(std::vector<uint32_t> &emitters);
	// The dynamic lights in world space, from the current transforms of their instances.
	void TransformDynamicLights();
	void UploadEmitters(const std::vector<uint32_t> &emitters, UploadBatchHandle batch);
	// Fills the opacity micromap tables from the base colour alpha of the alpha-tested primitives.
	void BakeOpacityMicromaps(const Scene::GLTFScene &scene, const std::vector<GpuMaterial> &materials);
	void UploadOpacityMicromapIndices(UploadBatchHandle batch);
	uint32_t UploadTexture(const Scene::Texture &texture,
						   Format format,
						   const char *debugName,
						   UploadBatchHandle &batch);
	// The RGBA8 mips as a BC7 chain, or false when they stay uncompressed.
	bool CompressBC7(const TextureDesc &desc, std::span<const TextureMipData> mips, DirectX::ScratchImage &compressed);
	// Returns the texture's bindless index.
	uint32_t UploadTextureMips(const TextureDesc &desc, std::span<const TextureMipData> mips, UploadBatchHandle &batch);

	// Writes (SceneCacheWriter) or reads (SceneCacheReader) everything but the textures.
	template<typename Archive>
	void SerialiseTables(Archive &archive,
						 std::vector<GpuMaterial> &materials,
						 std::vector<uint32_t> &emitters,
						 std::vector<GpuLight> &lights);
	void WriteSceneCache(SceneCacheWriter &writer,
						 const std::vector<GpuMaterial> &materials,
						 std::vector<uint32_t> &emitters);
	// Checks the indices between the tables loaded from a cache.
	bool ValidateTables(const std::vector<GpuMaterial> &materials, uint32_t textureCount) const;
	void ReleaseStaging();
	ResourceHandle UploadStream(
			const void *data, uint64_t sizeInBytes, uint32_t stride, const char *debugName, UploadBatchHandle batch);
	void ComputeBounds(const Scene::GLTFScene &scene);
	void FillInstances(const Scene::GLTFScene &scene);
	void DestroyLights();

	RenderContext *renderContext_ = nullptr;

	// CPU-side staging for the shared streams, released once BuildGPUScene has uploaded them.
	std::vector<GpuVertex> vertices_;
	std::vector<uint32_t> indices_;
	std::vector<uint32_t> meshletData_; // Scene::Meshlet rows, raw
	std::vector<uint32_t> meshletVertices_;
	std::vector<uint32_t> meshletTriangles_;

	// The textures BuildGPUScene has uploaded, by their pixels (shared by every material using the
	// image) and format, so each image goes to the GPU once.
	std::map<std::pair<const uint8_t *, Format>, uint32_t> uploadedTextures_;

	ResourceHandle streamBuffers_[(uint32_t) GPUStream::Count] = {};
	ResourceHandle vertexBuffer_ = InvalidResourceHandle;
	ResourceHandle primitiveBuffer_ = InvalidResourceHandle;
	ResourceHandle materialBuffer_ = InvalidResourceHandle;
	ResourceHandle instanceBuffer_ = InvalidResourceHandle;
	ResourceHandle emitterBuffer_ = InvalidResourceHandle;
	ResourceHandle lightEmitterBuffer_ = InvalidResourceHandle;
	uint32_t primitiveBufferIndex_ = InvalidResourceDescriptorIndex;
	uint32_t materialBufferIndex_ = InvalidResourceDescriptorIndex;

	// One set per frame in flight, so a rebuild can write the current frame's set while the GPU may
	// still read the last one. GPU upload heap buffers, so the CPU writes them in place and the GPU
	// reads them from video memory.
	struct LightFrameBuffers
	{
		ResourceHandle lights = InvalidResourceHandle;
		ResourceHandle nodes = InvalidResourceHandle;
		ResourceHandle leaves = InvalidResourceHandle;
		ResourceHandle parents = InvalidResourceHandle;
		ResourceHandle leafLights = InvalidResourceHandle;
		ResourceHandle power = InvalidResourceHandle;
		ResourceHandle emissionScales = InvalidResourceHandle; // written whole on every update
		size_t lightCapacity = 0;
		size_t nodeCapacity = 0;
		size_t leafCapacity = 0;
		size_t parentCapacity = 0;
		size_t leafLightCapacity = 0;
		size_t powerCapacity = 0;
		size_t emissionScaleCapacity = 0;
		uint32_t staticTreeGeneration = 0; // of the static part these hold, 0 for none
	};

	// The light tree input: the emissive triangles (one-sided area lights, with the average of their
	// emissive texture as the flux scale), the static emitters' first, then from staticLightCount_ on
	// the dynamic ones'.
	std::vector<GpuLight> emissiveLights_;
	std::vector<float> emissiveFluxScales_;
	uint32_t staticLightCount_ = 0;
	LightTreeBuildSettings lightTreeSettings_; // of the last build
	bool lightTreeBuilt_ = false;
	// Kept to write the static part of the light buffers of each frame in flight, once per build.
	LightTree staticLightTree_;
	uint32_t staticTreeGeneration_ = 0; // counts static tree builds
	// Kept for updates that only change emission scales, which still write the dynamic part.
	LightTree dynamicLightTree_;

	// Per emissive instance, its scene instance and SetEmissionScale's factor.
	std::vector<uint32_t> emissiveInstances_;
	std::vector<float> emissionScales_;
	std::vector<uint32_t> lightEmitters_; // per light, its emissive instance
	bool emissionScalesChanged_ = false;

	// Emissive instances that move (SetDynamicEmitters), and the lights of their triangles.
	std::vector<uint32_t> dynamicEmitterSelection_;
	std::vector<uint32_t> dynamicInstances_; // into instances
	std::vector<uint32_t> dynamicOfInstance_; // per scene instance, into dynamicInstances_ or UINT32_MAX
	std::vector<std::array<float, 12>> dynamicRestTransforms_;
	std::vector<uint8_t> dynamicRestMirrored_; // the rest transform mirrors, so CollectEmitters swapped corners
	std::vector<GpuLight> dynamicObjectLights_; // per dynamic light, its triangle in object space
	std::vector<uint32_t> dynamicLightInstance_; // per dynamic light, into dynamicInstances_
	bool dynamicLightsMoved_ = false;
	bool dynamicLightsRescaled_ = false; // a dynamic instance's emission scale changed
	bool dynamicTLASMoved_ = false;

	std::vector<LightFrameBuffers> lightFrames_;
	uint32_t lightFrame_ = 0;
	uint32_t lightCount_ = 0;
	float staticLightPowerShare_ = 1.0f;
	float staticLightTreeBuildMs_ = 0.0f;
	float dynamicLightTreeBuildMs_ = 0.0f;
	uint32_t lightTreeMaxDepth_ = 0;
	float lightTreeAverageDepth_ = 0.0f;

	std::vector<ResourceHandle> textures_;
	// Per GpuMaterial row; decides the geometry flags of each BLAS that uses it.
	std::vector<MaterialGeometryFlags> materialGeometry_;
	uint64_t pendingUploadBytes_ = 0;
	SceneCacheWriter *cacheWriter_ = nullptr; // during a BuildGPUScene that writes a cache
	double mipBuildMs_ = 0.0; // CPU time spent filtering texture mips during the last BuildGPUScene
	std::unique_ptr<ITextureCompressor> textureCompressor_; // during a BuildGPUScene
	double compressMs_ = 0.0; // time spent compressing textures to BC7 during the last BuildGPUScene
	uint32_t compressedTextures_ = 0;
	uint32_t uncompressedTextures_ = 0; // left as RGBA8 by CompressBC7 (not whole 4x4 blocks, or it failed)

	std::vector<AccelerationStructureHandle> blases_; // one per primitive desc, invalid for skipped primitives
	// The TLAS instances as built, in InstanceIndex() order, for UpdateAccelerationStructures.
	std::vector<TLASInstanceRef> tlasRefs_;
	std::vector<TLASInstance> tlasInstances_;
	std::vector<TLASInstance> ommTlasInstances_;
	std::vector<TLASInstance> omm2StateTlasInstances_;
	AccelerationStructureHandle tlas_ = InvalidAccelerationStructureHandle;
	ResourceHandle tlasSRV_ = InvalidResourceHandle;

	// DXR 1.2 opacity micromaps of the alpha-tested primitives (BakeOpacityMicromaps). The micromaps and
	// their states only stay on the CPU until BuildAccelerationStructures has built the array.
	std::vector<uint8_t> ommData_;
	std::vector<OpacityMicromap> ommMicromaps_;
	// Per triangle of the primitives with micromaps, an index into ommMicromaps_ or an
	// OpacityMicromapSpecialIndex; released once uploaded.
	std::vector<int32_t> ommIndices_;
	std::vector<uint32_t> ommIndexOffsets_; // per primitive desc, its first triangle in ommIndices_, or UINT32_MAX
	ResourceHandle ommIndexBuffer_ = InvalidResourceHandle;
	AccelerationStructureHandle ommArray_ = InvalidAccelerationStructureHandle;
	// The alpha-tested primitives' BLASes again, with the micromaps linked, and a TLAS that uses them
	// in place of the plain ones, so the micromaps can be switched on and off at run time.
	std::vector<AccelerationStructureHandle> ommBlases_;
	AccelerationStructureHandle ommTlas_ = InvalidAccelerationStructureHandle;
	ResourceHandle ommTlasSRV_ = InvalidResourceHandle;
	// The micromap TLAS again with the masks' instances flagged to read their micromaps in 2-state mode,
	// which a ray flag would also force on the blended surfaces'. Invalid when no mask has micromaps.
	AccelerationStructureHandle omm2StateTlas_ = InvalidAccelerationStructureHandle;
	ResourceHandle omm2StateTlasSRV_ = InvalidResourceHandle;

	float3 boundsMin_ = { 0.0f, 0.0f, 0.0f };
	float3 boundsMax_ = { 0.0f, 0.0f, 0.0f };
};
