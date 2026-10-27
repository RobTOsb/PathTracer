#pragma once

#include "Animation.h"
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

	// The scene's glTF animations, which play together: whether there are any, and the time of the last key
	// of any (they loop over it).
	bool HasAnimation() const
	{
		return !animationChannels_.empty();
	}

	float GetAnimationDuration() const
	{
		return animationDuration_;
	}

	// Poses the animated nodes at this many seconds into the animations. The scene instances they move are
	// the dynamic ones: their TLAS instances follow at the next UpdateAccelerationStructures and their lights,
	// which have a light tree of their own under the root group's last slot, at the next UpdateLights.
	void Animate(float seconds);

	// The scene instances an animation moves, through their own node or a parent.
	uint32_t GetDynamicInstanceCount() const
	{
		return static_cast<uint32_t>(dynamicInstances_.size());
	}

	// The scene instances with an emissive primitive, numbered in instance order (the load logs each one).
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

	// Picks the current frame's set of TLASes and rebuilds it if the dynamic instances have moved since it was
	// last built, and keeps last frame's transforms for the motion vectors. Records on the current frame's
	// graphics command list, so call it once per frame after BeginGraphicsCommandList and before any tracing.
	void UpdateAccelerationStructures();

	// Per TLAS instance (InstanceIndex()), last frame's object-to-world transform as three float4 rows of
	// the float3x4 that ObjectToWorld3x4() gives, for the motion vectors of moving geometry.
	ResourceHandle GetPreviousTransformBuffer() const
	{
		return previousTransforms_.empty() ? InvalidResourceHandle : previousTransforms_[previousTransformFrame_];
	}

	// Builds the static emitters' light tree on the first call and again whenever the settings change,
	// and the dynamic emitters' tree then too and whenever one of them has moved; otherwise it does
	// nothing. Call after BeginFrame, as a build writes the current frame's light buffers.
	void UpdateLights(const LightTreeBuildSettings &settings);

	bool HasGeometry() const
	{
		return !frameTlases_.empty();
	}

	// The current frame's TLAS (see UpdateAccelerationStructures).
	ResourceHandle GetTLASSRV(bool opacityMicromaps = false) const
	{
		const FrameTLASes &frame = frameTlases_[tlasFrame_];
		return opacityMicromaps && HasOpacityMicromaps() ? frame.ommTlasSRV : frame.tlasSRV;
	}

	// Last frame's TLAS, of the same kind, which still holds the dynamic instances where last frame traced them.
	ResourceHandle GetPreviousTLASSRV(bool opacityMicromaps = false) const
	{
		const FrameTLASes &frame = frameTlases_[PreviousTLASFrame()];
		return opacityMicromaps && HasOpacityMicromaps() ? frame.ommTlasSRV : frame.tlasSRV;
	}

	// The dynamic instances are elsewhere in the current frame's TLASes than in last frame's.
	bool MovedSinceLastFrame() const
	{
		return !frameTlases_.empty() && frameTlases_[tlasFrame_].poseGeneration != frameTlases_[PreviousTLASFrame()].poseGeneration;
	}

	// Some primitive has opacity micromaps and the device can trace them.
	bool HasOpacityMicromaps() const
	{
		return !frameTlases_.empty() && frameTlases_[0].ommTlas != InvalidAccelerationStructureHandle;
	}

	// Bindless index of the current frame's micromap TLAS whose alpha-tested instances read their micromaps in
	// 2-state mode, for the rays past RenderSettings::opacityMicromap2StateBounce. Only with HasOpacityMicromaps.
	uint32_t GetOpacityMicromap2StateTLASIndex() const
	{
		const FrameTLASes &frame = frameTlases_[tlasFrame_];
		const ResourceHandle srv = frame.omm2StateTlasSRV != InvalidResourceHandle ? frame.omm2StateTlasSRV : frame.ommTlasSRV;
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

	// A node an animation moves, through its own channels or a parent's. The table lists parents before
	// their children.
	struct AnimatedNode
	{
		uint32_t parent; // into animatedNodes_, or kStaticParent for a parent that does not move
		uint32_t instance; // into instances, or UINT32_MAX for a node without a mesh
		// The local transform at rest, in parts for the channels to replace, and whole for nodes without any.
		float translation[3];
		float rotation[4]; // quaternion xyzw
		float scale[3];
		float localMatrix[12];
		float parentMatrix[12]; // the world matrix of a parent that does not move, identity for a root
	};
	static constexpr uint32_t kStaticParent = UINT32_MAX;

	// A glTF animation channel, its keys in animationTimes_ and animationValues_ (see Scene::SampleAnimation).
	struct AnimationChannel
	{
		uint32_t node; // into animatedNodes_
		Scene::AnimationPath path;
		Scene::AnimationInterpolation interpolation;
		uint32_t keyCount;
		uint32_t firstTime;
		uint32_t firstValue;
	};

	void UploadStreams(UploadBatchHandle batch);
	void BuildMaterials(const Scene::GLTFScene &scene, UploadBatchHandle &batch, std::vector<GpuMaterial> &materials);
	void UploadMaterials(const std::vector<GpuMaterial> &materials, UploadBatchHandle batch);
	// Fills the emissive lights and returns the emitter buffer's contents.
	std::vector<uint32_t> CollectEmitters(const Scene::GLTFScene &scene, const std::vector<GpuMaterial> &materials);
	// Moves the lights of the dynamic instances to the end of the light buffer,
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
	// Returns each node's scene instance, UINT32_MAX for those without a mesh.
	std::vector<uint32_t> FillInstances(const Scene::GLTFScene &scene);
	// Fills the animation tables from the glTF's animations.
	void BuildAnimation(const Scene::GLTFScene &scene, const std::vector<uint32_t> &instanceOfNode);
	// From the animation tables, their duration and the dynamic instances. Call before PartitionDynamicLights.
	void PrepareAnimation();
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

	// The animations, kept in the scene cache.
	std::vector<AnimatedNode> animatedNodes_;
	std::vector<AnimationChannel> animationChannels_;
	std::vector<float> animationTimes_;
	std::vector<float> animationValues_;
	float animationDuration_ = 0.0f;

	// The scene instances that move, and the lights of their triangles.
	std::vector<uint32_t> dynamicInstances_; // into instances
	std::vector<uint32_t> dynamicOfInstance_; // per scene instance, into dynamicInstances_ or UINT32_MAX
	std::vector<uint8_t> dynamicRestMirrored_; // the rest transform mirrors, so CollectEmitters swapped corners
	// Per dynamic instance, its world transform when UpdateAccelerationStructures last ran.
	std::vector<std::array<float, 12>> dynamicLastFrameTransforms_;
	std::vector<GpuLight> dynamicObjectLights_; // per dynamic light, its triangle in object space
	std::vector<uint32_t> dynamicLightInstance_; // per dynamic light, into dynamicInstances_
	bool dynamicLightsMoved_ = false;
	bool dynamicLightsRescaled_ = false; // a dynamic instance's emission scale changed
	uint32_t dynamicPoseGeneration_ = 0; // counts the poses Animate has moved the dynamic instances to

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
	uint32_t tlasInstancesGeneration_ = 0; // the pose the TLAS instances above hold
	std::vector<uint32_t> dynamicTlasInstances_; // into tlasRefs_, those of the dynamic instances
	// GetPreviousTransformBuffer's, one per frame in flight. Only the dynamic instances' rows change.
	std::vector<ResourceHandle> previousTransforms_;
	uint32_t previousTransformFrame_ = 0;
	
	struct FrameTLASes
	{
		AccelerationStructureHandle tlas = InvalidAccelerationStructureHandle;
		AccelerationStructureHandle ommTlas = InvalidAccelerationStructureHandle;
		AccelerationStructureHandle omm2StateTlas = InvalidAccelerationStructureHandle;
		ResourceHandle tlasSRV = InvalidResourceHandle;
		ResourceHandle ommTlasSRV = InvalidResourceHandle;
		ResourceHandle omm2StateTlasSRV = InvalidResourceHandle;
		uint32_t poseGeneration = 0; // the pose of the dynamic instances it was built with
	};
	std::vector<FrameTLASes> frameTlases_; // empty without instanced geometry
	uint32_t tlasFrame_ = 0; // the set the current frame traces

	// The set last frame traced, which the current frame does not rebuild.
	uint32_t PreviousTLASFrame() const
	{
		const uint32_t frames = static_cast<uint32_t>(frameTlases_.size());
		return (tlasFrame_ + frames - 1) % frames;
	}

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
	// The alpha-tested primitives' BLASes again, with the micromaps linked, for the micromap TLASes.
	std::vector<AccelerationStructureHandle> ommBlases_;

	float3 boundsMin_ = { 0.0f, 0.0f, 0.0f };
	float3 boundsMax_ = { 0.0f, 0.0f, 0.0f };
};
