#pragma once

#include "Animation/AnimationData.h"
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
#include <string>
#include <utility>
#include <vector>

class FrameGraph;

namespace rg
{
	struct Pass;
}

namespace Scene
{
	class Texture;
}

class SceneCacheWriter;

struct MaterialGeometryFlags
{
	explicit MaterialGeometryFlags(const MaterialData &material)
	{
		passThrough = IsPassThroughMaterial(material);
		isOpaque = material.alphaMode == ALPHA_MODE_OPAQUE && !passThrough;
		// Shadow rays multiply in what these let through, once per triangle.
		noDuplicateAnyHit = passThrough || material.alphaMode == ALPHA_MODE_BLEND;
		// A mask's unknown micro-triangles may be taken as opaque or transparent. A blended surface's are
		// partly there, which only its any-hit shaders get right for camera and bounce rays (shadow rays force
		// 2-state on every micromap, see TraceShadowRay).
		opacityMicromap2State = material.alphaMode == ALPHA_MODE_MASK;
		instanceMask = material.alphaMode == ALPHA_MODE_BLEND ? INSTANCE_MASK_BLENDED : INSTANCE_MASK_SOLID;
	}

	bool passThrough; // IsPassThroughMaterial: shadow rays need its any-hit for every triangle, so no micromaps
	bool isOpaque;
	bool noDuplicateAnyHit;
	bool opacityMicromap2State;
	uint32_t instanceMask; // InstanceMaskBits
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

	// Whether the static, non-emissive primitives that share a world transform and an instance mask bit (the
	// parts of one plant, say) are one TLAS instance of a BLAS with a geometry per part, its prototype's,
	// rather than a TLAS instance and BLAS each. Hits find their primitive row at InstanceID() + GeometryIndex()
	// either way. SetMergeInstances rebuilds every BLAS and TLAS for the new choice: call it with the GPU idle,
	// after BeginGraphicsCommandList and SetDescriptorHeaps, before UpdateAccelerationStructures.
	bool GetMergeInstances() const
	{
		return mergeInstances_;
	}

	void SetMergeInstances(bool merge);

	// The scene's glTF animations and skins, with the vertices the skins deform, as tables for an
	// Animation::AnimationSystem. Valid until the scene loads again.
	Animation::AnimationData GetAnimationData() const;

	// Moves the scene to a pose an Animation::AnimationSystem evaluated from GetAnimationData(). The instances it
	// places are the dynamic ones: their TLAS instances follow at the next AddAccelerationStructurePasses and their
	// lights at the next UpdateLights. The skinned vertices go to the vertex buffer and their BLASes are rebuilt at the
	// next AddAccelerationStructurePasses. The cameras and punctual lights on posed nodes follow at once.
	void ApplyAnimationPose(const Animation::Pose &pose);

	// Primitives a skin deforms, which keep a BLAS of their own that is rebuilt for every new pose.
	uint32_t GetSkinnedPrimitiveCount() const
	{
		return static_cast<uint32_t>(skinnedPrimitives_.size());
	}

	// For the motion vectors of skinned vertices (FrameConstants::skinnedPrimitiveBases), invalid without skins: per
	// primitive desc a uint, 1 + its first skinned vertex or 0; and, this frame's after AddAccelerationStructurePasses,
	// per skinned vertex its object-space position last frame (float4). Upload heap buffers with SRVs.
	ResourceHandle GetSkinnedPrimitiveBaseBuffer() const
	{
		return skinnedPrimitiveBases_;
	}

	ResourceHandle GetPreviousSkinnedPositionBuffer() const
	{
		return previousSkinnedPositions_.empty() ? InvalidResourceHandle
												 : previousSkinnedPositions_[previousTransformFrame_];
	}

	// The scene's perspective glTF cameras: the first an animation moves is the one the view can follow while
	// it plays, and the still ones (none of their nodes or parents animated) are views to choose from.
	struct CameraPose
	{
		float3 position;
		float3 front; // the camera node's -Z
		float3 up; // its +Y
		float yFov; // radians
	};

	bool IsCameraAnimated() const
	{
		return FindAnimatedCamera() != UINT32_MAX;
	}

	// As the last ApplyAnimationPose posed it.
	CameraPose GetAnimatedCameraPose() const
	{
		return PoseOf(sceneCameras_[FindAnimatedCamera()]);
	}

	uint32_t GetStillCameraCount() const;
	CameraPose GetStillCameraPose(uint32_t still) const; // still < GetStillCameraCount(), in node order

	// The scene instances an animation or a skin moves, through their own node or a parent.
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

	// The emissive instances with a primitive of this material (its glTF index), in emissive instance order.
	std::vector<uint32_t> FindEmissiveInstances(uint32_t materialIndex) const;

	std::vector<uint32_t> ClusterEmissiveInstances(float linkDistance, uint32_t &clusterCount) const;

	void SetEmissionScale(uint32_t emissiveInstance, float scale);

	// Picks the current frame's set of TLASes and keeps last frame's transforms for the motion vectors, and adds the
	// Scene/TLAS update pass, which rebuilds the set, if the dynamic instances have moved since it was last built. With
	// skinned primitives, a new pose's vertices are copied into the vertex buffer first (Scene/Skinned vertices) and
	// the TLAS update pass rebuilds their BLASes before the TLASes. Call once per frame while recording the graph,
	// after BeginFrame (which frees the current frame's upload buffers), before any pass that traces.
	void AddAccelerationStructurePasses(FrameGraph &frame);

	// Per TLAS instance (InstanceIndex()), last frame's object-to-world transform as three float4 rows of
	// the float3x4 that ObjectToWorld3x4() gives, for the motion vectors of moving geometry.
	ResourceHandle GetPreviousTransformBuffer() const
	{
		return previousTransforms_.empty() ? InvalidResourceHandle : previousTransforms_[previousTransformFrame_];
	}

	// Swarms of moving, flashing point lights (AnimateLightSwarms in LightSwarms.slang) after the scene's lights in the
	// light buffer, to stress the light tree: each swarm is a BLAS of the two-level hierarchy. Scene-relative sizes, so
	// any scene gets them in a sensible place.
	struct LightSwarmDesc
	{
		uint32_t swarmCount = 0; // 0 = none
		uint32_t lightsPerSwarm = 0;
		float seconds = 0.0f; // where the animation is
		float speed = 1.0f;
		float flashRate = 1.5f; // flashes per second, on average
		float intensity = 0.02f; // W/sr of a light at the peak of its flash
		float radius = 0.02f; // of a swarm, a share of the scene's larger horizontal extent
		float spread = 0.6f; // share of the scene's horizontal extent, around its centre, the swarms wander
		float height = 0.2f; // share of the scene's height, from its floor, the swarms' centres fly through
		uint32_t seed = 1;
	};

	// The light tree's form and leaf size (see LightTreeBuilder).
	struct LightTreeOptions
	{
		bool twoLevel = true;
		uint32_t maxLeafLights = 4;
	};

	// Writes the current frame's uploads: the dynamic instances' transforms if they moved, the punctual (point and spot) lights and the
	// emission scales. After a scene load, or when the swarms' counts change, it also makes the light buffer and the
	// light tree builder's buffers, which waits for the GPU, and so does a new two-level layout. Call after BeginFrame,
	// which frees the current frame's buffers, and before BeginGraphicsCommandList.
	void UpdateLights(const LightTreeOptions &tree, const LightSwarmDesc &swarms);

	// Adds the move of the dynamic lights and the punctual lights into the light buffer (Lights/Dynamic lights,
	// TransformDynamicLights in DynamicLights.slang), the light swarms (Lights/Swarms), then the light tree's build
	// over every light (LightTreeBuilder::AddPasses). Call every frame after UpdateLights, before any pass that traces.
	void AddLightPasses(FrameGraph &frame);

	// Attaches to a ray tracing pass what the GPU writes of the scene that rays read: the current frame's TLAS (the
	// micromap one with opacityMicromaps) and its 2-state twin, last frame's shadow TLAS with previousShadows, and the
	// lights and their tree. The rest of the scene is uploaded once and stays out of the graph.
	void UseForTracing(rg::Pass &pass, FrameGraph &frame, bool opacityMicromaps, bool previousShadows) const;

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

	// Last frame's TLAS, which still holds the dynamic instances where last frame traced them, of the kind shadow
	// rays trace: with micromaps, the one whose masks read them in 2-state mode.
	ResourceHandle GetPreviousShadowTLASSRV(bool opacityMicromaps = false) const
	{
		const FrameTLASes &frame = frameTlases_[PreviousTLASFrame()];
		if (!opacityMicromaps || !HasOpacityMicromaps())
			return frame.tlasSRV;
		return frame.omm2StateTlasSRV != InvalidResourceHandle ? frame.omm2StateTlasSRV : frame.ommTlasSRV;
	}

	// Some TLAS instance moves (an animated node), so the TLASes may differ from frame to frame.
	bool HasDynamicInstances() const
	{
		return !dynamicTlasInstances_.empty();
	}

	// The dynamic instances are elsewhere in the current frame's TLASes than in last frame's. Never in the first
	// frame: its other set is still the one built at load, which no frame traced, so there is no last frame.
	bool MovedSinceLastFrame() const
	{
		return tlasFramesUpdated_ >= 2 &&
			   frameTlases_[tlasFrame_].poseGeneration != frameTlases_[PreviousTLASFrame()].poseGeneration;
	}

	// Some TLAS instance is a blended (ALPHA_MODE_BLEND) surface.
	bool HasBlendedInstances() const
	{
		return hasBlendedInstances_;
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
		const ResourceHandle srv =
				frame.omm2StateTlasSRV != InvalidResourceHandle ? frame.omm2StateTlasSRV : frame.ommTlasSRV;
		return srv != InvalidResourceHandle ? renderContext_->GetResourceViewInfo(srv).descriptorIndex : 0;
	}

	// Bindless indices of the material table (GpuMaterial) and primitive table (GetPrimitiveBuffer).
	uint32_t GetMaterialBufferIndex() const
	{
		return materialBufferIndex_;
	}

	uint32_t GetPrimitiveBufferIndex() const
	{
		return tlasPrimitiveBuffer_ != InvalidResourceHandle
					   ? renderContext_->GetResourceViewInfo(tlasPrimitiveBuffer_).descriptorIndex
					   : primitiveBufferIndex_;
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

	// PrimitiveGPUDesc rows, indexed by InstanceID() + GeometryIndex(): the primitive descs, then with merged
	// instances each prototype's rows again in its geometry order.
	ResourceHandle GetPrimitiveBuffer() const
	{
		return tlasPrimitiveBuffer_ != InvalidResourceHandle ? tlasPrimitiveBuffer_ : primitiveBuffer_;
	}

	ResourceHandle GetLightBuffer() const
	{
		return lights_;
	}

	// The light tree's outputs this frame, graph resources once AddLightPasses has run.
	rg::ResourceId GetLightTreeNodeBuffer() const
	{
		return lightTreeBuilder_->GetNodes();
	}

	// GpuLightTreeLeafEntry per light: its leaf position and its PMF within it, for the shader's PMF walk.
	rg::ResourceId GetLightLeafBuffer() const
	{
		return lightTreeBuilder_->GetLeafOfLight();
	}

	rg::ResourceId GetLightTreeLeafLightBuffer() const
	{
		return lightTreeBuilder_->GetLeafLights();
	}

	// GpuLightPowerEntry per light, for power light selection.
	rg::ResourceId GetLightPowerBuffer() const
	{
		return lightTreeBuilder_->GetPower();
	}

	// The two-level hierarchy's tables, SRVs (stand-ins for the single-level tree), and its shape.
	ResourceHandle GetLightTreeBlasBuffer() const
	{
		return lightTreeBuilder_->GetBlasBuffer();
	}

	ResourceHandle GetLightTreeLightBlasBuffer() const
	{
		return lightTreeBuilder_->GetLightBlasBuffer();
	}

	uint32_t GetLightTreeBlasLeafSize() const
	{
		return lightTreeBuilder_->GetBlasLeafSize();
	}

	uint32_t GetLightTreeTlasEntryOffset() const
	{
		return lightTreeBuilder_->GetTlasEntryOffset();
	}

	bool IsLightTreeTwoLevel() const
	{
		return lightTreeBuilder_->IsTwoLevel();
	}

	uint32_t GetLightTreeBlasCount() const
	{
		return lightTreeBuilder_->GetBlasCount();
	}

	uint32_t GetLightTreeMaxBlasDepth() const
	{
		return lightTreeBuilder_->GetMaxBlasDepth();
	}

	// The swarms' point lights, at the end of the light buffer.
	uint32_t GetSwarmLightCount() const
	{
		return swarmLightCount_;
	}

	// Maps a ray hit to the light of the emissive triangle it hit (uint). The first entry per TLAS
	// instance (indexed by InstanceIndex()) is the offset of that instance's triangle table in the
	// same buffer, or INVALID_LIGHT_INDEX if it does not emit; the table then holds the light index
	// of each triangle (indexed by PrimitiveIndex()), INVALID_LIGHT_INDEX for those that emit nothing.
	// Merged instances never emit (see SetMergeInstances).
	ResourceHandle GetEmitterBuffer() const
	{
		return tlasEmitterBuffer_ != InvalidResourceHandle ? tlasEmitterBuffer_ : emitterBuffer_;
	}

	// Per light, its emissive instance (uint): where its scale is in the emission scale buffer. The swarms' lights
	// share one slot after the scene's.
	ResourceHandle GetLightEmitterBuffer() const
	{
		return allLightEmitters_ != InvalidResourceHandle ? allLightEmitters_ : lightEmitterBuffer_;
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

	// The light buffer holds the emissive triangles, then the punctual (point and spot) lights. Triangle lights before
	// GetStaticLightCount() are static emitters, those from it on dynamic ones.
	uint32_t GetTriangleLightCount() const
	{
		return static_cast<uint32_t>(emissiveLights_.size());
	}

	uint32_t GetStaticLightCount() const
	{
		return staticLightCount_;
	}

	uint32_t GetPunctualLightCount() const
	{
		return static_cast<uint32_t>(punctualLights_.size());
	}

	// The light buffer's emissive triangles in its order (the dynamic ones at rest), with their flux scales (the
	// average of an emissive texture over the triangle).
	std::span<const GpuLight> GetTriangleLights() const
	{
		return emissiveLights_;
	}

	std::span<const float> GetTriangleLightFluxScales() const
	{
		return emissiveFluxScales_;
	}

	// Per light of the light buffer, its emission scale slot: a triangle light's emissive instance, then one slot per
	// punctual light.
	std::span<const uint32_t> GetLightEmitterSlots() const
	{
		return lightEmitters_;
	}

	// Levels of node groups every light is under in the last tree.
	uint32_t GetLightTreeDepth() const
	{
		return lightTreeBuilder_->GetDepth();
	}

	// The last tree's light count, lights per leaf and node mean quantisation (see LightTreeBuilder).
	uint32_t GetLightTreeLightCount() const
	{
		return lightTreeBuilder_->GetLightCount();
	}

	uint32_t GetLightTreeLeafSize() const
	{
		return lightTreeBuilder_->GetLeafSize();
	}

	float3 GetLightTreeOrigin() const
	{
		return lightTreeBuilder_->GetOrigin();
	}

	float GetLightTreeStep() const
	{
		return lightTreeBuilder_->GetStep();
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

	// A primitive instance that casts caustics, with its world bounds; reflects when a mirror lobe (metal or the coat)
	// casts them rather than only refraction, which sends them on down the sun's way.
	struct PhotonCaster
	{
		float3 lo;
		float3 hi;
		bool reflects;
	};
	void CollectPhotonCasters(float maxRoughness, std::vector<PhotonCaster> &casters) const;

	// The geometry counts the UI shows, from the last build of the BLASes and TLASes (so they follow
	// SetMergeInstances). Unique counts are over the primitives some instance places.
	struct GeometryStats
	{
		uint64_t uniquePrimitives = 0;
		uint64_t uniqueTriangles = 0;
		uint64_t uniqueVertices = 0;
		uint64_t primitiveInstances = 0; // a primitive of a scene instance's mesh, the TLAS instances unmerged
		uint64_t instancedTriangles = 0; // the triangles of every primitive instance
		uint64_t tlasInstances = 0;
		uint64_t blases = 0; // without the micromap twins
		uint64_t micromapBlases = 0;
		uint64_t prototypes = 0; // merged instances' multi-geometry BLASes
		uint64_t mergedPrimitiveInstances = 0; // primitive instances inside a merged TLAS instance
	};

	const GeometryStats &GetGeometryStats() const
	{
		return geometryStats_;
	}

	uint32_t GetTextureCount() const
	{
		return static_cast<uint32_t>(textures_.size());
	}

	uint32_t GetOpacityMicromapCount() const
	{
		return opacityMicromapCount_;
	}

	uint32_t GetSceneCameraCount() const
	{
		return static_cast<uint32_t>(sceneCameras_.size());
	}

	// The file the scene was loaded from, for the UI.
	const std::string &GetName() const
	{
		return name_;
	}

	void SetName(std::string name)
	{
		name_ = std::move(name);
	}

private:
	// One TLAS instance: a primitive of a scene instance's mesh.
	struct TLASInstanceRef
	{
		uint32_t instance; // into instances
		uint32_t meshPrimitive; // primitive within the instance's mesh
		uint32_t primitive; // into primitiveDescs
	};

	// Every primitive of every scene instance, in scene instance order: the TLAS instances, in InstanceIndex()
	// order, without merged instances. The emitter tables keep this order (emitters_).
	std::vector<TLASInstanceRef> EnumerateTLASInstances() const;

	// The BLASes, TLASes and the tables that follow the TLAS instances, for mergeInstances_. The BLASes are built
	// and compacted on the compute queue; the TLAS builds are recorded on the current graphics command list.
	void BuildInstanceStructures();
	void DestroyInstanceStructures();

	struct SceneCamera
	{
		float worldMatrix[12]; // at rest, then as ApplyAnimationPose last posed it
		float yFov; // radians
		uint32_t animatedNode; // into animatedNodes_, or UINT32_MAX for a camera that does not move
	};
	static CameraPose PoseOf(const SceneCamera &camera);
	uint32_t FindAnimatedCamera() const; // into sceneCameras_, UINT32_MAX for none

	void UploadStreams(UploadBatchHandle batch);
	void BuildMaterials(const Scene::GLTFScene &scene, UploadBatchHandle &batch, std::vector<GpuMaterial> &materials);
	void UploadMaterials(const std::vector<GpuMaterial> &materials, UploadBatchHandle batch);
	// Fills the emissive lights and returns the emitter buffer's contents.
	std::vector<uint32_t> CollectEmitters(const Scene::GLTFScene &scene, const std::vector<GpuMaterial> &materials);
	// Moves the lights of the dynamic instances to the end of the light buffer,
	// keeping the order within each part (the rigid dynamic lights before the skinned ones), and renumbers them in
	// the emitter buffer's contents. Also numbers the emissive instances and resets their emission scales. Needs the
	// staged indices and the skinned vertices (PrepareAnimation).
	void PartitionDynamicLights(std::vector<uint32_t> &emitters);
	// The dynamic lights' centroid bounds from their instances' current transforms, conservatively: each
	// instance's object-space bounds moved as a box.
	void UpdateDynamicLightBounds();
	// The skinned lights' object-space corners and their instances' centroid bounds from skinnedVertices_.
	void UpdateSkinnedLights();
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
	// Fills the animation tables from the glTF's animations and skins, and the cameras and punctual lights, which may
	// move. Call after the primitive descs are made.
	void BuildAnimation(const Scene::GLTFScene &scene, const std::vector<uint32_t> &instanceOfNode);
	// From the animation tables, the dynamic instances and the skinned primitives' bind vertices. Call while the
	// vertices are staged, before PartitionDynamicLights.
	void PrepareAnimation();
	void DestroySkinUploads();
	void DestroyLights();
	// The light buffer and the dynamic lights' object-space tables. Only while the GPU is idle.
	void CreateLightBuffers(std::span<const GpuLight> punctualLights);
	void DestroyLightBuffers();

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
	std::vector<uint32_t> emitters_; // emitterBuffer_'s contents, its first entries per EnumerateTLASInstances
	ResourceHandle lightEmitterBuffer_ = InvalidResourceHandle;
	// With merged instances: the primitive descs then the prototypes' rows, and the emitter buffer with one
	// leading entry per TLAS instance in front of emitters_ (whose offsets it adds that many to).
	ResourceHandle tlasPrimitiveBuffer_ = InvalidResourceHandle;
	ResourceHandle tlasEmitterBuffer_ = InvalidResourceHandle;
	bool mergeInstances_ = true;
	uint32_t primitiveBufferIndex_ = InvalidResourceDescriptorIndex;
	uint32_t materialBufferIndex_ = InvalidResourceDescriptorIndex;

	// One set per frame in flight, so the CPU can write the current frame's set while the GPU may still
	// read the last one. GPU upload heap buffers, so the CPU writes them in place and the GPU reads them
	// from video memory.
	struct LightFrameBuffers
	{
		ResourceHandle transforms =
				InvalidResourceHandle; // DynamicLightTransform per dynamic instance, when they moved
		ResourceHandle punctualLights = InvalidResourceHandle; // GpuLight, written whole every frame
		ResourceHandle emissionScales = InvalidResourceHandle; // written whole every frame
		// The skinned lights' object-space triangles (dynamicObjectLights_ from skinnedLightStart_), with transforms
		ResourceHandle skinnedLights = InvalidResourceHandle;
		size_t skinnedLightCapacity = 0;
		size_t transformCapacity = 0;
		size_t punctualLightCapacity = 0;
		size_t emissionScaleCapacity = 0;
	};

	// Every light, in video memory: the static triangle lights, uploaded once per builder reset, then the dynamic
	// ones and the punctual lights, which TransformDynamicLights writes on the GPU. One buffer for all frames in flight,
	// as the GPU writes it in queue order. Only a UAV descriptor, which the tree build reads through; tracing binds
	// it as a root SRV.
	ResourceHandle lights_ = InvalidResourceHandle;
	ResourceHandle dynamicObjectLightBuffer_ = InvalidResourceHandle; // dynamicObjectLights_
	ResourceHandle dynamicLightInstanceBuffer_ = InvalidResourceHandle; // dynamicLightInstance_
	PipelineHandle dynamicLightsPipeline_ = InvalidPipelineHandle;
	// lightEmitters_ and a slot for every swarm light, made with the light buffer.
	ResourceHandle allLightEmitters_ = InvalidResourceHandle;

	// The light swarms (LightSwarmDesc): their lights from swarmFirstLight_ on in the light buffer, written on the GPU
	// whenever their constants change (their time moves, or a setting), and once after the light buffer is made.
	PipelineHandle swarmPipeline_ = InvalidPipelineHandle;
	LightSwarmDesc swarms_;
	uint32_t swarmLightCount_ = 0;
	uint32_t swarmFirstLight_ = 0;
	bool swarmLightsWritten_ = false;
	LightSwarmConstants swarmWrittenConstants_ = {}; // what the light buffer's swarm lights were written with
	// Where the swarms fly this frame (scene units), for the kernel and the light tree's bounds.
	float3 swarmRegionMin_ = {};
	float3 swarmRegionSize_ = {};
	float swarmRadius_ = 0.0f;
	LightTreeOptions lightTreeOptions_;
	// The emission scales the static BLASes were last refit with.
	std::vector<float> refitEmissionScales_;
	// The light buffer's ranges as the two-level hierarchy's BLASes: one per run of a triangle emitter's lights, one per
	// punctual light, one per swarm.
	std::vector<LightBlasDesc> MakeLightBlases() const;
	uint32_t lightsPose_ = UINT32_MAX; // dynamicLightPose_ of the dynamic lights in lights_, UINT32_MAX for none
	bool transformsWritten_ = false; // this frame's upload holds the dynamic instances' transforms

	// The light tree input: the emissive triangles (one-sided area lights, with the average of their
	// emissive texture as the flux scale), the static emitters' first, then from staticLightCount_ on
	// the dynamic ones' at rest (the GPU moves them, see lights_). The tree treats them all alike.
	std::vector<GpuLight> emissiveLights_;
	std::vector<float> emissiveFluxScales_;
	uint32_t staticLightCount_ = 0;
	// False after a scene load, until UpdateLights has reset the builder for the new lights.
	bool lightTreeBuilt_ = false;
	std::unique_ptr<LightTreeBuilder> lightTreeBuilder_;
	// Of the light centroids, for the Morton codes: the static ones' from the last reset, the dynamic ones'
	// from their last move.
	float3 staticLightBoundsMin_ = {};
	float3 staticLightBoundsMax_ = {};
	float3 dynamicLightBoundsMin_ = {};
	float3 dynamicLightBoundsMax_ = {};

	// Per emissive instance, its scene instance and SetEmissionScale's factor. The punctual lights have the slots
	// after the emissive instances', at 1.
	std::vector<uint32_t> emissiveInstances_;
	std::vector<float> emissionScales_;
	std::vector<uint32_t> lightEmitters_; // per light, its emission scale slot
	// Gives the punctual lights their emission scale slots after the emissive instances' ones. Call after
	// PartitionDynamicLights, before UploadEmitters.
	void AppendPunctualLightEmitters();
	// The punctual lights as GpuLights, where they are now.
	std::vector<GpuLight> MakePunctualLights() const;

	// The KHR_lights_punctual point and spot lights, kept in the scene cache. They follow the triangle lights in the
	// light buffer. Directional lights are left out for now.
	struct PunctualLight
	{
		float3 intensity; // candela, colour times intensity
		float range; // 0 = infinite
		float3 position; // world space, at rest, then as ApplyAnimationPose last posed it
		uint32_t animatedNode; // into animatedNodes_, or UINT32_MAX for a light that does not move
		float3 direction; // spot only: world space, the way the cone points (the node's -Z), posed like position
		uint32_t spot; // 1 for a spot light, 0 for a point light
		float innerConeAngle; // spot only, radians from the axis: full intensity inside
		float outerConeAngle; // spot only: no light outside
	};
	std::vector<PunctualLight> punctualLights_;
	float3 punctualLightBoundsMin_ = {};
	float3 punctualLightBoundsMax_ = {};

	// The animation system's tables (Animation::AnimationData), kept in the scene cache: the nodes it poses, the
	// channels that move them, and the skins with their joints and the influences on each skinned vertex.
	std::vector<Animation::Node> animatedNodes_;
	std::vector<Animation::Channel> animationChannels_;
	std::vector<float> animationTimes_;
	std::vector<float> animationValues_;
	std::vector<Animation::Skin> skins_;
	std::vector<Animation::Joint> skinJoints_;
	std::vector<Animation::SkinnedPrimitive> skinnedPrimitives_; // in instance order
	std::vector<Animation::VertexInfluence> skinInfluences_;
	// Per skinned vertex, from the staged vertices: its bind pose for the animation system, and its row as the vertex
	// buffer holds it, which ApplyAnimationPose moves to the pose and AddAccelerationStructurePasses uploads.
	std::vector<Animation::SkinVertex> skinBindVertices_;
	std::vector<GpuVertex> skinnedVertices_;
	std::vector<uint8_t> primitiveSkinned_; // per primitive desc
	uint32_t skinPoseGeneration_ = 0; // counts the poses ApplyAnimationPose has given the skinned vertices
	uint32_t skinUploadGeneration_ = 0; // the pose the vertex buffer and the skinned BLASes hold
	std::vector<ResourceHandle> skinUploads_; // per frame in flight, skinnedVertices_ in an upload heap
	ResourceHandle skinnedPrimitiveBases_ = InvalidResourceHandle; // GetSkinnedPrimitiveBaseBuffer's
	std::vector<ResourceHandle> previousSkinnedPositions_; // per frame in flight, GetPreviousSkinnedPositionBuffer's
	// Per skinned vertex, its object-space position as the last frame traced it; valid once a frame has.
	std::vector<float4> skinnedLastFramePositions_;
	bool skinnedLastFrameValid_ = false;
	std::vector<AccelerationStructureHandle> skinnedBlases_; // blases_ of the skinned primitives, never compacted
	std::vector<SceneCamera> sceneCameras_; // every perspective camera, in node order

	// The scene instances that move, and the lights of their triangles.
	std::vector<uint32_t> dynamicInstances_; // into instances
	std::vector<uint32_t> dynamicOfInstance_; // per scene instance, into dynamicInstances_ or UINT32_MAX
	std::vector<uint8_t> dynamicRestMirrored_; // the rest transform mirrors, so CollectEmitters swapped corners
	// Per dynamic instance, its world transform when UpdateAccelerationStructures last ran.
	std::vector<std::array<float, 12>> dynamicLastFrameTransforms_;
	std::vector<GpuLight> dynamicObjectLights_; // per dynamic light, its triangle in object space
	std::vector<uint32_t> dynamicLightInstance_; // per dynamic light, into dynamicInstances_
	// The dynamic lights from skinnedLightStart_ on are triangles of skinned meshes. The anchor transform alone does
	// not place them, so ApplyAnimationPose rewrites their object-space corners from the skinned vertices.
	uint32_t skinnedLightStart_ = 0;
	std::vector<uint32_t> skinnedLightCorners_; // 3 per skinned light, into skinnedVertices_, in the light's order
	std::vector<uint32_t> skinnedLightInstances_; // the dynamic instances with skinned lights
	// Per dynamic instance, the bounds of its lights' object-space centroids, empty (min above max) without lights.
	std::vector<std::array<float3, 2>> dynamicObjectLightBounds_;
	bool dynamicLightsMoved_ = false;
	uint32_t dynamicLightPose_ = 0; // counts the moves of the dynamic lights
	uint32_t dynamicPoseGeneration_ = 0; // counts the poses ApplyAnimationPose has moved the dynamic instances to

	std::vector<LightFrameBuffers> lightFrames_;
	uint32_t lightFrame_ = 0;
	uint32_t lightCount_ = 0;

	std::vector<ResourceHandle> textures_;
	// Per GpuMaterial row; decides the geometry flags of each BLAS that uses it.
	std::vector<MaterialGeometryFlags> materialGeometry_;
	// Per GpuMaterial row, the roughness factor of its smoothest lobe that can make a caustic (CollectPhotonCasters),
	// FLT_MAX for none.
	std::vector<float> materialCasterRoughness_;
	// The same for its mirror lobes alone, metal and the coat.
	std::vector<float> materialCasterMirrorRoughness_;
	// Per primitiveDescs row, its object-space bounds (min, max), kept from the vertices before ReleaseStaging.
	std::vector<std::pair<float3, float3>> primitiveBounds_;
	uint64_t pendingUploadBytes_ = 0;
	SceneCacheWriter *cacheWriter_ = nullptr; // during a BuildGPUScene that writes a cache
	double mipBuildMs_ = 0.0; // CPU time spent filtering texture mips during the last BuildGPUScene
	std::unique_ptr<ITextureCompressor> textureCompressor_; // during a BuildGPUScene
	double compressMs_ = 0.0; // time spent compressing textures to BC7 during the last BuildGPUScene
	uint32_t compressedTextures_ = 0;
	uint32_t uncompressedTextures_ = 0; // left as RGBA8 by CompressBC7 (not whole 4x4 blocks, or it failed)

	// One per primitive desc, invalid for skipped primitives and, with merged instances, those only in prototypes.
	std::vector<AccelerationStructureHandle> blases_;
	// With merged instances, one per prototype (a set of primitives placed together), and its micromap twin
	// (invalid without micromaps).
	std::vector<AccelerationStructureHandle> prototypeBlases_;
	std::vector<AccelerationStructureHandle> prototypeOmmBlases_;
	// The TLAS instances as built, in InstanceIndex() order, for UpdateAccelerationStructures: a merged one by
	// the ref of its first part (merged instances are static).
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
		uint32_t skinGeneration = 0; // the skinned BLASes' pose it was built over, as their bounds are in it
	};
	std::vector<FrameTLASes> frameTlases_; // empty without instanced geometry
	bool hasBlendedInstances_ = false; // some TLAS instance is INSTANCE_MASK_BLENDED
	uint32_t tlasFrame_ = 0; // the set the current frame traces
	uint32_t tlasFramesUpdated_ = 0; // frames UpdateAccelerationStructures has picked a set for, up to 2

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

	GeometryStats geometryStats_;
	uint32_t opacityMicromapCount_ = 0; // ommMicromaps_.size() before BuildAccelerationStructures released them
	std::string name_;
};
