#include "RenderGPUScene.h"

#include "GLTFScene.h"
#include "LightTreeBuilder.h"
#include "PathTracerInterop.h"

#include "fmtlog.h"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cstring>


namespace
{
	// Staging memory for one upload batch is held until EndUpload, so flush every so often to keep
	// large texture sets from needing all of their staging buffers alive at once.
	constexpr uint64_t kMaxPendingUploadBytes = 256ull * 1024 * 1024;

	constexpr uint32_t kPositionStride = 3 * sizeof(float);

	bool IsUsablePrimitive(const Scene::Primitive &primitive)
	{
		return primitive.IsValid() && primitive.HasIndices() && primitive.GetIndexCount() >= 3;
	}

	// Transforms a point by the scene's row-major 3x4 layout (3x3 for row vectors, translation in
	// the 4th column of each row; see Scene::ComposeTRS).
	float3 TransformPoint(const float *m, float x, float y, float z)
	{
		return { x * m[0] + y * m[4] + z * m[8] + m[3],
				 x * m[1] + y * m[5] + z * m[9] + m[7],
				 x * m[2] + y * m[6] + z * m[10] + m[11] };
	}

	template<typename T>
	void WriteLightBuffer(RenderContext *renderContext,
						  ResourceHandle &buffer,
						  size_t &capacity,
						  const std::vector<T> &elements,
						  const char *debugName)
	{
		if (elements.size() > capacity)
		{
			if (buffer != InvalidResourceHandle)
			{
				renderContext->DestroyResource(buffer);
			}
			capacity = std::bit_ceil(elements.size());

			BufferDesc desc = {};
			desc.sizeInBytes = capacity * sizeof(T);
			desc.structureByteStride = sizeof(T);
			desc.debugName = debugName;
			buffer = renderContext->CreateUploadBuffer(desc, ResourceViewType::ShaderResource);
		}
		renderContext->UpdateUploadBuffer(buffer, elements.data(), 0, elements.size() * sizeof(T));
	}
} // namespace

RenderGPUScene::RenderGPUScene(RenderContext *renderContext) : renderContext_(renderContext)
{
	std::fill(std::begin(streamBuffers_), std::end(streamBuffers_), InvalidResourceHandle);
	std::fill(std::begin(streamBufferIndices), std::end(streamBufferIndices), InvalidResourceDescriptorIndex);
}

RenderGPUScene::~RenderGPUScene()
{
	if (tlas_ != InvalidAccelerationStructureHandle)
	{
		renderContext_->DestroyAccelerationStructure(tlas_);
	}
	for (AccelerationStructureHandle blas: blases_)
	{
		if (blas != InvalidAccelerationStructureHandle)
		{
			renderContext_->DestroyAccelerationStructure(blas);
		}
	}
	for (ResourceHandle texture: textures_)
	{
		renderContext_->DestroyResource(texture);
	}
	for (ResourceHandle buffer: streamBuffers_)
	{
		if (buffer != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(buffer);
		}
	}
	for (ResourceHandle buffer: { primitiveBuffer_, materialBuffer_, instanceBuffer_ })
	{
		if (buffer != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(buffer);
		}
	}
	DestroyLights();
}

void RenderGPUScene::DestroyLights()
{
	for (LightFrameBuffers &frame: lightFrames_)
	{
		for (ResourceHandle buffer: { frame.lights, frame.nodes })
		{
			if (buffer != InvalidResourceHandle)
			{
				renderContext_->DestroyResource(buffer);
			}
		}
	}
	lightFrames_.clear();
	lightCount_ = 0;
}

void RenderGPUScene::UpdateLights(std::span<const GpuLight> lights)
{
	const auto start = std::chrono::steady_clock::now();
	LightTree tree = BuildLightTree(lights);
	lightTreeBuildMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();

	const uint32_t lightCount = static_cast<uint32_t>(tree.lights.size());
	if (lightCount != lightCount_ && lightCount < lights.size())
	{
		logw("GPUScene: light tree skipped {} light(s) that are not point/area lights or emit nothing",
			 lights.size() - lightCount);
	}
	lightCount_ = lightCount;

	if (lightCount_ == 0)
	{
		// One unused element each, so the shader's bindings stay valid.
		tree.lights.push_back(GpuLight{});
		tree.nodes.push_back(GpuLightTreeNode{});
	}

	// BeginFrame has waited for the frame that last read this slot, so it can be overwritten.
	lightFrames_.resize(renderContext_->GetFramesInFlight());
	lightFrame_ = renderContext_->GetCurrentFrameIndex();
	LightFrameBuffers &frame = lightFrames_[lightFrame_];
	WriteLightBuffer(renderContext_, frame.lights, frame.lightCapacity, tree.lights, "Scene_Lights");
	WriteLightBuffer(renderContext_, frame.nodes, frame.nodeCapacity, tree.nodes, "Scene_LightTreeNodes");
}

void RenderGPUScene::BuildGPUScene(const Scene::GLTFScene &scene)
{
	meshDescs.clear();
	primitiveDescs.clear();
	positions_.clear();
	texCoords_.clear();
	normals_.clear();
	tangents_.clear();
	indices_.clear();
	meshletData_.clear();
	meshletVertices_.clear();
	meshletTriangles_.clear();

	// Materials are appended with one extra default entry at the end for primitives that have none.
	const uint32_t defaultMaterialIndex = static_cast<uint32_t>(scene.GetMaterialCount());

	size_t skippedPrimitives = 0;
	for (size_t m = 0; m < scene.GetMeshCount(); ++m)
	{
		const Scene::Mesh &mesh = scene.GetMesh(m);

		MeshGPUDesc meshDesc = {};
		meshDesc.primitiveOffset = static_cast<uint32_t>(primitiveDescs.size());
		meshDesc.primitiveCount = static_cast<uint32_t>(mesh.GetPrimitiveCount());
		meshDescs.push_back(meshDesc);

		for (const Scene::Primitive &primitive: mesh.GetPrimitives())
		{
			PrimitiveGPUDesc desc = {};
			const int32_t materialIndex = primitive.GetMaterialIndex();
			desc.materialIndex = (materialIndex >= 0 && static_cast<size_t>(materialIndex) < scene.GetMaterialCount())
										 ? static_cast<uint32_t>(materialIndex)
										 : defaultMaterialIndex;

			// Keep a row for every primitive so mesh/primitive indexing stays intact; unusable
			// ones simply have no geometry and get no BLAS.
			if (!IsUsablePrimitive(primitive))
			{
				++skippedPrimitives;
				primitiveDescs.push_back(desc);
				continue;
			}

			desc.vertexOffset = static_cast<uint32_t>(positions_.size() / 3);
			desc.vertexCount = static_cast<uint32_t>(primitive.GetVertexCount());
			desc.indexOffset = static_cast<uint32_t>(indices_.size());
			desc.indexCount = static_cast<uint32_t>(primitive.GetIndexCount());

			positions_.insert(positions_.end(), primitive.GetPositions().begin(), primitive.GetPositions().end());
			texCoords_.insert(texCoords_.end(), primitive.GetTexCoord0().begin(), primitive.GetTexCoord0().end());
			normals_.insert(normals_.end(), primitive.GetNormals().begin(), primitive.GetNormals().end());
			tangents_.insert(tangents_.end(), primitive.GetTangents().begin(), primitive.GetTangents().end());
			// Indices stay local to the primitive; vertexOffset rebases them.
			indices_.insert(indices_.end(), primitive.GetIndices().begin(), primitive.GetIndices().end());

			static_assert(sizeof(Scene::Meshlet) % sizeof(uint32_t) == 0);
			const auto &meshlets = primitive.GetMeshlets();
			desc.meshletOffset =
					static_cast<uint32_t>(meshletData_.size() / (sizeof(Scene::Meshlet) / sizeof(uint32_t)));
			desc.meshletCount = static_cast<uint32_t>(meshlets.size());
			desc.meshletVerticesOffset = static_cast<uint32_t>(meshletVertices_.size());
			desc.meshletTrianglesOffset = static_cast<uint32_t>(meshletTriangles_.size());
			const auto *meshletWords = reinterpret_cast<const uint32_t *>(meshlets.data());
			meshletData_.insert(meshletData_.end(),
								meshletWords,
								meshletWords + meshlets.size() * (sizeof(Scene::Meshlet) / sizeof(uint32_t)));
			meshletVertices_.insert(meshletVertices_.end(),
									primitive.GetMeshletVertices().begin(),
									primitive.GetMeshletVertices().end());
			meshletTriangles_.insert(meshletTriangles_.end(),
									 primitive.GetMeshletTriangles().begin(),
									 primitive.GetMeshletTriangles().end());

			primitiveDescs.push_back(desc);
		}
	}

	if (skippedPrimitives > 0)
	{
		logw("GPUScene: skipped {} primitive(s) missing indices or a required attribute", skippedPrimitives);
	}

	FillInstances(scene);
	ComputeBounds(scene);

	UploadBatchHandle batch = renderContext_->BeginUpload();
	pendingUploadBytes_ = 0;

	UploadStreams(batch);
	UploadMaterials(scene, batch);

	renderContext_->EndUpload(batch);

	logi("GPUScene: {} mesh(es), {} primitive(s), {} instance(s), {} vertices, {} triangles, {} texture(s)",
		 meshDescs.size(),
		 primitiveDescs.size(),
		 instances.size(),
		 positions_.size() / 3,
		 indices_.size() / 3,
		 textures_.size());

	// The GPU has its own copies now.
	for (std::vector<float> *stream: { &positions_, &texCoords_, &normals_, &tangents_ })
	{
		std::vector<float>().swap(*stream);
	}
	for (std::vector<uint32_t> *stream: { &indices_, &meshletData_, &meshletVertices_, &meshletTriangles_ })
	{
		std::vector<uint32_t>().swap(*stream);
	}
}

void RenderGPUScene::FillInstances(const Scene::GLTFScene &scene)
{
	instances.clear();
	for (size_t i = 0; i < scene.GetInstanceCount(); ++i)
	{
		const Scene::Instance &node = scene.GetInstance(i);
		const int32_t meshIndex = node.GetMeshIndex();
		if (meshIndex < 0 || static_cast<size_t>(meshIndex) >= meshDescs.size())
		{
			continue; // transform-only node (camera, light, empty parent...)
		}

		InstanceGPU instance = {};
		std::memcpy(instance.worldMatrix, node.GetWorldMatrix(), sizeof(instance.worldMatrix));
		instance.meshIndex = static_cast<uint32_t>(meshIndex);
		instances.push_back(instance);
	}
}

void RenderGPUScene::ComputeBounds(const Scene::GLTFScene &scene)
{
	float3 lo = { FLT_MAX, FLT_MAX, FLT_MAX };
	float3 hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

	for (const InstanceGPU &instance: instances)
	{
		const Scene::Mesh &mesh = scene.GetMesh(instance.meshIndex);
		for (const Scene::Primitive &primitive: mesh.GetPrimitives())
		{
			if (!IsUsablePrimitive(primitive))
			{
				continue;
			}
			const std::vector<float> &p = primitive.GetPositions();
			for (size_t v = 0; v + 2 < p.size(); v += 3)
			{
				const float3 w = TransformPoint(instance.worldMatrix, p[v], p[v + 1], p[v + 2]);
				lo = { std::min(lo.x, w.x), std::min(lo.y, w.y), std::min(lo.z, w.z) };
				hi = { std::max(hi.x, w.x), std::max(hi.y, w.y), std::max(hi.z, w.z) };
			}
		}
	}

	if (lo.x > hi.x)
	{
		lo = hi = { 0.0f, 0.0f, 0.0f };
	}
	boundsMin_ = lo;
	boundsMax_ = hi;
}

ResourceHandle RenderGPUScene::UploadStream(
		const void *data, uint64_t sizeInBytes, uint32_t stride, const char *debugName, UploadBatchHandle batch)
{
	BufferDesc desc = {};
	desc.sizeInBytes = sizeInBytes;
	desc.structureByteStride = stride;
	desc.debugName = debugName;
	pendingUploadBytes_ += sizeInBytes;
	return renderContext_->CreateBuffer(desc, ResourceViewType::ShaderResource, data, batch);
}

void RenderGPUScene::UploadStreams(UploadBatchHandle batch)
{
	struct StreamSource
	{
		GPUStream stream;
		const void *data;
		uint64_t sizeInBytes;
		uint32_t stride;
		const char *name;
	};

	const StreamSource sources[] = {
		{ GPUStream::Position,
		  positions_.data(),
		  positions_.size() * sizeof(float),
		  kPositionStride,
		  "Scene_Positions" },
		{ GPUStream::TexCoord0,
		  texCoords_.data(),
		  texCoords_.size() * sizeof(float),
		  2 * sizeof(float),
		  "Scene_TexCoord0" },
		{ GPUStream::Normal, normals_.data(), normals_.size() * sizeof(float), 3 * sizeof(float), "Scene_Normals" },
		{ GPUStream::Tangent, tangents_.data(), tangents_.size() * sizeof(float), 4 * sizeof(float), "Scene_Tangents" },
		{ GPUStream::Index, indices_.data(), indices_.size() * sizeof(uint32_t), sizeof(uint32_t), "Scene_Indices" },
		{ GPUStream::Meshlet,
		  meshletData_.data(),
		  meshletData_.size() * sizeof(uint32_t),
		  sizeof(Scene::Meshlet),
		  "Scene_Meshlets" },
		{ GPUStream::MeshletVertices,
		  meshletVertices_.data(),
		  meshletVertices_.size() * sizeof(uint32_t),
		  sizeof(uint32_t),
		  "Scene_MeshletVertices" },
		{ GPUStream::MeshletTriangles,
		  meshletTriangles_.data(),
		  meshletTriangles_.size() * sizeof(uint32_t),
		  sizeof(uint32_t),
		  "Scene_MeshletTriangles" },
	};

	for (const StreamSource &source: sources)
	{
		if (source.sizeInBytes == 0)
		{
			continue; // stays InvalidResourceDescriptorIndex
		}
		const ResourceHandle buffer = UploadStream(source.data, source.sizeInBytes, source.stride, source.name, batch);
		streamBuffers_[(uint32_t) source.stream] = buffer;
		streamBufferIndices[(uint32_t) source.stream] = renderContext_->GetResourceViewInfo(buffer).descriptorIndex;
	}

	if (!primitiveDescs.empty())
	{
		primitiveBuffer_ = UploadStream(primitiveDescs.data(),
										primitiveDescs.size() * sizeof(PrimitiveGPUDesc),
										sizeof(PrimitiveGPUDesc),
										"Scene_PrimitiveDescs",
										batch);
		primitiveBufferIndex_ = renderContext_->GetResourceViewInfo(primitiveBuffer_).descriptorIndex;
	}

	if (!instances.empty())
	{
		// Upload heap so UpdateInstances can rewrite transforms in place.
		BufferDesc desc = {};
		desc.sizeInBytes = instances.size() * sizeof(InstanceGPU);
		desc.structureByteStride = sizeof(InstanceGPU);
		desc.debugName = "Scene_Instances";
		instanceBuffer_ = renderContext_->CreateUploadBuffer(desc, ResourceViewType::ShaderResource, instances.data());
	}
}

uint32_t RenderGPUScene::UploadTexture(const Scene::Texture &texture,
									   Format format,
									   const char *debugName,
									   UploadBatchHandle &batch)
{
	if (!texture.IsValid())
	{
		return INVALIDDESCRIPTORHANDLE;
	}

	const uint32_t rowPitch = static_cast<uint32_t>(texture.GetWidth()) * 4;
	const uint64_t sizeInBytes = static_cast<uint64_t>(rowPitch) * texture.GetHeight();
	if (pendingUploadBytes_ > 0 && pendingUploadBytes_ + sizeInBytes > kMaxPendingUploadBytes)
	{
		renderContext_->EndUpload(batch);
		batch = renderContext_->BeginUpload();
		pendingUploadBytes_ = 0;
	}

	TextureDesc desc = {};
	desc.width = static_cast<uint>(texture.GetWidth());
	desc.height = static_cast<uint>(texture.GetHeight());
	desc.format = format;
	desc.debugName = debugName;

	TextureMipData mip = {};
	mip.pixels = texture.GetPixels();
	mip.rowPitch = rowPitch;
	mip.rowCount = desc.height;

	const ResourceHandle gpuTexture = renderContext_->CreateTexture(desc,
																	ResourceViewType::ShaderResource,
																	std::span<const TextureMipData>(&mip, 1),
																	batch);
	pendingUploadBytes_ += sizeInBytes;
	textures_.push_back(gpuTexture);
	return renderContext_->GetResourceViewInfo(gpuTexture).descriptorIndex;
}

void RenderGPUScene::UploadMaterials(const Scene::GLTFScene &scene, UploadBatchHandle &batch)
{
	std::vector<GpuMaterial> materials;
	materials.reserve(scene.GetMaterialCount() + 1);
	materialIsOpaque_.clear();
	materialIsOpaque_.reserve(scene.GetMaterialCount() + 1);

	for (size_t i = 0; i < scene.GetMaterialCount(); ++i)
	{
		const Scene::Material &material = scene.GetMaterial(i);
		const float *baseColour = material.GetBaseColour();

		GpuMaterial gpu = {};
		gpu.baseColour = float4(baseColour[0], baseColour[1], baseColour[2], baseColour[3]);
		gpu.roughness = material.GetRoughness();
		gpu.metallic = material.GetMetal();
		gpu.normalScale = material.GetNormalScale();
		gpu.alphaCutoff = material.GetAlphaCutoff();
		gpu.alphaMode = static_cast<uint32_t>(material.GetAlphaMode());
		gpu.transmission = material.GetTransmission();
		gpu.albedoTexture =
				UploadTexture(material.GetAlbedoTexture(), Format::R8G8B8A8_UNORM_SRGB, "Material_Albedo", batch);
		gpu.roughMetalTexture =
				UploadTexture(material.GetRoughMetalTexture(), Format::R8G8B8A8_UNORM, "Material_RoughMetal", batch);
		gpu.normalTexture =
				UploadTexture(material.GetNormalTexture(), Format::R8G8B8A8_UNORM, "Material_Normal", batch);
		materials.push_back(gpu);
		materialIsOpaque_.push_back(material.GetAlphaMode() == Scene::AlphaMode::Opaque);
	}

	// Default material (glTF spec defaults would be metallic 1, which reads badly on untextured
	// geometry, so use a plain grey dielectric instead).
	GpuMaterial fallback = {};
	fallback.baseColour = float4(0.8f, 0.8f, 0.8f, 1.0f);
	fallback.roughness = 0.5f;
	fallback.metallic = 0.0f;
	fallback.normalScale = 1.0f;
	fallback.albedoTexture = INVALIDDESCRIPTORHANDLE;
	fallback.roughMetalTexture = INVALIDDESCRIPTORHANDLE;
	fallback.normalTexture = INVALIDDESCRIPTORHANDLE;
	fallback.alphaCutoff = 0.5f;
	fallback.alphaMode = ALPHA_MODE_OPAQUE;
	materials.push_back(fallback);
	materialIsOpaque_.push_back(true);

	materialBuffer_ = UploadStream(materials.data(),
								   materials.size() * sizeof(GpuMaterial),
								   sizeof(GpuMaterial),
								   "Scene_Materials",
								   batch);
	materialBufferIndex_ = renderContext_->GetResourceViewInfo(materialBuffer_).descriptorIndex;
}

void RenderGPUScene::UpdateInstances(const Scene::GLTFScene &scene)
{
	const size_t previousCount = instances.size();
	FillInstances(scene);
	if (instanceBuffer_ != InvalidResourceHandle && instances.size() == previousCount)
	{
		renderContext_->UpdateUploadBuffer(instanceBuffer_,
										   instances.data(),
										   0,
										   instances.size() * sizeof(InstanceGPU));
	}
}

void RenderGPUScene::BuildAccelerationStructures()
{
	// Textures come off the copy queue in COMMON; move them to SHADER_RESOURCE for sampling.
	for (ResourceHandle texture: textures_)
	{
		renderContext_->TransitionResource(texture,
										   ResourceLayout::ShaderResource,
										   ResourceSync::AllShading,
										   ResourceAccess::ShaderResource);
	}

	const ResourceHandle positionBuffer = streamBuffers_[(uint32_t) GPUStream::Position];
	const ResourceHandle indexBuffer = streamBuffers_[(uint32_t) GPUStream::Index];

	blases_.assign(primitiveDescs.size(), InvalidAccelerationStructureHandle);
	for (size_t i = 0; i < primitiveDescs.size(); ++i)
	{
		const PrimitiveGPUDesc &desc = primitiveDescs[i];
		if (desc.indexCount == 0)
		{
			continue;
		}

		BLASGeometry geometry = {};
		geometry.vertexBuffer = positionBuffer;
		geometry.vertexBufferOffset = static_cast<uint64_t>(desc.vertexOffset) * kPositionStride;
		geometry.vertexCount = desc.vertexCount;
		geometry.vertexStride = kPositionStride;
		geometry.indexBuffer = indexBuffer;
		geometry.indexBufferOffset = static_cast<uint64_t>(desc.indexOffset) * sizeof(uint32_t);
		geometry.indexCount = desc.indexCount;
		// Only masked and blended materials need the shadow any-hit shader; opaque geometry skips it.
		geometry.isOpaque = materialIsOpaque_[desc.materialIndex];

		blases_[i] = renderContext_->CreateBLAS(geometry);
		renderContext_->BuildAccelerationStructure(blases_[i]);
	}

	renderContext_->UAVBarrierForAccelerationStructures();

	// One TLAS instance per (node, primitive). InstanceID is the primitive's row in the
	// PrimitiveGPUDesc table, which is all the hit shader needs to find its data.
	std::vector<TLASInstance> tlasInstances;
	for (const InstanceGPU &instance: instances)
	{
		// The scene stores row-vector matrices; DXR wants the 3x4 for column vectors.
		float transform[12];
		Scene::RowMajorToBLASTransform(instance.worldMatrix, transform);

		const MeshGPUDesc &mesh = meshDescs[instance.meshIndex];
		for (uint32_t p = 0; p < mesh.primitiveCount; ++p)
		{
			const uint32_t primitiveIndex = mesh.primitiveOffset + p;
			if (blases_[primitiveIndex] == InvalidAccelerationStructureHandle)
			{
				continue;
			}

			TLASInstance tlasInstance = {};
			tlasInstance.blas = blases_[primitiveIndex];
			std::memcpy(tlasInstance.transform, transform, sizeof(tlasInstance.transform));
			tlasInstance.instanceID = primitiveIndex;
			tlasInstance.instanceMask = 0xFF;
			tlasInstance.hitGroupIndex = 0;
			tlasInstances.push_back(tlasInstance);
		}
	}

	if (tlasInstances.empty())
	{
		logw("GPUScene: no instanced geometry, skipping TLAS build");
		return;
	}

	tlas_ = renderContext_->CreateTLAS(tlasInstances);
	renderContext_->BuildAccelerationStructure(tlas_);
	renderContext_->UAVBarrierForAccelerationStructures();
	tlasSRV_ = renderContext_->GetAccelerationStructureSRV(tlas_);

	logi("GPUScene: built {} BLAS(es) and a TLAS with {} instance(s)",
		 std::count_if(blases_.begin(),
					   blases_.end(),
					   [](AccelerationStructureHandle h) { return h != InvalidAccelerationStructureHandle; }),
		 tlasInstances.size());
}
