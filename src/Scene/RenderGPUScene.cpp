#include "RenderGPUScene.h"

#include "GLTFScene.h"
#include "LightTreeBuilder.h"
#include "PathTracerInterop.h"
#include "SceneCache.h"

#include "fmtlog.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <execution>
#include <future>
#include <numeric>
#include <stdexcept>
#include <unordered_map>


namespace
{
	// Staging memory for one upload batch is held until EndUpload, so flush every so often to keep
	// large texture sets from needing all of their staging buffers alive at once.
	constexpr uint64_t kMaxPendingUploadBytes = 256ull * 1024 * 1024;

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

	float Determinant3x3(const float *m)
	{
		return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
			   m[2] * (m[4] * m[9] - m[5] * m[8]);
	}

	float Luminance(const float3 &c)
	{
		return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
	}

	float SrgbToLinear(uint8_t value)
	{
		static const std::array<float, 256> table = []
		{
			std::array<float, 256> t = {};
			for (int i = 0; i < 256; ++i)
			{
				const float c = static_cast<float>(i) / 255.0f;
				t[i] = c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
			}
			return t;
		}();
		return table[value];
	}

	// Linear values quantised to 16 bits, fine enough to tell the darkest sRGB codes apart.
	uint8_t LinearToSrgb(float value)
	{
		static const std::vector<uint8_t> table = []
		{
			std::vector<uint8_t> t(65536);
			for (size_t i = 0; i < t.size(); ++i)
			{
				const float c = static_cast<float>(i) / 65535.0f;
				const float s = c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
				t[i] = static_cast<uint8_t>(s * 255.0f + 0.5f);
			}
			return t;
		}();
		return table[static_cast<uint32_t>(std::clamp(value, 0.0f, 1.0f) * 65535.0f + 0.5f)];
	}

	// One RGBA8 mip from the level above with a 2x2 box filter, the colour of sRGB textures averaged
	// in linear space. An odd size drops its last row or column, as D3D rounds mip sizes down.
	void DownsampleRgba8(const uint8_t *src,
						 uint32_t srcWidth,
						 uint32_t srcHeight,
						 uint8_t *dst,
						 uint32_t width,
						 uint32_t height,
						 bool srgb)
	{
		std::vector<uint32_t> rows(height);
		std::iota(rows.begin(), rows.end(), 0u);
		std::for_each(std::execution::par,
					  rows.begin(),
					  rows.end(),
					  [&](uint32_t y)
					  {
						  const size_t srcPitch = static_cast<size_t>(srcWidth) * 4;
						  const uint8_t *row0 = src + std::min(2 * y, srcHeight - 1) * srcPitch;
						  const uint8_t *row1 = src + std::min(2 * y + 1, srcHeight - 1) * srcPitch;
						  uint8_t *out = dst + static_cast<size_t>(y) * width * 4;
						  for (uint32_t x = 0; x < width; ++x, out += 4)
						  {
							  const uint32_t x0 = std::min(2 * x, srcWidth - 1) * 4;
							  const uint32_t x1 = std::min(2 * x + 1, srcWidth - 1) * 4;
							  const uint8_t *t[4] = { row0 + x0, row0 + x1, row1 + x0, row1 + x1 };
							  for (int c = 0; c < 4; ++c)
							  {
								  if (srgb && c < 3)
								  {
									  const float sum = SrgbToLinear(t[0][c]) + SrgbToLinear(t[1][c]) +
														SrgbToLinear(t[2][c]) + SrgbToLinear(t[3][c]);
									  out[c] = LinearToSrgb(0.25f * sum);
								  }
								  else
								  {
									  out[c] = static_cast<uint8_t>((t[0][c] + t[1][c] + t[2][c] + t[3][c] + 2) / 4);
								  }
							  }
						  }
					  });
	}

	// The levels below the texture's own pixels, down to 1x1, each filtered from the one above.
	std::vector<std::vector<uint8_t>> BuildMipChain(const Scene::Texture &texture, bool srgb)
	{
		uint32_t width = static_cast<uint32_t>(texture.GetWidth());
		uint32_t height = static_cast<uint32_t>(texture.GetHeight());
		std::vector<std::vector<uint8_t>> levels;
		// Reserved up front so the source pointer into the previous level stays valid.
		levels.reserve(std::bit_width(std::max(width, height)) - 1);
		const uint8_t *src = texture.GetPixels();
		while (width > 1 || height > 1)
		{
			const uint32_t mipWidth = std::max(width / 2, 1u);
			const uint32_t mipHeight = std::max(height / 2, 1u);
			std::vector<uint8_t> &level = levels.emplace_back(static_cast<size_t>(mipWidth) * mipHeight * 4);
			DownsampleRgba8(src, width, height, level.data(), mipWidth, mipHeight, srgb);
			src = level.data();
			width = mipWidth;
			height = mipHeight;
		}
		return levels;
	}

	// Nearest texel of an sRGB texture with wrapping, decoded to linear RGB.
	float3 FetchSrgbTexel(const Scene::Texture &texture, float u, float v)
	{
		const int32_t w = texture.GetWidth();
		const int32_t h = texture.GetHeight();
		const int32_t x = std::min(static_cast<int32_t>((u - floorf(u)) * static_cast<float>(w)), w - 1);
		const int32_t y = std::min(static_cast<int32_t>((v - floorf(v)) * static_cast<float>(h)), h - 1);
		const uint8_t *texel = texture.GetPixels() + (static_cast<size_t>(y) * w + x) * 4;
		return { SrgbToLinear(texel[0]), SrgbToLinear(texel[1]), SrgbToLinear(texel[2]) };
	}

	// Average of an sRGB texture over a triangle in UV space, from the centroids of its 16 sub-triangles
	// after splitting each edge in four. Only sets the light tree importance, so it need not be exact.
	float3 AverageSrgbTexture(const Scene::Texture &texture, const float2 (&uv)[3])
	{
		constexpr int kSplits = 4;
		const float2 e1 = uv[1] - uv[0];
		const float2 e2 = uv[2] - uv[0];
		float3 sum = { 0.0f, 0.0f, 0.0f };
		auto add = [&](float a, float b)
		{
			a /= kSplits;
			b /= kSplits;
			sum = sum + FetchSrgbTexel(texture, uv[0].x + e1.x * a + e2.x * b, uv[0].y + e1.y * a + e2.y * b);
		};
		for (int i = 0; i < kSplits; ++i)
		{
			for (int j = 0; i + j < kSplits; ++j)
			{
				add(i + 1.0f / 3.0f, j + 1.0f / 3.0f);
				if (i + j + 1 < kSplits)
				{
					add(i + 2.0f / 3.0f, j + 2.0f / 3.0f);
				}
			}
		}
		return sum * (1.0f / (kSplits * kSplits));
	}

	template<typename Remap>
	void RemapTextures(std::vector<GpuMaterial> &materials, std::vector<GpuLight> &lights, const Remap &remap)
	{
		auto apply = [&](uint32_t &texture)
		{
			if (texture != INVALIDDESCRIPTORHANDLE)
			{
				texture = remap(texture);
			}
		};
		for (GpuMaterial &material: materials)
		{
			for (uint32_t *texture: { &material.albedoTexture,
									  &material.roughMetalTexture,
									  &material.normalTexture,
									  &material.transmissionTexture,
									  &material.emissiveTexture })
			{
				apply(*texture);
			}
		}
		// MakeAreaLight keeps the texture in radiance.w, UINT_MAX when untextured.
		for (GpuLight &light: lights)
		{
			uint32_t texture = std::bit_cast<uint32_t>(light.radiance.w);
			apply(texture);
			light.radiance.w = std::bit_cast<float>(texture);
		}
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
	for (ResourceHandle buffer: { vertexBuffer_, primitiveBuffer_, materialBuffer_, instanceBuffer_, emitterBuffer_ })
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
		for (ResourceHandle buffer: { frame.lights, frame.nodes, frame.leaves, frame.parents, frame.leafLights })
		{
			if (buffer != InvalidResourceHandle)
			{
				renderContext_->DestroyResource(buffer);
			}
		}
	}
	lightFrames_.clear();
	lightCount_ = 0;
	lightTreeBuilt_ = false;
}

void RenderGPUScene::UpdateLights(const LightTreeBuildSettings &settings)
{
	// The emissive triangles are static, so the tree only changes with its settings.
	if (lightTreeBuilt_ && settings == lightTreeSettings_)
	{
		return;
	}
	lightTreeBuilt_ = true;
	lightTreeSettings_ = settings;

	const auto start = std::chrono::steady_clock::now();
	LightTree tree = BuildLightTree(emissiveLights_, emissiveFluxScales_, settings);
	lightTreeBuildMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
	lightTreeMaxDepth_ = tree.maxDepth;
	lightTreeAverageDepth_ = tree.averageDepth;
	lightCount_ = static_cast<uint32_t>(tree.lights.size());

	// The emitter buffer holds the indices of the emissive triangles. CollectEmitters only keeps
	// triangles the tree accepts, so they are also the tree's light indices.
	if (lightCount_ != emissiveLights_.size())
	{
		loge("GPUScene: the light tree skipped {} emissive triangle(s), emitter MIS weights will be wrong",
			 emissiveLights_.size() - lightCount_);
	}

	if (lightCount_ == 0)
	{
		// One unused element each, so the shader's bindings stay valid.
		tree.lights.push_back(GpuLight{});
		tree.nodes.push_back(GpuLightTreeNode{});
		tree.leafOfLight.push_back({});
		tree.parents.push_back(0);
		tree.leafLights.push_back({});
	}

	// BeginFrame has waited for the frame that last read this slot, so it can be overwritten.
	lightFrames_.resize(renderContext_->GetFramesInFlight());
	lightFrame_ = renderContext_->GetCurrentFrameIndex();
	LightFrameBuffers &frame = lightFrames_[lightFrame_];
	WriteLightBuffer(renderContext_, frame.lights, frame.lightCapacity, tree.lights, "Scene_Lights");
	WriteLightBuffer(renderContext_, frame.nodes, frame.nodeCapacity, tree.nodes, "Scene_LightTreeNodes");
	WriteLightBuffer(renderContext_, frame.leaves, frame.leafCapacity, tree.leafOfLight, "Scene_LightLeaves");
	WriteLightBuffer(renderContext_, frame.parents, frame.parentCapacity, tree.parents, "Scene_LightTreeParents");
	WriteLightBuffer(renderContext_, frame.leafLights, frame.leafLightCapacity, tree.leafLights, "Scene_LightTreeLeafLights");
}

void RenderGPUScene::BuildGPUScene(const Scene::GLTFScene &scene)
{
	BuildGPUScene(scene, {});
}

void RenderGPUScene::BuildGPUScene(const Scene::GLTFScene &scene, const std::filesystem::path &cachePath)
{
	meshDescs.clear();
	primitiveDescs.clear();
	vertices_.clear();
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

			desc.vertexOffset = static_cast<uint32_t>(vertices_.size());
			desc.vertexCount = static_cast<uint32_t>(primitive.GetVertexCount());
			desc.indexOffset = static_cast<uint32_t>(indices_.size());
			desc.indexCount = static_cast<uint32_t>(primitive.GetIndexCount());

			// The attribute arrays come out of the same vertex remap, so they hold one entry per vertex.
			const float *p = primitive.GetPositions().data();
			const float *n = primitive.GetNormals().data();
			const float *uv = primitive.GetTexCoord0().data();
			const float *t = primitive.GetTangents().data();
			for (size_t v = 0; v < desc.vertexCount; ++v)
			{
				GpuVertex &vertex = vertices_.emplace_back();
				vertex.position = float3(p[3 * v], p[3 * v + 1], p[3 * v + 2]);
				vertex.normal = float3(n[3 * v], n[3 * v + 1], n[3 * v + 2]);
				vertex.texCoord = float2(uv[2 * v], uv[2 * v + 1]);
				vertex.tangent = float4(t[4 * v], t[4 * v + 1], t[4 * v + 2], t[4 * v + 3]);
			}
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

	SceneCacheWriter writer;
	cacheWriter_ = !cachePath.empty() && writer.Open(cachePath) ? &writer : nullptr;

	UploadBatchHandle batch = renderContext_->BeginUpload();
	pendingUploadBytes_ = 0;
	mipBuildMs_ = 0.0;

	UploadStreams(batch);
	std::vector<GpuMaterial> materials;
	BuildMaterials(scene, batch, materials);
	UploadMaterials(materials, batch);
	std::vector<uint32_t> emitters = CollectEmitters(scene, materials);
	UploadEmitters(emitters, batch);

	renderContext_->EndUpload(batch);

	logi("GPUScene: {} mesh(es), {} primitive(s), {} instance(s), {} vertices, {} triangles, {} texture(s)",
		 meshDescs.size(),
		 primitiveDescs.size(),
		 instances.size(),
		 vertices_.size(),
		 indices_.size() / 3,
		 textures_.size());
	logi("GPUScene: built the texture mip chains in {:.0f} ms", mipBuildMs_);

	if (cacheWriter_ != nullptr)
	{
		cacheWriter_ = nullptr;
		WriteSceneCache(writer, materials, emitters);
	}

	ReleaseStaging();
}

void RenderGPUScene::ReleaseStaging()
{
	// The GPU has its own copies now.
	std::vector<GpuVertex>().swap(vertices_);
	for (std::vector<uint32_t> *stream: { &indices_, &meshletData_, &meshletVertices_, &meshletTriangles_ })
	{
		std::vector<uint32_t>().swap(*stream);
	}
}

template<typename Archive>
void RenderGPUScene::SerialiseTables(Archive &archive,
									 std::vector<GpuMaterial> &materials,
									 std::vector<uint32_t> &emitters,
									 std::vector<GpuLight> &lights)
{
	archive.Table(meshDescs);
	archive.Table(primitiveDescs);
	archive.Table(instances);
	archive.Table(vertices_);
	archive.Table(indices_);
	archive.Table(meshletData_);
	archive.Table(meshletVertices_);
	archive.Table(meshletTriangles_);
	archive.Table(materials);
	archive.Table(emitters);
	archive.Table(lights);
	archive.Table(emissiveFluxScales_);
	archive.Value(boundsMin_);
	archive.Value(boundsMax_);
}

void RenderGPUScene::WriteSceneCache(SceneCacheWriter &writer,
									 const std::vector<GpuMaterial> &materials,
									 std::vector<uint32_t> &emitters)
{
	// The file refers to textures by their order in it, which is the order of textures_.
	std::unordered_map<uint32_t, uint32_t> slotOfDescriptor;
	for (uint32_t i = 0; i < textures_.size(); ++i)
	{
		slotOfDescriptor[renderContext_->GetResourceViewInfo(textures_[i]).descriptorIndex] = i;
	}
	std::vector<GpuMaterial> cachedMaterials = materials;
	std::vector<GpuLight> cachedLights = emissiveLights_;
	RemapTextures(cachedMaterials, cachedLights, [&](uint32_t descriptor) { return slotOfDescriptor.at(descriptor); });

	SerialiseTables(writer, cachedMaterials, emitters, cachedLights);
	if (writer.Finish())
	{
		logi("GPUScene: wrote the scene cache '{}'", writer.GetPath().string());
	}
}

bool RenderGPUScene::ValidateTables(const std::vector<GpuMaterial> &materials, uint32_t textureCount) const
{
	auto validTexture = [&](uint32_t texture) { return texture == INVALIDDESCRIPTORHANDLE || texture < textureCount; };
	for (const GpuMaterial &material: materials)
	{
		for (uint32_t texture: { material.albedoTexture,
								 material.roughMetalTexture,
								 material.normalTexture,
								 material.transmissionTexture,
								 material.emissiveTexture })
		{
			if (!validTexture(texture))
			{
				return false;
			}
		}
	}
	for (const GpuLight &light: emissiveLights_)
	{
		if (!validTexture(std::bit_cast<uint32_t>(light.radiance.w)))
		{
			return false;
		}
	}
	for (const MeshGPUDesc &mesh: meshDescs)
	{
		if (static_cast<uint64_t>(mesh.primitiveOffset) + mesh.primitiveCount > primitiveDescs.size())
		{
			return false;
		}
	}
	for (const PrimitiveGPUDesc &primitive: primitiveDescs)
	{
		if (primitive.materialIndex >= materials.size() ||
			static_cast<uint64_t>(primitive.vertexOffset) + primitive.vertexCount > vertices_.size() ||
			static_cast<uint64_t>(primitive.indexOffset) + primitive.indexCount > indices_.size())
		{
			return false;
		}
	}
	for (const InstanceGPU &instance: instances)
	{
		if (instance.meshIndex >= meshDescs.size())
		{
			return false;
		}
	}
	return !materials.empty() && emissiveLights_.size() == emissiveFluxScales_.size();
}

bool RenderGPUScene::LoadGPUScene(const std::filesystem::path &cachePath)
{
	SceneCacheReader reader;
	if (!reader.Open(cachePath))
	{
		return false;
	}

	std::vector<GpuMaterial> materials;
	std::vector<uint32_t> emitters;
	SerialiseTables(reader, materials, emitters, emissiveLights_);
	const std::vector<SceneCacheTexture> &cachedTextures = reader.GetTextures();
	if (!reader.EndTables() || !ValidateTables(materials, static_cast<uint32_t>(cachedTextures.size())))
	{
		logw("GPUScene: '{}' is damaged", cachePath.string());
		meshDescs.clear();
		primitiveDescs.clear();
		instances.clear();
		emissiveLights_.clear();
		emissiveFluxScales_.clear();
		ReleaseStaging();
		return false;
	}

	UploadBatchHandle batch = renderContext_->BeginUpload();
	pendingUploadBytes_ = 0;

	UploadStreams(batch);

	// The next texture is read from disk while the current one uploads.
	std::vector<uint8_t> pixels;
	std::vector<uint8_t> nextPixels;
	auto readTexture = [&](uint32_t index)
	{
		return std::async(std::launch::async,
						  [&reader, &nextPixels, index] { return reader.ReadTexture(index, nextPixels); });
	};
	std::future<bool> pendingRead;
	if (!cachedTextures.empty())
	{
		pendingRead = readTexture(0);
	}
	std::vector<uint32_t> descriptorOfSlot;
	descriptorOfSlot.reserve(cachedTextures.size());
	for (uint32_t i = 0; i < cachedTextures.size(); ++i)
	{
		if (!pendingRead.get())
		{
			// The textures before this one are already on the GPU, so there is no falling back.
			throw std::runtime_error("failed to read a texture from '" + cachePath.string() +
									 "', delete it to rebuild it from the glTF");
		}
		pixels.swap(nextPixels);
		if (i + 1 < cachedTextures.size())
		{
			pendingRead = readTexture(i + 1);
		}

		const SceneCacheTexture &cached = cachedTextures[i];
		TextureDesc desc = {};
		desc.width = cached.width;
		desc.height = cached.height;
		desc.format = cached.srgb ? Format::R8G8B8A8_UNORM_SRGB : Format::R8G8B8A8_UNORM;
		desc.mipLevels = cached.mipLevels;
		desc.debugName = "Material_Texture";

		std::vector<TextureMipData> mips(cached.mipLevels);
		uint64_t offset = 0;
		for (uint32_t m = 0; m < mips.size(); ++m)
		{
			mips[m].pixels = pixels.data() + offset;
			mips[m].rowPitch = std::max(cached.width >> m, 1u) * 4;
			mips[m].rowCount = std::max(cached.height >> m, 1u);
			offset += SceneCacheMipBytes(cached.width, cached.height, m);
		}
		descriptorOfSlot.push_back(UploadTextureMips(desc, mips, batch));
	}

	RemapTextures(materials, emissiveLights_, [&](uint32_t slot) { return descriptorOfSlot[slot]; });
	UploadMaterials(materials, batch);
	UploadEmitters(emitters, batch);
	lightTreeBuilt_ = false;

	renderContext_->EndUpload(batch);

	logi("GPUScene: loaded '{}': {} mesh(es), {} primitive(s), {} instance(s), {} vertices, {} triangles, "
		 "{} texture(s), {} emissive triangle light(s)",
		 cachePath.string(),
		 meshDescs.size(),
		 primitiveDescs.size(),
		 instances.size(),
		 vertices_.size(),
		 indices_.size() / 3,
		 textures_.size(),
		 emissiveLights_.size());

	ReleaseStaging();
	return true;
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

	if (!vertices_.empty())
	{
		vertexBuffer_ = UploadStream(vertices_.data(),
									 vertices_.size() * sizeof(GpuVertex),
									 sizeof(GpuVertex),
									 "Scene_Vertices",
									 batch);
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

	TextureDesc desc = {};
	desc.width = static_cast<uint>(texture.GetWidth());
	desc.height = static_cast<uint>(texture.GetHeight());
	desc.format = format;
	desc.debugName = debugName;

	// The full chain, for the ray cone filtering in PathTracer.slang.
	const auto mipStart = std::chrono::steady_clock::now();
	const std::vector<std::vector<uint8_t>> levels = BuildMipChain(texture, format == Format::R8G8B8A8_UNORM_SRGB);
	mipBuildMs_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mipStart).count();

	std::vector<TextureMipData> mips(levels.size() + 1);
	for (uint32_t m = 0; m < mips.size(); ++m)
	{
		mips[m].pixels = m == 0 ? texture.GetPixels() : levels[m - 1].data();
		mips[m].rowPitch = std::max(desc.width >> m, 1u) * 4;
		mips[m].rowCount = std::max(desc.height >> m, 1u);
	}
	desc.mipLevels = static_cast<uint>(mips.size());

	if (cacheWriter_ != nullptr)
	{
		cacheWriter_->AddTexture(desc.width, desc.height, format == Format::R8G8B8A8_UNORM_SRGB, mips);
	}
	return UploadTextureMips(desc, mips, batch);
}

uint32_t RenderGPUScene::UploadTextureMips(const TextureDesc &desc,
										   std::span<const TextureMipData> mips,
										   UploadBatchHandle &batch)
{
	uint64_t sizeInBytes = 0;
	for (const TextureMipData &mip: mips)
	{
		sizeInBytes += static_cast<uint64_t>(mip.rowPitch) * mip.rowCount;
	}

	if (pendingUploadBytes_ > 0 && pendingUploadBytes_ + sizeInBytes > kMaxPendingUploadBytes)
	{
		renderContext_->EndUpload(batch);
		batch = renderContext_->BeginUpload();
		pendingUploadBytes_ = 0;
	}

	// The pixels go into a staging buffer here, so the levels can go out of scope after.
	const ResourceHandle gpuTexture = renderContext_->CreateTexture(desc, ResourceViewType::ShaderResource, mips, batch);
	pendingUploadBytes_ += sizeInBytes;
	textures_.push_back(gpuTexture);
	return renderContext_->GetResourceViewInfo(gpuTexture).descriptorIndex;
}

void RenderGPUScene::BuildMaterials(const Scene::GLTFScene &scene,
									UploadBatchHandle &batch,
									std::vector<GpuMaterial> &materials)
{
	materials.clear();
	materials.reserve(scene.GetMaterialCount() + 1);

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
		gpu.ior = material.GetIor();
		gpu.transmissionTexture = gpu.transmission > 0.0f ? UploadTexture(material.GetTransmissionTexture(),
																		  Format::R8G8B8A8_UNORM,
																		  "Material_Transmission",
																		  batch)
														  : INVALIDDESCRIPTORHANDLE;
		if (material.IsThinWalled())
		{
			gpu.transmissionFlags |= TRANSMISSION_THIN_WALLED;
			if (gpu.transmission > 0.0f && gpu.roughness <= 0.0f)
				gpu.transmissionFlags |= TRANSMISSION_PASS_THROUGH;
		}
		const float *attenuationColour = material.GetAttenuationColour();
		gpu.attenuationColour = float3(attenuationColour[0], attenuationColour[1], attenuationColour[2]);
		gpu.attenuationDistance = std::isfinite(material.GetAttenuationDistance()) && material.GetAttenuationDistance() > 0.0f
										  ? material.GetAttenuationDistance()
										  : FLT_MAX;
		gpu.albedoTexture =
				UploadTexture(material.GetAlbedoTexture(), Format::R8G8B8A8_UNORM_SRGB, "Material_Albedo", batch);
		gpu.roughMetalTexture =
				UploadTexture(material.GetRoughMetalTexture(), Format::R8G8B8A8_UNORM, "Material_RoughMetal", batch);
		gpu.normalTexture =
				UploadTexture(material.GetNormalTexture(), Format::R8G8B8A8_UNORM, "Material_Normal", batch);
		if (material.IsEmissive())
		{
			const float *emissive = material.GetEmissive();
			const float strength = material.GetEmissiveStrength();
			gpu.emissive = float3(emissive[0] * strength, emissive[1] * strength, emissive[2] * strength);
		}
		gpu.emissiveTexture = gpu.emissive.x > 0.0f || gpu.emissive.y > 0.0f || gpu.emissive.z > 0.0f
									  ? UploadTexture(material.GetEmissiveTexture(),
													  Format::R8G8B8A8_UNORM_SRGB,
													  "Material_Emissive",
													  batch)
									  : INVALIDDESCRIPTORHANDLE;
		materials.push_back(gpu);
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
	fallback.emissiveTexture = INVALIDDESCRIPTORHANDLE;
	fallback.alphaCutoff = 0.5f;
	fallback.alphaMode = ALPHA_MODE_OPAQUE;
	fallback.ior = 1.5f;
	fallback.transmissionTexture = INVALIDDESCRIPTORHANDLE;
	fallback.attenuationColour = float3(1.0f, 1.0f, 1.0f);
	fallback.attenuationDistance = FLT_MAX;
	materials.push_back(fallback);
}

void RenderGPUScene::UploadMaterials(const std::vector<GpuMaterial> &materials, UploadBatchHandle batch)
{
	materialGeometry_.clear();
	materialGeometry_.reserve(materials.size());
	for (const GpuMaterial &material: materials)
	{
		materialGeometry_.push_back(MaterialGeometryFlags(material));
	}

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

std::vector<RenderGPUScene::TLASInstanceRef> RenderGPUScene::EnumerateTLASInstances() const
{
	std::vector<TLASInstanceRef> refs;
	for (uint32_t i = 0; i < static_cast<uint32_t>(instances.size()); ++i)
	{
		const MeshGPUDesc &mesh = meshDescs[instances[i].meshIndex];
		for (uint32_t p = 0; p < mesh.primitiveCount; ++p)
		{
			// Primitives without geometry get no BLAS.
			const uint32_t primitive = mesh.primitiveOffset + p;
			if (primitiveDescs[primitive].indexCount != 0)
			{
				refs.push_back({ i, p, primitive });
			}
		}
	}
	return refs;
}

std::vector<uint32_t> RenderGPUScene::CollectEmitters(const Scene::GLTFScene &scene,
													  const std::vector<GpuMaterial> &materials)
{
	emissiveLights_.clear();
	emissiveFluxScales_.clear();

	// The per-instance offsets come first, then a triangle table for each emissive instance.
	const std::vector<TLASInstanceRef> tlasInstances = EnumerateTLASInstances();
	std::vector<uint32_t> emitters(std::max<size_t>(tlasInstances.size(), 1), INVALID_LIGHT_INDEX);
	uint32_t emissiveInstances = 0;

	for (size_t i = 0; i < tlasInstances.size(); ++i)
	{
		const TLASInstanceRef &ref = tlasInstances[i];
		const uint32_t materialIndex = primitiveDescs[ref.primitive].materialIndex;
		const GpuMaterial &material = materials[materialIndex];
		const float3 emissive = material.emissive;
		if (!(emissive.x > 0.0f || emissive.y > 0.0f || emissive.z > 0.0f))
		{
			continue;
		}

		const InstanceGPU &instance = instances[ref.instance];
		const Scene::Primitive &primitive = scene.GetMesh(instance.meshIndex).GetPrimitives()[ref.meshPrimitive];
		const std::vector<float> &positions = primitive.GetPositions();
		const std::vector<float> &texCoords = primitive.GetTexCoord0();
		const std::vector<uint32_t> &indices = primitive.GetIndices();
		const bool textured = material.emissiveTexture != INVALIDDESCRIPTORHANDLE;
		const Scene::Texture *texture = textured ? &scene.GetMaterial(materialIndex).GetEmissiveTexture() : nullptr;
		const float emissiveLuminance = Luminance(emissive);

		// The hit shader orients Ng by the object-space winding, so a mirroring transform swaps two
		// corners to keep the area light's front face (counter-clockwise) on the same side.
		const bool mirrored = Determinant3x3(instance.worldMatrix) < 0.0f;

		emitters[i] = static_cast<uint32_t>(emitters.size());
		++emissiveInstances;
		for (size_t t = 0; t + 2 < indices.size(); t += 3)
		{
			const uint32_t corners[3] = { indices[t], indices[t + (mirrored ? 2 : 1)], indices[t + (mirrored ? 1 : 2)] };
			float3 p[3];
			float2 uv[3];
			for (int c = 0; c < 3; ++c)
			{
				const size_t v = corners[c];
				p[c] = TransformPoint(instance.worldMatrix, positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
				uv[c] = 2 * v + 1 < texCoords.size() ? float2(texCoords[2 * v], texCoords[2 * v + 1]) : float2(0.0f, 0.0f);
			}

			GpuLight light;
			float fluxScale = 1.0f;
			if (texture != nullptr)
			{
				fluxScale = Luminance(emissive * AverageSrgbTexture(*texture, uv)) / emissiveLuminance;
				// Whole texture repeats do not change what wrap sampling reads, and shifting by them
				// keeps the UVs near zero where the light's half precision copies are accurate.
				const float shiftU = floorf(std::min({ uv[0].x, uv[1].x, uv[2].x }));
				const float shiftV = floorf(std::min({ uv[0].y, uv[1].y, uv[2].y }));
				for (float2 &corner: uv)
				{
					corner = float2(corner.x - shiftU, corner.y - shiftV);
				}
				light = MakeAreaLight(emissive, p[0], p[1], p[2], material.emissiveTexture, uv[0], uv[1], uv[2]);
			}
			else
			{
				light = MakeAreaLight(emissive, p[0], p[1], p[2]);
			}

			// Degenerate and (by the texture estimate) black triangles get no light. BSDF sampling
			// still finds whatever they emit, at full weight.
			uint32_t lightIndex = INVALID_LIGHT_INDEX;
			if (LightTreeFlux(light, fluxScale) > 0.0f)
			{
				lightIndex = static_cast<uint32_t>(emissiveLights_.size());
				emissiveLights_.push_back(light);
				emissiveFluxScales_.push_back(fluxScale);
			}
			emitters.push_back(lightIndex);
		}
	}

	lightTreeBuilt_ = false;
	if (emissiveInstances > 0)
	{
		logi("GPUScene: {} emissive triangle light(s) over {} TLAS instance(s)",
			 emissiveLights_.size(),
			 emissiveInstances);
	}
	return emitters;
}

void RenderGPUScene::UploadEmitters(const std::vector<uint32_t> &emitters, UploadBatchHandle batch)
{
	emitterBuffer_ = UploadStream(emitters.data(),
								  emitters.size() * sizeof(uint32_t),
								  sizeof(uint32_t),
								  "Scene_Emitters",
								  batch);
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
		// The position is the first member of each interleaved GpuVertex.
		geometry.vertexBuffer = vertexBuffer_;
		geometry.vertexBufferOffset = static_cast<uint64_t>(desc.vertexOffset) * sizeof(GpuVertex);
		geometry.vertexCount = desc.vertexCount;
		geometry.vertexStride = sizeof(GpuVertex);
		geometry.indexBuffer = indexBuffer;
		geometry.indexBufferOffset = static_cast<uint64_t>(desc.indexOffset) * sizeof(uint32_t);
		geometry.indexCount = desc.indexCount;
		const MaterialGeometryFlags &flags = materialGeometry_[desc.materialIndex];
		geometry.isOpaque = flags.isOpaque;
		geometry.noDuplicateAnyHit = flags.noDuplicateAnyHit;

		blases_[i] = renderContext_->CreateBLAS(geometry);
		renderContext_->BuildAccelerationStructure(blases_[i]);
	}

	renderContext_->UAVBarrierForAccelerationStructures();

	// One TLAS instance per (node, primitive). InstanceID is the primitive's row in the
	// PrimitiveGPUDesc table, which is all the hit shader needs to find its data. The order must
	// match the emitter buffer, which is indexed by InstanceIndex().
	std::vector<TLASInstance> tlasInstances;
	for (const TLASInstanceRef &ref: EnumerateTLASInstances())
	{
		// The scene stores row-vector matrices; DXR wants the 3x4 for column vectors.
		float transform[12];
		Scene::RowMajorToBLASTransform(instances[ref.instance].worldMatrix, transform);

		TLASInstance tlasInstance = {};
		tlasInstance.blas = blases_[ref.primitive];
		std::memcpy(tlasInstance.transform, transform, sizeof(tlasInstance.transform));
		tlasInstance.instanceID = ref.primitive;
		tlasInstance.instanceMask = 0xFF;
		tlasInstance.hitGroupIndex = 0;
		tlasInstances.push_back(tlasInstance);
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
