#include "RenderGPUScene.h"

#include "GLTFScene.h"
#include "LightTreeBuilder.h"
#include "PathTracerInterop.h"
#include "SceneCache.h"

#include "fmtlog.h"
#include "meshoptimizer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <execution>
#include <future>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace
{
	// Staging memory for one upload batch is held until EndUpload, so flush every so often to keep
	// large texture sets from needing all of their staging buffers alive at once.
	constexpr uint64_t kMaxPendingUploadBytes = 256ull * 1024 * 1024;
	constexpr int kOpacityMicromapStates = 4;
	constexpr int kOpacityMicromapMaxLevel = 6;
	constexpr float kOpacityMicromapTargetEdge = 3.0f;
	// BLEND_MIN_OPACITY in PathTracer/Materials.slang: the alpha below which a blended surface is cut.
	constexpr float kBlendMinOpacity = 1.0f / 255.0f;
	// D3D12_RAYTRACING_OPACITY_MICROMAP_STATE, two bits per micro-triangle in 4-state micromaps.
	constexpr uint8_t kMicroTriangleTransparent = 0;
	constexpr uint8_t kMicroTriangleOpaque = 1;
	constexpr uint8_t kMicroTriangleUnknownOpaque = 3;

	// A blended surface's micromap from its two cuts, in place of the first: transparent where nothing of
	// it is there, opaque where all of it is (null opaque: nowhere), unknown between, for the any-hit
	// shaders to blend. The states pair up two bits at a time, whatever order they are in.
	void CombineBlendMicromap(uint8_t *there, const uint8_t *opaque, size_t size)
	{
		for (size_t i = 0; i < size; ++i)
		{
			uint8_t combined = 0;
			for (int shift = 0; shift < 8; shift += 2)
			{
				const uint8_t thereState = (there[i] >> shift) & 3;
				const uint8_t opaqueState = opaque != nullptr ? (opaque[i] >> shift) & 3 : kMicroTriangleTransparent;
				const uint8_t state = thereState == kMicroTriangleTransparent ? kMicroTriangleTransparent
									  : opaqueState == kMicroTriangleOpaque	  ? kMicroTriangleOpaque
																			  : kMicroTriangleUnknownOpaque;
				combined |= static_cast<uint8_t>(state << shift);
			}
			there[i] = combined;
		}
	}
	constexpr uint32_t kNoOpacityMicromaps = UINT32_MAX;
	// glTF allows an IOR down to 1, an interface that hardly reflects or bends light at all, which the
	// BSDF's Fresnel and refraction handle badly.
	constexpr float kMinIor = 1.05f;

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
									  &material.emissiveTexture,
									  &material.subsurfaceWeightTexture,
									  &material.subsurfaceColourTexture,
									  &material.subsurfaceRadiusTexture,
									  &material.subsurfaceScaleTexture,
									  &material.anisotropyTexture,
									  &material.sheenColourTexture,
									  &material.sheenRoughnessTexture,
									  &material.coatWeightTexture,
									  &material.coatRoughnessTexture,
									  &material.coatNormalTexture })
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

	// Makes room for count elements (at least one, so the shader's bindings stay valid). Returns true
	// when the buffer was made again, which loses what it held.
	template<typename T>
	bool ReserveLightBuffer(RenderContext *renderContext,
							ResourceHandle &buffer,
							size_t &capacity,
							size_t count,
							const char *debugName)
	{
		if (buffer != InvalidResourceHandle && count <= capacity)
		{
			return false;
		}
		if (buffer != InvalidResourceHandle)
		{
			renderContext->DestroyResource(buffer);
		}
		capacity = std::bit_ceil(std::max<size_t>(count, 1));

		BufferDesc desc = {};
		desc.sizeInBytes = capacity * sizeof(T);
		desc.structureByteStride = sizeof(T);
		desc.debugName = debugName;
		buffer = renderContext->CreateUploadBuffer(desc, ResourceViewType::ShaderResource);
		return true;
	}

	// Writes elements from element first on.
	template<typename T>
	void WriteLightRange(RenderContext *renderContext, ResourceHandle buffer, size_t first, std::span<const T> elements)
	{
		if (!elements.empty())
		{
			renderContext->UpdateUploadBuffer(buffer, elements.data(), first * sizeof(T), elements.size_bytes());
		}
	}

	// Inverse of an affine float3x4 in TransformPoint's layout, which maps p to p A + t with
	// A[r][c] = m[4 r + c] and t = (m[3], m[7], m[11]).
	std::array<float, 12> InvertAffine(const float *m)
	{
		const float s = 1.0f / Determinant3x3(m);
		std::array<float, 12> inv;
		inv[0] = (m[5] * m[10] - m[6] * m[9]) * s;
		inv[1] = (m[2] * m[9] - m[1] * m[10]) * s;
		inv[2] = (m[1] * m[6] - m[2] * m[5]) * s;
		inv[4] = (m[6] * m[8] - m[4] * m[10]) * s;
		inv[5] = (m[0] * m[10] - m[2] * m[8]) * s;
		inv[6] = (m[2] * m[4] - m[0] * m[6]) * s;
		inv[8] = (m[4] * m[9] - m[5] * m[8]) * s;
		inv[9] = (m[1] * m[8] - m[0] * m[9]) * s;
		inv[10] = (m[0] * m[5] - m[1] * m[4]) * s;
		// p = (p' - t) A^-1, so the translation is -t A^-1.
		for (int c = 0; c < 3; ++c)
		{
			inv[4 * c + 3] = -(m[3] * inv[c] + m[7] * inv[4 + c] + m[11] * inv[8 + c]);
		}
		return inv;
	}

	// The light's triangle moved by m, its UVs kept.
	GpuLight TransformLight(GpuLight light, const float *m)
	{
		for (float4 *v: { &light.v1, &light.v2, &light.v3 })
		{
			const float3 p = TransformPoint(m, v->x, v->y, v->z);
			*v = float4(p.x, p.y, p.z, v->w);
		}
		return light;
	}
} // namespace

RenderGPUScene::RenderGPUScene(RenderContext *renderContext) : renderContext_(renderContext)
{
	std::fill(std::begin(streamBuffers_), std::end(streamBuffers_), InvalidResourceHandle);
	std::fill(std::begin(streamBufferIndices), std::end(streamBufferIndices), InvalidResourceDescriptorIndex);
}

RenderGPUScene::~RenderGPUScene()
{
	for (AccelerationStructureHandle tlas: { tlas_, ommTlas_, omm2StateTlas_ })
	{
		if (tlas != InvalidAccelerationStructureHandle)
		{
			renderContext_->DestroyAccelerationStructure(tlas);
		}
	}
	for (const std::vector<AccelerationStructureHandle> *list: { &blases_, &ommBlases_ })
	{
		for (AccelerationStructureHandle blas: *list)
		{
			if (blas != InvalidAccelerationStructureHandle)
			{
				renderContext_->DestroyAccelerationStructure(blas);
			}
		}
	}
	// After the BLASes that link it.
	if (ommArray_ != InvalidAccelerationStructureHandle)
	{
		renderContext_->DestroyAccelerationStructure(ommArray_);
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
	for (ResourceHandle buffer:
		 { vertexBuffer_, primitiveBuffer_, materialBuffer_, instanceBuffer_, emitterBuffer_, lightEmitterBuffer_, ommIndexBuffer_ })
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
		for (ResourceHandle buffer:
			 { frame.lights, frame.nodes, frame.leaves, frame.parents, frame.leafLights, frame.power, frame.emissionScales })
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
	// The static emitters' tree only changes with its settings, the dynamic emitters' also when they move
	// or their emission scales change. Other emission scales only need writing.
	const bool rebuildStatic = !lightTreeBuilt_ || settings != lightTreeSettings_;
	const bool rebuildDynamic = rebuildStatic || dynamicLightsMoved_ || dynamicLightsRescaled_;
	if (!rebuildDynamic && !emissionScalesChanged_)
	{
		return;
	}
	lightTreeBuilt_ = true;
	lightTreeSettings_ = settings;

	const std::span<const GpuLight> lights = emissiveLights_;
	const std::span<const float> fluxScales = emissiveFluxScales_;
	const uint32_t dynamicLightCount = static_cast<uint32_t>(lights.size()) - staticLightCount_;
	const bool dynamic = dynamicLightCount > 0;
	if (rebuildStatic)
	{
		// With dynamic emitters the root group leaves its last slot to their tree.
		LightTreeLayout layout;
		layout.rootWidth = dynamic ? LIGHT_TREE_WIDTH - 1 : LIGHT_TREE_WIDTH;
		const auto start = std::chrono::steady_clock::now();
		staticLightTree_ = BuildLightTree(lights.first(staticLightCount_), fluxScales.first(staticLightCount_), settings, layout);
		staticLightTreeBuildMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
		++staticTreeGeneration_;
	}

	// After the static tree's nodes, its lights and its leaf slots, which are one per static light.
	if (rebuildDynamic)
	{
		dynamicLightTree_ = {};
	}

	if (rebuildDynamic && dynamic)
	{
		const auto start = std::chrono::steady_clock::now();
		if (dynamicLightsMoved_)
		{
			TransformDynamicLights();
		}
		// Picked by the flux they emit now, so a light that is off gets no samples.
		std::vector<float> dynamicFluxScales(fluxScales.begin() + staticLightCount_, fluxScales.end());
		for (size_t j = 0; j < dynamicFluxScales.size(); ++j)
		{
			dynamicFluxScales[j] *= emissionScales_[lightEmitters_[staticLightCount_ + j]];
		}
		LightTreeLayout layout;
		layout.firstNode = static_cast<uint32_t>(staticLightTree_.nodes.size());
		layout.firstLight = staticLightCount_;
		layout.rootParent = LIGHT_TREE_WIDTH - 1;
		dynamicLightTree_ = BuildLightTree(lights.subspan(staticLightCount_), dynamicFluxScales, settings, layout);
		dynamicLightTreeBuildMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
	}
	
	const LightTree &dynamicTree = dynamicLightTree_;
	dynamicLightsMoved_ = false;
	dynamicLightsRescaled_ = false;
	emissionScalesChanged_ = false;

	const double flux = staticLightTree_.flux + dynamicTree.flux;
	staticLightPowerShare_ = flux > 0.0 ? static_cast<float>(staticLightTree_.flux / flux) : (staticLightCount_ > 0 ? 1.0f : 0.0f);
	lightCount_ = static_cast<uint32_t>(lights.size());
	// The dynamic emitters sit a level deeper, under the root group's last slot.
	const uint32_t treeLights = staticLightTree_.treeLights + dynamicTree.treeLights;
	lightTreeMaxDepth_ = std::max(staticLightTree_.maxDepth, dynamicTree.treeLights > 0 ? dynamicTree.maxDepth + 1 : 0);
	lightTreeAverageDepth_ = treeLights == 0 ? 0.0f
											 : (staticLightTree_.averageDepth * staticLightTree_.treeLights +
												(dynamicTree.averageDepth + 1.0f) * dynamicTree.treeLights) /
													   static_cast<float>(treeLights);

	// BeginFrame has waited for the frame that last read this slot, so it can be overwritten. Its
	// static part only needs writing once per static build, unless a buffer had to grow.
	lightFrames_.resize(renderContext_->GetFramesInFlight());
	lightFrame_ = renderContext_->GetCurrentFrameIndex();
	LightFrameBuffers &frame = lightFrames_[lightFrame_];
	const size_t staticNodes = staticLightTree_.nodes.size();
	const size_t staticGroups = staticLightTree_.parents.size();
	bool recreated = false;
	recreated |= ReserveLightBuffer<GpuLight>(renderContext_, frame.lights, frame.lightCapacity, lights.size(), "Scene_Lights");
	recreated |= ReserveLightBuffer<GpuLightTreeNode>(renderContext_,
													  frame.nodes,
													  frame.nodeCapacity,
													  staticNodes + (dynamic ? dynamicTree.nodes.size() : 0),
													  "Scene_LightTreeNodes");
	recreated |= ReserveLightBuffer<GpuLightTreeLeafEntry>(renderContext_,
														   frame.leaves,
														   frame.leafCapacity,
														   lights.size(),
														   "Scene_LightLeaves");
	recreated |= ReserveLightBuffer<uint32_t>(renderContext_,
											  frame.parents,
											  frame.parentCapacity,
											  staticGroups + (dynamic ? dynamicTree.parents.size() : 0),
											  "Scene_LightTreeParents");
	recreated |= ReserveLightBuffer<GpuLightTreeLeafEntry>(renderContext_,
														   frame.leafLights,
														   frame.leafLightCapacity,
														   staticLightCount_ + dynamicTree.leafLights.size(),
														   "Scene_LightTreeLeafLights");
	recreated |= ReserveLightBuffer<GpuLightAliasEntry>(renderContext_, frame.power, frame.powerCapacity, lights.size(), "Scene_LightPower");
	ReserveLightBuffer<float>(renderContext_,
							  frame.emissionScales,
							  frame.emissionScaleCapacity,
							  emissionScales_.size(),
							  "Scene_EmissionScales");
	WriteLightRange<float>(renderContext_, frame.emissionScales, 0, emissionScales_);

	if (recreated || frame.staticTreeGeneration != staticTreeGeneration_)
	{
		frame.staticTreeGeneration = staticTreeGeneration_;
		WriteLightRange(renderContext_, frame.lights, 0, lights.first(staticLightCount_));
		WriteLightRange<GpuLightTreeNode>(renderContext_, frame.nodes, 0, staticLightTree_.nodes);
		WriteLightRange<GpuLightTreeLeafEntry>(renderContext_, frame.leaves, 0, staticLightTree_.leafOfLight);
		WriteLightRange<uint32_t>(renderContext_, frame.parents, 0, staticLightTree_.parents);
		WriteLightRange<GpuLightTreeLeafEntry>(renderContext_, frame.leafLights, 0, staticLightTree_.leafLights);
		WriteLightRange<GpuLightAliasEntry>(renderContext_, frame.power, 0, staticLightTree_.powerTable);
	}
	if (dynamic)
	{
		const GpuLightTreeNode dynamicRoot = dynamicTree.flux > 0.0 ? dynamicTree.root : EmptyLightTreeNode();
		WriteLightRange<GpuLightTreeNode>(renderContext_, frame.nodes, LIGHT_TREE_WIDTH - 1, std::span(&dynamicRoot, 1));
		WriteLightRange(renderContext_, frame.lights, staticLightCount_, lights.subspan(staticLightCount_));
		WriteLightRange<GpuLightTreeNode>(renderContext_, frame.nodes, staticNodes, dynamicTree.nodes);
		WriteLightRange<GpuLightTreeLeafEntry>(renderContext_, frame.leaves, staticLightCount_, dynamicTree.leafOfLight);
		WriteLightRange<uint32_t>(renderContext_, frame.parents, staticGroups, dynamicTree.parents);
		WriteLightRange<GpuLightTreeLeafEntry>(renderContext_, frame.leafLights, staticLightCount_, dynamicTree.leafLights);
		WriteLightRange<GpuLightAliasEntry>(renderContext_, frame.power, staticLightCount_, dynamicTree.powerTable);
	}
}

void RenderGPUScene::SetDynamicInstanceTransform(uint32_t dynamicInstance, const float (&worldMatrix)[12])
{
	std::memcpy(instances[dynamicInstances_[dynamicInstance]].worldMatrix, worldMatrix, sizeof(worldMatrix));
	dynamicLightsMoved_ = true;
	dynamicTLASMoved_ = true;
}

void RenderGPUScene::SetEmissionScale(uint32_t emissiveInstance, float scale)
{
	scale = std::max(scale, 0.0f);
	if (emissiveInstance >= emissionScales_.size() || emissionScales_[emissiveInstance] == scale)
	{
		return;
	}
	emissionScales_[emissiveInstance] = scale;
	emissionScalesChanged_ = true;
	dynamicLightsRescaled_ |= GetDynamicInstanceOfEmitter(emissiveInstance) != UINT32_MAX;
}

void RenderGPUScene::TransformDynamicLights()
{
	std::vector<std::array<float, 12>> transforms(dynamicInstances_.size());
	std::vector<uint8_t> swap(dynamicInstances_.size());
	for (size_t i = 0; i < dynamicInstances_.size(); ++i)
	{
		const float *m = instances[dynamicInstances_[i]].worldMatrix;
		std::copy(m, m + 12, transforms[i].begin());
		// The object-space corners are in the order CollectEmitters gave them at rest, which already
		// accounts for a rest transform that mirrors.
		swap[i] = (Determinant3x3(m) < 0.0f) != (dynamicRestMirrored_[i] != 0);
	}
	for (size_t j = 0; j < dynamicObjectLights_.size(); ++j)
	{
		const uint32_t instance = dynamicLightInstance_[j];
		GpuLight light = TransformLight(dynamicObjectLights_[j], transforms[instance].data());
		if (swap[instance])
		{
			std::swap(light.v2, light.v3);
		}
		emissiveLights_[staticLightCount_ + j] = light;
	}
}

void RenderGPUScene::PartitionDynamicLights(std::vector<uint32_t> &emitters)
{
	dynamicInstances_.clear();
	dynamicOfInstance_.assign(instances.size(), UINT32_MAX);
	dynamicRestTransforms_.clear();
	dynamicRestMirrored_.clear();
	dynamicObjectLights_.clear();
	dynamicLightInstance_.clear();
	dynamicLightsMoved_ = false;
	dynamicLightsRescaled_ = false;
	dynamicTLASMoved_ = false;
	emissiveInstances_.clear();

	const uint32_t lightCount = static_cast<uint32_t>(emissiveLights_.size());
	staticLightCount_ = lightCount;
	const std::vector<TLASInstanceRef> refs = EnumerateTLASInstances();

	// Calls visit(scene instance, light) for each triangle light, through the emitter tables.
	auto forEachLight = [&](auto visit)
	{
		for (size_t i = 0; i < refs.size(); ++i)
		{
			const uint32_t table = emitters[i];
			if (table == INVALID_LIGHT_INDEX)
			{
				continue;
			}
			const uint32_t triangles = primitiveDescs[refs[i].primitive].indexCount / 3;
			for (uint32_t t = 0; t < triangles; ++t)
			{
				uint32_t &light = emitters[table + t];
				if (light != INVALID_LIGHT_INDEX)
				{
					visit(refs[i].instance, light);
				}
			}
		}
	};

	std::vector<uint32_t> instanceLights(instances.size(), 0);
	std::vector<uint8_t> emissive(instances.size(), 0);
	for (size_t i = 0; i < refs.size(); ++i)
	{
		emissive[refs[i].instance] |= emitters[i] != INVALID_LIGHT_INDEX;
	}
	forEachLight([&](uint32_t instance, uint32_t &) { ++instanceLights[instance]; });

	uint32_t ordinal = 0;
	std::vector<uint32_t> emissiveOfInstance(instances.size(), UINT32_MAX);
	for (uint32_t instance = 0; instance < static_cast<uint32_t>(instances.size()); ++instance)
	{
		if (!emissive[instance])
		{
			continue;
		}
		emissiveOfInstance[instance] = ordinal;
		emissiveInstances_.push_back(instance);
		const bool dynamic = std::find(dynamicEmitterSelection_.begin(), dynamicEmitterSelection_.end(), ordinal) !=
							 dynamicEmitterSelection_.end();
		const float *m = instances[instance].worldMatrix;
		logi("GPUScene: emissive instance {} (scene instance {} at {:.2f}, {:.2f}, {:.2f}): {} triangle light(s), {}",
			 ordinal,
			 instance,
			 m[3],
			 m[7],
			 m[11],
			 instanceLights[instance],
			 dynamic ? "dynamic" : "static");
		if (dynamic)
		{
			dynamicOfInstance_[instance] = static_cast<uint32_t>(dynamicInstances_.size());
			dynamicInstances_.push_back(instance);
			std::array<float, 12> &rest = dynamicRestTransforms_.emplace_back();
			std::copy(m, m + 12, rest.begin());
			dynamicRestMirrored_.push_back(Determinant3x3(m) < 0.0f);
		}
		++ordinal;
	}
	// Every emissive instance at its loaded emission, and each light's instance in the final order.
	emissionScales_.assign(emissiveInstances_.size(), 1.0f);
	emissionScalesChanged_ = true;
	auto findLightEmitters = [&]
	{
		lightEmitters_.assign(lightCount, 0);
		forEachLight([&](uint32_t instance, uint32_t &light) { lightEmitters_[light] = emissiveOfInstance[instance]; });
	};
	if (dynamicInstances_.empty())
	{
		findLightEmitters();
		return;
	}

	// Static lights first and dynamic ones after, each in their old order.
	std::vector<uint32_t> dynamicOfLight(lightCount, UINT32_MAX);
	forEachLight([&](uint32_t instance, uint32_t &light) { dynamicOfLight[light] = dynamicOfInstance_[instance]; });
	std::vector<uint32_t> newIndex(lightCount);
	uint32_t next = 0;
	for (uint32_t light = 0; light < lightCount; ++light)
	{
		if (dynamicOfLight[light] == UINT32_MAX)
		{
			newIndex[light] = next++;
		}
	}
	staticLightCount_ = next;

	std::vector<std::array<float, 12>> objectFromWorld;
	for (const std::array<float, 12> &rest: dynamicRestTransforms_)
	{
		objectFromWorld.push_back(InvertAffine(rest.data()));
	}
	for (uint32_t light = 0; light < lightCount; ++light)
	{
		const uint32_t dynamicInstance = dynamicOfLight[light];
		if (dynamicInstance != UINT32_MAX)
		{
			newIndex[light] = next++;
			dynamicLightInstance_.push_back(dynamicInstance);
			dynamicObjectLights_.push_back(TransformLight(emissiveLights_[light], objectFromWorld[dynamicInstance].data()));
		}
	}

	std::vector<GpuLight> lights(lightCount);
	std::vector<float> fluxScales(lightCount);
	for (uint32_t light = 0; light < lightCount; ++light)
	{
		lights[newIndex[light]] = emissiveLights_[light];
		fluxScales[newIndex[light]] = emissiveFluxScales_[light];
	}
	emissiveLights_ = std::move(lights);
	emissiveFluxScales_ = std::move(fluxScales);
	forEachLight([&](uint32_t, uint32_t &light) { light = newIndex[light]; });
	findLightEmitters();
	lightTreeBuilt_ = false;
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

	// The material textures go to the GPU, and into the cache, as BC7.
	textureCompressor_ = renderContext_->CreateTextureCompressor();
	if (textureCompressor_ != nullptr && !textureCompressor_->Initialise())
	{
		logw("GPUScene: the BC7 encoder failed to initialise, so the textures stay uncompressed");
		textureCompressor_.reset();
	}
	compressMs_ = 0.0;
	compressedTextures_ = 0;
	uncompressedTextures_ = 0;

	UploadBatchHandle batch = renderContext_->BeginUpload();
	pendingUploadBytes_ = 0;
	mipBuildMs_ = 0.0;
	uploadedTextures_.clear();

	UploadStreams(batch);
	std::vector<GpuMaterial> materials;
	BuildMaterials(scene, batch, materials);
	UploadMaterials(materials, batch);
	std::vector<uint32_t> emitters = CollectEmitters(scene, materials);
	PartitionDynamicLights(emitters);
	UploadEmitters(emitters, batch);
	BakeOpacityMicromaps(scene, materials);
	UploadOpacityMicromapIndices(batch);

	renderContext_->EndUpload(batch);
	uploadedTextures_.clear();
	textureCompressor_.reset();

	logi("GPUScene: {} mesh(es), {} primitive(s), {} instance(s), {} vertices, {} triangles, {} texture(s)",
		 meshDescs.size(),
		 primitiveDescs.size(),
		 instances.size(),
		 vertices_.size(),
		 indices_.size() / 3,
		 textures_.size());
	logi("GPUScene: built the texture mip chains in {:.0f} ms", mipBuildMs_);
	logi("GPUScene: compressed {} texture(s) to BC7 in {:.0f} ms, {} left as RGBA8",
		 compressedTextures_,
		 compressMs_,
		 uncompressedTextures_);

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
	std::vector<int32_t>().swap(ommIndices_);
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
	archive.Table(ommData_);
	archive.Table(ommMicromaps_);
	archive.Table(ommIndices_);
	archive.Table(ommIndexOffsets_);
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
								 material.emissiveTexture,
								 material.subsurfaceWeightTexture,
								 material.subsurfaceColourTexture,
								 material.subsurfaceRadiusTexture,
								 material.subsurfaceScaleTexture,
								 material.anisotropyTexture,
								 material.sheenColourTexture,
								 material.sheenRoughnessTexture,
								 material.coatWeightTexture,
								 material.coatRoughnessTexture,
								 material.coatNormalTexture })
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
	if (ommIndexOffsets_.size() != primitiveDescs.size())
	{
		return false;
	}
	for (size_t i = 0; i < primitiveDescs.size(); ++i)
	{
		if (ommIndexOffsets_[i] != kNoOpacityMicromaps &&
			static_cast<uint64_t>(ommIndexOffsets_[i]) + primitiveDescs[i].indexCount / 3 > ommIndices_.size())
		{
			return false;
		}
	}
	for (const OpacityMicromap &micromap: ommMicromaps_)
	{
		if (micromap.subdivisionLevel > kOpacityMicromapMaxLevel || micromap.states != kOpacityMicromapStates ||
			micromap.byteOffset + meshopt_opacityMapEntrySize(micromap.subdivisionLevel, micromap.states) >
					ommData_.size())
		{
			return false;
		}
	}
	for (int32_t index: ommIndices_)
	{
		if (index < OpacityMicromapFullyUnknownOpaque || index >= static_cast<int64_t>(ommMicromaps_.size()))
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
		ommData_.clear();
		ommMicromaps_.clear();
		ommIndexOffsets_.clear();
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
		desc.format = cached.format;
		desc.mipLevels = cached.mipLevels;
		desc.debugName = "Material_Texture";

		std::vector<TextureMipData> mips(cached.mipLevels);
		uint64_t offset = 0;
		for (uint32_t m = 0; m < mips.size(); ++m)
		{
			mips[m] = SceneCacheMipLayout(cached, m);
			mips[m].pixels = pixels.data() + offset;
			offset += SceneCacheMipBytes(cached, m);
		}
		descriptorOfSlot.push_back(UploadTextureMips(desc, mips, batch));
	}

	RemapTextures(materials, emissiveLights_, [&](uint32_t slot) { return descriptorOfSlot[slot]; });
	UploadMaterials(materials, batch);
	PartitionDynamicLights(emitters);
	UploadEmitters(emitters, batch);
	UploadOpacityMicromapIndices(batch);
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

	// The meshlet streams stay on the CPU: nothing on the GPU reads them.
	const StreamSource sources[] = {
		{ GPUStream::Index, indices_.data(), indices_.size() * sizeof(uint32_t), sizeof(uint32_t), "Scene_Indices" },
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

	const auto [uploaded, isNew] = uploadedTextures_.try_emplace({ texture.GetPixels(), format }, 0u);
	if (!isNew)
	{
		return uploaded->second;
	}

	TextureDesc desc = {};
	desc.width = static_cast<uint>(texture.GetWidth());
	desc.height = static_cast<uint>(texture.GetHeight());
	desc.format = format;
	desc.debugName = debugName;

	// The full chain, for the ray cone filtering in PathTracer/Geometry.slang.
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

	DirectX::ScratchImage compressed;
	if (CompressBC7(desc, mips, compressed))
	{
		desc.format = format == Format::R8G8B8A8_UNORM_SRGB ? Format::BC7_UNORM_SRGB : Format::BC7_UNORM;
		for (uint32_t m = 0; m < mips.size(); ++m)
		{
			const DirectX::Image *image = compressed.GetImage(m, 0, 0);
			mips[m].pixels = image->pixels;
			mips[m].rowPitch = static_cast<uint32_t>(image->rowPitch);
			mips[m].rowCount = static_cast<uint32_t>(image->slicePitch / image->rowPitch); // rows of blocks
		}
	}

	if (cacheWriter_ != nullptr)
	{
		cacheWriter_->AddTexture(desc.width, desc.height, desc.format, mips);
	}
	uploaded->second = UploadTextureMips(desc, mips, batch);
	return uploaded->second;
}

bool RenderGPUScene::CompressBC7(const TextureDesc &desc,
								 std::span<const TextureMipData> mips,
								 DirectX::ScratchImage &compressed)
{
	if (textureCompressor_ == nullptr)
	{
		return false;
	}
	// A block-compressed texture's top mip must be whole blocks.
	if (desc.width % 4 != 0 || desc.height % 4 != 0)
	{
		++uncompressedTextures_;
		return false;
	}

	const auto start = std::chrono::steady_clock::now();
	DirectX::ScratchImage source;
	if (FAILED(source.Initialize2D(DXGI_FORMAT_R8G8B8A8_UNORM, desc.width, desc.height, 1, mips.size())))
	{
		++uncompressedTextures_;
		return false;
	}
	for (uint32_t m = 0; m < mips.size(); ++m)
	{
		const DirectX::Image *image = source.GetImage(m, 0, 0);
		const uint8_t *src = static_cast<const uint8_t *>(mips[m].pixels);
		for (uint32_t row = 0; row < mips[m].rowCount; ++row)
		{
			std::memcpy(image->pixels + row * image->rowPitch, src + row * mips[m].rowPitch, mips[m].rowPitch);
		}
	}

	// The sRGB bytes are encoded as they are; the format only changes how they are sampled.
	const DXGI_FORMAT bcFormat =
			desc.format == Format::R8G8B8A8_UNORM_SRGB ? DXGI_FORMAT_BC7_UNORM_SRGB : DXGI_FORMAT_BC7_UNORM;
	const HRESULT hr = textureCompressor_->Compress(source, bcFormat, compressed);
	compressMs_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	if (FAILED(hr))
	{
		logw("GPUScene: BC7 compression of a {}x{} texture failed ({:#x}), so it stays uncompressed",
			 desc.width,
			 desc.height,
			 static_cast<uint32_t>(hr));
		++uncompressedTextures_;
		return false;
	}
	++compressedTextures_;
	return true;
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
	const ResourceHandle gpuTexture =
			renderContext_->CreateTexture(desc, ResourceViewType::ShaderResource, mips, batch);
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

		MaterialData gpu = {};
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
		// Strength along the rotation as one half2, which the texture's direction is rotated by.
		const float anisotropyStrength = std::clamp(material.GetAnisotropyStrength(), 0.0f, 1.0f);
		const float anisotropyRotation = material.GetAnisotropyRotation();
		gpu.anisotropy = float2(anisotropyStrength * std::cos(anisotropyRotation),
								anisotropyStrength * std::sin(anisotropyRotation));
		gpu.anisotropyTexture = anisotropyStrength > 0.0f ? UploadTexture(material.GetAnisotropyTexture(),
																		  Format::R8G8B8A8_UNORM,
																		  "Material_Anisotropy",
																		  batch)
														  : INVALIDDESCRIPTORHANDLE;
		// KHR_materials_clearcoat as OpenPBR coat. The weight and roughness textures are often one image
		// (R and G), which then shares its upload.
		gpu.coatWeightTexture = INVALIDDESCRIPTORHANDLE;
		gpu.coatRoughnessTexture = INVALIDDESCRIPTORHANDLE;
		gpu.coatNormalTexture = INVALIDDESCRIPTORHANDLE;
		gpu.coatNormalScale = 1.0f;
		if (material.HasClearcoat())
		{
			gpu.coatWeight = std::clamp(material.GetClearcoat(), 0.0f, 1.0f);
			gpu.coatRoughness = std::clamp(material.GetClearcoatRoughness(), 0.0f, 1.0f);
			gpu.coatNormalScale = material.GetClearcoatNormalScale();
			gpu.coatWeightTexture =
					UploadTexture(material.GetClearcoatTexture(), Format::R8G8B8A8_UNORM, "Material_Clearcoat", batch);
			gpu.coatRoughnessTexture = UploadTexture(material.GetClearcoatRoughnessTexture(),
													 Format::R8G8B8A8_UNORM,
													 "Material_ClearcoatRoughness",
													 batch);
			gpu.coatNormalTexture = UploadTexture(material.GetClearcoatNormalTexture(),
												  Format::R8G8B8A8_UNORM,
												  "Material_ClearcoatNormal",
												  batch);
		}
		// Only the file makes a material thin-walled (EXT_materials_thin_wall). Anisotropy stretches the
		// lobe even at roughness 0, and a rough coat roughens the base, so such a wall is never a delta
		// pass-through.
		if (material.IsThinWalled())
		{
			gpu.transmissionFlags |= TRANSMISSION_THIN_WALLED;
			if (gpu.transmission > 0.0f && gpu.roughness <= 0.0f && anisotropyStrength <= 0.0f &&
				(gpu.coatWeight <= 0.0f || gpu.coatRoughness <= 0.0f))
				gpu.transmissionFlags |= TRANSMISSION_PASS_THROUGH;
		}
		// KHR_materials_specular. The colour may exceed 1 (SetGltfSpecular moves the excess into the weight).
		const float *specularColour = material.GetSpecularColour();
		gpu.specularWeight = std::max(material.GetSpecular(), 0.0f);
		gpu.specularColour = float3(std::max(specularColour[0], 0.0f),
									std::max(specularColour[1], 0.0f),
									std::max(specularColour[2], 0.0f));
		gpu.specularTexture =
				UploadTexture(material.GetSpecularTexture(), Format::R8G8B8A8_UNORM, "Material_Specular", batch);
		gpu.specularColourTexture = UploadTexture(material.GetSpecularColourTexture(),
												  Format::R8G8B8A8_UNORM_SRGB,
												  "Material_SpecularColour",
												  batch);
		// KHR_materials_volume, which the shader uses in place of the base colour's tint on solid
		// transmission. Infinite packs as no absorption.
		const float *attenuationColour = material.GetAttenuationColour();
		gpu.attenuationColour = float3(std::clamp(attenuationColour[0], 0.0f, 1.0f),
									   std::clamp(attenuationColour[1], 0.0f, 1.0f),
									   std::clamp(attenuationColour[2], 0.0f, 1.0f));
		gpu.attenuationDistance = std::min(material.GetAttenuationDistance(), FLT_MAX);
		// OpenPBR subsurface from EXT_materials_subsurface. Like Blender, its colour is the base colour. On a
		// thin wall it is split between reflection and diffuse transmission to the other side by the
		// anisotropy; in a solid object it is a scattering medium whose mean free path is radius * scale.
		gpu.subsurfaceWeightTexture = INVALIDDESCRIPTORHANDLE;
		gpu.subsurfaceColourTexture = INVALIDDESCRIPTORHANDLE;
		gpu.subsurfaceRadiusTexture = INVALIDDESCRIPTORHANDLE;
		gpu.subsurfaceScaleTexture = INVALIDDESCRIPTORHANDLE;
		gpu.subsurfaceWeight = material.GetSubsurfaceWeight();
		gpu.subsurfaceColour = float3(baseColour[0], baseColour[1], baseColour[2]);
		gpu.subsurfaceWeightChannel = std::min(material.GetSubsurfaceWeightChannel(), 3u);
		if (gpu.subsurfaceWeight > 0.0f)
		{
			gpu.subsurfaceWeightTexture = UploadTexture(material.GetSubsurfaceWeightTexture(),
														Format::R8G8B8A8_UNORM,
														"Material_SubsurfaceWeight",
														batch);
			// sRGB, so it shares the albedo texture.
			gpu.subsurfaceColourTexture = UploadTexture(material.GetAlbedoTexture(),
														Format::R8G8B8A8_UNORM_SRGB,
														"Material_Albedo",
														batch);
		}
		if (material.IsThinWalled())
		{
			// -1 all reflected, 1 all transmitted.
			gpu.subsurfaceScatterAnisotropy = std::clamp(material.GetSubsurfaceScatterAnisotropy(), -1.0f, 1.0f);
		}
		else
		{
			const float *radius = material.GetSubsurfaceRadius();
			const float scale = std::max(material.GetSubsurfaceScale(), 0.0f);
			gpu.subsurfaceRadius = float3(std::max(radius[0], 0.0f) * scale,
										  std::max(radius[1], 0.0f) * scale,
										  std::max(radius[2], 0.0f) * scale);
			gpu.subsurfaceScatterAnisotropy = std::clamp(material.GetSubsurfaceScatterAnisotropy(), -0.99f, 0.99f);
			gpu.subsurfaceScaleChannel = std::min(material.GetSubsurfaceScaleChannel(), 3u);
			if (gpu.subsurfaceWeight > 0.0f)
			{
				gpu.subsurfaceRadiusTexture = UploadTexture(material.GetSubsurfaceRadiusTexture(),
															Format::R8G8B8A8_UNORM,
															"Material_SubsurfaceRadius",
															batch);
				gpu.subsurfaceScaleTexture = UploadTexture(material.GetSubsurfaceScaleTexture(),
														   Format::R8G8B8A8_UNORM,
														   "Material_SubsurfaceScale",
														   batch);
			}
		}
		// KHR_materials_sheen, which the shader turns into OpenPBR fuzz. sRGB leaves alpha linear, so a
		// roughness texture that is the colour texture's image shares its upload.
		gpu.sheenColourTexture = INVALIDDESCRIPTORHANDLE;
		gpu.sheenRoughnessTexture = INVALIDDESCRIPTORHANDLE;
		if (material.HasSheen())
		{
			const float *sheenColour = material.GetSheenColour();
			gpu.sheenColour = float3(std::clamp(sheenColour[0], 0.0f, 1.0f),
									 std::clamp(sheenColour[1], 0.0f, 1.0f),
									 std::clamp(sheenColour[2], 0.0f, 1.0f));
			gpu.sheenRoughness = std::clamp(material.GetSheenRoughness(), 0.0f, 1.0f);
			const Scene::Texture &colourTexture = material.GetSheenColourTexture();
			const Scene::Texture &roughnessTexture = material.GetSheenRoughnessTexture();
			const bool shared = colourTexture.IsValid() && roughnessTexture.IsValid() &&
								colourTexture.GetPixels() == roughnessTexture.GetPixels();
			gpu.sheenColourTexture =
					UploadTexture(colourTexture, Format::R8G8B8A8_UNORM_SRGB, "Material_SheenColour", batch);
			gpu.sheenRoughnessTexture = UploadTexture(roughnessTexture,
													  shared ? Format::R8G8B8A8_UNORM_SRGB : Format::R8G8B8A8_UNORM,
													  "Material_SheenRoughness",
													  batch);
		}
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
		materials.push_back(PackGpuMaterial(gpu));
	}

	// Default material (glTF spec defaults would be metallic 1, which reads badly on untextured
	// geometry, so use a plain grey dielectric instead).
	MaterialData fallback = {};
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
	fallback.subsurfaceColour = float3(1.0f, 1.0f, 1.0f);
	fallback.subsurfaceWeightTexture = INVALIDDESCRIPTORHANDLE;
	fallback.subsurfaceColourTexture = INVALIDDESCRIPTORHANDLE;
	fallback.subsurfaceRadiusTexture = INVALIDDESCRIPTORHANDLE;
	fallback.subsurfaceScaleTexture = INVALIDDESCRIPTORHANDLE;
	fallback.anisotropyTexture = INVALIDDESCRIPTORHANDLE;
	fallback.sheenColourTexture = INVALIDDESCRIPTORHANDLE;
	fallback.sheenRoughnessTexture = INVALIDDESCRIPTORHANDLE;
	fallback.coatNormalScale = 1.0f;
	fallback.coatWeightTexture = INVALIDDESCRIPTORHANDLE;
	fallback.coatRoughnessTexture = INVALIDDESCRIPTORHANDLE;
	fallback.coatNormalTexture = INVALIDDESCRIPTORHANDLE;
	fallback.specularWeight = 1.0f;
	fallback.specularColour = float3(1.0f, 1.0f, 1.0f);
	fallback.specularTexture = INVALIDDESCRIPTORHANDLE;
	fallback.specularColourTexture = INVALIDDESCRIPTORHANDLE;
	materials.push_back(PackGpuMaterial(fallback));
}

void RenderGPUScene::UploadMaterials(const std::vector<GpuMaterial> &sourceMaterials, UploadBatchHandle batch)
{
	std::vector<GpuMaterial> materials = sourceMaterials;
	materialGeometry_.clear();
	materialGeometry_.reserve(materials.size());
	for (GpuMaterial &packed: materials)
	{
		// Clamped here rather than baked, so cached scenes get it too.
		if (UnpackGpuMaterial(packed).ior < kMinIor)
		{
			packed.transmissionIor = (packed.transmissionIor & 0xFFFFu) |
									 (static_cast<uint32_t>(DirectX::PackedVector::XMConvertFloatToHalf(kMinIor)) << 16);
		}
		const MaterialData material = UnpackGpuMaterial(packed);
		// Derived here rather than baked, so cached scenes get it too.
		if (IsSimpleMaterial(packed))
		{
			packed.emissiveFlags.y |= GPU_MATERIAL_SIMPLE;
		}
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
		const MaterialData material = UnpackGpuMaterial(materials[materialIndex]);
		// A blended emitter is only there by its alpha, which the light has its factor of; a texture's alpha
		// would need the albedo texture in the light, so NEE leaves that out.
		const float3 emissive = material.alphaMode == ALPHA_MODE_BLEND ? material.emissive * std::clamp(material.baseColour.w, 0.0f, 1.0f)
																	   : material.emissive;
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
			const uint32_t corners[3] = { indices[t],
										  indices[t + (mirrored ? 2 : 1)],
										  indices[t + (mirrored ? 1 : 2)] };
			float3 p[3];
			float2 uv[3];
			for (int c = 0; c < 3; ++c)
			{
				const size_t v = corners[c];
				p[c] = TransformPoint(instance.worldMatrix,
									  positions[3 * v],
									  positions[3 * v + 1],
									  positions[3 * v + 2]);
				uv[c] = 2 * v + 1 < texCoords.size() ? float2(texCoords[2 * v], texCoords[2 * v + 1])
													 : float2(0.0f, 0.0f);
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
	// At least one entry, so the shader's binding stays valid without lights.
	const uint32_t noLight = 0;
	lightEmitterBuffer_ = UploadStream(lightEmitters_.empty() ? &noLight : lightEmitters_.data(),
									   std::max<size_t>(lightEmitters_.size(), 1) * sizeof(uint32_t),
									   sizeof(uint32_t),
									   "Scene_LightEmitters",
									   batch);
}

void RenderGPUScene::BakeOpacityMicromaps(const Scene::GLTFScene &scene, const std::vector<GpuMaterial> &materials)
{
	const auto start = std::chrono::steady_clock::now();
	ommData_.clear();
	ommMicromaps_.clear();
	ommIndices_.clear();
	ommIndexOffsets_.assign(primitiveDescs.size(), kNoOpacityMicromaps);

	// meshopt_opacityMapRasterize cuts at alpha 0.5 of the plane it reads.
	struct AlphaPlane
	{
		const uint8_t *data;
		size_t stride;
		size_t pitch;
		uint32_t width;
		uint32_t height;
	};
	// The texture's own alpha where the cut is at half of it, otherwise a copy remapped to put the cut
	// there, kept per material and cut (in texture alpha, in (0, 1]).
	std::map<std::pair<uint32_t, float>, std::vector<uint8_t>> remappedAlpha;
	std::map<std::pair<uint32_t, float>, AlphaPlane> planes;
	auto planeFor = [&](uint32_t materialIndex, const Scene::Texture &texture, float threshold) -> const AlphaPlane *
	{
		auto [plane, isNew] = planes.try_emplace({ materialIndex, threshold });
		if (!isNew)
		{
			return &plane->second;
		}
		const uint32_t width = static_cast<uint32_t>(texture.GetWidth());
		const uint32_t height = static_cast<uint32_t>(texture.GetHeight());
		if (std::fabs(threshold - 0.5f) < 0.5f / 255.0f)
		{
			plane->second = { texture.GetPixels() + 3, 4, static_cast<size_t>(width) * 4, width, height };
			return &plane->second;
		}
		// Piecewise linear, threshold to 0.5, so the same texels pass.
		std::array<uint8_t, 256> remap;
		for (int v = 0; v < 256; ++v)
		{
			const float a = static_cast<float>(v) / 255.0f;
			const float mapped = a < threshold ? 0.5f * a / threshold
											   : (threshold < 1.0f ? 0.5f + 0.5f * (a - threshold) / (1.0f - threshold) : 1.0f);
			remap[v] = static_cast<uint8_t>(std::clamp(mapped * 255.0f + 0.5f, 0.0f, 255.0f));
		}
		std::vector<uint8_t> &alpha = remappedAlpha[{ materialIndex, threshold }];
		alpha.resize(static_cast<size_t>(width) * height);
		const uint8_t *pixels = texture.GetPixels();
		for (size_t i = 0; i < alpha.size(); ++i)
		{
			alpha[i] = remap[pixels[4 * i + 3]];
		}
		plane->second = { alpha.data(), 1, width, width, height };
		return &plane->second;
	};

	// Each micromap before compaction: its level and offset, and the triangle and plane it comes from.
	// A blended surface's is cut twice: plane where nothing of it is there, opaquePlane (null if it never
	// is) where all of it is, see CombineBlendMicromap. It never reads them in 2-state mode
	// (MaterialGeometryFlags::opacityMicromap2State).
	struct Entry
	{
		float uv[3][2];
		const AlphaPlane *plane;
		const AlphaPlane *opaquePlane;
		bool blended;
	};
	std::vector<uint8_t> levels;
	std::vector<uint32_t> offsets;
	std::vector<Entry> entries;
	size_t dataSize = 0;
	uint32_t ommPrimitives = 0;
	uint32_t uniformPrimitives = 0;
	uint32_t blendedPrimitives = 0;

	for (size_t p = 0; p < primitiveDescs.size(); ++p)
	{
		const PrimitiveGPUDesc &desc = primitiveDescs[p];
		if (desc.indexCount == 0)
		{
			continue;
		}
		const MaterialData material = UnpackGpuMaterial(materials[desc.materialIndex]);
		// Pass-through surfaces always need the shadow any-hit shader for what they let through.
		const bool blended = material.alphaMode == ALPHA_MODE_BLEND;
		if ((material.alphaMode != ALPHA_MODE_MASK && !blended) || (material.transmissionFlags & TRANSMISSION_PASS_THROUGH) != 0)
		{
			continue;
		}

		// The any-hit shaders cut where baseColour.w * texture alpha < the cutoff.
		const Scene::Texture *texture = desc.materialIndex < scene.GetMaterialCount() &&
												scene.GetMaterial(desc.materialIndex).GetAlbedoTexture().IsValid()
										? &scene.GetMaterial(desc.materialIndex).GetAlbedoTexture()
										: nullptr;
		const float factor = material.baseColour.w;
		const float cutoff = blended ? kBlendMinOpacity : material.alphaCutoff;
		const uint32_t triangleCount = desc.indexCount / 3;

		const bool neverCut = cutoff <= 0.0f;
		const bool alwaysCut = !neverCut && (texture == nullptr ? factor < cutoff : !(factor >= cutoff));
		// A blended surface of one alpha that is partly there needs the any-hit shaders everywhere.
		if (blended && texture == nullptr && !alwaysCut && factor < 1.0f)
		{
			continue;
		}
		ommIndexOffsets_[p] = static_cast<uint32_t>(ommIndices_.size());
		blendedPrimitives += blended ? 1 : 0;
		if (neverCut || alwaysCut || texture == nullptr)
		{
			// One alpha over the whole primitive.
			const bool opaque = neverCut || (texture == nullptr && !alwaysCut);
			ommIndices_.insert(ommIndices_.end(),
							   triangleCount,
							   opaque ? OpacityMicromapFullyOpaque : OpacityMicromapFullyTransparent);
			++uniformPrimitives;
			continue;
		}

		const AlphaPlane *plane = planeFor(desc.materialIndex, *texture, cutoff / factor);
		const AlphaPlane *opaquePlane =
				blended && factor >= 1.0f ? planeFor(desc.materialIndex, *texture, 1.0f / factor) : nullptr;

		// One micromap per distinct UV triangle, at a level for its texel area.
		const GpuVertex *vertices = &vertices_[desc.vertexOffset];
		const uint32_t *indices = &indices_[desc.indexOffset];
		std::vector<uint8_t> primitiveLevels(triangleCount);
		std::vector<uint32_t> sources(triangleCount);
		std::vector<int> primitiveIndices(triangleCount);
		const size_t count = meshopt_opacityMapMeasure(primitiveLevels.data(),
													   sources.data(),
													   primitiveIndices.data(),
													   indices,
													   desc.indexCount,
													   &vertices[0].texCoord.x,
													   desc.vertexCount,
													   sizeof(GpuVertex),
													   plane->width,
													   plane->height,
													   kOpacityMicromapMaxLevel,
													   kOpacityMicromapTargetEdge);

		const int32_t base = static_cast<int32_t>(levels.size());
		for (size_t e = 0; e < count; ++e)
		{
			levels.push_back(primitiveLevels[e]);
			offsets.push_back(static_cast<uint32_t>(dataSize));
			dataSize += meshopt_opacityMapEntrySize(primitiveLevels[e], kOpacityMicromapStates);
			Entry &entry = entries.emplace_back();
			for (int c = 0; c < 3; ++c)
			{
				const float2 &uv = vertices[indices[3 * sources[e] + c]].texCoord;
				entry.uv[c][0] = uv.x;
				entry.uv[c][1] = uv.y;
			}
			entry.plane = plane;
			entry.opaquePlane = opaquePlane;
			entry.blended = blended;
		}
		for (int index: primitiveIndices)
		{
			ommIndices_.push_back(base + index);
		}
		++ommPrimitives;
	}

	if (ommIndices_.empty())
	{
		return;
	}

	std::vector<uint8_t> data(dataSize);
	std::vector<uint32_t> order(entries.size());
	std::iota(order.begin(), order.end(), 0u);
	std::for_each(std::execution::par,
				  order.begin(),
				  order.end(),
				  [&](uint32_t e)
				  {
					  const Entry &entry = entries[e];
					  meshopt_opacityMapRasterize(&data[offsets[e]],
												  levels[e],
												  kOpacityMicromapStates,
												  entry.uv[0],
												  entry.uv[1],
												  entry.uv[2],
												  entry.plane->data,
												  entry.plane->stride,
												  entry.plane->pitch,
												  entry.plane->width,
												  entry.plane->height);
					  if (!entry.blended)
					  {
						  return;
					  }
					  const size_t size = meshopt_opacityMapEntrySize(levels[e], kOpacityMicromapStates);
					  std::vector<uint8_t> opaque;
					  if (entry.opaquePlane != nullptr)
					  {
						  opaque.resize(size);
						  meshopt_opacityMapRasterize(opaque.data(),
													  levels[e],
													  kOpacityMicromapStates,
													  entry.uv[0],
													  entry.uv[1],
													  entry.uv[2],
													  entry.opaquePlane->data,
													  entry.opaquePlane->stride,
													  entry.opaquePlane->pitch,
													  entry.opaquePlane->width,
													  entry.opaquePlane->height);
					  }
					  CombineBlendMicromap(&data[offsets[e]], opaque.empty() ? nullptr : opaque.data(), size);
				  });

	// Across the whole scene: one copy of each distinct micromap, and special indices for those in one state.
	const size_t rasterised = levels.size();
	static_assert(sizeof(int) == sizeof(int32_t));
	const size_t count = meshopt_opacityMapCompact(data.data(),
												   data.size(),
												   levels.data(),
												   offsets.data(),
												   levels.size(),
												   reinterpret_cast<int *>(ommIndices_.data()),
												   ommIndices_.size(),
												   kOpacityMicromapStates);
	data.resize(count == 0 ? 0 : offsets[count - 1] + meshopt_opacityMapEntrySize(levels[count - 1], kOpacityMicromapStates));
	ommData_ = std::move(data);
	ommMicromaps_.resize(count);
	for (size_t i = 0; i < count; ++i)
	{
		ommMicromaps_[i] = { offsets[i], levels[i], static_cast<uint16_t>(kOpacityMicromapStates) };
	}

	const size_t special = std::count_if(ommIndices_.begin(), ommIndices_.end(), [](int32_t i) { return i < 0; });
	logi("GPUScene: opacity micromaps for {} alpha-tested primitive(s) ({} of one alpha, {} blended): {} triangles, "
		 "{:.0f}% in one state, {} micromaps ({} before compaction), {:.2f} MB, in {:.0f} ms",
		 ommPrimitives + uniformPrimitives,
		 uniformPrimitives,
		 blendedPrimitives,
		 ommIndices_.size(),
		 100.0 * static_cast<double>(special) / static_cast<double>(ommIndices_.size()),
		 count,
		 rasterised,
		 ommData_.size() / (1024.0 * 1024.0),
		 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

void RenderGPUScene::UploadOpacityMicromapIndices(UploadBatchHandle batch)
{
	if (!ommIndices_.empty())
	{
		ommIndexBuffer_ = UploadStream(ommIndices_.data(),
									   ommIndices_.size() * sizeof(int32_t),
									   sizeof(int32_t),
									   "Scene_OpacityMicromapIndices",
									   batch);
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

	const ResourceHandle indexBuffer = streamBuffers_[(uint32_t) GPUStream::Index];

	// The micromaps are built first, as the BLASes that link them read them.
	const bool micromaps = renderContext_->SupportsOpacityMicromaps() && ommIndexBuffer_ != InvalidResourceHandle;
	if (micromaps && !ommMicromaps_.empty())
	{
		ommArray_ = renderContext_->CreateOpacityMicromapArray(ommData_, ommMicromaps_);
	}
	std::vector<uint8_t>().swap(ommData_);
	std::vector<OpacityMicromap>().swap(ommMicromaps_);

	blases_.assign(primitiveDescs.size(), InvalidAccelerationStructureHandle);
	ommBlases_.assign(primitiveDescs.size(), InvalidAccelerationStructureHandle);
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

		if (micromaps && ommIndexOffsets_[i] != kNoOpacityMicromaps)
		{
			geometry.opacityMicromapArray = ommArray_;
			geometry.opacityMicromapIndexBuffer = ommIndexBuffer_;
			geometry.opacityMicromapIndexBufferOffset = static_cast<uint64_t>(ommIndexOffsets_[i]) * sizeof(int32_t);
			ommBlases_[i] = renderContext_->CreateBLAS(geometry);
		}
	}

	// Built and compacted in submissions of their own, which leaves them none of the build's memory.
	std::vector<AccelerationStructureHandle> builtBlases;
	for (const std::vector<AccelerationStructureHandle> *list: { &blases_, &ommBlases_ })
	{
		std::copy_if(list->begin(),
					 list->end(),
					 std::back_inserter(builtBlases),
					 [](AccelerationStructureHandle h) { return h != InvalidAccelerationStructureHandle; });
	}
	renderContext_->BuildCompactedBLASes(builtBlases);

	// One TLAS instance per (node, primitive). InstanceID is the primitive's row in the
	// PrimitiveGPUDesc table, which is all the hit shader needs to find its data. The order must
	// match the emitter buffer, which is indexed by InstanceIndex().
	std::vector<TLASInstance> tlasInstances;
	std::vector<TLASInstance> ommTlasInstances;
	std::vector<TLASInstance> omm2StateTlasInstances;
	bool anyMicromaps = false;
	bool anyMaskMicromaps = false;
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

		if (ommBlases_[ref.primitive] != InvalidAccelerationStructureHandle)
		{
			tlasInstance.blas = ommBlases_[ref.primitive];
			anyMicromaps = true;
		}
		ommTlasInstances.push_back(tlasInstance);

		if (ommBlases_[ref.primitive] != InvalidAccelerationStructureHandle &&
			materialGeometry_[primitiveDescs[ref.primitive].materialIndex].opacityMicromap2State)
		{
			tlasInstance.flags = TLASInstanceFlagForceOpacityMicromap2State;
			anyMaskMicromaps = true;
		}
		omm2StateTlasInstances.push_back(tlasInstance);
	}

	if (tlasInstances.empty())
	{
		logw("GPUScene: no instanced geometry, skipping TLAS build");
		return;
	}

	tlas_ = renderContext_->CreateTLAS(tlasInstances);
	renderContext_->BuildAccelerationStructure(tlas_);
	if (anyMicromaps)
	{
		ommTlas_ = renderContext_->CreateTLAS(ommTlasInstances);
		renderContext_->BuildAccelerationStructure(ommTlas_);
	}
	// Without a mask's micromaps 2-state mode changes nothing, and the micromap TLAS serves for it too.
	if (anyMaskMicromaps)
	{
		omm2StateTlas_ = renderContext_->CreateTLAS(omm2StateTlasInstances);
		renderContext_->BuildAccelerationStructure(omm2StateTlas_);
	}
	tlasRefs_ = EnumerateTLASInstances();
	tlasInstances_ = std::move(tlasInstances);
	ommTlasInstances_ = anyMicromaps ? std::move(ommTlasInstances) : std::vector<TLASInstance>();
	omm2StateTlasInstances_ = anyMaskMicromaps ? std::move(omm2StateTlasInstances) : std::vector<TLASInstance>();
	renderContext_->UAVBarrierForAccelerationStructures();
	tlasSRV_ = renderContext_->GetAccelerationStructureSRV(tlas_);
	if (anyMicromaps)
	{
		ommTlasSRV_ = renderContext_->GetAccelerationStructureSRV(ommTlas_);
	}
	if (anyMaskMicromaps)
	{
		omm2StateTlasSRV_ = renderContext_->GetAccelerationStructureSRV(omm2StateTlas_);
	}

	logi("GPUScene: built {} BLAS(es) ({} with opacity micromaps) and {} TLAS(es) with {} instance(s)",
		 builtBlases.size(),
		 std::count_if(ommBlases_.begin(),
					   ommBlases_.end(),
					   [](AccelerationStructureHandle h) { return h != InvalidAccelerationStructureHandle; }),
		 1 + (anyMicromaps ? 1 : 0) + (anyMaskMicromaps ? 1 : 0),
		 tlasInstances_.size());
}

void RenderGPUScene::UpdateAccelerationStructures()
{
	if (!dynamicTLASMoved_ || tlas_ == InvalidAccelerationStructureHandle)
	{
		return;
	}
	dynamicTLASMoved_ = false;

	for (size_t i = 0; i < tlasRefs_.size(); ++i)
	{
		const uint32_t instance = tlasRefs_[i].instance;
		if (dynamicOfInstance_[instance] == UINT32_MAX)
		{
			continue;
		}
		float transform[12];
		Scene::RowMajorToBLASTransform(instances[instance].worldMatrix, transform);
		std::memcpy(tlasInstances_[i].transform, transform, sizeof(tlasInstances_[i].transform));
		if (!ommTlasInstances_.empty())
		{
			std::memcpy(ommTlasInstances_[i].transform, transform, sizeof(ommTlasInstances_[i].transform));
		}
		if (!omm2StateTlasInstances_.empty())
		{
			std::memcpy(omm2StateTlasInstances_[i].transform, transform, sizeof(omm2StateTlasInstances_[i].transform));
		}
	}

	renderContext_->UpdateTLAS(tlas_, tlasInstances_);
	if (ommTlas_ != InvalidAccelerationStructureHandle)
	{
		renderContext_->UpdateTLAS(ommTlas_, ommTlasInstances_);
	}
	if (omm2StateTlas_ != InvalidAccelerationStructureHandle)
	{
		renderContext_->UpdateTLAS(omm2StateTlas_, omm2StateTlasInstances_);
	}
	renderContext_->UAVBarrierForAccelerationStructures();
}
