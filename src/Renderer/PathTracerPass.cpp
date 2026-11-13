#include "PathTracerPass.h"

#include "Camera.h"
#include "RenderSettings.h"
#include "Scene/RenderGPUScene.h"
#include "SkyAtmospherePass.h"

#include "DLSSUtil.h"

#include "fmtlog.h"

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <math.h>
#include <stdexcept>

#undef min
#undef max

namespace
{
	// HitPayload in PathTracer/RayTracing.slang, which carries the hit's motion under DLSS_RR.
	constexpr uint32_t kMaxPayloadSizeInBytes = 48;
	constexpr uint32_t kMaxPayloadSizeInBytesRR = 56;

	// Halton phases before the jitter repeats. The DLSS guide asks for 8 * (output / render)^2, so
	// 72 at Ultra Performance, and the RR guide for at least 32 with no reason to limit it.
	constexpr uint32_t kJitterPhaseCount = 128;

	static_assert(sizeof(GpuPhotonRecord) == 48, "GpuPhotonRecord must match the HLSL structured buffer layout");
	static_assert(PHOTON_MAP_BLOCKS == PHOTON_MAP_BLOCK, "ScanPhotonTotals scans the blocks in one group of a block's size");
} // namespace

PathTracerPass::PathTracerPass(RenderContext *renderContext,
							   const RenderGPUScene &scene,
							   const Camera &camera,
							   const RenderSettings &settings,
							   const SkyAtmospherePass &atmosphere,
							   bool rayReconstruction,
							   bool reference) :
	renderContext_(renderContext), scene_(scene), camera_(camera), settings_(settings), atmosphere_(atmosphere),
	rayReconstruction_(rayReconstruction), reference_(reference)
{
	serSupported_ = renderContext->GetShaderExecutionReorderingSupport().supported;

	// Build the pipeline the settings start with now rather than on the first frame.
	SelectPipeline();

	width_ = renderContext->GetRenderWidth();
	height_ = renderContext->GetRenderHeight();

	output_ = CreateTarget(Format::R16G16B16A16_FLOAT, "PathTracer_Radiance");
	if (reference_)
	{
		accumulation_ = CreateTarget(Format::R32G32B32A32_FLOAT, "PathTracer_Accumulation");
	}

	if (rayReconstruction_)
	{
		guides_.diffuseAlbedo = CreateTarget(Format::R8G8B8A8_UNORM, "PathTracer_DiffuseAlbedo");
		guides_.specularAlbedo = CreateTarget(Format::R16G16B16A16_FLOAT, "PathTracer_SpecularAlbedo");
		guides_.normalRoughness = CreateTarget(Format::R16G16B16A16_FLOAT, "PathTracer_NormalRoughness");
		guides_.motionVectors = CreateTarget(Format::R16G16_FLOAT, "PathTracer_MotionVectors");
		guides_.depth = CreateTarget(Format::R32_FLOAT, "PathTracer_Depth");
		guides_.specularHitDistance = CreateTarget(Format::R16_FLOAT, "PathTracer_SpecularHitDistance");
		guides_.specularMotionVectors = CreateTarget(Format::R16G16_FLOAT, "PathTracer_SpecularMotionVectors");
		guides_.shadowChanges = CreateTarget(Format::R8_UNORM, "PathTracer_ShadowChanges");
		guides_.responsivity = CreateTarget(Format::R16_FLOAT, "PathTracer_Responsivity");
		guides_.psrOutcome = CreateTarget(Format::R16_FLOAT, "PathTracer_PsrOutcome");

		BufferDesc cameraStateDesc = {};
		cameraStateDesc.sizeInBytes = static_cast<uint64_t>(width_) * height_ * sizeof(GpuCameraState);
		cameraStateDesc.structureByteStride = sizeof(GpuCameraState);
		cameraStateDesc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		cameraStateDesc.debugName = "PathTracer_CameraState";
		cameraState_ = renderContext->CreateBuffer(cameraStateDesc, ResourceViewType::UnorderedAccess);
	}

	BufferDesc deferredBranchDesc = {};
	deferredBranchDesc.sizeInBytes = static_cast<uint64_t>(width_) * height_ * sizeof(GpuDeferredBranch);
	deferredBranchDesc.structureByteStride = sizeof(GpuDeferredBranch);
	deferredBranchDesc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	deferredBranchDesc.debugName = "PathTracer_DeferredBranches";
	deferredBranches_ = renderContext->CreateBuffer(deferredBranchDesc, ResourceViewType::UnorderedAccess);

	frameConstants_.resize(renderContext->GetFramesInFlight());
	for (ResourceHandle &constants: frameConstants_)
	{
		constants = renderContext->CreateConstantBuffer(sizeof(FrameConstants), "PathTracer_FrameConstants");
	}

	const char *photonEntries[PhotonPassCount] = { "UpdatePhotonMap", "ScanPhotonBlocks", "ScanPhotonTotals", "SplatPhotons",
												   "CompositeCaustics" };
	for (uint32_t pass = 0; pass < PhotonPassCount; ++pass)
	{
		photonPipelines_[pass] = renderContext->CreateComputePipeline(PATHTRACER_SHADER_DIR "PhotonMap.slang", photonEntries[pass]);
		if (photonPipelines_[pass] == InvalidPipelineHandle)
		{
			throw std::runtime_error("Failed to create the photon map pipelines (see the shader log above).");
		}
	}
	BufferDesc rectDesc = {};
	rectDesc.sizeInBytes = PHOTON_MAX_RECTS * sizeof(GpuPhotonRect);
	rectDesc.structureByteStride = sizeof(GpuPhotonRect);
	rectDesc.debugName = "PathTracer_PhotonRects";
	photonRects_.resize(renderContext->GetFramesInFlight());
	for (ResourceHandle &rects: photonRects_)
	{
		rects = renderContext->CreateUploadBuffer(rectDesc, ResourceViewType::ShaderResource);
	}
}

PathTracerPass::~PathTracerPass()
{
	for (ResourceHandle target: { output_,
								  guides_.diffuseAlbedo,
								  guides_.specularAlbedo,
								  guides_.normalRoughness,
								  guides_.motionVectors,
								  guides_.depth,
								  guides_.specularHitDistance,
								  guides_.specularMotionVectors,
								  guides_.shadowChanges,
								  guides_.responsivity,
								  guides_.psrOutcome,
								  cameraState_,
								  deferredBranches_,
								  accumulation_,
								  photonDensity_[0],
								  photonDensity_[1],
								  photonTexelPrefix_,
								  photonBlockPrefix_,
								  photonFeedback_,
								  photonRecords_,
								  photonRecordCount_,
								  photonReceivers_,
								  photonCaustics_,
								  photonHistory_[0],
								  photonHistory_[1] })
	{
		if (target != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(target);
		}
	}
	for (ResourceHandle constants: frameConstants_)
	{
		renderContext_->DestroyResource(constants);
	}
	for (ResourceHandle rects: photonRects_)
	{
		renderContext_->DestroyResource(rects);
	}
}

const PathTracerPass::Pipeline *PathTracerPass::SelectPipeline()
{
	const bool atmosphere = atmosphere_.IsEnabled();
	const bool inlineShadows = settings_.inlineShadowRays;
	// The guides are only traced for ray reconstruction.
	const bool psr = rayReconstruction_ && settings_.primarySurfaceReplacement;
	const bool sgLightTree = settings_.lights.sgImportance;

	// A failed opacity micromap or SER compile is logged once, and the pipeline without them takes over.
	if (settings_.opacityMicromaps && scene_.HasOpacityMicromaps() && !ommFailed_)
	{
		const bool ser = settings_.shaderExecutionReordering && serSupported_ && !serFailed_;
		if (const Pipeline *pipeline = GetPipeline(ser, atmosphere, inlineShadows, true, psr, sgLightTree))
		{
			return pipeline;
		}
		loge("Failed to create the opacity micromap pipeline, tracing without them");
		ommFailed_ = true;
	}

	if (settings_.shaderExecutionReordering && serSupported_ && !serFailed_)
	{
		if (const Pipeline *pipeline = GetPipeline(true, atmosphere, inlineShadows, false, psr, sgLightTree))
		{
			return pipeline;
		}
		loge("Failed to create the shader execution reordering pipeline, falling back to TraceRay");
		serFailed_ = true;
	}

	const Pipeline *pipeline = GetPipeline(false, atmosphere, inlineShadows, false, psr, sgLightTree);
	if (pipeline == nullptr)
	{
		throw std::runtime_error("Failed to create the path tracing pipeline (see the shader log above).");
	}
	return pipeline;
}

const PathTracerPass::Pipeline *PathTracerPass::GetPipeline(bool shaderExecutionReordering, bool atmosphere,
															bool inlineShadows, bool opacityMicromaps, bool psr,
															bool sgLightTree)
{
	Pipeline &pipeline = pipelines_[shaderExecutionReordering ? 1 : 0][atmosphere ? 1 : 0][inlineShadows ? 1 : 0]
								   [opacityMicromaps ? 1 : 0][psr ? 1 : 0][sgLightTree ? 1 : 0];
	if (pipeline.pipeline == InvalidPipelineHandle)
	{
		pipeline = CreatePipeline(shaderExecutionReordering, atmosphere, inlineShadows, opacityMicromaps, psr, sgLightTree);
	}
	return pipeline.pipeline != InvalidPipelineHandle ? &pipeline : nullptr;
}

PathTracerPass::Pipeline PathTracerPass::CreatePipeline(bool shaderExecutionReordering, bool atmosphere,
														 bool inlineShadows, bool opacityMicromaps, bool psr,
														 bool sgLightTree) const
{
	RaytracingPipelineDesc desc = {};
	desc.exports = { "RayGen", "PhotonRayGen", "Miss", "ClosestHit", "AnyHit" };
	desc.hitGroups = { RaytracingHitGroup{ "HitGroup", "ClosestHit", "AnyHit", "" } };
	// Inline shadow rays are ray queries, so that permutation has no shadow shaders.
	if (!inlineShadows)
	{
		desc.exports.insert(desc.exports.end(), { "ShadowMiss", "ShadowClosestHit", "ShadowAnyHit" });
		desc.hitGroups.push_back(RaytracingHitGroup{ "ShadowHitGroup", "ShadowClosestHit", "ShadowAnyHit", "" });
	}
	desc.maxPayloadSizeInBytes = rayReconstruction_ ? kMaxPayloadSizeInBytesRR : kMaxPayloadSizeInBytes;
	desc.maxAttributeSizeInBytes = 8; // triangle barycentrics
	desc.maxTraceRecursionDepth = 1; // primary and shadow rays are both traced from raygen
	if (rayReconstruction_)
	{
		desc.defines.push_back("DLSS_RR");
	}
	if (psr)
	{
		desc.defines.push_back("PSR");
	}
	if (reference_)
	{
		desc.defines.push_back("REFERENCE");
	}
	if (atmosphere)
	{
		desc.defines.push_back("ATMOSPHERE");
	}
	if (inlineShadows)
	{
		desc.defines.push_back("INLINE_SHADOWS");
	}
	if (sgLightTree)
	{
		desc.defines.push_back("SG_LIGHT_TREE");
	}
	if (shaderExecutionReordering)
	{
		// HitObject and MaybeReorderThread are Shader Model 6.9.
		desc.defines.push_back("SER");
		desc.profile = "lib_6_9";
	}
	if (opacityMicromaps)
	{
		// So are the ray query flags that let the inline rays meet micromaps.
		desc.defines.push_back("OPACITY_MICROMAPS");
		desc.profile = "lib_6_9";
		desc.allowOpacityMicromaps = true;
	}

	Pipeline result;
	result.opacityMicromaps = opacityMicromaps;
	result.pipeline = renderContext_->CreateRaytracingPipeline(PATHTRACER_SHADER_DIR "PathTracer.slang", desc);
	if (result.pipeline == InvalidPipelineHandle)
	{
		return result;
	}

	// The paths' table and the photons', which differ only in their ray generation shader.
	auto createTable = [&](const char *rayGen)
	{
		ShaderTableHandle table = renderContext_->CreateShaderTable(result.pipeline);
		renderContext_->AddRayGenShaderRecord(table, rayGen);
		// Record order matches STANDARD_RAY_INDEX and SHADOW_RAY_INDEX in PathTracer/RayTracing.slang.
		renderContext_->AddMissShaderRecord(table, "Miss");
		renderContext_->AddHitGroupShaderRecord(table, "HitGroup");
		if (!inlineShadows)
		{
			renderContext_->AddMissShaderRecord(table, "ShadowMiss");
			renderContext_->AddHitGroupShaderRecord(table, "ShadowHitGroup");
		}
		renderContext_->BuildShaderTable(table);
		return table;
	};
	result.shaderTable = createTable("RayGen");
	result.photonShaderTable = createTable("PhotonRayGen");
	return result;
}

void PathTracerPass::PreparePhotons(FrameConstants &constants, bool sunActive, float sunAngularRadius)
{
	const PhotonSettings &settings = settings_.photons;
	constants.photons = {};
	// The reference converges to the path tracer's own caustics.
	if (!settings.enabled || !sunActive || reference_)
	{
		return;
	}
	const float maxRoughness = std::clamp(settings.maxRoughness, 0.0f, 1.0f);
	scene_.CollectPhotonCasters(maxRoughness, photonCasters_);
	if (photonCasters_.empty())
	{
		return;
	}

	// The plane perpendicular to the sun, photons starting on it above the whole scene so that nothing is skipped.
	const float3 s = constants.sunDirection;
	const float3 helper = fabsf(s.y) < 0.99f ? float3{ 0.0f, 1.0f, 0.0f } : float3{ 1.0f, 0.0f, 0.0f };
	float3 u = Cross(helper, s);
	NormaliseFloat3(u);
	const float3 v = Cross(s, u);
	const float3 sceneLo = scene_.GetBoundsMin();
	const float3 sceneHi = scene_.GetBoundsMax();
	const float3 sceneExtent = sceneHi - sceneLo;
	const float sceneDiagonal = std::max(sqrtf(Dot(sceneExtent, sceneExtent)), 1.0e-3f);
	float planeHeight = -FLT_MAX;
	for (uint32_t corner = 0; corner < 8; ++corner)
	{
		const float3 c = { (corner & 1) != 0 ? sceneHi.x : sceneLo.x,
						   (corner & 2) != 0 ? sceneHi.y : sceneLo.y,
						   (corner & 4) != 0 ? sceneHi.z : sceneLo.z };
		planeHeight = std::max(planeHeight, Dot(c, s));
	}
	planeHeight += 0.01f * sceneDiagonal;
	// Directions across the sun disc drift sideways on the way down, which the rectangles widen for.
	const float spread = tanf(std::min(sunAngularRadius, 0.5f));

	// One rectangle per caster, or one around them all past PHOTON_MAX_RECTS.
	photonRectData_.clear();
	auto addRect = [&](const float3 &lo, const float3 &hi)
	{
		float2 a = { FLT_MAX, FLT_MAX };
		float2 b = { -FLT_MAX, -FLT_MAX };
		float lowest = FLT_MAX;
		for (uint32_t corner = 0; corner < 8; ++corner)
		{
			const float3 c = { (corner & 1) != 0 ? hi.x : lo.x, (corner & 2) != 0 ? hi.y : lo.y, (corner & 4) != 0 ? hi.z : lo.z };
			const float pu = Dot(c, u);
			const float pv = Dot(c, v);
			a = { std::min(a.x, pu), std::min(a.y, pv) };
			b = { std::max(b.x, pu), std::max(b.y, pv) };
			lowest = std::min(lowest, Dot(c, s));
		}
		const float margin = (planeHeight - lowest) * spread + 1.0e-4f * sceneDiagonal;
		GpuPhotonRect rect = {};
		rect.lo = { a.x - margin, a.y - margin };
		rect.hi = { b.x + margin, b.y + margin };
		photonRectData_.push_back(rect);
	};
	if (photonCasters_.size() <= PHOTON_MAX_RECTS)
	{
		for (const auto &[lo, hi]: photonCasters_)
		{
			addRect(lo, hi);
		}
	}
	else
	{
		float3 lo = photonCasters_[0].first;
		float3 hi = photonCasters_[0].second;
		for (const auto &[casterLo, casterHi]: photonCasters_)
		{
			lo = { std::min(lo.x, casterLo.x), std::min(lo.y, casterLo.y), std::min(lo.z, casterLo.z) };
			hi = { std::max(hi.x, casterHi.x), std::max(hi.y, casterHi.y), std::max(hi.z, casterHi.z) };
		}
		addRect(lo, hi);
	}
	float2 lo = { FLT_MAX, FLT_MAX };
	float2 hi = { -FLT_MAX, -FLT_MAX };
	for (const GpuPhotonRect &rect: photonRectData_)
	{
		lo = { std::min(lo.x, rect.lo.x), std::min(lo.y, rect.lo.y) };
		hi = { std::max(hi.x, rect.hi.x), std::max(hi.y, rect.hi.y) };
	}
	if (!(hi.x > lo.x) || !(hi.y > lo.y))
	{
		return;
	}

	// The photon map: a square over the rectangles, its side a power of two and its corner on a grid of an eighth of
	// that, so that casters moving a little leave it where it is (moving it starts the map over).
	const float extent = std::max(hi.x - lo.x, hi.y - lo.y);
	float mapScale = exp2f(ceilf(log2f(extent * 1.0625f)));
	float2 mapOrigin = { 0.0f, 0.0f };
	for (int attempt = 0; attempt < 8; ++attempt)
	{
		const float step = mapScale / 8.0f;
		mapOrigin = { floorf((0.5f * (lo.x + hi.x) - 0.5f * mapScale) / step) * step,
					  floorf((0.5f * (lo.y + hi.y) - 0.5f * mapScale) / step) * step };
		if (mapOrigin.x + mapScale >= hi.x && mapOrigin.y + mapScale >= hi.y)
		{
			break;
		}
		mapScale *= 2.0f;
	}

	const ResourceHandle rects = photonRects_[renderContext_->GetCurrentFrameIndex()];
	renderContext_->UpdateUploadBuffer(rects, photonRectData_.data(), 0, photonRectData_.size() * sizeof(GpuPhotonRect));

	// The buffers, on first use.
	if (photonRecords_ == InvalidResourceHandle)
	{
		// The sums the passes add to start at zero, and the shares and running sums read as empty: those buffers are
		// created from zeros, copied in on the copy queue now. (ClearUnorderedAccessViewUint, which they used to be cleared
		// with on their first frame, read a stale shader-visible descriptor: debug layer #646, then a GPU hang in release.)
		const uint64_t pixels = static_cast<uint64_t>(width_) * height_;
		std::vector<uint8_t> zeros(std::max<uint64_t>(static_cast<uint64_t>(PHOTON_MAP_TEXELS) * 4 * sizeof(uint32_t),
													  pixels * 4 * sizeof(uint32_t)),
								   0);
		const UploadBatchHandle batch = renderContext_->BeginUpload();
		auto create = [&](uint64_t count, uint32_t stride, const char *debugName, bool zeroed)
		{
			BufferDesc desc = {};
			desc.sizeInBytes = count * stride;
			desc.structureByteStride = stride;
			desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
			desc.debugName = debugName;
			return zeroed ? renderContext_->CreateBuffer(desc, ResourceViewType::UnorderedAccess, zeros.data(), batch)
						  : renderContext_->CreateBuffer(desc, ResourceViewType::UnorderedAccess);
		};
		photonDensity_[0] = create(PHOTON_MAP_TEXELS, sizeof(float), "PathTracer_PhotonDensity0", true);
		photonDensity_[1] = create(PHOTON_MAP_TEXELS, sizeof(float), "PathTracer_PhotonDensity1", true);
		photonTexelPrefix_ = create(PHOTON_MAP_TEXELS, sizeof(float), "PathTracer_PhotonTexelPrefix", false);
		photonBlockPrefix_ = create(PHOTON_MAP_BLOCKS + 1, sizeof(float), "PathTracer_PhotonBlockPrefix", true);
		photonFeedback_ = create(PHOTON_MAP_TEXELS, 4 * sizeof(uint32_t), "PathTracer_PhotonFeedback", true);
		photonRecords_ = create(PHOTON_RECORD_CAPACITY, sizeof(GpuPhotonRecord), "PathTracer_PhotonRecords", false);
		photonRecordCount_ = create(4, sizeof(uint32_t), "PathTracer_PhotonRecordCount", true);
		photonReceivers_ = create(2 * pixels, 4 * sizeof(float), "PathTracer_PhotonReceivers", false);
		photonCaustics_ = create(pixels, 4 * sizeof(uint32_t), "PathTracer_PhotonCaustics", true);
		photonHistory_[0] = create(pixels, 2 * sizeof(float), "PathTracer_PhotonHistory0", true);
		photonHistory_[1] = create(pixels, 2 * sizeof(float), "PathTracer_PhotonHistory1", true);
		renderContext_->EndUpload(batch);
		photonMapValid_ = false;
	}

	// The map starts over when its plane or square moves, when the UI asks, or after a frame without photons; without
	// adaptation every frame starts from even shares.
	const bool moved = Dot(s, photonSunDirection_) < 0.999999f || mapScale != photonMapScale_ ||
					   mapOrigin.x != photonMapOrigin_.x || mapOrigin.y != photonMapOrigin_.y;
	photonReset_ = !photonMapValid_ || moved || photonResetCount_ != settings.resetCount ||
				   photonAdaptive_ != settings.adaptive || !settings.adaptive;
	photonSunDirection_ = s;
	photonMapScale_ = mapScale;
	photonMapOrigin_ = mapOrigin;
	photonResetCount_ = settings.resetCount;
	photonAdaptive_ = settings.adaptive;
	photonParity_ ^= 1u;

	auto index = [&](ResourceHandle handle) { return renderContext_->GetResourceViewInfo(handle).descriptorIndex; };
	const float3 irradiance = settings_.sun.colour * settings_.sun.intensity;
	const float peak = std::max({ irradiance.x, irradiance.y, irradiance.z });
	PhotonParameters &photons = constants.photons;
	photons.basisU = u;
	photons.basisV = v;
	photons.planeHeight = planeHeight;
	photons.mapScale = mapScale;
	photons.mapOrigin = mapOrigin;
	photons.enabled = 1;
	photons.photonCount = static_cast<uint>(std::clamp(settings.photonsPerFrame, 1, 1 << 24));
	photons.rects = index(rects);
	photons.rectCount = static_cast<uint>(photonRectData_.size());
	photons.maxBounces = static_cast<uint>(std::clamp(settings.maxBounces, 1, 64));
	photons.maxAlpha = maxRoughness * maxRoughness;
	photons.density = index(photonDensity_[photonParity_]);
	photons.blockPrefix = index(photonBlockPrefix_);
	photons.texelPrefix = index(photonTexelPrefix_);
	photons.records = index(photonRecords_);
	photons.recordCount = index(photonRecordCount_);
	photons.feedback = index(photonFeedback_);
	photons.receivers = index(photonReceivers_);
	photons.width = width_;
	photons.height = height_;
	photons.footprintScale = std::max(settings.footprintScale, 0.01f);
	// A uniform disc of angular radius r spreads its directions by r / 2 each way.
	photons.sunSpread = constants.sunIsDelta != 0 ? 0.0f : 0.5f * sunAngularRadius;
	photons.irradianceUnit = std::max(peak, 1.0e-30f) / PHOTON_IRRADIANCE_SCALE;
}

PhotonPassConstants PathTracerPass::PhotonPassConstantsForFrame() const
{
	const PhotonSettings &settings = settings_.photons;
	auto index = [&](ResourceHandle handle) { return renderContext_->GetResourceViewInfo(handle).descriptorIndex; };
	PhotonPassConstants constants = {};
	constants.densityPrevious = index(photonDensity_[photonParity_ ^ 1u]);
	constants.historyPrevious = index(photonHistory_[photonParity_ ^ 1u]);
	constants.history = index(photonHistory_[photonParity_]);
	constants.caustics = index(photonCaustics_);
	constants.output = index(output_);
	constants.reset = photonReset_ ? 1u : 0u;
	constants.targetArea = std::max(settings.targetArea, 0.01f);
	constants.varianceGain = std::max(settings.varianceGain, 0.0f);
	constants.temporalWeight = std::clamp(settings.temporalWeight, 0.0f, 1.0f);
	constants.minDensity = std::max(settings.minDensity, 1.0e-3f);
	constants.maxDensity = std::max(settings.maxDensity, constants.minDensity);
	constants.frameIndex = renderContext_->GetCurrentFrame();
	return constants;
}

void PathTracerPass::DispatchPhotonMap(RenderContext *renderContext, ResourceHandle constantBuffer)
{
	const PhotonPassConstants constants = PhotonPassConstantsForFrame();
	// The scans read only the frame constants, so their root signatures have no root constants.
	auto bind = [&](PhotonPass pass)
	{
		renderContext->SetComputePipeline(photonPipelines_[pass]);
		if (pass != PhotonPassScanBlocks && pass != PhotonPassScanTotals)
		{
			renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
		}
		renderContext->SetComputeConstantBufferByName("g_Frame", constantBuffer);
	};
	auto computeBarrier = [&]()
	{
		renderContext->GlobalBarrier(ResourceSync::ComputeShading,
									 ResourceAccess::UnorderedAccess,
									 ResourceSync::ComputeShading,
									 ResourceAccess::UnorderedAccess);
	};

	bind(PhotonPassUpdate);
	renderContext->Dispatch(PHOTON_MAP_TEXELS / PHOTON_GROUP_SIZE, 1, 1);
	computeBarrier();
	bind(PhotonPassScanBlocks);
	renderContext->Dispatch(PHOTON_MAP_BLOCKS, 1, 1);
	computeBarrier();
	bind(PhotonPassScanTotals);
	renderContext->Dispatch(1, 1, 1);
	renderContext->GlobalBarrier(ResourceSync::ComputeShading,
								 ResourceAccess::UnorderedAccess,
								 ResourceSync::Raytracing,
								 ResourceAccess::UnorderedAccess);
}

void PathTracerPass::DispatchPhotonSplat(RenderContext *renderContext, ResourceHandle constantBuffer)
{
	const PhotonPassConstants constants = PhotonPassConstantsForFrame();
	auto bind = [&](PhotonPass pass)
	{
		renderContext->SetComputePipeline(photonPipelines_[pass]);
		renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
		renderContext->SetComputeConstantBufferByName("g_Frame", constantBuffer);
	};
	renderContext->GlobalBarrier(ResourceSync::Raytracing,
								 ResourceAccess::UnorderedAccess,
								 ResourceSync::ComputeShading,
								 ResourceAccess::UnorderedAccess);
	bind(PhotonPassSplat);
	renderContext->Dispatch(PHOTON_RECORD_CAPACITY / PHOTON_GROUP_SIZE, 1, 1);
	renderContext->GlobalBarrier(ResourceSync::ComputeShading,
								 ResourceAccess::UnorderedAccess,
								 ResourceSync::ComputeShading,
								 ResourceAccess::UnorderedAccess);
	bind(PhotonPassComposite);
	renderContext->Dispatch((width_ + PHOTON_SCREEN_GROUP_SIZE - 1) / PHOTON_SCREEN_GROUP_SIZE,
							(height_ + PHOTON_SCREEN_GROUP_SIZE - 1) / PHOTON_SCREEN_GROUP_SIZE,
							1);
	renderContext->GlobalBarrier(ResourceSync::ComputeShading, ResourceAccess::UnorderedAccess, ResourceSync::All, ResourceAccess::UnorderedAccess);
	photonMapValid_ = true;
}

ResourceHandle PathTracerPass::CreateTarget(Format format, const char *debugName) const
{
	TextureDesc desc = {};
	desc.width = width_;
	desc.height = height_;
	desc.format = format;
	desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	desc.debugName = debugName;
	return renderContext_->CreateTexture(desc, ResourceViewType::UnorderedAccess);
}

void PathTracerPass::Execute(RenderContext *renderContext)
{
	if (!scene_.HasGeometry())
	{
		return;
	}

	// Index 0 of the Halton sequence is the pixel corner, so start at 1.
	jitter_ = rayReconstruction_ ? dlssJitter(jitterIndex_ % kJitterPhaseCount + 1) : float2(0.0f, 0.0f);
	++jitterIndex_;

	FrameConstants constants = {};
	constants.invView = InvertMatrix(camera_.GetViewMatrix());
	constants.invProj = InvertMatrix(camera_.GetProjectionMatrix());
	constants.viewProj = camera_.GetViewMatrix() * camera_.GetProjectionMatrix();
	constants.prevViewProj = camera_.GetPreviousViewMatrix() * camera_.GetPreviousProjectionMatrix();
	const float4x4 prevInvView = InvertMatrix(camera_.GetPreviousViewMatrix());
	constants.prevCameraPosition = { prevInvView._41, prevInvView._42, prevInvView._43 };
	constants.jitter = jitter_;

	// Sun disc of angular radius r. 1 - cos(r) = 2 sin^2(r / 2) stays accurate for the tiny
	// physical sun; the disc's radiance is its irradiance spread over the solid angle.
	const SunSettings &sun = settings_.sun;
	const float3 irradiance = sun.colour * sun.intensity;
	const float halfRadius = 0.5f * DegreesToRadians(sun.angularRadiusDegrees);
	const float oneMinusCosMax = 2.0f * sinf(halfRadius) * sinf(halfRadius);
	const bool sunIsDelta = oneMinusCosMax < 1.0e-12f;
	const float solidAngle = 2.0f * Pi * oneMinusCosMax;

	constants.sunDirection = NormaliseFloat3Copy(sun.GetDirection());
	constants.sunIsDelta = sunIsDelta ? 1u : 0u;
	constants.sunOneMinusCosMax = sunIsDelta ? 0.0f : oneMinusCosMax;
	constants.sunRadiance = sunIsDelta ? irradiance : irradiance * (1.0f / solidAngle);
	constants.sunPdf = sunIsDelta ? 1.0f : 1.0f / solidAngle;

	// Split next event estimation evenly between the sun and the light buffer when both emit,
	// otherwise give it all to whichever does.
	const LightSettings &lights = settings_.lights;
	const bool atmosphere = atmosphere_.IsEnabled();
	const bool sunActive = std::max({ irradiance.x, irradiance.y, irradiance.z }) > 0.0f &&
						   (!atmosphere || atmosphere_.IsSunAboveHorizon());
	const bool lightsActive = lights.enabled && scene_.GetLightCount() > 0;
	constants.lightCount = lightsActive ? scene_.GetLightCount() : 0u;
	constants.lightSampling = static_cast<uint>(lights.sampling);
	constants.lightTreeBounces = static_cast<uint>(std::max(lights.treeBounces, 0));
	constants.lightTreePowerMix = std::clamp(lights.treePowerMix, 0.0f, 1.0f);
	constants.lightTreeLeafLevel = std::max(scene_.GetLightTreeDepth(), 1u) - 1;
	constants.lightTreeLightCount = scene_.GetLightTreeLightCount();
	constants.lightTreeLeafSize = scene_.GetLightTreeLeafSize();
	constants.lightTreeOrigin = scene_.GetLightTreeOrigin();
	constants.lightTreeStep = scene_.GetLightTreeStep();
	constants.sunSelectProbability = sunActive ? (lightsActive ? 0.5f : 1.0f) : 0.0f;
	constants.lightTreeGlossyBounces = static_cast<uint>(std::max(lights.glossyImportanceBounces, 0));
	constants.lightTreeFarFieldSharpness = lights.farFieldImportance ? lights.farFieldSharpness : FLT_MAX;
	constants.lightTreeDiffuseOnlyAlpha = lights.diffuseOnlyRoughLobes
											  ? lights.diffuseOnlyRoughness * lights.diffuseOnlyRoughness
											  : FLT_MAX;

	// Tokuyoshi and Kaplanyan's kernel, 2 * sigma^2 of the pixel filter, and their alpha^2 cap.
	// The reference leaves it out unless asked to keep the real-time approximations.
	const bool approximations = !reference_ || settings_.reference.approximations;
	const SpecularAASettings &specularAA = settings_.specularAA;
	const bool specularAAEnabled = specularAA.enabled && approximations;
	constants.specularAAVariance = specularAAEnabled ? 2.0f * 0.15915494f * std::max(specularAA.strength, 0.0f) : 0.0f;
	constants.specularAANormalMapScale = specularAAEnabled && specularAA.normalMaps ? std::max(specularAA.strength, 0.0f) : 0.0f;
	constants.specularAAMaxVariance = std::max(specularAA.maxVariance, 0.0f);

	// Enough bits to tell every material row apart, up to the 16 that MaybeReorderThread honours.
	uint hintBits = 0;
	while (hintBits < 16 && (1u << hintBits) < scene_.GetMaterialCount())
	{
		++hintBits;
	}
	constants.serCoherenceHintBits = hintBits;
	constants.shadowInstanceMask = settings_.blendedShadows ? 0xFFu : 0xFFu & ~static_cast<uint>(INSTANCE_MASK_BLENDED);
	constants.blendGuides = settings_.blendGuides && scene_.HasBlendedInstances() ? 1u : 0u;
	constants.materialOverrides = (settings_.specularMaterials ? 0u : MATERIAL_OVERRIDE_NO_SPECULAR) |
								  (settings_.transmissiveMaterials ? 0u : MATERIAL_OVERRIDE_NO_TRANSMISSION);

	// One pixel's angle at the render resolution. DLSS reconstructs at the output resolution, so the
	// texture footprint shrinks by render / output width, the usual upscaler mip bias.
	constants.rayConePixelSpread = atanf(2.0f * tanf(0.5f * camera_.GetFovY()) / static_cast<float>(height_));
	constants.rayConeBsdfSpread = settings_.rayConeBsdfSpread;
	const float upscalerBias = log2f(static_cast<float>(width_) / static_cast<float>(renderContext->GetWindowWidth()));
	constants.textureLodScale =
			settings_.rayConeTextureLod ? exp2f(settings_.textureLodBias + upscalerBias) : 0.0f;
	constants.anisotropicSegments = static_cast<uint>(std::max(settings_.anisotropicSegments, 0));
	// TracePath counts splits in 7 bits.
	constants.glassReflectionSplits = static_cast<uint>(std::clamp(settings_.glassReflectionSplits, 0, 127));
	constants.indirectClamp = settings_.clampIndirect && approximations ? std::max(settings_.indirectClampValue, 0.0f) : FLT_MAX;
	constants.directClamp = settings_.clampDirect && approximations ? std::max(settings_.directClampValue, 0.0f) : FLT_MAX;
	constants.opaqueAlphaDiffuseBounces = settings_.opaqueAlphaDiffuseBounces > 0 && approximations
												  ? static_cast<uint>(settings_.opaqueAlphaDiffuseBounces)
												  : UINT32_MAX;
	constants.reducedMaterialDiffuseBounces = settings_.reducedMaterialDiffuseBounces > 0 && approximations
													  ? static_cast<uint>(settings_.reducedMaterialDiffuseBounces)
													  : UINT32_MAX;
	constants.reflectionMinProbability = std::clamp(settings_.reflectionMinProbability, 0.0f, 0.95f);
	const float reflectionMaxRoughness = std::clamp(settings_.reflectionMinProbabilityMaxRoughness, 0.0f, 1.0f);
	constants.reflectionMinProbabilityMaxAlpha = reflectionMaxRoughness * reflectionMaxRoughness;
	constants.subsurfaceComponents = (settings_.subsurfaceZeroScattering ? SUBSURFACE_ZERO_SCATTERING : 0u) |
									 (settings_.subsurfaceSingleScattering ? SUBSURFACE_SINGLE_SCATTERING : 0u) |
									 (settings_.subsurfaceMultipleScattering ? SUBSURFACE_MULTIPLE_SCATTERING : 0u);
	constants.subsurfaceRadiusScale = std::max(settings_.subsurfaceRadiusScale, 0.0f);
	constants.diffuseRoughnessScale = std::clamp(settings_.diffuseRoughnessScale, 0.0f, 1.0f);
	constants.opacityMicromap2StateScene = scene_.HasOpacityMicromaps() ? scene_.GetOpacityMicromap2StateTLASIndex() : 0u;
	constants.opacityMicromap2StateBounce = settings_.opacityMicromap2StateBounce >= 0
													? static_cast<uint>(settings_.opacityMicromap2StateBounce)
													: UINT32_MAX;
	constants.guideSamples = static_cast<uint>(std::clamp(settings_.guideSamples, 0, 8));

	constants.skyRadiance = settings_.skyColour * settings_.skyIntensity;

	constants.psrMaxRoughness = settings_.psrMaxRoughness;
	constants.psrStrongerLobe = settings_.psrStrongerLobe ? 1u : 0u;
	constants.psrMaxMinification = settings_.psrCurvedSurfaces ? std::max(settings_.psrMaxMinification, 1.0f) : -1.0f;
	constants.specularHitDistanceRay = settings_.specularHitDistanceRay ? 1u : 0u;
	constants.specularHitDistanceHoldRoughness =
			settings_.holdSpecularHitDistance ? settings_.holdHitDistanceMaxRoughness : -1.0f;
	constants.specularMotionMaxRoughness = settings_.specularMotionMaxRoughness;
	constants.psrSky = settings_.psrSky ? 1u : 0u;
	constants.psrGuideSamples = static_cast<uint>(std::clamp(settings_.psrGuideSamples, 0, 8));
	// Only Preset F reads the mask, so no other preset pays for it.
	if (settings_.rrPreset == DLSSRayReconstructionPreset::PresetF)
	{
		constants.reflectionResponsivity = settings_.reflectionResponsivity;
		constants.reflectionResponsivityMaxRoughness = std::max(settings_.reflectionResponsivityMaxRoughness, 1e-3f);
	}

	if (atmosphere)
	{
		const SkyAtmospherePass::SceneAir air = atmosphere_.GetSceneAir();
		const AtmosphereSettings &atmosphereSettings = settings_.atmosphere;
		constants.atmosphereRayleighScattering = air.rayleighScattering;
		constants.atmosphereMieScattering = air.mieScattering;
		constants.atmosphereExtinction = air.extinction;
		constants.atmosphereMiePhaseG = atmosphereSettings.miePhaseG;
		constants.atmosphereViewHeight = atmosphere_.GetViewHeight();
		constants.atmosphereBottomRadius = atmosphere_.GetParameters().bottomRadius;
		// Rays that leave the scene cross about its diameter of air before reaching the sky.
		const float3 extent = scene_.GetBoundsMax() - scene_.GetBoundsMin();
		constants.atmosphereMissDistance = std::max(sqrtf(Dot(extent, extent)), 1.0e-3f);
		constants.atmosphereInScatterSegments = static_cast<uint>(std::max(atmosphereSettings.inScatterSegments, 0));
		constants.atmosphereVolumetricShadows = atmosphereSettings.volumetricShadows ? 1u : 0u;
		constants.atmosphereSkyVisibility = atmosphereSettings.skyVisibility ? 1u : 0u;
		constants.atmosphereSkyVisibilityDistance =
				atmosphereSettings.skyVisibilityDistance > 0.0f ? atmosphereSettings.skyVisibilityDistance : FLT_MAX;
		constants.clouds = atmosphere_.GetCloudParameters();
	}
	constants.frameIndex = renderContext->GetCurrentFrame();
	PreparePhotons(constants, sunActive, halfRadius * 2.0f);
	// TracePath keeps the diffuse bounces in 4 bits, up to one past the budget.
	constants.maxDiffuseBounces = static_cast<uint>(std::clamp(settings_.maxDiffuseBounces, 0, kMaxBounces));
	constants.maxSpecularBounces = static_cast<uint>(std::clamp(settings_.maxSpecularBounces, 0, kMaxBounces));
	constants.samplesPerFrame = static_cast<uint>(std::clamp(settings_.samplesPerPixel, 1, kMaxSamplesPerPixel));

	// RTXPT's threshold scale, with its middle grey's radiance from the exposure in the shader. The reference keeps it
	// only with its approximations, where auto exposure makes it follow the image (RTXPT has the same caveat).
	constants.fireflyFilterThreshold = settings_.fireflyFilter && approximations
											   ? std::max(settings_.fireflyFilterThreshold, 0.0f) * (1000.0f / sqrtf(0.18f))
											   : 0.0f;
	constants.exposureBuffer = exposure_ != InvalidResourceHandle ? renderContext->GetResourceViewInfo(exposure_).descriptorIndex
																   : INVALIDDESCRIPTORHANDLE;

	if (reference_)
	{
		// Anything the image depends on starts the sum over: the constants, but for those that change every frame
		// whatever else does, the atmosphere's LUTs (the sun is in the constants), and the UI's restart.
		FrameConstants key = constants;
		key.prevViewProj = {};
		key.prevCameraPosition = {};
		key.frameIndex = 0;
		key.opacityMicromap2StateScene = 0;
		key.previousScene = 0;
		key.samplesPerFrame = 0; // the reference's own, set below
		key.photons.rects = 0; // a buffer per frame in flight
		const ReferenceSettings &reference = settings_.reference;
		if (accumulatedSamples_ == 0 || std::memcmp(&key, &accumulationKey_, sizeof(key)) != 0 ||
			std::memcmp(&atmosphere_.GetParameters(), &accumulationAtmosphere_, sizeof(AtmosphereParameters)) != 0 ||
			reference.restartCount != accumulationRestartCount_)
		{
			accumulationKey_ = key;
			accumulationAtmosphere_ = atmosphere_.GetParameters();
			accumulationRestartCount_ = reference.restartCount;
			accumulatedSamples_ = 0;
			accumulationStart_ = std::chrono::steady_clock::now();
			accumulationSeconds_ = 0.0f;
		}

		// Done: the output keeps the mean.
		const uint target = static_cast<uint>(std::max(reference.samples, 0));
		if (target > 0 && accumulatedSamples_ >= target)
		{
			return;
		}
		uint samples = static_cast<uint>(std::max(reference.samplesPerFrame, 1));
		if (target > 0)
		{
			samples = std::min(samples, target - accumulatedSamples_);
		}
		constants.referenceSampleCount = accumulatedSamples_;
		constants.samplesPerFrame = samples;
		accumulatedSamples_ += samples;
		// Up to when this frame is recorded, not when the GPU finishes it.
		accumulationSeconds_ =
				std::chrono::duration<float>(std::chrono::steady_clock::now() - accumulationStart_).count();
	}

	const Pipeline *pipeline = SelectPipeline();

	// Moving shadows are found against last frame's TLAS of the kind this pipeline's shadow rays trace.
	if (rayReconstruction_ && settings_.shadowChangeMask && scene_.MovedSinceLastFrame())
	{
		constants.shadowChanges = 1;
		constants.previousScene =
				renderContext->GetResourceViewInfo(scene_.GetPreviousShadowTLASSRV(pipeline->opacityMicromaps)).descriptorIndex;
	}

	// The slot for this frame index was last read framesInFlight frames ago, and BeginFrame has
	// already waited on that frame's fence.
	const ResourceHandle constantBuffer = frameConstants_[renderContext->GetCurrentFrameIndex()];
	renderContext->UpdateConstantBuffer(constantBuffer, &constants, sizeof(constants));

	renderContext->BeginEvent("PathTracer");
	renderContext->BeginProfileScope("PathTracer");
	const bool photons = constants.photons.enabled != 0;

	// The output, the guides and the camera state, in their u0..u8 register order.
	const ResourceHandle targets[] = { output_,
									   guides_.diffuseAlbedo,
									   guides_.specularAlbedo,
									   guides_.normalRoughness,
									   guides_.motionVectors,
									   guides_.depth,
									   guides_.specularHitDistance,
									   guides_.specularMotionVectors,
									   cameraState_ };
	const uint32_t targetCount = rayReconstruction_ ? 9 : 1;
	for (uint32_t i = 0; i < targetCount; ++i)
	{
		renderContext->TransitionResource(targets[i],
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::Raytracing,
										  ResourceAccess::UnorderedAccess);
	}
	renderContext->TransitionResource(deferredBranches_,
									  ResourceLayout::UnorderedAccess,
									  ResourceSync::Raytracing,
									  ResourceAccess::UnorderedAccess);
	if (rayReconstruction_)
	{
		for (ResourceHandle guide: { guides_.shadowChanges, guides_.responsivity, guides_.psrOutcome })
		{
			renderContext->TransitionResource(guide,
											  ResourceLayout::UnorderedAccess,
											  ResourceSync::Raytracing,
											  ResourceAccess::UnorderedAccess);
		}
	}

	// Bound again after the photon map's passes, whose compute pipelines replace the root signature.
	auto bindRaytracing = [&]()
	{
		renderContext->SetRaytracingPipeline(pipeline->pipeline);
		renderContext->SetRaytracingConstantBuffer(0, constantBuffer);
		renderContext->SetRaytracingSRV(0, scene_.GetTLASSRV(pipeline->opacityMicromaps));
		// Scene buffers are root SRVs, registers must match PathTracer/Resources.slang.
		renderContext->SetRaytracingSRV(1, scene_.GetPrimitiveBuffer());
		renderContext->SetRaytracingSRV(2, scene_.GetStreamBuffer(GPUStream::Index));
		renderContext->SetRaytracingSRV(3, scene_.GetVertexBuffer());
		renderContext->SetRaytracingSRV(4, scene_.GetMaterialBuffer());
		renderContext->SetRaytracingSRV(5, scene_.GetLightBuffer());
		renderContext->SetRaytracingSRV(6, scene_.GetLightTreeNodeBuffer());
		renderContext->SetRaytracingSRV(7, scene_.GetLightLeafBuffer());
		renderContext->SetRaytracingSRV(8, scene_.GetEmitterBuffer());
		renderContext->SetRaytracingSRV(10, scene_.GetLightTreeLeafLightBuffer());
		renderContext->SetRaytracingSRV(13, scene_.GetLightPowerBuffer());
		renderContext->SetRaytracingSRV(14, scene_.GetLightEmitterBuffer());
		renderContext->SetRaytracingSRV(15, scene_.GetEmissionScaleBuffer());
		if (rayReconstruction_)
		{
			renderContext->SetRaytracingSRV(16, scene_.GetPreviousTransformBuffer());
		}

		if (atmosphere)
		{
			renderContext->SetRaytracingSRV(11, atmosphere_.GetSkyViewLUT());
			renderContext->SetRaytracingSRV(12, atmosphere_.GetSceneLighting());
		}

		for (uint32_t i = 0; i < targetCount; ++i)
		{
			renderContext->SetRaytracingUAV(i, targets[i]);
		}
		renderContext->SetRaytracingUAV(9, deferredBranches_);
		if (rayReconstruction_)
		{
			renderContext->SetRaytracingUAV(10, guides_.shadowChanges);
			renderContext->SetRaytracingUAV(11, guides_.responsivity);
			renderContext->SetRaytracingUAV(12, guides_.psrOutcome);
		}
		if (reference_)
		{
			renderContext->TransitionResource(accumulation_,
											  ResourceLayout::UnorderedAccess,
											  ResourceSync::Raytracing,
											  ResourceAccess::UnorderedAccess);
			renderContext->SetRaytracingUAV(1, accumulation_);
		}
	};

	// Caustic photons: the photon map adapted and scanned, then the photons traced into their records, before the paths
	// mark the receivers. A map that photons skipped a frame of starts over.
	const ResourceHandle photonBuffers[] = { photonDensity_[0], photonDensity_[1], photonTexelPrefix_, photonBlockPrefix_,
											 photonFeedback_,	photonRecords_,	   photonRecordCount_, photonReceivers_,
											 photonCaustics_,	photonHistory_[0], photonHistory_[1] };
	if (!photons)
	{
		photonMapValid_ = false;
	}
	else
	{
		renderContext->BeginProfileScope("Photons");
		for (ResourceHandle buffer: photonBuffers)
		{
			renderContext->TransitionResource(buffer, ResourceLayout::UnorderedAccess, ResourceSync::All, ResourceAccess::UnorderedAccess);
		}
		// Last frame's splat and composite read and wrote what this frame's passes write.
		renderContext->GlobalBarrier(ResourceSync::All, ResourceAccess::UnorderedAccess, ResourceSync::All, ResourceAccess::UnorderedAccess);
		DispatchPhotonMap(renderContext, constantBuffer);
		bindRaytracing();
		const uint32_t rows = (constants.photons.photonCount + PHOTON_DISPATCH_WIDTH - 1) / PHOTON_DISPATCH_WIDTH;
		renderContext->DispatchRays(pipeline->photonShaderTable, PHOTON_DISPATCH_WIDTH, rows, 1);
		renderContext->EndProfileScope();
	}

	bindRaytracing();
	renderContext->DispatchRays(pipeline->shaderTable, width_, height_, 1);

	// The photons splatted onto the receivers the paths marked, and added to the output.
	if (photons)
	{
		renderContext->BeginProfileScope("Caustics");
		DispatchPhotonSplat(renderContext, constantBuffer);
		renderContext->EndProfileScope();
	}

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
