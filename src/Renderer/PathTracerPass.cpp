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
								  cameraState_,
								  deferredBranches_,
								  accumulation_ })
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
}

const PathTracerPass::Pipeline *PathTracerPass::SelectPipeline()
{
	const bool atmosphere = atmosphere_.IsEnabled();
	const bool inlineShadows = settings_.inlineShadowRays;
	// The guides are only traced for ray reconstruction.
	const bool psr = rayReconstruction_ && settings_.primarySurfaceReplacement;

	// A failed opacity micromap or SER compile is logged once, and the pipeline without them takes over.
	if (settings_.opacityMicromaps && scene_.HasOpacityMicromaps() && !ommFailed_)
	{
		const bool ser = settings_.shaderExecutionReordering && serSupported_ && !serFailed_;
		if (const Pipeline *pipeline = GetPipeline(ser, atmosphere, inlineShadows, true, psr))
		{
			return pipeline;
		}
		loge("Failed to create the opacity micromap pipeline, tracing without them");
		ommFailed_ = true;
	}

	if (settings_.shaderExecutionReordering && serSupported_ && !serFailed_)
	{
		if (const Pipeline *pipeline = GetPipeline(true, atmosphere, inlineShadows, false, psr))
		{
			return pipeline;
		}
		loge("Failed to create the shader execution reordering pipeline, falling back to TraceRay");
		serFailed_ = true;
	}

	const Pipeline *pipeline = GetPipeline(false, atmosphere, inlineShadows, false, psr);
	if (pipeline == nullptr)
	{
		throw std::runtime_error("Failed to create the path tracing pipeline (see the shader log above).");
	}
	return pipeline;
}

const PathTracerPass::Pipeline *PathTracerPass::GetPipeline(bool shaderExecutionReordering, bool atmosphere,
															bool inlineShadows, bool opacityMicromaps, bool psr)
{
	Pipeline &pipeline = pipelines_[shaderExecutionReordering ? 1 : 0][atmosphere ? 1 : 0][inlineShadows ? 1 : 0]
								   [opacityMicromaps ? 1 : 0][psr ? 1 : 0];
	if (pipeline.pipeline == InvalidPipelineHandle)
	{
		pipeline = CreatePipeline(shaderExecutionReordering, atmosphere, inlineShadows, opacityMicromaps, psr);
	}
	return pipeline.pipeline != InvalidPipelineHandle ? &pipeline : nullptr;
}

PathTracerPass::Pipeline PathTracerPass::CreatePipeline(bool shaderExecutionReordering, bool atmosphere,
														 bool inlineShadows, bool opacityMicromaps, bool psr) const
{
	RaytracingPipelineDesc desc = {};
	desc.exports = { "RayGen", "Miss", "ClosestHit", "AnyHit" };
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

	result.shaderTable = renderContext_->CreateShaderTable(result.pipeline);
	renderContext_->AddRayGenShaderRecord(result.shaderTable, "RayGen");
	// Record order matches STANDARD_RAY_INDEX and SHADOW_RAY_INDEX in PathTracer/RayTracing.slang.
	renderContext_->AddMissShaderRecord(result.shaderTable, "Miss");
	renderContext_->AddHitGroupShaderRecord(result.shaderTable, "HitGroup");
	if (!inlineShadows)
	{
		renderContext_->AddMissShaderRecord(result.shaderTable, "ShadowMiss");
		renderContext_->AddHitGroupShaderRecord(result.shaderTable, "ShadowHitGroup");
	}
	renderContext_->BuildShaderTable(result.shaderTable);
	return result;
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
	constants.staticLightCount = lightsActive ? scene_.GetStaticLightCount() : 0u;
	constants.staticLightPowerShare = scene_.GetStaticLightPowerShare();
	constants.lightSampling = static_cast<uint>(lights.sampling);
	constants.lightTreeBounces = static_cast<uint>(std::max(lights.treeBounces, 0));
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
	// TracePath counts splits in 8 bits.
	constants.glassReflectionSplits = static_cast<uint>(std::clamp(settings_.glassReflectionSplits, 0, 255));
	constants.blendLayerSplits = static_cast<uint>(std::clamp(settings_.blendLayerSplits, 0, 255));
	constants.indirectClamp = settings_.clampIndirect && approximations ? std::max(settings_.indirectClampValue, 0.0f) : FLT_MAX;
	constants.directClamp = settings_.clampDirect && approximations ? std::max(settings_.directClampValue, 0.0f) : FLT_MAX;
	constants.subsurfaceComponents = (settings_.subsurfaceZeroScattering ? SUBSURFACE_ZERO_SCATTERING : 0u) |
									 (settings_.subsurfaceSingleScattering ? SUBSURFACE_SINGLE_SCATTERING : 0u) |
									 (settings_.subsurfaceMultipleScattering ? SUBSURFACE_MULTIPLE_SCATTERING : 0u);
	constants.subsurfaceRadiusScale = std::max(settings_.subsurfaceRadiusScale, 0.0f);
	constants.opacityMicromap2StateScene = scene_.HasOpacityMicromaps() ? scene_.GetOpacityMicromap2StateTLASIndex() : 0u;
	constants.opacityMicromap2StateBounce = settings_.opacityMicromap2StateBounce >= 0
													? static_cast<uint>(settings_.opacityMicromap2StateBounce)
													: UINT32_MAX;

	constants.skyRadiance = settings_.skyColour * settings_.skyIntensity;

	constants.psrMaxRoughness = settings_.psrMaxRoughness;
	constants.psrStrongerLobe = settings_.psrStrongerLobe ? 1u : 0u;
	constants.psrMaxMinification = settings_.psrCurvedSurfaces ? std::max(settings_.psrMaxMinification, 1.0f) : -1.0f;
	constants.specularHitDistanceHoldRoughness =
			settings_.holdSpecularHitDistance ? settings_.holdHitDistanceMaxRoughness : -1.0f;

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
		constants.clouds = atmosphere_.GetCloudParameters();
	}
	constants.frameIndex = renderContext->GetCurrentFrame();
	// TracePath keeps the diffuse bounces in 4 bits, up to one past the budget.
	constants.maxDiffuseBounces = static_cast<uint>(std::clamp(settings_.maxDiffuseBounces, 0, kMaxBounces));
	constants.maxSpecularBounces = static_cast<uint>(std::clamp(settings_.maxSpecularBounces, 0, kMaxBounces));
	constants.samplesPerFrame = static_cast<uint>(std::clamp(settings_.samplesPerPixel, 1, kMaxSamplesPerPixel));

	if (reference_)
	{
		// Anything the image depends on starts the sum over: the constants, but for those that change every frame
		// whatever else does, the atmosphere's LUTs (the sun is in the constants), and the UI's restart.
		FrameConstants key = constants;
		key.prevViewProj = {};
		key.prevCameraPosition = {};
		key.frameIndex = 0;
		key.samplesPerFrame = 0; // the reference's own, set below
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

	// The slot for this frame index was last read framesInFlight frames ago, and BeginFrame has
	// already waited on that frame's fence.
	const ResourceHandle constantBuffer = frameConstants_[renderContext->GetCurrentFrameIndex()];
	renderContext->UpdateConstantBuffer(constantBuffer, &constants, sizeof(constants));

	const Pipeline *pipeline = SelectPipeline();

	renderContext->BeginEvent("PathTracer");
	renderContext->BeginProfileScope("PathTracer");

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
	renderContext->SetRaytracingSRV(9, scene_.GetLightTreeParentBuffer());
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
	if (reference_)
	{
		renderContext->TransitionResource(accumulation_,
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::Raytracing,
										  ResourceAccess::UnorderedAccess);
		renderContext->SetRaytracingUAV(1, accumulation_);
	}
	renderContext->DispatchRays(pipeline->shaderTable, width_, height_, 1);

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
