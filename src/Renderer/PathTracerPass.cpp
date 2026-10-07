#include "PathTracerPass.h"

#include "Camera.h"
#include "RenderSettings.h"
#include "Scene/RenderGPUScene.h"

#include "DLSSUtil.h"

#include "fmtlog.h"

#include <algorithm>
#include <cfloat>
#include <math.h>
#include <stdexcept>

#undef min
#undef max

namespace
{
	constexpr uint32_t kMaxPayloadSizeInBytes = 76;

	// Halton phases before the jitter repeats. The DLSS guide asks for 8 * (output / render)^2, so
	// 72 at Ultra Performance, and the RR guide for at least 32 with no reason to limit it.
	constexpr uint32_t kJitterPhaseCount = 128;
} // namespace

PathTracerPass::PathTracerPass(RenderContext *renderContext,
							   const RenderGPUScene &scene,
							   const Camera &camera,
							   const RenderSettings &settings,
							   bool rayReconstruction) :
	renderContext_(renderContext), scene_(scene), camera_(camera), settings_(settings),
	rayReconstruction_(rayReconstruction)
{
	serSupported_ = renderContext->GetShaderExecutionReorderingSupport().supported;

	// Build the pipeline the settings start with now rather than on the first frame.
	SelectPipeline();

	width_ = renderContext->GetRenderWidth();
	height_ = renderContext->GetRenderHeight();

	output_ = CreateTarget(Format::R16G16B16A16_FLOAT, "PathTracer_Radiance");

	if (rayReconstruction_)
	{
		guides_.diffuseAlbedo = CreateTarget(Format::R8G8B8A8_UNORM, "PathTracer_DiffuseAlbedo");
		guides_.specularAlbedo = CreateTarget(Format::R16G16B16A16_FLOAT, "PathTracer_SpecularAlbedo");
		guides_.normalRoughness = CreateTarget(Format::R16G16B16A16_FLOAT, "PathTracer_NormalRoughness");
		guides_.motionVectors = CreateTarget(Format::R16G16_FLOAT, "PathTracer_MotionVectors");
		guides_.depth = CreateTarget(Format::R32_FLOAT, "PathTracer_Depth");
		guides_.specularHitDistance = CreateTarget(Format::R16_FLOAT, "PathTracer_SpecularHitDistance");
	}

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
								  guides_.specularHitDistance })
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
	// A failed SER compile is logged once and the plain pipeline takes over.
	if (settings_.shaderExecutionReordering && serSupported_ && !serFailed_)
	{
		if (const Pipeline *pipeline = GetPipeline(true))
		{
			return pipeline;
		}
		loge("Failed to create the shader execution reordering pipeline, falling back to TraceRay");
		serFailed_ = true;
	}

	const Pipeline *pipeline = GetPipeline(false);
	if (pipeline == nullptr)
	{
		throw std::runtime_error("Failed to create the path tracing pipeline (see the shader log above).");
	}
	return pipeline;
}

const PathTracerPass::Pipeline *PathTracerPass::GetPipeline(bool shaderExecutionReordering)
{
	Pipeline &pipeline = pipelines_[shaderExecutionReordering ? 1 : 0];
	if (pipeline.pipeline == InvalidPipelineHandle)
	{
		pipeline = CreatePipeline(shaderExecutionReordering);
	}
	return pipeline.pipeline != InvalidPipelineHandle ? &pipeline : nullptr;
}

PathTracerPass::Pipeline PathTracerPass::CreatePipeline(bool shaderExecutionReordering) const
{
	RaytracingPipelineDesc desc = {};
	desc.exports = { "RayGen", "Miss", "ClosestHit", "AnyHit", "ShadowMiss", "ShadowClosestHit", "ShadowAnyHit" };
	desc.hitGroups = { RaytracingHitGroup{ "HitGroup", "ClosestHit", "AnyHit", "" },
					   RaytracingHitGroup{ "ShadowHitGroup", "ShadowClosestHit", "ShadowAnyHit", "" } };
	desc.maxPayloadSizeInBytes = kMaxPayloadSizeInBytes;
	desc.maxAttributeSizeInBytes = 8; // triangle barycentrics
	desc.maxTraceRecursionDepth = 1; // primary and shadow rays are both traced from raygen
	if (rayReconstruction_)
	{
		desc.defines.push_back("DLSS_RR");
	}
	if (shaderExecutionReordering)
	{
		// HitObject and MaybeReorderThread are Shader Model 6.9.
		desc.defines.push_back("SER");
		desc.profile = "lib_6_9";
	}

	Pipeline result;
	result.pipeline = renderContext_->CreateRaytracingPipeline(PATHTRACER_SHADER_DIR "PathTracer.slang", desc);
	if (result.pipeline == InvalidPipelineHandle)
	{
		return result;
	}

	result.shaderTable = renderContext_->CreateShaderTable(result.pipeline);
	renderContext_->AddRayGenShaderRecord(result.shaderTable, "RayGen");
	// Record order matches STANDARD_RAY_INDEX and SHADOW_RAY_INDEX in PathTracer.slang.
	renderContext_->AddMissShaderRecord(result.shaderTable, "Miss");
	renderContext_->AddMissShaderRecord(result.shaderTable, "ShadowMiss");
	renderContext_->AddHitGroupShaderRecord(result.shaderTable, "HitGroup");
	renderContext_->AddHitGroupShaderRecord(result.shaderTable, "ShadowHitGroup");
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
	const bool sunActive = std::max({ irradiance.x, irradiance.y, irradiance.z }) > 0.0f;
	const bool lightsActive = lights.enabled && scene_.GetLightCount() > 0;
	constants.lightCount = lightsActive ? scene_.GetLightCount() : 0u;
	constants.lightSampling = static_cast<uint>(lights.sampling);
	constants.sunSelectProbability = sunActive ? (lightsActive ? 0.5f : 1.0f) : 0.0f;
	constants.lightTreeGlossyBounces = static_cast<uint>(std::max(lights.glossyImportanceBounces, 0));
	constants.lightTreeFarFieldSharpness = lights.farFieldImportance ? lights.farFieldSharpness : FLT_MAX;
	constants.lightTreeDiffuseOnlyAlpha = lights.diffuseOnlyRoughLobes
											  ? lights.diffuseOnlyRoughness * lights.diffuseOnlyRoughness
											  : FLT_MAX;

	// Enough bits to tell every material row apart, up to the 16 that MaybeReorderThread honours.
	uint hintBits = 0;
	while (hintBits < 16 && (1u << hintBits) < scene_.GetMaterialCount())
	{
		++hintBits;
	}
	constants.serCoherenceHintBits = hintBits;

	// One pixel's angle at the render resolution. DLSS reconstructs at the output resolution, so the
	// texture footprint shrinks by render / output width, the usual upscaler mip bias.
	constants.rayConePixelSpread = atanf(2.0f * tanf(0.5f * camera_.GetFovY()) / static_cast<float>(height_));
	constants.rayConeBsdfSpread = settings_.rayConeBsdfSpread;
	const float upscalerBias = log2f(static_cast<float>(width_) / static_cast<float>(renderContext->GetWindowWidth()));
	constants.textureLodScale =
			settings_.rayConeTextureLod ? exp2f(settings_.textureLodBias + upscalerBias) : 0.0f;

	constants.skyRadiance = settings_.skyColour * settings_.skyIntensity;
	constants.frameIndex = renderContext->GetCurrentFrame();
	constants.maxBounces = static_cast<uint>(settings_.maxBounces);

	// The slot for this frame index was last read framesInFlight frames ago, and BeginFrame has
	// already waited on that frame's fence.
	const ResourceHandle constantBuffer = frameConstants_[renderContext->GetCurrentFrameIndex()];
	renderContext->UpdateConstantBuffer(constantBuffer, &constants, sizeof(constants));

	const Pipeline *pipeline = SelectPipeline();

	renderContext->BeginEvent("PathTracer");
	renderContext->BeginProfileScope("PathTracer");

	// The output and then the guides, in their u0..u6 register order.
	const ResourceHandle targets[] = { output_,
									   guides_.diffuseAlbedo,
									   guides_.specularAlbedo,
									   guides_.normalRoughness,
									   guides_.motionVectors,
									   guides_.depth,
									   guides_.specularHitDistance };
	const uint32_t targetCount = rayReconstruction_ ? 7 : 1;
	for (uint32_t i = 0; i < targetCount; ++i)
	{
		renderContext->TransitionResource(targets[i],
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::Raytracing,
										  ResourceAccess::UnorderedAccess);
	}

	renderContext->SetRaytracingPipeline(pipeline->pipeline);
	renderContext->SetRaytracingConstantBuffer(0, constantBuffer);
	renderContext->SetRaytracingSRV(0, scene_.GetTLASSRV());
	// Scene buffers are root SRVs, registers must match PathTracer.slang.
	renderContext->SetRaytracingSRV(1, scene_.GetPrimitiveBuffer());
	renderContext->SetRaytracingSRV(2, scene_.GetStreamBuffer(GPUStream::Index));
	renderContext->SetRaytracingSRV(3, scene_.GetStreamBuffer(GPUStream::Position));
	renderContext->SetRaytracingSRV(4, scene_.GetStreamBuffer(GPUStream::Normal));
	renderContext->SetRaytracingSRV(5, scene_.GetStreamBuffer(GPUStream::TexCoord0));
	renderContext->SetRaytracingSRV(6, scene_.GetStreamBuffer(GPUStream::Tangent));
	renderContext->SetRaytracingSRV(7, scene_.GetMaterialBuffer());
	renderContext->SetRaytracingSRV(8, scene_.GetLightBuffer());
	renderContext->SetRaytracingSRV(9, scene_.GetLightTreeNodeBuffer());
	renderContext->SetRaytracingSRV(10, scene_.GetLightLeafBuffer());
	renderContext->SetRaytracingSRV(11, scene_.GetEmitterBuffer());
	renderContext->SetRaytracingSRV(12, scene_.GetLightTreeParentBuffer());
	renderContext->SetRaytracingSRV(13, scene_.GetLightTreeLeafLightBuffer());
	for (uint32_t i = 0; i < targetCount; ++i)
	{
		renderContext->SetRaytracingUAV(i, targets[i]);
	}
	renderContext->DispatchRays(pipeline->shaderTable, width_, height_, 1);

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
