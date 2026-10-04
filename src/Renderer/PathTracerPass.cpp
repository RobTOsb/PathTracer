#include "PathTracerPass.h"

#include "Camera.h"
#include "RenderSettings.h"
#include "Scene/RenderGPUScene.h"

#include <math.h>
#include <stdexcept>

namespace
{
	// Must cover HitPayload in PathTracer.slang (14 x 4 bytes). The driver sizes the ray stack from
	// this, so keep it tight.
	constexpr uint32_t kMaxPayloadSizeInBytes = 56;
} // namespace

PathTracerPass::PathTracerPass(RenderContext *renderContext,
							   const RenderGPUScene &scene,
							   const Camera &camera,
							   const RenderSettings &settings) :
	renderContext_(renderContext), scene_(scene), camera_(camera), settings_(settings)
{
	RaytracingPipelineDesc desc = {};
	desc.exports = { "RayGen", "Miss", "ClosestHit", "ShadowMiss", "ShadowClosestHit", "ShadowAnyHit" };
	desc.hitGroups = { RaytracingHitGroup{ "HitGroup", "ClosestHit", "", "" },
					   RaytracingHitGroup{ "ShadowHitGroup", "ShadowClosestHit", "ShadowAnyHit", "" } };
	desc.maxPayloadSizeInBytes = kMaxPayloadSizeInBytes;
	desc.maxAttributeSizeInBytes = 8; // triangle barycentrics
	desc.maxTraceRecursionDepth = 1; // primary and shadow rays are both traced from raygen

	pipeline_ = renderContext->CreateRaytracingPipeline(PATHTRACER_SHADER_DIR "PathTracer.slang", desc);
	if (pipeline_ == InvalidPipelineHandle)
	{
		throw std::runtime_error("Failed to create the path tracing pipeline (see the shader log above).");
	}

	shaderTable_ = renderContext->CreateShaderTable(pipeline_);
	renderContext->AddRayGenShaderRecord(shaderTable_, "RayGen");
	// Record order matches STANDARD_RAY_INDEX and SHADOW_RAY_INDEX in PathTracer.slang.
	renderContext->AddMissShaderRecord(shaderTable_, "Miss");
	renderContext->AddMissShaderRecord(shaderTable_, "ShadowMiss");
	renderContext->AddHitGroupShaderRecord(shaderTable_, "HitGroup");
	renderContext->AddHitGroupShaderRecord(shaderTable_, "ShadowHitGroup");
	renderContext->BuildShaderTable(shaderTable_);

	width_ = renderContext->GetRenderWidth();
	height_ = renderContext->GetRenderHeight();

	TextureDesc outputDesc = {};
	outputDesc.width = width_;
	outputDesc.height = height_;
	outputDesc.format = Format::R16G16B16A16_FLOAT;
	outputDesc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	outputDesc.debugName = "PathTracer_Radiance";
	output_ = renderContext->CreateTexture(outputDesc, ResourceViewType::UnorderedAccess).handle;

	frameConstants_.resize(renderContext->GetFramesInFlight());
	for (ResourceHandle &constants: frameConstants_)
	{
		constants = renderContext->CreateConstantBuffer(sizeof(FrameConstants), "PathTracer_FrameConstants").handle;
	}
}

PathTracerPass::~PathTracerPass()
{
	renderContext_->DestroyResource(output_);
	for (ResourceHandle constants: frameConstants_)
	{
		renderContext_->DestroyResource(constants);
	}
}

void PathTracerPass::Execute(RenderContext *renderContext)
{
	if (!scene_.HasGeometry())
	{
		return;
	}

	FrameConstants constants = {};
	constants.invView = InvertMatrix(camera_.GetViewMatrix());
	constants.invProj = InvertMatrix(camera_.GetProjectionMatrix());

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

	constants.skyRadiance = settings_.skyColour * settings_.skyIntensity;
	constants.frameIndex = renderContext->GetCurrentFrame();
	constants.maxBounces = static_cast<uint>(settings_.maxBounces);

	// The slot for this frame index was last read framesInFlight frames ago, and BeginFrame has
	// already waited on that frame's fence.
	const ResourceHandle constantBuffer = frameConstants_[renderContext->GetCurrentFrameIndex()];
	renderContext->UpdateConstantBuffer(constantBuffer, &constants, sizeof(constants));

	renderContext->BeginEvent("PathTracer");
	renderContext->BeginProfileScope("PathTracer");

	renderContext->TransitionResource(output_,
									  ResourceLayout::UnorderedAccess,
									  ResourceSync::Raytracing,
									  ResourceAccess::UnorderedAccess);

	renderContext->SetRaytracingPipeline(pipeline_);
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
	renderContext->SetRaytracingUAV(0, output_);
	renderContext->DispatchRays(shaderTable_, width_, height_, 1);

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
