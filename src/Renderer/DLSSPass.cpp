#include "DLSSPass.h"

#include "D3D12RenderContext.h"

#include "Camera.h"
#include "PathTracerPass.h"

#include "fmtlog.h"

#include <cstring>

DLSSPass::DLSSPass(RenderContext *renderContext,
				   NGXContext &ngx,
				   const PathTracerPass &pathTracer,
				   const Camera &camera,
				   DLSSMode mode,
				   DLSSRayReconstructionPreset preset) :
	renderContext_(renderContext), ngx_(ngx), pathTracer_(pathTracer), camera_(camera)
{
	featureParams_.kind = DLSSFeatureKind::RayReconstruction;
	featureParams_.mode = mode;
	featureParams_.rrPreset = preset;
	featureParams_.renderWidth = pathTracer.GetWidth();
	featureParams_.renderHeight = pathTracer.GetHeight();
	featureParams_.outputWidth = renderContext->GetWindowWidth();
	featureParams_.outputHeight = renderContext->GetWindowHeight();
	featureParams_.hdr = true; // linear radiance, exposed by the tonemapper afterwards
	featureParams_.depthInverted = true; // the camera uses reversed Z
	featureParams_.packedNormalRoughness = true; // roughness in normalRoughness.w

	TextureDesc desc = {};
	desc.width = featureParams_.outputWidth;
	desc.height = featureParams_.outputHeight;
	desc.format = Format::R16G16B16A16_FLOAT;
	desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	desc.debugName = "DLSS_Output";
	output_ = renderContext->CreateTexture(desc, ResourceViewType::UnorderedAccess).handle;
}

DLSSPass::~DLSSPass()
{
	renderContext_->DestroyResource(output_);
}

void DLSSPass::Execute(RenderContext *renderContext)
{
	// NGX records straight into the native command list, and main only ever creates the D3D12 backend.
	D3D12RenderContext *d3dContext = static_cast<D3D12RenderContext *>(renderContext);
	ID3D12GraphicsCommandList *commandList = d3dContext->GetNativeGraphicsCommandList();

	renderContext->BeginEvent("DLSS-RR");
	renderContext->BeginProfileScope("DLSS-RR");

	// DLSS reads its inputs from compute shaders and writes the output as a UAV, and returns every
	// resource to these states when it is done.
	const PathTracerGuides &guides = pathTracer_.GetGuides();
	const ResourceHandle inputs[] = { pathTracer_.GetOutput(), guides.diffuseAlbedo,	guides.specularAlbedo,
									  guides.normalRoughness,  guides.motionVectors, guides.depth,
									  guides.specularHitDistance };
	for (ResourceHandle input: inputs)
	{
		renderContext->TransitionResource(input,
										  ResourceLayout::ShaderResource,
										  ResourceSync::ComputeShading,
										  ResourceAccess::ShaderResource);
	}
	renderContext->TransitionResource(output_,
									  ResourceLayout::UnorderedAccess,
									  ResourceSync::ComputeShading,
									  ResourceAccess::UnorderedAccess);

	bool evaluated = ngx_.EnsureFeature(featureParams_, commandList);
	if (evaluated)
	{
		DLSSEvalInputs eval = {};
		eval.colourInput = d3dContext->GetNativeResource(pathTracer_.GetOutput());
		eval.colourOutput = d3dContext->GetNativeResource(output_);
		eval.depth = d3dContext->GetNativeResource(guides.depth);
		eval.motionVectors = d3dContext->GetNativeResource(guides.motionVectors);
		eval.diffuseAlbedo = d3dContext->GetNativeResource(guides.diffuseAlbedo);
		eval.specularAlbedo = d3dContext->GetNativeResource(guides.specularAlbedo);
		eval.normalRoughness = d3dContext->GetNativeResource(guides.normalRoughness);
		eval.specularHitDistance = d3dContext->GetNativeResource(guides.specularHitDistance);
		eval.renderWidth = featureParams_.renderWidth;
		eval.renderHeight = featureParams_.renderHeight;
		eval.reset = reset_;

		// DLSS wants the jitter as applied to the projection, which shifts the image the opposite
		// way to the offset the camera rays were sampled at.
		const float2 jitter = pathTracer_.GetJitter();
		eval.jitterOffsetX = -jitter.x;
		eval.jitterOffsetY = -jitter.y;

		// Row-major with row vectors, as DirectXMath stores them; the projection is unjittered.
		const float4x4 view = camera_.GetViewMatrix();
		const float4x4 projection = camera_.GetProjectionMatrix();
		static_assert(sizeof(view) == sizeof(eval.worldToView));
		std::memcpy(eval.worldToView, &view, sizeof(eval.worldToView));
		std::memcpy(eval.viewToClip, &projection, sizeof(eval.viewToClip));

		evaluated = ngx_.EvaluateFeature(eval, commandList);
	}

	// NGX binds its own descriptor heaps, root signature and pipeline. Every later pass binds its
	// own pipeline, but the heaps are only set once per frame.
	renderContext->SetDescriptorHeaps();

	if (evaluated)
	{
		reset_ = false;
	}
	else if (!reportedFailure_)
	{
		loge("DLSS Ray Reconstruction failed to run (see the NGX log above); the output will be stale.");
		reportedFailure_ = true;
	}

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
