#include "TonemapPass.h"

#include "RenderSettings.h"

#include <math.h>
#include <stdexcept>

namespace
{
	constexpr uint32_t kGroupSize = 8;
	constexpr float kPaperWhiteNits = 203.0f; // BT.2408 reference white for HDR10
}

TonemapPass::TonemapPass(RenderContext *renderContext, ResourceHandle hdrInput, const RenderSettings &settings) :
	renderContext_(renderContext), hdrInput_(hdrInput), settings_(settings)
{
	pipeline_ = renderContext->CreateComputePipeline(PATHTRACER_SHADER_DIR "Tonemap.slang", "main");
	if (pipeline_ == InvalidPipelineHandle)
	{
		throw std::runtime_error("Failed to create the tonemap pipeline (see the shader log above).");
	}

	// Same size and format as the back buffer so it can be copied over directly (swapchain
	// buffers cannot be bound as UAVs).
	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();

	TextureDesc desc = {};
	desc.width = width_;
	desc.height = height_;
	desc.format = renderContext->GetBackBufferFormat();
	desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	desc.debugName = "Tonemap_LDR";
	ldrOutput_ = renderContext->CreateTexture(desc, ResourceViewType::UnorderedAccess);
}

TonemapPass::~TonemapPass()
{
	renderContext_->DestroyResource(ldrOutput_);
}

void TonemapPass::Execute(RenderContext *renderContext)
{
	renderContext->BeginEvent("Tonemap");
	renderContext->BeginProfileScope("Tonemap");

	renderContext->TransitionResource(hdrInput_,
									  ResourceLayout::UnorderedAccess,
									  ResourceSync::ComputeShading,
									  ResourceAccess::UnorderedAccess);
	renderContext->TransitionResource(ldrOutput_,
									  ResourceLayout::UnorderedAccess,
									  ResourceSync::ComputeShading,
									  ResourceAccess::UnorderedAccess);

	TonemapConstants constants = {};
	constants.exposure = exp2f(settings_.exposureEV);
	constants.toneMapOperator = static_cast<uint>(settings_.toneMapOperator);
	constants.width = width_;
	constants.height = height_;
	constants.hdr10 = renderContext->GetUseHDR() ? 1u : 0u;
	constants.paperWhiteNits = kPaperWhiteNits;

	renderContext->SetComputePipeline(pipeline_);
	renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
	renderContext->SetComputeUAV(0, hdrInput_);
	renderContext->SetComputeUAV(1, ldrOutput_);
	renderContext->Dispatch((width_ + kGroupSize - 1) / kGroupSize, (height_ + kGroupSize - 1) / kGroupSize, 1);

	const ResourceHandle backBuffer = renderContext->GetCurrentBackBufferHandle();
	renderContext->TransitionResource(ldrOutput_, ResourceLayout::CopySource, ResourceSync::Copy, ResourceAccess::CopySource);
	renderContext->TransitionResource(backBuffer, ResourceLayout::CopyDest, ResourceSync::Copy, ResourceAccess::CopyDest);
	renderContext->CopyTexture(backBuffer, ldrOutput_);

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
