#include "TonemapPass.h"

#include "RenderSettings.h"

#include <chrono>
#include <math.h>
#include <stdexcept>

namespace
{
	constexpr uint32_t kGroupSize = 8;
}

TonemapPass::TonemapPass(RenderContext *renderContext,
						 ResourceHandle hdrInput,
						 ResourceHandle exposure,
						 const RenderSettings &settings) :
	renderContext_(renderContext), hdrInput_(hdrInput), exposure_(exposure), settings_(settings)
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
	constants.toneMapOperator = static_cast<uint>(settings_.toneMapOperator);
	constants.width = width_;
	constants.height = height_;
	constants.hdr10 = renderContext->GetUseHDR() ? 1u : 0u;
	constants.paperWhiteNits = settings_.paperWhiteNits;
	constants.peakNits = settings_.peakNits;
	constants.whiteClip = settings_.whiteClip;
	constants.hdrGammaCorrection = settings_.hdrGammaCorrection ? 1u : 0u;
	constants.grainStrength = settings_.filmGrain ? settings_.filmGrainStrength : 0.0f;
	constants.grainSize = settings_.filmGrainSize;
	constants.grainColour = settings_.filmGrainColour;
	constants.grainResponse = settings_.filmGrainResponse;
	if (settings_.filmGrainFps > 0.0f)
	{
		// Wall-clock time so the grain changes at the chosen rate whatever the frame rate.
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime_).count();
		constants.grainSeed = static_cast<uint>(static_cast<uint64_t>(seconds * settings_.filmGrainFps));
	}

	renderContext->SetComputePipeline(pipeline_);
	renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
	renderContext->SetComputeUAV(0, hdrInput_);
	renderContext->SetComputeUAV(1, ldrOutput_);
	renderContext->SetComputeUAV(2, exposure_);
	renderContext->Dispatch((width_ + kGroupSize - 1) / kGroupSize, (height_ + kGroupSize - 1) / kGroupSize, 1);

	const ResourceHandle backBuffer = renderContext->GetCurrentBackBufferHandle();
	renderContext->TransitionResource(ldrOutput_, ResourceLayout::CopySource, ResourceSync::Copy, ResourceAccess::CopySource);
	renderContext->TransitionResource(backBuffer, ResourceLayout::CopyDest, ResourceSync::Copy, ResourceAccess::CopyDest);
	renderContext->CopyTexture(backBuffer, ldrOutput_);

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
