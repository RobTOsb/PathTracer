#include "TonemapPass.h"

#include "BloomPass.h"
#include "FrameGraph.h"
#include "PathTracerInterop.h"
#include "RenderSettings.h"

#include <chrono>
#include <math.h>
#include <stdexcept>

namespace
{
	constexpr uint32_t kGroupSize = 8;
}

TonemapPass::TonemapPass(RenderContext *renderContext,
						 const BloomPass &bloom,
						 ResourceHandle exposure,
						 const RenderSettings &settings) :
	bloom_(bloom), exposure_(exposure), settings_(settings)
{
	pipeline_ = renderContext->CreateComputePipeline(PATHTRACER_SHADER_DIR "Tonemap.slang", "main");
	if (pipeline_ == InvalidPipelineHandle)
	{
		throw std::runtime_error("Failed to create the tonemap pipeline (see the shader log above).");
	}

	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();
	format_ = renderContext->GetBackBufferFormat();
}

void TonemapPass::AddPasses(FrameGraph &frame)
{
	TonemapConstants constants = {};
	constants.toneMapOperator = static_cast<uint>(settings_.toneMapOperator);
	constants.width = width_;
	constants.height = height_;
	constants.hdr10 = frame.rc.GetUseHDR() ? 1u : 0u;
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

	TextureDesc desc = {};
	desc.width = width_;
	desc.height = height_;
	desc.format = format_;
	desc.debugName = "Tonemap_LDR";
	const rg::ResourceId ldr = frame.graph.CreateTexture(desc);
	const rg::ResourceId hdr = bloom_.GetOutput();
	const rg::ResourceId exposure = frame.ImportBuffer(exposure_, "AutoExposure_Exposure");

	// The shader reads its input through a UAV.
	frame.AddPass(rg::Pass::Compute("Tonemap")
								.Use(hdr, rg::Access::ComputeShaderReadWrite)
								.Use(exposure, rg::Access::ComputeShaderReadWrite)
								.Writes(ldr)
								.Executes(
										[this, constants, hdr, ldr, exposure](rg::PassContext &ctx)
										{
											ctx.rc.SetComputePipeline(pipeline_);
											ctx.rc.SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
											ctx.rc.SetComputeUAV(0, ctx.Get(hdr));
											ctx.rc.SetComputeUAV(1, ctx.Get(ldr));
											ctx.rc.SetComputeUAV(2, ctx.Get(exposure));
											ctx.rc.Dispatch((width_ + kGroupSize - 1) / kGroupSize,
															(height_ + kGroupSize - 1) / kGroupSize,
															1);
										}));

	const rg::ResourceId backBuffer = frame.backBuffer;
	frame.AddPass(rg::Pass::Copy("Tonemap/Copy to back buffer")
								.Reads(ldr)
								.Writes(backBuffer)
								.Executes([ldr, backBuffer](rg::PassContext &ctx)
										  { ctx.rc.CopyTexture(ctx.Get(backBuffer), ctx.Get(ldr)); }));
}
