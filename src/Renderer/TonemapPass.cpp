#include "TonemapPass.h"

#include "BloomPass.h"
#include "FilmLut.h"
#include "FrameGraph.h"
#include "PathTracerInterop.h"
#include "RenderSettings.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <math.h>
#include <span>
#include <stdexcept>

namespace
{
	constexpr uint32_t kGroupSize = 8;
}

TonemapPass::TonemapPass(RenderContext *renderContext,
						 const BloomPass &bloom,
						 ResourceHandle exposure,
						 const RenderSettings &settings) :
	renderContext_(renderContext), bloom_(bloom), exposure_(exposure), settings_(settings)
{
	pipeline_ = renderContext->CreateComputePipeline(PATHTRACER_SHADER_DIR "Tonemap.slang", "main");
	if (pipeline_ == InvalidPipelineHandle)
	{
		throw std::runtime_error("Failed to create the tonemap pipeline (see the shader log above).");
	}

	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();
	format_ = renderContext->GetBackBufferFormat();

	// 2^3 identity: the corners of the RGB cube map to themselves (halves 0 and 1).
	std::array<uint16_t, 8 * 4> identity = {};
	for (uint i = 0; i < 8; ++i)
	{
		identity[i * 4 + 0] = (i & 1) ? 0x3C00 : 0;
		identity[i * 4 + 1] = (i & 2) ? 0x3C00 : 0;
		identity[i * 4 + 2] = (i & 4) ? 0x3C00 : 0;
		identity[i * 4 + 3] = 0x3C00;
	}
	identityLut_ = CreateFilmLutTexture(2, identity.data(), "Tonemap_IdentityLut");
	filmLuts_.resize(GetFilmLutFiles().size());
}

TonemapPass::~TonemapPass()
{
	renderContext_->DestroyResource(identityLut_.texture);
	for (const FilmLutTexture &lut: filmLuts_)
	{
		if (lut.texture != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(lut.texture);
		}
	}
}

TonemapPass::FilmLutTexture TonemapPass::CreateFilmLutTexture(uint size, const uint16_t *texels, const char *name)
{
	TextureDesc desc = {};
	desc.dimension = TextureDimension::Texture3D;
	desc.width = size;
	desc.height = size;
	desc.depthOrArraySize = size;
	desc.format = Format::R16G16B16A16_FLOAT;
	desc.debugName = name;
	TextureMipData mip = {};
	mip.pixels = texels;
	mip.rowPitch = size * 4 * sizeof(uint16_t);
	mip.rowCount = size;
	mip.depth = size;
	const UploadBatchHandle batch = renderContext_->BeginUpload();
	FilmLutTexture lut;
	lut.texture = renderContext_->CreateTexture(desc,
												ResourceViewType::ShaderResource,
												std::span<const TextureMipData>(&mip, 1),
												batch);
	renderContext_->EndUpload(batch);
	lut.size = size;
	return lut;
}

const TonemapPass::FilmLutTexture &TonemapPass::GetFilmLut(int index)
{
	const std::vector<FilmLutFile> &files = GetFilmLutFiles();
	if (index < 1 || index > static_cast<int>(files.size()))
	{
		return identityLut_;
	}
	FilmLutTexture &lut = filmLuts_[index - 1];
	if (lut.texture == InvalidResourceHandle && !lut.failed)
	{
		try
		{
			const FilmLutData data = LoadFilmLut(files[index - 1].path);
			lut = CreateFilmLutTexture(data.size, data.texels.data(), "Tonemap_FilmLut");
		}
		catch (const std::exception &error)
		{
			printf("%s\n", error.what());
			lut.failed = true;
		}
	}
	return lut.failed ? identityLut_ : lut;
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
	constants.vignettePower = settings_.vignette ? std::max(settings_.vignetteStrength, 0.0f) : 0.0f;
	const FilmLutTexture &filmLut = GetFilmLut(settings_.filmLut);
	const ResourceHandle filmLutTexture = filmLut.texture;
	constants.filmLutStrength = filmLut.texture != identityLut_.texture ? std::clamp(settings_.filmLutStrength, 0.0f, 1.0f) : 0.0f;
	constants.filmLutSize = static_cast<float>(filmLut.size);
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
										[this, constants, hdr, ldr, exposure, filmLutTexture](rg::PassContext &ctx)
										{
											ctx.rc.SetComputePipeline(pipeline_);
											ctx.rc.SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
											ctx.rc.SetComputeUAV(0, ctx.Get(hdr));
											ctx.rc.SetComputeUAV(1, ctx.Get(ldr));
											ctx.rc.SetComputeUAV(2, ctx.Get(exposure));
											ctx.rc.SetComputeSRVByName("g_FilmLut", filmLutTexture);
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
