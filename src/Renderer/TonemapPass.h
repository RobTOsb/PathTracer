#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include <chrono>
#include <vector>

class BloomPass;
struct RenderSettings;

// Exposes, tone maps and sRGB-encodes the HDR radiance into an LDR target in a compute pass, then
// copies it onto the current back buffer. The LDR target is a graph transient the size and format of
// the back buffer, since swapchain buffers cannot be bound as UAVs.
class TonemapPass final : public RenderPass
{
public:
	// The HDR image is bloom's output. exposure: AutoExposurePass's exposure buffer.
	TonemapPass(RenderContext *renderContext, const BloomPass &bloom, ResourceHandle exposure, const RenderSettings &settings);
	~TonemapPass() override;

	void AddPasses(FrameGraph &frame) override;

private:
	struct FilmLutTexture
	{
		ResourceHandle texture = InvalidResourceHandle;
		uint size = 0;
		bool failed = false; // the file did not load; logged once, then treated as no LUT
	};

	FilmLutTexture CreateFilmLutTexture(uint size, const uint16_t *texels, const char *name);
	// The texture for RenderSettings::filmLut, loaded on first use; the identity LUT when none or it failed to load.
	const FilmLutTexture &GetFilmLut(int index);

	RenderContext *renderContext_ = nullptr;
	const BloomPass &bloom_;
	ResourceHandle exposure_ = InvalidResourceHandle;
	const RenderSettings &settings_;

	PipelineHandle pipeline_ = InvalidPipelineHandle;
	Format format_ = Format::Unknown;
	uint width_ = 0;
	uint height_ = 0;

	FilmLutTexture identityLut_; // bound when no LUT is chosen, so the slot is always valid
	std::vector<FilmLutTexture> filmLuts_; // per GetFilmLutFiles() entry

	std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now(); // animates the film grain
};
