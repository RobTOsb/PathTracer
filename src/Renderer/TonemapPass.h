#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include <chrono>

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

	void AddPasses(FrameGraph &frame) override;

private:
	const BloomPass &bloom_;
	ResourceHandle exposure_ = InvalidResourceHandle;
	const RenderSettings &settings_;

	PipelineHandle pipeline_ = InvalidPipelineHandle;
	Format format_ = Format::Unknown;
	uint width_ = 0;
	uint height_ = 0;

	std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now(); // animates the film grain
};
