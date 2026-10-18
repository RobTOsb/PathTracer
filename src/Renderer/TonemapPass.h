#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include <chrono>

struct RenderSettings;

// Exposes, tone maps and sRGB-encodes the HDR radiance into an LDR target in a compute pass, then
// copies it onto the current back buffer. Leaves the back buffer in the CopyDest layout.
class TonemapPass final : public RenderPass
{
public:
	TonemapPass(RenderContext *renderContext, ResourceHandle hdrInput, const RenderSettings &settings);
	~TonemapPass() override;

	void Execute(RenderContext *renderContext) override;

private:
	RenderContext *renderContext_ = nullptr;
	ResourceHandle hdrInput_ = InvalidResourceHandle;
	const RenderSettings &settings_;

	PipelineHandle pipeline_ = InvalidPipelineHandle;
	ResourceHandle ldrOutput_ = InvalidResourceHandle;
	uint width_ = 0;
	uint height_ = 0;

	std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now(); // animates the film grain
};
