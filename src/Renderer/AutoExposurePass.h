#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include <chrono>

struct RenderSettings;

class AutoExposurePass final : public RenderPass
{
public:
	AutoExposurePass(RenderContext *renderContext, ResourceHandle hdrInput, const RenderSettings &settings);
	~AutoExposurePass() override;

	void Execute(RenderContext *renderContext) override;

	ResourceHandle GetExposure() const
	{
		return exposure_;
	}

private:
	RenderContext *renderContext_ = nullptr;
	ResourceHandle hdrInput_ = InvalidResourceHandle;
	const RenderSettings &settings_;

	PipelineHandle histogramPipeline_ = InvalidPipelineHandle;
	PipelineHandle exposurePipeline_ = InvalidPipelineHandle;
	ResourceHandle histogram_ = InvalidResourceHandle; // uint per bin
	ResourceHandle keySceneLuminance_ = InvalidResourceHandle; // one float
	ResourceHandle exposure_ = InvalidResourceHandle; // one float
	uint width_ = 0;
	uint height_ = 0;

	// The adaptation starts over at the first frame with auto exposure on.
	bool adapting_ = false;
	std::chrono::steady_clock::time_point lastTime_;
};
