#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include <chrono>

struct RenderSettings;

class AutoExposurePass final : public RenderPass
{
public:
	AutoExposurePass(RenderContext *renderContext, HdrSource hdrInput, const RenderSettings &settings);
	~AutoExposurePass() override;

	void AddPasses(FrameGraph &frame) override;

	ResourceHandle GetExposure() const
	{
		return exposure_;
	}

	// Copies of the exposure for the path tracer, by frame parity: frame f's copy pass writes [f & 1], which the
	// path tracer reads before it in frame f + 2. So the path tracer, on the async compute queue, never waits for
	// this frame's or the last frame's graphics work to know the exposure.
	ResourceHandle GetTracingExposure(uint32_t parity) const
	{
		return tracingExposure_[parity & 1];
	}

private:
	RenderContext *renderContext_ = nullptr;
	HdrSource hdrInput_;
	const RenderSettings &settings_;

	PipelineHandle histogramPipeline_ = InvalidPipelineHandle;
	PipelineHandle exposurePipeline_ = InvalidPipelineHandle;
	ResourceHandle histogram_ = InvalidResourceHandle; // uint per bin
	ResourceHandle keySceneLuminance_ = InvalidResourceHandle; // one float
	ResourceHandle exposure_ = InvalidResourceHandle; // one float
	ResourceHandle tracingExposure_[2] = { InvalidResourceHandle, InvalidResourceHandle };
	uint width_ = 0;
	uint height_ = 0;

	// The adaptation starts over at the first frame with auto exposure on.
	bool adapting_ = false;
	std::chrono::steady_clock::time_point lastTime_;
};
