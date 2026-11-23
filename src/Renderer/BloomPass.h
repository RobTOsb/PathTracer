#pragma once

#include "RenderPass.h"

#include <utility>
#include <vector>

#include "RenderContext.h"
#include "RenderGraph.h"
#include "RenderSettings.h"

class Camera;
struct BloomConstants;

// Blends bloom into the HDR image in compute passes (Bloom.slang): the image is filtered down into a
// chain of half-resolution levels, the levels are summed back up into the first, and their mean
// replaces RenderSettings::bloomStrength of the image in a copy the size of the window. With bloom off
// it only copies the image. Without RenderSettings::bloomSun the first level leaves out the pixels that
// look at the sun disc. The levels and the output are graph transients, one graph pass per dispatch.
class BloomPass final : public RenderPass
{
public:
	// exposure: AutoExposurePass's exposure buffer.
	BloomPass(RenderContext *renderContext,
			  HdrSource hdrInput,
			  ResourceHandle exposure,
			  const Camera &camera,
			  const RenderSettings &settings);
	~BloomPass() override;

	void AddPasses(FrameGraph &frame) override;

	// This frame's output (RGBA16F, the window's size), once AddPasses has run.
	rg::ResourceId GetOutput() const
	{
		return output_;
	}

private:
	struct Level
	{
		uint width = 0;
		uint height = 0;
	};

	PipelineHandle CreatePipeline(const char *entryPoint) const;
	void SetSunMask(BloomConstants &constants) const;

	RenderContext *renderContext_ = nullptr;
	HdrSource hdrInput_;
	ResourceHandle exposure_ = InvalidResourceHandle;
	const Camera &camera_;
	const RenderSettings &settings_;

	PipelineHandle downsamplePipeline_ = InvalidPipelineHandle;
	PipelineHandle upsamplePipeline_ = InvalidPipelineHandle;
	PipelineHandle compositePipeline_ = InvalidPipelineHandle;

	Level levels_[kMaxBloomLevels];
	int levelCount_ = 0; // levels that fit, fewer than kMaxBloomLevels for a small window
	rg::ResourceId output_;
	uint width_ = 0;
	uint height_ = 0;
};
