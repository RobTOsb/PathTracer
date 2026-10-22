#pragma once

#include "RenderPass.h"

#include "RenderContext.h"
#include "RenderSettings.h"

class Camera;
struct BloomConstants;

// Blends bloom into the HDR image in compute passes (Bloom.slang): the image is filtered down into a
// chain of half-resolution levels, the levels are summed back up into the first, and their mean
// replaces RenderSettings::bloomStrength of the image in a copy the size of the window. With bloom off
// it only copies the image. Without RenderSettings::bloomSun the first level leaves out the pixels that
// look at the sun disc.
class BloomPass final : public RenderPass
{
public:
	BloomPass(RenderContext *renderContext, ResourceHandle hdrInput, const Camera &camera, const RenderSettings &settings);
	~BloomPass() override;

	void Execute(RenderContext *renderContext) override;

	ResourceHandle GetOutput() const
	{
		return output_;
	}

private:
	struct Level
	{
		ResourceHandle texture = InvalidResourceHandle; // RGBA16F, written as a UAV
		ResourceHandle srv = InvalidResourceHandle;
		uint width = 0;
		uint height = 0;
	};

	PipelineHandle CreatePipeline(const char *entryPoint) const;
	void Dispatch(RenderContext *renderContext, const BloomConstants &constants) const;
	void SetSunMask(BloomConstants &constants) const;

	RenderContext *renderContext_ = nullptr;
	ResourceHandle hdrInput_ = InvalidResourceHandle;
	ResourceHandle hdrInputSRV_ = InvalidResourceHandle;
	const Camera &camera_;
	const RenderSettings &settings_;

	PipelineHandle downsamplePipeline_ = InvalidPipelineHandle;
	PipelineHandle upsamplePipeline_ = InvalidPipelineHandle;
	PipelineHandle compositePipeline_ = InvalidPipelineHandle;

	Level levels_[kMaxBloomLevels];
	int levelCount_ = 0; // levels allocated, fewer than kMaxBloomLevels for a small window
	ResourceHandle output_ = InvalidResourceHandle; // RGBA16F, the window's size
	uint width_ = 0;
	uint height_ = 0;
};
