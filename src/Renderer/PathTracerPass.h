#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include <vector>

class Camera;
class RenderGPUScene;
struct RenderSettings;

// Traces one path per pixel into an HDR radiance target (RGBA16F) using the DXR pipeline in
// PathTracer.slang. Leaves the output in the UnorderedAccess layout.
class PathTracerPass final : public RenderPass
{
public:
	PathTracerPass(RenderContext *renderContext,
				   const RenderGPUScene &scene,
				   const Camera &camera,
				   const RenderSettings &settings);
	~PathTracerPass() override;

	void Execute(RenderContext *renderContext) override;

	ResourceHandle GetOutput() const
	{
		return output_;
	}

private:
	RenderContext *renderContext_ = nullptr;
	const RenderGPUScene &scene_;
	const Camera &camera_;
	const RenderSettings &settings_;

	PipelineHandle pipeline_ = InvalidPipelineHandle;
	ShaderTableHandle shaderTable_ = InvalidShaderTableHandle;
	ResourceHandle output_ = InvalidResourceHandle;
	std::vector<ResourceHandle> frameConstants_; // one per frame in flight, CPU-written each frame
	uint width_ = 0;
	uint height_ = 0;
};
