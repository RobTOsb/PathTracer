#pragma once

#include "RenderPass.h"

#include "DLSSCommon.h"
#include "RenderContext.h"

class Camera;
class NGXContext;
class PathTracerPass;

class DLSSPass final : public RenderPass
{
public:
	DLSSPass(RenderContext *renderContext,
			 NGXContext &ngx,
			 const PathTracerPass &pathTracer,
			 const Camera &camera,
			 DLSSMode mode,
			 DLSSRayReconstructionPreset preset);
	~DLSSPass() override;

	void Execute(RenderContext *renderContext) override;

	ResourceHandle GetOutput() const
	{
		return output_;
	}

private:
	RenderContext *renderContext_ = nullptr;
	NGXContext &ngx_;
	const PathTracerPass &pathTracer_;
	const Camera &camera_;

	DLSSFeatureParams featureParams_;
	ResourceHandle output_ = InvalidResourceHandle;
	bool reset_ = true; // no history yet
	bool reportedFailure_ = false;
};
