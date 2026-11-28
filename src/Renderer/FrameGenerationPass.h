#pragma once

#include "RenderContext.h"

class FrameGraph;
class PathTracerPass;
class Streamline;

class FrameGenerationPass
{
public:
	FrameGenerationPass(RenderContext *renderContext, Streamline &streamline, const PathTracerPass &pathTracer);

	void AddPasses(FrameGraph &frame);

private:
	Streamline &streamline_;
	const PathTracerPass &pathTracer_;

	PipelineHandle compositePipeline_ = InvalidPipelineHandle;
	Format format_ = Format::Unknown; // the back buffer's
	uint32_t width_ = 0;
	uint32_t height_ = 0;
	bool reportedFailure_ = false;
};
