#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include "Maths.h"
#include "PathTracerInterop.h"

class PathTracerPass;
struct RenderSettings;

// Debug view of the DLSS-RR guides: while RenderSettings::guideView picks one, draws it over the back buffer
// after TonemapPass, nearest-upscaled from render resolution, or with RenderSettings::guideViewChange how much it
// changed since last frame at the point the motion vectors lead to. Keeps the guide in a history at render
// resolution for that. Does nothing while guideView is off. Leaves the back buffer in the CopyDest layout.
class GuideViewPass final : public RenderPass
{
public:
	GuideViewPass(RenderContext *renderContext, const PathTracerPass &pathTracer, const RenderSettings &settings);
	~GuideViewPass() override;

	void Execute(RenderContext *renderContext) override;

private:
	RenderContext *renderContext_ = nullptr;
	const PathTracerPass &pathTracer_;
	const RenderSettings &settings_;

	PipelineHandle pipeline_ = InvalidPipelineHandle;
	ResourceHandle history_[2] = { InvalidResourceHandle, InvalidResourceHandle }; // RGBA16F, alternate frames
	ResourceHandle output_ = InvalidResourceHandle; // back buffer format, copied onto it
	uint historyIndex_ = 0; // the one written this frame
	uint width_ = 0;
	uint height_ = 0;

	GuideView lastGuide_ = GUIDE_VIEW_OFF; // what last frame's history holds
	float2 lastJitter_ = { 0.0f, 0.0f };
};
