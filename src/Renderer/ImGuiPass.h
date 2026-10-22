#pragma once

#include "RenderPass.h"

#include <cstdint>

class Camera;
class PathTracerPass;
class RenderGPUScene;
struct RenderSettings;

class ImGuiPass final : public RenderPass
{
public:
	// rayReconstructionSupported greys out the DLSS controls when NGX reports no DLSS-RR. The path tracer reports
	// the reference's progress.
	ImGuiPass(RenderSettings &settings,
			  const Camera &camera,
			  bool rayReconstructionSupported,
			  const RenderGPUScene &scene,
			  const PathTracerPass &pathTracer);

	void Execute(RenderContext *renderContext) override;

private:
	void BuildUI(RenderContext *renderContext);

	RenderSettings &settings_;
	const Camera &camera_;
	const bool rayReconstructionSupported_;
	const RenderGPUScene &scene_;
	const PathTracerPass &pathTracer_;
};
