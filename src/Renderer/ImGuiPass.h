#pragma once

#include "RenderPass.h"

#include <cstdint>

class Camera;
class PathTracerPass;
class RenderContext;
class RenderGPUScene;
struct RenderSettings;

namespace rg
{
	class RenderGraph;
}

// The settings UI and the stats window. BuildUI runs at the start of the frame, before anything reads the settings,
// so that they hold still while the frame is recorded and while the graph runs its callbacks later on; the pass
// only draws what BuildUI made over the back buffer.
class ImGuiPass final : public RenderPass
{
public:
	// rayReconstructionSupported greys out the DLSS controls when NGX reports no DLSS-RR. The path tracer reports
	// the reference's progress, and the graph its passes' GPU times.
	ImGuiPass(RenderSettings &settings,
			  const Camera &camera,
			  bool rayReconstructionSupported,
			  const RenderGPUScene &scene,
			  const PathTracerPass &pathTracer,
			  const rg::RenderGraph &graph);

	// Starts the ImGui frame and builds the windows, which may change the settings.
	void BuildUI(RenderContext *renderContext);

	void AddPasses(FrameGraph &frame) override;

private:
	// The frame rate, memory and pass timings, in a window of their own.
	void BuildStatsWindow(RenderContext *renderContext);

	RenderSettings &settings_;
	const Camera &camera_;
	const bool rayReconstructionSupported_;
	const RenderGPUScene &scene_;
	const PathTracerPass &pathTracer_;
	const rg::RenderGraph &graph_;
};
