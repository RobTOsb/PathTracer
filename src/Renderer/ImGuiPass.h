#pragma once

#include "RenderPass.h"

#include <chrono>
#include <cstdint>
#include <memory>

class Camera;
class FrameGenerationPass;
class PathTracerPass;
class Streamline;
class RenderContext;
class RenderGPUScene;
struct RenderSettings;

namespace rg
{
	class RenderGraph;
}

// The settings UI and the stats window. BuildUI runs at the start of the frame, before anything reads the settings,
// so that they hold still while the frame is recorded and while the graph runs its callbacks later on; the pass
// only draws what BuildUI made over the back buffer, or, while DLSS Frame Generation is on, leaves that to its
// FrameGenerationPass.
class ImGuiPass final : public RenderPass
{
public:
	// Streamline's support greys out the DLSS controls it has no feature for, and its frame generation state is shown.
	// frameGeneration, null without DLSS-RR, draws the UI while frame generation is on. The path tracer reports the
	// reference's progress, and the graph its passes' GPU times.
	ImGuiPass(RenderSettings &settings,
			  const Camera &camera,
			  const Streamline &streamline,
			  std::unique_ptr<FrameGenerationPass> frameGeneration,
			  const RenderGPUScene &scene,
			  const PathTracerPass &pathTracer,
			  const rg::RenderGraph &graph);
	~ImGuiPass() override;

	// Starts the ImGui frame and builds the windows, which may change the settings.
	void BuildUI(RenderContext *renderContext);

	void AddPasses(FrameGraph &frame) override;

private:
	// The frame rate, memory and pass timings, in a window of their own.
	void BuildStatsWindow(RenderContext *renderContext);
	// The "DLSS Frame Generation" header: frame generation and Reflex.
	void BuildFrameGenerationUI(bool reference);
	// Counts this frame into the rendered and output frame rates.
	void UpdateFrameRates();

	RenderSettings &settings_;
	const Camera &camera_;
	const Streamline &streamline_;
	const bool rayReconstructionSupported_;
	std::unique_ptr<FrameGenerationPass> frameGeneration_;

	// Frames rendered and presented (generated ones included) over the current half second, and the rates the last
	// one measured.
	std::chrono::steady_clock::time_point rateWindowStart_ = std::chrono::steady_clock::now();
	uint32_t renderedInWindow_ = 0;
	uint32_t presentedInWindow_ = 0;
	float renderedFps_ = 0.0f;
	float outputFps_ = 0.0f;
	const RenderGPUScene &scene_;
	const PathTracerPass &pathTracer_;
	const rg::RenderGraph &graph_;
};
