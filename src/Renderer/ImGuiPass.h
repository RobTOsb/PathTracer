#pragma once

#include "RenderPass.h"

class Camera;
struct RenderSettings;

// Builds the settings UI and draws ImGui on top of the back buffer. Edits to RenderSettings take
// effect from the next frame, since the passes before this one have already been recorded.
// Leaves the back buffer in the RenderTarget layout.
class ImGuiPass final : public RenderPass
{
public:
	ImGuiPass(RenderSettings &settings, const Camera &camera);

	void Execute(RenderContext *renderContext) override;

private:
	void BuildUI(RenderContext *renderContext);

	RenderSettings &settings_;
	const Camera &camera_;
};
