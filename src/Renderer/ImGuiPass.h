#pragma once

#include "RenderPass.h"

#include <cstdint>

class Camera;
struct RenderSettings;

class ImGuiPass final : public RenderPass
{
public:
	// rayReconstructionSupported greys out the DLSS controls when NGX reports no DLSS-RR.
	ImGuiPass(RenderSettings &settings, const Camera &camera, bool rayReconstructionSupported, uint32_t lightCount);

	void Execute(RenderContext *renderContext) override;

private:
	void BuildUI(RenderContext *renderContext);

	RenderSettings &settings_;
	const Camera &camera_;
	const bool rayReconstructionSupported_;
	const uint32_t lightCount_;
};
