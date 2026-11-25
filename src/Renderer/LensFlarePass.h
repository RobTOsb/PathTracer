#pragma once

#include "RenderPass.h"

#include <vector>

#include "PathTracerInterop.h"
#include "PathTracerPass.h"
#include "RenderContext.h"
#include "RenderGraph.h"
#include "RenderSettings.h"

class BloomPass;
class Camera;
class RenderGPUScene;

class LensFlarePass final : public RenderPass
{
public:
	// exposure: AutoExposurePass's exposure buffer.
	LensFlarePass(RenderContext *renderContext,
				  const RenderGPUScene &scene,
				  const BloomPass &bloom,
				  ResourceHandle exposure,
				  const Camera &camera,
				  const RenderSettings &settings);
	~LensFlarePass() override;

	void AddPasses(FrameGraph &frame) override;

	// What PathTracerPass::SetFlareVisibility needs.
	PathTracerPass::FlareVisibility GetVisibility() const;

	// The lenses LensFlareSettings::lens picks from, for the UI.
	static int GetLensCount();
	static const char *GetLensName(int lens);

private:
	// The lens at rest: its surfaces at each channel's wavelength, ghosts and first-order (paraxial) properties.
	void BuildLens(int lens);
	void BuildCandidates();
	// The starburst texture for an iris of this many blades (0 = round).
	void BuildStarburst(int blades);
	PipelineHandle CreateGraphicsPipeline(const char *vertexEntry, const char *pixelEntry) const;

	RenderContext *renderContext_ = nullptr;
	const RenderGPUScene &scene_;
	const BloomPass &bloom_;
	ResourceHandle exposure_ = InvalidResourceHandle;
	const Camera &camera_;
	const RenderSettings &settings_;

	PipelineHandle selectPipeline_ = InvalidPipelineHandle;
	PipelineHandle tracePipeline_ = InvalidPipelineHandle;
	PipelineHandle compositePipeline_ = InvalidPipelineHandle;
	PipelineHandle ghostPipeline_ = InvalidPipelineHandle;
	PipelineHandle starburstPipeline_ = InvalidPipelineHandle;
	CommandSignatureHandle drawSignature_ = InvalidCommandSignatureHandle;

	// The lens.
	int lens_ = -1; // LensFlareSettings::lens it was built for
	std::vector<GpuFlareSurface> surfaces_; // one prescription, at the d line
	std::vector<GpuFlareGhost> ghosts_;
	uint32_t stopSurface_ = 0;
	float focalLength_ = 0.0f; // mm
	float sensorZ_ = 0.0f; // the back focal plane
	float entranceZ_ = 0.0f; // the entrance pupil's plane
	float entranceRadius_ = 0.0f; // the ray grid's half side
	float openFNumber_ = 0.0f; // the f-number with the stop wide open
	float openStopRadius_ = 0.0f;

	// Static buffers.
	ResourceHandle surfaceBuffer_ = InvalidResourceHandle;
	ResourceHandle ghostBuffer_ = InvalidResourceHandle;
	ResourceHandle candidates_ = InvalidResourceHandle;
	ResourceHandle samples_ = InvalidResourceHandle;
	uint32_t candidateCount_ = 0;

	// Persistent buffers, imported into the graph.
	ResourceHandle results_ = InvalidResourceHandle;
	ResourceHandle lights_ = InvalidResourceHandle;
	ResourceHandle args_ = InvalidResourceHandle;
	ResourceHandle ghostVertices_ = InvalidResourceHandle;
	ResourceHandle ghostApertures_ = InvalidResourceHandle;
	ResourceHandle quads_ = InvalidResourceHandle;
	uint32_t quadCapacity_ = 0;

	std::vector<ResourceHandle> constantBuffers_; // one per frame in flight

	ResourceHandle starburst_ = InvalidResourceHandle;
	int starburstBlades_ = -1;
	struct Retired
	{
		ResourceHandle resource;
		uint64_t frame;
	};
	std::vector<Retired> retired_;
	uint64_t frameCount_ = 0;

	uint width_ = 0;
	uint height_ = 0;
	uint flareWidth_ = 0;
	uint flareHeight_ = 0;
};
