#pragma once

#include "RenderPass.h"

#include "RenderContext.h"
#include "Streamline.h"

class Camera;
class PathTracerPass;
struct RenderSettings;

class DLSSPass final : public RenderPass
{
public:
	DLSSPass(RenderContext *renderContext,
			 Streamline &streamline,
			 const PathTracerPass &pathTracer,
			 const Camera &camera,
			 const RenderSettings &settings);
	~DLSSPass() override;

	void AddPasses(FrameGraph &frame) override;

	// This frame's output, a graph transient, once AddPasses has run.
	rg::ResourceId GetOutput() const
	{
		return output_;
	}

private:
	RenderContext *renderContext_ = nullptr;
	Streamline &streamline_;
	const PathTracerPass &pathTracer_;
	const Camera &camera_;
	const RenderSettings &settings_;

	// main rebuilds the pass when the mode, preset or sizes change.
	DLSSMode mode_ = DLSSMode::Off;
	DLSSRayReconstructionPreset preset_ = DLSSRayReconstructionPreset::Default;
	rg::ResourceId output_;
	TextureDesc outputDesc_;
	bool reset_ = true; // no history yet
	bool reportedFailure_ = false;
};
