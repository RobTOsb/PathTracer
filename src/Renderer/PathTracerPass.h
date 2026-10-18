#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include "Maths.h"

#include <vector>

class Camera;
class RenderGPUScene;
class SkyAtmospherePass;
struct RenderSettings;

// Render-resolution guide buffers written alongside the radiance for DLSS Ray Reconstruction.
struct PathTracerGuides
{
	ResourceHandle diffuseAlbedo = InvalidResourceHandle; // RGBA8
	ResourceHandle specularAlbedo = InvalidResourceHandle; // RGBA16F
	ResourceHandle normalRoughness = InvalidResourceHandle; // RGBA16F, world-space normal + linear roughness in .w
	ResourceHandle motionVectors = InvalidResourceHandle; // RG16F, render pixels
	ResourceHandle depth = InvalidResourceHandle; // R32F, reversed-Z hardware depth
	ResourceHandle specularHitDistance = InvalidResourceHandle;
	ResourceHandle specularMotionVectors = InvalidResourceHandle; // RG16F
};

// Traces one path per pixel into an HDR radiance target (RGBA16F) using the DXR pipeline in
// PathTracer.slang, at the render context's render resolution. With rayReconstruction set it
// also writes PathTracerGuides and jitters the camera along the DLSS Halton sequence instead of
// randomly. With RenderSettings::shaderExecutionReordering on a device with DXR 1.2 it traces with
// the SER pipeline instead. While the sky atmosphere is on, it traces with the ATMOSPHERE permutation,
// which takes its sky and sunlight from the atmosphere's LUTs and scatters sunlight in the scene's air.
// With RenderSettings::opacityMicromaps on a scene that has them, it traces the micromap TLAS with the
// OPACITY_MICROMAPS permutation.
// Leaves the output and the guides in the UnorderedAccess layout.
class PathTracerPass final : public RenderPass
{
public:
	PathTracerPass(RenderContext *renderContext,
				   const RenderGPUScene &scene,
				   const Camera &camera,
				   const RenderSettings &settings,
				   const SkyAtmospherePass &atmosphere,
				   bool rayReconstruction);
	~PathTracerPass() override;

	void Execute(RenderContext *renderContext) override;

	ResourceHandle GetOutput() const
	{
		return output_;
	}

	const PathTracerGuides &GetGuides() const
	{
		return guides_;
	}

	// The sub-pixel offset of this frame's camera rays from the pixel centre, in render pixels
	// with +y down. Zero without ray reconstruction.
	float2 GetJitter() const
	{
		return jitter_;
	}

	uint GetWidth() const
	{
		return width_;
	}

	uint GetHeight() const
	{
		return height_;
	}

private:
	struct Pipeline
	{
		PipelineHandle pipeline = InvalidPipelineHandle;
		ShaderTableHandle shaderTable = InvalidShaderTableHandle;
		bool opacityMicromaps = false; // traces the scene's micromap TLAS
	};

	const Pipeline *SelectPipeline();
	const Pipeline *GetPipeline(bool shaderExecutionReordering, bool atmosphere, bool inlineShadows, bool opacityMicromaps);
	Pipeline CreatePipeline(bool shaderExecutionReordering, bool atmosphere, bool inlineShadows, bool opacityMicromaps) const;
	ResourceHandle CreateTarget(Format format, const char *debugName) const;

	RenderContext *renderContext_ = nullptr;
	const RenderGPUScene &scene_;
	const Camera &camera_;
	const RenderSettings &settings_;
	const SkyAtmospherePass &atmosphere_;
	const bool rayReconstruction_;

	Pipeline pipelines_[2][2][2][2]; // indexed by shaderExecutionReordering, atmosphere, inline shadows, then opacity micromaps
	bool serSupported_ = false;
	bool serFailed_ = false; // the SER pipeline did not compile, so stay on the plain one
	bool ommFailed_ = false; // the opacity micromap pipeline did not compile, so trace without them
	ResourceHandle output_ = InvalidResourceHandle;
	PathTracerGuides guides_;
	ResourceHandle cameraState_ = InvalidResourceHandle; // GpuCameraState per pixel, DLSS-RR only
	std::vector<ResourceHandle> frameConstants_; // one per frame in flight, CPU-written each frame
	uint width_ = 0;
	uint height_ = 0;

	uint jitterIndex_ = 0;
	float2 jitter_ = { 0.0f, 0.0f };
};
