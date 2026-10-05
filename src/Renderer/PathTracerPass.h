#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include "Maths.h"

#include <vector>

class Camera;
class RenderGPUScene;
struct RenderSettings;

// Render-resolution guide buffers written alongside the radiance for DLSS Ray Reconstruction.
struct PathTracerGuides
{
	ResourceHandle diffuseAlbedo = InvalidResourceHandle; // RGBA8
	ResourceHandle specularAlbedo = InvalidResourceHandle; // RGBA16F
	ResourceHandle normalRoughness = InvalidResourceHandle; // RGBA16F, world-space normal + linear roughness in .w
	ResourceHandle motionVectors = InvalidResourceHandle; // RG16F, render pixels
	ResourceHandle depth = InvalidResourceHandle; // R32F, reversed-Z hardware depth
	ResourceHandle specularHitDistance = InvalidResourceHandle; // R16F, world units
};

// Traces one path per pixel into an HDR radiance target (RGBA16F) using the DXR pipeline in
// PathTracer.slang, at the render context's render resolution. With rayReconstruction set it
// also writes PathTracerGuides and jitters the camera along the DLSS Halton sequence instead of
// randomly. Leaves the output and the guides in the UnorderedAccess layout.
class PathTracerPass final : public RenderPass
{
public:
	PathTracerPass(RenderContext *renderContext,
				   const RenderGPUScene &scene,
				   const Camera &camera,
				   const RenderSettings &settings,
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
	ResourceHandle CreateTarget(Format format, const char *debugName) const;

	RenderContext *renderContext_ = nullptr;
	const RenderGPUScene &scene_;
	const Camera &camera_;
	const RenderSettings &settings_;
	const bool rayReconstruction_;

	PipelineHandle pipeline_ = InvalidPipelineHandle;
	ShaderTableHandle shaderTable_ = InvalidShaderTableHandle;
	ResourceHandle output_ = InvalidResourceHandle;
	PathTracerGuides guides_;
	std::vector<ResourceHandle> frameConstants_; // one per frame in flight, CPU-written each frame
	uint width_ = 0;
	uint height_ = 0;

	uint jitterIndex_ = 0;
	float2 jitter_ = { 0.0f, 0.0f };
};
