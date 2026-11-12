#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include "Maths.h"
#include "PathTracerInterop.h"
#include "SkyAtmosphereInterop.h"

#include <chrono>
#include <utility>
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
	// R8, 1 where a moving object's shadow changed the light since last frame: DLSS-RR's bias current colour mask.
	ResourceHandle shadowChanges = InvalidResourceHandle;
	// R16F, -1..1: DLSS-RR's responsivity mask (Preset F only), positive on smooth reflectors so their reflections keep
	// less history (FrameConstants::reflectionResponsivity).
	ResourceHandle responsivity = InvalidResourceHandle;
};

// Traces one path per pixel into an HDR radiance target (RGBA16F) using the DXR pipeline in
// PathTracer.slang, at the render context's render resolution. With rayReconstruction set it
// also writes PathTracerGuides and jitters the camera along the DLSS Halton sequence instead of
// randomly. With RenderSettings::shaderExecutionReordering on a device with DXR 1.2 it traces with
// the SER pipeline instead. While the sky atmosphere is on, it traces with the ATMOSPHERE permutation,
// which takes its sky and sunlight from the atmosphere's LUTs and scatters sunlight in the scene's air.
// With RenderSettings::opacityMicromaps on a scene that has them, it traces the micromap TLAS with the
// OPACITY_MICROMAPS permutation. With rayReconstruction and RenderSettings::primarySurfaceReplacement, the
// guides follow mirrors and glass in the PSR permutation. With LightSettings::sgImportance, the light tree ranks its
// nodes by their SG lighting in the SG_LIGHT_TREE permutation.
// With reference set instead (the REFERENCE permutation) it sums RenderSettings::reference.samplesPerFrame
// randomly jittered paths per pixel each frame into an accumulation target and outputs their mean, starting
// over whenever the camera or anything else the image depends on changes, and stops tracing once it has
// RenderSettings::reference.samples.
// With RenderSettings::photons on and a sun (not in the reference), it first adapts the photon map and traces caustic
// photons from it (PhotonRayGen, PathTracer/Photons.slang), then after the paths splats them onto the surfaces the
// camera sees and adds them to the output (PhotonMap.slang): Adaptive Anisotropic Photon Scattering.
// Leaves the output and the guides in the UnorderedAccess layout.
class PathTracerPass final : public RenderPass
{
public:
	PathTracerPass(RenderContext *renderContext,
				   const RenderGPUScene &scene,
				   const Camera &camera,
				   const RenderSettings &settings,
				   const SkyAtmospherePass &atmosphere,
				   bool rayReconstruction,
				   bool reference);
	~PathTracerPass() override;

	void Execute(RenderContext *renderContext) override;

	ResourceHandle GetOutput() const
	{
		return output_;
	}

	void SetExposureBuffer(ResourceHandle exposure)
	{
		exposure_ = exposure;
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

	// GpuCameraState per render pixel, DLSS-RR only.
	ResourceHandle GetCameraState() const
	{
		return cameraState_;
	}

	// This frame's FrameConstants, once Execute has written them.
	ResourceHandle GetFrameConstants() const
	{
		return frameConstants_[renderContext_->GetCurrentFrameIndex()];
	}

	uint GetWidth() const
	{
		return width_;
	}

	uint GetHeight() const
	{
		return height_;
	}

	bool IsReference() const
	{
		return reference_;
	}

	// The reference's paths per pixel so far, and the wall-clock seconds they took.
	uint GetAccumulatedSamples() const
	{
		return accumulatedSamples_;
	}

	float GetAccumulationSeconds() const
	{
		return accumulationSeconds_;
	}

private:
	struct Pipeline
	{
		PipelineHandle pipeline = InvalidPipelineHandle;
		ShaderTableHandle shaderTable = InvalidShaderTableHandle;
		ShaderTableHandle photonShaderTable = InvalidShaderTableHandle; // PhotonRayGen with the same hit groups
		bool opacityMicromaps = false; // traces the scene's micromap TLAS
	};

	const Pipeline *SelectPipeline();
	const Pipeline *GetPipeline(bool shaderExecutionReordering, bool atmosphere, bool inlineShadows, bool opacityMicromaps,
								bool psr, bool sgLightTree);
	Pipeline CreatePipeline(bool shaderExecutionReordering, bool atmosphere, bool inlineShadows, bool opacityMicromaps,
							bool psr, bool sgLightTree) const;
	ResourceHandle CreateTarget(Format format, const char *debugName) const;
	// Fills constants.photons for this frame (the photon map over the casters), enabled or not, and says whether the
	// photon map starts over.
	void PreparePhotons(FrameConstants &constants, bool sunActive, float sunAngularRadius);
	// PhotonMap.slang's passes before the photons, and after the paths.
	void DispatchPhotonMap(RenderContext *renderContext, ResourceHandle constantBuffer);
	void DispatchPhotonSplat(RenderContext *renderContext, ResourceHandle constantBuffer);
	PhotonPassConstants PhotonPassConstantsForFrame() const;

	RenderContext *renderContext_ = nullptr;
	const RenderGPUScene &scene_;
	const Camera &camera_;
	const RenderSettings &settings_;
	const SkyAtmospherePass &atmosphere_;
	const bool rayReconstruction_;
	const bool reference_;

	// Indexed by shaderExecutionReordering, atmosphere, inline shadows, opacity micromaps, then primary surface replacement.
	Pipeline pipelines_[2][2][2][2][2][2];
	bool serSupported_ = false;
	bool serFailed_ = false; // the SER pipeline did not compile, so stay on the plain one
	bool ommFailed_ = false; // the opacity micromap pipeline did not compile, so trace without them
	ResourceHandle output_ = InvalidResourceHandle;
	PathTracerGuides guides_;
	ResourceHandle cameraState_ = InvalidResourceHandle; // GpuCameraState per pixel, DLSS-RR only
	ResourceHandle deferredBranches_ = InvalidResourceHandle; // GpuDeferredBranch per pixel
	ResourceHandle exposure_ = InvalidResourceHandle;

	// Caustic photons (created on first use): the photon map's shares (two, last frame's and this one's), running sums
	// and feedback, the photon records, and per pixel the receivers, the splatted irradiance and the caustics' history
	// (two). The caster rectangles per frame in flight.
	enum PhotonPass
	{
		PhotonPassUpdate,
		PhotonPassScanBlocks,
		PhotonPassScanTotals,
		PhotonPassSplat,
		PhotonPassComposite,
		PhotonPassCount
	};
	PipelineHandle photonPipelines_[PhotonPassCount] = {};
	ResourceHandle photonDensity_[2] = { InvalidResourceHandle, InvalidResourceHandle };
	ResourceHandle photonTexelPrefix_ = InvalidResourceHandle;
	ResourceHandle photonBlockPrefix_ = InvalidResourceHandle;
	ResourceHandle photonFeedback_ = InvalidResourceHandle;
	ResourceHandle photonRecords_ = InvalidResourceHandle;
	ResourceHandle photonRecordCount_ = InvalidResourceHandle;
	ResourceHandle photonReceivers_ = InvalidResourceHandle;
	ResourceHandle photonCaustics_ = InvalidResourceHandle;
	ResourceHandle photonHistory_[2] = { InvalidResourceHandle, InvalidResourceHandle };
	std::vector<ResourceHandle> photonRects_;
	std::vector<std::pair<float3, float3>> photonCasters_;
	std::vector<GpuPhotonRect> photonRectData_;
	bool photonMapValid_ = false; // photons ran last frame with this map
	bool photonReset_ = false; // this frame's photon map starts over
	uint photonParity_ = 0; // which of the two shares and histories is this frame's
	uint photonResetCount_ = 0;
	bool photonAdaptive_ = true; // the last frame's PhotonSettings::adaptive
	float3 photonSunDirection_ = { 0.0f, 0.0f, 0.0f }; // the map's plane and square, a change of which starts it over
	float2 photonMapOrigin_ = { 0.0f, 0.0f };
	float photonMapScale_ = 0.0f;
	std::vector<ResourceHandle> frameConstants_; // one per frame in flight, CPU-written each frame
	uint width_ = 0;
	uint height_ = 0;

	uint jitterIndex_ = 0;
	float2 jitter_ = { 0.0f, 0.0f };

	// Reference only. The key is the last dispatch's constants without what changes every frame anyway.
	ResourceHandle accumulation_ = InvalidResourceHandle; // RGBA32F sum of the paths so far
	uint accumulatedSamples_ = 0;
	FrameConstants accumulationKey_ = {};
	AtmosphereParameters accumulationAtmosphere_ = {};
	uint accumulationRestartCount_ = 0;
	std::chrono::steady_clock::time_point accumulationStart_;
	float accumulationSeconds_ = 0.0f;
};
