#pragma once

#include "RenderPass.h"

#include "RenderContext.h"
#include "RenderGraph.h"

#include "Maths.h"
#include "PathTracerInterop.h"
#include "Scene/RenderGPUScene.h"
#include "SkyAtmosphereInterop.h"

#include <chrono>
#include <functional>
#include <utility>
#include <vector>

class Camera;
class SkyAtmospherePass;
struct RenderSettings;

// Render-resolution guides written alongside the radiance for DLSS Ray Reconstruction, as this frame's graph resources.
// The paths write all of them in full every frame, so they are transients, but for the specular hit distance and PSR
// outcome, which the paths read back the next frame.
struct PathTracerGuides
{
	rg::ResourceId diffuseAlbedo; // RGBA8
	rg::ResourceId specularAlbedo; // RGBA16F
	rg::ResourceId normalRoughness; // RGBA16F, world-space normal + linear roughness in .w
	rg::ResourceId motionVectors; // RG16F, render pixels
	rg::ResourceId depth; // R32F, reversed-Z hardware depth
	rg::ResourceId specularHitDistance; // R16F, imported
	rg::ResourceId specularMotionVectors; // RG16F
	// R8, 1 where a moving object's shadow changed the light since last frame: DLSS-RR's bias current colour mask.
	rg::ResourceId shadowChanges;
	// R16F, -1..1: DLSS-RR's responsivity mask (Preset F only), positive on smooth reflectors so their reflections keep
	// less history (FrameConstants::reflectionResponsivity).
	rg::ResourceId responsivity;
	// R16F, PsrOutcome per pixel: what became of its PSR chain, for GUIDE_VIEW_PSR_OUTCOME only. Imported.
	rg::ResourceId psrOutcome;
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
// Its graph passes: Photons/Map update, Photons/Scan blocks, Photons/Scan totals and Photons/Trace before the paths
// (with photons), PathTracer/Paths, then Photons/Splat and Photons/Composite (with photons).
// What a frame builds and nothing reads the next frame is a graph transient: the radiance, the guides but for two, the
// deferred glass branches, and the photons' prefix sums, records, receivers and splatted sums. The descriptors of those
// the constants name are only known once the graph has placed them, so the first of the frame's passes to run writes
// the frame constants.
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

	void AddPasses(FrameGraph &frame) override;

	// This frame's radiance (RGBA16F), a graph transient: valid from AddPasses until the frame's graph has run.
	rg::ResourceId GetOutput() const
	{
		return output_;
	}

	// This frame's guides, DLSS-RR only.
	const PathTracerGuides &GetGuides() const
	{
		return guides_;
	}

	// The exposure the firefly filter follows: one copy per frame parity, so the paths read the copy the frame before
	// last wrote (AutoExposurePass::GetTracingExposure).
	void SetExposureBuffers(ResourceHandle even, ResourceHandle odd)
	{
		exposure_[0] = even;
		exposure_[1] = odd;
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
	// Binds the root arguments every DispatchRays of the pipeline takes, from a pass that read the scene and the sky
	// (RenderGPUScene::UseForTracing, SkyAtmospherePass::UseForTracing).
	void BindRaytracing(rg::PassContext &ctx, const Pipeline &pipeline, ResourceHandle constantBuffer, bool atmosphere) const;
	// The real handle of one of this frame's graph resources, or none.
	ResourceHandle Handle(rg::ResourceId resource) const
	{
		return resource.IsValid() ? graph_->GetHandle(resource) : InvalidResourceHandle;
	}
	uint DescriptorIndex(rg::ResourceId resource) const;
	const Pipeline *GetPipeline(bool shaderExecutionReordering, bool atmosphere, bool inlineShadows, bool opacityMicromaps,
								bool psr, bool sgLightTree);
	Pipeline CreatePipeline(bool shaderExecutionReordering, bool atmosphere, bool inlineShadows, bool opacityMicromaps,
							bool psr, bool sgLightTree) const;
	ResourceHandle CreateTarget(Format format, const char *debugName) const;
	// A render-resolution target of this frame's graph.
	rg::ResourceId CreateTransientTarget(FrameGraph &frame, Format format, const char *debugName) const;
	// Fills constants.photons for this frame (the photon map over the casters), enabled or not, and says whether the
	// photon map starts over.
	void PreparePhotons(FrameConstants &constants, bool sunActive, float sunAngularRadius);
	// The photon buffers kept from frame to frame, made zeroed on first use.
	void CreatePhotonBuffers();

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
	// This frame's graph, its radiance, guides and deferred glass branches (GpuDeferredBranch per pixel).
	const rg::RenderGraph *graph_ = nullptr;
	rg::ResourceId output_;
	PathTracerGuides guides_;
	rg::ResourceId deferredBranches_;
	// The guides the paths read back the next frame, DLSS-RR only.
	ResourceHandle specularHitDistance_ = InvalidResourceHandle;
	ResourceHandle psrOutcome_ = InvalidResourceHandle;
	ResourceHandle cameraState_ = InvalidResourceHandle; // GpuCameraState per pixel, DLSS-RR only
	ResourceHandle exposure_[2] = { InvalidResourceHandle, InvalidResourceHandle };

	// Caustic photons (created on first use): the photon map's shares (two, last frame's and this one's), running sums,
	// feedback and coverage (and its sums, two), the photon records, and per pixel the receivers and the splatted
	// irradiance. The caster rectangles per frame in flight.
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
	PhotonPassConstants PhotonPassConstantsForFrame() const;
	// Binds a photon map pass, with this frame's transients' descriptors in its constants.
	void BindPhotonPass(RenderContext &rc, PhotonPass pass, ResourceHandle constantBuffer,
						const PhotonPassConstants &constants) const;
	// PhotonMap.slang's passes before the photons are traced: the map adapted, then scanned.
	// writeConstants writes the frame constants if no pass has yet. Without active, the passes do nothing.
	void AddPhotonMapPasses(FrameGraph &frame, ResourceHandle constantBuffer, const PhotonPassConstants &constants,
							const std::function<void(RenderContext &)> &writeConstants, bool active);
	// Kept from frame to frame: the shares, the block sums (whose total the next update reads), the feedback and the
	// coverage.
	ResourceHandle photonDensity_[2] = { InvalidResourceHandle, InvalidResourceHandle };
	ResourceHandle photonBlockPrefix_ = InvalidResourceHandle;
	ResourceHandle photonFeedback_ = InvalidResourceHandle;
	ResourceHandle photonCoverage_ = InvalidResourceHandle;
	ResourceHandle photonCoverageSums_ = InvalidResourceHandle;
	// This frame's transients, while photons run.
	rg::ResourceId photonTexelPrefix_;
	rg::ResourceId photonRecords_;
	rg::ResourceId photonRecordCount_;
	rg::ResourceId photonReceivers_;
	rg::ResourceId photonCaustics_;
	std::vector<ResourceHandle> photonRects_;
	std::vector<RenderGPUScene::PhotonCaster> photonCasters_;
	std::vector<GpuPhotonRect> photonRectData_;
	bool photonMapValid_ = false; // photons ran last frame with this map
	bool photonReset_ = false; // this frame's photon map starts over
	uint photonParity_ = 0; // which of the two shares and coverage sums is this frame's
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
