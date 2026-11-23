#pragma once

#include "RenderPass.h"

#include "RenderContext.h"
#include "RenderGraph.h"

#include "SkyAtmosphereInterop.h"

#include <vector>

namespace rg
{
	struct Pass;
}

class Camera;
class RenderGPUScene;
struct AtmosphereSettings;
struct RenderSettings;

class SkyAtmospherePass final : public RenderPass
{
public:
	SkyAtmospherePass(RenderContext *renderContext,
					  const RenderGPUScene &scene,
					  const Camera &camera,
					  const RenderSettings &settings);
	~SkyAtmospherePass() override;

	// The LUTs that changed, the sky-view LUT and scene lighting, the clouds' textures and the sky importance
	// pyramid, one graph pass per dispatch.
	void AddPasses(FrameGraph &frame) override;

	// Attaches what a ray tracing pass reads of this frame's sky: the LUTs and scene lighting while the sky is on,
	// the clouds' textures while they are, and the sky importance pyramid while it is built.
	void UseForTracing(rg::Pass &pass, FrameGraph &frame) const;

	bool IsEnabled() const;

	// The parameters and the camera's distance from the planet centre (km) of the last Execute.
	const AtmosphereParameters &GetParameters() const
	{
		return parameters_;
	}

	float GetViewHeight() const
	{
		return viewHeight_;
	}

	// This frame's sky-view LUT, a graph transient (a pass that reads it gets its SRV), once AddPasses has run.
	rg::ResourceId GetSkyViewLUT() const
	{
		return skyView_;
	}

	// ATMOSPHERE_LIGHTING_ROWS float4s, a graph transient.
	rg::ResourceId GetSceneLighting() const
	{
		return sceneLighting_;
	}

	// The cloud layer of the last Execute, enabled = 0 while the clouds are off.
	const CloudParameters &GetCloudParameters() const
	{
		return cloudParameters_;
	}

	// The sky importance pyramid (SkyImportance.slang) this frame builds, a graph transient read through its UAV, or
	// invalid when sky sampling is off.
	rg::ResourceId GetSkyImportance() const
	{
		return skyImportance_;
	}

	// False once the planet hides the sun's centre from the camera's altitude.
	bool IsSunAboveHorizon() const;

	// The air at the camera's altitude, per scene unit (metres) and scaled by the scene air and haze
	// scales, which the path tracer treats as filling the scene homogeneously.
	struct SceneAir
	{
		float3 rayleighScattering;
		float3 mieScattering;
		float3 extinction;
	};
	
	SceneAir GetSceneAir() const;

	// The Earth of the paper's reference implementation, with the settings' scales applied.
	static AtmosphereParameters MakeParameters(const AtmosphereSettings &settings);

private:
	PipelineHandle CreatePipeline(const char *shader, const char *entryPoint) const;
	// A 3D texture for depth > 1.
	ResourceHandle CreateLUT(uint width,
							 uint height,
							 const char *debugName,
							 ResourceHandle &srv,
							 Format format = Format::R16G16B16A16_FLOAT,
							 uint depth = 1) const;
	CloudParameters MakeCloudParameters() const;
	// Declared every frame the clouds are on; dispatches only with generate.
	void AddCloudNoisePass(FrameGraph &frame, const char *name, PipelineHandle pipeline, ResourceHandle texture,
						   const char *debugName, const char *output, uint groupsX, uint groupsY, uint groupsZ,
						   bool generate);

	RenderContext *renderContext_ = nullptr;
	const RenderGPUScene &scene_;
	const Camera &camera_;
	const RenderSettings &settings_;

	PipelineHandle transmittancePipeline_ = InvalidPipelineHandle;
	PipelineHandle multiScatteringPipeline_ = InvalidPipelineHandle;
	PipelineHandle skyViewPipeline_ = InvalidPipelineHandle;
	PipelineHandle sceneLightingPipeline_ = InvalidPipelineHandle;
	PipelineHandle shapeNoisePipeline_ = InvalidPipelineHandle;
	PipelineHandle detailNoisePipeline_ = InvalidPipelineHandle;
	PipelineHandle weatherMapPipeline_ = InvalidPipelineHandle;
	PipelineHandle cloudShadowPipeline_ = InvalidPipelineHandle;
	PipelineHandle cloudPanoramaPipeline_ = InvalidPipelineHandle;
	PipelineHandle skyImportanceTexelsPipeline_ = InvalidPipelineHandle;
	PipelineHandle skyImportanceReducePipeline_ = InvalidPipelineHandle;

	ResourceHandle transmittanceLUT_ = InvalidResourceHandle;
	ResourceHandle transmittanceSRV_ = InvalidResourceHandle;
	ResourceHandle multiScatteringLUT_ = InvalidResourceHandle;
	ResourceHandle multiScatteringSRV_ = InvalidResourceHandle;
	// Built every frame, so graph transients.
	rg::ResourceId skyView_;
	rg::ResourceId sceneLighting_;
	std::vector<ResourceHandle> parameterBuffers_; // one per frame in flight, CPU-written each frame

	// Rendered on the first frame with clouds.
	ResourceHandle shapeNoise_ = InvalidResourceHandle;
	ResourceHandle shapeNoiseSRV_ = InvalidResourceHandle;
	ResourceHandle detailNoise_ = InvalidResourceHandle;
	ResourceHandle detailNoiseSRV_ = InvalidResourceHandle;
	ResourceHandle weatherMap_ = InvalidResourceHandle;
	ResourceHandle weatherMapSRV_ = InvalidResourceHandle;
	bool cloudNoiseValid_ = false;
	// Rendered again whenever the cloud parameters change.
	ResourceHandle cloudShadowMap_ = InvalidResourceHandle;
	ResourceHandle cloudShadowMapSRV_ = InvalidResourceHandle;
	std::vector<ResourceHandle> cloudBuffers_; // one per frame in flight
	CloudParameters cloudParameters_ = {};
	CloudParameters shadowMapParameters_ = {}; // what the shadow map holds
	bool shadowMapValid_ = false;
	// Rendered every frame with clouds, unless CloudSettings::indirectPanorama is off.
	ResourceHandle cloudPanorama_ = InvalidResourceHandle;
	ResourceHandle cloudPanoramaSRV_ = InvalidResourceHandle;
	CloudParameters panoramaParameters_ = {}; // what the panorama holds, the wind offset left out
	bool panoramaValid_ = false;
	uint32_t panoramaFrame_ = 0;
	// Built every frame while AtmosphereSettings::skySampling is on, a graph transient.
	rg::ResourceId skyImportance_;

	AtmosphereParameters parameters_ = {};
	AtmosphereParameters lutParameters_ = {}; // what the transmittance and multiple scattering LUTs hold
	bool lutsValid_ = false;
	float viewHeight_ = 0.0f;
	float3 sunDirection_ = { 0.0f, 1.0f, 0.0f };
};
