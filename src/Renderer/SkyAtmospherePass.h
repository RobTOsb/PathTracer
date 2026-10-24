#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include "SkyAtmosphereInterop.h"

#include <vector>

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

	void Execute(RenderContext *renderContext) override;

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

	// SRV of the sky-view LUT.
	ResourceHandle GetSkyViewLUT() const
	{
		return skyViewSRV_;
	}

	// ATMOSPHERE_LIGHTING_ROWS float4s.
	ResourceHandle GetSceneLighting() const
	{
		return sceneLighting_;
	}

	// The cloud layer of the last Execute, enabled = 0 while the clouds are off.
	const CloudParameters &GetCloudParameters() const
	{
		return cloudParameters_;
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
	void RenderCloudNoise(RenderContext *renderContext, PipelineHandle pipeline, ResourceHandle texture, const char *output,
						  uint groupsX, uint groupsY, uint groupsZ);

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

	ResourceHandle transmittanceLUT_ = InvalidResourceHandle;
	ResourceHandle transmittanceSRV_ = InvalidResourceHandle;
	ResourceHandle multiScatteringLUT_ = InvalidResourceHandle;
	ResourceHandle multiScatteringSRV_ = InvalidResourceHandle;
	ResourceHandle skyViewLUT_ = InvalidResourceHandle;
	ResourceHandle skyViewSRV_ = InvalidResourceHandle;
	ResourceHandle sceneLighting_ = InvalidResourceHandle;
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

	AtmosphereParameters parameters_ = {};
	AtmosphereParameters lutParameters_ = {}; // what the transmittance and multiple scattering LUTs hold
	bool lutsValid_ = false;
	float viewHeight_ = 0.0f;
	float3 sunDirection_ = { 0.0f, 1.0f, 0.0f };
};
