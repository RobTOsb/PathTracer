#pragma once

#include "RenderPass.h"

#include "RenderContext.h"

#include "SkyAtmosphereInterop.h"

#include <vector>

class Camera;
struct AtmosphereSettings;
struct RenderSettings;

class SkyAtmospherePass final : public RenderPass
{
public:
	SkyAtmospherePass(RenderContext *renderContext, const Camera &camera, const RenderSettings &settings);
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

	// SRV of the sky-view LUT (t14 of the path tracer's ATMOSPHERE permutation).
	ResourceHandle GetSkyViewLUT() const
	{
		return skyViewSRV_;
	}

	// ATMOSPHERE_LIGHTING_ROWS float4s (t15 of the path tracer's ATMOSPHERE permutation).
	ResourceHandle GetSceneLighting() const
	{
		return sceneLighting_;
	}

	// False once the planet hides the sun's centre from the camera's altitude.
	bool IsSunAboveHorizon() const;

	// The air at the camera's altitude, per scene unit (metres) and scaled by the aerial perspective
	// scale, which the path tracer treats as filling the scene homogeneously.
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
	PipelineHandle CreatePipeline(const char *entryPoint) const;
	ResourceHandle CreateLUT(uint width, uint height, const char *debugName, ResourceHandle &srv) const;

	RenderContext *renderContext_ = nullptr;
	const Camera &camera_;
	const RenderSettings &settings_;

	PipelineHandle transmittancePipeline_ = InvalidPipelineHandle;
	PipelineHandle multiScatteringPipeline_ = InvalidPipelineHandle;
	PipelineHandle skyViewPipeline_ = InvalidPipelineHandle;
	PipelineHandle sceneLightingPipeline_ = InvalidPipelineHandle;

	ResourceHandle transmittanceLUT_ = InvalidResourceHandle;
	ResourceHandle transmittanceSRV_ = InvalidResourceHandle;
	ResourceHandle multiScatteringLUT_ = InvalidResourceHandle;
	ResourceHandle multiScatteringSRV_ = InvalidResourceHandle;
	ResourceHandle skyViewLUT_ = InvalidResourceHandle;
	ResourceHandle skyViewSRV_ = InvalidResourceHandle;
	ResourceHandle sceneLighting_ = InvalidResourceHandle;
	std::vector<ResourceHandle> parameterBuffers_; // one per frame in flight, CPU-written each frame

	AtmosphereParameters parameters_ = {};
	AtmosphereParameters lutParameters_ = {}; // what the transmittance and multiple scattering LUTs hold
	bool lutsValid_ = false;
	float viewHeight_ = 0.0f;
	float3 sunDirection_ = { 0.0f, 1.0f, 0.0f };
};
