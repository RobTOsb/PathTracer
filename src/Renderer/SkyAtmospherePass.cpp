#include "SkyAtmospherePass.h"

#include "Camera.h"
#include "RenderSettings.h"

#include <algorithm>
#include <cstring>
#include <math.h>
#include <stdexcept>
#include <string>

#undef min
#undef max

namespace
{
	constexpr uint32_t kGroupSize = 8;

	// glTF scenes are in metres, the atmosphere in kilometres.
	constexpr float kKmPerSceneUnit = 0.001f;

	uint32_t GroupCount(uint32_t size)
	{
		return (size + kGroupSize - 1) / kGroupSize;
	}
} // namespace

SkyAtmospherePass::SkyAtmospherePass(RenderContext *renderContext,
									 const Camera &camera,
									 const RenderSettings &settings) :
	renderContext_(renderContext), camera_(camera), settings_(settings)
{
	transmittancePipeline_ = CreatePipeline("TransmittanceLUT");
	multiScatteringPipeline_ = CreatePipeline("MultiScatteringLUT");
	skyViewPipeline_ = CreatePipeline("SkyViewLUT");
	sceneLightingPipeline_ = CreatePipeline("SceneLighting");

	transmittanceLUT_ =
			CreateLUT(TRANSMITTANCE_LUT_WIDTH, TRANSMITTANCE_LUT_HEIGHT, "SkyAtmosphere_Transmittance", transmittanceSRV_);
	multiScatteringLUT_ = CreateLUT(MULTI_SCATTERING_LUT_SIZE,
									MULTI_SCATTERING_LUT_SIZE,
									"SkyAtmosphere_MultiScattering",
									multiScatteringSRV_);
	skyViewLUT_ = CreateLUT(SKY_VIEW_LUT_WIDTH, SKY_VIEW_LUT_HEIGHT, "SkyAtmosphere_SkyView", skyViewSRV_);

	BufferDesc lightingDesc = {};
	lightingDesc.sizeInBytes = ATMOSPHERE_LIGHTING_ROWS * sizeof(float4);
	lightingDesc.structureByteStride = sizeof(float4);
	lightingDesc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	lightingDesc.debugName = "SkyAtmosphere_SceneLighting";
	sceneLighting_ = renderContext->CreateBuffer(lightingDesc, ResourceViewType::UnorderedAccess);

	parameterBuffers_.resize(renderContext->GetFramesInFlight());
	for (ResourceHandle &buffer: parameterBuffers_)
	{
		buffer = renderContext->CreateConstantBuffer(sizeof(AtmosphereParameters), "SkyAtmosphere_Parameters");
	}

	parameters_ = MakeParameters(settings.atmosphere);
}

SkyAtmospherePass::~SkyAtmospherePass()
{
	// Views before the textures they view.
	for (ResourceHandle resource: { transmittanceSRV_,
									multiScatteringSRV_,
									skyViewSRV_,
									transmittanceLUT_,
									multiScatteringLUT_,
									skyViewLUT_,
									sceneLighting_ })
	{
		renderContext_->DestroyResource(resource);
	}
	for (ResourceHandle buffer: parameterBuffers_)
	{
		renderContext_->DestroyResource(buffer);
	}
}

PipelineHandle SkyAtmospherePass::CreatePipeline(const char *entryPoint) const
{
	const PipelineHandle pipeline =
			renderContext_->CreateComputePipeline(PATHTRACER_SHADER_DIR "SkyAtmosphereLUTs.slang", entryPoint);
	if (pipeline == InvalidPipelineHandle)
	{
		throw std::runtime_error(std::string("Failed to create the sky atmosphere pipeline ") + entryPoint +
								 " (see the shader log above).");
	}
	return pipeline;
}

ResourceHandle SkyAtmospherePass::CreateLUT(uint width, uint height, const char *debugName, ResourceHandle &srv) const
{
	TextureDesc desc = {};
	desc.width = width;
	desc.height = height;
	desc.format = Format::R16G16B16A16_FLOAT;
	desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	desc.debugName = debugName;
	const ResourceHandle texture = renderContext_->CreateTexture(desc, ResourceViewType::UnorderedAccess);

	TextureViewDesc viewDesc = {};
	viewDesc.viewType = ResourceViewType::ShaderResource;
	srv = renderContext_->CreateTextureView(texture, viewDesc);
	return texture;
}

AtmosphereParameters SkyAtmospherePass::MakeParameters(const AtmosphereSettings &settings)
{
	AtmosphereParameters p = {};
	p.bottomRadius = 6360.0f;
	p.topRadius = 6460.0f;
	p.groundAlbedo = settings.groundAlbedo;

	// Rayleigh scattering with a scale height of 8 km.
	p.rayleighDensityExpScale = -1.0f / 8.0f;
	p.rayleighScattering = float3(5.802e-3f, 13.558e-3f, 33.1e-3f) * settings.rayleighScale;

	// Mie scattering with a scale height of 1.2 km.
	p.mieDensityExpScale = -1.0f / 1.2f;
	p.mieScattering = float3(3.996e-3f, 3.996e-3f, 3.996e-3f) * settings.mieScale;
	p.mieExtinction = float3(4.40e-3f, 4.40e-3f, 4.40e-3f) * settings.mieScale;
	p.mieAbsorption = p.mieExtinction - p.mieScattering;
	p.miePhaseG = settings.miePhaseG;

	// Ozone: a tent of width 30 km peaking at 25 km.
	p.absorptionDensity0LayerWidth = 25.0f;
	p.absorptionDensity0ConstantTerm = -2.0f / 3.0f;
	p.absorptionDensity0LinearTerm = 1.0f / 15.0f;
	p.absorptionDensity1ConstantTerm = 8.0f / 3.0f;
	p.absorptionDensity1LinearTerm = -1.0f / 15.0f;
	p.absorptionExtinction = float3(0.650e-3f, 1.881e-3f, 0.085e-3f) * settings.ozoneScale;

	p.multipleScatteringFactor = settings.multipleScatteringFactor;
	return p;
}

bool SkyAtmospherePass::IsEnabled() const
{
	return settings_.atmosphere.enabled;
}

bool SkyAtmospherePass::IsSunAboveHorizon() const
{
	// The sun's ray from the camera clears the planet unless it dips below the horizon, whose zenith
	// cosine is -sqrt(1 - (R / h)^2).
	const float ratio = parameters_.bottomRadius / viewHeight_;
	return sunDirection_.y >= -sqrtf(std::max(0.0f, 1.0f - ratio * ratio));
}

SkyAtmospherePass::SceneAir SkyAtmospherePass::GetSceneAir() const
{
	// As SampleAtmosphereMedium in SkyAtmosphere.slang.
	const AtmosphereParameters &p = parameters_;
	const float height = viewHeight_ - p.bottomRadius;
	const float densityMie = expf(p.mieDensityExpScale * height);
	const float densityRay = expf(p.rayleighDensityExpScale * height);
	const float ozoneLinear = height < p.absorptionDensity0LayerWidth
									  ? p.absorptionDensity0LinearTerm * height + p.absorptionDensity0ConstantTerm
									  : p.absorptionDensity1LinearTerm * height + p.absorptionDensity1ConstantTerm;
	const float densityOzo = std::clamp(ozoneLinear, 0.0f, 1.0f);

	const float airScale = kKmPerSceneUnit * settings_.atmosphere.sceneAirScale;
	const float hazeScale = kKmPerSceneUnit * settings_.atmosphere.sceneHazeScale;
	SceneAir air;
	air.rayleighScattering = p.rayleighScattering * (densityRay * airScale);
	air.mieScattering = p.mieScattering * (densityMie * hazeScale);
	air.extinction = p.mieExtinction * (densityMie * hazeScale) +
					 (p.rayleighScattering * densityRay + p.absorptionExtinction * densityOzo) * airScale;
	return air;
}

void SkyAtmospherePass::Execute(RenderContext *renderContext)
{
	if (!IsEnabled())
	{
		return;
	}

	const AtmosphereSettings &settings = settings_.atmosphere;
	parameters_ = MakeParameters(settings);
	sunDirection_ = NormaliseFloat3Copy(settings_.sun.GetDirection());

	// The camera's altitude, kept clear of the ground where the LUT parameterisations break down.
	const float altitude = settings.altitudeKm + camera_.GetPosition().y * kKmPerSceneUnit;
	viewHeight_ = parameters_.bottomRadius + std::max(altitude, PLANET_RADIUS_OFFSET);

	// The slot for this frame index was last read framesInFlight frames ago, and BeginFrame has
	// already waited on that frame's fence.
	const ResourceHandle parameterBuffer = parameterBuffers_[renderContext->GetCurrentFrameIndex()];
	renderContext->UpdateConstantBuffer(parameterBuffer, &parameters_, sizeof(parameters_));

	AtmosphereLUTConstants constants = {};
	constants.viewHeight = viewHeight_;
	constants.sunZenithCos = sunDirection_.y;

	renderContext->BeginEvent("SkyAtmosphere");
	renderContext->BeginProfileScope("SkyAtmosphere");

	if (!lutsValid_ || std::memcmp(&lutParameters_, &parameters_, sizeof(parameters_)) != 0)
	{
		renderContext->TransitionResource(transmittanceLUT_,
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::ComputeShading,
										  ResourceAccess::UnorderedAccess);
		renderContext->SetComputePipeline(transmittancePipeline_);
		renderContext->SetComputeConstantBufferByName("g_Atmosphere", parameterBuffer);
		renderContext->SetComputeUAVByName("g_Output", transmittanceLUT_);
		renderContext->Dispatch(GroupCount(TRANSMITTANCE_LUT_WIDTH), GroupCount(TRANSMITTANCE_LUT_HEIGHT), 1);
		renderContext->TransitionResource(transmittanceLUT_,
										  ResourceLayout::ShaderResource,
										  ResourceSync::ComputeShading,
										  ResourceAccess::ShaderResource);

		// One group of MULTI_SCATTERING_SAMPLES threads per texel.
		renderContext->TransitionResource(multiScatteringLUT_,
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::ComputeShading,
										  ResourceAccess::UnorderedAccess);
		renderContext->SetComputePipeline(multiScatteringPipeline_);
		renderContext->SetComputeConstantBufferByName("g_Atmosphere", parameterBuffer);
		renderContext->SetComputeSRVByName("g_TransmittanceLUT", transmittanceSRV_);
		renderContext->SetComputeUAVByName("g_Output", multiScatteringLUT_);
		renderContext->Dispatch(MULTI_SCATTERING_LUT_SIZE, MULTI_SCATTERING_LUT_SIZE, 1);
		renderContext->TransitionResource(multiScatteringLUT_,
										  ResourceLayout::ShaderResource,
										  ResourceSync::ComputeShading,
										  ResourceAccess::ShaderResource);

		lutParameters_ = parameters_;
		lutsValid_ = true;
	}

	renderContext->TransitionResource(skyViewLUT_,
									  ResourceLayout::UnorderedAccess,
									  ResourceSync::ComputeShading,
									  ResourceAccess::UnorderedAccess);
	renderContext->SetComputePipeline(skyViewPipeline_);
	renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
	renderContext->SetComputeConstantBufferByName("g_Atmosphere", parameterBuffer);
	renderContext->SetComputeSRVByName("g_TransmittanceLUT", transmittanceSRV_);
	renderContext->SetComputeSRVByName("g_MultiScatteringLUT", multiScatteringSRV_);
	renderContext->SetComputeUAVByName("g_Output", skyViewLUT_);
	renderContext->Dispatch(GroupCount(SKY_VIEW_LUT_WIDTH), GroupCount(SKY_VIEW_LUT_HEIGHT), 1);
	renderContext->TransitionResource(skyViewLUT_,
									  ResourceLayout::ShaderResource,
									  ResourceSync::Raytracing,
									  ResourceAccess::ShaderResource);

	renderContext->TransitionResource(sceneLighting_,
									  ResourceLayout::UnorderedAccess,
									  ResourceSync::ComputeShading,
									  ResourceAccess::UnorderedAccess);
	renderContext->SetComputePipeline(sceneLightingPipeline_);
	renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
	renderContext->SetComputeConstantBufferByName("g_Atmosphere", parameterBuffer);
	renderContext->SetComputeSRVByName("g_TransmittanceLUT", transmittanceSRV_);
	renderContext->SetComputeSRVByName("g_MultiScatteringLUT", multiScatteringSRV_);
	renderContext->SetComputeUAVByName("g_SceneLighting", sceneLighting_);
	renderContext->Dispatch(1, 1, 1);
	renderContext->TransitionResource(sceneLighting_,
									  ResourceLayout::ShaderResource,
									  ResourceSync::Raytracing,
									  ResourceAccess::ShaderResource);

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
