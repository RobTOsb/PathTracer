#include "SkyAtmospherePass.h"

#include "Camera.h"
#include "RenderSettings.h"
#include "Scene/RenderGPUScene.h"

#include <algorithm>
#include <cfloat>
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

	// The cloud noise kernels' groups are 4 x 4 x 4.
	constexpr uint32_t kNoiseGroupSize = 4;

	// Most of the scene's projection onto the cloud shadow map's plane along a low sun, scene units.
	constexpr float kMaxCloudShadowMapSize = 50000.0f;

	uint32_t GroupCount(uint32_t size)
	{
		return (size + kGroupSize - 1) / kGroupSize;
	}
} // namespace

SkyAtmospherePass::SkyAtmospherePass(RenderContext *renderContext,
									 const RenderGPUScene &scene,
									 const Camera &camera,
									 const RenderSettings &settings) :
	renderContext_(renderContext), scene_(scene), camera_(camera), settings_(settings)
{
	transmittancePipeline_ = CreatePipeline("SkyAtmosphereLUTs.slang", "TransmittanceLUT");
	multiScatteringPipeline_ = CreatePipeline("SkyAtmosphereLUTs.slang", "MultiScatteringLUT");
	skyViewPipeline_ = CreatePipeline("SkyAtmosphereLUTs.slang", "SkyViewLUT");
	sceneLightingPipeline_ = CreatePipeline("SkyAtmosphereLUTs.slang", "SceneLighting");
	shapeNoisePipeline_ = CreatePipeline("CloudTextures.slang", "ShapeNoise");
	detailNoisePipeline_ = CreatePipeline("CloudTextures.slang", "DetailNoise");
	weatherMapPipeline_ = CreatePipeline("CloudTextures.slang", "WeatherMap");
	cloudShadowPipeline_ = CreatePipeline("CloudTextures.slang", "ShadowMap");

	transmittanceLUT_ =
			CreateLUT(TRANSMITTANCE_LUT_WIDTH, TRANSMITTANCE_LUT_HEIGHT, "SkyAtmosphere_Transmittance", transmittanceSRV_);
	multiScatteringLUT_ = CreateLUT(MULTI_SCATTERING_LUT_SIZE,
									MULTI_SCATTERING_LUT_SIZE,
									"SkyAtmosphere_MultiScattering",
									multiScatteringSRV_);
	skyViewLUT_ = CreateLUT(SKY_VIEW_LUT_WIDTH, SKY_VIEW_LUT_HEIGHT, "SkyAtmosphere_SkyView", skyViewSRV_);
	cloudShadowMap_ = CreateLUT(CLOUD_SHADOW_MAP_SIZE,
								CLOUD_SHADOW_MAP_SIZE,
								"Clouds_ShadowMap",
								cloudShadowMapSRV_,
								Format::R16_FLOAT);
	// Here rather than on the first frame with clouds, since views made during a frame only reach the shader
	// visible heap at the next frame's start.
	shapeNoise_ = CreateLUT(CLOUD_SHAPE_NOISE_SIZE,
							CLOUD_SHAPE_NOISE_SIZE,
							"Clouds_ShapeNoise",
							shapeNoiseSRV_,
							Format::R8G8B8A8_UNORM,
							CLOUD_SHAPE_NOISE_SIZE);
	detailNoise_ = CreateLUT(CLOUD_DETAIL_NOISE_SIZE,
							 CLOUD_DETAIL_NOISE_SIZE,
							 "Clouds_DetailNoise",
							 detailNoiseSRV_,
							 Format::R8G8B8A8_UNORM,
							 CLOUD_DETAIL_NOISE_SIZE);
	weatherMap_ = CreateLUT(CLOUD_WEATHER_MAP_SIZE,
							CLOUD_WEATHER_MAP_SIZE,
							"Clouds_WeatherMap",
							weatherMapSRV_,
							Format::R16G16_FLOAT);

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
	cloudBuffers_.resize(renderContext->GetFramesInFlight());
	for (ResourceHandle &buffer: cloudBuffers_)
	{
		buffer = renderContext->CreateConstantBuffer(sizeof(CloudParameters), "Clouds_Parameters");
	}

	parameters_ = MakeParameters(settings.atmosphere);
}

SkyAtmospherePass::~SkyAtmospherePass()
{
	// Views before the textures they view.
	for (ResourceHandle resource: { transmittanceSRV_,
									multiScatteringSRV_,
									skyViewSRV_,
									shapeNoiseSRV_,
									detailNoiseSRV_,
									weatherMapSRV_,
									cloudShadowMapSRV_,
									transmittanceLUT_,
									multiScatteringLUT_,
									skyViewLUT_,
									shapeNoise_,
									detailNoise_,
									weatherMap_,
									cloudShadowMap_,
									sceneLighting_ })
	{
		if (resource != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(resource);
		}
	}
	for (ResourceHandle buffer: parameterBuffers_)
	{
		renderContext_->DestroyResource(buffer);
	}
	for (ResourceHandle buffer: cloudBuffers_)
	{
		renderContext_->DestroyResource(buffer);
	}
}

PipelineHandle SkyAtmospherePass::CreatePipeline(const char *shader, const char *entryPoint) const
{
	const PipelineHandle pipeline =
			renderContext_->CreateComputePipeline(std::string(PATHTRACER_SHADER_DIR) + shader, entryPoint);
	if (pipeline == InvalidPipelineHandle)
	{
		throw std::runtime_error(std::string("Failed to create the sky atmosphere pipeline ") + entryPoint +
								 " (see the shader log above).");
	}
	return pipeline;
}

ResourceHandle SkyAtmospherePass::CreateLUT(uint width,
											uint height,
											const char *debugName,
											ResourceHandle &srv,
											Format format,
											uint depth) const
{
	TextureDesc desc = {};
	desc.dimension = depth > 1 ? TextureDimension::Texture3D : TextureDimension::Texture2D;
	desc.width = width;
	desc.height = height;
	desc.depthOrArraySize = depth;
	desc.format = format;
	desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	desc.debugName = debugName;
	const ResourceHandle texture = renderContext_->CreateTexture(desc, ResourceViewType::UnorderedAccess);

	TextureViewDesc viewDesc = {};
	viewDesc.viewType = ResourceViewType::ShaderResource;
	viewDesc.dimension = desc.dimension;
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

CloudParameters SkyAtmospherePass::MakeCloudParameters() const
{
	const CloudSettings &settings = settings_.clouds;
	const float3 sun = sunDirection_;
	CloudParameters c = {};
	if (!settings.enabled)
	{
		return c;
	}

	c.enabled = 1;
	c.planetRadius = parameters_.bottomRadius;
	c.atmosphereTopRadius = parameters_.topRadius;
	c.bottomRadius = parameters_.bottomRadius + std::max(settings.bottomKm, 0.0f);
	c.topRadius = std::max(parameters_.bottomRadius + settings.topKm, c.bottomRadius + 0.01f);
	c.coverage = settings.coverage;
	c.extinction = std::max(settings.density, 0.0f);
	c.windOffset = settings.windOffsetKm;
	c.sceneHeight = parameters_.bottomRadius + settings_.atmosphere.altitudeKm;
	c.sunDirection = sun;
	c.viewSteps = static_cast<uint>(std::max(settings.viewSteps, 1));
	c.indirectSteps = static_cast<uint>(std::max(settings.indirectSteps, 1));
	c.shapeNoise = renderContext_->GetResourceViewInfo(shapeNoiseSRV_).descriptorIndex;
	c.detailNoise = renderContext_->GetResourceViewInfo(detailNoiseSRV_).descriptorIndex;
	c.weatherMap = renderContext_->GetResourceViewInfo(weatherMapSRV_).descriptorIndex;
	c.shadowMap = renderContext_->GetResourceViewInfo(cloudShadowMapSRV_).descriptorIndex;
	c.transmittanceLut = renderContext_->GetResourceViewInfo(transmittanceSRV_).descriptorIndex;

	// The shadow map covers the square of the scene's floor that its bounds project onto along the sun (as
	// CloudShadowMapUv does), with a margin.
	float3 lo = scene_.GetBoundsMin();
	float3 hi = scene_.GetBoundsMax();
	if (!(lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z))
	{
		lo = hi = float3(0.0f, 0.0f, 0.0f);
	}
	const float slope = 1.0f / std::max(sun.y, 0.01f);
	float2 xzMin = { FLT_MAX, FLT_MAX };
	float2 xzMax = { -FLT_MAX, -FLT_MAX };
	for (int corner = 0; corner < 8; ++corner)
	{
		const float3 p = { (corner & 1) ? hi.x : lo.x, (corner & 2) ? hi.y : lo.y, (corner & 4) ? hi.z : lo.z };
		const float drop = (p.y - lo.y) * slope;
		const float2 xz = { p.x - sun.x * drop, p.z - sun.z * drop };
		xzMin = { std::min(xzMin.x, xz.x), std::min(xzMin.y, xz.y) };
		xzMax = { std::max(xzMax.x, xz.x), std::max(xzMax.y, xz.y) };
	}
	const float side = std::min(std::max(xzMax.x - xzMin.x, xzMax.y - xzMin.y) * 1.1f + 1.0f, kMaxCloudShadowMapSize);
	c.shadowMapOrigin = { 0.5f * (xzMin.x + xzMax.x - side), 0.5f * (xzMin.y + xzMax.y - side) };
	c.shadowMapInvSize = 1.0f / side;
	c.shadowMapPlane = lo.y;
	return c;
}

void SkyAtmospherePass::RenderCloudNoise(RenderContext *renderContext,
										 PipelineHandle pipeline,
										 ResourceHandle texture,
										 const char *output,
										 uint groupsX,
										 uint groupsY,
										 uint groupsZ)
{
	renderContext->TransitionResource(texture,
									  ResourceLayout::UnorderedAccess,
									  ResourceSync::ComputeShading,
									  ResourceAccess::UnorderedAccess);
	renderContext->SetComputePipeline(pipeline);
	renderContext->SetComputeUAVByName(output, texture);
	renderContext->Dispatch(groupsX, groupsY, groupsZ);
	// Read by the shadow map kernel and by the path tracer.
	renderContext->TransitionResource(texture,
									  ResourceLayout::ShaderResource,
									  ResourceSync::AllShading,
									  ResourceAccess::ShaderResource);
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
		// Also read by the path tracer's clouds.
		renderContext->TransitionResource(transmittanceLUT_,
										  ResourceLayout::ShaderResource,
										  ResourceSync::AllShading,
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

	cloudParameters_ = MakeCloudParameters();
	if (cloudParameters_.enabled != 0)
	{
		if (!cloudNoiseValid_)
		{
			const uint shapeGroups = CLOUD_SHAPE_NOISE_SIZE / kNoiseGroupSize;
			const uint detailGroups = CLOUD_DETAIL_NOISE_SIZE / kNoiseGroupSize;
			RenderCloudNoise(renderContext, shapeNoisePipeline_, shapeNoise_, "g_NoiseOutput", shapeGroups, shapeGroups, shapeGroups);
			RenderCloudNoise(renderContext, detailNoisePipeline_, detailNoise_, "g_NoiseOutput", detailGroups, detailGroups, detailGroups);
			RenderCloudNoise(renderContext,
							 weatherMapPipeline_,
							 weatherMap_,
							 "g_WeatherOutput",
							 GroupCount(CLOUD_WEATHER_MAP_SIZE),
							 GroupCount(CLOUD_WEATHER_MAP_SIZE),
							 1);
			cloudNoiseValid_ = true;
		}

		if (!shadowMapValid_ || std::memcmp(&shadowMapParameters_, &cloudParameters_, sizeof(CloudParameters)) != 0)
		{
			const ResourceHandle cloudBuffer = cloudBuffers_[renderContext->GetCurrentFrameIndex()];
			renderContext->UpdateConstantBuffer(cloudBuffer, &cloudParameters_, sizeof(cloudParameters_));
			renderContext->TransitionResource(cloudShadowMap_,
											  ResourceLayout::UnorderedAccess,
											  ResourceSync::ComputeShading,
											  ResourceAccess::UnorderedAccess);
			renderContext->SetComputePipeline(cloudShadowPipeline_);
			renderContext->SetComputeConstantBufferByName("g_Clouds", cloudBuffer);
			renderContext->SetComputeUAVByName("g_ShadowMapOutput", cloudShadowMap_);
			renderContext->Dispatch(GroupCount(CLOUD_SHADOW_MAP_SIZE), GroupCount(CLOUD_SHADOW_MAP_SIZE), 1);
			renderContext->TransitionResource(cloudShadowMap_,
											  ResourceLayout::ShaderResource,
											  ResourceSync::Raytracing,
											  ResourceAccess::ShaderResource);
			shadowMapParameters_ = cloudParameters_;
			shadowMapValid_ = true;
		}
	}

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
