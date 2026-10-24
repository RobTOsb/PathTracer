#include "BloomPass.h"

#include "Camera.h"
#include "PathTracerInterop.h"

#include <algorithm>
#include <math.h>
#include <stdexcept>
#include <string>

namespace
{
	constexpr uint32_t kGroupSize = 8;

	// Output pixels around the sun disc also left out, for the soft edge DLSS and filtering leave.
	constexpr float kSunMaskMarginPixels = 3.0f;

	uint GroupCount(uint size)
	{
		return (size + kGroupSize - 1) / kGroupSize;
	}

	void TransitionToUAV(RenderContext *renderContext, ResourceHandle texture)
	{
		renderContext->TransitionResource(texture,
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::ComputeShading,
										  ResourceAccess::UnorderedAccess);
	}

	void TransitionToSRV(RenderContext *renderContext, ResourceHandle texture)
	{
		renderContext->TransitionResource(texture,
										  ResourceLayout::ShaderResource,
										  ResourceSync::ComputeShading,
										  ResourceAccess::ShaderResource);
	}
}

BloomPass::BloomPass(RenderContext *renderContext,
					 ResourceHandle hdrInput,
					 ResourceHandle exposure,
					 const Camera &camera,
					 const RenderSettings &settings) :
	renderContext_(renderContext), hdrInput_(hdrInput), exposure_(exposure), camera_(camera), settings_(settings)
{
	downsamplePipeline_ = CreatePipeline("Downsample");
	upsamplePipeline_ = CreatePipeline("Upsample");
	compositePipeline_ = CreatePipeline("Composite");

	TextureViewDesc srvDesc = {};
	srvDesc.viewType = ResourceViewType::ShaderResource;
	hdrInputSRV_ = renderContext->CreateTextureView(hdrInput_, srvDesc);

	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();

	TextureDesc desc = {};
	desc.width = width_;
	desc.height = height_;
	desc.format = Format::R16G16B16A16_FLOAT;
	desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	desc.debugName = "Bloom_Output";
	output_ = renderContext->CreateTexture(desc, ResourceViewType::UnorderedAccess);

	// Separate textures rather than the mips of one, so each level is transitioned on its own.
	uint levelWidth = width_;
	uint levelHeight = height_;
	while (levelCount_ < kMaxBloomLevels && levelWidth > 1 && levelHeight > 1)
	{
		levelWidth = (levelWidth + 1) / 2;
		levelHeight = (levelHeight + 1) / 2;

		Level &level = levels_[levelCount_];
		level.width = levelWidth;
		level.height = levelHeight;
		desc.width = levelWidth;
		desc.height = levelHeight;
		desc.debugName = "Bloom_Level" + std::to_string(levelCount_);
		level.texture = renderContext->CreateTexture(desc, ResourceViewType::UnorderedAccess);
		level.srv = renderContext->CreateTextureView(level.texture, srvDesc);
		++levelCount_;
	}
}

BloomPass::~BloomPass()
{
	// Views before the textures they view.
	for (int i = 0; i < levelCount_; ++i)
	{
		renderContext_->DestroyResource(levels_[i].srv);
		renderContext_->DestroyResource(levels_[i].texture);
	}
	renderContext_->DestroyResource(hdrInputSRV_);
	renderContext_->DestroyResource(output_);
}

PipelineHandle BloomPass::CreatePipeline(const char *entryPoint) const
{
	const PipelineHandle pipeline = renderContext_->CreateComputePipeline(PATHTRACER_SHADER_DIR "Bloom.slang", entryPoint);
	if (pipeline == InvalidPipelineHandle)
	{
		throw std::runtime_error(std::string("Failed to create the bloom pipeline ") + entryPoint +
								 " (see the shader log above).");
	}
	return pipeline;
}

void BloomPass::Dispatch(RenderContext *renderContext, const BloomConstants &constants) const
{
	renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
	renderContext->Dispatch(GroupCount(constants.width), GroupCount(constants.height), 1);
}

void BloomPass::SetSunMask(BloomConstants &constants) const
{
	// The projection's scales give the tangents of the half fields of view.
	const float4x4 projection = camera_.GetProjectionMatrix();
	const float tanHalfFovX = 1.0f / projection._11;
	const float tanHalfFovY = 1.0f / projection._22;
	constants.tanHalfFov = float2(tanHalfFovX, tanHalfFovY);

	const float3 sun = NormaliseFloat3Copy(settings_.sun.GetDirection());
	const float3 right = camera_.GetRight();
	const float3 up = camera_.GetUp();
	const float3 front = camera_.GetFront();
	constants.sunDirectionView = float3(sun.x * right.x + sun.y * right.y + sun.z * right.z,
										sun.x * up.x + sun.y * up.y + sun.z * up.z,
										sun.x * front.x + sun.y * front.y + sun.z * front.z);

	// A sun without a disc cannot be seen, so there is nothing to leave out.
	const float sunRadius = DegreesToRadians(settings_.sun.angularRadiusDegrees);
	if (settings_.bloomSun || sunRadius <= 0.0f)
	{
		return;
	}
	// One output pixel's angle at the centre of the screen.
	const float pixelAngle = 2.0f * tanHalfFovY / static_cast<float>(height_);
	const float halfRadius = 0.5f * (sunRadius + kSunMaskMarginPixels * pixelAngle);
	constants.sunMaskOneMinusCos = 2.0f * sinf(halfRadius) * sinf(halfRadius);
}

void BloomPass::Execute(RenderContext *renderContext)
{
	renderContext->BeginEvent("Bloom");
	renderContext->BeginProfileScope("Bloom");

	TransitionToSRV(renderContext, hdrInput_);

	const int levelCount = settings_.bloom ? std::clamp(settings_.bloomLevels, 1, levelCount_) : 0;

	// Down the chain, each level filtered from the one above it.
	renderContext->SetComputePipeline(downsamplePipeline_);
	for (int i = 0; i < levelCount; ++i)
	{
		const Level &level = levels_[i];
		const uint inputWidth = i == 0 ? width_ : levels_[i - 1].width;
		const uint inputHeight = i == 0 ? height_ : levels_[i - 1].height;

		BloomConstants constants = {};
		constants.inputTexelSize = float2(1.0f / inputWidth, 1.0f / inputHeight);
		constants.outputTexelSize = float2(1.0f / level.width, 1.0f / level.height);
		constants.width = level.width;
		constants.height = level.height;
		constants.karisAverage = i == 0 ? 1u : 0u;
		if (i == 0)
		{
			SetSunMask(constants);
		}

		TransitionToUAV(renderContext, level.texture);
		renderContext->SetComputeSRVByName("g_Input", i == 0 ? hdrInputSRV_ : levels_[i - 1].srv);
		renderContext->SetComputeUAVByName("g_Output", level.texture);
		if (i == 0)
		{
			renderContext->SetComputeUAVByName("g_Exposure", exposure_);
		}
		Dispatch(renderContext, constants);
		TransitionToSRV(renderContext, level.texture);
	}

	// Back up, each level adding the sum of the ones below it.
	renderContext->SetComputePipeline(upsamplePipeline_);
	for (int i = levelCount - 1; i > 0; --i)
	{
		const Level &below = levels_[i];
		const Level &level = levels_[i - 1];

		BloomConstants constants = {};
		constants.inputTexelSize = float2(1.0f / below.width, 1.0f / below.height);
		constants.outputTexelSize = float2(1.0f / level.width, 1.0f / level.height);
		constants.width = level.width;
		constants.height = level.height;

		TransitionToUAV(renderContext, level.texture);
		renderContext->SetComputeSRVByName("g_Input", below.srv);
		renderContext->SetComputeUAVByName("g_Output", level.texture);
		Dispatch(renderContext, constants);
		TransitionToSRV(renderContext, level.texture);
	}

	BloomConstants constants = {};
	constants.inputTexelSize = float2(1.0f / levels_[0].width, 1.0f / levels_[0].height);
	constants.outputTexelSize = float2(1.0f / width_, 1.0f / height_);
	constants.width = width_;
	constants.height = height_;
	constants.strength = levelCount > 0 ? std::clamp(settings_.bloomStrength, 0.0f, 1.0f) : 0.0f;
	constants.levelCount = static_cast<float>(std::max(levelCount, 1));

	TransitionToUAV(renderContext, output_);
	renderContext->SetComputePipeline(compositePipeline_);
	renderContext->SetComputeSRVByName("g_Input", hdrInputSRV_);
	renderContext->SetComputeSRVByName("g_Bloom", levels_[0].srv);
	renderContext->SetComputeUAVByName("g_Output", output_);
	Dispatch(renderContext, constants);

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
