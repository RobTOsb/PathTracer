#include "BloomPass.h"

#include "Camera.h"
#include "FrameGraph.h"
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
}

BloomPass::BloomPass(RenderContext *renderContext,
					 HdrSource hdrInput,
					 ResourceHandle exposure,
					 const Camera &camera,
					 const RenderSettings &settings) :
	renderContext_(renderContext), hdrInput_(std::move(hdrInput)), exposure_(exposure), camera_(camera), settings_(settings)
{
	downsamplePipeline_ = CreatePipeline("Downsample");
	upsamplePipeline_ = CreatePipeline("Upsample");
	compositePipeline_ = CreatePipeline("Composite");

	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();

	// Separate textures rather than the mips of one, each level half the one above rounded up.
	uint levelWidth = width_;
	uint levelHeight = height_;
	while (levelCount_ < kMaxBloomLevels && levelWidth > 1 && levelHeight > 1)
	{
		levelWidth = (levelWidth + 1) / 2;
		levelHeight = (levelHeight + 1) / 2;
		levels_[levelCount_].width = levelWidth;
		levels_[levelCount_].height = levelHeight;
		++levelCount_;
	}
}

BloomPass::~BloomPass()
{
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

void BloomPass::AddPasses(FrameGraph &frame)
{
	const int levelCount = settings_.bloom ? std::clamp(settings_.bloomLevels, 1, levelCount_) : 0;

	const rg::ResourceId input = hdrInput_();
	const rg::ResourceId exposure = frame.ImportBuffer(exposure_, "AutoExposure_Exposure");
	TextureDesc desc = {};
	desc.format = Format::R16G16B16A16_FLOAT;
	rg::ResourceId levels[kMaxBloomLevels];
	for (int i = 0; i < levelCount; ++i)
	{
		desc.width = levels_[i].width;
		desc.height = levels_[i].height;
		desc.debugName = "Bloom_Level" + std::to_string(i);
		levels[i] = frame.graph.CreateTexture(desc);
	}
	desc.width = width_;
	desc.height = height_;
	desc.debugName = "Bloom_Output";
	output_ = frame.graph.CreateTexture(desc);

	// The graph binds a transient's SRV or UAV as each pass uses it, the input's too.
	auto dispatch = [](rg::PassContext &ctx, const BloomConstants &constants)
	{
		ctx.rc.SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
		ctx.rc.Dispatch(GroupCount(constants.width), GroupCount(constants.height), 1);
	};

	// Down the chain, each level filtered from the one above it. The first also reads the exposure for its
	// firefly weights; every level binds it, since setting the pipeline resets the bindings.
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

		const rg::ResourceId source = i == 0 ? input : levels[i - 1];
		const rg::ResourceId target = levels[i];
		frame.AddPass(rg::Pass::Compute("Bloom/Down " + std::to_string(i))
									.Reads(source)
									.Use(exposure, rg::Access::ComputeShaderReadWrite)
									.Writes(target)
									.Executes(
											[this, dispatch, constants, i, source, target, exposure](rg::PassContext &ctx)
											{
												ctx.rc.SetComputePipeline(downsamplePipeline_);
												ctx.rc.SetComputeSRVByName("g_Input", ctx.Get(source));
												ctx.rc.SetComputeUAVByName("g_Output", ctx.Get(target));
												ctx.rc.SetComputeUAVByName("g_Exposure", ctx.Get(exposure));
												dispatch(ctx, constants);
											}));
	}

	// Back up, each level adding the sum of the ones below it.
	for (int i = levelCount - 1; i > 0; --i)
	{
		const Level &below = levels_[i];
		const Level &level = levels_[i - 1];

		BloomConstants constants = {};
		constants.inputTexelSize = float2(1.0f / below.width, 1.0f / below.height);
		constants.outputTexelSize = float2(1.0f / level.width, 1.0f / level.height);
		constants.width = level.width;
		constants.height = level.height;

		const rg::ResourceId source = levels[i];
		const rg::ResourceId target = levels[i - 1];
		frame.AddPass(rg::Pass::Compute("Bloom/Up " + std::to_string(i - 1))
									.Reads(source)
									.Use(target, rg::Access::ComputeShaderReadWrite)
									.Executes(
											[this, dispatch, constants, source, target](rg::PassContext &ctx)
											{
												ctx.rc.SetComputePipeline(upsamplePipeline_);
												ctx.rc.SetComputeSRVByName("g_Input", ctx.Get(source));
												ctx.rc.SetComputeUAVByName("g_Output", ctx.Get(target));
												dispatch(ctx, constants);
											}));
	}

	BloomConstants constants = {};
	constants.inputTexelSize = float2(1.0f / levels_[0].width, 1.0f / levels_[0].height);
	constants.outputTexelSize = float2(1.0f / width_, 1.0f / height_);
	constants.width = width_;
	constants.height = height_;
	constants.strength = levelCount > 0 ? std::clamp(settings_.bloomStrength, 0.0f, 1.0f) : 0.0f;
	constants.levelCount = static_cast<float>(std::max(levelCount, 1));

	// With bloom off nothing has written a level, and the strength of 0 leaves the image as it is, so the image
	// stands in for the bloom it would have blended in.
	const rg::ResourceId bloom = levelCount > 0 ? levels[0] : rg::ResourceId{};
	const rg::ResourceId output = output_;
	rg::Pass composite = rg::Pass::Compute("Bloom/Composite");
	composite.Reads(input);
	if (bloom.IsValid())
	{
		composite.Reads(bloom);
	}
	frame.AddPass(std::move(composite.Writes(output).Executes(
			[this, dispatch, constants, input, bloom, output](rg::PassContext &ctx)
			{
				ctx.rc.SetComputePipeline(compositePipeline_);
				ctx.rc.SetComputeSRVByName("g_Input", ctx.Get(input));
				ctx.rc.SetComputeSRVByName("g_Bloom", bloom.IsValid() ? ctx.Get(bloom) : ctx.Get(input));
				ctx.rc.SetComputeUAVByName("g_Output", ctx.Get(output));
				dispatch(ctx, constants);
			})));
}
