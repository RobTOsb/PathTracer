#include "AutoExposurePass.h"

#include "FrameGraph.h"
#include "PathTracerInterop.h"
#include "RenderSettings.h"

#include <math.h>
#include <stdexcept>
#include <string>

namespace
{
	// Must match AutoExposure.slang.
	constexpr uint32_t kHistogramGroupSize = 16;
	constexpr uint32_t kHistogramSize = 64;
} // namespace

AutoExposurePass::AutoExposurePass(RenderContext *renderContext, HdrSource hdrInput, const RenderSettings &settings) :
	renderContext_(renderContext), hdrInput_(std::move(hdrInput)), settings_(settings)
{
	for (auto [pipeline, entryPoint]: { std::pair{ &histogramPipeline_, "CalculateHistogram" },
										std::pair{ &exposurePipeline_, "CalculateExposure" } })
	{
		*pipeline = renderContext->CreateComputePipeline(PATHTRACER_SHADER_DIR "AutoExposure.slang", entryPoint);
		if (*pipeline == InvalidPipelineHandle)
		{
			throw std::runtime_error(std::string("Failed to create the auto exposure pipeline ") + entryPoint +
									 " (see the shader log above).");
		}
	}

	// The image after DLSS, or the path tracer's at the window's resolution without it.
	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();

	// The histogram starts empty, after which CalculateExposure empties it again each frame; the exposure
	// starts at 1 for whatever reads it before the first CalculateExposure.
	const uint32_t zeros[kHistogramSize] = {};
	const float one = 1.0f;
	const UploadBatchHandle batch = renderContext->BeginUpload();
	auto createBuffer = [&](uint32_t count, const void *data, const char *debugName) {
		BufferDesc desc = {};
		desc.sizeInBytes = count * sizeof(uint32_t);
		desc.structureByteStride = sizeof(uint32_t);
		desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		desc.debugName = debugName;
		return renderContext->CreateBuffer(desc, ResourceViewType::UnorderedAccess, data, batch);
	};
	histogram_ = createBuffer(kHistogramSize, zeros, "AutoExposure_Histogram");
	keySceneLuminance_ = createBuffer(1, &one, "AutoExposure_KeySceneLuminance");
	exposure_ = createBuffer(1, &one, "AutoExposure_Exposure");
	tracingExposure_[0] = createBuffer(1, &one, "AutoExposure_TracingExposure0");
	tracingExposure_[1] = createBuffer(1, &one, "AutoExposure_TracingExposure1");
	renderContext->EndUpload(batch);
}

AutoExposurePass::~AutoExposurePass()
{
	for (ResourceHandle buffer: { histogram_, keySceneLuminance_, exposure_, tracingExposure_[0], tracingExposure_[1] })
	{
		renderContext_->DestroyResource(buffer);
	}
}

void AutoExposurePass::AddPasses(FrameGraph &frame)
{
	const bool enabled = settings_.autoExposure;
	const auto now = std::chrono::steady_clock::now();
	const float frameTime = enabled && adapting_ ? std::chrono::duration<float>(now - lastTime_).count() : 0.0f;
	adapting_ = enabled;
	lastTime_ = now;

	AutoExposureConstants constants = {};
	constants.bufferDimensions = uint2(width_, height_);
	constants.frameTime = frameTime;
	constants.exposureBias = exp2f(settings_.exposureEV);
	constants.maxExposure = exp2f(settings_.maxAutoExposureEV);
	const rg::ResourceId hdrInput = hdrInput_();
	constants.inputBufferHandle = INVALIDDESCRIPTORHANDLE; // a transient, filled in when the histogram runs
	constants.histogramHandle = frame.DescriptorIndex(histogram_);
	constants.keySceneLuminanceHandle = frame.DescriptorIndex(keySceneLuminance_);
	constants.exposureHandle = frame.DescriptorIndex(exposure_);
	constants.enabled = enabled ? 1u : 0u;

	// Every buffer is read and written through its UAV, and so is the image.
	const rg::ResourceId histogram = frame.ImportBuffer(histogram_, "AutoExposure_Histogram");
	const rg::ResourceId keySceneLuminance = frame.ImportBuffer(keySceneLuminance_, "AutoExposure_KeySceneLuminance");
	const rg::ResourceId exposure = frame.ImportBuffer(exposure_, "AutoExposure_Exposure");
	if (enabled)
	{
		frame.AddPass(rg::Pass::Compute("AutoExposure/Histogram")
									.Use(hdrInput, rg::Access::ComputeShaderReadWrite)
									.Use(histogram, rg::Access::ComputeShaderReadWrite)
									.Executes(
											[this, constants, hdrInput](rg::PassContext &ctx)
											{
												AutoExposureConstants placed = constants;
												placed.inputBufferHandle = ctx.rc.GetResourceViewInfo(ctx.Get(hdrInput)).descriptorIndex;
												ctx.rc.SetComputePipeline(histogramPipeline_);
												ctx.rc.SetCompute32BitConstants(&placed, sizeof(placed) / sizeof(uint32_t));
												ctx.rc.Dispatch((width_ + kHistogramGroupSize - 1) / kHistogramGroupSize,
																(height_ + kHistogramGroupSize - 1) / kHistogramGroupSize,
																1);
											}));
	}

	frame.AddPass(rg::Pass::Compute("AutoExposure/Exposure")
								.Use(histogram, rg::Access::ComputeShaderReadWrite)
								.Use(keySceneLuminance, rg::Access::ComputeShaderReadWrite)
								.Use(exposure, rg::Access::ComputeShaderReadWrite)
								.Executes(
										[this, constants](rg::PassContext &ctx)
										{
											ctx.rc.SetComputePipeline(exposurePipeline_);
											ctx.rc.SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
											ctx.rc.Dispatch(1, 1, 1);
										}));

	const rg::ResourceId tracing =
			frame.ImportBuffer(tracingExposure_[frame.rc.GetCurrentFrame() & 1], "AutoExposure_TracingExposure");
	frame.AddPass(rg::Pass::Copy("AutoExposure/Copy for tracing")
						  .Reads(exposure)
						  .Writes(tracing)
						  .Executes(
								  [exposure, tracing](rg::PassContext &ctx)
								  { ctx.rc.CopyBufferRegion(ctx.Get(tracing), 0, ctx.Get(exposure), 0, sizeof(float)); }));
}
