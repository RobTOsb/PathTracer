#include "AutoExposurePass.h"

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

	void UAVBarrier(RenderContext *renderContext, ResourceHandle buffer)
	{
		renderContext->TransitionResource(buffer,
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::ComputeShading,
										  ResourceAccess::UnorderedAccess);
	}
} // namespace

AutoExposurePass::AutoExposurePass(RenderContext *renderContext, ResourceHandle hdrInput, const RenderSettings &settings) :
	renderContext_(renderContext), hdrInput_(hdrInput), settings_(settings)
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
	renderContext->EndUpload(batch);
}

AutoExposurePass::~AutoExposurePass()
{
	for (ResourceHandle buffer: { histogram_, keySceneLuminance_, exposure_ })
	{
		renderContext_->DestroyResource(buffer);
	}
}

void AutoExposurePass::Execute(RenderContext *renderContext)
{
	renderContext->BeginEvent("AutoExposure");
	renderContext->BeginProfileScope("AutoExposure");

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
	constants.inputBufferHandle = renderContext->GetResourceViewInfo(hdrInput_).descriptorIndex;
	constants.histogramHandle = renderContext->GetResourceViewInfo(histogram_).descriptorIndex;
	constants.keySceneLuminanceHandle = renderContext->GetResourceViewInfo(keySceneLuminance_).descriptorIndex;
	constants.exposureHandle = renderContext->GetResourceViewInfo(exposure_).descriptorIndex;
	constants.enabled = enabled ? 1u : 0u;

	// Every barrier also orders the buffers after last frame's readers.
	UAVBarrier(renderContext, histogram_);
	UAVBarrier(renderContext, keySceneLuminance_);
	UAVBarrier(renderContext, exposure_);

	if (enabled)
	{
		renderContext->TransitionResource(hdrInput_,
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::ComputeShading,
										  ResourceAccess::UnorderedAccess);
		renderContext->SetComputePipeline(histogramPipeline_);
		renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
		renderContext->Dispatch((width_ + kHistogramGroupSize - 1) / kHistogramGroupSize,
								(height_ + kHistogramGroupSize - 1) / kHistogramGroupSize,
								1);
		UAVBarrier(renderContext, histogram_);
	}

	renderContext->SetComputePipeline(exposurePipeline_);
	renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
	renderContext->Dispatch(1, 1, 1);
	UAVBarrier(renderContext, exposure_);

	renderContext->EndProfileScope();
	renderContext->EndEvent();
}
