#include "GuideViewPass.h"

#include "PathTracerPass.h"
#include "RenderSettings.h"

#include <stdexcept>

namespace
{
	constexpr uint32_t kGroupSize = 8;
}

GuideViewPass::GuideViewPass(RenderContext *renderContext, const PathTracerPass &pathTracer, const RenderSettings &settings) :
	renderContext_(renderContext), pathTracer_(pathTracer), settings_(settings)
{
	pipeline_ = renderContext->CreateComputePipeline(PATHTRACER_SHADER_DIR "GuideView.slang", "main");
	if (pipeline_ == InvalidPipelineHandle)
	{
		throw std::runtime_error("Failed to create the guide view pipeline (see the shader log above).");
	}

	TextureDesc historyDesc = {};
	historyDesc.width = pathTracer.GetWidth();
	historyDesc.height = pathTracer.GetHeight();
	historyDesc.format = Format::R16G16B16A16_FLOAT;
	historyDesc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	historyDesc.debugName = "GuideView_History";
	for (ResourceHandle &history: history_)
	{
		history = renderContext->CreateTexture(historyDesc, ResourceViewType::UnorderedAccess);
	}

	// As TonemapPass's, since swapchain buffers cannot be bound as UAVs.
	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();
	TextureDesc outputDesc = {};
	outputDesc.width = width_;
	outputDesc.height = height_;
	outputDesc.format = renderContext->GetBackBufferFormat();
	outputDesc.usageFlags = ResourceUsageFlags::UnorderedAccess;
	outputDesc.debugName = "GuideView_Output";
	output_ = renderContext->CreateTexture(outputDesc, ResourceViewType::UnorderedAccess);
}

GuideViewPass::~GuideViewPass()
{
	for (ResourceHandle history: history_)
	{
		renderContext_->DestroyResource(history);
	}
	renderContext_->DestroyResource(output_);
}

void GuideViewPass::Execute(RenderContext *renderContext)
{
	const GuideView guide = settings_.guideView;
	const float2 jitter = pathTracer_.GetJitter();
	if (guide == GUIDE_VIEW_OFF)
	{
		lastGuide_ = GUIDE_VIEW_OFF;
		lastJitter_ = jitter;
		return;
	}

	renderContext->BeginEvent("GuideView");
	renderContext->BeginProfileScope("GuideView");

	const PathTracerGuides &guides = pathTracer_.GetGuides();
	ResourceHandle source = InvalidResourceHandle;
	switch (guide)
	{
	case GUIDE_VIEW_DIFFUSE_ALBEDO:
		source = guides.diffuseAlbedo;
		break;
	case GUIDE_VIEW_SPECULAR_ALBEDO:
		source = guides.specularAlbedo;
		break;
	case GUIDE_VIEW_NORMAL:
	case GUIDE_VIEW_ROUGHNESS:
		source = guides.normalRoughness;
		break;
	case GUIDE_VIEW_DEPTH:
		source = guides.depth;
		break;
	case GUIDE_VIEW_MOTION_VECTORS:
		source = guides.motionVectors;
		break;
	case GUIDE_VIEW_SHADOW_CHANGES:
		source = guides.shadowChanges;
		break;
	default:
		source = guides.specularMotionVectors;
		break;
	}

	const ResourceHandle previous = history_[historyIndex_ ^ 1];
	const ResourceHandle current = history_[historyIndex_];
	const ResourceHandle cameraState = pathTracer_.GetCameraState();
	for (ResourceHandle resource: { source, guides.motionVectors, previous, current, output_, cameraState })
	{
		renderContext->TransitionResource(resource,
										  ResourceLayout::UnorderedAccess,
										  ResourceSync::ComputeShading,
										  ResourceAccess::UnorderedAccess);
	}

	GuideViewConstants constants = {};
	constants.renderSize = { pathTracer_.GetWidth(), pathTracer_.GetHeight() };
	constants.outputSize = { width_, height_ };
	constants.prevJitter = lastJitter_;
	constants.guide = static_cast<uint>(guide);
	constants.change = settings_.guideViewChange ? 1u : 0u;
	constants.gain = settings_.guideViewGain;
	constants.historyValid = lastGuide_ == guide ? 1u : 0u;
	constants.hdr10 = renderContext->GetUseHDR() ? 1u : 0u;
	constants.paperWhiteNits = settings_.paperWhiteNits;
	constants.showHeld = settings_.guideViewHeld ? 1u : 0u;

	renderContext->SetComputePipeline(pipeline_);
	renderContext->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
	renderContext->SetComputeConstantBufferByName("g_Frame", pathTracer_.GetFrameConstants());
	renderContext->SetComputeUAV(0, source);
	renderContext->SetComputeUAV(1, guides.motionVectors);
	renderContext->SetComputeUAV(2, previous);
	renderContext->SetComputeUAV(3, current);
	renderContext->SetComputeUAV(4, output_);
	renderContext->SetComputeUAV(5, cameraState);
	renderContext->Dispatch((width_ + kGroupSize - 1) / kGroupSize, (height_ + kGroupSize - 1) / kGroupSize, 1);

	const ResourceHandle backBuffer = renderContext->GetCurrentBackBufferHandle();
	renderContext->TransitionResource(output_, ResourceLayout::CopySource, ResourceSync::Copy, ResourceAccess::CopySource);
	renderContext->TransitionResource(backBuffer, ResourceLayout::CopyDest, ResourceSync::Copy, ResourceAccess::CopyDest);
	renderContext->CopyTexture(backBuffer, output_);

	renderContext->EndProfileScope();
	renderContext->EndEvent();

	historyIndex_ ^= 1;
	lastGuide_ = guide;
	lastJitter_ = jitter;
}
