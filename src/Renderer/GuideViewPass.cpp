#include "GuideViewPass.h"

#include "FrameGraph.h"
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

	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();
	outputFormat_ = renderContext->GetBackBufferFormat();
}

GuideViewPass::~GuideViewPass()
{
	for (ResourceHandle history: history_)
	{
		renderContext_->DestroyResource(history);
	}
}

void GuideViewPass::AddPasses(FrameGraph &frame)
{
	const GuideView guide = settings_.guideView;
	const float2 jitter = pathTracer_.GetJitter();
	if (guide == GUIDE_VIEW_OFF)
	{
		lastGuide_ = GUIDE_VIEW_OFF;
		lastJitter_ = jitter;
		return;
	}

	const PathTracerGuides &guides = pathTracer_.GetGuides();
	rg::ResourceId source;
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
	case GUIDE_VIEW_RESPONSIVITY:
		source = guides.responsivity;
		break;
	case GUIDE_VIEW_PSR_OUTCOME:
		source = guides.psrOutcome;
		break;
	default:
		source = guides.specularMotionVectors;
		break;
	}

	const ResourceHandle previous = history_[historyIndex_ ^ 1];
	const ResourceHandle current = history_[historyIndex_];
	const ResourceHandle cameraState = pathTracer_.GetCameraState();
	const ResourceHandle frameConstants = pathTracer_.GetFrameConstants();

	GuideViewConstants constants = {};
	constants.renderSize = { pathTracer_.GetWidth(), pathTracer_.GetHeight() };
	constants.outputSize = { width_, height_ };
	constants.prevJitter = lastJitter_;
	constants.guide = static_cast<uint>(guide);
	constants.change = settings_.guideViewChange ? 1u : 0u;
	constants.gain = settings_.guideViewGain;
	constants.historyValid = lastGuide_ == guide ? 1u : 0u;
	constants.hdr10 = frame.rc.GetUseHDR() ? 1u : 0u;
	constants.paperWhiteNits = settings_.paperWhiteNits;
	constants.showHeld = settings_.guideViewHeld ? 1u : 0u;

	TextureDesc outputDesc = {};
	outputDesc.width = width_;
	outputDesc.height = height_;
	outputDesc.format = outputFormat_;
	outputDesc.debugName = "GuideView_Output";
	const rg::ResourceId output = frame.graph.CreateTexture(outputDesc);

	// Every input is bound as a UAV. The guide shown may be the motion vectors it also reads.
	const rg::ResourceId motionVectors = guides.motionVectors;
	rg::Pass pass = rg::Pass::Compute("GuideView");
	FrameGraph::Use(pass, source, rg::Access::ComputeShaderReadWrite);
	FrameGraph::Use(pass, motionVectors, rg::Access::ComputeShaderReadWrite);
	frame.UseTexture(pass, previous, "GuideView_History", rg::Access::ComputeShaderReadWrite);
	frame.UseTexture(pass, current, "GuideView_NewHistory", rg::Access::ComputeShaderReadWrite);
	frame.UseBuffer(pass, cameraState, "PathTracer_CameraState", rg::Access::ComputeShaderReadWrite);
	pass.Writes(output);
	frame.AddPass(std::move(pass.Executes(
			[this, constants, frameConstants, source, motionVectors, previous, current, cameraState, output](rg::PassContext &ctx)
			{
				ctx.rc.SetComputePipeline(pipeline_);
				ctx.rc.SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
				ctx.rc.SetComputeConstantBufferByName("g_Frame", frameConstants);
				ctx.rc.SetComputeUAV(0, ctx.Get(source));
				ctx.rc.SetComputeUAV(1, ctx.Get(motionVectors));
				ctx.rc.SetComputeUAV(2, previous);
				ctx.rc.SetComputeUAV(3, current);
				ctx.rc.SetComputeUAV(4, ctx.Get(output));
				ctx.rc.SetComputeUAV(5, cameraState);
				ctx.rc.Dispatch((width_ + kGroupSize - 1) / kGroupSize, (height_ + kGroupSize - 1) / kGroupSize, 1);
			})));

	const rg::ResourceId backBuffer = frame.backBuffer;
	frame.AddPass(rg::Pass::Copy("GuideView/Copy to back buffer")
								.Reads(output)
								.Writes(backBuffer)
								.Executes([output, backBuffer](rg::PassContext &ctx)
										  { ctx.rc.CopyTexture(ctx.Get(backBuffer), ctx.Get(output)); }));

	historyIndex_ ^= 1;
	lastGuide_ = guide;
	lastJitter_ = jitter;
}
