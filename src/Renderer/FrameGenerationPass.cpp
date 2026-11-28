#include "FrameGenerationPass.h"

#include "D3D12RenderContext.h"
#include "Streamline.h"

#include "FrameGraph.h"
#include "PathTracerPass.h"

#include "fmtlog.h"

#include <stdexcept>

namespace
{
	constexpr uint32_t kGroupSize = 8;

	// UIComposite.slang's.
	struct UICompositeConstants
	{
		uint32_t width;
		uint32_t height;
	};
} // namespace

FrameGenerationPass::FrameGenerationPass(RenderContext *renderContext,
										 Streamline &streamline,
										 const PathTracerPass &pathTracer) :
	streamline_(streamline), pathTracer_(pathTracer)
{
	compositePipeline_ = renderContext->CreateComputePipeline(PATHTRACER_SHADER_DIR "UIComposite.slang", "main");
	if (compositePipeline_ == InvalidPipelineHandle)
	{
		throw std::runtime_error("Failed to create the UI composite pipeline (see the shader log above).");
	}
	format_ = renderContext->GetBackBufferFormat();
	width_ = renderContext->GetWindowWidth();
	height_ = renderContext->GetWindowHeight();
}

void FrameGenerationPass::AddPasses(FrameGraph &frame)
{
	TextureDesc desc = {};
	desc.width = width_;
	desc.height = height_;
	desc.format = format_;
	desc.debugName = "FrameGeneration_Scene";
	const rg::ResourceId scene = frame.graph.CreateTexture(desc);
	desc.debugName = "FrameGeneration_UI";
	const rg::ResourceId ui = frame.graph.CreateTexture(desc);
	const rg::ResourceId backBuffer = frame.backBuffer;
	const PathTracerGuides guides = pathTracer_.GetGuides();

	frame.AddPass(rg::Pass::Copy("FrameGeneration/Copy scene")
						  .Reads(backBuffer)
						  .Writes(scene)
						  .Executes([backBuffer, scene](rg::PassContext &ctx)
									{ ctx.rc.CopyTexture(ctx.Get(scene), ctx.Get(backBuffer)); }));

	frame.AddPass(rg::Pass::Graphics("FrameGeneration/UI")
						  .Use(ui, rg::Access::ColourAttachment)
						  .Executes(
								  [ui](rg::PassContext &ctx)
								  {
									  const Resource renderTarget = ctx.GetResource(ui);
									  float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
									  ctx.rc.ClearRenderTarget(renderTarget, clear);
									  ctx.rc.SetRenderTargets(&renderTarget, 1);
									  ctx.rc.ImguiRender();
								  }));

	// Streamline copies all four as they are tagged, in the copy-source layout this pass puts them in.
	FrameGenerationInputs inputs;
	inputs.renderWidth = pathTracer_.GetWidth();
	inputs.renderHeight = pathTracer_.GetHeight();
	inputs.outputWidth = width_;
	inputs.outputHeight = height_;
	frame.AddPass(
			rg::Pass::Copy("FrameGeneration/Tag inputs")
					.Reads(guides.depth)
					.Reads(guides.motionVectors)
					.Reads(scene)
					.Reads(ui)
					.Executes(
							[this, &graph = frame.graph, inputs, guides, scene, ui](rg::PassContext &ctx) mutable
							{
								D3D12RenderContext *d3dContext = static_cast<D3D12RenderContext *>(&ctx.rc);
								auto native = [&](rg::ResourceId resource)
								{ return d3dContext->GetNativeResource(graph.GetHandle(resource)); };
								inputs.depth = native(guides.depth);
								inputs.motionVectors = native(guides.motionVectors);
								inputs.hudless = native(scene);
								inputs.ui = native(ui);
								const bool tagged =
										streamline_.TagFrameGeneration(inputs,
																	   d3dContext->GetNativeGraphicsCommandList());
								// In case Streamline's copies bound heaps of their own.
								ctx.rc.SetDescriptorHeaps();
								if (!tagged && !reportedFailure_)
								{
									loge("DLSS Frame Generation's inputs could not be tagged (see the Streamline log "
										 "above)");
									reportedFailure_ = true;
								}
							}));

	const UICompositeConstants constants = { width_, height_ };
	frame.AddPass(rg::Pass::Compute("FrameGeneration/UI composite")
						  .Use(ui, rg::Access::ComputeShaderRead)
						  .Use(scene, rg::Access::ComputeShaderReadWrite)
						  .Executes(
								  [this, constants, ui, scene](rg::PassContext &ctx)
								  {
									  ctx.rc.SetComputePipeline(compositePipeline_);
									  ctx.rc.SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
									  ctx.rc.SetComputeSRV(0, ctx.Get(ui));
									  ctx.rc.SetComputeUAV(0, ctx.Get(scene));
									  ctx.rc.Dispatch((width_ + kGroupSize - 1) / kGroupSize,
													  (height_ + kGroupSize - 1) / kGroupSize,
													  1);
								  }));

	frame.AddPass(rg::Pass::Copy("FrameGeneration/Copy to back buffer")
						  .Reads(scene)
						  .Writes(backBuffer)
						  .Executes([scene, backBuffer](rg::PassContext &ctx)
									{ ctx.rc.CopyTexture(ctx.Get(backBuffer), ctx.Get(scene)); }));
}
