#include "DLSSPass.h"

#include "D3D12RenderContext.h"

#include "Camera.h"
#include "FrameGraph.h"
#include "PathTracerPass.h"
#include "RenderSettings.h"

#include "fmtlog.h"

#include <cstring>

DLSSPass::DLSSPass(RenderContext *renderContext,
				   NGXContext &ngx,
				   const PathTracerPass &pathTracer,
				   const Camera &camera,
				   const RenderSettings &settings) :
	renderContext_(renderContext), ngx_(ngx), pathTracer_(pathTracer), camera_(camera), settings_(settings)
{
	// main rebuilds the pass when the mode or preset changes.
	featureParams_.kind = DLSSFeatureKind::RayReconstruction;
	featureParams_.mode = settings.dlssMode;
	featureParams_.rrPreset = settings.rrPreset;
	featureParams_.renderWidth = pathTracer.GetWidth();
	featureParams_.renderHeight = pathTracer.GetHeight();
	featureParams_.outputWidth = renderContext->GetWindowWidth();
	featureParams_.outputHeight = renderContext->GetWindowHeight();
	featureParams_.hdr = true; // linear radiance, exposed by the tonemapper afterwards
	featureParams_.depthInverted = true; // the camera uses reversed Z
	featureParams_.packedNormalRoughness = true; // roughness in normalRoughness.w

	outputDesc_.width = featureParams_.outputWidth;
	outputDesc_.height = featureParams_.outputHeight;
	outputDesc_.format = Format::R16G16B16A16_FLOAT;
	outputDesc_.usageFlags = ResourceUsageFlags::UnorderedAccess;
	outputDesc_.debugName = "DLSS_Output";
}

DLSSPass::~DLSSPass() = default;

void DLSSPass::AddPasses(FrameGraph &frame)
{
	// DLSS reads its inputs from compute shaders and writes the output as a UAV, and returns every
	// resource to these states when it is done.
	// The reflections' motion comes as motion vectors, never as a hit distance for RR to derive it from.
	// NGX is handed the resources themselves, so it reads the frame's transients wherever the graph placed them.
	const PathTracerGuides guides = pathTracer_.GetGuides();
	const rg::ResourceId radiance = pathTracer_.GetOutput();
	output_ = frame.graph.CreateTexture(outputDesc_, ResourceViewType::UnorderedAccess);
	const rg::ResourceId output = output_;
	rg::Pass pass = rg::Pass::Compute("DLSS-RR");
	for (rg::ResourceId input: { radiance,
								 guides.diffuseAlbedo,
								 guides.specularAlbedo,
								 guides.normalRoughness,
								 guides.motionVectors,
								 guides.depth,
								 guides.specularMotionVectors,
								 guides.shadowChanges,
								 guides.responsivity })
	{
		FrameGraph::Use(pass, input, rg::Access::ComputeShaderRead);
	}
	pass.Use(output, rg::Access::ComputeShaderWrite);

	// DLSS wants the jitter as applied to the projection, which shifts the image the opposite
	// way to the offset the camera rays were sampled at. The projection is unjittered.
	const float2 jitter = pathTracer_.GetJitter();
	const float4x4 view = camera_.GetViewMatrix();
	const float4x4 projection = camera_.GetProjectionMatrix();
	const bool responsivity =
			featureParams_.rrPreset == DLSSRayReconstructionPreset::PresetF && settings_.reflectionResponsivity != 0.0f;
	const bool cameraCut = settings_.cameraCut;

	frame.AddPass(std::move(pass.Executes(
			[this, &graph = frame.graph, jitter, view, projection, responsivity, cameraCut, guides, radiance,
			 output](rg::PassContext &ctx)
			{
				// NGX records straight into the native command list, and main only ever creates the D3D12 backend.
				D3D12RenderContext *d3dContext = static_cast<D3D12RenderContext *>(&ctx.rc);
				ID3D12GraphicsCommandList *commandList = d3dContext->GetNativeGraphicsCommandList();
				auto native = [&](rg::ResourceId resource) { return d3dContext->GetNativeResource(graph.GetHandle(resource)); };

				bool evaluated = ngx_.EnsureFeature(featureParams_, commandList);
				if (evaluated)
				{
					DLSSEvalInputs eval = {};
					eval.colourInput = native(radiance);
					eval.colourOutput = native(output);
					eval.depth = native(guides.depth);
					eval.motionVectors = native(guides.motionVectors);
					eval.diffuseAlbedo = native(guides.diffuseAlbedo);
					eval.specularAlbedo = native(guides.specularAlbedo);
					eval.normalRoughness = native(guides.normalRoughness);
					eval.specularMotionVectors = native(guides.specularMotionVectors);
					eval.biasCurrentColourMask = native(guides.shadowChanges);
					if (responsivity)
					{
						eval.responsivityMask = native(guides.responsivity);
					}
					eval.renderWidth = featureParams_.renderWidth;
					eval.renderHeight = featureParams_.renderHeight;
					eval.reset = reset_ || cameraCut;
					eval.jitterOffsetX = -jitter.x;
					eval.jitterOffsetY = -jitter.y;

					// Row-major with row vectors, as DirectXMath stores them.
					static_assert(sizeof(view) == sizeof(eval.worldToView));
					std::memcpy(eval.worldToView, &view, sizeof(eval.worldToView));
					std::memcpy(eval.viewToClip, &projection, sizeof(eval.viewToClip));

					evaluated = ngx_.EvaluateFeature(eval, commandList);
				}

				// NGX binds its own descriptor heaps, root signature and pipeline. Every later pass binds its
				// own pipeline, but the heaps are only set once per frame.
				ctx.rc.SetDescriptorHeaps();

				if (evaluated)
				{
					reset_ = false;
				}
				else if (!reportedFailure_)
				{
					loge("DLSS Ray Reconstruction failed to run (see the NGX log above); the output will be stale.");
					reportedFailure_ = true;
				}
			})));
}
