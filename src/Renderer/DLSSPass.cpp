#include "DLSSPass.h"

#include "D3D12RenderContext.h"

#include "Camera.h"
#include "FrameGraph.h"
#include "PathTracerPass.h"
#include "RenderSettings.h"

#include "fmtlog.h"

#include <cstring>

DLSSPass::DLSSPass(RenderContext *renderContext,
				   Streamline &streamline,
				   const PathTracerPass &pathTracer,
				   const Camera &camera,
				   const RenderSettings &settings) :
	renderContext_(renderContext), streamline_(streamline), pathTracer_(pathTracer), camera_(camera),
	settings_(settings)
{
	mode_ = settings.dlssMode;
	preset_ = settings.rrPreset;

	outputDesc_.width = renderContext->GetWindowWidth();
	outputDesc_.height = renderContext->GetWindowHeight();
	outputDesc_.format = Format::R16G16B16A16_FLOAT;
	outputDesc_.usageFlags = ResourceUsageFlags::UnorderedAccess;
	outputDesc_.debugName = "DLSS_Output";
}

DLSSPass::~DLSSPass() = default;

void DLSSPass::AddPasses(FrameGraph &frame)
{
	// DLSS reads its inputs from compute shaders and writes the output as a UAV; Streamline is told they are in those
	// states, so it transitions nothing and leaves them as they were.
	// The reflections' motion comes as motion vectors, never as a hit distance for RR to derive it from.
	// Streamline is handed the resources themselves, so it reads the frame's transients wherever the graph placed them.
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

	RayReconstructionInputs eval;
	eval.mode = mode_;
	eval.preset = preset_;
	eval.renderWidth = pathTracer_.GetWidth();
	eval.renderHeight = pathTracer_.GetHeight();
	eval.outputWidth = outputDesc_.width;
	eval.outputHeight = outputDesc_.height;
	// DLSS wants the jitter as applied to the projection, which shifts the image the opposite
	// way to the offset the camera rays were sampled at. The projection is unjittered.
	const float2 jitter = pathTracer_.GetJitter();
	eval.jitterX = -jitter.x;
	eval.jitterY = -jitter.y;
	// Row-major with row vectors, as DirectXMath stores them.
	const float4x4 view = camera_.GetViewMatrix();
	const float4x4 projection = camera_.GetProjectionMatrix();
	const float4x4 prevView = camera_.GetPreviousViewMatrix();
	const float4x4 prevProjection = camera_.GetPreviousProjectionMatrix();
	static_assert(sizeof(view) == sizeof(eval.worldToView));
	std::memcpy(eval.worldToView, &view, sizeof(eval.worldToView));
	std::memcpy(eval.viewToClip, &projection, sizeof(eval.viewToClip));
	std::memcpy(eval.prevWorldToView, &prevView, sizeof(eval.prevWorldToView));
	std::memcpy(eval.prevViewToClip, &prevProjection, sizeof(eval.prevViewToClip));
	const float3 position = camera_.GetPosition();
	const float3 up = camera_.GetUp();
	const float3 right = camera_.GetRight();
	const float3 forward = camera_.GetFront();
	std::memcpy(eval.cameraPosition, &position, sizeof(eval.cameraPosition));
	std::memcpy(eval.cameraUp, &up, sizeof(eval.cameraUp));
	std::memcpy(eval.cameraRight, &right, sizeof(eval.cameraRight));
	std::memcpy(eval.cameraForward, &forward, sizeof(eval.cameraForward));
	eval.nearZ = camera_.GetNearZ();
	eval.farZ = camera_.GetFarZ();
	eval.fovY = camera_.GetFovY();
	const bool responsivity =
			preset_ == DLSSRayReconstructionPreset::PresetF && settings_.reflectionResponsivity != 0.0f;
	const bool cameraCut = settings_.cameraCut;

	frame.AddPass(std::move(pass.Executes(
			[this, &graph = frame.graph, eval, responsivity, cameraCut, guides, radiance, output](
					rg::PassContext &ctx) mutable
			{
				// Streamline records into the native command list (its proxy), and main only ever creates the D3D12
				// backend.
				D3D12RenderContext *d3dContext = static_cast<D3D12RenderContext *>(&ctx.rc);
				ID3D12GraphicsCommandList *commandList = d3dContext->GetNativeGraphicsCommandList();
				auto native = [&](rg::ResourceId resource)
				{ return d3dContext->GetNativeResource(graph.GetHandle(resource)); };

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
				eval.reset = reset_ || cameraCut;

				const bool evaluated = streamline_.EvaluateRayReconstruction(eval, commandList);

				// DLSS binds its own descriptor heaps, root signature and pipeline. Every later pass binds its
				// own pipeline, but the heaps are only set once per frame.
				ctx.rc.SetDescriptorHeaps();

				if (evaluated)
				{
					reset_ = false;
				}
				else if (!reportedFailure_)
				{
					loge("DLSS Ray Reconstruction failed to run (see the Streamline log above); the output will be "
						 "stale.");
					reportedFailure_ = true;
				}
			})));
}
