#include "ImGuiPass.h"

#include "Camera.h"
#include "FilmLut.h"
#include "FrameGraph.h"
#include "LensFlarePass.h"
#include "LightTreeInterop.h"
#include "PathTracerPass.h"
#include "RenderContext.h"
#include "RenderSettings.h"
#include "Scene/RenderGPUScene.h"

#include "imgui.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace
{
	struct ToneMapperEntry
	{
		ToneMappingOperator op;
		const char *name;
	};

	// Operators Tonemap.slang implements.
	constexpr ToneMapperEntry kToneMappers[] = {
		{ TONEMAP_ACES_FITTED, "ACES (fitted)" },
		{ TONEMAP_KHRONOS_PBR_NEUTRAL, "Khronos PBR Neutral" },
		{ TONEMAP_REINHARD, "Reinhard" },
		{ TONEMAP_NEUTWO, "Neutwo" },
		{ TONEMAP_OFF, "Off (clamp)" },
	};

	template<typename T>
	struct ComboEntry
	{
		T value;
		const char *name;
	};

	constexpr ComboEntry<GuideView> kGuideViews[] = {
		{ GUIDE_VIEW_OFF, "Off" },
		{ GUIDE_VIEW_DIFFUSE_ALBEDO, "Diffuse albedo" },
		{ GUIDE_VIEW_SPECULAR_ALBEDO, "Specular albedo" },
		{ GUIDE_VIEW_NORMAL, "Normal" },
		{ GUIDE_VIEW_ROUGHNESS, "Roughness" },
		{ GUIDE_VIEW_DEPTH, "Depth" },
		{ GUIDE_VIEW_MOTION_VECTORS, "Motion vectors" },
		{ GUIDE_VIEW_SPECULAR_MOTION_VECTORS, "Specular motion vectors" },
		{ GUIDE_VIEW_SHADOW_CHANGES, "Shadow changes" },
		{ GUIDE_VIEW_RESPONSIVITY, "Responsivity" },
		{ GUIDE_VIEW_PSR_OUTCOME, "PSR outcome" },
	};

	constexpr ComboEntry<DLSSMode> kDLSSModes[] = {
		{ DLSSMode::Off, "Off" },
		{ DLSSMode::DLAA, "DLAA" },
		{ DLSSMode::MaxQuality, "Quality" },
		{ DLSSMode::Balanced, "Balanced" },
		{ DLSSMode::MaxPerformance, "Performance" },
		{ DLSSMode::UltraPerformance, "Ultra Performance" },
	};

	constexpr ComboEntry<LightSamplingMode> kLightSamplingModes[] = {
		{ LIGHT_SAMPLING_TREE, "Light tree" },
		{ LIGHT_SAMPLING_UNIFORM, "Uniform" },
		{ LIGHT_SAMPLING_POWER, "Power" },
	};

	constexpr ComboEntry<DLSSRayReconstructionPreset> kRRPresets[] = {
		{ DLSSRayReconstructionPreset::Default, "Default" },
		{ DLSSRayReconstructionPreset::PresetD, "D" },
		{ DLSSRayReconstructionPreset::PresetE, "E" },
		{ DLSSRayReconstructionPreset::PresetF, "F" },
	};

	// The reference's paths per pixel to stop at, 0 = never stop.
	constexpr ComboEntry<int> kReferenceSampleCounts[] = {
		{ 64, "64" }, { 256, "256" }, { 1024, "1K" }, { 4096, "4K" }, { 16384, "16K" }, { 0, "Unlimited" },
	};

	template<typename T, size_t Count>
	void EnumCombo(const char *label, const ComboEntry<T> (&entries)[Count], T &value)
	{
		size_t current = 0;
		for (size_t i = 0; i < Count; ++i)
		{
			if (entries[i].value == value)
			{
				current = i;
			}
		}
		if (ImGui::BeginCombo(label, entries[current].name))
		{
			for (size_t i = 0; i < Count; ++i)
			{
				if (ImGui::Selectable(entries[i].name, i == current))
				{
					value = entries[i].value;
				}
			}
			ImGui::EndCombo();
		}
	}

	struct ByteString
	{
		char text[32];
	};

	ByteString FormatBytes(uint64_t bytes)
	{
		ByteString result;
		const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
		if (mib >= 1024.0)
		{
			snprintf(result.text, sizeof(result.text), "%.2f GB", mib / 1024.0);
		}
		else
		{
			snprintf(result.text, sizeof(result.text), "%.1f MB", mib);
		}
		return result;
	}

	void MemorySegmentRow(const char *name, const GPUMemorySegment &segment)
	{
		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		ImGui::TextUnformatted(name);
		ImGui::TableNextColumn();
		ImGui::TextUnformatted(FormatBytes(segment.allocator.allocationBytes).text);
		ImGui::TableNextColumn();
		ImGui::TextUnformatted(FormatBytes(segment.allocator.blockBytes).text);
		ImGui::TableNextColumn();
		ImGui::TextUnformatted(FormatBytes(segment.usageBytes).text);
		ImGui::TableNextColumn();
		ImGui::TextUnformatted(FormatBytes(segment.budgetBytes).text);
	}

	void MemoryUI(const GPUMemoryStats &stats)
	{
		const GPUMemorySegment &local = stats.local;
		const float fraction =
				local.budgetBytes > 0 ? static_cast<float>(static_cast<double>(local.usageBytes) / local.budgetBytes)
									  : 0.0f;
		char overlay[96];
		snprintf(overlay,
				 sizeof(overlay),
				 "%s / %s",
				 FormatBytes(local.usageBytes).text,
				 FormatBytes(local.budgetBytes).text);
		ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), overlay);
		ImGui::SetItemTooltip("This process's video memory against the budget the OS gives it, past which it "
							  "starts paging to system memory");

		constexpr ImGuiTableFlags tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
											   ImGuiTableFlags_SizingStretchProp;
		if (ImGui::BeginTable("Segments", 5, tableFlags))
		{
			ImGui::TableSetupColumn("Segment");
			ImGui::TableSetupColumn("In use");
			ImGui::TableSetupColumn("Reserved");
			ImGui::TableSetupColumn("Process");
			ImGui::TableSetupColumn("Budget");
			ImGui::TableHeadersRow();
			MemorySegmentRow(stats.unifiedMemory ? "Unified" : "Video", local);
			if (!stats.unifiedMemory)
			{
				MemorySegmentRow("System", stats.nonLocal);
			}
			ImGui::EndTable();
		}
		ImGui::TextDisabled("In use / reserved: the allocator's resources and the heaps holding them. "
							"Process: everything, swap chain and pipelines included.");
	}

	struct CountString
	{
		char text[32];
	};

	// With thousands separators: 2,500,000,000.
	CountString FormatCount(uint64_t count)
	{
		char digits[24];
		const int length = snprintf(digits, sizeof(digits), "%llu", static_cast<unsigned long long>(count));
		CountString result;
		int out = 0;
		for (int i = 0; i < length; ++i)
		{
			if (i > 0 && (length - i) % 3 == 0)
			{
				result.text[out++] = ',';
			}
			result.text[out++] = digits[i];
		}
		result.text[out] = '\0';
		return result;
	}

	void BeginStatRow(const char *name, const char *tooltip)
	{
		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		ImGui::TextUnformatted(name);
		if (tooltip)
		{
			ImGui::SetItemTooltip("%s", tooltip);
		}
		ImGui::TableNextColumn();
	}

	void CountRow(const char *name, uint64_t count, const char *tooltip = nullptr)
	{
		BeginStatRow(name, tooltip);
		ImGui::TextUnformatted(FormatCount(count).text);
	}

	void SceneUI(const RenderGPUScene &scene, const AnimationSettings &animation)
	{
		if (!scene.GetName().empty())
		{
			ImGui::TextUnformatted(scene.GetName().c_str());
		}

		const RenderGPUScene::GeometryStats &stats = scene.GetGeometryStats();
		constexpr ImGuiTableFlags tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
		if (ImGui::BeginTable("SceneStats", 2, tableFlags))
		{
			CountRow("Scene instances",
					 scene.GetInstanceCount(),
					 "Mesh nodes in the scene, each placing its mesh's primitives");
			CountRow("Meshes", scene.GetMeshDescCount());
			CountRow("Unique primitives",
					 stats.uniquePrimitives,
					 "Mesh primitives some instance places, each a BLAS of its own unmerged");
			CountRow("Unique triangles",
					 stats.uniqueTriangles,
					 "Triangles stored once per primitive, however many times it is placed");
			CountRow("Unique vertices", stats.uniqueVertices);
			CountRow("Primitive instances",
					 stats.primitiveInstances,
					 "Every primitive of every scene instance: the TLAS instances without merging");
			CountRow("Instanced triangles",
					 stats.instancedTriangles,
					 "Triangles summed over every placement, what the scene would be if nothing were shared");
			if (stats.uniqueTriangles > 0)
			{
				BeginStatRow("Instancing ratio", "Instanced triangles per unique triangle");
				ImGui::Text("%.1fx",
							static_cast<double>(stats.instancedTriangles) / static_cast<double>(stats.uniqueTriangles));
			}
			CountRow("TLAS instances", stats.tlasInstances);
			CountRow("BLASes", stats.blases, "Per primitive placed on its own, plus one per merged prototype");
			if (stats.prototypes > 0)
			{
				CountRow("Merged prototypes",
						 stats.prototypes,
						 "Multi-geometry BLASes standing for co-located static parts");
				CountRow("Merged placements",
						 stats.mergedPrimitiveInstances,
						 "Primitive instances inside a merged TLAS instance");
			}
			if (stats.micromapBlases > 0)
			{
				CountRow("Micromap BLASes",
						 stats.micromapBlases,
						 "Twins of the alpha-tested BLASes with their opacity micromaps linked");
				CountRow("Opacity micromaps", scene.GetOpacityMicromapCount());
			}
			CountRow("Materials", scene.GetMaterialCount(), "Including the default material");
			CountRow("Textures", scene.GetTextureCount());
			CountRow("Triangle lights", scene.GetTriangleLightCount(), "Emissive triangles, counted per placement");
			if (scene.GetTriangleLightCount() > scene.GetStaticLightCount())
			{
				CountRow("Dynamic triangle lights", scene.GetTriangleLightCount() - scene.GetStaticLightCount());
			}
			CountRow("Emissive instances", scene.GetEmissiveInstanceCount());
			CountRow("Point / spot lights", scene.GetPunctualLightCount());
			if (scene.GetDynamicInstanceCount() > 0)
			{
				CountRow("Animated instances", scene.GetDynamicInstanceCount());
			}
			if (animation.available)
			{
				BeginStatRow("Animation length", nullptr);
				ImGui::Text("%.2f s", animation.duration);
			}
			if (animation.skinCount > 0)
			{
				CountRow("Skins", animation.skinCount);
				CountRow("Joints", animation.jointCount);
				CountRow("Skinned primitives", scene.GetSkinnedPrimitiveCount());
				CountRow("Skinned vertices", animation.skinnedVertices);
			}
			if (scene.GetSceneCameraCount() > 0)
			{
				CountRow("Cameras", scene.GetSceneCameraCount());
			}
			const float3 size = scene.GetBoundsMax() - scene.GetBoundsMin();
			BeginStatRow("Bounds", "World-space size of the instanced geometry");
			ImGui::Text("%.1f x %.1f x %.1f", size.x, size.y, size.z);
			ImGui::EndTable();
		}
	}

	// The graph's pass timings, averaged over recent frames, in groups by the part of their names before '/': one row
	// per group, opening to its passes.
	void PassTimingTable(const std::vector<rg::PassTiming> &timings)
	{
		struct Group
		{
			std::string name;
			float averageMs = 0.0f;
			float medianMs = 0.0f;
			bool timed = false;
			std::vector<const rg::PassTiming *> passes;
		};
		std::vector<Group> groups;
		for (const rg::PassTiming &timing: timings)
		{
			const std::string name = timing.name.substr(0, timing.name.find('/'));
			auto it = std::find_if(groups.begin(), groups.end(), [&](const Group &g) { return g.name == name; });
			if (it == groups.end())
			{
				it = groups.insert(groups.end(), Group{ name });
			}
			it->passes.push_back(&timing);
			if (timing.stats.sampleCount > 0)
			{
				it->averageMs += timing.stats.averageMs;
				it->medianMs += timing.stats.medianMs;
				it->timed = true;
			}
		}

		auto timeColumns = [](bool timed, float averageMs, float medianMs)
		{
			ImGui::TableNextColumn();
			timed ? ImGui::Text("%6.2f", averageMs) : ImGui::TextUnformatted("     -");
			ImGui::TableNextColumn();
			timed ? ImGui::Text("%6.2f", medianMs) : ImGui::TextUnformatted("     -");
		};

		if (!ImGui::BeginTable("Pass timings", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg))
		{
			return;
		}
		ImGui::TableSetupColumn("GPU time", ImGuiTableColumnFlags_WidthStretch);
		ImGui::TableSetupColumn("Avg ms");
		ImGui::TableSetupColumn("Median ms");
		ImGui::TableHeadersRow();
		for (const Group &group: groups)
		{
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			const bool single = group.passes.size() == 1 && group.passes[0]->name == group.name;
			const ImGuiTreeNodeFlags leaf = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
			const bool open = ImGui::TreeNodeEx(group.name.c_str(), single ? leaf : ImGuiTreeNodeFlags_None);
			timeColumns(group.timed, group.averageMs, group.medianMs);
			if (single || !open)
			{
				continue;
			}
			for (const rg::PassTiming *pass: group.passes)
			{
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				const size_t slash = pass->name.find('/');
				ImGui::TreeNodeEx(pass->name.c_str(),
								  leaf,
								  "%s",
								  slash == std::string::npos ? pass->name.c_str() : pass->name.c_str() + slash + 1);
				timeColumns(pass->stats.sampleCount > 0, pass->stats.averageMs, pass->stats.medianMs);
			}
			ImGui::TreePop();
		}
		ImGui::EndTable();
	}
} // namespace

ImGuiPass::ImGuiPass(RenderSettings &settings,
					 const Camera &camera,
					 bool rayReconstructionSupported,
					 const RenderGPUScene &scene,
					 const PathTracerPass &pathTracer,
					 const rg::RenderGraph &graph) :
	settings_(settings), camera_(camera), rayReconstructionSupported_(rayReconstructionSupported), scene_(scene),
	pathTracer_(pathTracer), graph_(graph)
{
}

void ImGuiPass::AddPasses(FrameGraph &frame)
{
	const rg::ResourceId backBuffer = frame.backBuffer;
	frame.AddPass(rg::Pass::Graphics("ImGui")
								.Use(backBuffer, rg::Access::ColourAttachment)
								.Executes(
										[backBuffer](rg::PassContext &ctx)
										{
											const Resource renderTarget = ctx.GetResource(backBuffer);
											ctx.rc.SetRenderTargets(&renderTarget, 1);
											ctx.rc.ImguiRender();
										}));
}

void ImGuiPass::BuildStatsWindow(RenderContext *renderContext)
{
	// Its own window, top right, out of the way of the settings.
	const ImGuiIO &io = ImGui::GetIO();
	ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 16.0f, 16.0f), ImGuiCond_FirstUseEver, ImVec2(1.0f, 0.0f));
	if (!ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing))
	{
		ImGui::End();
		return;
	}

	ImGui::Text("%.1f fps (%.2f ms)  %ux%u -> %ux%u, %u spp",
				io.Framerate,
				1000.0f / io.Framerate,
				renderContext->GetRenderWidth(),
				renderContext->GetRenderHeight(),
				renderContext->GetWindowWidth(),
				renderContext->GetWindowHeight(),
				pathTracer_.IsReference() ? pathTracer_.GetAccumulatedSamples()
										  : static_cast<uint>(settings_.samplesPerPixel));
	const GPUMemoryStats memoryStats = renderContext->GetMemoryStats();
	ImGui::Text("VRAM %s / %s (allocator %s)",
				FormatBytes(memoryStats.local.usageBytes).text,
				FormatBytes(memoryStats.local.budgetBytes).text,
				FormatBytes(memoryStats.local.allocator.blockBytes).text);
	if (const std::optional<GPUProfileScopeStats> frame = renderContext->GetProfileScopeStats(kGpuFrameScope))
	{
		ImGui::Text("GPU frame %.2f ms (median %.2f)", frame->averageMs, frame->medianMs);
	}

	ImGui::Separator();
	const std::vector<rg::PassTiming> timings = graph_.GetPassTimings(*renderContext);
	ImGui::Text("Render graph: %zu passes in %u batches, %s transient",
				timings.size(),
				graph_.GetTotalBatchCount(),
				FormatBytes(graph_.GetTransientMemorySize()).text);
	PassTimingTable(timings);
	ImGui::End();
}

void ImGuiPass::BuildUI(RenderContext *renderContext)
{
	renderContext->ImguiNewFrame();
	BuildStatsWindow(renderContext);

	ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Path Tracer"))
	{
		ImGui::End();
		return;
	}

	// Without the approximations the reference ignores the firefly clamp and specular antialiasing settings.
	const bool reference = settings_.renderMode == RenderMode::Reference;
	const bool unbiasedReference = reference && !settings_.reference.approximations;

	if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
	{
		SceneUI(scene_, settings_.animation);
	}

	if (ImGui::CollapsingHeader("Render mode", ImGuiTreeNodeFlags_DefaultOpen))
	{
		int mode = static_cast<int>(settings_.renderMode);
		ImGui::RadioButton("Real-time", &mode, static_cast<int>(RenderMode::RealTime));
		ImGui::SetItemTooltip("One path per pixel per frame, denoised and upscaled by DLSS-RR unless its mode is off");
		ImGui::SameLine();
		ImGui::RadioButton("Reference", &mode, static_cast<int>(RenderMode::Reference));
		ImGui::SetItemTooltip("Paths accumulated at the window's resolution without DLSS, starting over whenever the "
							  "camera or a setting the image depends on changes");
		settings_.renderMode = static_cast<RenderMode>(mode);

		if (!reference)
		{
			ImGui::SliderInt("Samples per pixel##RealTime", &settings_.samplesPerPixel, 1, kMaxSamplesPerPixel);
			ImGui::SetItemTooltip("Paths per pixel each frame, averaged. Under DLSS-RR they all start through the "
								  "frame's jitter, so they cut the noise RR is given but add no antialiasing");
		}

		// The passes are rebuilt for a new mode before the next frame.
		if (reference && pathTracer_.IsReference())
		{
			ReferenceSettings &referenceSettings = settings_.reference;
			const uint samples = pathTracer_.GetAccumulatedSamples();
			const float seconds = pathTracer_.GetAccumulationSeconds();
			if (referenceSettings.samples > 0)
			{
				char overlay[64];
				snprintf(overlay, sizeof(overlay), "%u / %d spp, %.1f s", samples, referenceSettings.samples, seconds);
				ImGui::ProgressBar(static_cast<float>(samples) / static_cast<float>(referenceSettings.samples),
								   ImVec2(-FLT_MIN, 0.0f),
								   overlay);
			}
			else
			{
				ImGui::Text("%u spp, %.1f s", samples, seconds);
			}

			ImGui::TextUnformatted("Samples per pixel");
			for (size_t i = 0; i < IM_ARRAYSIZE(kReferenceSampleCounts); ++i)
			{
				if (i > 0)
				{
					ImGui::SameLine();
				}
				ImGui::RadioButton(kReferenceSampleCounts[i].name,
								   &referenceSettings.samples,
								   kReferenceSampleCounts[i].value);
			}
			ImGui::SliderInt("Samples per frame", &referenceSettings.samplesPerFrame, 1, 16);
			ImGui::SetItemTooltip("Paths per pixel each frame traces one after another; too many in a heavy scene "
								  "can make a frame long enough for the driver to reset the GPU");
			ImGui::Checkbox("Real-time approximations", &referenceSettings.approximations);
			ImGui::SetItemTooltip("Keep the firefly clamp and specular antialiasing, as in the real-time image. "
								  "Off: the reference converges to the unbiased image");
			if (ImGui::Button("Restart"))
			{
				++referenceSettings.restartCount;
			}
		}
	}

		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
	{
		const uint32_t stillCameras = scene_.GetStillCameraCount();
		if (stillCameras > 0)
		{
			char label[32];
			const auto viewName = [&label](uint32_t view) {
				if (view == 0)
				{
					return "User";
				}
				snprintf(label, sizeof(label), "Camera %u", view);
				return static_cast<const char *>(label);
			};
			if (ImGui::BeginCombo("View", viewName(settings_.view)))
			{
				for (uint32_t view = 0; view <= stillCameras; ++view)
				{
					if (ImGui::Selectable(viewName(view), settings_.view == view))
					{
						settings_.view = view;
					}
				}
				ImGui::EndCombo();
			}
			ImGui::SetItemTooltip("User is the free camera. Camera 1 and on are the scene's glTF cameras that do not "
								  "move, their field of view too; they ignore the mouse and keys, and back on User "
								  "the free camera is where it was left");
		}
		const float3 position = camera_.GetPosition();
		const float3 front = camera_.GetFront();
		ImGui::Text("Position  %.2f %.2f %.2f", position.x, position.y, position.z);
		ImGui::Text("Forward   %.2f %.2f %.2f", front.x, front.y, front.z);
		ImGui::DragFloat("Speed", &settings_.cameraSpeed, 0.1f, 0.01f, 1000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::TextDisabled("WASD move, Q/E down/up, Shift faster, RMB drag to look");
		if (ImGui::Button("Save view"))
		{
			settings_.cameraBookmarkRequest = RenderSettings::CameraBookmarkRequest::Save;
		}
		ImGui::SameLine();
		ImGui::BeginDisabled(!settings_.cameraBookmarkExists);
		if (ImGui::Button("Restore view"))
		{
			settings_.cameraBookmarkRequest = RenderSettings::CameraBookmarkRequest::Restore;
		}
		ImGui::EndDisabled();
		ImGui::TextDisabled("Saved beside the scene and restored at startup");
	}

	if (ImGui::CollapsingHeader("DLSS Ray Reconstruction", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (!rayReconstructionSupported_)
		{
			ImGui::TextDisabled("Not supported on this GPU/driver (see the NGX log)");
		}
		else if (reference)
		{
			ImGui::TextDisabled("Off while rendering the reference");
		}
		ImGui::BeginDisabled(!rayReconstructionSupported_ || reference);
		EnumCombo("Mode", kDLSSModes, settings_.dlssMode);
		EnumCombo("Preset", kRRPresets, settings_.rrPreset);
		ImGui::Checkbox("Primary surface replacement", &settings_.primarySurfaceReplacement);
		ImGui::BeginDisabled(!settings_.primarySurfaceReplacement);
		ImGui::SliderFloat("Mirror and glass roughness", &settings_.psrMaxRoughness, 0.0f, 0.3f, "%.3f");
		ImGui::Checkbox("Chains take the stronger lobe", &settings_.psrStrongerLobe);
		ImGui::SetItemTooltip("Past the first surface, the path through smooth glass always takes the lobe the guides "
							  "follow (mostly the refraction), so they never keep last frame's when it would have taken "
							  "the other. Leaves out that other lobe's light: the glass's faint internal reflections");
		ImGui::Checkbox("Curved mirrors and glass", &settings_.psrCurvedSurfaces);
		ImGui::SetItemTooltip("Motion vectors traced through the curvature of the mirrors and glass (ray differentials), "
							  "so curved ones keep PSR. Off: only those that barely bend the image keep it");
		ImGui::BeginDisabled(!settings_.psrCurvedSurfaces);
		ImGui::SliderFloat("Max minification", &settings_.psrMaxMinification, 1.0f, 64.0f, "%.1f");
		ImGui::SetItemTooltip("How far a curved chain may shrink the image of the surface it shows before PSR gives up "
							  "on it, its detail then being finer than a pixel");
		ImGui::EndDisabled();
		ImGui::Checkbox("Chains into the sky", &settings_.psrSky);
		ImGui::SetItemTooltip("A mirror or glass chain that leaves into the sky puts the sky in the guides' place, "
							  "moving as the chain turns it, instead of keeping the mirror's or glass's own motion. "
							  "Off: what such chains show (stars, distant lights through glass) moves with the surface");
		ImGui::EndDisabled();
		ImGui::SliderFloat("Specular MV max roughness", &settings_.specularMotionMaxRoughness, 0.0f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("Surfaces at most this rough get specular motion vectors that follow their reflection, "
							  "rougher ones the surface's own (Bevy Solari: 0.25). 1 = always the reflection");
		ImGui::Checkbox("Traced specular hit distance", &settings_.specularHitDistanceRay);
		ImGui::SetItemTooltip("Places the reflection for the specular motion vectors with a ray along the lobe's "
							  "dominant direction, the same every frame, rather than the path's sampled reflection. "
							  "PSR's virtual surfaces and glass panes keep the path's");
		ImGui::Checkbox("Hold specular hit distance", &settings_.holdSpecularHitDistance);
		ImGui::BeginDisabled(!settings_.holdSpecularHitDistance);
		ImGui::SliderFloat("Hold max roughness", &settings_.holdHitDistanceMaxRoughness, 0.0f, 1.0f, "%.2f");
		ImGui::EndDisabled();
		ImGui::Checkbox("Moving shadow mask", &settings_.shadowChangeMask);
		ImGui::SetItemTooltip("While the animation moves things, traces each pixel's light sample shadow ray in last "
							  "frame's TLAS too, and where its shadow changed tells DLSS-RR to favour the current frame "
							  "over its history, so moving shadows do not ghost");
		ImGui::BeginDisabled(settings_.rrPreset != DLSSRayReconstructionPreset::PresetF);
		ImGui::SliderFloat("Reflection responsivity", &settings_.reflectionResponsivity, 0.0f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("Preset F only. Tells DLSS-RR to keep less history on smooth reflectors the camera sees, "
							  "this much at a mirror, so bright reflections do not blur and trail across a polished "
							  "surface in motion. Too much may flicker. 0 = off");
		ImGui::SliderFloat("Responsivity max roughness", &settings_.reflectionResponsivityMaxRoughness, 0.01f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("The responsivity falls to none at reflectors this (perceptually) rough");
		ImGui::EndDisabled();
		ImGui::SliderInt("Guide samples", &settings_.guideSamples, 0, 8, settings_.guideSamples <= 0 ? "off" : "%d");
		ImGui::SetItemTooltip("Extra camera rays per pixel, around the jittered sample within a pixel's width, whose "
							  "surfaces the albedo, normal and roughness guides average in. Geometry finer than a "
							  "pixel (needles, distant twigs) then gives DLSS-RR a steady mix instead of flipping "
							  "between leaf and sky with the jitter, which it shows as crawl. Each ray is a closest hit "
							  "and a material load, no shading. Depth and motion stay the camera ray's. 0 = off");
		EnumCombo("Guide view", kGuideViews, settings_.guideView);
		ImGui::SetItemTooltip("Shows one of the guides given to DLSS-RR in place of the image, at render resolution");
		ImGui::BeginDisabled(settings_.guideView == GUIDE_VIEW_OFF);
		ImGui::Checkbox("Change since last frame", &settings_.guideViewChange);
		ImGui::SetItemTooltip("How much the guide differs from last frame's at the point the motion vectors lead to, "
							  "as heat: where DLSS-RR finds its history does not match. Depth is compared with where "
							  "last frame's camera would have seen the point. Blue: came from off screen");
		ImGui::Checkbox("Show held pixels", &settings_.guideViewHeld);
		ImGui::SetItemTooltip("Tints magenta the pixels that kept last frame's guides this frame, because the path took "
							  "the weaker lobe of a glass or mirror in their PSR chain");
		ImGui::SliderFloat("Guide view gain", &settings_.guideViewGain, 0.1f, 100.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
		ImGui::EndDisabled();
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Sun", ImGuiTreeNodeFlags_DefaultOpen))
	{
		SunSettings &sun = settings_.sun;
		// Drags of 0.1 degrees per pixel, finer than a slider's whole range across its width.
		ImGui::DragFloat("Azimuth", &sun.azimuthDegrees, 0.1f, -180.0f, 180.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
		// Below the horizon the atmosphere's planet hides the sun, leaving the twilight sky.
		const float minElevation = settings_.atmosphere.enabled ? -20.0f : 0.0f;
		ImGui::DragFloat("Elevation", &sun.elevationDegrees, 0.1f, minElevation, 180.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Angular radius",
						   &sun.angularRadiusDegrees,
						   0.0f,
						   10.0f,
						   "%.3f deg",
						   ImGuiSliderFlags_Logarithmic);
		ImGui::SameLine();
		if (ImGui::SmallButton("Physical"))
		{
			sun.angularRadiusDegrees = SUN_ANGULAR_RADIUS_DEFAULT * 180.0f / Pi;
		}
		ImGui::ColorEdit3("Colour", &sun.colour.x);
		ImGui::DragFloat("Intensity", &sun.intensity, 0.05f, 0.0f, 1000.0f, "%.2f");
	}

	if (ImGui::CollapsingHeader("Caustic photons", ImGuiTreeNodeFlags_DefaultOpen))
	{
		PhotonSettings &photons = settings_.photons;
		ImGui::Checkbox("Enabled##photons", &photons.enabled);
		ImGui::SetItemTooltip("Sun photons through smooth glass and mirrors, splatted as anisotropic footprints onto the "
							  "diffuse surfaces the camera sees directly (Adaptive Anisotropic Photon Scattering). The sun "
							  "disc found through glass after such a surface is left out, so it is not counted twice; "
							  "caustics seen through glass or in mirrors are the path tracer's own");
		ImGui::BeginDisabled(!photons.enabled);
		ImGui::SliderInt("Photons per frame", &photons.photonsPerFrame, 1024, 4194304, "%d", ImGuiSliderFlags_Logarithmic);
		ImGui::SliderFloat("Footprint scale##photons", &photons.footprintScale, 0.25f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("A photon's footprint is this many times the spacing to its neighbours on its branch across, "
							  "curved casters, rough lobes and the sun's size included: higher is smoother and blurrier");
		ImGui::SliderFloat("Max roughness##photons", &photons.maxRoughness, 0.0f, 0.5f, "%.2f");
		ImGui::SetItemTooltip("Lobes below this perceptual roughness pass photons, all of their light up to half of it and "
							  "less and less from there, the path tracer counting the rest; materials below it cast "
							  "caustics");
		ImGui::SliderInt("Max bounces##photons", &photons.maxBounces, 1, 16);
		ImGui::Checkbox("Cull casters to view", &photons.cullToView);
		ImGui::SetItemTooltip("The photon map spans only the casters whose caustics may land in view, so its texels are "
							  "finer; mirrors are always kept. A change in that set starts the map over");
		ImGui::Checkbox("Adaptive photon map", &photons.adaptive);
		ImGui::SetItemTooltip("Each part of the sun's view of the casters gets photons by feedback: more where they land "
							  "wide on screen, fewer where they land small or out of view, and a share kept for casters "
							  "out of view");
		ImGui::BeginDisabled(!photons.adaptive);
		ImGui::SliderFloat("Target area (pixels)", &photons.targetArea, 0.25f, 64.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("Pixels a photon should cover on screen: the map moves photons towards where they cover more");
		ImGui::SliderFloat("Temporal weight##photons", &photons.temporalWeight, 0.0f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("How fast the map follows its feedback, against its neighbourhood's last shares: lower is "
							  "steadier");
		ImGui::EndDisabled();
		if (ImGui::Button("Reset photon map"))
		{
			++photons.resetCount;
		}
		ImGui::EndDisabled();
	}

	if (settings_.animation.available && ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen))
	{
		AnimationSettings &animation = settings_.animation;
		ImGui::Text("%u instance(s) move", scene_.GetDynamicInstanceCount());
		if (animation.skinCount > 0)
		{
			ImGui::Text("%u skin(s), %u joint(s), %u skinned vertices", animation.skinCount, animation.jointCount,
						animation.skinnedVertices);
			ImGui::Text("Pose %.2f ms on the CPU, %.2f ms of it skinning", animation.evaluateMs, animation.skinningMs);
		}
		ImGui::Checkbox("Play", &animation.play);
		ImGui::SetItemTooltip(animation.skinCount > 0 ? "Rebuilds the TLAS and the skinned meshes' BLASes every frame"
													  : "Rebuilds the TLAS every frame");
		ImGui::SliderFloat("Playback speed", &animation.speed, 0.0f, 4.0f, "%.2fx");
		ImGui::SliderFloat("Time", &animation.seconds, 0.0f, animation.duration, "%.2f s");
		if (scene_.IsCameraAnimated())
		{
			ImGui::Checkbox("Scene camera", &animation.sceneCamera);
			ImGui::SetItemTooltip("While playing with the User view, it follows the scene's animated camera and "
								  "ignores the mouse and keys. Paused, the free camera carries on from there");
		}
	}

	if (ImGui::CollapsingHeader("Lights", ImGuiTreeNodeFlags_DefaultOpen))
	{
		LightSettings &lights = settings_.lights;
		const uint32_t lightCount = scene_.GetLightCount();
		ImGui::Text("%u emissive triangle(s), %u of them dynamic",
					scene_.GetTriangleLightCount(),
					scene_.GetTriangleLightCount() - scene_.GetStaticLightCount());
		ImGui::Text("%u point / spot light(s)", scene_.GetPunctualLightCount());
		ImGui::Text("Tree depth: %u (4-wide), rebuilt on the GPU every frame", scene_.GetLightTreeDepth());
		if (lights.bunnyCount > 0)
		{
			ImGui::Checkbox("Flash bunnies", &lights.flashBunnies);
			ImGui::SetItemTooltip("The %u emissive bunnies flash on their own, the violet one like a broken light",
								  lights.bunnyCount);
		}
		if (lights.clusterCount > 0)
		{
			ImGui::Checkbox("Flash light clusters", &lights.flashClusters);
			ImGui::SetItemTooltip("The %u clusters of emitters switch on and off at their own random times",
								  lights.clusterCount);
		}
		ImGui::BeginDisabled(lightCount == 0);
		ImGui::Checkbox("Enabled", &lights.enabled);
		EnumCombo("Sampling", kLightSamplingModes, lights.sampling);
		ImGui::BeginDisabled(lights.sampling != LIGHT_SAMPLING_TREE);
		ImGui::SliderInt("Light tree bounces", &lights.treeBounces, 0, 16);
		ImGui::SetItemTooltip("Path vertices (0 = primary hit) that select lights with the tree, later ones "
							  "select them in proportion to their flux");
		ImGui::Checkbox("Diffuse-only importance", &lights.diffuseOnlyRoughLobes);
		ImGui::SetItemTooltip("Skip the glossy light tree importance for lobes at least this rough");
		ImGui::SameLine();
		ImGui::BeginDisabled(!lights.diffuseOnlyRoughLobes);
		ImGui::SliderFloat("Min roughness", &lights.diffuseOnlyRoughness, 0.0f, 1.0f, "%.2f");
		ImGui::EndDisabled();
		ImGui::SliderInt("Glossy importance bounces", &lights.glossyImportanceBounces, 0, 16);
		ImGui::SetItemTooltip("Path vertices (0 = primary hit) that get the glossy light tree importance, "
							  "later ones use the diffuse one only");
		ImGui::Checkbox("SG importance", &lights.sgImportance);
		ImGui::SetItemTooltip("Rank light tree nodes by their spherical Gaussian lighting (diffuse and anisotropic "
							  "glossy integrals) instead of the cheap bound over the cone to each node, oriented by "
							  "the normal. More accurate, much more expensive; compiles its own pipeline");
		ImGui::BeginDisabled(!lights.sgImportance);
		ImGui::Checkbox("Far-field importance", &lights.farFieldImportance);
		ImGui::SetItemTooltip("SG importance: treat small, distant light clusters wholly above the horizon as point "
							  "lights");
		ImGui::SameLine();
		ImGui::BeginDisabled(!lights.farFieldImportance);
		ImGui::SliderFloat("Min sharpness", &lights.farFieldSharpness, 1.0f, 1000.0f, "%.0f",
						   ImGuiSliderFlags_Logarithmic);
		ImGui::EndDisabled();
		ImGui::EndDisabled();
		ImGui::SliderInt("Lights per leaf", &lights.maxLeafLights, 1, static_cast<int>(LIGHT_TREE_MAX_LEAF_LIGHTS));
		ImGui::SetItemTooltip("Light tree leaf size; a leaf's lights are picked in proportion to their flux");
		ImGui::SliderFloat("Power mix", &lights.treePowerMix, 0.0f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("Share of the tree's light samples that select by power instead, weighed by both "
							  "PMFs: caps the noise of lights the tree underrates (point lights off glossy lobes "
							  "or right next to a surface). 0 = the tree alone");
		ImGui::EndDisabled();
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Specular antialiasing", ImGuiTreeNodeFlags_DefaultOpen))
	{
		SpecularAASettings &specularAA = settings_.specularAA;
		ImGui::BeginDisabled(unbiasedReference);
		ImGui::Checkbox("Enabled##SpecularAA", &specularAA.enabled);
		ImGui::SetItemTooltip("Widens the glossy lobes of the surfaces the camera sees, directly and in mirrors and "
							  "through glass, along each axis by how much their normals turn across the pixel's footprint "
							  "(DLSS-RR builds only), so sub-pixel detail stops speckling. Seen directly (and inside the "
							  "object seen), lobes with roughness 0 are left alone, so mirrors and smooth glass stay sharp");
		ImGui::BeginDisabled(!specularAA.enabled);
		ImGui::SliderFloat("Strength##SpecularAA", &specularAA.strength, 0.0f, 4.0f, "%.2f");
		ImGui::Checkbox("Normal map variance", &specularAA.normalMaps);
		ImGui::SetItemTooltip("Toksvig: also widen by the spread of normals that a normal map's mips average away");
		ImGui::Checkbox("Glass##SpecularAA", &specularAA.glass);
		ImGui::SetItemTooltip("Also widen glass, so finely rippled or distant glass stops speckling. Glass widened past "
							  "PSR's roughness limit loses its PSR guides there");
		ImGui::SliderFloat("Sharp radius", &specularAA.sharpRadiusPixels, 0.0f, 32.0f, "%.1f px");
		ImGui::SetItemTooltip("Curvature with a radius above this many pixels stays sharp (left to the jitter, as for "
							  "mirror balls); tighter curvature, like thin tubes and ripples, is widened. 0 widens all");
		ImGui::SliderFloat("Max alpha^2 added", &specularAA.maxVariance, 0.0f, 1.0f, "%.3f");
		ImGui::SetItemTooltip("Cap on what each axis of the lobe's GGX alpha^2 grows by");
		ImGui::EndDisabled();
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Sky", ImGuiTreeNodeFlags_DefaultOpen))
	{
		AtmosphereSettings &atmosphere = settings_.atmosphere;
		ImGui::Checkbox("Atmosphere", &atmosphere.enabled);
		ImGui::SetItemTooltip("Physically based sky and aerial perspective (Hillaire 2020) lit by the sun, "
							  "instead of a constant sky colour");
		if (atmosphere.enabled)
		{
			ImGui::SliderFloat("Altitude", &atmosphere.altitudeKm, 0.0f, 50.0f, "%.2f km");
			ImGui::SetItemTooltip("Of the scene's origin above the ground");
			ImGui::SliderFloat("Scene air scale", &atmosphere.sceneAirScale, 0.0f, 10000.0f, "%.4g", ImGuiSliderFlags_Logarithmic);
			ImGui::SetItemTooltip("How many times denser than real air the scene's molecules are (Rayleigh scattering "
								  "and ozone): a nearly even blue fog; 1 = physical, the sky is unaffected");
			ImGui::SliderFloat("Scene haze scale", &atmosphere.sceneHazeScale, 0.0f, 10000.0f, "%.4g", ImGuiSliderFlags_Logarithmic);
			ImGui::SetItemTooltip("How many times denser than real air the scene's aerosols are (Mie scattering): the "
								  "dust and haze that show light shafts; 1 = physical, the sky is unaffected");
			ImGui::SliderInt("In-scattering segments", &atmosphere.inScatterSegments, 0, 8);
			ImGui::SetItemTooltip("Path segments (1 = camera rays only) that gather the sunlight scattered "
								  "in the scene's air, at one shadow ray each; later ones are only dimmed");
			ImGui::Checkbox("Volumetric shadows", &atmosphere.volumetricShadows);
			ImGui::SetItemTooltip("Shadow rays from the in-scattering points towards the sun, so geometry casts "
								  "light shafts");
			ImGui::Checkbox("Sky visibility", &atmosphere.skyVisibility);
			ImGui::SetItemTooltip("The air's multiple-scattered light, that of an open sky, counts only where a "
								  "shadow ray in a random direction reaches the sky, so enclosed spaces stay clear");
			if (atmosphere.skyVisibility)
			{
				ImGui::SliderFloat("Sky visibility distance", &atmosphere.skyVisibilityDistance, 0.0f, 10000.0f, "%.0f m",
								   ImGuiSliderFlags_Logarithmic);
				ImGui::SetItemTooltip("How far the sky visibility ray looks for occluders, 0 for all the way to the sky. "
									  "It is traced only on segments the camera looks along");
			}
			ImGui::Checkbox("Fog-aware guides", &atmosphere.fogGuides);
			ImGui::SetItemTooltip("DLSS-RR's guides fade with the air's transmittance to the surface: towards a white "
								  "albedo, no reflection and full roughness as haze covers it, so the haze's light is "
								  "denoised as a smooth signal instead of lighting on the surface behind it");
			ImGui::Checkbox("Sky sampling", &atmosphere.skySampling);
			ImGui::SetItemTooltip("Light samples also go to the sky without the sun, in proportion to its brightness "
								  "(an importance map rebuilt each frame), weighed against BSDF sampling. Off leaves the "
								  "sky to BSDF sampling alone");
			if (atmosphere.skySampling)
			{
				ImGui::SliderFloat("Min sky sample share", &atmosphere.skySampleShare, 0.0f, 1.0f, "%.2f");
				ImGui::SetItemTooltip("The sun and the sky split their light samples by how much light each gives the "
									  "surface (the sun by its angle and the clouds' shadow); the sky keeps at least this "
									  "part, and all of them once the sun has set");
			}
			ImGui::SliderFloat("Rayleigh scale", &atmosphere.rayleighScale, 0.0f, 10.0f, "%.2f");
			ImGui::SliderFloat("Mie scale", &atmosphere.mieScale, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
			ImGui::SetItemTooltip("Aerosols: haze, and the glow around the sun");
			ImGui::SliderFloat("Mie anisotropy", &atmosphere.miePhaseG, -0.99f, 0.99f, "%.2f");
			ImGui::SliderFloat("Ozone scale", &atmosphere.ozoneScale, 0.0f, 10.0f, "%.2f");
			ImGui::ColorEdit3("Ground albedo", &atmosphere.groundAlbedo.x);
			ImGui::SliderFloat("Multiple scattering", &atmosphere.multipleScatteringFactor, 0.0f, 2.0f, "%.2f");

			CloudSettings &clouds = settings_.clouds;
			ImGui::Checkbox("Clouds", &clouds.enabled);
			ImGui::SetItemTooltip("Volumetric clouds that the sky is seen through and that shadow the sun");
			if (clouds.enabled)
			{
				ImGui::SliderFloat("Cloud base", &clouds.bottomKm, 0.0f, 10.0f, "%.2f km");
				ImGui::SliderFloat("Cloud top", &clouds.topKm, 0.0f, 15.0f, "%.2f km");
				ImGui::SetItemTooltip("Only cumulus reach it, stratus stay in the lowest 30%% of the layer");
				ImGui::SliderFloat("Coverage", &clouds.coverage, 0.0f, 1.0f, "%.2f");
				ImGui::SetItemTooltip("Added to the weather map's, from patchy at 0 to overcast at 1");
				ImGui::SliderFloat("Cloud density", &clouds.density, 0.01f, 100.0f, "%.3g /km", ImGuiSliderFlags_Logarithmic);
				ImGui::SetItemTooltip("Extinction where the cloud is densest");
				ImGui::SliderFloat("Wind speed", &clouds.windSpeed, 0.0f, 100.0f, "%.1f m/s");
				ImGui::SetItemTooltip("Still in the reference");
				ImGui::SliderFloat("Wind direction", &clouds.windDirectionDegrees, -180.0f, 180.0f, "%.0f deg");
				ImGui::SliderInt("View steps", &clouds.viewSteps, 1, 256);
				ImGui::SetItemTooltip("Most march steps through the clouds of rays the camera sees through (0.2 km apart "
									  "where fewer reach across the layer)");
				ImGui::SliderInt("Indirect steps", &clouds.indirectSteps, 1, 128);
				ImGui::SetItemTooltip("Most march steps through the clouds of every other ray that leaves the scene (with "
									  "the indirect panorama, only in the reference)");
				ImGui::Checkbox("Indirect panorama", &clouds.indirectPanorama);
				ImGui::SetItemTooltip("Rays whose sky the camera does not see, not even in a mirror, read the cloudy sky "
									  "from a panorama rendered from the camera each frame instead of marching the clouds");
			}
		}
		else
		{
			ImGui::ColorEdit3("Sky colour", &settings_.skyColour.x);
			ImGui::DragFloat("Sky intensity", &settings_.skyIntensity, 0.01f, 0.0f, 100.0f, "%.2f");
		}
	}

	if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::SliderInt("Diffuse bounces", &settings_.maxDiffuseBounces, 0, kMaxBounces);
		ImGui::SetItemTooltip("Indirect bounces off diffuse, diffuse transmission, subsurface and fuzz lobes");
		ImGui::SliderInt("Specular bounces", &settings_.maxSpecularBounces, 0, kMaxBounces);
		ImGui::SetItemTooltip("Indirect bounces off specular reflection and transmission and coat lobes, and through "
							  "thin smooth walls");
		ImGui::BeginDisabled(unbiasedReference);
		ImGui::Checkbox("Clamp indirect", &settings_.clampIndirect);
		ImGui::SetItemTooltip("Firefly clamp: each indirect light contribution is scaled down to at most this in its "
							  "brightest channel (biased darker)");
		ImGui::BeginDisabled(!settings_.clampIndirect);
		ImGui::SliderFloat("Indirect clamp", &settings_.indirectClampValue, 0.1f, 1000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::EndDisabled();
		ImGui::Checkbox("Clamp direct", &settings_.clampDirect);
		ImGui::SetItemTooltip("Firefly clamp for the light the visible surface scatters (sparkling highlights on glossy "
							  "metal and rough glass). Lights seen directly or in mirror-like surfaces are not clamped");
		ImGui::BeginDisabled(!settings_.clampDirect);
		ImGui::SliderFloat("Direct clamp", &settings_.directClampValue, 0.1f, 1000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::EndDisabled();
		ImGui::Checkbox("Firefly filter", &settings_.fireflyFilter);
		ImGui::SetItemTooltip("RTXPT's firefly filter: the light a path finds is scaled down to at most the threshold "
							  "times the exposure's middle grey, times a path probability heuristic that shrinks with "
							  "every bounce, the more the wider and less likely the sample. Lights seen directly or in "
							  "mirror-like surfaces are left alone. Biased darker");
		ImGui::BeginDisabled(!settings_.fireflyFilter);
		ImGui::SliderFloat("Firefly threshold", &settings_.fireflyFilterThreshold, 0.001f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("Higher clamps less. Better light sampling allows a higher threshold");
		ImGui::EndDisabled();
		ImGui::EndDisabled();
		ImGui::SliderFloat("Min reflection probability", &settings_.reflectionMinProbability, 0.0f, 0.95f, "%.2f");
		ImGui::SetItemTooltip("Visible surfaces with a sharp reflection sample it at least this often, the other lobes "
							  "less: small bright reflections (lights in a polished floor) show in more frames, and the "
							  "diffuse lobe, which denoises better, gets the noise instead. Unbiased. 0 = off");
		ImGui::BeginDisabled(settings_.reflectionMinProbability <= 0.0f);
		ImGui::SliderFloat("Max roughness##ReflectionMinProbability",
						   &settings_.reflectionMinProbabilityMaxRoughness, 0.0f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("Reflections at most this rough (perceptual) get the minimum probability; rougher ones "
							  "are found by light sampling well enough");
		ImGui::EndDisabled();

		const ShaderExecutionReorderingSupport ser = renderContext->GetShaderExecutionReorderingSupport();
		ImGui::BeginDisabled(!ser.supported);
		ImGui::Checkbox("Shader execution reordering", &settings_.shaderExecutionReordering);
		ImGui::EndDisabled();
		ImGui::SetItemTooltip("DXR 1.2 SER pipeline: reorders bounce hits by material before shading them");
		if (!ser.supported)
		{
			ImGui::TextDisabled("Needs shader model 6.9 and raytracing tier 1.2");
		}
		else if (!ser.actuallyReorders)
		{
			ImGui::TextDisabled("The driver does not actually reorder");
		}

		ImGui::Checkbox("Inline shadow rays", &settings_.inlineShadowRays);
		ImGui::SetItemTooltip("Shadow rays as inline ray queries, keeping the path's state in registers instead of "
							  "saving it around a TraceRay");

		ImGui::Checkbox("Async compute", &settings_.asyncCompute);
		ImGui::SetItemTooltip("The sky and cloud passes on the async compute queue, overlapping the acceleration "
							  "structure update and light tree build on the graphics queue, where the path tracing and "
							  "everything after it then run. Off: every pass on the graphics queue");

		ImGui::Checkbox("Merge co-located instances", &settings_.mergeInstances);
		ImGui::SetItemTooltip("Static primitives with the same world matrix (the parts of one tree, say) as one TLAS "
							  "instance of a BLAS with a geometry per part, instead of an instance and BLAS each, "
							  "so rays enter one BVH per object rather than one per part. Rebuilds every BLAS and "
							  "TLAS when changed");

		const bool micromaps = scene_.HasOpacityMicromaps();
		ImGui::BeginDisabled(!micromaps);
		ImGui::Checkbox("Opacity micromaps", &settings_.opacityMicromaps);
		ImGui::SetItemTooltip("DXR 1.2 opacity micromaps on the alpha-tested and blended geometry: the alpha test "
							  "only runs where an alpha edge crosses a micro-triangle, or a blended surface is partly "
							  "there");
		ImGui::BeginDisabled(!settings_.opacityMicromaps);
		ImGui::SliderInt("OMM 2-state from bounce",
						 &settings_.opacityMicromap2StateBounce,
						 -1,
						 8,
						 settings_.opacityMicromap2StateBounce < 0 ? "never" : "%d");
		ImGui::SetItemTooltip("Rays from path vertices at this bounce on (0 = camera rays) read the masks' micromaps "
							  "in 2-state mode and skip their alpha test entirely. Blended surfaces keep theirs. Shadow "
							  "rays always read every micromap in 2-state mode");
		ImGui::EndDisabled();
		ImGui::EndDisabled();
		if (!micromaps)
		{
			ImGui::TextDisabled(renderContext->SupportsOpacityMicromaps() ? "The scene has no alpha-tested geometry"
																		  : "Needs shader model 6.9 and raytracing tier 1.2");
		}

		ImGui::SliderInt("Opaque alpha after diffuse bounces", &settings_.opaqueAlphaDiffuseBounces, 0, 8,
						 settings_.opaqueAlphaDiffuseBounces <= 0 ? "off" : "%d");
		ImGui::SetItemTooltip("Rays leaving a path after this many diffuse bounces skip blended surfaces and treat "
							  "alpha-tested ones as opaque, with no alpha test. Biased: cutouts may block light there");
		ImGui::SliderInt("Reduced materials after diffuse bounces", &settings_.reducedMaterialDiffuseBounces, 0, 8,
						 settings_.reducedMaterialDiffuseBounces <= 0 ? "off" : "%d");
		ImGui::SetItemTooltip("Surfaces reached after this many diffuse bounces skip their normal maps and shade solid "
							  "subsurface as plain diffuse instead of scattering light through the object");

		ImGui::Checkbox("Specular", &settings_.specularMaterials);
		ImGui::SetItemTooltip("Off: no specular reflection, metals turn diffuse and glass only refracts");
		ImGui::SameLine();
		ImGui::Checkbox("Transmission", &settings_.transmissiveMaterials);
		ImGui::SetItemTooltip("Off: transmissive materials turn opaque");
		ImGui::Checkbox("Blended shadows", &settings_.blendedShadows);
		ImGui::SetItemTooltip("Alpha-blended surfaces cast shadows, by chance at their alpha. Off for decals, which "
							  "would shadow the wall they lie on");
		ImGui::SameLine();
		ImGui::Checkbox("Blended guides", &settings_.blendGuides);
		ImGui::SetItemTooltip("DLSS-RR guides mix an alpha-blended surface with what is behind it by its opacity, "
							  "instead of showing whichever layer the camera ray took (noisy albedo on decals)");
		ImGui::SliderInt("Glass reflection splits", &settings_.glassReflectionSplits, 0, 4);
		ImGui::SetItemTooltip("Smooth glass in front of the camera, thin panes or solid, whose reflection is traced as "
							  "well as what it lets through, instead of one or the other. 0 = off");

		ImGui::TextUnformatted("Subsurface scattering");
		ImGui::Checkbox("Zero##Subsurface", &settings_.subsurfaceZeroScattering);
		ImGui::SetItemTooltip("Light straight through the medium, path traced");
		ImGui::SameLine();
		ImGui::Checkbox("Single##Subsurface", &settings_.subsurfaceSingleScattering);
		ImGui::SetItemTooltip("Light scattered once in the medium, path traced");
		ImGui::SameLine();
		ImGui::Checkbox("Multiple##Subsurface", &settings_.subsurfaceMultipleScattering);
		ImGui::SetItemTooltip("Light scattered more than once, from a diffusion profile with single scattering "
							  "taken out. All three off: OpenPBR's Lambertian stand-in");
		ImGui::SliderFloat("Subsurface radius scale",
						   &settings_.subsurfaceRadiusScale,
						   0.01f,
						   10.0f,
						   "%.2f",
						   ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("Multiplies every material's mean free path");
		ImGui::SliderFloat("Diffuse roughness", &settings_.diffuseRoughnessScale, 0.0f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("The base's diffuse roughness (EON) as this fraction of its specular roughness, "
							  "0 = Lambertian");

		ImGui::Checkbox("Ray cone texture LOD", &settings_.rayConeTextureLod);
		ImGui::SetItemTooltip("Filters textures over each ray's cone footprint instead of always sampling the top mip");
		ImGui::BeginDisabled(!settings_.rayConeTextureLod);
		ImGui::SliderFloat("Texture LOD bias", &settings_.textureLodBias, -4.0f, 4.0f, "%.2f mips");
		ImGui::SetItemTooltip("On top of the DLSS bias, log2(render / output width)");
		ImGui::SliderFloat("Bounce cone spread", &settings_.rayConeBsdfSpread, 0.0f, 2.0f, "%.2f");
		ImGui::SetItemTooltip("How much each BSDF sample widens the cone, 1 = the solid angle the sample stands for, "
							  "0 = only the camera's pixel spread");
		ImGui::SliderInt("Anisotropic segments", &settings_.anisotropicSegments, 0,
						 settings_.maxDiffuseBounces + settings_.maxSpecularBounces + 1);
		ImGui::SetItemTooltip("Path segments whose hits filter textures anisotropically (16x), the rest trilinearly. "
							  "0 = none, 1 = camera hits only");
		ImGui::EndDisabled();

		ImGui::Checkbox("Auto exposure", &settings_.autoExposure);
		ImGui::SetItemTooltip("Exposure from each frame's luminance histogram, adapting over time");
		ImGui::SliderFloat(settings_.autoExposure ? "Exposure compensation" : "Exposure", &settings_.exposureEV, -10.0f,
						   10.0f, "%.2f EV");
		if (settings_.autoExposure)
		{
			ImGui::SliderFloat("Max auto exposure", &settings_.maxAutoExposureEV, -10.0f, 20.0f, "%.2f EV");
			ImGui::SetItemTooltip("The most the auto exposure may brighten the image, compensation included");
		}

		if (renderContext->GetUseHDR())
		{
			ImGui::SliderFloat("Paper white", &settings_.paperWhiteNits, 80.0f, 400.0f, "%.0f nits");
			ImGui::SliderFloat("Peak luminance", &settings_.peakNits, 400.0f, 4000.0f, "%.0f nits");
			ImGui::Checkbox("Gamma correction (2.2)", &settings_.hdrGammaCorrection);
			ImGui::SetItemTooltip("Re-encodes with sRGB and decodes with a 2.2 power, as SDR displays do, so "
								  "shadows are not lifted compared with the SDR output");
		}

		int current = 0;
		for (int i = 0; i < IM_ARRAYSIZE(kToneMappers); ++i)
		{
			if (kToneMappers[i].op == settings_.toneMapOperator)
			{
				current = i;
			}
		}
		if (ImGui::BeginCombo("Tone mapper", kToneMappers[current].name))
		{
			for (int i = 0; i < IM_ARRAYSIZE(kToneMappers); ++i)
			{
				if (ImGui::Selectable(kToneMappers[i].name, i == current))
				{
					settings_.toneMapOperator = kToneMappers[i].op;
				}
			}
			ImGui::EndCombo();
		}
		if (renderContext->GetUseHDR())
		{
			ImGui::SetItemTooltip("HDR10: applied as a grade below paper white, Neutwo maps the result to the peak");
		}

		// Neutwo is always the final curve in HDR; in SDR only when it is the selected operator.
		if (renderContext->GetUseHDR() ? settings_.toneMapOperator != TONEMAP_OFF
									   : settings_.toneMapOperator == TONEMAP_NEUTWO)
		{
			ImGui::SliderFloat("White clip", &settings_.whiteClip, 1.0f, 1000.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
			ImGui::SetItemTooltip("Scene value (1 = paper white) that the Neutwo curve maps exactly to its peak");
		}

		{
			const std::vector<FilmLutFile> &luts = GetFilmLutFiles();
			settings_.filmLut = std::clamp(settings_.filmLut, 0, static_cast<int>(luts.size()));
			const char *preview = settings_.filmLut == 0 ? "None" : luts[settings_.filmLut - 1].name.c_str();
			if (ImGui::BeginCombo("Film LUT", preview))
			{
				if (ImGui::Selectable("None", settings_.filmLut == 0))
				{
					settings_.filmLut = 0;
				}
				for (int i = 0; i < static_cast<int>(luts.size()); ++i)
				{
					if (ImGui::Selectable(luts[i].name.c_str(), settings_.filmLut == i + 1))
					{
						settings_.filmLut = i + 1;
					}
				}
				ImGui::EndCombo();
			}
			ImGui::SetItemTooltip("A 3D look LUT from assets/luts (made from .cube files by tools/cube_to_dds.py), "
								  "applied to the display-encoded SDR image after the tone mapper. In HDR the graded "
								  "image sets the mid-tones and the highlights extend above paper white; with the tone "
								  "mapper off in HDR it is not applied");
			ImGui::BeginDisabled(settings_.filmLut == 0);
			ImGui::SliderFloat("LUT strength", &settings_.filmLutStrength, 0.0f, 1.0f, "%.2f");
			ImGui::SetItemTooltip("Blend from the tone mapper's output (0) to the graded image (1)");
			ImGui::EndDisabled();
		}

		ImGui::Checkbox("Bloom", &settings_.bloom);
		ImGui::BeginDisabled(!settings_.bloom);
		ImGui::SliderFloat("Bloom strength", &settings_.bloomStrength, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("How much of the image the bloom replaces; it moves light around without adding any");
		ImGui::SliderInt("Bloom levels", &settings_.bloomLevels, 1, kMaxBloomLevels);
		ImGui::SetItemTooltip("Half-resolution levels averaged into the bloom, each twice as blurry as the last");
		ImGui::Checkbox("Bloom the sun", &settings_.bloomSun);
		ImGui::SetItemTooltip("Off: the sun disc seen directly is left out of the bloom, its reflections still bloom");
		ImGui::EndDisabled();

		LensFlareSettings &flare = settings_.lensFlare;
		ImGui::Checkbox("Lens flare", &flare.enabled);
		ImGui::SetItemTooltip("Ghosts traced through a real lens prescription and the iris's diffraction starburst, "
							  "for the brightest lights in view");
		ImGui::BeginDisabled(!flare.enabled);
		ImGui::Combo(
				"Flare lens",
				&flare.lens,
				[](void *, int index) { return LensFlarePass::GetLensName(index); },
				nullptr,
				LensFlarePass::GetLensCount());
		ImGui::SetItemTooltip("The lens the ghosts are traced through: its focal length, open f-number and (surface "
							  "count). More surfaces give more ghosts; long lenses give small, tight ones. The f-number "
							  "slider cannot open a lens past its own");
		ImGui::SliderFloat("Flare min intensity", &flare.minIntensity, 0.001f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("A light flares only once its irradiance at the lens, exposed like the image, reaches this "
							  "luminance; it fades in up to twice it. Occluded lights are dimmed first");
		ImGui::SliderFloat("Flare compression", &flare.compression, 0.0f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("How a flare grows with its light's brightness past the minimum intensity: 1 = in "
							  "proportion, 0.5 = as its square root, 0 = every flare as bright as one at the minimum");
		ImGui::SliderInt("Flare max lights", &flare.maxLights, 1, FLARE_MAX_LIGHTS);
		ImGui::SetItemTooltip("The brightest lights above the minimum that get a flare; each costs about the same");
		ImGui::SliderFloat("Flare f-number", &flare.fNumber, 2.0f, 16.0f, "f/%.1f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("Closing the iris clips the ghosts smaller and spreads the starburst wider");
		ImGui::SliderInt("Iris blades", &flare.apertureBlades, 0, 32);
		ImGui::SetItemTooltip("The iris polygon the ghosts and the starburst take their shape from, below 3 = round");
		ImGui::SliderFloat("Ghost intensity", &flare.ghostIntensity, 0.0f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("1 = what the lens's coated surfaces reflect");
		ImGui::SliderFloat("Ghost size", &flare.ghostSize, 0.05f, 1.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("Each ghost's size about its centre, 1 = what the lens gives; smaller ghosts are brighter");
		ImGui::SliderFloat("Ghost softness", &flare.ghostSoftness, 0.0f, 0.5f, "%.3f");
		ImGui::SetItemTooltip("How far inside its edge each ghost fades, as a share of the iris; 0 = sharp outlines");
		ImGui::SliderInt("Ghost blur passes", &flare.ghostBlurPasses, 0, 6);
		ImGui::SetItemTooltip("Dual Kawase blur over the ghosts (not the starbursts): each pass doubles its reach. "
							  "0 = off");
		ImGui::BeginDisabled(flare.ghostBlurPasses <= 0);
		ImGui::SliderFloat("Ghost blur offset", &flare.ghostBlurOffset, 0.0f, 4.0f, "%.2f");
		ImGui::SetItemTooltip("How far apart the blur's taps are, in half texels of each pass; larger is wider but can "
							  "show a grid pattern past about 2");
		ImGui::EndDisabled();
		ImGui::SliderFloat("Starburst intensity", &flare.starburstIntensity, 0.0f, 4.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("Share of the light's irradiance spread into the starburst");
		ImGui::SliderFloat("Starburst size", &flare.starburstSize, 0.05f, 2.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("Half the starburst's side over half the screen's height, at f/8; it scales with the f-number");
		ImGui::EndDisabled();

		ImGui::Checkbox("Film grain", &settings_.filmGrain);
		ImGui::BeginDisabled(!settings_.filmGrain);
		ImGui::SliderFloat("Grain strength", &settings_.filmGrainStrength, 0.0f, 0.2f, "%.3f");
		ImGui::SetItemTooltip("Standard deviation of the grain in sRGB-encoded units at mid-grey");
		ImGui::SliderFloat("Grain size", &settings_.filmGrainSize, 1.0f, 4.0f, "%.2f px");
		ImGui::SliderFloat("Grain colour", &settings_.filmGrainColour, 0.0f, 1.0f, "%.2f");
		ImGui::SetItemTooltip("0 = monochrome (luminance) grain, 1 = independent grain per colour channel");
		ImGui::SliderFloat("Grain mid-tone bias", &settings_.filmGrainResponse, 0.0f, 4.0f, "%.2f");
		ImGui::SetItemTooltip("How strongly the grain fades out towards black and white, 0 = the same grain everywhere");
		ImGui::SliderFloat("Grain frame rate", &settings_.filmGrainFps, 0.0f, 60.0f, "%.0f fps");
		ImGui::SetItemTooltip("How often the grain pattern changes, 0 = static");
		ImGui::EndDisabled();

		ImGui::Checkbox("Vignette", &settings_.vignette);
		ImGui::BeginDisabled(!settings_.vignette);
		ImGui::SliderFloat("Vignette strength", &settings_.vignetteStrength, 0.0f, 2.0f, "%.2f");
		ImGui::SetItemTooltip("How strongly the image darkens towards the corners; the centre is unchanged, 0 = none");
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Memory", ImGuiTreeNodeFlags_DefaultOpen))
	{
		MemoryUI(renderContext->GetMemoryStats());
	}

	ImGui::End();
}
