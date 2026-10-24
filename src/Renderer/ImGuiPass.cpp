#include "ImGuiPass.h"

#include "Camera.h"
#include "LightTreeInterop.h"
#include "PathTracerPass.h"
#include "RenderContext.h"
#include "RenderSettings.h"
#include "Scene/RenderGPUScene.h"

#include "imgui.h"

#include <cfloat>
#include <cstdio>

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

	void ProfileRow(RenderContext *renderContext, const char *scope)
	{
		if (const GPUProfileScopeStats *stats = renderContext->GetProfileScopeStats(scope))
		{
			ImGui::Text("%-12s %6.2f ms (median %.2f)", scope, stats->averageMs, stats->medianMs);
		}
	}
} // namespace

ImGuiPass::ImGuiPass(RenderSettings &settings,
					 const Camera &camera,
					 bool rayReconstructionSupported,
					 const RenderGPUScene &scene,
					 const PathTracerPass &pathTracer) :
	settings_(settings), camera_(camera), rayReconstructionSupported_(rayReconstructionSupported), scene_(scene),
	pathTracer_(pathTracer)
{
}

void ImGuiPass::Execute(RenderContext *renderContext)
{
	renderContext->ImguiNewFrame();
	BuildUI(renderContext);

	renderContext->BeginEvent("ImGui");

	const ResourceHandle backBuffer = renderContext->GetCurrentBackBufferHandle();
	renderContext->TransitionResource(backBuffer,
									  ResourceLayout::RenderTarget,
									  ResourceSync::RenderTarget,
									  ResourceAccess::RenderTarget);

	const Resource renderTarget = renderContext->GetCurrentBackBuffer();
	renderContext->SetRenderTargets(&renderTarget, 1);
	renderContext->ImguiRender();

	renderContext->EndEvent();
}

void ImGuiPass::BuildUI(RenderContext *renderContext)
{
	ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Path Tracer"))
	{
		ImGui::End();
		return;
	}

	const ImGuiIO &io = ImGui::GetIO();
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
	if (settings_.atmosphere.enabled)
	{
		ProfileRow(renderContext, "SkyAtmosphere");
	}
	ProfileRow(renderContext, "PathTracer");
	if (settings_.dlssMode != DLSSMode::Off)
	{
		ProfileRow(renderContext, "DLSS-RR");
	}
	ProfileRow(renderContext, "Bloom");
	ProfileRow(renderContext, "Tonemap");

	// Without the approximations the reference ignores the firefly clamp and specular antialiasing settings.
	const bool reference = settings_.renderMode == RenderMode::Reference;
	const bool unbiasedReference = reference && !settings_.reference.approximations;

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
		ImGui::Checkbox("Curved mirrors and glass", &settings_.psrCurvedSurfaces);
		ImGui::SetItemTooltip("Motion vectors traced through the curvature of the mirrors and glass (ray differentials), "
							  "so curved ones keep PSR. Off: only those that barely bend the image keep it");
		ImGui::BeginDisabled(!settings_.psrCurvedSurfaces);
		ImGui::SliderFloat("Max minification", &settings_.psrMaxMinification, 1.0f, 16.0f, "%.1f");
		ImGui::SetItemTooltip("How far a curved chain may shrink the image of the surface it shows before PSR gives up "
							  "on it, its detail then being finer than a pixel");
		ImGui::EndDisabled();
		ImGui::EndDisabled();
		ImGui::Checkbox("Hold specular hit distance", &settings_.holdSpecularHitDistance);
		ImGui::BeginDisabled(!settings_.holdSpecularHitDistance);
		ImGui::SliderFloat("Hold max roughness", &settings_.holdHitDistanceMaxRoughness, 0.0f, 1.0f, "%.2f");
		ImGui::EndDisabled();
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Sun", ImGuiTreeNodeFlags_DefaultOpen))
	{
		SunSettings &sun = settings_.sun;
		ImGui::SliderFloat("Azimuth", &sun.azimuthDegrees, -180.0f, 180.0f, "%.1f deg");
		// Below the horizon the atmosphere's planet hides the sun, leaving the twilight sky.
		const float minElevation = settings_.atmosphere.enabled ? -20.0f : 0.0f;
		ImGui::SliderFloat("Elevation", &sun.elevationDegrees, minElevation, 180.0f, "%.1f deg");
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
		if (settings_.atmosphere.enabled)
		{
			ImGui::TextDisabled("At the top of the atmosphere, which tints and dims it");
		}
	}

	if (ImGui::CollapsingHeader("Lights", ImGuiTreeNodeFlags_DefaultOpen))
	{
		LightSettings &lights = settings_.lights;
		const uint32_t lightCount = scene_.GetLightCount();
		const uint32_t dynamicLightCount = lightCount - scene_.GetStaticLightCount();
		ImGui::Text("%u static emissive triangle(s), tree built in %.3f ms",
					scene_.GetStaticLightCount(),
					scene_.GetStaticLightTreeBuildMs());
		if (scene_.GetDynamicInstanceCount() > 0)
		{
			ImGui::Text("%u dynamic emissive triangle(s), tree built in %.3f ms",
						dynamicLightCount,
						scene_.GetDynamicLightTreeBuildMs());
			ImGui::Checkbox("Animate dynamic emitters", &lights.animateDynamic);
			ImGui::SetItemTooltip("Move the dynamic emitters round a circle, which rebuilds their light tree and the "
								  "TLAS every frame");
		}
		ImGui::Text("Tree depth: %u max, %.1f average (4-wide)",
					scene_.GetLightTreeMaxDepth(),
					scene_.GetLightTreeAverageDepth());
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
		ImGui::SliderInt("Lights per leaf", &lights.maxLeafLights, 1, static_cast<int>(LIGHT_TREE_MAX_LEAF_LIGHTS));
		ImGui::SetItemTooltip("Light tree leaf size; a leaf's lights are picked in proportion to their flux");
		ImGui::Checkbox("Orientation splits", &lights.orientationSplits);
		ImGui::SetItemTooltip("Light tree build: keep emitters that face the same way together, by also weighing "
							  "the spread of each side's normals in the split cost");
		ImGui::Checkbox("Far-field importance", &lights.farFieldImportance);
		ImGui::SetItemTooltip("Treat small, distant light clusters wholly above the horizon as point lights");
		ImGui::SameLine();
		ImGui::BeginDisabled(!lights.farFieldImportance);
		ImGui::SliderFloat("Min sharpness",
						   &lights.farFieldSharpness,
						   1.0f,
						   1000.0f,
						   "%.0f",
						   ImGuiSliderFlags_Logarithmic);
		ImGui::EndDisabled();
		ImGui::EndDisabled();
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Specular antialiasing", ImGuiTreeNodeFlags_DefaultOpen))
	{
		SpecularAASettings &specularAA = settings_.specularAA;
		ImGui::BeginDisabled(unbiasedReference);
		ImGui::Checkbox("Enabled##SpecularAA", &specularAA.enabled);
		ImGui::SetItemTooltip("Widens the glossy lobes of opaque surfaces the camera sees by how much their normals "
							  "turn across the pixel (DLSS-RR builds only), so sub-pixel highlights stop flickering");
		ImGui::BeginDisabled(!specularAA.enabled);
		ImGui::SliderFloat("Strength##SpecularAA", &specularAA.strength, 0.0f, 4.0f, "%.2f");
		ImGui::Checkbox("Normal map variance", &specularAA.normalMaps);
		ImGui::SetItemTooltip("Toksvig: also widen by the spread of normals that a normal map's mips average away");
		ImGui::SliderFloat("Max alpha^2 added", &specularAA.maxVariance, 0.0f, 1.0f, "%.3f");
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
				ImGui::SetItemTooltip("Most march steps through the clouds of every other ray that leaves the scene");
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

		const bool micromaps = scene_.HasOpacityMicromaps();
		ImGui::BeginDisabled(!micromaps);
		ImGui::Checkbox("Opacity micromaps", &settings_.opacityMicromaps);
		ImGui::SetItemTooltip("DXR 1.2 opacity micromaps on the alpha-tested geometry: the alpha test only runs "
							  "where an alpha edge crosses a micro-triangle");
		ImGui::BeginDisabled(!settings_.opacityMicromaps);
		ImGui::SliderInt("OMM 2-state from bounce",
						 &settings_.opacityMicromap2StateBounce,
						 -1,
						 8,
						 settings_.opacityMicromap2StateBounce < 0 ? "never" : "%d");
		ImGui::SetItemTooltip("Rays from path vertices at this bounce on (0 = camera rays) read the micromaps in "
							  "2-state mode and skip the alpha test entirely");
		ImGui::EndDisabled();
		ImGui::EndDisabled();
		if (!micromaps)
		{
			ImGui::TextDisabled(renderContext->SupportsOpacityMicromaps() ? "The scene has no alpha-tested geometry"
																		  : "Needs shader model 6.9 and raytracing tier 1.2");
		}

		ImGui::Checkbox("Specular", &settings_.specularMaterials);
		ImGui::SetItemTooltip("Off: no specular reflection, metals turn diffuse and glass only refracts");
		ImGui::SameLine();
		ImGui::Checkbox("Transmission", &settings_.transmissiveMaterials);
		ImGui::SetItemTooltip("Off: transmissive materials turn opaque");
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

		ImGui::SliderFloat("Exposure", &settings_.exposureEV, -10.0f, 10.0f, "%.2f EV");

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

		ImGui::Checkbox("Bloom", &settings_.bloom);
		ImGui::BeginDisabled(!settings_.bloom);
		ImGui::SliderFloat("Bloom strength", &settings_.bloomStrength, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip("How much of the image the bloom replaces; it moves light around without adding any");
		ImGui::SliderInt("Bloom levels", &settings_.bloomLevels, 1, kMaxBloomLevels);
		ImGui::SetItemTooltip("Half-resolution levels averaged into the bloom, each twice as blurry as the last");
		ImGui::Checkbox("Bloom the sun", &settings_.bloomSun);
		ImGui::SetItemTooltip("Off: the sun disc seen directly is left out of the bloom, its reflections still bloom");
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
	}

	if (ImGui::CollapsingHeader("Memory"))
	{
		MemoryUI(memoryStats);
	}

	if (ImGui::CollapsingHeader("Camera"))
	{
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

	ImGui::End();
}
