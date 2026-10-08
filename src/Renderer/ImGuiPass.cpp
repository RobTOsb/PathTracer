#include "ImGuiPass.h"

#include "Camera.h"
#include "LightTreeInterop.h"
#include "RenderContext.h"
#include "RenderSettings.h"
#include "Scene/RenderGPUScene.h"

#include "imgui.h"

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
	};

	constexpr ComboEntry<DLSSRayReconstructionPreset> kRRPresets[] = {
		{ DLSSRayReconstructionPreset::Default, "Default" },
		{ DLSSRayReconstructionPreset::PresetD, "D" },
		{ DLSSRayReconstructionPreset::PresetE, "E" },
		{ DLSSRayReconstructionPreset::PresetF, "F" },
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
					 const RenderGPUScene &scene) :
	settings_(settings), camera_(camera), rayReconstructionSupported_(rayReconstructionSupported), scene_(scene)
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
	ImGui::Text("%.1f fps (%.2f ms)  %ux%u -> %ux%u, 1 spp",
				io.Framerate,
				1000.0f / io.Framerate,
				renderContext->GetRenderWidth(),
				renderContext->GetRenderHeight(),
				renderContext->GetWindowWidth(),
				renderContext->GetWindowHeight());
	if (settings_.atmosphere.enabled)
	{
		ProfileRow(renderContext, "SkyAtmosphere");
	}
	ProfileRow(renderContext, "PathTracer");
	if (settings_.dlssMode != DLSSMode::Off)
	{
		ProfileRow(renderContext, "DLSS-RR");
	}
	ProfileRow(renderContext, "Tonemap");

	if (ImGui::CollapsingHeader("DLSS Ray Reconstruction", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (!rayReconstructionSupported_)
		{
			ImGui::TextDisabled("Not supported on this GPU/driver (see the NGX log)");
		}
		ImGui::BeginDisabled(!rayReconstructionSupported_);
		EnumCombo("Mode", kDLSSModes, settings_.dlssMode);
		EnumCombo("Preset", kRRPresets, settings_.rrPreset);
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
		ImGui::Text("%u emissive triangle(s), tree built in %.3f ms", lightCount, scene_.GetLightTreeBuildMs());
		ImGui::Text("Tree depth: %u max, %.1f average (4-wide)",
					scene_.GetLightTreeMaxDepth(),
					scene_.GetLightTreeAverageDepth());
		ImGui::BeginDisabled(lightCount == 0);
		ImGui::Checkbox("Enabled", &lights.enabled);
		EnumCombo("Sampling", kLightSamplingModes, lights.sampling);
		ImGui::BeginDisabled(lights.sampling != LIGHT_SAMPLING_TREE);
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
			ImGui::SliderFloat("Aerial perspective scale",
							   &atmosphere.aerialPerspectiveScale,
							   1.0f,
							   10000.0f,
							   "%.0f",
							   ImGuiSliderFlags_Logarithmic);
			ImGui::SetItemTooltip("How many times denser than real air the scene's air is, so that haze and light "
								  "shafts show at the scale of a building; 1 = physical");
			ImGui::SliderInt("In-scattering segments", &atmosphere.inScatterSegments, 0, 8);
			ImGui::SetItemTooltip("Path segments (1 = camera rays only) that gather the sunlight scattered "
								  "in the scene's air, at one shadow ray each; later ones are only dimmed");
			ImGui::Checkbox("Volumetric shadows", &atmosphere.volumetricShadows);
			ImGui::SetItemTooltip("Shadow rays from the in-scattering points towards the sun, so geometry casts "
								  "light shafts");
			ImGui::SliderFloat("Rayleigh scale", &atmosphere.rayleighScale, 0.0f, 10.0f, "%.2f");
			ImGui::SliderFloat("Mie scale", &atmosphere.mieScale, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
			ImGui::SetItemTooltip("Aerosols: haze, and the glow around the sun");
			ImGui::SliderFloat("Mie anisotropy", &atmosphere.miePhaseG, -0.99f, 0.99f, "%.2f");
			ImGui::SliderFloat("Ozone scale", &atmosphere.ozoneScale, 0.0f, 10.0f, "%.2f");
			ImGui::ColorEdit3("Ground albedo", &atmosphere.groundAlbedo.x);
			ImGui::SliderFloat("Multiple scattering", &atmosphere.multipleScatteringFactor, 0.0f, 2.0f, "%.2f");
		}
		else
		{
			ImGui::ColorEdit3("Sky colour", &settings_.skyColour.x);
			ImGui::DragFloat("Sky intensity", &settings_.skyIntensity, 0.01f, 0.0f, 100.0f, "%.2f");
		}
	}

	if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::SliderInt("Max bounces", &settings_.maxBounces, 0, 32);

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

		ImGui::Checkbox("Ray cone texture LOD", &settings_.rayConeTextureLod);
		ImGui::SetItemTooltip("Filters textures over each ray's cone footprint instead of always sampling the top mip");
		ImGui::BeginDisabled(!settings_.rayConeTextureLod);
		ImGui::SliderFloat("Texture LOD bias", &settings_.textureLodBias, -4.0f, 4.0f, "%.2f mips");
		ImGui::SetItemTooltip("On top of the DLSS bias, log2(render / output width)");
		ImGui::SliderFloat("Bounce cone spread", &settings_.rayConeBsdfSpread, 0.0f, 2.0f, "%.2f");
		ImGui::SetItemTooltip("How much each BSDF sample widens the cone, 1 = the solid angle the sample stands for, "
							  "0 = only the camera's pixel spread");
		ImGui::EndDisabled();

		ImGui::SliderFloat("Exposure", &settings_.exposureEV, -10.0f, 10.0f, "%.2f EV");

		if (renderContext->GetUseHDR())
		{
			ImGui::SliderFloat("Paper white", &settings_.paperWhiteNits, 80.0f, 400.0f, "%.0f nits");
			ImGui::SliderFloat("Peak luminance", &settings_.peakNits, 400.0f, 4000.0f, "%.0f nits");
			ImGui::SliderFloat("Roll-off knee", &settings_.rolloffKnee, 0.2f, 0.95f, "%.2f");
			ImGui::TextDisabled("HDR10 output: the tone mapper only applies to SDR");
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
	}

	if (ImGui::CollapsingHeader("Camera"))
	{
		const float3 position = camera_.GetPosition();
		const float3 front = camera_.GetFront();
		ImGui::Text("Position  %.2f %.2f %.2f", position.x, position.y, position.z);
		ImGui::Text("Forward   %.2f %.2f %.2f", front.x, front.y, front.z);
		ImGui::DragFloat("Speed", &settings_.cameraSpeed, 0.1f, 0.01f, 1000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::TextDisabled("WASD move, Q/E down/up, Shift faster, RMB drag to look");
	}

	ImGui::End();
}
