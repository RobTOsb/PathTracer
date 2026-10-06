#include "ImGuiPass.h"

#include "Camera.h"
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
		ImGui::SliderFloat("Elevation", &sun.elevationDegrees, 0.0f, 90.0f, "%.1f deg");
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

	if (ImGui::CollapsingHeader("Lights", ImGuiTreeNodeFlags_DefaultOpen))
	{
		LightSettings &lights = settings_.lights;
		const uint32_t lightCount = scene_.GetLightCount();
		ImGui::Text("%u point/area light(s), tree built in %.3f ms", lightCount, scene_.GetLightTreeBuildMs());
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
		ImGui::EndDisabled();
		ImGui::DragFloat("Light intensity",
						 &lights.intensityScale,
						 0.01f,
						 0.0f,
						 1000.0f,
						 "%.2f",
						 ImGuiSliderFlags_Logarithmic);
		ImGui::Checkbox("Animate", &lights.animate);
		ImGui::SameLine();
		ImGui::BeginDisabled(!lights.animate);
		ImGui::DragFloat("Speed", &lights.animationSpeed, 0.01f, 0.0f, 10.0f, "%.2f");
		ImGui::EndDisabled();
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Sky", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::ColorEdit3("Sky colour", &settings_.skyColour.x);
		ImGui::DragFloat("Sky intensity", &settings_.skyIntensity, 0.01f, 0.0f, 100.0f, "%.2f");
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

		ImGui::SliderFloat("Exposure", &settings_.exposureEV, -10.0f, 10.0f, "%.2f EV");

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
