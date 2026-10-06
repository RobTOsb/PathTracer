#include "Camera.h"
#include "D3D12RenderContext.h"
#include "DLSSCommon.h"
#include "GLTFScene.h"
#include "RenderContext.h"
#include "Window.h"

#include "Renderer/DLSSPass.h"
#include "Renderer/ImGuiPass.h"
#include "Renderer/PathTracerPass.h"
#include "Renderer/RenderPass.h"
#include "Renderer/RenderSettings.h"
#include "Renderer/TonemapPass.h"
#include "Scene/RenderGPUScene.h"

#include "fmtlog.h"
#include "imgui.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <math.h>
#include <memory>
#include <random>
#include <span>
#include <stdio.h>
#include <vector>

#include <Windows.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#pragma comment(lib, "dxgi.lib")

#undef min
#undef max

// Peak luminance in nits that the monitor showing the window reports, or 0 if it can't be found.
static float QueryDisplayPeakNits(void *hwnd)
{
	using Microsoft::WRL::ComPtr;

	const HMONITOR monitor = MonitorFromWindow(static_cast<HWND>(hwnd), MONITOR_DEFAULTTONEAREST);
	ComPtr<IDXGIFactory1> factory;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
	{
		return 0.0f;
	}

	ComPtr<IDXGIAdapter1> adapter;
	for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a)
	{
		ComPtr<IDXGIOutput> output;
		for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o)
		{
			ComPtr<IDXGIOutput6> output6;
			DXGI_OUTPUT_DESC1 desc = {};
			if (SUCCEEDED(output.As(&output6)) && SUCCEEDED(output6->GetDesc1(&desc)) && desc.Monitor == monitor)
			{
				return desc.MaxLuminance;
			}
		}
	}
	return 0.0f;
}

constexpr const char *kDefaultScenePath = "h:\\Intelsponza\\glTF\\Intelsponza.gltf";
//constexpr const char *kDefaultScenePath = "h:\\glTF-Sample-Assets\\Models\\Sponza\\glTF\\Sponza.gltf";
//constexpr const char *kDefaultScenePath = "H:\\PathTracer\\assets\\TestScene\\scene.gltf";

constexpr uint kFramesInFlight = 2;
constexpr float kMouseSensitivity = 0.15f; // degrees per pixel
constexpr uint32_t kTestLightCount = 1024;
constexpr float kTestLightMinRadius = 0.005f; // fraction of the scene radius, the largest is 10x this
constexpr float kTestLightMinPeriod = 10.0f; // seconds per swing along one axis
constexpr float kTestLightMaxPeriod = 30.0f;

void UpdateCamera(Window &window, Camera &camera, RenderSettings &settings, float deltaSeconds)
{
	// Always drain the deltas so a drag starts from where the cursor is, not where it was.
	const float mouseDeltaX = window.GetMouseDeltaX();
	const float mouseDeltaY = window.GetMouseDeltaY();
	if (window.IsRightMouseButtonDown())
	{
		camera.RotateCamera(-mouseDeltaX, -mouseDeltaY, kMouseSensitivity);
	}

	// Mouse wheel scales the movement speed.
	const float wheel = window.GetMouseWheelDelta();
	if (wheel != 0.0f)
	{
		settings.cameraSpeed = std::clamp(settings.cameraSpeed * powf(1.2f, wheel), 0.01f, 1000.0f);
	}

	float distance = settings.cameraSpeed * deltaSeconds;
	if (window.WasKeyPressed(0x10 /* VK_SHIFT */))
	{
		distance *= 4.0f;
	}

	if (window.WasKeyPressed('W'))
		camera.MoveForward(distance);
	if (window.WasKeyPressed('S'))
		camera.MoveBackward(distance);
	if (window.WasKeyPressed('A'))
		camera.StrafeLeft(distance);
	if (window.WasKeyPressed('D'))
		camera.StrafeRight(distance);
	if (window.WasKeyPressed('E'))
		camera.SetPosition(camera.GetPosition() + float3(0.0f, distance, 0.0f));
	if (window.WasKeyPressed('Q'))
		camera.SetPosition(camera.GetPosition() - float3(0.0f, distance, 0.0f));
}

// Places the camera in front of the scene bounds looking down -Z, and scales the move speed to it.
void FrameScene(const RenderGPUScene &gpuScene, Camera &camera, RenderSettings &settings)
{
	const float3 lo = gpuScene.GetBoundsMin();
	const float3 hi = gpuScene.GetBoundsMax();
	const float3 centre = (lo + hi) * 0.5f;
	const float3 extent = hi - lo;
	const float radius = std::max(0.5f * sqrtf(Dot(extent, extent)), 0.01f);

	const float distance = radius / tanf(0.5f * camera.GetFovY());
	camera.SetPosition(float3(centre.x, centre.y, centre.z + distance));
	settings.cameraSpeed = std::max(radius * 0.25f, 0.01f);
}

// A test light moving on a Lissajous path: each axis swings sinusoidally around the centre with its
// own amplitude, angular frequency and phase.
struct TestLight
{
	float3 intensity;
	float radius;
	float3 centre;
	float3 amplitude;
	float3 frequency; // radians per second
	float3 phase;

	float3 GetPosition(float time) const
	{
		return { centre.x + amplitude.x * sinf(frequency.x * time + phase.x),
				 centre.y + amplitude.y * sinf(frequency.y * time + phase.y),
				 centre.z + amplitude.z * sinf(frequency.z * time + phase.z) };
	}
};

std::vector<TestLight> CreateTestLights(const RenderGPUScene &gpuScene, uint32_t count)
{
	const float3 lo = gpuScene.GetBoundsMin();
	const float3 hi = gpuScene.GetBoundsMax();
	const float3 extent = hi - lo;
	const float radius = std::max(0.5f * sqrtf(Dot(extent, extent)), 0.01f);
	// Mean of a log-uniform distribution over [0.1, 10] is 9.9 / ln(100).
	const float meanIntensity = 0.25f * radius * radius / static_cast<float>(count);
	const float baseIntensity = meanIntensity * logf(100.0f) / 9.9f;

	std::mt19937 rng(1234u);
	std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	auto axisMotion = [&](float &centre, float &amplitude, float &frequency, float &phase)
	{
		amplitude = 0.1f + 0.15f * uniform(rng);
		centre = 0.05f + amplitude + (0.9f - 2.0f * amplitude) * uniform(rng);
		frequency = 2.0f * Pi / (kTestLightMinPeriod + (kTestLightMaxPeriod - kTestLightMinPeriod) * uniform(rng));
		phase = 2.0f * Pi * uniform(rng);
	};

	std::vector<TestLight> lights;
	lights.reserve(count);
	for (uint32_t i = 0; i < count; ++i)
	{
		TestLight light;
		float3 t, a;
		axisMotion(t.x, a.x, light.frequency.x, light.phase.x);
		axisMotion(t.y, a.y, light.frequency.y, light.phase.y);
		axisMotion(t.z, a.z, light.frequency.z, light.phase.z);
		light.centre = lo + extent * t;
		light.amplitude = extent * a;

		// Fully bright HSV colour with a random hue and saturation.
		const float hue = 6.0f * uniform(rng);
		const float saturation = 0.3f + 0.7f * uniform(rng);
		const float3 hueColour = { std::clamp(fabsf(hue - 3.0f) - 1.0f, 0.0f, 1.0f),
								   std::clamp(2.0f - fabsf(hue - 2.0f), 0.0f, 1.0f),
								   std::clamp(2.0f - fabsf(hue - 4.0f), 0.0f, 1.0f) };
		const float3 colour = float3(1.0f, 1.0f, 1.0f) * (1.0f - saturation) + hueColour * saturation;

		light.intensity = colour * (baseIntensity * powf(10.0f, 2.0f * uniform(rng) - 1.0f));
		light.radius = radius * kTestLightMinRadius * powf(10.0f, uniform(rng));
		lights.push_back(light);
	}
	return lights;
}

void PlaceTestLights(std::span<const TestLight> testLights, float time, std::vector<GpuLight> &lights)
{
	lights.clear();
	for (const TestLight &light: testLights)
	{
		lights.push_back(MakePointLight(light.intensity, light.GetPosition(time), 0.0f, light.radius));
	}
}

void BuildPasses(RenderContext *renderContext,
				 NGXContext &ngx,
				 const RenderGPUScene &gpuScene,
				 const Camera &camera,
				 RenderSettings &settings,
				 std::vector<std::unique_ptr<RenderPass>> &passes)
{
	passes.clear();

	DLSSOptimalSettings optimal;
	const bool rayReconstruction = settings.dlssMode != DLSSMode::Off && ngx.IsRayReconstructionSupported() &&
								   ngx.GetOptimalSettings(DLSSFeatureKind::RayReconstruction,
														  settings.dlssMode,
														  renderContext->GetWindowWidth(),
														  renderContext->GetWindowHeight(),
														  optimal);
	if (rayReconstruction)
	{
		renderContext->SetRenderResolution(optimal.optimalRenderWidth, optimal.optimalRenderHeight);
	}
	else
	{
		settings.dlssMode = DLSSMode::Off;
		renderContext->SetRenderResolution(0, 0);
	}
	logi("Render resolution {}x{}, output {}x{}, DLSS-RR {}",
		 renderContext->GetRenderWidth(),
		 renderContext->GetRenderHeight(),
		 renderContext->GetWindowWidth(),
		 renderContext->GetWindowHeight(),
		 rayReconstruction ? "on" : "off");

	auto pathTracer = std::make_unique<PathTracerPass>(renderContext, gpuScene, camera, settings, rayReconstruction);
	ResourceHandle radiance = pathTracer->GetOutput();
	if (rayReconstruction)
	{
		auto dlss = std::make_unique<DLSSPass>(renderContext,
											   ngx,
											   *pathTracer,
											   camera,
											   settings.dlssMode,
											   settings.rrPreset);
		radiance = dlss->GetOutput();
		passes.push_back(std::move(pathTracer));
		passes.push_back(std::move(dlss));
	}
	else
	{
		passes.push_back(std::move(pathTracer));
	}
	passes.push_back(std::make_unique<TonemapPass>(renderContext, radiance, settings));
	passes.push_back(std::make_unique<ImGuiPass>(settings, camera, ngx.IsRayReconstructionSupported(), gpuScene));
}

int Run(const char *scenePath)
{
	Window window(WindowDesc{ "PathTracer", 1920, 1080, true });

	std::unique_ptr<RenderContext> renderContext(
			CreateRenderContext(&window, RenderAPI::D3D12, kFramesInFlight, false, true));
	renderContext->CreatePerFrameResources();
	renderContext->InitialiseImgui(window.GetNativeWindow());

	Scene::GLTFScene scene;
	logi("Loading '{}'", scenePath);
	fmtlog::poll();
	if (Scene::GLTFScene::MakeGLTFScene(scenePath, scene) != Scene::SCENE_OK)
	{
		loge("Failed to load scene '{}'", scenePath);
		return 1;
	}
	scene.Print();

	RenderGPUScene gpuScene(renderContext.get());
	gpuScene.BuildGPUScene(scene);

	// The scene is static for now, so the acceleration structures are built once in their own
	// submission before the first frame.
	renderContext->BeginFrame();
	renderContext->BeginGraphicsCommandList();
	renderContext->SetDescriptorHeaps();
	gpuScene.BuildAccelerationStructures();
	renderContext->EndGraphicsCommandList();
	renderContext->EndFrame();
	renderContext->WaitForGPUIdle();

	if (!gpuScene.HasGeometry())
	{
		loge("Scene '{}' has no renderable triangle geometry", scenePath);
		return 1;
	}

	const std::vector<TestLight> testLights = CreateTestLights(gpuScene, kTestLightCount);
	std::vector<GpuLight> lights;
	float lightTime = 0.0f;

	RenderSettings settings;
	if (renderContext->GetUseHDR())
	{
		const float peakNits = QueryDisplayPeakNits(window.GetNativeWindow());
		if (peakNits > 0.0f)
		{
			settings.peakNits = peakNits;
		}
		logi("HDR10 output, display peak {} nits", settings.peakNits);
	}
	Camera camera(90.0f, window.GetAspectRatio(), 0.01f, 1000.0f, false, true);
	FrameScene(gpuScene, camera, settings);

	// Declared before the passes so it outlives the DLSS feature they use. Without it (no NVIDIA
	// GPU, old driver, missing nvngx_dlssd.dll) DLSS-RR is simply unavailable.
	NGXContext ngx;
	ngx.Initialise(static_cast<D3D12RenderContext *>(renderContext.get())->GetDevice());

	std::vector<std::unique_ptr<RenderPass>> passes;
	BuildPasses(renderContext.get(), ngx, gpuScene, camera, settings, passes);
	DLSSMode builtMode = settings.dlssMode;
	DLSSRayReconstructionPreset builtPreset = settings.rrPreset;

	auto lastTime = std::chrono::steady_clock::now();
	while (window.ProcessEvents())
	{
		const auto now = std::chrono::steady_clock::now();
		const float deltaSeconds = std::chrono::duration<float>(now - lastTime).count();
		lastTime = now;

		// The UI changes these while the previous frame is recorded, so they apply from here.
		if (settings.dlssMode != builtMode || settings.rrPreset != builtPreset)
		{
			renderContext->WaitForGPUIdle();
			ngx.ReleaseFeature();
			BuildPasses(renderContext.get(), ngx, gpuScene, camera, settings, passes);
			builtMode = settings.dlssMode;
			builtPreset = settings.rrPreset;
		}

		// Last frame's matrices, for the motion vectors.
		camera.UpdatePreviousMatrices();
		UpdateCamera(window, camera, settings, deltaSeconds);

		if (settings.lights.animate)
		{
			lightTime += deltaSeconds * settings.lights.animationSpeed;
		}

		renderContext->BeginFrame();

		// After BeginFrame, which frees this frame's light buffers.
		PlaceTestLights(testLights, lightTime, lights);
		gpuScene.UpdateLights(lights);

		renderContext->BeginGraphicsCommandList();
		renderContext->SetDescriptorHeaps();

		for (const std::unique_ptr<RenderPass> &pass: passes)
		{
			pass->Execute(renderContext.get());
		}

		renderContext->TransitionResource(renderContext->GetCurrentBackBufferHandle(),
										  ResourceLayout::Present,
										  ResourceSync::None,
										  ResourceAccess::NoAccess);

		renderContext->EndGraphicsCommandList();
		renderContext->EndFrame();
		renderContext->Present();

		fmtlog::poll();
	}

	renderContext->WaitForGPUIdle();
	return 0;
}

int main(int argc, char **argv)
{
	// A command-line path overrides the default.
	const char *scenePath = argc >= 2 ? argv[1] : kDefaultScenePath;
	if (scenePath[0] == '\0')
	{
		fprintf(stderr, "Usage: %s <scene.gltf|scene.glb>, or set kDefaultScenePath in main.cpp\n", argv[0]);
		return 1;
	}

	int result = 1;
	try
	{
		result = Run(scenePath);
	}
	catch (const std::exception &e)
	{
		loge("Fatal: {}", e.what());
	}

	fmtlog::poll(true);
	return result;
}
