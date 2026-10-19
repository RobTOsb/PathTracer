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
#include "Renderer/SkyAtmospherePass.h"
#include "Renderer/TonemapPass.h"
#include "Scene/RenderGPUScene.h"

#include "fmtlog.h"
#include "imgui.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <math.h>
#include <memory>
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

//constexpr const char *kDefaultScenePath = "h:\\AMD-GI1\\assets\\EmissiveMeshes\\EmissiveMeshes.glb";
constexpr const char *kDefaultScenePath = "h:\\Intelsponza\\glTF\\Intelsponza.gltf";
//constexpr const char *kDefaultScenePath = "h:\\glTF-Sample-Assets\\Models\\Sponza\\glTF\\Sponza.gltf";
// constexpr const char *kDefaultScenePath = "H:\\PathTracer\\assets\\TestScene\\scene.gltf";
//constexpr const char *kDefaultScenePath = "H:\\PathTracer\\assets\\SubsurfaceScene\\scene.gltf";
//constexpr const char *kDefaultScenePath = "h:\\AMD-GI1\\assets\\SphereRoom\\SphereRoom.glb";
//constexpr const char *kDefaultScenePath = "h:\\SubwayInterior\\glTF\\SubwayInterior.gltf";
//constexpr const char *kDefaultScenePath = "h:\\SanMiguel.glb";

constexpr uint kFramesInFlight = 2;
constexpr uint kSwapchainImageCount = 3;
constexpr float kMouseSensitivity = 0.15f; // degrees per pixel

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

	auto atmosphere = std::make_unique<SkyAtmospherePass>(renderContext, camera, settings);
	auto pathTracer =
			std::make_unique<PathTracerPass>(renderContext, gpuScene, camera, settings, *atmosphere, rayReconstruction);
	ResourceHandle radiance = pathTracer->GetOutput();
	passes.push_back(std::move(atmosphere));
	if (rayReconstruction)
	{
		auto dlss = std::make_unique<DLSSPass>(renderContext, ngx, *pathTracer, camera, settings);
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

// A glTF loads from the .scene file beside it when that is at least as new, and otherwise is
// imported and written to it for next time. A .scene path loads only that file.
bool LoadScene(const std::filesystem::path &scenePath, RenderGPUScene &gpuScene)
{
	const auto start = std::chrono::steady_clock::now();
	auto logLoadTime = [&]
	{
		logi("Loaded the scene in {:.2f} s",
			 std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count());
	};

	if (scenePath.extension() == ".scene")
	{
		logi("Loading '{}'", scenePath.string());
		fmtlog::poll();
		if (!gpuScene.LoadGPUScene(scenePath))
		{
			loge("Failed to load scene '{}'", scenePath.string());
			return false;
		}
		logLoadTime();
		return true;
	}

	std::filesystem::path cachePath = scenePath;
	cachePath.replace_extension(".scene");
	std::error_code cacheError;
	std::error_code sceneError;
	const auto cacheTime = std::filesystem::last_write_time(cachePath, cacheError);
	const auto sceneTime = std::filesystem::last_write_time(scenePath, sceneError);
	if (!cacheError && !sceneError && cacheTime >= sceneTime)
	{
		logi("Loading '{}'", cachePath.string());
		fmtlog::poll();
		if (gpuScene.LoadGPUScene(cachePath))
		{
			logLoadTime();
			return true;
		}
		logi("Rebuilding '{}' from the glTF", cachePath.string());
	}

	Scene::GLTFScene scene;
	logi("Loading '{}'", scenePath.string());
	fmtlog::poll();
	if (Scene::GLTFScene::MakeGLTFScene(scenePath.string().c_str(), scene) != Scene::SCENE_OK)
	{
		loge("Failed to load scene '{}'", scenePath.string());
		return false;
	}
	gpuScene.BuildGPUScene(scene, cachePath);
	logLoadTime();
	return true;
}

int Run(const char *scenePath)
{
	Window window(WindowDesc{ "PathTracer", 1920, 1080, true });

	std::unique_ptr<RenderContext> renderContext(
			CreateRenderContext(&window, RenderAPI::D3D12, kFramesInFlight, kSwapchainImageCount, false, true));
	renderContext->CreatePerFrameResources();
	renderContext->InitialiseImgui(window.GetNativeWindow());

	RenderGPUScene gpuScene(renderContext.get());
	if (!LoadScene(scenePath, gpuScene))
	{
		return 1;
	}

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

		renderContext->BeginFrame();

		// After BeginFrame, which frees this frame's light buffers. Rebuilds the tree only when
		// its settings change.
		LightTreeBuildSettings lightTreeSettings;
		lightTreeSettings.maxLeafLights = static_cast<uint32_t>(settings.lights.maxLeafLights);
		lightTreeSettings.orientationSplits = settings.lights.orientationSplits;
		gpuScene.UpdateLights(lightTreeSettings);

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
		fprintf(stderr, "Usage: %s <scene.gltf|scene.glb|scene.scene>, or set kDefaultScenePath in main.cpp\n", argv[0]);
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
