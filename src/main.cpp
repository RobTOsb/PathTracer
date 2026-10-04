#include "Camera.h"
#include "GLTFScene.h"
#include "RenderContext.h"
#include "Window.h"

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
#include <stdio.h>
#include <vector>

// constexpr const char *kDefaultScenePath = "h:\\AMD-GI1\\assets\\intel_sponza\\intelsponza.glb";
constexpr const char *kDefaultScenePath = "H:\\PathTracer\\assets\\TestScene\\scene.gltf";

constexpr uint kFramesInFlight = 2;
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

	const ImGuiIO &io = ImGui::GetIO();
	if (io.WantCaptureKeyboard)
	{
		return;
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

int Run(const char *scenePath)
{
	Window window(WindowDesc{ "PathTracer", 1920, 1080, true });

	std::unique_ptr<RenderContext> renderContext(CreateRenderContext(&window, RenderAPI::D3D12, kFramesInFlight));
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

	RenderSettings settings;
	Camera camera(90.0f, window.GetAspectRatio(), 0.01f, 1000.0f, false, true);
	FrameScene(gpuScene, camera, settings);

	std::vector<std::unique_ptr<RenderPass>> passes;
	auto pathTracer = std::make_unique<PathTracerPass>(renderContext.get(), gpuScene, camera, settings);
	const ResourceHandle radiance = pathTracer->GetOutput();
	passes.push_back(std::move(pathTracer));
	passes.push_back(std::make_unique<TonemapPass>(renderContext.get(), radiance, settings));
	passes.push_back(std::make_unique<ImGuiPass>(settings, camera));

	auto lastTime = std::chrono::steady_clock::now();
	while (window.ProcessEvents())
	{
		const auto now = std::chrono::steady_clock::now();
		const float deltaSeconds = std::chrono::duration<float>(now - lastTime).count();
		lastTime = now;

		UpdateCamera(window, camera, settings, deltaSeconds);

		renderContext->BeginFrame();
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
