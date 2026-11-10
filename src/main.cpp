#include "Camera.h"
#include "D3D12RenderContext.h"
#include "DLSSCommon.h"
#include "GLTFScene.h"
#include "RenderContext.h"
#include "Window.h"

#include "Renderer/AutoExposurePass.h"
#include "Renderer/BloomPass.h"
#include "Renderer/DLSSPass.h"
#include "Renderer/GuideViewPass.h"
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
#include <fstream>
#include <iomanip>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>
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
//constexpr const char *kDefaultScenePath = "h:\\Intelsponza\\glTF\\Intelsponza.gltf";
//constexpr const char *kDefaultScenePath = "h:\\gltf-research-scenes\\scenes\\cornell-caustic\\gltf\\cornell_caustic_extended.gltf";
//constexpr const char *kDefaultScenePath = "h:\\glTF-Sample-Assets\\Models\\TransmissionThinwallTestGrid\\glTF-Binary\\TransmissionThinwallTestGrid.glb";
//constexpr const char *kDefaultScenePath = "H:\\bistro_roughmetal.glb";
//constexpr const char *kDefaultScenePath = "h:\\suntemple.glb";
//constexpr const char *kDefaultScenePath = "h:\\SanMiguel.glb";
//constexpr const char *kDefaultScenePath = "H:\\PathTracer\\assets\\SubsurfaceScene\\scene.gltf";
//constexpr const char *kDefaultScenePath = "h:\\AMD-GI1\\assets\\SphereRoom\\SphereRoom.glb";
//constexpr const char *kDefaultScenePath = "h:\\ZeroDay_pbr.glb";
constexpr const char *kDefaultScenePath = "h:\\landscape\\view-0.glb";
//constexpr const char *kDefaultScenePath = "h:\\modern_apartment.glb";
//constexpr const char *kDefaultScenePath = "h:\\LucyCheckerboard.glb";

constexpr uint kFramesInFlight = 1;
constexpr uint kSwapchainImageCount = 2;
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

// The camera bookmark: position, orientation and move speed, as text in a .camera file beside the scene.
std::filesystem::path CameraBookmarkPath(const std::filesystem::path &scenePath)
{
	std::filesystem::path path = scenePath;
	return path.replace_extension(".camera");
}

bool SaveCameraBookmark(const std::filesystem::path &path, const Camera &camera, const RenderSettings &settings)
{
	std::ofstream file(path);
	const float3 position = camera.GetPosition();
	file << std::setprecision(9) << "position " << position.x << ' ' << position.y << ' ' << position.z << '\n'
		 << "yaw " << camera.GetYaw() << '\n'
		 << "pitch " << camera.GetPitch() << '\n'
		 << "speed " << settings.cameraSpeed << '\n';
	return static_cast<bool>(file);
}

bool LoadCameraBookmark(const std::filesystem::path &path, Camera &camera, RenderSettings &settings)
{
	std::ifstream file(path);
	std::string key;
	float3 position;
	float yaw = 0.0f;
	float pitch = 0.0f;
	float speed = 0.0f;
	if (!(file >> key >> position.x >> position.y >> position.z >> key >> yaw >> key >> pitch >> key >> speed))
	{
		return false;
	}
	camera.SetPosition(position);
	camera.SetOrientation(yaw, pitch);
	settings.cameraSpeed = speed;
	return true;
}

void BuildPasses(RenderContext *renderContext,
				 NGXContext &ngx,
				 const RenderGPUScene &gpuScene,
				 const Camera &camera,
				 RenderSettings &settings,
				 std::vector<std::unique_ptr<RenderPass>> &passes)
{
	passes.clear();

	// The reference renders at the window's resolution without DLSS, leaving the DLSS mode for when it ends.
	const bool reference = settings.renderMode == RenderMode::Reference;
	DLSSOptimalSettings optimal;
	const bool rayReconstruction = !reference && settings.dlssMode != DLSSMode::Off &&
								   ngx.IsRayReconstructionSupported() &&
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
		if (!reference)
		{
			settings.dlssMode = DLSSMode::Off;
		}
		renderContext->SetRenderResolution(0, 0);
	}
	logi("Render resolution {}x{}, output {}x{}, {}",
		 renderContext->GetRenderWidth(),
		 renderContext->GetRenderHeight(),
		 renderContext->GetWindowWidth(),
		 renderContext->GetWindowHeight(),
		 reference ? "reference" : rayReconstruction ? "DLSS-RR on" : "DLSS-RR off");

	auto atmosphere = std::make_unique<SkyAtmospherePass>(renderContext, gpuScene, camera, settings);
	auto pathTracer = std::make_unique<PathTracerPass>(
			renderContext, gpuScene, camera, settings, *atmosphere, rayReconstruction, reference);
	const PathTracerPass &pathTracerPass = *pathTracer;
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
	auto autoExposure = std::make_unique<AutoExposurePass>(renderContext, radiance, settings);
	const ResourceHandle exposure = autoExposure->GetExposure();
	passes.push_back(std::move(autoExposure));
	auto bloom = std::make_unique<BloomPass>(renderContext, radiance, exposure, camera, settings);
	radiance = bloom->GetOutput();
	passes.push_back(std::move(bloom));
	passes.push_back(std::make_unique<TonemapPass>(renderContext, radiance, exposure, settings));
	if (rayReconstruction)
	{
		passes.push_back(std::make_unique<GuideViewPass>(renderContext, pathTracerPass, settings));
	}
	passes.push_back(
			std::make_unique<ImGuiPass>(settings, camera, ngx.IsRayReconstructionSupported(), gpuScene, pathTracerPass));
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

// Uniform in [0, 1) from an integer (PCG).
float Hash01(uint32_t x)
{
	x = x * 747796405u + 2891336453u;
	x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
	return static_cast<float>((x >> 22u) ^ x) * (1.0f / 4294967296.0f);
}

float SmoothStep(float edge0, float edge1, float x)
{
	const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	return t * t * (3.0f - 2.0f * t);
}

// On for a little over half of every 1.2 s, with 60 ms fades.
float BeatFlash(float seconds)
{
	const float phase = seconds / 1.2f - floorf(seconds / 1.2f);
	return SmoothStep(0.0f, 0.06f, phase) * (1.0f - SmoothStep(0.55f, 0.61f, phase));
}

// Two 120 ms blinks every 1.7 s, like a beacon, and dark in between.
float BeaconFlash(float seconds)
{
	const float t = (seconds / 1.7f - floorf(seconds / 1.7f)) * 1.7f;
	return (t < 0.12f || (t >= 0.3f && t < 0.42f)) ? 1.0f : 0.0f;
}

// A failing light: each 1.5 s it is either steady with a faint 25 Hz buzz, stuttering on and off at random,
// or dead.
float BrokenLight(float seconds)
{
	const uint32_t episode = static_cast<uint32_t>(seconds / 1.5f);
	const uint32_t tick = static_cast<uint32_t>(seconds * 25.0f);
	const float state = Hash01(episode * 0x9E3779B9u + 17u);
	if (state < 0.15f)
	{
		return 0.0f;
	}
	if (state < 0.5f)
	{
		return Hash01(tick ^ 0x85EBCA6Bu) < 0.45f ? 0.0f : 0.6f + 0.4f * Hash01(tick * 7u + 1u);
	}
	return 0.9f + 0.1f * Hash01(tick);
}

// Intel Sponza's emissive bunnies, by their glTF materials (Material.001 to .003): pink, blue and the violet one
// that is broken.
struct FlashingBunny
{
	uint32_t emissiveInstance;
	float (*pattern)(float seconds);
};

std::vector<FlashingBunny> FindBunnies(const std::filesystem::path &scenePath, const RenderGPUScene &gpuScene)
{
	std::vector<FlashingBunny> bunnies;
	if (scenePath.stem() != "Intelsponza")
	{
		return bunnies;
	}
	constexpr struct
	{
		uint32_t material;
		float (*pattern)(float);
	} kBunnies[] = { { 41, BeatFlash }, { 42, BeaconFlash }, { 43, BrokenLight } };
	for (const auto &bunny: kBunnies)
	{
		for (uint32_t emissiveInstance: gpuScene.FindEmissiveInstances(bunny.material))
		{
			bunnies.push_back({ emissiveInstance, bunny.pattern });
		}
	}
	return bunnies;
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

	// The acceleration structures are built once in their own submission before the first frame.
	// Animating the scene then only rebuilds the TLASes of the frames that trace it in a new pose.
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
	const std::filesystem::path bookmarkPath = CameraBookmarkPath(scenePath);
	settings.cameraBookmarkExists = std::filesystem::exists(bookmarkPath);
	if (settings.cameraBookmarkExists && LoadCameraBookmark(bookmarkPath, camera, settings))
	{
		logi("Camera restored from '{}'", bookmarkPath.string());
	}

	// Declared before the passes so it outlives the DLSS feature they use. Without it (no NVIDIA
	// GPU, old driver, missing nvngx_dlssd.dll) DLSS-RR is simply unavailable.
	NGXContext ngx;
	ngx.Initialise(static_cast<D3D12RenderContext *>(renderContext.get())->GetDevice());

	std::vector<std::unique_ptr<RenderPass>> passes;
	BuildPasses(renderContext.get(), ngx, gpuScene, camera, settings, passes);
	RenderMode builtRenderMode = settings.renderMode;
	DLSSMode builtMode = settings.dlssMode;
	DLSSRayReconstructionPreset builtPreset = settings.rrPreset;

	const std::vector<FlashingBunny> bunnies = FindBunnies(scenePath, gpuScene);
	settings.lights.bunnyCount = static_cast<uint32_t>(bunnies.size());
	std::vector<float> bunnyScales(bunnies.size(), 1.0f);
	float flashSeconds = 0.0f;

	auto lastTime = std::chrono::steady_clock::now();
	float animatedSeconds = -1.0f; // the time the scene was last posed at, none yet
	const float freeFovY = camera.GetFovY();
	bool followedSceneCamera = false; // last frame's view was the animated scene camera's
	float followedSeconds = 0.0f; // the animation time it was posed at
	uint32_t shownView = 0; // last frame's RenderSettings::view
	// The user's free camera while a still scene camera is the view, to go back to.
	float3 userPosition = camera.GetPosition();
	float userYaw = camera.GetYaw();
	float userPitch = camera.GetPitch();
	while (window.ProcessEvents())
	{
		const auto now = std::chrono::steady_clock::now();
		const float deltaSeconds = std::chrono::duration<float>(now - lastTime).count();
		lastTime = now;

		// The UI changes these while the previous frame is recorded, so they apply from here.
		if (settings.renderMode != builtRenderMode || settings.dlssMode != builtMode || settings.rrPreset != builtPreset)
		{
			renderContext->WaitForGPUIdle();
			ngx.ReleaseFeature();
			BuildPasses(renderContext.get(), ngx, gpuScene, camera, settings, passes);
			builtRenderMode = settings.renderMode;
			builtMode = settings.dlssMode;
			builtPreset = settings.rrPreset;
		}

		// Last frame's matrices, for the motion vectors.
		camera.UpdatePreviousMatrices();
		// A still scene camera keeps the user's view to come back to. Restoring the bookmark goes back to it.
		if (settings.cameraBookmarkRequest == RenderSettings::CameraBookmarkRequest::Restore ||
			settings.view > gpuScene.GetStillCameraCount())
		{
			settings.view = 0;
		}
		if (settings.view != shownView && shownView == 0)
		{
			userPosition = camera.GetPosition();
			userYaw = camera.GetYaw();
			userPitch = camera.GetPitch();
		}
		else if (settings.view != shownView && settings.view == 0)
		{
			camera.SetPosition(userPosition);
			camera.SetOrientation(userYaw, userPitch);
		}
		const bool followSceneCamera = settings.view == 0 && gpuScene.IsCameraAnimated() && settings.animation.play &&
									   settings.animation.sceneCamera && settings.renderMode != RenderMode::Reference;
		if (settings.view == 0 && !followSceneCamera)
		{
			UpdateCamera(window, camera, settings, deltaSeconds);
		}
		if (settings.cameraBookmarkRequest == RenderSettings::CameraBookmarkRequest::Save)
		{
			settings.cameraBookmarkExists = SaveCameraBookmark(bookmarkPath, camera, settings);
			if (settings.cameraBookmarkExists)
			{
				logi("Camera saved to '{}'", bookmarkPath.string());
			}
			else
			{
				loge("Could not write '{}'", bookmarkPath.string());
			}
		}
		else if (settings.cameraBookmarkRequest == RenderSettings::CameraBookmarkRequest::Restore &&
				 !LoadCameraBookmark(bookmarkPath, camera, settings))
		{
			loge("Could not read '{}'", bookmarkPath.string());
		}
		settings.cameraBookmarkRequest = RenderSettings::CameraBookmarkRequest::None;

		if (gpuScene.HasAnimation())
		{
			AnimationSettings &animation = settings.animation;
			const float duration = gpuScene.GetAnimationDuration();
			// The reference holds the scene still, at the time the UI sets.
			if (animation.play && settings.renderMode != RenderMode::Reference)
			{
				animation.seconds += deltaSeconds * animation.speed;
			}
			animation.seconds = duration > 0.0f ? animation.seconds - floorf(animation.seconds / duration) * duration : 0.0f;
			if (animation.seconds != animatedSeconds)
			{
				gpuScene.Animate(animation.seconds);
				animatedSeconds = animation.seconds;
				// What the reference sums changes, so it starts over.
				++settings.reference.restartCount;
			}
		}

		// The scene cameras, the animated one posed with the animation above. Choosing another view, the animated
		// camera starting, stopping (back to the free camera's field of view) and looping are cuts.
		settings.cameraCut = settings.view != shownView || followSceneCamera != followedSceneCamera;
		if (settings.view != 0)
		{
			const RenderGPUScene::CameraPose pose = gpuScene.GetStillCameraPose(settings.view - 1);
			camera.SetView(pose.position, pose.front, pose.up);
			camera.SetFovY(pose.yFov, window.GetAspectRatio());
		}
		else if (followSceneCamera)
		{
			const RenderGPUScene::CameraPose pose = gpuScene.GetAnimatedCameraPose();
			camera.SetView(pose.position, pose.front, pose.up);
			camera.SetFovY(pose.yFov, window.GetAspectRatio());
			settings.cameraCut |= settings.animation.seconds < followedSeconds;
			followedSeconds = settings.animation.seconds;
		}
		else if (followedSceneCamera || shownView != 0)
		{
			camera.SetFovY(freeFovY, window.GetAspectRatio());
		}
		followedSceneCamera = followSceneCamera;
		shownView = settings.view;

		// Like the animations, the flashing holds still in the reference, which starts over if a bunny changes.
		if (settings.lights.flashBunnies && settings.renderMode != RenderMode::Reference)
		{
			flashSeconds += deltaSeconds;
		}
		for (size_t i = 0; i < bunnies.size(); ++i)
		{
			const float scale = settings.lights.flashBunnies ? bunnies[i].pattern(flashSeconds) : 1.0f;
			if (scale != bunnyScales[i])
			{
				gpuScene.SetEmissionScale(bunnies[i].emissiveInstance, scale);
				bunnyScales[i] = scale;
				++settings.reference.restartCount;
			}
		}

		if (settings.atmosphere.enabled && settings.clouds.enabled && settings.renderMode != RenderMode::Reference)
		{
			const float direction = DegreesToRadians(settings.clouds.windDirectionDegrees);
			const float distanceKm = settings.clouds.windSpeed * 0.001f * deltaSeconds;
			settings.clouds.windOffsetKm.x += sinf(direction) * distanceKm;
			settings.clouds.windOffsetKm.y += cosf(direction) * distanceKm;
		}

		renderContext->BeginFrame();

		// After BeginFrame, which frees this frame's light buffers.
		gpuScene.UpdateLights();

		renderContext->BeginGraphicsCommandList();
		renderContext->SetDescriptorHeaps();
		if (settings.mergeInstances != gpuScene.GetMergeInstances())
		{
			// The old acceleration structures may still be traced by frames in flight.
			renderContext->WaitForGPUIdle();
			const auto start = std::chrono::steady_clock::now();
			gpuScene.SetMergeInstances(settings.mergeInstances);
			logi("Rebuilt the acceleration structures {} merged instances in {:.2f} s",
				 settings.mergeInstances ? "with" : "without",
				 std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
		}
		gpuScene.UpdateAccelerationStructures();
		// Every light, static and dynamic, into a new tree on the GPU each frame.
		gpuScene.BuildLightTree(static_cast<uint32_t>(settings.lights.maxLeafLights));

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
