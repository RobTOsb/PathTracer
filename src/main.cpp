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
#include <numeric>
#include <random>
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
constexpr const char *kDefaultScenePath = "h:\\Intelsponza\\glTF\\Intelsponza.gltf";
//constexpr const char *kDefaultScenePath = "h:\\gltf-research-scenes\\scenes\\breakfast-room\\gltf\\breakfast_room_extended.gltf";
//constexpr const char *kDefaultScenePath = "h:\\glTF-Sample-Assets\\Models\\GlassHurricaneCandleHolder\\glTF-Binary\\GlassHurricaneCandleHolder.glb";
//constexpr const char *kDefaultScenePath = "H:\\bistro.glb";
//constexpr const char *kDefaultScenePath = "h:\\suntemple.glb";
//constexpr const char *kDefaultScenePath = "H:\\PathTracer\\assets\\SubsurfaceScene\\scene.gltf";
//constexpr const char *kDefaultScenePath = "h:\\AMD-GI1\\assets\\SphereRoom\\SphereRoom.glb";
//constexpr const char *kDefaultScenePath = "h:\\SanMiguel.glb";

// Intel Sponza's emissive instances, numbered as the load logs them: from 0 the lamps' bulbs, then a
// 1 m quad in master_material with the bulb material (left alone), the textured plane and the red sphere.
constexpr uint32_t kBulbEmitterCount = 22;
constexpr uint32_t kQuadEmitter = 23;
constexpr uint32_t kSphereEmitter = 24;
// The quad spins about the vertical through its centre, once per period.
constexpr float kQuadSpinPeriod = 6.0f; // seconds
// The sphere's emission swings between its glTF strength and this share of it, once per period.
constexpr float kSpherePulsePeriod = 8.0f; // seconds
constexpr float kSpherePulseMin = 0.0f;
// Each bulb stays lit, then dark, for a random time in these ranges (seconds).
constexpr float kBulbLitMin = 0.3f;
constexpr float kBulbLitMax = 4.0f;
constexpr float kBulbDarkMin = 0.05f;
constexpr float kBulbDarkMax = 0.8f;

constexpr uint kFramesInFlight = 2;
constexpr uint kSwapchainImageCount = 3;
constexpr float kMouseSensitivity = 0.15f; // degrees per pixel

// Emissive instances whose lights get a light tree of their own, rebuilt whenever they move or change
// brightness: the quad and the bulbs, so that dark bulbs get no light samples. The sphere (329,616
// triangles) would take too long to rebuild every frame, so it stays static and only its emission scales.
std::vector<uint32_t> DynamicEmitters()
{
	std::vector<uint32_t> emitters(kBulbEmitterCount);
	std::iota(emitters.begin(), emitters.end(), 0u);
	emitters.push_back(kQuadEmitter);
	return emitters;
}

struct LightAnimation
{
	float seconds = 0.0f; // of animation, which stops while it is switched off
	std::mt19937 random{ 1 };
	// Per bulb: lit or not, and the time left until it switches.
	std::vector<bool> bulbLit;
	std::vector<float> bulbSecondsLeft;
};

void AnimateLights(RenderGPUScene &gpuScene, LightAnimation &animation, float deltaSeconds)
{
	animation.seconds += deltaSeconds;
	const uint32_t emitterCount = gpuScene.GetEmissiveInstanceCount();

	const uint32_t quad = gpuScene.GetDynamicInstanceOfEmitter(kQuadEmitter);
	if (quad != UINT32_MAX)
	{
		const float angle = 2.0f * Pi * animation.seconds / kQuadSpinPeriod;
		const float c = cosf(angle);
		const float s = sinf(angle);
		// Row vectors (p' = p A + t, see InstanceGPU): the rest transform, then a turn about +Y, which
		// leaves the centre (the object space origin) at t.
		const float rotation[3][3] = { { c, 0.0f, -s }, { 0.0f, 1.0f, 0.0f }, { s, 0.0f, c } };
		const float *rest = gpuScene.GetDynamicInstanceRestTransform(quad);
		float m[12];
		for (int r = 0; r < 3; ++r)
		{
			for (int col = 0; col < 3; ++col)
			{
				m[4 * r + col] = rest[4 * r] * rotation[0][col] + rest[4 * r + 1] * rotation[1][col] +
								 rest[4 * r + 2] * rotation[2][col];
			}
			m[4 * r + 3] = rest[4 * r + 3];
		}
		gpuScene.SetDynamicInstanceTransform(quad, m);
	}

	if (kSphereEmitter < emitterCount)
	{
		const float pulse = 0.5f + 0.5f * cosf(2.0f * Pi * animation.seconds / kSpherePulsePeriod);
		gpuScene.SetEmissionScale(kSphereEmitter, kSpherePulseMin + (1.0f - kSpherePulseMin) * pulse);
	}

	auto randomSeconds = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(animation.random); };
	const uint32_t bulbs = std::min(kBulbEmitterCount, emitterCount);
	if (animation.bulbLit.size() != bulbs)
	{
		// All lit, each for its own time, so that they do not switch together.
		animation.bulbLit.assign(bulbs, true);
		animation.bulbSecondsLeft.resize(bulbs);
		for (float &secondsLeft: animation.bulbSecondsLeft)
		{
			secondsLeft = randomSeconds(0.0f, kBulbLitMax);
		}
	}
	for (uint32_t i = 0; i < bulbs; ++i)
	{
		float &secondsLeft = animation.bulbSecondsLeft[i];
		secondsLeft -= deltaSeconds;
		while (secondsLeft <= 0.0f)
		{
			const bool lit = !animation.bulbLit[i];
			animation.bulbLit[i] = lit;
			secondsLeft += lit ? randomSeconds(kBulbLitMin, kBulbLitMax) : randomSeconds(kBulbDarkMin, kBulbDarkMax);
		}
		gpuScene.SetEmissionScale(i, animation.bulbLit[i] ? 1.0f : 0.0f);
	}
}

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

int Run(const char *scenePath)
{
	Window window(WindowDesc{ "PathTracer", 1920, 1080, true });

	std::unique_ptr<RenderContext> renderContext(
			CreateRenderContext(&window, RenderAPI::D3D12, kFramesInFlight, kSwapchainImageCount, false, true));
	renderContext->CreatePerFrameResources();
	renderContext->InitialiseImgui(window.GetNativeWindow());

	RenderGPUScene gpuScene(renderContext.get());
	gpuScene.SetDynamicEmitters(DynamicEmitters());
	if (!LoadScene(scenePath, gpuScene))
	{
		return 1;
	}

	// The acceleration structures are built once in their own submission before the first frame.
	// Moving the dynamic emitters then only rebuilds the TLAS, in the frame they move.
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

	auto lastTime = std::chrono::steady_clock::now();
	LightAnimation lightAnimation;
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
		UpdateCamera(window, camera, settings, deltaSeconds);
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

		if (settings.lights.animate && gpuScene.GetEmissiveInstanceCount() > 0)
		{
			AnimateLights(gpuScene, lightAnimation, deltaSeconds);
			// What the reference sums changes, so it starts over.
			++settings.reference.restartCount;
		}

		if (settings.atmosphere.enabled && settings.clouds.enabled && settings.renderMode != RenderMode::Reference)
		{
			const float direction = DegreesToRadians(settings.clouds.windDirectionDegrees);
			const float distanceKm = settings.clouds.windSpeed * 0.001f * deltaSeconds;
			settings.clouds.windOffsetKm.x += sinf(direction) * distanceKm;
			settings.clouds.windOffsetKm.y += cosf(direction) * distanceKm;
		}

		renderContext->BeginFrame();

		// After BeginFrame, which frees this frame's light buffers. Rebuilds the static emitters'
		// tree only when its settings change, the dynamic emitters' also when they have moved.
		LightTreeBuildSettings lightTreeSettings;
		lightTreeSettings.maxLeafLights = static_cast<uint32_t>(settings.lights.maxLeafLights);
		lightTreeSettings.orientationSplits = settings.lights.orientationSplits;
		gpuScene.UpdateLights(lightTreeSettings);

		renderContext->BeginGraphicsCommandList();
		renderContext->SetDescriptorHeaps();
		gpuScene.UpdateAccelerationStructures();

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
