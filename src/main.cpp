#include "Camera.h"
#include "D3D12RenderContext.h"
#include "GLTFScene.h"
#include "RenderContext.h"
#include "Streamline.h"
#include "Window.h"

#include "Animation/AnimationSystem.h"
#include "Renderer/AutoExposurePass.h"
#include "Renderer/BloomPass.h"
#include "Renderer/DLSSPass.h"
#include "Renderer/FrameGenerationPass.h"
#include "Renderer/FrameGraph.h"
#include "Renderer/GuideViewPass.h"
#include "Renderer/ImGuiPass.h"
#include "Renderer/LensFlarePass.h"
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
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>

#include <Windows.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

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
//constexpr const char *kDefaultScenePath = "h:\\gltf-research-scenes\\scenes\\crown\\gltf\\crown_extended.gltf";
//constexpr const char *kDefaultScenePath = "h:\\glTF-Sample-Assets\\Models\\TransmissionThinwallTestGrid\\glTF-Binary\\TransmissionThinwallTestGrid.glb";
//constexpr const char *kDefaultScenePath = "H:\\bistro_roughmetal.glb";
//constexpr const char *kDefaultScenePath = "h:\\suntemple.glb";
//constexpr const char *kDefaultScenePath = "h:\\AbandonedRoom.glb";
//constexpr const char *kDefaultScenePath = "H:\\PathTracer\\assets\\SubsurfaceScene\\scene.gltf";
//constexpr const char *kDefaultScenePath = "h:\\AMD-GI1\\assets\\SphereRoom\\SphereRoom.glb";
//constexpr const char *kDefaultScenePath = "h:\\ZeroDay_pbr.glb";
//constexpr const char *kDefaultScenePath = "h:\\landscape\\view-0.glb";
//constexpr const char *kDefaultScenePath = "h:\\modern_apartment.glb";
//constexpr const char *kDefaultScenePath = "h:\\LucyCheckerboard.glb";
constexpr const char *kDefaultScenePath = "h:\\StressTest\\StressTest.glb";
//constexpr const char *kDefaultScenePath = "h:\\TheMetro.glb";
//constexpr const char *kDefaultScenePath = "h:\\AncientCathedral.glb";
//constexpr const char *kDefaultScenePath = "h:\\DarkFantasyChurch.glb";
//constexpr const char *kDefaultScenePath = "h:\\SubwayStation.glb";
//constexpr const char *kDefaultScenePath = "h:\\VintageBar.glb";
//constexpr const char *kDefaultScenePath = "H:\\PineForest\\PineForest.gltf";

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

// What BuildPasses made, besides the passes.
struct BuiltPasses
{
	// The UI's pass, which builds the UI at the start of each frame.
	ImGuiPass *ui = nullptr;
	// The passes at the front of the list (the sky and clouds) that run on the async compute queue when it is on,
	// alongside the acceleration structure and light tree on graphics.
	size_t onAsyncCompute = 0;
	// DLSS Frame Generation can run: DLSS-RR is on, so the path tracer writes the depth and motion vectors it reads.
	bool frameGeneration = false;
};

// Makes the passes in the order they add their graph passes.
BuiltPasses BuildPasses(RenderContext *renderContext,
					   Streamline &streamline,
					   const RenderGPUScene &gpuScene,
					   const Camera &camera,
					   RenderSettings &settings,
					   const rg::RenderGraph &graph,
					   std::vector<std::unique_ptr<RenderPass>> &passes)
{
	passes.clear();

	// The reference renders at the window's resolution without DLSS, leaving the DLSS mode for when it ends.
	const bool reference = settings.renderMode == RenderMode::Reference;
	uint32_t renderWidth = 0;
	uint32_t renderHeight = 0;
	const bool rayReconstruction = !reference && settings.dlssMode != DLSSMode::Off &&
								   streamline.IsRayReconstructionSupported() &&
								   streamline.GetOptimalRenderSize(settings.dlssMode,
																   renderContext->GetWindowWidth(),
																   renderContext->GetWindowHeight(),
																   renderWidth,
																   renderHeight);
	if (rayReconstruction)
	{
		renderContext->SetRenderResolution(renderWidth, renderHeight);
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
	PathTracerPass &pathTracerPass = *pathTracer;
	// The HDR image the post passes read: DLSS-RR's output, or the path tracer's, both transients of each frame's graph.
	HdrSource radiance = [&pathTracerPass] { return pathTracerPass.GetOutput(); };
	passes.push_back(std::move(atmosphere));
	if (rayReconstruction)
	{
		auto dlss = std::make_unique<DLSSPass>(renderContext, streamline, *pathTracer, camera, settings);
		radiance = [&dlssPass = *dlss] { return dlssPass.GetOutput(); };
		passes.push_back(std::move(pathTracer));
		passes.push_back(std::move(dlss));
	}
	else
	{
		passes.push_back(std::move(pathTracer));
	}
	auto autoExposure = std::make_unique<AutoExposurePass>(renderContext, radiance, settings);
	const ResourceHandle exposure = autoExposure->GetExposure();
	// The firefly filter's threshold follows the exposure of the frame before last.
	pathTracerPass.SetExposureBuffers(autoExposure->GetTracingExposure(0), autoExposure->GetTracingExposure(1));
	passes.push_back(std::move(autoExposure));
	auto bloom = std::make_unique<BloomPass>(renderContext, radiance, exposure, camera, settings);
	const BloomPass &bloomPass = *bloom;
	passes.push_back(std::move(bloom));
	// The lens flare adds onto bloom's output; the path tracer measures its lights after the paths.
	auto lensFlare = std::make_unique<LensFlarePass>(renderContext, gpuScene, bloomPass, exposure, camera, settings);
	pathTracerPass.SetFlareVisibility(lensFlare->GetVisibility());
	passes.push_back(std::move(lensFlare));
	passes.push_back(std::make_unique<TonemapPass>(renderContext, bloomPass, exposure, settings));
	if (rayReconstruction)
	{
		passes.push_back(std::make_unique<GuideViewPass>(renderContext, pathTracerPass, settings));
	}
	const bool frameGeneration = rayReconstruction && streamline.IsFrameGenerationSupported();
	auto imgui = std::make_unique<ImGuiPass>(
			settings,
			camera,
			streamline,
			frameGeneration ? std::make_unique<FrameGenerationPass>(renderContext, streamline, pathTracerPass) : nullptr,
			gpuScene,
			pathTracerPass,
			graph);
	BuiltPasses built;
	built.ui = imgui.get();
	built.onAsyncCompute = 1;
	built.frameGeneration = frameGeneration;
	passes.push_back(std::move(imgui));
	return built;
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

// The stress test's swarms of emitters, as clusters of emissive instances: cluster c is members[first[c]] to
// members[first[c + 1] - 1].
struct LightClusters
{
	std::vector<uint32_t> first;
	std::vector<uint32_t> members;
	uint32_t Count() const { return first.empty() ? 0u : static_cast<uint32_t>(first.size() - 1); }
};

// Emitters closer than this are in one cluster. Within a swarm, neighbours are at most about 11 m apart, while
// make_stress_scene.py keeps the swarms at least 15 m apart when they don't move.
constexpr float kLightClusterLinkMetres = 13.0f;

LightClusters FindLightClusters(const std::filesystem::path &scenePath, const RenderGPUScene &gpuScene)
{
	LightClusters clusters;
	if (scenePath.stem() != "StressTest")
	{
		return clusters;
	}
	const auto start = std::chrono::steady_clock::now();
	uint32_t count = 0;
	const std::vector<uint32_t> clusterOf = gpuScene.ClusterEmissiveInstances(kLightClusterLinkMetres, count);
	clusters.first.assign(count + 1, 0);
	for (uint32_t c: clusterOf)
	{
		++clusters.first[c + 1];
	}
	for (uint32_t c = 0; c < count; ++c)
	{
		clusters.first[c + 1] += clusters.first[c];
	}
	clusters.members.resize(clusterOf.size());
	std::vector<uint32_t> next(clusters.first.begin(), clusters.first.end() - 1);
	for (uint32_t i = 0; i < static_cast<uint32_t>(clusterOf.size()); ++i)
	{
		clusters.members[next[clusterOf[i]]++] = i;
	}
	logi("Found {} light cluster(s) among {} emissive instance(s) in {:.2f} s",
		 count,
		 clusterOf.size(),
		 std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count());
	return clusters;
}

// Cluster c's emission: its own period of 1.5 to 6 s and phase, and each period on (60 %) or off by a hash, with a
// 0.25 s fade from the last period's state into this one's.
float ClusterFlash(uint32_t cluster, float seconds)
{
	const float period = 1.5f + 4.5f * Hash01(cluster * 3u + 1u);
	const float t = seconds + period * Hash01(cluster * 3u + 2u);
	const uint32_t episode = static_cast<uint32_t>(t / period);
	auto on = [&](uint32_t e) { return Hash01((cluster * 0x9E3779B9u) ^ (e * 0x85EBCA6Bu + 0x27D4EB2Fu)) < 0.6f ? 1.0f : 0.0f; };
	const float from = on(episode - 1u);
	const float to = on(episode);
	return from + (to - from) * SmoothStep(0.0f, 0.25f, t - static_cast<float>(episode) * period);
}

int Run(const char *scenePath)
{
	Window window(WindowDesc{ "PathTracer", 1920, 1080, true });

	// Before the first D3D12 or DXGI call, so that the device, factory and swap chain the render context makes are
	// Streamline's proxies.
	Streamline::Initialise();
	std::unique_ptr<RenderContext> renderContext(CreateRenderContext(
			&window,
			RenderAPI::D3D12,
			kFramesInFlight,
			kSwapchainImageCount,
			false,
			true,
			[](void *device) { Streamline::AttachDevice(static_cast<ID3D12Device *>(device)); }));
	// Declared after the render context, so it shuts Streamline down before the device goes, and before everything
	// else here, so it outlives the DLSS-RR feature the passes use. Without an NVIDIA GPU and a driver that runs it, or
	// without nvngx_dlssd.dll, DLSS-RR is simply unavailable.
	Streamline streamline(static_cast<D3D12RenderContext *>(renderContext.get())->GetDevice()->GetNativeDevice().Get());
	renderContext->CreatePerFrameResources();
	renderContext->InitialiseImgui(window.GetNativeWindow());

	// Every frame's GPU work, recorded into it each frame. Declared after the context, so its GPU objects go first.
	rg::RenderGraph graph;
	graph.SetProfilePasses(true);

	RenderGPUScene gpuScene(renderContext.get());
	if (!LoadScene(scenePath, gpuScene))
	{
		return 1;
	}
	gpuScene.SetName(std::filesystem::path(scenePath).filename().string());

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

	// The scene's animations and skins, posed on the CPU from the GPU scene's tables, which it reads for as long as the
	// scene is loaded. The frame loop applies each new pose back to the scene.
	Animation::AnimationSystem animator(gpuScene.GetAnimationData());

	RenderSettings settings;
	settings.animation.available = animator.HasAnimation();
	settings.animation.duration = animator.GetDuration();
	settings.animation.skinCount = animator.GetSkinCount();
	settings.animation.jointCount = animator.GetJointCount();
	settings.animation.skinnedVertices = animator.GetSkinnedVertexCount();
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

	std::vector<std::unique_ptr<RenderPass>> passes;
	BuiltPasses built = BuildPasses(renderContext.get(), streamline, gpuScene, camera, settings, graph, passes);
	RenderMode builtRenderMode = settings.renderMode;
	DLSSMode builtMode = settings.dlssMode;
	DLSSRayReconstructionPreset builtPreset = settings.rrPreset;

	const std::vector<FlashingBunny> bunnies = FindBunnies(scenePath, gpuScene);
	settings.lights.bunnyCount = static_cast<uint32_t>(bunnies.size());
	std::vector<float> bunnyScales(bunnies.size(), 1.0f);
	float flashSeconds = 0.0f;

	const LightClusters clusters = FindLightClusters(scenePath, gpuScene);
	settings.lights.clusterCount = clusters.Count();
	std::vector<float> clusterScales(clusters.Count(), 1.0f);
	float clusterSeconds = 0.0f;

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
	while (true)
	{
		// The frame starts with the Reflex sleep, before the input it answers is read.
		streamline.BeginFrame();
		if (!window.ProcessEvents())
		{
			break;
		}
		const auto now = std::chrono::steady_clock::now();
		const float deltaSeconds = std::chrono::duration<float>(now - lastTime).count();
		lastTime = now;

		// The UI first, so that the settings it changes hold still while the frame is recorded and while the graph
		// runs the passes' callbacks.
		built.ui->BuildUI(renderContext.get());
		if (settings.renderMode != builtRenderMode || settings.dlssMode != builtMode || settings.rrPreset != builtPreset)
		{
			renderContext->WaitForGPUIdle();
			streamline.FreeRayReconstruction();
			built = BuildPasses(renderContext.get(), streamline, gpuScene, camera, settings, graph, passes);
			builtRenderMode = settings.renderMode;
			builtMode = settings.dlssMode;
			builtPreset = settings.rrPreset;
		}
		// Before the frame tags anything: whether ImGuiPass hands this frame to DLSS-G follows from it. Frame generation
		// needs Reflex, so it keeps it on.
		FrameGenerationOptions frameGeneration;
		frameGeneration.enabled = built.frameGeneration && settings.frameGeneration;
		frameGeneration.framesToGenerate = static_cast<uint32_t>(std::max(settings.framesToGenerate, 1));
		frameGeneration.dynamic = settings.dynamicFrameGeneration;
		streamline.SetFrameGeneration(frameGeneration);
		streamline.SetReflexMode(streamline.IsFrameGenerationOn() && settings.reflexMode == ReflexMode::Off
										 ? ReflexMode::LowLatency
										 : settings.reflexMode);

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

		// A scene with skins but no channels is still posed once, which puts its skinned meshes in the joints' pose.
		if (!animator.IsEmpty())
		{
			AnimationSettings &animation = settings.animation;
			const float duration = animator.GetDuration();
			// The reference holds the scene still, at the time the UI sets.
			if (animation.play && settings.renderMode != RenderMode::Reference)
			{
				animation.seconds += deltaSeconds * animation.speed;
			}
			animation.seconds = duration > 0.0f ? animation.seconds - floorf(animation.seconds / duration) * duration : 0.0f;
			if (animation.seconds != animatedSeconds)
			{
				gpuScene.ApplyAnimationPose(animator.Evaluate(animation.seconds));
				animation.evaluateMs = static_cast<float>(animator.GetEvaluateMs());
				animation.skinningMs = static_cast<float>(animator.GetSkinningMs());
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
		if (settings.lights.flashClusters && settings.renderMode != RenderMode::Reference)
		{
			clusterSeconds += deltaSeconds;
		}
		for (uint32_t c = 0; c < clusters.Count(); ++c)
		{
			const float scale = settings.lights.flashClusters ? ClusterFlash(c, clusterSeconds) : 1.0f;
			if (scale != clusterScales[c])
			{
				for (uint32_t k = clusters.first[c]; k < clusters.first[c + 1]; ++k)
				{
					gpuScene.SetEmissionScale(clusters.members[k], scale);
				}
				clusterScales[c] = scale;
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

		// The light swarms, held still in the reference, which starts over when they move or change.
		if (settings.swarms.enabled && settings.swarms.animate && settings.renderMode != RenderMode::Reference)
		{
			settings.swarms.seconds += deltaSeconds;
		}
		{
			static LightSwarmSettings shownSwarms = {};
			const LightSwarmSettings &s = settings.swarms;
			const bool changed = s.enabled != shownSwarms.enabled || s.swarmCount != shownSwarms.swarmCount ||
								 s.lightsPerSwarm != shownSwarms.lightsPerSwarm || s.seconds != shownSwarms.seconds ||
								 s.speed != shownSwarms.speed || s.flashRate != shownSwarms.flashRate ||
								 s.intensity != shownSwarms.intensity || s.radius != shownSwarms.radius ||
								 s.spread != shownSwarms.spread || s.height != shownSwarms.height;
			if (changed)
			{
				shownSwarms = s;
				++settings.reference.restartCount;
			}
		}

		streamline.MarkSimulationEnd();
		streamline.MarkRenderSubmitStart();
		renderContext->BeginFrame();

		// After BeginFrame, which frees this frame's light buffers.
		{
			RenderGPUScene::LightTreeOptions tree;
			tree.twoLevel = settings.lights.twoLevelTree;
			tree.maxLeafLights = static_cast<uint32_t>(std::max(settings.lights.maxLeafLights, 1));
			const LightSwarmSettings &swarmSettings = settings.swarms;
			RenderGPUScene::LightSwarmDesc swarms;
			swarms.swarmCount = swarmSettings.enabled ? static_cast<uint32_t>(std::max(swarmSettings.swarmCount, 0)) : 0u;
			swarms.lightsPerSwarm = static_cast<uint32_t>(std::max(swarmSettings.lightsPerSwarm, 1));
			swarms.seconds = swarmSettings.seconds;
			swarms.speed = swarmSettings.speed;
			swarms.flashRate = swarmSettings.flashRate;
			swarms.intensity = swarmSettings.intensity;
			swarms.radius = swarmSettings.radius;
			swarms.spread = swarmSettings.spread;
			swarms.height = swarmSettings.height;
			gpuScene.UpdateLights(tree, swarms);
		}

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
		// The frame as a graph. The passes' AddPasses do the CPU work and declare the GPU work, which the graph
		// schedules, places the barriers and queue syncs between and runs. With async compute, the sky and clouds go on
		// the compute queue while the graphics queue updates the TLAS and builds the light tree, which share nothing
		// with them, so the two overlap; a submit then hands the sky to graphics, where the path tracer, the denoiser
		// and the rest run. Without, everything is on graphics. A section can't use a resource on both queues, even
		// only to read it, so nothing that reads the scene or sky can overlap the path tracer.
		graph.Reset();
		FrameGraph frame(graph, *renderContext);
		frame.backBuffer = frame.ImportTexture(renderContext->GetCurrentBackBufferHandle(), "BackBuffer");
		frame.queue = settings.asyncCompute ? QueueType::Compute : QueueType::Graphics;
		for (size_t i = 0; i < built.onAsyncCompute; ++i)
		{
			passes[i]->AddPasses(frame);
		}
		frame.queue = QueueType::Graphics;
		gpuScene.AddAccelerationStructurePasses(frame);
		// Every light, static and dynamic, into the light tree on the GPU each frame: rebuilt whole, or its BLASes
		// refit under a new TLAS.
		gpuScene.AddLightPasses(frame);
		if (settings.asyncCompute)
		{
			graph.Submit();
		}
		for (size_t i = built.onAsyncCompute; i < passes.size(); ++i)
		{
			passes[i]->AddPasses(frame);
		}
		if (!graph.Compile(*renderContext))
		{
			throw std::runtime_error("Render graph: " + graph.GetError());
		}
		// A new compile (the first frame, or a frame whose passes differ from the last compile's) replaces the page of
		// the graph: its passes, batches, barriers and transient memory.
		if (!graph.WasLastCompileReused())
		{
			std::ofstream("RenderGraph.html") << graph.DumpHtml("PathTracer frame");
			logi("Render graph compiled: {} batches, {:.1f} MB of transients, written to RenderGraph.html",
				 graph.GetTotalBatchCount(),
				 static_cast<double>(graph.GetTransientMemorySize()) / (1024.0 * 1024.0));
		}

		renderContext->BeginProfileScope(kGpuFrameScope);
		graph.Execute(*renderContext);
		renderContext->EndProfileScope();

		renderContext->TransitionResource(renderContext->GetCurrentBackBufferHandle(),
										  ResourceLayout::Present,
										  ResourceSync::None,
										  ResourceAccess::NoAccess);

		renderContext->EndGraphicsCommandList();
		renderContext->EndFrame();
		streamline.MarkRenderSubmitEnd();
		// With frame generation on, Streamline's swap chain queues the frame and its generated ones to present on its
		// own queue.
		streamline.MarkPresentStart();
		renderContext->Present();
		streamline.MarkPresentEnd();
		streamline.UpdateFrameGenerationState();

		fmtlog::poll();
	}

	renderContext->WaitForGPUIdle();
	graph.ReleaseResources();
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
