#include "Streamline.h"

#include "Maths.h"
#include "fmtlog.h"

#include "directx/d3d12.h"

#include <sl.h>
#include <sl_dlss_d.h>
#include <sl_dlss_g.h>
#include <sl_helpers.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace
{
	// The renderer's one viewport.
	const sl::ViewportHandle kViewport{ 0u };

	// A custom engine without an NVIDIA application id, under the project id it has always used with NGX.
	constexpr char kProjectId[] = "6D5B51F0-076B-486D-9995-5A561043F5C1";
	constexpr char kEngineVersion[] = "1.0";

	bool g_initialised = false;
	bool g_deviceAttached = false;

	// Streamline reports its errors here, including the ones raised later from inside the D3D12 and DXGI proxies.
	void LogMessage(sl::LogType type, const char *message)
	{
		std::string text(message ? message : "");
		while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
		{
			text.pop_back();
		}
		switch (type)
		{
			case sl::LogType::eError:
				loge("Streamline: {}", text);
				break;
			case sl::LogType::eWarn:
				logw("Streamline: {}", text);
				break;
			default:
				logi("Streamline: {}", text);
				break;
		}
	}

	sl::DLSSMode ToSl(DLSSMode mode)
	{
		switch (mode)
		{
			case DLSSMode::DLAA:
				return sl::DLSSMode::eDLAA;
			case DLSSMode::MaxQuality:
				return sl::DLSSMode::eMaxQuality;
			case DLSSMode::Balanced:
				return sl::DLSSMode::eBalanced;
			case DLSSMode::MaxPerformance:
				return sl::DLSSMode::eMaxPerformance;
			case DLSSMode::UltraPerformance:
				return sl::DLSSMode::eUltraPerformance;
			case DLSSMode::Off:
			default:
				return sl::DLSSMode::eOff;
		}
	}

	float4x4 LoadMatrix(const float (&m)[16])
	{
		float4x4 result;
		static_assert(sizeof(result) == sizeof(m));
		std::memcpy(&result, m, sizeof(result));
		return result;
	}

	// Both are row-major with row vectors, translation in the last row.
	void StoreMatrix(sl::float4x4 &out, const float4x4 &m)
	{
		static_assert(sizeof(out) == sizeof(m));
		std::memcpy(&out, &m, sizeof(out));
	}

	sl::float3 ToSl(const float (&v)[3])
	{
		return sl::float3(v[0], v[1], v[2]);
	}

	bool HasStatus(sl::DLSSGStatus status, sl::DLSSGStatus flag)
	{
		return (static_cast<uint32_t>(status) & static_cast<uint32_t>(flag)) != 0;
	}

	// The first reason in a DLSS-G status, or null when it is running.
	const char *DescribeFrameGenerationStatus(sl::DLSSGStatus status)
	{
		if (status == sl::DLSSGStatus::eOk)
		{
			return nullptr;
		}
		if (HasStatus(status, sl::DLSSGStatus::eFailResolutionTooLow))
		{
			return "the output is too small";
		}
		if (HasStatus(status, sl::DLSSGStatus::eFailReflexNotDetectedAtRuntime))
		{
			return "Reflex was not detected";
		}
		if (HasStatus(status, sl::DLSSGStatus::eFailHDRFormatNotSupported))
		{
			return "the swap chain's HDR format is not supported";
		}
		if (HasStatus(status, sl::DLSSGStatus::eFailCommonConstantsInvalid))
		{
			return "the frame's constants are invalid";
		}
		if (HasStatus(status, sl::DLSSGStatus::eFailGetCurrentBackBufferIndexNotCalled))
		{
			return "GetCurrentBackBufferIndex was not called";
		}
		return "unknown failure (see the Streamline log)";
	}

	bool IsSupported(sl::Feature feature, const sl::AdapterInfo &adapter, const char *name)
	{
		const sl::Result result = slIsFeatureSupported(feature, adapter);
		if (result != sl::Result::eOk)
		{
			logw("Streamline: {} is not supported on this GPU/driver/OS ({})", name, sl::getResultAsStr(result));
			return false;
		}
		sl::FeatureVersion version{};
		if (slGetFeatureVersion(feature, version) == sl::Result::eOk)
		{
			logi("Streamline: {} supported (plugin {}, NGX {})",
				 name,
				 version.versionSL.toStr(),
				 version.versionNGX.toStr());
		}
		return true;
	}
} // namespace

bool Streamline::Initialise()
{
	if (g_initialised)
	{
		return true;
	}

	// PCL, which the latency markers go to, is loaded whatever the list says.
	static const sl::Feature kFeatures[] = { sl::kFeatureDLSS_RR, sl::kFeatureDLSS_G, sl::kFeatureReflex };
	sl::Preferences preferences{};
	preferences.showConsole = false;
	preferences.logLevel = sl::LogLevel::eDefault;
	preferences.logMessageCallback = LogMessage;
	preferences.featuresToLoad = kFeatures;
	preferences.numFeaturesToLoad = static_cast<uint32_t>(std::size(kFeatures));
	preferences.engine = sl::EngineType::eCustom;
	preferences.engineVersion = kEngineVersion;
	preferences.projectId = kProjectId;
	preferences.renderAPI = sl::RenderAPI::eD3D12;
	// Not the default eAllowOTA | eLoadDownloadedPlugins: the DLSS that runs is the one the build deploys, so a newer
	// one downloaded in the background can't change the image between runs. The command list state is restored by the
	// caller after each evaluate. DLSS-G's inputs are tagged per frame token.
	preferences.flags =
			sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;

	const sl::Result result = slInit(preferences);
	if (result != sl::Result::eOk)
	{
		loge("Streamline: slInit failed ({}); DLSS is unavailable", sl::getResultAsStr(result));
		return false;
	}
	g_initialised = true;
	return true;
}

void Streamline::AttachDevice(ID3D12Device *device)
{
	if (!g_initialised || g_deviceAttached)
	{
		return;
	}
	// Starts the plugins, which hook the queues and swap chain made from here on.
	const sl::Result result = slSetD3DDevice(device);
	if (result != sl::Result::eOk)
	{
		loge("Streamline: slSetD3DDevice failed ({}); DLSS is unavailable", sl::getResultAsStr(result));
		return;
	}
	g_deviceAttached = true;
}

Streamline::Streamline(ID3D12Device *device)
{
	if (!g_initialised)
	{
		return;
	}
	if (!g_deviceAttached)
	{
		loge("Streamline: the device was not attached before the swap chain was made; DLSS is unavailable");
		return;
	}

	LUID luid = device->GetAdapterLuid();
	sl::AdapterInfo adapter{};
	adapter.deviceLUID = reinterpret_cast<uint8_t *>(&luid);
	adapter.deviceLUIDSizeInBytes = sizeof(luid);
	rrSupported_ = IsSupported(sl::kFeatureDLSS_RR, adapter, "DLSS-RR");

	bool pclLoaded = false;
	pclLoaded_ = slIsFeatureLoaded(sl::kFeaturePCL, pclLoaded) == sl::Result::eOk && pclLoaded;
	const bool reflexSupported = IsSupported(sl::kFeatureReflex, adapter, "Reflex");
	if (reflexSupported)
	{
		sl::ReflexState reflexState{};
		reflexAvailable_ = slReflexGetState(reflexState) == sl::Result::eOk && reflexState.lowLatencyAvailable;
	}
	// DLSS-G paces its presents with Reflex and finds each frame by its latency markers.
	fgSupported_ = reflexSupported && pclLoaded_ && IsSupported(sl::kFeatureDLSS_G, adapter, "DLSS-G");
	if (fgSupported_)
	{
		UpdateFrameGenerationState();
	}
}

Streamline::~Streamline()
{
	if (!g_initialised)
	{
		return;
	}
	// Frees the DLSS-RR feature along with the plugins. The caller has waited for the GPU.
	const sl::Result result = slShutdown();
	if (result != sl::Result::eOk)
	{
		logw("Streamline: slShutdown failed ({})", sl::getResultAsStr(result));
	}
	g_initialised = false;
	g_deviceAttached = false;
}

bool Streamline::GetOptimalRenderSize(
		DLSSMode mode, uint32_t outputWidth, uint32_t outputHeight, uint32_t &renderWidth, uint32_t &renderHeight) const
{
	if (!rrSupported_ || mode == DLSSMode::Off)
	{
		return false;
	}

	sl::DLSSDOptions options{};
	options.mode = ToSl(mode);
	options.outputWidth = outputWidth;
	options.outputHeight = outputHeight;
	sl::DLSSDOptimalSettings settings{};
	const sl::Result result = slDLSSDGetOptimalSettings(options, settings);
	if (result != sl::Result::eOk || settings.optimalRenderWidth == 0 || settings.optimalRenderHeight == 0)
	{
		logw("Streamline: no DLSS-RR render size for {}x{} ({})",
			 outputWidth,
			 outputHeight,
			 sl::getResultAsStr(result));
		return false;
	}
	renderWidth = settings.optimalRenderWidth;
	renderHeight = settings.optimalRenderHeight;
	return true;
}

void Streamline::BeginFrame()
{
	frameToken_ = nullptr;
	if (!g_initialised)
	{
		return;
	}
	sl::FrameToken *token = nullptr;
	if (slGetNewFrameToken(token) != sl::Result::eOk)
	{
		return;
	}
	frameToken_ = token;
	// Every frame, whatever the Reflex mode: off, it only paces to a frame limit, of which there is none.
	if (reflexModeSet_)
	{
		slReflexSleep(*frameToken_);
	}
	SetMarker(static_cast<uint32_t>(sl::PCLMarker::eSimulationStart));
}

void Streamline::SetMarker(uint32_t marker) const
{
	if (pclLoaded_ && frameToken_)
	{
		slPCLSetMarker(static_cast<sl::PCLMarker>(marker), *frameToken_);
	}
}

void Streamline::MarkSimulationEnd()
{
	SetMarker(static_cast<uint32_t>(sl::PCLMarker::eSimulationEnd));
}

void Streamline::MarkRenderSubmitStart()
{
	SetMarker(static_cast<uint32_t>(sl::PCLMarker::eRenderSubmitStart));
}

void Streamline::MarkRenderSubmitEnd()
{
	SetMarker(static_cast<uint32_t>(sl::PCLMarker::eRenderSubmitEnd));
}

void Streamline::MarkPresentStart()
{
	SetMarker(static_cast<uint32_t>(sl::PCLMarker::ePresentStart));
}

void Streamline::MarkPresentEnd()
{
	SetMarker(static_cast<uint32_t>(sl::PCLMarker::ePresentEnd));
}

void Streamline::SetReflexMode(ReflexMode mode)
{
	if (!g_initialised || (reflexModeSet_ && mode == reflexMode_))
	{
		return;
	}
	// Without the low-latency mode the options still go in (off), so that the sleep runs.
	sl::ReflexOptions options{};
	options.mode = reflexAvailable_ ? static_cast<sl::ReflexMode>(mode) : sl::ReflexMode::eOff;
	const sl::Result result = slReflexSetOptions(options);
	if (result != sl::Result::eOk)
	{
		if (!reflexModeSet_)
		{
			logw("Streamline: slReflexSetOptions failed ({})", sl::getResultAsStr(result));
		}
		return;
	}
	reflexModeSet_ = true;
	reflexMode_ = mode;
}

void Streamline::SetFrameGeneration(const FrameGenerationOptions &requested)
{
	if (!fgSupported_)
	{
		fgOn_ = false;
		return;
	}
	FrameGenerationOptions wanted = requested;
	wanted.framesToGenerate = std::clamp(wanted.framesToGenerate, 1u, std::max(fgState_.maxFramesToGenerate, 1u));
	wanted.dynamic = wanted.dynamic && fgState_.dynamicSupported;
	if (fgOptionsSet_ && wanted.enabled == fgOptions_.enabled &&
		wanted.framesToGenerate == fgOptions_.framesToGenerate && wanted.dynamic == fgOptions_.dynamic)
	{
		return;
	}

	// Off frees DLSS-G's memory, as it is turned off in the settings rather than for a pause menu. The swap chain stays
	// Streamline's proxy, which renders off screen and presents on DLSS-G's queue even then.
	sl::DLSSGOptions options{};
	options.mode = !wanted.enabled	? sl::DLSSGMode::eOff
				   : wanted.dynamic ? sl::DLSSGMode::eDynamic
									: sl::DLSSGMode::eOn;
	options.numFramesToGenerate = wanted.framesToGenerate;
	options.dynamicTargetFrameRate = 0.0f; // the display's refresh rate
	const sl::Result result = slDLSSGSetOptions(kViewport, options);
	if (result != sl::Result::eOk)
	{
		loge("Streamline: slDLSSGSetOptions failed ({})", sl::getResultAsStr(result));
		fgOn_ = false;
		return;
	}
	if (wanted.enabled)
	{
		logi("Streamline: DLSS-G on, {}x{}", wanted.framesToGenerate + 1, wanted.dynamic ? " at most (dynamic)" : "");
	}
	else if (fgOptionsSet_)
	{
		logi("Streamline: DLSS-G off");
	}
	fgOptionsSet_ = true;
	fgOptions_ = wanted;
	fgOn_ = wanted.enabled;
}

bool Streamline::TagFrameGeneration(const FrameGenerationInputs &in, ID3D12GraphicsCommandList *commandList)
{
	if (!fgOn_ || !frameToken_)
	{
		return false;
	}

	// eOnlyValidNow: Streamline copies each one in this command list now, as they are the frame graph's transients (or
	// the scene before the UI goes on) and hold something else by present. They are in the copy-source layout, so it
	// records no barriers on them.
	constexpr uint32_t kState = D3D12_RESOURCE_STATE_COPY_SOURCE;
	const sl::Extent renderExtent{ 0, 0, in.renderWidth, in.renderHeight };
	const sl::Extent outputExtent{ 0, 0, in.outputWidth, in.outputHeight };
	sl::Resource depth(sl::ResourceType::eTex2d, in.depth, kState);
	sl::Resource motionVectors(sl::ResourceType::eTex2d, in.motionVectors, kState);
	sl::Resource hudless(sl::ResourceType::eTex2d, in.hudless, kState);
	sl::Resource ui(sl::ResourceType::eTex2d, in.ui, kState);
	const sl::ResourceTag tags[4] = {
		sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &renderExtent),
		sl::ResourceTag(&motionVectors,
						sl::kBufferTypeMotionVectors,
						sl::ResourceLifecycle::eOnlyValidNow,
						&renderExtent),
		sl::ResourceTag(&hudless, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eOnlyValidNow, &outputExtent),
		sl::ResourceTag(&ui, sl::kBufferTypeUIColorAndAlpha, sl::ResourceLifecycle::eOnlyValidNow, &outputExtent),
	};
	const uint32_t count = in.ui ? 4u : 3u;
	const sl::Result result = slSetTagForFrame(*frameToken_, kViewport, tags, count, commandList);
	if (result != sl::Result::eOk)
	{
		loge("Streamline: DLSS-G tagging failed ({})", sl::getResultAsStr(result));
		return false;
	}
	return true;
}

void Streamline::UpdateFrameGenerationState()
{
	if (!fgSupported_)
	{
		return;
	}
	// No options: the VRAM estimate they could ask for is too slow for every frame.
	sl::DLSSGState state{};
	if (slDLSSGGetState(kViewport, state, nullptr) != sl::Result::eOk)
	{
		return;
	}
	fgState_.maxFramesToGenerate = std::max(state.numFramesToGenerateMax, 1u);
	fgState_.dynamicSupported = state.bIsDynamicMFGSupported == sl::Boolean::eTrue;
	fgState_.framesPresented = fgOn_ ? std::max(state.numFramesActuallyPresented, 1u) : 1u;
	fgState_.minWidthOrHeight = state.minWidthOrHeight;
	fgState_.error = fgOn_ ? DescribeFrameGenerationStatus(state.status) : nullptr;
}

bool Streamline::EvaluateRayReconstruction(const RayReconstructionInputs &in, ID3D12GraphicsCommandList *commandList)
{
	if (!rrSupported_ || !frameToken_ || in.mode == DLSSMode::Off)
	{
		return false;
	}
	const sl::FrameToken &frame = *frameToken_;

	const float4x4 worldToView = LoadMatrix(in.worldToView);
	const float4x4 viewToClip = LoadMatrix(in.viewToClip);
	const float4x4 viewToWorld = InvertMatrix(worldToView);
	const float4x4 clipToView = InvertMatrix(viewToClip);
	// Row vectors: this frame's clip space back to the world, then into last frame's clip space.
	const float4x4 clipToPrevClip =
			clipToView * viewToWorld * LoadMatrix(in.prevWorldToView) * LoadMatrix(in.prevViewToClip);

	// Streamline compares the options with the last frame's and recreates the feature when they differ.
	sl::DLSSDOptions options{};
	options.mode = ToSl(in.mode);
	options.outputWidth = in.outputWidth;
	options.outputHeight = in.outputHeight;
	options.colorBuffersHDR = sl::Boolean::eTrue; // linear radiance, exposed by the tonemapper afterwards
	options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked; // roughness in normalRoughness.w
	StoreMatrix(options.worldToCameraView, worldToView);
	StoreMatrix(options.cameraViewToWorld, viewToWorld);
	const sl::DLSSDPreset preset = static_cast<sl::DLSSDPreset>(in.preset);
	options.dlaaPreset = preset;
	options.qualityPreset = preset;
	options.balancedPreset = preset;
	options.performancePreset = preset;
	options.ultraPerformancePreset = preset;
	options.ultraQualityPreset = preset;
	sl::Result result = slDLSSDSetOptions(kViewport, options);
	if (result != sl::Result::eOk)
	{
		loge("Streamline: slDLSSDSetOptions failed ({})", sl::getResultAsStr(result));
		return false;
	}

	sl::Constants constants{};
	StoreMatrix(constants.cameraViewToClip, viewToClip);
	StoreMatrix(constants.clipToCameraView, clipToView);
	StoreMatrix(constants.clipToLensClip, CreateIdentityMatrix());
	StoreMatrix(constants.clipToPrevClip, clipToPrevClip);
	StoreMatrix(constants.prevClipToClip, InvertMatrix(clipToPrevClip));
	constants.jitterOffset = sl::float2(in.jitterX, in.jitterY);
	// The motion vectors are in render pixels; this takes them to -1..1.
	constants.mvecScale =
			sl::float2(1.0f / static_cast<float>(in.renderWidth), 1.0f / static_cast<float>(in.renderHeight));
	constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
	constants.cameraPos = ToSl(in.cameraPosition);
	constants.cameraUp = ToSl(in.cameraUp);
	constants.cameraRight = ToSl(in.cameraRight);
	constants.cameraFwd = ToSl(in.cameraForward);
	constants.cameraNear = in.nearZ;
	constants.cameraFar = in.farZ;
	constants.cameraFOV = in.fovY;
	constants.cameraAspectRatio = static_cast<float>(in.outputWidth) / static_cast<float>(in.outputHeight);
	constants.depthInverted = sl::Boolean::eTrue; // reversed Z
	constants.cameraMotionIncluded = sl::Boolean::eTrue;
	constants.motionVectors3D = sl::Boolean::eFalse;
	constants.reset = in.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	constants.orthographicProjection = sl::Boolean::eFalse;
	constants.motionVectorsDilated = sl::Boolean::eFalse;
	constants.motionVectorsJittered = sl::Boolean::eFalse;
	result = slSetConstants(constants, frame, kViewport);
	if (result != sl::Result::eOk)
	{
		loge("Streamline: slSetConstants failed ({})", sl::getResultAsStr(result));
		return false;
	}

	// The buffers go in as local tags of this evaluate, which Streamline reads in place without copying. Their states
	// are the ones DLSS-RR wants (read in compute, output as a UAV), so it transitions nothing.
	constexpr uint32_t kRead =
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	constexpr uint32_t kWrite = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	const sl::Extent renderExtent{ 0, 0, in.renderWidth, in.renderHeight };
	const sl::Extent outputExtent{ 0, 0, in.outputWidth, in.outputHeight };

	constexpr size_t kMaxTags = 10;
	std::vector<sl::Resource> resources;
	std::vector<sl::ResourceTag> tags;
	resources.reserve(kMaxTags); // the tags point into it
	tags.reserve(kMaxTags);
	auto tag = [&](ID3D12Resource *resource, sl::BufferType type, uint32_t state, const sl::Extent &extent)
	{
		if (resource)
		{
			resources.emplace_back(sl::ResourceType::eTex2d, resource, state);
			tags.emplace_back(&resources.back(), type, sl::ResourceLifecycle::eValidUntilEvaluate, &extent);
		}
	};
	tag(in.colourInput, sl::kBufferTypeScalingInputColor, kRead, renderExtent);
	tag(in.colourOutput, sl::kBufferTypeScalingOutputColor, kWrite, outputExtent);
	tag(in.depth, sl::kBufferTypeDepth, kRead, renderExtent);
	tag(in.motionVectors, sl::kBufferTypeMotionVectors, kRead, renderExtent);
	tag(in.diffuseAlbedo, sl::kBufferTypeAlbedo, kRead, renderExtent);
	tag(in.specularAlbedo, sl::kBufferTypeSpecularAlbedo, kRead, renderExtent);
	tag(in.normalRoughness, sl::kBufferTypeNormalRoughness, kRead, renderExtent);
	tag(in.specularMotionVectors, sl::kBufferTypeSpecularMotionVectors, kRead, renderExtent);
	tag(in.biasCurrentColourMask, sl::kBufferTypeBiasCurrentColorHint, kRead, renderExtent);
	tag(in.responsivityMask, sl::kBufferTypeResponsivityMask, kRead, renderExtent);

	std::vector<const sl::BaseStructure *> inputs;
	inputs.reserve(1 + tags.size());
	inputs.push_back(&kViewport);
	for (const sl::ResourceTag &t: tags)
	{
		inputs.push_back(&t);
	}

	result = slEvaluateFeature(sl::kFeatureDLSS_RR,
							   frame,
							   inputs.data(),
							   static_cast<uint32_t>(inputs.size()),
							   commandList);
	if (result != sl::Result::eOk)
	{
		loge("Streamline: DLSS-RR evaluate failed ({})", sl::getResultAsStr(result));
		return false;
	}
	return true;
}

void Streamline::FreeRayReconstruction()
{
	if (rrSupported_)
	{
		// Fails harmlessly when DLSS-RR never ran.
		slFreeResources(sl::kFeatureDLSS_RR, kViewport);
	}
}
