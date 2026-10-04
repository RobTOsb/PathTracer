#include "DLSSCommon.h"

#include "D3D12Device.h"
#include "D3D12Buffer.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd_d3d.h>

#include <stdio.h>
#include <string>

#include <Windows.h>

static std::wstring GetExecutableDirectory()
{
	wchar_t modulePath[MAX_PATH] = {};
	const DWORD len = GetModuleFileNameW(nullptr, modulePath, MAX_PATH);
	if (len == 0 || len == MAX_PATH)
		return L".";

	std::wstring path(modulePath, len);
	const size_t slash = path.find_last_of(L"\\/");
	if (slash == std::wstring::npos)
		return L".";
	return path.substr(0, slash);
}

static const char *kNGXProjectId = "6D5B51F0-076B-486D-9995-5A561043F5C1";
static const char *kNGXEngineVersion = "1.0";

static NVSDK_NGX_PerfQuality_Value ToNGXPerfQuality(DLSSMode mode)
{
	switch (mode)
	{
		case DLSSMode::DLAA:
			return NVSDK_NGX_PerfQuality_Value_DLAA;
		case DLSSMode::MaxQuality:
			return NVSDK_NGX_PerfQuality_Value_MaxQuality;
		case DLSSMode::Balanced:
			return NVSDK_NGX_PerfQuality_Value_Balanced;
		case DLSSMode::MaxPerformance:
			return NVSDK_NGX_PerfQuality_Value_MaxPerf;
		case DLSSMode::UltraPerformance:
			return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
		case DLSSMode::Off:
		default:
			return NVSDK_NGX_PerfQuality_Value_Balanced;
	}
}

static unsigned int ToNGXDLSSPreset(DLSSPreset preset)
{
	return static_cast<unsigned int>(preset);
}

static unsigned int ToNGXRayReconstructionPreset(DLSSRayReconstructionPreset preset)
{
	return static_cast<unsigned int>(preset);
}

NGXContext::~NGXContext()
{
	Shutdown();
}

bool NGXContext::Initialise(D3D12Device *device, const wchar_t *dllSearchPath)
{
	if (initialised_)
		return true;

	device_ = device;
	ID3D12Device *nativeDevice = device_->GetNativeDevice().Get();

	// Feature-DLL search paths. NGX must be given ABSOLUTE paths to find the
	// nvngx_dlss*.dll snippets; the executable directory (where the build copies
	// them) is the primary path, with any caller-supplied path as a fallback.
	const std::wstring exeDir = GetExecutableDirectory();
	const std::wstring fallbackDir = dllSearchPath ? std::wstring(dllSearchPath) : std::wstring();

	const wchar_t *paths[2] = {};
	unsigned int numPaths = 0;
	paths[numPaths++] = exeDir.c_str();
	if (!fallbackDir.empty())
		paths[numPaths++] = fallbackDir.c_str();

	NVSDK_NGX_FeatureCommonInfo featureInfo = {};
	featureInfo.PathListInfo.Path = paths;
	featureInfo.PathListInfo.Length = numPaths;

	printf("[NGX] Feature-DLL search dir: %ls\n", exeDir.c_str());

	NVSDK_NGX_Result res = NVSDK_NGX_D3D12_Init_with_ProjectID(kNGXProjectId,
															   NVSDK_NGX_ENGINE_TYPE_CUSTOM,
															   kNGXEngineVersion,
															   exeDir.c_str(),
															   nativeDevice,
															   &featureInfo,
															   NVSDK_NGX_Version_API);
	printf("[NGX] Init_with_ProjectID(\"%s\") -> 0x%08x\n", kNGXProjectId, (unsigned int) res);
	if (NVSDK_NGX_FAILED(res))
	{
		printf("[NGX] Init failed; DLSS disabled.\n");
		fflush(stdout);
		return false;
	}

	// Capability parameters carry the *_Available flags and the optimal-settings
	// callbacks. These are owned by NGX; never destroy them.
	res = NVSDK_NGX_D3D12_GetCapabilityParameters(&capabilityParams_);
	printf("[NGX] GetCapabilityParameters -> 0x%08x (params=%p)\n", (unsigned int) res, (void *) capabilityParams_);
	if (NVSDK_NGX_FAILED(res) || !capabilityParams_)
	{
		printf("[NGX] GetCapabilityParameters failed; DLSS disabled.\n");
		fflush(stdout);
		NVSDK_NGX_D3D12_Shutdown1(nativeDevice);
		return false;
	}

	// DLSS-SR availability. FeatureInitResult tells us *why* it is unavailable.
	{
		int available = 0;
		NVSDK_NGX_Parameter_GetI(capabilityParams_, NVSDK_NGX_Parameter_SuperSampling_Available, &available);

		int initResult = 0;
		NVSDK_NGX_Parameter_GetI(capabilityParams_,
								 NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
								 &initResult);
		int needsUpdatedDriver = 0;
		NVSDK_NGX_Parameter_GetI(capabilityParams_,
								 NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
								 &needsUpdatedDriver);
		int minMajor = 0, minMinor = 0;
		NVSDK_NGX_Parameter_GetI(capabilityParams_,
								 NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,
								 &minMajor);
		NVSDK_NGX_Parameter_GetI(capabilityParams_,
								 NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,
								 &minMinor);

		srSupported_ = (available != 0);
		printf("[NGX] DLSS-SR: available=%d initResult=0x%08x needsUpdatedDriver=%d minDriver=%d.%d\n",
			   available,
			   (unsigned int) initResult,
			   needsUpdatedDriver,
			   minMajor,
			   minMinor);
	}

	// DLSS-RR (Ray Reconstruction) availability.
	{
		int available = 0;
		NVSDK_NGX_Parameter_GetI(capabilityParams_,
								 NVSDK_NGX_Parameter_SuperSamplingDenoising_Available,
								 &available);

		int initResult = 0;
		NVSDK_NGX_Parameter_GetI(capabilityParams_,
								 NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult,
								 &initResult);
		int needsUpdatedDriver = 0;
		NVSDK_NGX_Parameter_GetI(capabilityParams_,
								 NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver,
								 &needsUpdatedDriver);

		rrSupported_ = (available != 0);
		printf("[NGX] DLSS-RR: available=%d initResult=0x%08x needsUpdatedDriver=%d\n",
			   available,
			   (unsigned int) initResult,
			   needsUpdatedDriver);
	}

	printf("[NGX] DLSS-SR supported: %d, DLSS-RR supported: %d\n", srSupported_, rrSupported_);
	fflush(stdout);

	initialised_ = true;
	return true;
}

void NGXContext::Shutdown()
{
	if (!initialised_)
		return;

	ReleaseFeatureInternal();

	// Capability params are owned by NGX and freed by Shutdown; do not destroy.
	capabilityParams_ = nullptr;

	if (device_)
		NVSDK_NGX_D3D12_Shutdown1(device_->GetNativeDevice().Get());

	device_ = nullptr;
	srSupported_ = false;
	rrSupported_ = false;
	initialised_ = false;
}

bool NGXContext::GetOptimalSettings(DLSSFeatureKind kind,
									DLSSMode mode,
									uint32_t outputWidth,
									uint32_t outputHeight,
									DLSSOptimalSettings &out) const
{
	if (!initialised_ || !capabilityParams_)
		return false;

	const NVSDK_NGX_PerfQuality_Value perfQuality = ToNGXPerfQuality(mode);

	unsigned int optimalW = 0, optimalH = 0;
	unsigned int maxW = 0, maxH = 0, minW = 0, minH = 0;
	float sharpness = 0.0f;

	NVSDK_NGX_Result res;
	if (kind == DLSSFeatureKind::RayReconstruction)
	{
		res = NGX_DLSSD_GET_OPTIMAL_SETTINGS(capabilityParams_,
											 outputWidth,
											 outputHeight,
											 perfQuality,
											 &optimalW,
											 &optimalH,
											 &maxW,
											 &maxH,
											 &minW,
											 &minH,
											 &sharpness);
	}
	else
	{
		res = NGX_DLSS_GET_OPTIMAL_SETTINGS(capabilityParams_,
											outputWidth,
											outputHeight,
											perfQuality,
											&optimalW,
											&optimalH,
											&maxW,
											&maxH,
											&minW,
											&minH,
											&sharpness);
	}

	if (NVSDK_NGX_FAILED(res) || optimalW == 0 || optimalH == 0)
		return false;

	out.optimalRenderWidth = optimalW;
	out.optimalRenderHeight = optimalH;
	out.optimalSharpness = sharpness;
	out.renderWidthMin = minW;
	out.renderHeightMin = minH;
	out.renderWidthMax = maxW;
	out.renderHeightMax = maxH;
	return true;
}

bool NGXContext::ParamsMatch(const DLSSFeatureParams &a, const DLSSFeatureParams &b) const
{
	return a.kind == b.kind && a.mode == b.mode && a.preset == b.preset && a.rrPreset == b.rrPreset &&
		   a.renderWidth == b.renderWidth && a.renderHeight == b.renderHeight && a.outputWidth == b.outputWidth &&
		   a.outputHeight == b.outputHeight && a.hdr == b.hdr && a.autoExposure == b.autoExposure &&
		   a.depthInverted == b.depthInverted && a.packedNormalRoughness == b.packedNormalRoughness;
}

bool NGXContext::EnsureFeature(const DLSSFeatureParams &params, ID3D12GraphicsCommandList *cmdList)
{
	if (!initialised_ || !capabilityParams_)
		return false;

	if (params.kind == DLSSFeatureKind::RayReconstruction && !rrSupported_)
		return false;
	if (params.kind == DLSSFeatureKind::SuperResolution && !srSupported_)
		return false;

	// Reuse the existing feature if nothing changed.
	if (hasActiveFeature_ && featureHandle_ && ParamsMatch(activeParams_, params))
		return true;

	// Parameters changed (or first creation): tear down the old instance. The
	// caller is responsible for having flushed the GPU when sizes change.
	ReleaseFeatureInternal();

	// Common feature-create flags.
	int createFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes; // motion vectors at render res
	if (params.hdr)
		createFlags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
	if (params.depthInverted)
		createFlags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
	if (params.autoExposure)
		createFlags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

	NVSDK_NGX_Result res;

	if (params.kind == DLSSFeatureKind::RayReconstruction)
	{
		const unsigned int preset = ToNGXRayReconstructionPreset(params.rrPreset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_,
								  NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
								  preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_,
								  NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
								  preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_,
								  NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
								  preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_,
								  NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
								  preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_,
								  NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
								  preset);

		NVSDK_NGX_DLSSD_Create_Params createParams = {};
		createParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
		createParams.InRoughnessMode = params.packedNormalRoughness ? NVSDK_NGX_DLSS_Roughness_Mode_Packed
																	 : NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;
		createParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;
		createParams.InWidth = params.renderWidth;
		createParams.InHeight = params.renderHeight;
		createParams.InTargetWidth = params.outputWidth;
		createParams.InTargetHeight = params.outputHeight;
		createParams.InPerfQualityValue = ToNGXPerfQuality(params.mode);
		createParams.InFeatureCreateFlags = createFlags;

		res = NGX_D3D12_CREATE_DLSSD_EXT(cmdList, 1, 1, &featureHandle_, capabilityParams_, &createParams);
	}
	else
	{
		// Per-mode render-preset hints.
		const unsigned int preset = ToNGXDLSSPreset(params.preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, preset);
		NVSDK_NGX_Parameter_SetUI(capabilityParams_,
								  NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
								  preset);

		NVSDK_NGX_DLSS_Create_Params createParams = {};
		createParams.Feature.InWidth = params.renderWidth;
		createParams.Feature.InHeight = params.renderHeight;
		createParams.Feature.InTargetWidth = params.outputWidth;
		createParams.Feature.InTargetHeight = params.outputHeight;
		createParams.Feature.InPerfQualityValue = ToNGXPerfQuality(params.mode);
		createParams.InFeatureCreateFlags = createFlags;

		res = NGX_D3D12_CREATE_DLSS_EXT(cmdList, 1, 1, &featureHandle_, capabilityParams_, &createParams);
	}

	if (NVSDK_NGX_FAILED(res) || !featureHandle_)
	{
		printf("NGX create feature failed: 0x%08x\n", (unsigned int) res);
		featureHandle_ = nullptr;
		return false;
	}

	// Allocate the app-managed scratch buffer. Most current DLSS builds report 0;
	// we still honour whatever size NGX asks for.
	const NVSDK_NGX_Feature feature = (params.kind == DLSSFeatureKind::RayReconstruction)
											  ? NVSDK_NGX_Feature_RayReconstruction
											  : NVSDK_NGX_Feature_SuperSampling;
	size_t scratchSize = 0;
	const NVSDK_NGX_Result scratchRes =
			NVSDK_NGX_D3D12_GetScratchBufferSize(feature, capabilityParams_, &scratchSize);
	if (NVSDK_NGX_SUCCEED(scratchRes) && scratchSize > 0)
	{
		D3D12ResourceDesc1 desc = D3D12ResourceDesc1::Buffer(scratchSize);
		desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		scratchBuffer_ = device_->CreateBuffer(desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_BARRIER_LAYOUT_UNDEFINED);
		scratchBuffer_->GetResource()->SetName(L"NGXDLSSScratch");
		scratchBufferSize_ = scratchSize;

		// Bind the scratch buffer for the lifetime of this feature instance.
		NVSDK_NGX_Parameter_SetD3d12Resource(capabilityParams_, NVSDK_NGX_Parameter_Scratch, scratchBuffer_->GetResource());
		NVSDK_NGX_Parameter_SetULL(capabilityParams_, NVSDK_NGX_Parameter_Scratch_SizeInBytes, scratchBufferSize_);
	}
	else
	{
		scratchBuffer_.reset();
		scratchBufferSize_ = 0;
	}

	activeParams_ = params;
	hasActiveFeature_ = true;
	return true;
}

bool NGXContext::EvaluateFeature(const DLSSEvalInputs &inputs, ID3D12GraphicsCommandList *cmdList)
{
	if (!hasActiveFeature_ || !featureHandle_ || !capabilityParams_)
		return false;

	NVSDK_NGX_Result res;

	if (activeParams_.kind == DLSSFeatureKind::RayReconstruction)
	{
		NVSDK_NGX_D3D12_DLSSD_Eval_Params evalParams = {};
		evalParams.pInColor = inputs.colourInput;
		evalParams.pInOutput = inputs.colourOutput;
		evalParams.pInDepth = inputs.depth;
		evalParams.pInMotionVectors = inputs.motionVectors;
		evalParams.InJitterOffsetX = inputs.jitterOffsetX;
		evalParams.InJitterOffsetY = inputs.jitterOffsetY;
		evalParams.InReset = inputs.reset ? 1 : 0;
		evalParams.InMVScaleX = 1.0f;
		evalParams.InMVScaleY = 1.0f;
		evalParams.InRenderSubrectDimensions.Width = inputs.renderWidth;
		evalParams.InRenderSubrectDimensions.Height = inputs.renderHeight;
		evalParams.pInDiffuseAlbedo = inputs.diffuseAlbedo;
		evalParams.pInSpecularAlbedo = inputs.specularAlbedo;
		// Packed normal+roughness: bind the same texture as normals; the feature
		// was created with Roughness_Mode_Packed so it reads roughness from .w.
		evalParams.pInNormals = inputs.normalRoughness;
		evalParams.pInRoughness = inputs.normalRoughness;
		evalParams.pInSpecularHitDistance = inputs.specularHitDistance;
		evalParams.pInWorldToViewMatrix = const_cast<float *>(inputs.worldToView);
		evalParams.pInViewToClipMatrix = const_cast<float *>(inputs.viewToClip);

		res = NGX_D3D12_EVALUATE_DLSSD_EXT(cmdList, featureHandle_, capabilityParams_, &evalParams);
	}
	else
	{
		NVSDK_NGX_D3D12_DLSS_Eval_Params evalParams = {};
		evalParams.Feature.pInColor = inputs.colourInput;
		evalParams.Feature.pInOutput = inputs.colourOutput;
		evalParams.Feature.InSharpness = 0.0f;
		evalParams.pInDepth = inputs.depth;
		evalParams.pInMotionVectors = inputs.motionVectors;
		evalParams.InJitterOffsetX = inputs.jitterOffsetX;
		evalParams.InJitterOffsetY = inputs.jitterOffsetY;
		evalParams.InReset = inputs.reset ? 1 : 0;
		evalParams.InMVScaleX = 1.0f;
		evalParams.InMVScaleY = 1.0f;
		evalParams.InRenderSubrectDimensions.Width = inputs.renderWidth;
		evalParams.InRenderSubrectDimensions.Height = inputs.renderHeight;

		res = NGX_D3D12_EVALUATE_DLSS_EXT(cmdList, featureHandle_, capabilityParams_, &evalParams);
	}

	if (NVSDK_NGX_FAILED(res))
	{
		printf("NGX evaluate feature failed: 0x%08x\n", (unsigned int) res);
		return false;
	}
	return true;
}

void NGXContext::ReleaseFeatureInternal()
{
	if (featureHandle_)
	{
		NVSDK_NGX_D3D12_ReleaseFeature(featureHandle_);
		featureHandle_ = nullptr;
	}
	scratchBuffer_.reset();
	scratchBufferSize_ = 0;
	hasActiveFeature_ = false;
	activeParams_ = {};
}

void NGXContext::ReleaseFeature()
{
	ReleaseFeatureInternal();
}
