#pragma once

#include <cstdint>
#include <memory>

// Forward-declare the D3D12 interfaces we only reference by pointer. We avoid
// including <d3d12.h> here: this header is pulled into TUs (e.g. main.cpp) that
// rely on the DirectX-Headers copy of d3d12.h being included first, and pulling
// in the Windows SDK <d3d12.h> ahead of it breaks that (missing newer enums).
struct ID3D12Resource;
struct ID3D12GraphicsCommandList;

class D3D12Device;
class D3D12Buffer;

// Opaque NGX types, forward-declared so this header stays NGX-header-free.
struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;

// Upscaling quality preset (mirrors the old sl::DLSSMode; values are the UI order).
enum class DLSSMode : uint32_t
{
	Off = 0,
	DLAA,
	MaxQuality,
	Balanced,
	MaxPerformance,
	UltraPerformance,
};


enum class DLSSPreset : uint32_t
{
	Default = 0,
	PresetJ = 10,
	PresetK = 11,
	PresetL = 12,
	PresetM = 13,
};


enum class DLSSRayReconstructionPreset : uint32_t
{
	Default = 0,
	PresetD = 4,
	PresetE = 5,
	PresetF = 6
};

// DLSS-optimal render resolution for a given output size and mode.
struct DLSSOptimalSettings
{
	uint32_t optimalRenderWidth = 0;
	uint32_t optimalRenderHeight = 0;
	float optimalSharpness = 0.0f;
	uint32_t renderWidthMin = 0;
	uint32_t renderHeightMin = 0;
	uint32_t renderWidthMax = 0;
	uint32_t renderHeightMax = 0;
};

// Which upscaler a set of feature parameters belongs to.
enum class DLSSFeatureKind
{
	SuperResolution, // DLSS-SR
	RayReconstruction, // DLSS-RR
};

// Parameters that define a DLSS feature instance. When any of these change the
// feature must be recreated (mirrors the old option-resubmit / slFreeResources).
struct DLSSFeatureParams
{
	DLSSFeatureKind kind = DLSSFeatureKind::SuperResolution;
	DLSSMode mode = DLSSMode::Balanced;
	DLSSPreset preset = DLSSPreset::Default; // used when kind == SuperResolution
	DLSSRayReconstructionPreset rrPreset = DLSSRayReconstructionPreset::Default; // used when kind == RayReconstruction
	uint32_t renderWidth = 0;
	uint32_t renderHeight = 0;
	uint32_t outputWidth = 0;
	uint32_t outputHeight = 0;
	bool hdr = true;
	bool autoExposure = false;
	bool depthInverted = true; // reversed-Z
	bool packedNormalRoughness = true; // DLSS-RR: roughness in normals.w
};

// Per-frame inputs handed to DLSS at evaluate time (all resources at render res
// except colourOutput which is at output res).
struct DLSSEvalInputs
{
	ID3D12Resource *colourInput = nullptr;
	ID3D12Resource *colourOutput = nullptr;
	ID3D12Resource *depth = nullptr;
	ID3D12Resource *motionVectors = nullptr;
	float jitterOffsetX = 0.0f; // render-pixel space
	float jitterOffsetY = 0.0f;
	uint32_t renderWidth = 0;
	uint32_t renderHeight = 0;
	bool reset = false;

	// DLSS-RR guide buffers (only read when kind == RayReconstruction).
	ID3D12Resource *diffuseAlbedo = nullptr;
	ID3D12Resource *specularAlbedo = nullptr;
	ID3D12Resource *normalRoughness = nullptr;
	// World-space distance to the reflection hit. Optional (null = not supplied), but it is what lets
	// RR reproject reflections against the reflected geometry rather than the surface they sit on.
	ID3D12Resource *specularHitDistance = nullptr;
	// Row-major 4x4 view/proj matrices for RR specular reprojection.
	float worldToView[16] = {};
	float viewToClip[16] = {};
};

// Owns the NGX SDK lifetime and the DLSS-SR / DLSS-RR feature instances.
//
// Not thread-safe (neither is NGX). All calls must come from the render thread.
class NGXContext
{
public:
	NGXContext() = default;
	~NGXContext();

	NGXContext(const NGXContext &) = delete;
	NGXContext &operator=(const NGXContext &) = delete;

	// Initialises NGX for the given D3D12 device. dllSearchPath (optional) is an
	// extra directory NGX will search for the nvngx_dlss*.dll snippets, on top of
	// the application directory. Returns false on failure (DLSS unsupported).
	bool Initialise(D3D12Device *device, const wchar_t *dllSearchPath = nullptr);
	void Shutdown();

	bool IsInitialised() const
	{
		return initialised_;
	}

	// GPU capability checks (populated during Initialise from the capability params).
	bool IsSuperResolutionSupported() const
	{
		return srSupported_;
	}
	bool IsRayReconstructionSupported() const
	{
		return rrSupported_;
	}

	// Query DLSS-optimal render resolution for an output size + mode. These do not
	// touch the GPU and may be called every frame.
	bool GetOptimalSettings(DLSSFeatureKind kind,
							DLSSMode mode,
							uint32_t outputWidth,
							uint32_t outputHeight,
							DLSSOptimalSettings &out) const;

	// Ensure a feature instance matching params exists, (re)creating it on the
	// given command list if the parameters changed since the last create. Also
	// (re)allocates the manual scratch buffer when the size requirement changes.
	// Must be called with an open command list before EvaluateFeature.
	bool EnsureFeature(const DLSSFeatureParams &params, ID3D12GraphicsCommandList *cmdList);

	// Evaluate the active feature. EnsureFeature must have succeeded this frame.
	bool EvaluateFeature(const DLSSEvalInputs &inputs, ID3D12GraphicsCommandList *cmdList);

	// Release any active feature and free its scratch buffer. Safe to call when
	// nothing is active. The caller must have flushed the GPU first.
	void ReleaseFeature();

private:
	bool ParamsMatch(const DLSSFeatureParams &a, const DLSSFeatureParams &b) const;
	void ReleaseFeatureInternal();

	D3D12Device *device_ = nullptr;
	bool initialised_ = false;
	bool srSupported_ = false;
	bool rrSupported_ = false;

	// Capability parameter block, owned by NGX (do not destroy). Used for support
	// queries and optimal-settings callbacks.
	NVSDK_NGX_Parameter *capabilityParams_ = nullptr;

	// The single live feature instance (SR or RR). Recreated when params change.
	NVSDK_NGX_Handle *featureHandle_ = nullptr;
	DLSSFeatureParams activeParams_ = {};
	bool hasActiveFeature_ = false;

	// App-managed scratch buffer for the active feature.
	std::unique_ptr<D3D12Buffer> scratchBuffer_;
	uint64_t scratchBufferSize_ = 0;
};
