#pragma once

#include <cstdint>

struct ID3D12Device;
struct ID3D12Resource;
struct ID3D12GraphicsCommandList;

namespace sl
{
	struct FrameToken;
}

// Upscaling quality mode, in the UI's order.
enum class DLSSMode : uint32_t
{
	Off = 0,
	DLAA,
	MaxQuality,
	Balanced,
	MaxPerformance,
	UltraPerformance,
};

// DLSS-RR render preset; the values are sl::DLSSDPreset's.
enum class DLSSRayReconstructionPreset : uint32_t
{
	Default = 0,
	PresetD = 4,
	PresetE = 5,
	PresetF = 6
};

// NVIDIA Reflex low-latency mode; the values are sl::ReflexMode's.
enum class ReflexMode : uint32_t
{
	Off = 0,
	LowLatency,
	LowLatencyWithBoost,
};

// What DLSS Frame Generation is asked for, applied by Streamline::SetFrameGeneration.
struct FrameGenerationOptions
{
	bool enabled = false;
	// Generated frames presented after each rendered one: 1 = 2x, up to the GPU's FrameGenerationState max.
	uint32_t framesToGenerate = 1;
	// DLSS-G picks the multiplier each frame (up to framesToGenerate) to hold the display's refresh rate. Only where
	// FrameGenerationState::dynamicSupported.
	bool dynamic = false;
};

// What DLSS Frame Generation last reported, refreshed after each present.
struct FrameGenerationState
{
	uint32_t maxFramesToGenerate = 1; // 1 on GPUs without multi frame generation
	bool dynamicSupported = false;
	// Frames presented for the last rendered one, generated included: 1 when off, 2 at 2x.
	uint32_t framesPresented = 1;
	uint32_t minWidthOrHeight = 0; // the smallest output it runs at
	// Why it is not running although on; null when it runs (or is off).
	const char *error = nullptr;
};

// The buffers DLSS-G interpolates between frames, tagged once per frame before present. Streamline copies each when
// tagged, so every one must be in the copy-source layout at that point and may be reused right after.
struct FrameGenerationInputs
{
	ID3D12Resource *depth = nullptr; // render resolution, the same as DLSS-RR's
	ID3D12Resource *motionVectors = nullptr; // render resolution, the same as DLSS-RR's
	ID3D12Resource *hudless = nullptr; // the back buffer's image before the UI, output resolution and format
	// Optional, output resolution: the UI alone over black, colour premultiplied by alpha, so that
	// back buffer = ui.rgb + (1 - ui.a) * hudless.
	ID3D12Resource *ui = nullptr;
	uint32_t renderWidth = 0;
	uint32_t renderHeight = 0;
	uint32_t outputWidth = 0;
	uint32_t outputHeight = 0;
};

// What DLSS-RR is given each frame. Every resource is at the render resolution but the output, which is at the output
// resolution. Inputs must be in a shader-resource layout and the output in the unordered-access layout; Streamline then
// records no barriers of its own.
struct RayReconstructionInputs
{
	DLSSMode mode = DLSSMode::Off;
	DLSSRayReconstructionPreset preset = DLSSRayReconstructionPreset::Default;
	uint32_t renderWidth = 0;
	uint32_t renderHeight = 0;
	uint32_t outputWidth = 0;
	uint32_t outputHeight = 0;

	ID3D12Resource *colourInput = nullptr;
	ID3D12Resource *colourOutput = nullptr;
	ID3D12Resource *depth = nullptr; // hardware depth, reversed Z
	ID3D12Resource *motionVectors = nullptr; // render pixels, camera motion included
	ID3D12Resource *diffuseAlbedo = nullptr;
	ID3D12Resource *specularAlbedo = nullptr;
	ID3D12Resource *normalRoughness = nullptr; // roughness in .w
	ID3D12Resource *specularMotionVectors = nullptr;
	// Optional, R8: 1 where the history is out of date though the motion vectors match, so the current frame wins.
	ID3D12Resource *biasCurrentColourMask = nullptr;
	// Optional, R16F or R8_SNORM, -1..1, Preset F only: positive keeps less history, negative more.
	ID3D12Resource *responsivityMask = nullptr;

	// Render-pixel offset as applied to the projection.
	float jitterX = 0.0f;
	float jitterY = 0.0f;
	bool reset = false;

	// Row-major with row vectors (DirectXMath), unjittered.
	float worldToView[16] = {};
	float viewToClip[16] = {};
	float prevWorldToView[16] = {};
	float prevViewToClip[16] = {};

	float cameraPosition[3] = {};
	float cameraUp[3] = {};
	float cameraRight[3] = {};
	float cameraForward[3] = {};
	float nearZ = 0.0f;
	float farZ = 0.0f;
	float fovY = 0.0f; // radians
};

// NVIDIA Streamline with DLSS Ray Reconstruction, DLSS Frame Generation and Reflex.
// Not thread-safe. Every call comes from the render thread.
class Streamline
{
public:
	// slInit with DLSS-RR, DLSS-G and Reflex (PCL comes with them). On failure (old OS, missing DLLs) Streamline stays
	// off and the interposer passes D3D12 and DXGI straight through.
	static bool Initialise();
	// slSetD3DDevice: from the render context's device-created callback.
	static void AttachDevice(ID3D12Device *device);

	explicit Streamline(ID3D12Device *device);
	~Streamline();

	Streamline(const Streamline &) = delete;
	Streamline &operator=(const Streamline &) = delete;

	// False without an NVIDIA GPU and driver that run DLSS-RR, or if Initialise failed.
	bool IsRayReconstructionSupported() const
	{
		return rrSupported_;
	}

	// The render resolution DLSS-RR wants for an output size and mode. Touches no GPU state.
	bool GetOptimalRenderSize(DLSSMode mode,
							  uint32_t outputWidth,
							  uint32_t outputHeight,
							  uint32_t &renderWidth,
							  uint32_t &renderHeight) const;

	// False without a GPU, driver and OS that run DLSS-G (hardware-accelerated GPU scheduling included), or without
	// Reflex, which it needs.
	bool IsFrameGenerationSupported() const
	{
		return fgSupported_;
	}
	// False where the driver has no Reflex low-latency mode. The latency markers are sent regardless.
	bool IsReflexAvailable() const
	{
		return reflexAvailable_;
	}

	// The start of a frame, before its input is read: a new frame token, then the Reflex sleep (which paces the frame
	// to keep the GPU's queue short) and the simulation-start marker. Once per frame, before everything below.
	void BeginFrame();
	// PC Latency markers around this frame's stages, all with BeginFrame's token: the CPU work up to recording, the
	// recording and submission, and the present.
	void MarkSimulationEnd();
	void MarkRenderSubmitStart();
	void MarkRenderSubmitEnd();
	void MarkPresentStart();
	void MarkPresentEnd();

	// Sets the Reflex mode if it changed. Call at least once.
	void SetReflexMode(ReflexMode mode);

	// Turns DLSS-G on or off or changes its multiplier, if anything changed. Once per frame before the frame's tags
	// and present. It runs only with this frame's constants set (EvaluateRayReconstruction sets them) and its inputs
	// tagged.
	void SetFrameGeneration(const FrameGenerationOptions &options);
	// Whether the last SetFrameGeneration turned it on, so the frame must tag its inputs.
	bool IsFrameGenerationOn() const
	{
		return fgOn_;
	}
	// Records Streamline's copies of the inputs into the command list. Every resource must be in the copy-source
	// layout.
	bool TagFrameGeneration(const FrameGenerationInputs &inputs, ID3D12GraphicsCommandList *commandList);
	// Reads DLSS-G's state after a present.
	void UpdateFrameGenerationState();
	const FrameGenerationState &GetFrameGenerationState() const
	{
		return fgState_;
	}

	// Records DLSS-RR into the command list. Streamline creates the feature on first use and recreates it whenever the
	// mode, preset or sizes change. The caller restores the command list's descriptor heaps and pipeline afterwards.
	bool EvaluateRayReconstruction(const RayReconstructionInputs &inputs, ID3D12GraphicsCommandList *commandList);

	// Frees DLSS-RR's GPU memory. The GPU must be idle.
	void FreeRayReconstruction();

private:
	void SetMarker(uint32_t marker) const;

	bool rrSupported_ = false;
	bool fgSupported_ = false;
	bool reflexAvailable_ = false;
	bool pclLoaded_ = false;
	sl::FrameToken *frameToken_ = nullptr; // this frame's, owned by Streamline

	bool fgOn_ = false;
	bool fgOptionsSet_ = false;
	FrameGenerationOptions fgOptions_;
	FrameGenerationState fgState_;
	bool reflexModeSet_ = false;
	ReflexMode reflexMode_ = ReflexMode::Off;
};
