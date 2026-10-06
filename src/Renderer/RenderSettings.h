#pragma once

#include "DLSSCommon.h"
#include "PathTracerInterop.h"

#include <math.h>

struct SunSettings
{
	float azimuthDegrees = 30.0f; // rotation around +Y, 0 = towards +Z
	float elevationDegrees = 50.0f; // angle above the horizon
	float angularRadiusDegrees = 0.267865f; // physical sun (SUN_ANGULAR_RADIUS_DEFAULT radians)
	float3 colour = { 1.0f, 0.95f, 0.88f };
	float intensity = 5.0f; // irradiance scale

	// World-space direction pointing towards the sun (Y up).
	float3 GetDirection() const
	{
		const float azimuth = DegreesToRadians(azimuthDegrees);
		const float elevation = DegreesToRadians(elevationDegrees);
		return { cosf(elevation) * sinf(azimuth), sinf(elevation), cosf(elevation) * cosf(azimuth) };
	}
};

// The point and area lights in the scene's light buffer.
struct LightSettings
{
	bool enabled = true;
	float intensityScale = 1.0f; // multiplies every light, so the tree does not need rebuilding
	LightSamplingMode sampling = LIGHT_SAMPLING_TREE;
	// Light tree: skip the glossy importance for lobes at least this rough (perceptual roughness).
	bool diffuseOnlyRoughLobes = true;
	float diffuseOnlyRoughness = 0.7f;
	bool animate = true; // moves the test lights, the tree is rebuilt every frame either way
	float animationSpeed = 1.0f;
};

// Everything the UI can tweak; passes read it every frame.
struct RenderSettings
{
	SunSettings sun;
	LightSettings lights;

	float3 skyColour = { 0.45f, 0.65f, 1.0f };
	float skyIntensity = 1.0f;

	int maxBounces = 8;
	// DXR 1.2 shader execution reordering: switches to the path tracing pipeline that reorders
	// threads by material before shading each hit. Ignored where the device has no SER.
	bool shaderExecutionReordering = true;

	float exposureEV = 0.0f;
	ToneMappingOperator toneMapOperator = TONEMAP_ACES_FITTED; // SDR output only

	// HDR10 output: no SDR curve, scene 1.0 = paperWhiteNits and only highlights above the knee
	// are compressed towards peakNits.
	float paperWhiteNits = 203.0f; // BT.2408 reference white
	float peakNits = 1000.0f; // overwritten with the display's reported peak at startup
	float rolloffKnee = 0.6f; // fraction of peakNits

	DLSSMode dlssMode = DLSSMode::MaxQuality;
	DLSSRayReconstructionPreset rrPreset = DLSSRayReconstructionPreset::PresetF;

	float cameraSpeed = 5.0f; // world units per second
};
