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

// Everything the UI can tweak; passes read it every frame.
struct RenderSettings
{
	SunSettings sun;

	float3 skyColour = { 0.45f, 0.65f, 1.0f };
	float skyIntensity = 1.0f;

	int maxBounces = 8;

	float exposureEV = 0.0f;
	ToneMappingOperator toneMapOperator = TONEMAP_ACES_FITTED;

	DLSSMode dlssMode = DLSSMode::MaxQuality;
	DLSSRayReconstructionPreset rrPreset = DLSSRayReconstructionPreset::PresetF;

	float cameraSpeed = 5.0f; // world units per second
};
