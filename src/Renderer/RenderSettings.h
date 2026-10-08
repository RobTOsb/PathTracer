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

// The emissive triangles in the scene's light buffer.
struct LightSettings
{
	// Next event estimation of the light buffer. Without it emissive surfaces are only found by
	// BSDF sampling.
	bool enabled = true;
	LightSamplingMode sampling = LIGHT_SAMPLING_TREE;
	// Light tree: skip the glossy importance for lobes at least this rough (perceptual roughness).
	bool diffuseOnlyRoughLobes = true;
	float diffuseOnlyRoughness = 0.7f;
	// Light tree: vertices from this bounce on (0 = primary hit) skip the glossy importance.
	int glossyImportanceBounces = 1;
	// Light tree: nodes wholly above the horizon whose SG light is at least this sharp (squared
	// distance over spatial variance) use the cheap point light form of the importance.
	bool farFieldImportance = true;
	float farFieldSharpness = 16.0f;
	// Light tree: lights per leaf, picked by flux within it, 1 to LIGHT_TREE_MAX_LEAF_LIGHTS. More
	// makes the tree shallower at the cost of ignoring where the leaf's lights are relative to each other.
	int maxLeafLights = 4;
	// Light tree build: the split cost also weighs how far each side's emitter normals spread.
	bool orientationSplits = true;
};

// Sky and aerial perspective of Hillaire 2020 (SkyAtmospherePass), lit by the sun. While it is on, the
// sun's colour and intensity are taken at the top of the atmosphere and the constant sky is unused.
struct AtmosphereSettings
{
	bool enabled = true;
	float altitudeKm = 0.0f; // of the scene's y = 0 above the ground
	// The scene's air is this many times denser than real air at its altitude, so that its haze and
	// light shafts show at the scale of a building. 1 is physical.
	float aerialPerspectiveScale = 100.0f;
	// Path segments (1 = camera rays only) that gather the sunlight the air scatters into them; the
	// air only dims the later ones.
	int inScatterSegments = 1;
	// In-scattering traces a shadow ray towards the sun, so geometry casts light shafts.
	bool volumetricShadows = true;

	// Scales of Earth's atmosphere.
	float rayleighScale = 1.0f;
	float mieScale = 1.0f; // aerosols
	float miePhaseG = 0.8f;
	float ozoneScale = 1.0f;
	float3 groundAlbedo = { 0.3f, 0.3f, 0.3f };
	float multipleScatteringFactor = 1.0f;
};

// Everything the UI can tweak; passes read it every frame.
struct RenderSettings
{
	SunSettings sun;
	LightSettings lights;
	AtmosphereSettings atmosphere;

	float3 skyColour = { 0.45f, 0.65f, 1.0f };
	float skyIntensity = 1.0f;

	int maxBounces = 4;
	// DXR 1.2 shader execution reordering: switches to the path tracing pipeline that reorders
	// threads by material before shading each hit. Ignored where the device has no SER.
	bool shaderExecutionReordering = true;
	
	bool rayConeTextureLod = true;
	float textureLodBias = 0.0f;
	float rayConeBsdfSpread = 1.0f;

	float exposureEV = 0.0f;
	ToneMappingOperator toneMapOperator = TONEMAP_ACES_FITTED; // SDR output only

	// HDR10 output: no SDR curve, scene 1.0 = paperWhiteNits and only highlights above the knee
	// are compressed towards peakNits.
	float paperWhiteNits = 203.0f; // BT.2408 reference white
	float peakNits = 1000.0f; // overwritten with the display's reported peak at startup
	float rolloffKnee = 0.8f; // fraction of peakNits

	DLSSMode dlssMode = DLSSMode::MaxQuality;
	DLSSRayReconstructionPreset rrPreset = DLSSRayReconstructionPreset::PresetF;

	float cameraSpeed = 5.0f; // world units per second
};
