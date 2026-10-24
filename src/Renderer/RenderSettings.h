#pragma once

#include "DLSSCommon.h"
#include "PathTracerInterop.h"

#include <math.h>

struct SunSettings
{
	float azimuthDegrees = 30.0f; // rotation around +Y, 0 = towards +Z
	float elevationDegrees = 50.0f; // angle above the horizon
	float angularRadiusDegrees = 0.267865f / 2.0f; // physical sun (SUN_ANGULAR_RADIUS_DEFAULT radians)
	float3 colour = { 1.0f, 0.95f, 0.88f };
	float intensity = 20.0f; // irradiance scale

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
	// Light tree: path vertices before this bounce (0 = primary hit) walk the tree, later ones select by
	// flux, which costs one lookup instead of a walk per light sample and per emitter hit's MIS.
	int treeBounces = 1;
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
	bool orientationSplits = false;
	// Animates the scene's emitters (AnimateLights in main.cpp): the quad spins, which rebuilds the TLAS,
	// the sphere pulses and the bulbs flash, and the dynamic emitters' light tree is rebuilt every frame.
	bool animate = true;
};

// Geometric specular antialiasing of the opaque surfaces the camera sees: glossy lobes widen by how much
// the normals turn across the pixel, so highlights smaller than a pixel stop flickering with the jitter.
struct SpecularAASettings
{
	bool enabled = true;
	float strength = 1.0f; // scales the kernel, 1 = Tokuyoshi and Kaplanyan's
	// The normal maps' variance as their mips average it away (Toksvig).
	bool normalMaps = true;
	float maxVariance = 0.18f; // cap on what is added to alpha^2
};

// Sky and aerial perspective of Hillaire 2020 (SkyAtmospherePass), lit by the sun. While it is on, the
// sun's colour and intensity are taken at the top of the atmosphere and the constant sky is unused.
struct AtmosphereSettings
{
	bool enabled = true;
	float altitudeKm = 0.0f; // of the scene's y = 0 above the ground
	// The scene's air is this many times denser than real air at its altitude, so that its haze and
	// light shafts show at the scale of a building; 1 is physical, and the sky is unaffected. Molecules
	// (Rayleigh scattering and ozone) scatter almost evenly in all directions, a blue fog, while aerosols
	// (Mie scattering) are the dust and haze whose forward scattering shows light shafts.
	float sceneAirScale = 1.0f; // molecules
	float sceneHazeScale = 1.0f; // aerosols
	// Path segments (1 = camera rays only) that gather the sunlight the air scatters into them; the
	// air only dims the later ones.
	int inScatterSegments = 1;
	// In-scattering traces a shadow ray towards the sun, so geometry casts light shafts.
	bool volumetricShadows = true;
	// The multiple-scattered light of the LUT is that of an open sky. In-scattering traces a ray in a random
	// direction and keeps it only if that reaches the sky, so enclosed and shadowed spaces are not lit as if
	// they stood in the open.
	bool skyVisibility = true;

	// Scales of Earth's atmosphere.
	float rayleighScale = 1.0f;
	float mieScale = 1.0f; // aerosols
	float miePhaseG = 0.8f;
	float ozoneScale = 1.0f;
	float3 groundAlbedo = { 0.3f, 0.3f, 0.3f };
	float multipleScatteringFactor = 1.0f;
};

// A layer of volumetric clouds in the atmosphere (Clouds.slang), atmosphere only: rays leaving the scene see the
// sky through them and the sunlight reaching the scene is shadowed by them.
struct CloudSettings
{
	bool enabled = true;
	float bottomKm = 1.5f; // cloud base above the ground
	float topKm = 8.0f; // top of the layer, which only cumulus reach
	float coverage = 0.3f; // added to the weather map's, from patchy at 0 to overcast at 1
	float density = 1.0f; // extinction per km where the cloud is densest
	// The wind carries the clouds towards windDirectionDegrees (around +Y, 0 = towards +Z, as the sun's azimuth).
	// The reference holds them still.
	float windSpeed = 10.0f; // m/s
	float windDirectionDegrees = 60.0f;
	// Most march steps along rays that leave the scene: those the camera sees through, and the rest.
	int viewSteps = 16;
	int indirectSteps = 16;
	// Kept by the main loop: how far the wind has carried the clouds in x and z.
	float2 windOffsetKm = { 0.0f, 0.0f };
};

constexpr int kMaxSamplesPerPixel = 8;
constexpr int kMaxBounces = 8; // of each kind, diffuse and specular
constexpr int kMaxBloomLevels = 8; // half resolution down to 1/256

enum class RenderMode
{
	RealTime, // one path per pixel per frame, denoised and upscaled by DLSS-RR when dlssMode is on
	Reference, // paths accumulated at the window's resolution without DLSS
};

struct ReferenceSettings
{
	int samples = 1024; // paths per pixel to stop at, 0 = never stop
	int samplesPerFrame = 1; // paths per pixel each dispatch traces in turn
	// The firefly clamp and specular antialiasing, biased approximations for the real-time image. Off makes the
	// reference converge to the unbiased image; on matches what DLSS-RR is given.
	bool approximations = false;
	// The UI bumps this to start the accumulation over.
	uint restartCount = 0;
};

// Everything the UI can tweak; passes read it every frame.
struct RenderSettings
{
	RenderMode renderMode = RenderMode::RealTime;
	ReferenceSettings reference;

	SunSettings sun;
	LightSettings lights;
	SpecularAASettings specularAA;
	AtmosphereSettings atmosphere;
	CloudSettings clouds;

	float3 skyColour = { 0.45f, 0.65f, 1.0f };
	float skyIntensity = 1.0f;

	// Indirect bounces of each kind a path may take, 0 to kMaxBounces each: diffuse (diffuse, diffuse
	// transmission, subsurface and fuzz lobes) and specular (specular reflection and transmission, coat,
	// pass-throughs).
	int maxDiffuseBounces = 2;
	int maxSpecularBounces = 8;
	// Real-time: paths per pixel each frame, averaged, 1 to kMaxSamplesPerPixel. Under DLSS-RR they all start
	// through the frame's jitter.
	int samplesPerPixel = 1;
	// Firefly clamp: each contribution of indirect light (more than one scattering away from the surface the
	// camera sees, PSR's mirrors and glass and pass-throughs not counting) is scaled down to at most
	// indirectClampValue in its largest channel. Biased darker.
	bool clampIndirect = true;
	float indirectClampValue = 5.0f;
	// DXR 1.2 shader execution reordering: switches to the path tracing pipeline that reorders
	// threads by material before shading each hit. Ignored where the device has no SER.
	bool shaderExecutionReordering = true;
	// Shadow rays as inline ray queries (INLINE_SHADOWS) instead of TraceRay with the shadow hit group.
	bool inlineShadowRays = true;
	bool opacityMicromaps = true;
	int opacityMicromap2StateBounce = 1;
	bool specularMaterials = true;
	bool transmissiveMaterials = true;
	int glassReflectionSplits = 2;
	// Solid subsurface materials: the orders of scattering inside the object that are rendered, light
	// straight through and one scattering event path traced, the rest from a diffusion profile. All off
	// falls back to OpenPBR's Lambertian stand-in. The radius scale multiplies every material's radius.
	bool subsurfaceZeroScattering = true;
	bool subsurfaceSingleScattering = true;
	bool subsurfaceMultipleScattering = true;
	float subsurfaceRadiusScale = 1.0f;

	bool rayConeTextureLod = true;
	float textureLodBias = 0.0f;
	float rayConeBsdfSpread = 1.0f;
	int anisotropicSegments = 1;

	float exposureEV = 0.0f; // the exposure, or with autoExposure its compensation
	// Exposure from the luminance histogram of each frame (AutoExposurePass), adapting over time, at most
	// maxAutoExposureEV with the compensation included.
	bool autoExposure = true;
	float maxAutoExposureEV = 4.0f;
	ToneMappingOperator toneMapOperator = TONEMAP_NEUTWO;
	float whiteClip = 100.0f; // scene value the Neutwo curve maps to its peak (1.0 = paper white)

	// HDR10 output: scene 1.0 = paperWhiteNits.
	float paperWhiteNits = 203.0f; // BT.2408 reference white
	float peakNits = 1000.0f; // overwritten with the display's reported peak at startup
	bool hdrGammaCorrection = true; // match the 2.2 gamma SDR displays decode with

	// Bloom, blended into the HDR image after upscaling: the mean of bloomLevels ever blurrier half-resolution
	// levels replaces bloomStrength of the image.
	bool bloom = true;
	float bloomStrength = 0.1f;
	int bloomLevels = 6; // 1 to kMaxBloomLevels, each doubles the radius
	bool bloomSun = false;

	// Film grain, added after tone mapping in display-encoded space.
	bool filmGrain = true;
	float filmGrainStrength = 0.008f; // standard deviation in sRGB-encoded units at mid-grey
	float filmGrainSize = 1.0f; // output pixels
	float filmGrainColour = 1.0f; // 0 = monochrome, 1 = independent per channel
	float filmGrainResponse = 1.0f; // mid-tone emphasis, 0 = the same grain at every level
	float filmGrainFps = 60.0f; // how often the grain pattern changes, 0 = static

	DLSSMode dlssMode = DLSSMode::MaxQuality;
	DLSSRayReconstructionPreset rrPreset = DLSSRayReconstructionPreset::PresetF;
	// Primary surface replacement: the guides of a mirror or pane of glass (at most this rough, and
	// reflecting or transmitting mostly through its specular lobe) describe the surface it reflects or
	// refracts, so RR denoises that surface's texture detail as it would on screen instead of blurring
	// it as the mirror's or glass's lighting.
	bool primarySurfaceReplacement = true;
	float psrMaxRoughness = 0.1f;
	// Curved mirrors and glass keep PSR too, with motion vectors traced through them by ray differentials,
	// unless they shrink the image of the surface they show by more than psrMaxMinification (linear).
	bool psrCurvedSurfaces = true;
	float psrMaxMinification = 64.0f;
	// Past the camera's first surface, a chain through smooth glass always takes the lobe it follows, so its guides
	// never hold last frame's; the other lobe's light (faint internal reflections) is left out.
	bool psrStrongerLobe = true;
	// Surfaces at most this rough keep last frame's specular hit distance in frames whose path took
	// the diffuse lobe, so the guide does not flicker between a distance and none.
	bool holdSpecularHitDistance = true;
	float holdHitDistanceMaxRoughness = 0.25f;
	// Debug: one guide in place of the image (GuideViewPass), or with guideViewChange how much it changed since
	// last frame along the motion vectors, as heat scaled by guideViewGain. guideViewHeld tints the pixels that kept
	// last frame's guides because their PSR chain broke by chance.
	GuideView guideView = GUIDE_VIEW_OFF;
	bool guideViewChange = false;
	bool guideViewHeld = false;
	float guideViewGain = 1.0f;

	float cameraSpeed = 5.0f; // world units per second
	// Set by the Camera panel, carried out (and cleared) by the main loop: the camera bookmark beside the
	// scene, so captures for an A/B comparison can be taken from exactly the same view.
	enum class CameraBookmarkRequest
	{
		None,
		Save,
		Restore
	};
	CameraBookmarkRequest cameraBookmarkRequest = CameraBookmarkRequest::None;
	bool cameraBookmarkExists = false; // the main loop keeps it current, for the Restore button
};
