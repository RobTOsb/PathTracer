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
	// Light tree: rank nodes by the SG lighting integrals of Tokuyoshi et al. (the SG_LIGHT_TREE permutation) rather
	// than the cheap bound over the cone to each node, oriented by the normal. More accurate, far more code.
	bool sgImportance = true;
	// Light tree, SG importance: nodes wholly above the horizon whose SG light is at least this sharp (squared
	// distance over spatial variance) use the cheap point light form of the importance.
	bool farFieldImportance = true;
	float farFieldSharpness = 16.0f;
	// Light tree: lights per leaf, picked by flux within it, 1 to LIGHT_TREE_MAX_LEAF_LIGHTS. More
	// makes the tree shallower at the cost of ignoring where the leaf's lights are relative to each other.
	int maxLeafLights = 4;
	// Light tree: the share of the tree's light samples that select by power instead (defensive sampling), which
	// bounds the noise of a light the tree's importance underrates. 0 for the tree alone.
	float treePowerMix = 0.0f;
	// Intel Sponza's emissive bunnies flash on clocks of their own, one like a broken light, so that the light
	// tree's lights change every frame. The main loop finds them (the checkbox shows only if it did) and holds
	// them still in the reference.
	bool flashBunnies = true;
	uint32_t bunnyCount = 0;
	bool flashClusters = true;
	uint32_t clusterCount = 0;
};

// The scene's glTF animations (RenderGPUScene::Animate), which play together and loop, except in the reference.
// While they play, the TLAS is rebuilt every frame (the light tree is rebuilt every frame regardless).
struct AnimationSettings
{
	bool play = false;
	float speed = 1.0f;
	// Into the animations: the main loop advances it while they play, and the UI can set it.
	float seconds = 0.0f;
	// While they play (not in the reference) and the view is the user's, it follows the scene's animated glTF
	// camera, its field of view too, and ignores the mouse and keys. Paused, the free camera carries on from there.
	bool sceneCamera = true;
};

// Geometric specular antialiasing of the surfaces the camera sees past mirrors and glass (PSR's chain)
struct SpecularAASettings
{
	bool enabled = true;
	float strength = 1.0f; // scales the kernel, 1 = Tokuyoshi and Kaplanyan's
	// The normal maps' variance as their mips average it away (Toksvig).
	bool normalMaps = true;
	// Glass (not pass-through walls) widens too, which can turn its PSR off where it widens past PSR's roughness.
	bool glass = true;
	// Curvature radius in pixels above which the turn of the normals is left to the jitter (half widened there,
	// ramping as radius^-4), so mirror balls stay sharp while thin tubes and ripples are prefiltered. 0 = widen all.
	float sharpRadiusPixels = 4.0f;
	float maxVariance = 0.18f; // cap on what is added to alpha^2, per axis
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
	// How far the sky visibility ray looks for occluders, metres; 0 for all the way to the sky. Most of what keeps an
	// enclosed space clear is close by, and a ray through all of a large scene costs as much as a light's shadow ray.
	float skyVisibilityDistance = 250.0f;

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
	// Rays whose sky the camera does not see, not even in a mirror, read the cloudy sky from a panorama rendered from
	// the camera each frame instead of marching the clouds (indirectSteps then only counts in the reference).
	bool indirectPanorama = true;
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

// Caustics from sun photons (PathTracer/Photons.slang): traced from the sun through smooth glass and mirrors into a
// world-space hash grid, read at the first diffuse surface of camera paths.
struct PhotonSettings
{
	bool enabled = true;
	int photonsPerFrame = 262144;
	// A photon's footprint is this many times the area it stands for across (two standard deviations of its Gaussian):
	// higher is smoother and blurrier.
	float footprintScale = 4.0f;
	// Perceptual roughness up to which a lobe passes photons (and counts as smooth on camera paths), also the
	// roughness factor up to which a material's primitives are aimed at as casters.
	float maxRoughness = 0.10f;
	int maxBounces = 8; // smooth bounces a photon may take
	// The photon map adapts each texel's share of the photons from feedback (Yang and Ouyang's AAPS): more where its
	// photons land wide on screen or the caustics flicker, fewer where they land small or out of view. Off: even.
	bool adaptive = true;
	float targetArea = 4.0f; // pixels a photon's area should cover on screen
	float varianceGain = 64.0f; // photons per texel a fully flickering caustic adds
	float temporalWeight = 0.5f; // weight of a texel's new share against its neighbourhood's last ones
	float minDensity = 0.25f; // photons per texel over a caster, at least, so that new caustics are found
	float maxDensity = 4096.0f; // photons per texel, at most
	// The UI bumps this to start the photon map over.
	uint resetCount = 0;
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
	AnimationSettings animation;
	SpecularAASettings specularAA;
	PhotonSettings photons;
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
	bool clampIndirect = false;
	float indirectClampValue = 5.0f;
	// The same for direct light: what the surface the camera sees scatters, by light sampling or a sample of a
	// lobe that is not mirror-like. Light the camera sees, also as a mirror-like surface's image, is not clamped.
	bool clampDirect = false;
	float directClampValue = 1.0f;
	// RTXPT's firefly filter
	bool fireflyFilter = true;
	float fireflyFilterThreshold = 0.050f;
	// Camera-visible opaque surfaces whose reflection is at most reflectionMinProbabilityMaxRoughness (perceptual)
	// rough pick that lobe with at least this probability, the other lobes' shares scaled down: unbiased, it moves
	// noise from small bright reflections (emitters in polished floors) to the diffuse lobe. 0 = off, at most 0.95.
	float reflectionMinProbability = 0.5f;
	float reflectionMinProbabilityMaxRoughness = 0.3f;
	// DXR 1.2 shader execution reordering: switches to the path tracing pipeline that reorders
	// threads by material before shading each hit. Ignored where the device has no SER.
	bool shaderExecutionReordering = true;
	// Shadow rays as inline ray queries (INLINE_SHADOWS) instead of TraceRay with the shadow hit group.
	bool inlineShadowRays = true;
	bool opacityMicromaps = true;
	int opacityMicromap2StateBounce = 1;
	// Performance approximations past this many diffuse bounces, 0 = off (FrameConstants): rays skip blended
	// instances and alpha tests; hits load materials without normal maps, solid subsurface as Lambertian.
	int opaqueAlphaDiffuseBounces = 1;
	int reducedMaterialDiffuseBounces = 1;
	// Static primitives placed together (the same world matrix, the parts of one plant) as one TLAS instance
	// of a multi-geometry BLAS (RenderGPUScene::SetMergeInstances). Changing it rebuilds every BLAS and TLAS.
	bool mergeInstances = true;
	bool specularMaterials = true;
	bool transmissiveMaterials = true;
	// Blended (ALPHA_MODE_BLEND) surfaces block shadow rays, by chance at their alpha. Off suits decals, which
	// would otherwise shadow the wall they lie on.
	bool blendedShadows = false;
	// DLSS-RR guides of blended surfaces: the layer mixed with what is behind it by opacity, deterministically,
	// instead of whichever layer the camera ray took by chance (FrameConstants::blendGuides).
	bool blendGuides = true;
	int glassReflectionSplits = 2;
	// Solid subsurface materials: the orders of scattering inside the object that are rendered, light
	// straight through and one scattering event path traced, the rest from a diffusion profile. All off
	// falls back to OpenPBR's Lambertian stand-in. The radius scale multiplies every material's radius.
	bool subsurfaceZeroScattering = true;
	bool subsurfaceSingleScattering = true;
	bool subsurfaceMultipleScattering = true;
	float subsurfaceRadiusScale = 1.0f;
	// glTF has no diffuse roughness: the base's EON diffuse takes this fraction of the specular roughness, 0 = Lambert.
	float diffuseRoughnessScale = 0.33f;

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
	float psrMaxRoughness = 0.01f;
	// Curved mirrors and glass keep PSR too, with motion vectors traced through them by ray differentials,
	// unless they shrink the image of the surface they show by more than psrMaxMinification (linear).
	bool psrCurvedSurfaces = true;
	float psrMaxMinification = 64.0f;
	// A chain that leaves into the sky puts the sky in the guides' place (FrameConstants::psrSky).
	bool psrSky = true;
	// Pixels that followed a PSR chain take their guides from this many chains walked from fixed points of the pixel,
	// 1 = the centre alone (FrameConstants::psrGuideSamples). 0 = the path's jittered chain.
	int psrGuideSamples = 1;
	// Past the camera's first surface, a chain through smooth glass always takes the lobe it follows, so its guides
	// never hold last frame's; the other lobe's light (faint internal reflections) is left out.
	bool psrStrongerLobe = true;
	// The specular hit distance of the guides' surface, which places its reflection for the specular motion
	// vectors, from a ray RayGen traces along the lobe's dominant direction rather than from the path's sampled
	// reflection, so it is the same every frame for the same view. PSR's virtual surfaces and panes of glass keep
	// the path's, and the hold below.
	bool specularHitDistanceRay = true;
	// Surfaces at most this rough keep last frame's specular hit distance in frames whose path took
	// the diffuse lobe, so the guide does not flicker between a distance and none.
	bool holdSpecularHitDistance = true;
	float holdHitDistanceMaxRoughness = 0.25f;
	// Only surfaces at most this rough get specular motion vectors that follow their reflection; rougher ones get
	// the surface's own, since one hit distance cannot place a glossy blur of things at many depths (a light's
	// streak on a shiny floor) and the field it gives jumps across the highlight. Bevy Solari's cut-off
	// (GGX alpha 0.0625). 1 = always the reflection.
	float specularMotionMaxRoughness = 0.25f;
	// While the animated instances move, the surface the guides describe also traces its light sample's shadow
	// ray in last frame's TLAS, and where the light was blocked differently DLSS-RR is told to trust the current
	// frame over its history (bias current colour mask), so moving shadows do not ghost.
	bool shadowChangeMask = true;
	// Preset F only: DLSS-RR's responsivity mask on the smooth reflectors the camera sees, so it keeps less history
	// for their reflections (an emitter's image trailing across a polished floor), this much at a mirror and falling
	// to none at reflectionResponsivityMaxRoughness (perceptual). 0 = off.
	float reflectionResponsivity = 0.5f;
	float reflectionResponsivityMaxRoughness = 0.25f;
	// DLSS-RR albedo/normal/roughness guides averaged over this many extra sub-pixel camera rays, 0 = off
	// (FrameConstants::guideSamples, at most 8).
	int guideSamples = 2;
	// Set by the main loop for a frame whose view does not follow on from the last (another view chosen, the
	// scene camera starting, stopping or looping), so DLSS drops its history.
	bool cameraCut = false;
	// Debug: one guide in place of the image (GuideViewPass), or with guideViewChange how much it changed since
	// last frame along the motion vectors, as heat scaled by guideViewGain. guideViewHeld tints the pixels that kept
	// last frame's guides because their PSR chain broke by chance.
	GuideView guideView = GUIDE_VIEW_OFF;
	bool guideViewChange = false;
	bool guideViewHeld = false;
	float guideViewGain = 1.0f;

	// The view: 0 = the user's free camera, k = the scene's k-th still glTF camera
	// (RenderGPUScene::GetStillCameraPose(k - 1)), its field of view too, ignoring the mouse and keys. Back on 0,
	// the free camera is where it was left.
	uint32_t view = 0;
	float cameraSpeed = 5.0f; // world units per second
	// Set by the Camera panel, carried out (and cleared) by the main loop: the camera bookmark beside the
	// scene, so captures for an A/B comparison can be taken from exactly the same view. Restoring goes back to
	// the user's view.
	enum class CameraBookmarkRequest
	{
		None,
		Save,
		Restore
	};
	CameraBookmarkRequest cameraBookmarkRequest = CameraBookmarkRequest::None;
	bool cameraBookmarkExists = false; // the main loop keeps it current, for the Restore button
};
