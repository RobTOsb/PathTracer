#ifndef PATH_TRACER_INTEROP_H
#define PATH_TRACER_INTEROP_H

#include "ShaderInterop.h"

// CloudParameters. PathTracer/Resources.slang imports the SkyAtmosphere module, which already declares it, and
// defines SKY_ATMOSPHERE_IMPORTED so that its types are not declared twice.
#ifndef SKY_ATMOSPHERE_IMPORTED
#include "SkyAtmosphereInterop.h"
#endif

// Per-frame constants for the path tracer (cbuffer packing rules apply on the GPU side).
struct FrameConstants
{
	RowMajor float4x4 invView;
	RowMajor float4x4 invProj;
	// Unjittered view-projections for the DLSS-RR depth and motion vector guides.
	RowMajor float4x4 viewProj;
	RowMajor float4x4 prevViewProj;

	// Directional sun: a uniform disc of angular radius r around sunDirection. Everything derived
	// from the light settings is computed once on the CPU rather than per sample.
	float3 sunDirection; // normalised, towards the sun
	float sunOneMinusCosMax; // 1 - cos(r), 0 for a delta sun
	float3 sunRadiance; // radiance of the disc (irradiance / solid angle), or the irradiance for a delta sun
	float sunPdf; // solid angle PDF of uniform cone sampling, 1 / solid angle

	float3 skyRadiance; // constant sky colour * intensity, returned for every missed ray
	uint frameIndex;

	// Bounce budgets, each counting the indirect bounces of its kind: diffuse (diffuse, diffuse transmission,
	// subsurface and fuzz lobes) and specular (specular reflection and transmission, coat, pass-throughs).
	uint maxDiffuseBounces;
	uint sunIsDelta; // zero angular radius, cannot be hit by BSDF sampling
	float2 jitter;

	// Next event estimation picks the sun with sunSelectProbability, otherwise one of the
	// lightCount lights in the light buffer. lightCount is 0 while the lights are switched off.
	uint lightCount;
	uint lightSampling; // LightSamplingMode
	uint materialOverrides; // MaterialOverrideFlags
	float sunSelectProbability;

	float lightTreeDiffuseOnlyAlpha;
	// Shader execution reordering (SER pipeline only): bits of the material index used as the
	// coherence hint, enough for every material row and at most 16.
	uint serCoherenceHintBits;
	uint lightTreeGlossyBounces; // path vertices from this bounce on use the diffuse light tree importance only
	float lightTreeFarFieldSharpness; // SG sharpness from which tree nodes count as point lights, FLT_MAX = off

	// Ray cone texture filtering (Akenine-Moller et al. 2021). Camera rays start as cones of
	// rayConePixelSpread radians, and each BSDF sample widens the cone by rayConeBsdfSpread times
	// the angle of a cone covering the solid angle (1 / pdf) the sample stands for.
	float rayConePixelSpread;
	float rayConeBsdfSpread;
	float textureLodScale; // multiplies the cone footprint (2^mip bias), 0 samples the top mip everywhere
	uint anisotropicSegments;

	// Sky atmosphere (ATMOSPHERE permutation only, see SkyAtmospherePass). The scene sits at one
	// altitude of the atmosphere, whose air fills it homogeneously. Coefficients are per scene unit,
	// scaled up by the scene air (Rayleigh, ozone) and haze (Mie) scales; the sun and the sky-view LUT are not.
	float3 atmosphereRayleighScattering;
	float atmosphereViewHeight; // km from the planet centre, where the sky-view LUT was rendered
	float3 atmosphereMieScattering;
	float atmosphereBottomRadius; // km
	float3 atmosphereExtinction;
	float atmosphereMiePhaseG;
	float atmosphereMissDistance; // air in front of the sky along rays that leave the scene, scene units
	uint atmosphereInScatterSegments; // path segments (0 = camera ray) that gather in-scattered sunlight
	uint atmosphereVolumetricShadows; // in-scattering traces a shadow ray towards the sun
	// SubsurfaceComponents traced for solid subsurface materials; 0 keeps OpenPBR's Lambertian stand-in.
	uint subsurfaceComponents;

	// DLSS-RR guides (DLSS_RR permutation only). Mirrors and glass at most psrMaxRoughness (perceptual) rough
	// are described by the surface they reflect or refract, primary surface replacement (PSR permutation only).
	// Surfaces at most specularHitDistanceHoldRoughness rough keep last frame's specular hit
	// distance when their path took another lobe, instead of reporting none; negative = off.
	float psrMaxRoughness;
	float specularHitDistanceHoldRoughness;
	uint glassReflectionSplits;
	float subsurfaceRadiusScale; // multiplies every material's subsurface radius

	// Opacity micromaps (OPACITY_MICROMAPS permutation only): path vertices from this bounce on (0 = the
	// camera ray) trace with RAY_FLAG_FORCE_OMM_2_STATE, taking the micromaps' unknown micro-triangles as
	// opaque or transparent instead of running the alpha test. UINT_MAX = never.
	uint opacityMicromap2StateBounce;
	// LIGHT_SAMPLING_TREE: path vertices before this bounce (0 = the primary hit) select lights with the
	// tree, later ones and those that transmit by power, which skips the tree walks there.
	uint lightTreeBounces;
	// Firefly clamp: indirect light (reaching the surface the camera sees after more than one scattering, see
	// ClampFirefly) is scaled down to at most this in its largest channel per contribution. FLT_MAX = off.
	float indirectClamp;
	// ATMOSPHERE: the air's multiple-scattered light counts only where a ray in a random direction from the
	// in-scattering point reaches the sky.
	uint atmosphereSkyVisibility;

	// Last frame's camera position, for the motion vectors of what curved mirrors and glass show.
	float3 prevCameraPosition;
	// Chains of curved mirrors and glass keep PSR, with motion vectors traced through them, unless they
	// shrink the image of the surface they end on by more than this (linear); negative = off, which
	// leaves only the chains that barely bend the virtual image.
	float psrMaxMinification;

	// Geometric specular antialiasing of the opaque surfaces the camera sees (Tokuyoshi and Kaplanyan 2019):
	// GGX alpha^2 grows by specularAAVariance times the squared turn of the vertex normals across the pixel,
	// plus specularAANormalMapScale times the normal maps' variance from their mips (Toksvig), together at
	// most specularAAMaxVariance. Zero turns either off.
	float specularAAVariance;
	float specularAANormalMapScale;
	float specularAAMaxVariance;
	// Reference (REFERENCE permutation only): the paths per pixel already summed in the accumulation target.
	uint referenceSampleCount;

	// Paths per pixel this dispatch traces: averaged into the output, or under REFERENCE added to the sum.
	uint samplesPerFrame;
	uint maxSpecularBounces; // see maxDiffuseBounces
	// The light buffer holds the static emitters first, then from staticLightCount on the dynamic ones.
	// Power selection picks the static part with staticLightPowerShare, its share of the flux, then a
	// light within the part from its own alias table.
	uint staticLightCount;
	float staticLightPowerShare;

	// ATMOSPHERE: the cloud layer seen past the scene's geometry and shadowing its sunlight (see Clouds.slang).
	CloudParameters clouds;

	// PSR: past the camera's first surface, a chain through smooth solid glass takes the lobe it follows rather than
	// a random one (PsrStrongerLobe), leaving out the other's light. 0 = off.
	uint psrStrongerLobe;
	// Firefly clamp of direct light: as indirectClamp, for light the surface the camera sees scatters (see
	// ClampFirefly). Light the camera sees, also in a mirror-like surface, is never clamped. FLT_MAX = off.
	float directClamp;
	// DLSS-RR: at most this many splits of the camera ray at alpha-blended surfaces (see TracePath), which share
	// their count with glassReflectionSplits. Past them a blended surface is there or not by chance.
	uint blendLayerSplits;
	uint pad1;
};

// Path-traced subsurface scattering (FrameConstants::subsurfaceComponents): which orders of
// scattering inside a solid subsurface object are rendered.
UnscopedEnum
enum SubsurfaceComponents
{
	SUBSURFACE_ZERO_SCATTERING = 1, // light straight through the medium, path traced
	SUBSURFACE_SINGLE_SCATTERING = 2, // one scattering event, path traced
	SUBSURFACE_MULTIPLE_SCATTERING = 4, // the rest, from the multiple-scattering diffusion profile
};

// Debug switches applied to every material as it is loaded (FrameConstants::materialOverrides).
UnscopedEnum
enum MaterialOverrideFlags
{
	// No specular reflection: metals become diffuse in their base colour and dielectrics lose their
	// reflection lobe. The interface keeps its IOR, so glass still refracts.
	MATERIAL_OVERRIDE_NO_SPECULAR = 1,
	// No transmission: transmissive materials become opaque, and shadow rays stop passing them.
	MATERIAL_OVERRIDE_NO_TRANSMISSION = 2,
};

// How next event estimation picks one light from the light buffer.
UnscopedEnum
enum LightSamplingMode
{
	LIGHT_SAMPLING_TREE = 0, // SG light tree (Tokuyoshi et al. 2024), power selection past lightTreeBounces
	LIGHT_SAMPLING_UNIFORM = 1,
	LIGHT_SAMPLING_POWER = 2, // in proportion to flux, wherever the shading point is
};

// glTF alphaMode, matches Scene::AlphaMode.
UnscopedEnum
enum GpuAlphaMode
{
	ALPHA_MODE_OPAQUE = 0,
	ALPHA_MODE_MASK = 1,
	ALPHA_MODE_BLEND = 2,
};

// GpuMaterial::transmissionFlags.
UnscopedEnum
enum GpuTransmissionFlags
{
	// Material::IsThinWalled (EXT_materials_thin_wall): both sides are outside, light crosses without
	// refracting and without entering a medium.
	TRANSMISSION_THIN_WALLED = 1,
	// Thin-walled, transmissive and smooth (roughness factor 0, so every texel is too), which makes
	// its transmission a delta lobe straight through. Shadow rays pass such surfaces in
	// ShadowAnyHit, so their geometry is non-opaque in the BLAS.
	TRANSMISSION_PASS_THROUGH = 2,
};

// A glTF material as the CPU builds it and the shaders read it, unpacked from its GpuMaterial row.
// Texture fields are bindless descriptor-heap indices, INVALIDDESCRIPTORHANDLE when the material has
// no such map.
struct MaterialData
{
	float4 baseColour; // base colour factor, .a = alpha
	float roughness;
	float metallic;
	float normalScale;
	float alphaCutoff; // ALPHA_MODE_MASK only
	uint albedoTexture; // sRGB, sampled as linear, .a = alpha
	uint roughMetalTexture; // glTF packing: G = roughness, B = metallic
	uint normalTexture; // tangent-space, RGB = XYZ * 0.5 + 0.5
	uint alphaMode; // GpuAlphaMode
	float transmission; // KHR_materials_transmission factor
	float ior; // KHR_materials_ior
	uint transmissionFlags; // GpuTransmissionFlags
	uint transmissionTexture; // R multiplies transmission
	float3 emissive; // emissiveFactor * KHR_materials_emissive_strength, linear, front faces only
	uint emissiveTexture; // sRGB, multiplies emissive
	// Not thin-walled only: white light comes out as attenuationColour after attenuationDistance in
	// the medium. FLT_MAX or more: no absorption (always, while KHR_materials_volume is not loaded).
	float3 attenuationColour;
	float attenuationDistance;
	// OpenPBR subsurface from EXT_materials_subsurface, coloured by the base colour. Thin-walled: split
	// between reflection and diffuse transmission to the other side by subsurfaceScatterAnisotropy.
	// Solid: a scattering medium under the surface whose multiple-scattering albedo is the colour.
	float3 subsurfaceColour;
	float subsurfaceWeight;
	uint subsurfaceWeightTexture; // channel subsurfaceWeightChannel multiplies subsurfaceWeight
	uint subsurfaceColourTexture; // sRGB, RGB multiplies subsurfaceColour
	// KHR_materials_anisotropy: strength * (cos, sin) of the rotation from the tangent, 0 when
	// isotropic. The texture's RG direction (* 2 - 1) is rotated by it, B multiplies the strength.
	float2 anisotropy;
	uint anisotropyTexture;
	// Solid subsurface only: the medium's mean free path per channel in scene units (radius * radius
	// scale), 0 without one.
	float3 subsurfaceRadius;
	// Solid: the medium's Henyey-Greenstein g. Thin-walled: -1 all reflected, 1 all transmitted.
	float subsurfaceScatterAnisotropy;
	// Solid subsurface only, linear: RGB multiplies subsurfaceRadius, and so does the scale texture's
	// channel subsurfaceScaleChannel.
	uint subsurfaceRadiusTexture;
	uint subsurfaceScaleTexture;
	// The channel, 0 to 3 for R to A, of the weight and scale textures that holds the scalar.
	uint subsurfaceWeightChannel;
	uint subsurfaceScaleChannel;
	// KHR_materials_sheen, which the shader turns into OpenPBR fuzz: linear colour (black for none) and
	// roughness factors.
	float3 sheenColour;
	float sheenRoughness;
	uint sheenColourTexture; // sRGB, RGB multiplies sheenColour
	uint sheenRoughnessTexture; // A multiplies sheenRoughness, the colour texture when it is the same image
	// KHR_materials_clearcoat, which the shader turns into a white OpenPBR coat of IOR 1.5: weight (0 for
	// none) and roughness factors in [0, 1], and the coat normal map's scale.
	float coatWeight;
	float coatRoughness;
	float coatNormalScale;
	uint coatWeightTexture; // R multiplies coatWeight
	uint coatRoughnessTexture; // G multiplies coatRoughness
	uint coatNormalTexture; // tangent space like normalTexture; without one the coat follows the vertex normal
	// KHR_materials_specular: the dielectric F0 scaled by the weight and tinted by the colour, which may
	// exceed 1. Metals are left alone.
	float specularWeight;
	float3 specularColour;
	uint specularTexture; // A multiplies specularWeight
	uint specularColourTexture; // sRGB, RGB multiplies specularColour
};

// GpuMaterial::emissiveFlags bits above the half emissive blue.
#define GPU_MATERIAL_ALPHA_MODE_SHIFT 16 // GpuAlphaMode, 2 bits
#define GPU_MATERIAL_TRANSMISSION_FLAGS_SHIFT 18 // GpuTransmissionFlags, 2 bits
#define GPU_MATERIAL_WEIGHT_CHANNEL_SHIFT 20 // subsurfaceWeightChannel, 2 bits
#define GPU_MATERIAL_SCALE_CHANNEL_SHIFT 22 // subsurfaceScaleChannel, 2 bits
// No transmission, subsurface, coat, sheen or anisotropy, so the path tracer shades it with only the
// base's diffuse and specular lobes (SimplifyMaterial). Set at upload, see IsSimpleMaterial.
#define GPU_MATERIAL_SIMPLE (1u << 24)

// One MaterialData per row of a StructuredBuffer, packed to 144 bytes: the factors as halves (x in the
// low 16 bits), the coat's [0, 1] weight and roughness as unorm8, the texture indices whole, since the
// descriptor heap holds up to a million of them. The fields are MaterialData's; PackGpuMaterial and LoadGpuMaterial
// (Resources.slang) convert.
struct GpuMaterial
{
	uint2 baseColour; // half4
	uint roughnessMetallic; // half2
	uint normalScaleAlphaCutoff; // half2
	uint albedoTexture;
	uint roughMetalTexture;
	uint normalTexture;
	uint transmissionIor; // half2
	uint2 emissiveFlags; // half3 emissive, then the GPU_MATERIAL_* bits
	uint transmissionTexture;
	uint emissiveTexture;
	uint anisotropy; // half2
	uint anisotropyTexture;
	uint subsurfaceWeightTexture;
	uint subsurfaceColourTexture;
	uint2 subsurfaceColourWeight; // half4
	uint2 subsurfaceRadiusAnisotropy; // half4: radius, then scatter anisotropy
	uint2 attenuation; // half4: colour, then distance, infinite for no absorption
	uint subsurfaceRadiusTexture;
	uint subsurfaceScaleTexture;
	uint2 sheenColourRoughness; // half4: colour, then roughness
	uint sheenColourTexture;
	uint sheenRoughnessTexture;
	uint coatWeightRoughnessNormalScale; // unorm8 weight, unorm8 roughness, then half normal scale
	uint coatWeightTexture;
	uint coatRoughnessTexture;
	uint coatNormalTexture;
	uint2 specularColourWeight; // half4: colour, then weight
	uint specularTexture;
	uint specularColourTexture;
};

#ifdef __cplusplus
// Halves clamped to the finite range, except that an attenuation distance beyond it becomes infinity.
inline GpuMaterial PackGpuMaterial(const MaterialData &m)
{
	using namespace DirectX::PackedVector;
	auto half = [](float v) { return static_cast<uint32_t>(XMConvertFloatToHalf(std::clamp(v, -65504.0f, 65504.0f))); };
	auto pair = [&](float lo, float hi) { return half(lo) | (half(hi) << 16); };
	auto unorm8 = [](float v) { return static_cast<uint32_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
	const uint32_t infinity = 0x7C00;
	const uint32_t flags = (m.alphaMode << GPU_MATERIAL_ALPHA_MODE_SHIFT) |
						   (m.transmissionFlags << GPU_MATERIAL_TRANSMISSION_FLAGS_SHIFT) |
						   (m.subsurfaceWeightChannel << GPU_MATERIAL_WEIGHT_CHANNEL_SHIFT) |
						   (m.subsurfaceScaleChannel << GPU_MATERIAL_SCALE_CHANNEL_SHIFT);

	GpuMaterial p = {};
	p.baseColour = uint2(pair(m.baseColour.x, m.baseColour.y), pair(m.baseColour.z, m.baseColour.w));
	p.roughnessMetallic = pair(m.roughness, m.metallic);
	p.normalScaleAlphaCutoff = pair(m.normalScale, m.alphaCutoff);
	p.albedoTexture = m.albedoTexture;
	p.roughMetalTexture = m.roughMetalTexture;
	p.normalTexture = m.normalTexture;
	p.transmissionIor = pair(m.transmission, m.ior);
	p.emissiveFlags = uint2(pair(m.emissive.x, m.emissive.y), half(m.emissive.z) | flags);
	p.transmissionTexture = m.transmissionTexture;
	p.emissiveTexture = m.emissiveTexture;
	p.anisotropy = pair(m.anisotropy.x, m.anisotropy.y);
	p.anisotropyTexture = m.anisotropyTexture;
	p.subsurfaceWeightTexture = m.subsurfaceWeightTexture;
	p.subsurfaceColourTexture = m.subsurfaceColourTexture;
	p.subsurfaceColourWeight = uint2(pair(m.subsurfaceColour.x, m.subsurfaceColour.y),
									 pair(m.subsurfaceColour.z, m.subsurfaceWeight));
	p.subsurfaceRadiusAnisotropy = uint2(pair(m.subsurfaceRadius.x, m.subsurfaceRadius.y),
										 pair(m.subsurfaceRadius.z, m.subsurfaceScatterAnisotropy));
	p.attenuation = uint2(pair(m.attenuationColour.x, m.attenuationColour.y),
						  half(m.attenuationColour.z) |
								  ((m.attenuationDistance > 65504.0f ? infinity : half(m.attenuationDistance)) << 16));
	p.subsurfaceRadiusTexture = m.subsurfaceRadiusTexture;
	p.subsurfaceScaleTexture = m.subsurfaceScaleTexture;
	p.sheenColourRoughness = uint2(pair(m.sheenColour.x, m.sheenColour.y), pair(m.sheenColour.z, m.sheenRoughness));
	p.sheenColourTexture = m.sheenColourTexture;
	p.sheenRoughnessTexture = m.sheenRoughnessTexture;
	p.coatWeightRoughnessNormalScale = unorm8(m.coatWeight) | (unorm8(m.coatRoughness) << 8) | (half(m.coatNormalScale) << 16);
	p.coatWeightTexture = m.coatWeightTexture;
	p.coatRoughnessTexture = m.coatRoughnessTexture;
	p.coatNormalTexture = m.coatNormalTexture;
	p.specularColourWeight = uint2(pair(m.specularColour.x, m.specularColour.y), pair(m.specularColour.z, m.specularWeight));
	p.specularTexture = m.specularTexture;
	p.specularColourTexture = m.specularColourTexture;
	return p;
}

inline MaterialData UnpackGpuMaterial(const GpuMaterial &p)
{
	using namespace DirectX::PackedVector;
	auto lo = [](uint32_t bits) { return XMConvertHalfToFloat(static_cast<HALF>(bits & 0xFFFF)); };
	auto hi = [](uint32_t bits) { return XMConvertHalfToFloat(static_cast<HALF>(bits >> 16)); };
	const uint32_t flags = p.emissiveFlags.y;

	MaterialData m = {};
	m.baseColour = float4(lo(p.baseColour.x), hi(p.baseColour.x), lo(p.baseColour.y), hi(p.baseColour.y));
	m.roughness = lo(p.roughnessMetallic);
	m.metallic = hi(p.roughnessMetallic);
	m.normalScale = lo(p.normalScaleAlphaCutoff);
	m.alphaCutoff = hi(p.normalScaleAlphaCutoff);
	m.albedoTexture = p.albedoTexture;
	m.roughMetalTexture = p.roughMetalTexture;
	m.normalTexture = p.normalTexture;
	m.alphaMode = (flags >> GPU_MATERIAL_ALPHA_MODE_SHIFT) & 3;
	m.transmission = lo(p.transmissionIor);
	m.ior = hi(p.transmissionIor);
	m.transmissionFlags = (flags >> GPU_MATERIAL_TRANSMISSION_FLAGS_SHIFT) & 3;
	m.transmissionTexture = p.transmissionTexture;
	m.emissive = float3(lo(p.emissiveFlags.x), hi(p.emissiveFlags.x), lo(flags));
	m.emissiveTexture = p.emissiveTexture;
	m.attenuationColour = float3(lo(p.attenuation.x), hi(p.attenuation.x), lo(p.attenuation.y));
	m.attenuationDistance = hi(p.attenuation.y);
	m.subsurfaceColour = float3(lo(p.subsurfaceColourWeight.x), hi(p.subsurfaceColourWeight.x),
								lo(p.subsurfaceColourWeight.y));
	m.subsurfaceWeight = hi(p.subsurfaceColourWeight.y);
	m.subsurfaceWeightTexture = p.subsurfaceWeightTexture;
	m.subsurfaceColourTexture = p.subsurfaceColourTexture;
	m.anisotropy = float2(lo(p.anisotropy), hi(p.anisotropy));
	m.anisotropyTexture = p.anisotropyTexture;
	m.subsurfaceRadius = float3(lo(p.subsurfaceRadiusAnisotropy.x), hi(p.subsurfaceRadiusAnisotropy.x),
								lo(p.subsurfaceRadiusAnisotropy.y));
	m.subsurfaceScatterAnisotropy = hi(p.subsurfaceRadiusAnisotropy.y);
	m.subsurfaceRadiusTexture = p.subsurfaceRadiusTexture;
	m.subsurfaceScaleTexture = p.subsurfaceScaleTexture;
	m.subsurfaceWeightChannel = (flags >> GPU_MATERIAL_WEIGHT_CHANNEL_SHIFT) & 3;
	m.subsurfaceScaleChannel = (flags >> GPU_MATERIAL_SCALE_CHANNEL_SHIFT) & 3;
	m.sheenColour = float3(lo(p.sheenColourRoughness.x), hi(p.sheenColourRoughness.x), lo(p.sheenColourRoughness.y));
	m.sheenRoughness = hi(p.sheenColourRoughness.y);
	m.sheenColourTexture = p.sheenColourTexture;
	m.sheenRoughnessTexture = p.sheenRoughnessTexture;
	m.coatWeight = (p.coatWeightRoughnessNormalScale & 0xFF) / 255.0f;
	m.coatRoughness = ((p.coatWeightRoughnessNormalScale >> 8) & 0xFF) / 255.0f;
	m.coatNormalScale = hi(p.coatWeightRoughnessNormalScale);
	m.coatWeightTexture = p.coatWeightTexture;
	m.coatRoughnessTexture = p.coatRoughnessTexture;
	m.coatNormalTexture = p.coatNormalTexture;
	m.specularColour = float3(lo(p.specularColourWeight.x), hi(p.specularColourWeight.x), lo(p.specularColourWeight.y));
	m.specularWeight = hi(p.specularColourWeight.y);
	m.specularTexture = p.specularTexture;
	m.specularColourTexture = p.specularColourTexture;
	return m;
}

// Whether LoadMaterial would leave every layer above the base off for this row: the same tests on the
// same (unpacked half) values. A specular weight factor still counts as simple (it only changes the IOR),
// a specular colour or texture does not.
inline bool IsSimpleMaterial(const GpuMaterial &p)
{
	const MaterialData m = UnpackGpuMaterial(p);
	return !(m.transmission > 0.0f) && !(m.subsurfaceWeight > 0.0f) && !(m.coatWeight > 0.0f) &&
		   !(m.sheenColour.x > 0.0f || m.sheenColour.y > 0.0f || m.sheenColour.z > 0.0f) &&
		   m.anisotropy.x == 0.0f && m.anisotropy.y == 0.0f && m.specularColour.x == 1.0f &&
		   m.specularColour.y == 1.0f && m.specularColour.z == 1.0f &&
		   m.specularTexture == INVALIDDESCRIPTORHANDLE && m.specularColourTexture == INVALIDDESCRIPTORHANDLE;
}
#endif

struct GpuVertex
{
	float3 position;
	float3 normal;
	float2 texCoord; // TEXCOORD_0
	float4 tangent; // .w = bitangent sign
};

// One per render pixel under DLSS_RR: what TracePath keeps for the camera between the path segments
// that touch it, packed by PathTracer/CameraState.slang. Each part is loaded and stored on its own.
struct GpuCameraGuides
{
	float3 position; // its w in flags
	uint normal; // octahedral snorm16x2
	uint3 albedos; // half: diffuse rgb, then specular rgb
	uint roughnessCurvature; // half2
	float specularHitDistance;
	uint flags; // CAMERA_GUIDES_* in CameraState.slang
	float2 chainMotion;
};

// Only an active chain is stored.
struct GpuSpecularChain
{
	float4 rotor;
	uint2 throughputFraction; // half: throughput rgb, primary fraction
	float primaryDistance;
	float length;
	float pixelDeflection;
	uint flags; // CAMERA_CHAIN_* in CameraState.slang
	// half: dP xyz then dD xyz of the pixelU, pixelV and motion differentials
	uint3 pixelU;
	uint3 pixelV;
	uint3 motion;
};

struct GpuPaneReflection
{
	uint2 albedoLuminance; // half: albedo rgb, luminance
	float distance;
	float curvature;
	float hitDistance;
};

struct GpuCameraState
{
	GpuCameraGuides guides;
	GpuSpecularChain chain;
	GpuPaneReflection panes;
	uint2 transmittance; // half: camera transmittance rgb, then unused
	float distance;
	float3 cameraDirection; // the camera ray's, rather than held in registers through the path
	uint3 layerAlbedos; // half: diffuse rgb, specular rgb of the blended layers the guides do not describe
};

// One per render pixel: the branch a glass split leaves for later (PathTracer/GlassSplit.slang).
struct GpuDeferredBranch
{
	float3 origin;
	float coneWidth;
	float3 throughput;
	float coneSpread;
	float3 direction;
	float rrInvScale; // the branch's Russian roulette scale, see TracePath
};

// Root constants for the tonemap compute pass (bound as g_PushConstants).
// The exposure (a linear multiplier) that Bloom.slang and Tonemap.slang apply is not in their constants but in
// AutoExposurePass's exposure buffer, one float.
struct TonemapConstants
{
	uint toneMapOperator; // ToneMappingOperator
	uint width;
	uint height;
	uint hdr10; // 1: output Rec.2020 PQ (HDR10) instead of sRGB
	float paperWhiteNits; // luminance of diffuse white (1.0) when hdr10 is set
	float peakNits; // display peak luminance, the HDR curve maps whiteClip to it when hdr10 is set
	float whiteClip; // scene value (1.0 = paper white) the Neutwo curve maps exactly to its peak
	uint hdrGammaCorrection; // 1: re-encode sRGB and decode with 2.2 so HDR shadows match SDR displays
	float grainStrength; // film grain standard deviation in sRGB-encoded units at mid-grey, 0 = off
	float grainSize; // grain lattice spacing in output pixels
	float grainColour; // 0: monochrome grain, 1: independent grain per channel
	float grainResponse; // exponent on the mid-tone bell, 0 = the same grain at every level
	uint grainSeed; // changes whenever the grain pattern should change
};

// The DLSS-RR guide GuideViewPass shows in place of the image.
UnscopedEnum
enum GuideView
{
	GUIDE_VIEW_OFF = 0,
	GUIDE_VIEW_DIFFUSE_ALBEDO = 1,
	GUIDE_VIEW_SPECULAR_ALBEDO = 2,
	GUIDE_VIEW_NORMAL = 3,
	GUIDE_VIEW_ROUGHNESS = 4,
	GUIDE_VIEW_DEPTH = 5,
	GUIDE_VIEW_MOTION_VECTORS = 6,
	GUIDE_VIEW_SPECULAR_MOTION_VECTORS = 7
};

// Root constants of GuideView.slang.
struct GuideViewConstants
{
	uint2 renderSize;
	uint2 outputSize;
	float2 prevJitter; // last frame's FrameConstants::jitter
	uint guide; // GuideView
	uint change; // 1: how much the guide changed since last frame along the motion vectors, as heat
	float gain; // scales the heat, or the depth and motion vectors' colours
	uint historyValid; // 0: last frame showed another guide or none, so there is nothing to compare with
	uint hdr10; // as TonemapConstants
	float paperWhiteNits;
	uint showHeld; // 1: tint the pixels whose guides were held this frame (PrimaryGuides::hold)
};

// Root constants for the bloom compute passes (bound as g_PushConstants). Generated root signatures hold
// at most 16 dwords of root constants, which this fills.
struct BloomConstants
{
	float2 inputTexelSize; // 1 / the size of the texture the pass filters (bloom level 0 for Composite)
	float2 outputTexelSize; // 1 / the output's size
	uint width; // output size
	uint height;
	uint karisAverage; // 1: the first downsample, which weighs its boxes down by their brightness, exposed as displayed
	float strength; // Composite: how much of the image is replaced by the bloom
	float levelCount; // Composite: bloom levels summed into level 0
	float2 tanHalfFov; // horizontal, vertical
	float3 sunDirectionView; // towards the sun in the camera's (right, up, front) basis
	float sunMaskOneMinusCos; // first downsample: 1 - cos of the angle around the sun it leaves out, 0 = none
};

// Root constants of the auto exposure kernels (AutoExposure.slang). Handles are descriptor heap indices of UAVs.
struct AutoExposureConstants
{
	uint2 bufferDimensions;
	float frameTime; // seconds the key luminance adapts over, 0 = start from this frame's
	float exposureBias; // linear, 2^EV: multiplies the auto exposure, or is the exposure while it is off
	float maxExposure; // linear, the most the auto exposure may reach, bias included
	uint inputBufferHandle; // RWTexture2D<float4>, the HDR image
	uint histogramHandle; // RWStructuredBuffer<uint>, HISTOGRAM_SIZE bins
	uint keySceneLuminanceHandle; // RWStructuredBuffer<float>, the adapted luminance
	uint exposureHandle; // RWStructuredBuffer<float>, the result
	uint enabled; // 0: CalculateExposure only writes exposureBias
};

#ifdef __cplusplus
static_assert(sizeof(AutoExposureConstants) <= 16 * 4, "AutoExposureConstants must fit the root constants");
static_assert(sizeof(FrameConstants) == 624,"FrameConstants must match the HLSL cbuffer layout");
static_assert(sizeof(BloomConstants) <= 16 * 4, "BloomConstants must fit the 16 root constants of a generated root signature");
static_assert(sizeof(GpuMaterial) == 144,"GpuMaterial must match the structured buffer stride");
static_assert(sizeof(GpuVertex) == 48 && offsetof(GpuVertex, position) == 0,
			  "GpuVertex must match the structured buffer stride");
static_assert(sizeof(GpuCameraGuides) == 48 && sizeof(GpuSpecularChain) == 76 && sizeof(GpuPaneReflection) == 20,
			  "GpuCameraState parts must match their HLSL layout");
static_assert(sizeof(GpuCameraState) == 180, "GpuCameraState must match the structured buffer stride");
static_assert(sizeof(GpuDeferredBranch) == 48, "GpuDeferredBranch must match the structured buffer stride");
static_assert(sizeof(TonemapConstants) == 52, "TonemapConstants must fit the root constants");
static_assert(sizeof(GuideViewConstants) <= 16 * 4, "GuideViewConstants must fit the root constants");
#endif

#endif // PATH_TRACER_INTEROP_H
