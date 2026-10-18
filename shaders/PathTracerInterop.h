#ifndef PATH_TRACER_INTEROP_H
#define PATH_TRACER_INTEROP_H

#include "ShaderInterop.h"

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

	uint maxBounces;
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
	// scaled up by the aerial perspective scale; the sun and the sky-view LUT are not.
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
	// are described by the surface they reflect or refract, primary surface replacement; negative = off.
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
	uint pad0;
	uint pad1;
	uint pad2;

	// Last frame's camera position, for the motion vectors of what curved mirrors and glass show.
	float3 prevCameraPosition;
	// Chains of curved mirrors and glass keep PSR, with motion vectors traced through them, unless they
	// shrink the image of the surface they end on by more than this (linear); negative = off, which
	// leaves only the chains that barely bend the virtual image.
	float psrMaxMinification;
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
	LIGHT_SAMPLING_TREE = 0, // SG light tree (Tokuyoshi et al. 2024)
	LIGHT_SAMPLING_UNIFORM = 1,
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
};

// GpuMaterial::emissiveFlags bits above the half emissive blue.
#define GPU_MATERIAL_ALPHA_MODE_SHIFT 16 // GpuAlphaMode, 2 bits
#define GPU_MATERIAL_TRANSMISSION_FLAGS_SHIFT 18 // GpuTransmissionFlags, 2 bits
#define GPU_MATERIAL_WEIGHT_CHANNEL_SHIFT 20 // subsurfaceWeightChannel, 2 bits
#define GPU_MATERIAL_SCALE_CHANNEL_SHIFT 22 // subsurfaceScaleChannel, 2 bits

// One MaterialData per row of a StructuredBuffer, packed to 128 bytes (one aligned cache line, four
// spare dwords at the end): the factors as halves (x in the low 16 bits), the texture indices whole, since the descriptor heap
// holds up to a million of them. The fields are MaterialData's; PackGpuMaterial and LoadGpuMaterial
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
	uint4 reserved;
};

#ifdef __cplusplus
// Halves clamped to the finite range, except that an attenuation distance beyond it becomes infinity.
inline GpuMaterial PackGpuMaterial(const MaterialData &m)
{
	using namespace DirectX::PackedVector;
	auto half = [](float v) { return static_cast<uint32_t>(XMConvertFloatToHalf(std::clamp(v, -65504.0f, 65504.0f))); };
	auto pair = [&](float lo, float hi) { return half(lo) | (half(hi) << 16); };
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
	return m;
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
// that touch it, packed by PathTracer/CameraState.slang.
struct GpuCameraState
{
	float3 guidePosition; // the guides' position, its w in flags
	uint guideNormal; // octahedral snorm16x2
	uint3 guideAlbedos; // half: diffuse rgb, then specular rgb
	uint guideRoughnessCurvature; // half2
	float guideSpecularHitDistance;
	float chainPrimaryDistance;
	float chainLength;
	float chainPixelDeflection;
	float4 chainRotor;
	uint2 chainThroughputFraction; // half: throughput rgb, primary fraction
	uint2 paneAlbedoLuminance; // half: albedo rgb, luminance
	float paneDistance;
	uint paneCurvatureHitDistance; // half2
	uint2 transmittance; // half: camera transmittance rgb, then unused
	float distance;
	uint flags; // CAMERA_STATE_* in CameraState.slang
	// half: dP xyz then dD xyz of SpecularChain's pixelU, pixelV and motion differentials while the chain
	// is active. Once it has ended, chainPixelU.xy holds PrimaryGuides::chainMotion (float2).
	uint3 chainPixelU;
	uint3 chainPixelV;
	uint3 chainMotion;
};

// Root constants for the tonemap compute pass (bound as g_PushConstants).
struct TonemapConstants
{
	float exposure; // linear multiplier, 2^EV
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

#ifdef __cplusplus
static_assert(sizeof(FrameConstants) == 480,"FrameConstants must match the HLSL cbuffer layout");
static_assert(sizeof(GpuMaterial) == 128,"GpuMaterial must match the structured buffer stride");
static_assert(sizeof(GpuVertex) == 48 && offsetof(GpuVertex, position) == 0,
			  "GpuVertex must match the structured buffer stride");
static_assert(sizeof(GpuCameraState) == 140, "GpuCameraState must match the structured buffer stride");
static_assert(sizeof(TonemapConstants) == 56, "TonemapConstants must fit the root constants");
#endif

#endif // PATH_TRACER_INTEROP_H
