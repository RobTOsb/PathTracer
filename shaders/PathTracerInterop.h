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
	uint pad1;
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
	uint pad0;

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
	uint pad2;

	// DLSS-RR guides (DLSS_RR permutation only). Mirrors at most psrMaxRoughness (perceptual) rough
	// are described by the surface they reflect, primary surface replacement; negative = off.
	// Surfaces at most specularHitDistanceHoldRoughness rough keep last frame's specular hit
	// distance when their path took another lobe, instead of reporting none; negative = off.
	float psrMaxRoughness;
	float specularHitDistanceHoldRoughness;
	float2 pad3;
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

// One row per glTF material in a StructuredBuffer. Texture fields are bindless descriptor-heap
// indices, INVALIDDESCRIPTORHANDLE when the material has no such map.
struct GpuMaterial
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
	uint pad0;
	uint pad1;
	uint pad2;
	float3 emissive; // emissiveFactor * KHR_materials_emissive_strength, linear, front faces only
	uint emissiveTexture; // sRGB, multiplies emissive
};

struct GpuVertex
{
	float3 position;
	float3 normal;
	float2 texCoord; // TEXCOORD_0
	float4 tangent; // .w = bitangent sign
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
	float peakNits; // display peak luminance, highlights roll off towards it when hdr10 is set
	float rolloffKnee; // fraction of peakNits where the highlight roll-off starts
};

#ifdef __cplusplus
static_assert(sizeof(FrameConstants) == 448,"FrameConstants must match the HLSL cbuffer layout");
static_assert(sizeof(GpuMaterial) == 80, "GpuMaterial must match the structured buffer stride");
static_assert(sizeof(GpuVertex) == 48 && offsetof(GpuVertex, position) == 0,
			  "GpuVertex must match the structured buffer stride");
static_assert(sizeof(TonemapConstants) == 32, "TonemapConstants must fit the root constants");
#endif

#endif // PATH_TRACER_INTEROP_H
