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
	float lightIntensityScale; // multiplies every light's radiance or intensity
	float sunSelectProbability;

	float lightTreeDiffuseOnlyAlpha;
	uint framePad0;
	uint framePad1;
	uint framePad2;
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
};

#ifdef __cplusplus
static_assert(sizeof(FrameConstants) == 352,"FrameConstants must match the HLSL cbuffer layout");
static_assert(sizeof(GpuMaterial) == 64, "GpuMaterial must match the structured buffer stride");
static_assert(sizeof(TonemapConstants) == 24, "TonemapConstants must fit the root constants");
#endif

#endif // PATH_TRACER_INTEROP_H
