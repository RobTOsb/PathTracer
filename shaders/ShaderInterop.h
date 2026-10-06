#ifndef SHADER_INTEROP_H
#define SHADER_INTEROP_H

#ifdef __cplusplus
#include <algorithm>
#include <math.h>
#include <cstddef>
#include <cstring>
#include <stdint.h>
#include "Maths.h"

#define AlignAs(x) alignas(x)
#define RowMajor
#define UnscopedEnum

// Pack a float3 into R11G11B10_FLOAT (same layout as DXGI_FORMAT_R11G11B10_FLOAT).
// Components must be non-negative; range up to ~65504.
inline uint32_t PackR11G11B10(float3 v)
{
	using namespace DirectX::PackedVector;
	XMFLOAT3PK pk;
	XMStoreFloat3PK(&pk, XMLoadFloat3(&v));
	return pk.v;
}

// Pack two float32 values as float16 into a uint32.
// lo occupies bits [15:0], hi occupies bits [31:16].
inline uint32_t PackF16Pair(float lo, float hi)
{
	using namespace DirectX::PackedVector;
	return static_cast<uint32_t>(XMConvertFloatToHalf(lo)) | (static_cast<uint32_t>(XMConvertFloatToHalf(hi)) << 16);
}

// Reinterpret a uint32 bit pattern as a float (HLSL asfloat equivalent).
inline float AsFloatBits(uint32_t bits)
{
	float f;
	std::memcpy(&f, &bits, sizeof(f));
	return f;
}

#else
#define AlignAs(x)
#define RowMajor row_major
// Slang enums are scoped by default; keep C-style unqualified enumerator access on both sides.
#define UnscopedEnum [UnscopedEnum]
#define ResourceHeap(index) ResourceDescriptorHeap[index]
#define SamplerHeap(index) SamplerDescriptorHeap[index]

#endif

struct GpuVirtualAddressRange
{
	uint2 startAddress; // 64-bit GPU virtual address, split into two 32-bit halves
	uint2 sizeInBytes;
};

struct GpuVirtualAddressRangeAndStride
{
	uint2 startAddress;
	uint2 sizeInBytes;
	uint2 strideInBytes;
};

#ifndef QUARTER_PI // pi/4
#define QUARTER_PI 0.78539816339744830961566084581988
#endif

#ifndef HALF_PI // pi/2
#define HALF_PI 1.5707963267948966192313216916398
#endif

#ifndef PI // pi
#define PI 3.1415926535897932384626433832795
#endif

#ifndef TWO_PI // 2pi
#define TWO_PI 6.283185307179586476925286766559
#endif

#ifndef FOUR_PI // 4pi
#define FOUR_PI 12.566370614359172953850573533118
#endif

#ifndef INV_PI // 1/pi
#define INV_PI 0.31830988618379067153776752674503
#endif

#ifndef INV_TWO_PI // 1/(2pi)
#define INV_TWO_PI 0.15915494309189533576888376337251
#endif

#ifndef INV_FOUR_PI // 1/(4pi)
#define INV_FOUR_PI 0.07957747154594766788444188168626
#endif

#ifndef TWO_INV_PI // 2/pi
#define TWO_INV_PI 0.63661977236758134307553505349006
#endif

#ifndef UINT_MAX
#define UINT_MAX 0xFFFFFFFF
#endif

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

#ifndef INT_MIN
#define INT_MIN -2147483649
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38
#endif

#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.192092896e-07
#endif

#ifndef DBL_MAX
#define DBL_MAX 1.7976931348623158e+308
#endif

#ifndef DBL_MIN
#define DBL_MIN 2.2250738585072014e-308
#endif

#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131e-016
#endif

#define INVALIDDESCRIPTORHANDLE 0xFFFFFFFF

// Angular radius of the physical sun disc in radians (~0.53 degrees full-angle).
#define SUN_ANGULAR_RADIUS_DEFAULT 0.004675f

#define DISPLAY_CURVE_SRGB 0
#define DISPLAY_CURVE_ST2084 1
#define DISPLAY_CURVE_LINEAR 2

UnscopedEnum
enum ToneMappingOperator
{
	TONEMAP_REINHARD = 0,
	TONEMAP_REINHARD_EXTENDED = 1,
	TONEMAP_KHRONOS_PBR_NEUTRAL = 2,
	TONEMAP_ACES_FAST = 3,
	TONEMAP_ACES_FITTED = 4,
	TONEMAP_ACES_FILM = 5,
	TONEMAP_UNCHARTED2 = 6,
	TONEMAP_AGX = 7,
	TONEMAP_AGX_FITTED = 8,
	TONEMAP_OFF = 9
};

#ifndef ENVIRONMENT_LIGHT_BOOST
#define ENVIRONMENT_LIGHT_BOOST 10.0f
#endif

// NOTE: named GpuLight (not Light) to avoid clashing with the CPU-side authoring
// class `Light` declared in Scene.h, which shares this global namespace.
UnscopedEnum
enum GpuLightType
{
	kLight_Point = 0xFFF0FF80,
	kLight_Spot,
	kLight_Direction,
	kLight_Environment,
	kLight_Area,
	kLight_Count
};

struct GpuLight
{
	float4 radiance; // .xyz = radiance, .w = radiance_map
	float4 v1; // .xyz = 1st vertex pos., .w = 1st packed UVs
	float4 v2; // .xyz = 2nd vertex pos., .w = 2nd packed UVs
	float4 v3; // .xyz = 3rd vertex pos., .w = 3rd packed UVs

	/**
	 * Get the type of the current light.
	 * @return GpuLightType enum specifying type of light.
	 */
	GpuLightType GetLightType()
	{
		// All delta lights have a pack value of 0.0.xxx with the type of light in w
		// To avoid increasing the size of the struct we use this pack variable to distinguish if the light is
		// a delta light or an area light. There is the potential for an area light to have v3==0.0.xxx so the
		// light types are represented using bit patterns that cannot occur in float32/16 numbers
#ifndef __cplusplus
		uint index = asuint(v3.w);
#else
		const uint index = *reinterpret_cast<uint *>(&v3.w);
#endif
		if (index < (uint) kLight_Point)
		{
			return kLight_Area;
		}
		return (GpuLightType) index;
	}

	// The member variable can be interpreted differently depending on the actual type of light being stored
	// The following lists the interpretation for each light type
	// Note: float3s are considered to have same storage size as float4s
	//       All lights apart (from AreaLights) must have asfloat(-1) in the radiance.w variable
	// struct PointLight
	//{
	//    float3 intensity; /**< The light luminous intensity (lm/sr) */
	//    float4 position; /**< The light world space position in xyz with w=range */
	//    float radius; /**< v2.x - sphere radius, 0 for a true point light */
	//};
	// struct SpotLight
	//{
	//    float3 intensity; /**< The light luminous intensity (lm/sr) in xyz */
	//    float4 position; /**< The light world space direction in xyz with w=range */
	//    float3 direction; /**< The light world space direction to the light */
	//    float2 angles; /**< x=angle cutoff scale, y=angle cutoff offset */
	//    /**< The cutoff angles are the cosine of the light cutoff angle from center of light (radians) */
	//};
	// struct DirectionLight
	//{
	//    float3 irradiance; /**< The light illuminance (lm/m^2) in radiance.xyz */
	//    // radiance.w = asfloat(UINT_MAX) sentinel
	//    float3 direction; /**< The light world space direction to the light in v2.xyz */
	//    float range;      /**< v2.w */
	//    float sunAngularRadius; /**< v3.x - angular radius of the sun in radians (physical sun ~0.004675) */
	//    // v3.w = packed light-type code (kLight_Direction)
	//};
	// struct EnvironmentLight
	//{
	//    uint mips; /**< The number of mip map levels in each face of the texture (radiance.x) */
	//    uint width; /**< The environment map texture width/height (radiance.y) */
	//    uint mapIndex; /**< Bindless descriptor-heap index of the environment cube map (radiance.z) */
	//    // radiance.w = asfloat(UINT_MAX) sentinel
	//};
};

#ifdef __cplusplus
// Compose a float4 from a float3 plus a w component (DirectXMath has no such ctor).
inline float4 MakeFloat4(float3 const &xyz, float w)
{
	return float4(xyz.x, xyz.y, xyz.z, w);
}

// Normalise a float3 direction using DirectXMath.
inline float3 NormaliseFloat3Copy(float3 const &v)
{
	float3 out;
	XMStoreFloat3(&out, XMVector3Normalize(XMLoadFloat3(&v)));
	return out;
}

// Pack a float2 UV as two float16 values into the bit pattern of a float (asfloat of packHalf2x16).
inline float PackUVsToFloat(float2 const &uv)
{
	return AsFloatBits(PackF16Pair(uv.x, uv.y));
}

/**
 * Make a light type from an area light.
 * @param radiance Colour and value of light.
 * @param vertex1  1st vertex position (vertices are expected counter-clockwise in right-handed system).
 * @param vertex2  2nd vertex position.
 * @param vertex3  3rd vertex position.
 * @return A light with the correctly set internal values.
 */
inline GpuLight MakeAreaLight(float3 const radiance, float3 const vertex1, float3 const vertex2, float3 const vertex3)
{
	// Create the new light
	GpuLight light = {};
	light.radiance = MakeFloat4(radiance, AsFloatBits(UINT_MAX));
	light.v1 = MakeFloat4(vertex1, 0.0f);
	light.v2 = MakeFloat4(vertex2, 0.0f);
	light.v3 = MakeFloat4(vertex3, 0.0f);
	return light;
}

/**
 * Make a light type from a textured area light.
 * @param radiance Colour and value of light texture multiplier.
 * @param vertex1  1st vertex position (vertices are expected counter-clockwise in right-handed system).
 * @param vertex2  2nd vertex position.
 * @param vertex3  3rd vertex position.
 * @param texture  Index of radiance map texture.
 * @param uv1      1st vertex uv texture parameter.
 * @param uv2      2nd vertex uv texture parameter.
 * @param uv3      3rd vertex uv texture parameter.
 * @return A light with the correctly set internal values.
 */
inline GpuLight MakeAreaLight(float3 const radiance,
							  float3 const vertex1,
							  float3 const vertex2,
							  float3 const vertex3,
							  uint const texture,
							  float2 const uv1,
							  float2 const uv2,
							  float2 const uv3)
{
	// Create the new light
	GpuLight light = {};
	light.radiance = MakeFloat4(radiance, AsFloatBits(texture));
	light.v1 = MakeFloat4(vertex1, PackUVsToFloat(uv1));
	light.v2 = MakeFloat4(vertex2, PackUVsToFloat(uv2));
	light.v3 = MakeFloat4(vertex3, PackUVsToFloat(uv3));
	return light;
}

/**
 * Make a light type from a point light.
 * @param intensity Colour and intensity of light.
 * @param position  Position of the light.
 * @param range     Maximum distance from the light where lighting has an effect.
 * @param radius    Radius of a spherical emitter with the same intensity, 0 for a delta point light.
 * @return A light with the correctly set internal values.
 */
inline GpuLight MakePointLight(float3 const intensity, float3 const position, float const range, float const radius = 0.0f)
{
	GpuLight light = {};
	light.radiance = MakeFloat4(intensity, AsFloatBits(UINT_MAX));
	light.v1 = MakeFloat4(position, range);
	light.v2 = float4(radius, 0.0f, 0.0f, 0.0f);
	light.v3 = float4(0.0f, 0.0f, 0.0f, AsFloatBits(static_cast<uint32_t>(kLight_Point)));
	return light;
}

/**
 * Make a light type from a spot light.
 * @param intensity      Colour and intensity of light.
 * @param position       Position of the light.
 * @param range          Maximum distance from the light where lighting has an effect.
 * @param direction      The direction to the light along cones view axis.
 * @param outerConeAngle The maximum angle from the cones view axis to the outside of the cone.
 * @param innerConeAngle The angle from the cones view axis to the inside of the cones penumbra region.
 * @return A light with the correctly set internal values.
 */
inline GpuLight MakeSpotLight(float3 const intensity,
							  float3 const position,
							  float const range,
							  float3 const direction,
							  float const outerConeAngle,
							  float const innerConeAngle)
{
	float const cosOuter = -cosf(outerConeAngle);
	float const lightAngleScale = 1.0f / (std::max) (0.001f, cosf(innerConeAngle) + cosOuter);
	float const lightAngleOffset = cosOuter * lightAngleScale;
	float const sinAngle = sinf(outerConeAngle);
	float const tanAngle = tanf(outerConeAngle);
	GpuLight light = {};
	light.radiance = MakeFloat4(intensity, AsFloatBits(UINT_MAX));
	light.v1 = MakeFloat4(position, range);
	light.v2 = MakeFloat4(NormaliseFloat3Copy(direction), sinAngle);
	light.v3 = float4(lightAngleScale, lightAngleOffset, tanAngle, AsFloatBits(static_cast<uint32_t>(kLight_Spot)));
	return light;
}

/**
 * Make a light type from a directional light.
 * @param radiance         Colour and value of light.
 * @param direction        Direction to the light along lights view axis.
 * @param range            Maximum distance from the light where lighting has an effect.
 * @param sunAngularRadius Angular radius of the sun disc in radians (physical sun ~0.004675).
 * @return A light with the correctly set internal values.
 */
inline GpuLight MakeDirectionalLight(float3 const radiance,
									 float3 const direction,
									 float const range,
									 float const sunAngularRadius = SUN_ANGULAR_RADIUS_DEFAULT)
{
	GpuLight light = {};
	light.radiance = MakeFloat4(radiance, AsFloatBits(UINT_MAX));
	light.v2 = MakeFloat4(NormaliseFloat3Copy(direction), range);
	// v3.x carries the sun angular radius; v3.w carries the packed light-type code.
	light.v3 = float4(sunAngularRadius, 0.0f, 0.0f, AsFloatBits(static_cast<uint32_t>(kLight_Direction)));
	return light;
}

/**
 * Make a light type from a environment light.
 * @param mips     The number of mip levels contained in the texture (must be atleast 1).
 * @param width    The width of each face in the environment cube map (height must equal width).
 * @param mapIndex Bindless descriptor-heap index of the environment cube map (TextureCube).
 * @return A light with the correctly set internal values.
 */
inline GpuLight MakeEnvironmentLight(uint const mips, uint const width, uint const mapIndex)
{
	GpuLight light = {};
	light.radiance = float4(AsFloatBits(mips), AsFloatBits(width), AsFloatBits(mapIndex), AsFloatBits(UINT_MAX));
	light.v3 = float4(0.0f, 0.0f, 0.0f, AsFloatBits(static_cast<uint32_t>(kLight_Environment)));
	return light;
}

// CPU-side scene light type codes (must match Scene.h's LightType enum class values).
// Kept here so the scene can build a GpuLight without ShaderInterop depending on Scene.h.
enum SceneLightTypeCode
{
	kSceneLight_Directional = 0,
	kSceneLight_Point = 1,
	kSceneLight_Spot = 2,
	kSceneLight_Sun = 3,
	// Environment lights are not built through MakeGpuLight (they carry map index / mips /
	// width rather than positional/directional data); they are built directly via
	// MakeEnvironmentLight. This code exists for parity with Scene.h's LightType enum.
	kSceneLight_Environment = 4,
};

/**
 * Build a packed GpuLight from CPU-side scene light parameters.
 * @param sceneType         SceneLightTypeCode (Directional/Point/Spot/Sun).
 * @param radiance          Colour * intensity to emit.
 * @param positionWS        World-space position (point / spot).
 * @param directionWS       World-space direction the light emits along (spot / directional / sun).
 * @param range             Falloff range (0 = infinite).
 * @param innerConeAngle    Spot inner cone half-angle (radians).
 * @param outerConeAngle    Spot outer cone half-angle (radians).
 * @param sunAngularRadius  Sun disc angular radius (radians); only used by directional/sun lights.
 * @return A GpuLight with the correct packed layout for its type.
 *
 * NOTE: Sun is treated as a directional light on the GPU; its disc/atmospheric
 *       behaviour is handled by the sky-atmosphere system, not the light buffer.
 */
inline GpuLight MakeGpuLight(uint32_t sceneType,
							 float3 const &radiance,
							 float3 const &positionWS,
							 float3 const &directionWS,
							 float range,
							 float innerConeAngle,
							 float outerConeAngle,
							 float sunAngularRadius = SUN_ANGULAR_RADIUS_DEFAULT)
{
	switch (sceneType)
	{
		case kSceneLight_Point:
			return MakePointLight(radiance, positionWS, range);
		case kSceneLight_Spot:
			return MakeSpotLight(radiance, positionWS, range, directionWS, outerConeAngle, innerConeAngle);
		case kSceneLight_Directional:
		case kSceneLight_Sun:
		default:
			return MakeDirectionalLight(radiance, directionWS, range, sunAngularRadius);
	}
}
#endif

/**
 * Check if a light is a delta light.
 * @param light The light to be checked.
 * @return True if a delta light.
 */
inline bool IsDeltaLight(GpuLight light)
{
#if defined(DISABLE_DELTA_LIGHTS)
	return false;
#elif !defined(DISABLE_AREA_LIGHTS) && !defined(DISABLE_ENVIRONMENT_LIGHTS)
	const GpuLightType type = light.GetLightType();
	return type != kLight_Area && type != kLight_Environment;
#elif !defined(DISABLE_AREA_LIGHTS)
	return light.GetLightType() != kLight_Area;
#elif defined(DISABLE_AREA_LIGHTS) && defined(DISABLE_ENVIRONMENT_LIGHTS)
	return true;
#else /*defined(DISABLE_AREA_LIGHTS)*/
	return light.GetLightType() != kLight_Environment;
#endif
}

/**
 * Check if a light has a known position.
 * @note Lights such as directional and environment do not have positions and only directions.
 * @param light The light to be checked.
 * @return True if light has a position.
 */
inline bool HasLightPosition(GpuLight light)
{
#if !defined(DISABLE_DELTA_LIGHTS) && !defined(DISABLE_ENVIRONMENT_LIGHTS)
	const GpuLightType type = light.GetLightType();
	return type != kLight_Direction && type != kLight_Environment;
#elif defined(DISABLE_DELTA_LIGHTS) && !defined(DISABLE_ENVIRONMENT_LIGHTS) && !defined(DISABLE_AREA_LIGHTS)
	return light.GetLightType() != kLight_Environment;
#elif !defined(DISABLE_DELTA_LIGHTS) && defined(DISABLE_ENVIRONMENT_LIGHTS)
	return light.GetLightType() != kLight_Direction;
#elif defined(DISABLE_DELTA_LIGHTS) && defined(DISABLE_ENVIRONMENT_LIGHTS) && !defined(DISABLE_AREA_LIGHTS)
	return true;
#else
	return false;
#endif
}

#endif // SHADER_INTEROP_H
