#ifndef SKY_ATMOSPHERE_INTEROP_H
#define SKY_ATMOSPHERE_INTEROP_H

#ifdef __cplusplus
#include "Maths.h"
#endif

// Look-up tables of "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
// (Hillaire 2020), all RGBA16F.
#define TRANSMITTANCE_LUT_WIDTH 256u
#define TRANSMITTANCE_LUT_HEIGHT 64u
#define MULTI_SCATTERING_LUT_SIZE 32u
#define SKY_VIEW_LUT_WIDTH 192u
#define SKY_VIEW_LUT_HEIGHT 108u

// Threads per multiple scattering texel, one per direction of an 8 x 8 sphere grid.
#define MULTI_SCATTERING_SAMPLES 64u

// Kept clear of the ground so that the LUT parameterisations stay defined. Also a normalised height
// offset in the multiple scattering LUT, so it must stay this small whatever the unit system (km).
#define PLANET_RADIUS_OFFSET 0.002f

// The planet and its atmosphere, in kilometres with the planet centre at the origin. Scattering and
// extinction coefficients are per km at the density peak; the LUTs are rendered for a sun of unit
// illuminance, which the path tracer then scales by its own.
struct AtmosphereParameters
{
	float3 rayleighScattering;
	float rayleighDensityExpScale; // -1 / scale height
	float3 mieScattering;
	float mieDensityExpScale; // -1 / scale height
	float3 mieExtinction;
	float miePhaseG; // Cornette-Shanks asymmetry
	float3 mieAbsorption;
	float bottomRadius;
	float3 absorptionExtinction; // ozone
	float topRadius;
	float3 groundAlbedo;
	float multipleScatteringFactor;

	// Ozone density: a tent of two linear layers, below and above absorptionDensity0LayerWidth.
	float absorptionDensity0LayerWidth;
	float absorptionDensity0ConstantTerm;
	float absorptionDensity0LinearTerm;
	float absorptionDensity1ConstantTerm;
	float absorptionDensity1LinearTerm;
	float pad0;
	float pad1;
	float pad2;
};

// Root constants of the LUT kernels in SkyAtmosphereLUTs.slang (bound as g_PushConstants).
struct AtmosphereLUTConstants
{
	float viewHeight; // km from the planet centre, where the sky-view LUT and the scene lighting are taken
	float sunZenithCos; // cosine of the sun's angle from the zenith there
	float pad0;
	float pad1;
};

// What the path tracer needs from the LUTs at the scene, written by the SceneLighting kernel as two
// float4 rows of a structured buffer.
#define ATMOSPHERE_LIGHTING_SUN_TRANSMITTANCE 0 // .rgb, top of the atmosphere to the scene, 0 below the horizon
#define ATMOSPHERE_LIGHTING_MULTI_SCATTERING 1 // .rgb, Psi_ms for a unit sun illuminance
#define ATMOSPHERE_LIGHTING_ROWS 2

#ifdef __cplusplus
static_assert(sizeof(AtmosphereParameters) % 16 == 0, "AtmosphereParameters must match the HLSL cbuffer layout");
static_assert(sizeof(AtmosphereParameters) == 128, "AtmosphereParameters must match the HLSL cbuffer layout");
static_assert(sizeof(AtmosphereLUTConstants) == 16, "AtmosphereLUTConstants must fit the root constants");
#endif

#endif // SKY_ATMOSPHERE_INTEROP_H
