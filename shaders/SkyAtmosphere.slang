// Sky and atmosphere model of "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
// (Hillaire 2020): participating media, phase functions and the LUT parameterisations, shared by the
// LUT kernels in SkyAtmosphereLUTs.slang and the path tracer. Distances are in km, with the planet
// centre at the origin and +Y up. Nothing here binds resources.

#include "SkyAtmosphereInterop.h"

static const float kAtmospherePi = 3.14159265358979;

// The interop macros for importers, which do not see this module's preprocessor state.
static const uint kTransmittanceLutWidth = TRANSMITTANCE_LUT_WIDTH;
static const uint kTransmittanceLutHeight = TRANSMITTANCE_LUT_HEIGHT;
static const uint kMultiScatteringLutSize = MULTI_SCATTERING_LUT_SIZE;
static const uint kSkyViewLutWidth = SKY_VIEW_LUT_WIDTH;
static const uint kSkyViewLutHeight = SKY_VIEW_LUT_HEIGHT;
static const uint kMultiScatteringSamples = MULTI_SCATTERING_SAMPLES;
static const float kPlanetRadiusOffset = PLANET_RADIUS_OFFSET;
static const uint kAtmosphereLightingSunTransmittance = ATMOSPHERE_LIGHTING_SUN_TRANSMITTANCE;
static const uint kAtmosphereLightingMultiScattering = ATMOSPHERE_LIGHTING_MULTI_SCATTERING;
static const uint kCloudShapeNoiseSize = CLOUD_SHAPE_NOISE_SIZE;
static const uint kCloudDetailNoiseSize = CLOUD_DETAIL_NOISE_SIZE;
static const uint kCloudWeatherMapSize = CLOUD_WEATHER_MAP_SIZE;
static const uint kCloudShadowMapSize = CLOUD_SHADOW_MAP_SIZE;

// Distance from r0 along rd to the nearest intersection with the sphere at s0 of radius sR, or -1
// if there is none.
float RaySphereIntersectNearest(float3 r0, float3 rd, float3 s0, float sR)
{
    float a = dot(rd, rd);
    float3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sR * sR);
    float delta = b * b - 4.0 * a * c;
    if (delta < 0.0 || a == 0.0)
        return -1.0;

    float sol0 = (-b - sqrt(delta)) / (2.0 * a);
    float sol1 = (-b + sqrt(delta)) / (2.0 * a);
    if (sol0 < 0.0 && sol1 < 0.0)
        return -1.0;
    if (sol0 < 0.0)
        return max(0.0, sol1);
    if (sol1 < 0.0)
        return max(0.0, sol0);
    return max(0.0, min(sol0, sol1));
}

// Moves worldPos to the top of the atmosphere along worldDir if it starts above it. False if the ray
// misses the atmosphere altogether.
bool MoveToTopAtmosphere(inout float3 worldPos, float3 worldDir, float atmosphereTopRadius)
{
    float viewHeight = length(worldPos);
    if (viewHeight > atmosphereTopRadius)
    {
        float tTop = RaySphereIntersectNearest(worldPos, worldDir, float3(0.0, 0.0, 0.0), atmosphereTopRadius);
        if (tTop < 0.0)
            return false;
        float3 upOffset = worldPos / viewHeight * -PLANET_RADIUS_OFFSET;
        worldPos = worldPos + worldDir * tTop + upOffset;
    }
    return true;
}

// viewHeight^2 - bottomRadius^2, factored so that it keeps its precision near the ground, where the
// squares (~4e7 km^2 for the Earth) differ by little more than their float spacing.
float HeightSquaredOverGround(float viewHeight, float bottomRadius)
{
    return (viewHeight - bottomRadius) * (viewHeight + bottomRadius);
}

// Sub-texel mapping that keeps [0, 1] on texel centres, so the LUT ends are sampled exactly.
float FromUnitToSubUvs(float u, float resolution)
{
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

float FromSubUvsToUnit(float u, float resolution)
{
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

// Participating media

struct AtmosphereMedium
{
    float3 scattering;
    float3 extinction;
    float3 scatteringMie;
    float3 scatteringRay;
};

// The medium at height km above the ground.
AtmosphereMedium SampleAtmosphereMedium(AtmosphereParameters atmosphere, float height)
{
    float densityMie = exp(atmosphere.mieDensityExpScale * height);
    float densityRay = exp(atmosphere.rayleighDensityExpScale * height);
    float densityOzo = saturate(height < atmosphere.absorptionDensity0LayerWidth
                                    ? atmosphere.absorptionDensity0LinearTerm * height + atmosphere.absorptionDensity0ConstantTerm
                                    : atmosphere.absorptionDensity1LinearTerm * height + atmosphere.absorptionDensity1ConstantTerm);

    AtmosphereMedium m;
    m.scatteringMie = densityMie * atmosphere.mieScattering;
    m.scatteringRay = densityRay * atmosphere.rayleighScattering;
    m.scattering = m.scatteringMie + m.scatteringRay;
    // Rayleigh scattering does not absorb, ozone absorbs without scattering.
    m.extinction = densityMie * atmosphere.mieExtinction + m.scatteringRay + densityOzo * atmosphere.absorptionExtinction;
    return m;
}

// Phase functions, cosTheta between the light's direction of travel and the scattered one: 1 looking
// straight into the sun.

float RayleighPhase(float cosTheta)
{
    float factor = 3.0 / (16.0 * kAtmospherePi);
    return factor * (1.0 + cosTheta * cosTheta);
}

float CornetteShanksMiePhase(float g, float cosTheta)
{
    float k = 3.0 / (8.0 * kAtmospherePi) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
}

// Transmittance LUT, parameterised as in Bruneton 2017: x by the distance to the top of the
// atmosphere between its minimum and maximum, y by the distance to the horizon.

float2 TransmittanceLutUv(AtmosphereParameters atmosphere, float viewHeight, float viewZenithCosAngle)
{
    float H = sqrt(max(0.0, atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius));
    float rho = sqrt(max(0.0, HeightSquaredOverGround(viewHeight, atmosphere.bottomRadius)));

    float discriminant = viewHeight * viewHeight * (viewZenithCosAngle * viewZenithCosAngle - 1.0) +
                         atmosphere.topRadius * atmosphere.topRadius;
    float d = max(0.0, -viewHeight * viewZenithCosAngle + sqrt(max(0.0, discriminant))); // to the top of the atmosphere

    float dMin = atmosphere.topRadius - viewHeight;
    float dMax = rho + H;
    return float2((d - dMin) / (dMax - dMin), rho / H);
}

void TransmittanceLutParams(AtmosphereParameters atmosphere, float2 uv, out float viewHeight, out float viewZenithCosAngle)
{
    float H = sqrt(atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius);
    float rho = H * uv.y;
    viewHeight = sqrt(rho * rho + atmosphere.bottomRadius * atmosphere.bottomRadius);

    float dMin = atmosphere.topRadius - viewHeight;
    float dMax = rho + H;
    float d = dMin + uv.x * (dMax - dMin);
    viewZenithCosAngle = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * viewHeight * d);
    viewZenithCosAngle = clamp(viewZenithCosAngle, -1.0, 1.0);
}

// Multiple scattering LUT: x by the sun's zenith cosine, y by the height through the atmosphere.
float2 MultiScatteringLutUv(AtmosphereParameters atmosphere, float viewHeight, float sunZenithCosAngle)
{
    float2 uv = saturate(float2(sunZenithCosAngle * 0.5 + 0.5,
                                (viewHeight - atmosphere.bottomRadius) / (atmosphere.topRadius - atmosphere.bottomRadius)));
    return float2(FromUnitToSubUvs(uv.x, MULTI_SCATTERING_LUT_SIZE), FromUnitToSubUvs(uv.y, MULTI_SCATTERING_LUT_SIZE));
}

// Sky-view LUT: x by the view azimuth relative to the sun's, y by the view zenith angle, both
// squashed towards the horizon and the sun where the sky changes fastest. The upper half of the
// LUT holds the sky, the lower half the views that hit the ground.

void SkyViewLutParams(float bottomRadius, float viewHeight, float2 uv, out float viewZenithCosAngle, out float lightViewCosAngle)
{
    // Constrain uvs to valid sub texel range (avoid zenith derivative issue making LUT usage visible)
    uv = float2(FromSubUvsToUnit(uv.x, SKY_VIEW_LUT_WIDTH), FromSubUvsToUnit(uv.y, SKY_VIEW_LUT_HEIGHT));

    float vHorizon = sqrt(max(0.0, HeightSquaredOverGround(viewHeight, bottomRadius)));
    float cosBeta = vHorizon / viewHeight; // ground to horizon cosine
    float beta = acos(cosBeta);
    float zenithHorizonAngle = kAtmospherePi - beta;

    if (uv.y < 0.5)
    {
        float coord = 1.0 - 2.0 * uv.y;
        coord = 1.0 - coord * coord;
        viewZenithCosAngle = cos(zenithHorizonAngle * coord);
    }
    else
    {
        float coord = uv.y * 2.0 - 1.0;
        viewZenithCosAngle = cos(zenithHorizonAngle + beta * coord * coord);
    }

    float coord = uv.x * uv.x;
    // The last column maps to exactly -1, where the view azimuth is opposite the sun's and the ray
    // march's frame would collapse, so stay just inside +-1.
    lightViewCosAngle = clamp(-(coord * 2.0 - 1.0), -0.9999, 0.9999);
}

float2 SkyViewLutUv(float bottomRadius, float viewHeight, bool intersectGround, float viewZenithCosAngle, float lightViewCosAngle)
{
    float vHorizon = sqrt(max(0.0, HeightSquaredOverGround(viewHeight, bottomRadius)));
    float cosBeta = vHorizon / viewHeight; // ground to horizon cosine
    float beta = acos(cosBeta);
    float zenithHorizonAngle = kAtmospherePi - beta;

    float2 uv;
    if (!intersectGround)
    {
        float coord = saturate(acos(viewZenithCosAngle) / zenithHorizonAngle);
        coord = 1.0 - sqrt(1.0 - coord);
        uv.y = coord * 0.5;
    }
    else
    {
        float coord = saturate((acos(viewZenithCosAngle) - zenithHorizonAngle) / beta);
        uv.y = sqrt(coord) * 0.5 + 0.5;
    }
    uv.x = sqrt(saturate(-lightViewCosAngle * 0.5 + 0.5));

    // Constrain uvs to valid sub texel range (avoid zenith derivative issue making LUT usage visible)
    return float2(FromUnitToSubUvs(uv.x, SKY_VIEW_LUT_WIDTH), FromUnitToSubUvs(uv.y, SKY_VIEW_LUT_HEIGHT));
}

// Sky-view LUT coordinates of world direction dir (+Y up) seen from viewHeight km above the planet
// centre, with the sun towards sunDir.
float2 SkyViewLutUvFromDirection(float bottomRadius, float viewHeight, float3 dir, float3 sunDir)
{
    // The azimuth between the view and the sun, in the horizontal plane.
    float2 viewH = dir.xz;
    float2 sunH = sunDir.xz;
    float lengths2 = dot(viewH, viewH) * dot(sunH, sunH);
    float lightViewCosAngle = lengths2 > 1.0e-12 ? clamp(dot(viewH, sunH) * rsqrt(lengths2), -1.0, 1.0) : 1.0;

    bool intersectGround = RaySphereIntersectNearest(float3(0.0, viewHeight, 0.0), dir, float3(0.0, 0.0, 0.0), bottomRadius) >= 0.0;
    return SkyViewLutUv(bottomRadius, viewHeight, intersectGround, dir.y, lightViewCosAngle);
}
