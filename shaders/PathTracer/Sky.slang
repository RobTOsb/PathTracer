// The sun, the sky, the clouds and the scene's air (single scattering with volumetric shadows).
import Clouds;
import SkyAtmosphere;
import PathTracer.Math;
import PathTracer.Random;
import PathTracer.RayTracing;
import PathTracer.Resources;
import PathTracer.Shadows;

bool SunIsDelta()
{
    return g_Frame.sunIsDelta != 0;
}

// True when dir falls inside the sun disc.
bool InSunDisc(float3 dir)
{
    return 1.0 - dot(dir, g_Frame.sunDirection) <= g_Frame.sunOneMinusCosMax;
}

// Of the sunlight through the atmosphere down to the scene, zero once the planet hides the sun.
float3 SunTransmittance()
{
#ifdef ATMOSPHERE
    return g_AtmosphereLighting[kAtmosphereLightingSunTransmittance].rgb;
#else
    return float3(1.0, 1.0, 1.0);
#endif
}

// Irradiance of the sun at the top of the atmosphere: sunRadiance is the disc's radiance over its
// 1 / sunPdf of solid angle, or the irradiance itself with a pdf of 1 for a delta sun.
float3 SunIlluminance()
{
    return g_Frame.sunRadiance / g_Frame.sunPdf;
}

// Radiance of the sun disc seen along dir, zero outside it.
float3 SunRadiance(float3 dir)
{
    return !SunIsDelta() && InSunDisc(dir) ? g_Frame.sunRadiance * SunTransmittance() : float3(0.0, 0.0, 0.0);
}

// The sky without the sun disc, which is handled separately so it can be MIS weighted: a constant
// colour, or the atmosphere lit by the sun.
float3 EnvironmentRadiance(float3 dir)
{
#ifdef ATMOSPHERE
    float2 uv = SkyViewLutUvFromDirection(g_Frame.atmosphereBottomRadius, g_Frame.atmosphereViewHeight, dir, g_Frame.sunDirection);
    return SunIlluminance() * g_SkyViewLUT.SampleLevel(g_LinearClamp, uv, 0.0).rgb;
#else
    return g_Frame.skyRadiance;
#endif
}

#ifdef ATMOSPHERE
CloudLayer SceneClouds()
{
    CloudLayer layer = { g_Frame.clouds, g_LinearWrap, g_LinearClamp };
    return layer;
}
#endif

// The share of the sunlight that the clouds let through to a scene position, from their shadow map.
float CloudShadow(float3 position)
{
#ifdef ATMOSPHERE
    if (g_Frame.clouds.enabled == 0)
        return 1.0;
    Texture2D<float> shadowMap = ResourceDescriptorHeap[g_Frame.clouds.shadowMap];
    return shadowMap.SampleLevel(g_LinearClamp, CloudShadowMapUv(g_Frame.clouds, position), 0.0);
#else
    return 1.0;
#endif
}

// Solid angle PDF of SampleSkyImportance producing dir: its texel's share of the sky importance pyramid's total over the
// texel's 4 pi / texels of solid angle. 0 without the pyramid or with a black sky.
float SkyImportancePdf(float3 dir)
{
#ifdef ATMOSPHERE
    if (g_Frame.skyImportance == INVALIDDESCRIPTORHANDLE)
        return 0.0;
    RWStructuredBuffer<float> pyramid = ResourceDescriptorHeap[g_Frame.skyImportance];
    float total = pyramid[kSkyImportanceCount - 1u];
    if (!(total > 0.0))
        return 0.0;
    uint2 texel = min(uint2(SkyImportanceUv(dir) * float(kSkyImportanceSize)), kSkyImportanceSize - 1u);
    return pyramid[SkyImportanceIndex(0u, texel)] / total *
           (float(kSkyImportanceSize * kSkyImportanceSize) / (4.0 * kAtmospherePi));
#else
    return 0.0;
#endif
}

// The sky's luminance irradiance on a surface open to it, roughly: the pyramid's integral over the sphere
// (total * 4 pi / texels) times a quarter, which an even sky's cosine-weighted hemisphere would give. 0 without the
// pyramid.
float SkyIrradianceEstimate()
{
#ifdef ATMOSPHERE
    if (g_Frame.skyImportance == INVALIDDESCRIPTORHANDLE)
        return 0.0;
    RWStructuredBuffer<float> pyramid = ResourceDescriptorHeap[g_Frame.skyImportance];
    return max(pyramid[kSkyImportanceCount - 1u], 0.0) * (kAtmospherePi / float(kSkyImportanceSize * kSkyImportanceSize));
#else
    return 0.0;
#endif
}

// A direction towards the sky in proportion to its luminance without the sun disc (hierarchical sample warping):
// from the pyramid's top, u picks one of the 2 x 2 children of the texel so far by their sums, first the column and
// then the row, and is stretched back over [0, 1) each time, down to a texel that it then places the direction in.
// False without the pyramid or with a black sky.
bool SampleSkyImportance(float2 u, out float3 dir, out float pdf)
{
    dir = float3(0.0, 1.0, 0.0);
    pdf = 0.0;
#ifdef ATMOSPHERE
    if (g_Frame.skyImportance == INVALIDDESCRIPTORHANDLE)
        return false;
    RWStructuredBuffer<float> pyramid = ResourceDescriptorHeap[g_Frame.skyImportance];
    float total = pyramid[kSkyImportanceCount - 1u];
    if (!(total > 0.0))
        return false;
    uint2 texel = uint2(0u, 0u);
    [loop]
    for (int level = int(kSkyImportanceLevels) - 2; level >= 0; --level)
    {
        texel *= 2u;
        uint corner = SkyImportanceIndex(uint(level), texel);
        uint width = kSkyImportanceSize >> uint(level);
        float w00 = pyramid[corner];
        float w10 = pyramid[corner + 1u];
        float w01 = pyramid[corner + width];
        float w11 = pyramid[corner + width + 1u];
        float left = w00 + w01;
        float right = w10 + w11;
        float pLeft = left / max(left + right, 1.0e-30);
        float pTop;
        if (u.x < pLeft)
        {
            u.x /= pLeft;
            pTop = w00 / max(left, 1.0e-30);
        }
        else
        {
            u.x = (u.x - pLeft) / (1.0 - pLeft);
            texel.x += 1u;
            pTop = w10 / max(right, 1.0e-30);
        }
        if (u.y < pTop)
        {
            u.y /= pTop;
        }
        else
        {
            u.y = (u.y - pTop) / (1.0 - pTop);
            texel.y += 1u;
        }
        u = min(u, 0.99999994);
    }
    float w = pyramid[SkyImportanceIndex(0u, texel)];
    if (!(w > 0.0))
        return false;
    dir = normalize(SkyImportanceDirection((float2(texel) + u) / float(kSkyImportanceSize)));
    pdf = w / total * (float(kSkyImportanceSize * kSkyImportanceSize) / (4.0 * kAtmospherePi));
    return true;
#else
    return false;
#endif
}

// Radiance from beyond the scene along a ray from origin that leaves it: the sky times skyWeight (its MIS weight
// against the sky's light sampling) and the light of the sun disc already weighted for MIS (sun), with the clouds in
// front. Rays the camera sees through (view) march the clouds in
// more steps; rays whose sky the camera sees, also in a mirror-like surface (seen), take the clouds' detail noise.
// The others read the cloud panorama, the cloudy sky as the camera sees it, where there is one: their light only
// reaches the camera after a scattering that blurs it, which hides that it was seen from elsewhere. The reference
// marches every ray.
float3 SkyRadiance(float3 origin, float3 dir, float3 sun, float skyWeight, bool view, bool seen, inout uint rng)
{
#ifdef ATMOSPHERE
    CloudParameters c = g_Frame.clouds;
    if (c.enabled != 0)
    {
#ifndef REFERENCE
        if (!view && !seen && c.skyPanorama != INVALIDDESCRIPTORHANDLE)
        {
            Texture2D<float4> panorama = ResourceDescriptorHeap[c.skyPanorama];
            float4 cloudy = panorama.SampleLevel(g_LinearClamp, CloudPanoramaUv(dir), 0.0);
            return skyWeight * SunIlluminance() * cloudy.rgb + cloudy.a * sun;
        }
#endif
        float4 cloudy = CloudySky(SceneClouds(), EnvironmentRadiance(dir), CloudSpacePosition(c, origin), dir,
                                  view ? c.viewSteps : c.indirectSteps, SunIlluminance(),
                                  EnvironmentRadiance(float3(0.0, 1.0, 0.0)), Rand3(rng), view || seen);
        return skyWeight * cloudy.rgb + cloudy.a * sun;
    }
#endif
    return skyWeight * EnvironmentRadiance(dir) + sun;
}

// Transmittance of the scene's air over distance (capped at the distance to the sky), 1 without the
// atmosphere.
float3 AtmosphereTransmittance(float distance)
{
#ifdef ATMOSPHERE
    return exp(-g_Frame.atmosphereExtinction * min(distance, g_Frame.atmosphereMissDistance));
#else
    return float3(1.0, 1.0, 1.0);
#endif
}

#ifdef ATMOSPHERE
// Sunlight the air scatters towards origin along a segment, estimated at one distance along it. The
// single scattering is shadowed by a ray towards the sun, the LUT's multiple scattering, where view (a segment the
// camera looks along), by one in a random direction out to atmosphereSkyVisibilityDistance.
float3 AtmosphereInScattering(float3 origin, float3 dir, float segmentLength, bool view, inout uint rng)
{
    float3 sigmaT = g_Frame.atmosphereExtinction;
    float sigmaMean = (sigmaT.x + sigmaT.y + sigmaT.z) * (1.0 / 3.0);
    float tauMean = sigmaMean * segmentLength;
    if (!(tauMean > 0.0))
        return float3(0.0, 0.0, 0.0);

    // s follows the mean transmittance normalised over the segment, or is uniform where that is
    // inaccurate in float. weight is the transmittance to s over its pdf.
    float3 u = Rand3(rng);
    float s;
    float3 weight;
    if (tauMean < 1.0e-3)
    {
        s = u.x * segmentLength;
        weight = exp(-sigmaT * s) * segmentLength;
    }
    else
    {
        float norm = 1.0 - exp(-tauMean);
        s = -log(1.0 - u.x * norm) / sigmaMean;
        weight = exp((sigmaMean - sigmaT) * s) * (norm / sigmaMean);
    }

    float3 scattering = g_Frame.atmosphereRayleighScattering + g_Frame.atmosphereMieScattering;
    float3 inScattered = g_AtmosphereLighting[kAtmosphereLightingMultiScattering].rgb * scattering;
    if (view && g_Frame.atmosphereSkyVisibility != 0)
    {
        // The LUT's multiple scattering arrives from the whole of an open sky, so only the fraction of
        // directions that reach it lights the point, estimated by one uniform direction on the sphere
        // (a cone of 1 - cos = 2).
        float2 v = Rand3(rng).xy;
        inScattered *= TraceShadowRay(origin + dir * s, SampleUniformCone(float3(0.0, 1.0, 0.0), 2.0, v),
                                      g_Frame.atmosphereSkyVisibilityDistance);
    }
    float3 sunTransmittance = SunTransmittance();
    if (any(sunTransmittance > 0.0))
    {
        float cosTheta = dot(dir, g_Frame.sunDirection);
        float3 phaseScattering = g_Frame.atmosphereRayleighScattering * RayleighPhase(cosTheta) +
                                 g_Frame.atmosphereMieScattering * CornetteShanksMiePhase(g_Frame.atmosphereMiePhaseG, cosTheta);
        float3 visibility = float3(1.0, 1.0, 1.0);
        if (g_Frame.atmosphereVolumetricShadows != 0)
        {
            // Towards a point on the disc, so the light shafts get the sun's penumbrae.
            float3 sunDir = SunIsDelta() ? g_Frame.sunDirection : SampleUniformCone(g_Frame.sunDirection, g_Frame.sunOneMinusCosMax, u.yz);
            visibility = TraceShadowRay(origin + dir * s, sunDir, FLT_MAX);
        }
        inScattered += sunTransmittance * visibility * CloudShadow(origin + dir * s) * phaseScattering;
    }
    return SunIlluminance() * inScattered * weight;
}
#endif
