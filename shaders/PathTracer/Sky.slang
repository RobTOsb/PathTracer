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

// Radiance from beyond the scene along a ray from origin that leaves it: the sky and the light of the sun disc
// already weighted for MIS (sun), with the clouds in front. Rays the camera sees through (view) march the clouds in
// more steps.
float3 SkyRadiance(float3 origin, float3 dir, float3 sun, bool view, inout uint rng)
{
    float3 sky = EnvironmentRadiance(dir);
#ifdef ATMOSPHERE
    CloudParameters c = g_Frame.clouds;
    if (c.enabled != 0)
    {
        float3 pos = CloudSpacePosition(c, origin);
        CloudSegment cloud = MarchClouds(SceneClouds(), pos, dir, view ? c.viewSteps : c.indirectSteps, SunIlluminance(),
                                         EnvironmentRadiance(float3(0.0, 1.0, 0.0)), Rand3(rng));
        if (cloud.transmittance < 1.0)
        {
            // The air between the origin and the cloud dims it, and scatters in the part of the sky's light that
            // its transmittance says comes from in front of the cloud, which the cloud does not hide.
            AtmosphereParameters atmosphere = (AtmosphereParameters)0;
            atmosphere.bottomRadius = c.planetRadius;
            atmosphere.topRadius = c.atmosphereTopRadius;
            Texture2D<float4> transmittanceLut = ResourceDescriptorHeap[c.transmittanceLut];
            float3 cloudPos = pos + dir * cloud.distance;
            float r = length(pos);
            float cloudR = length(cloudPos);
            float3 toTop = transmittanceLut.SampleLevel(g_LinearClamp, TransmittanceLutUv(atmosphere, r, dot(dir, pos) / r), 0.0).rgb;
            float3 cloudToTop =
                transmittanceLut.SampleLevel(g_LinearClamp, TransmittanceLutUv(atmosphere, cloudR, dot(dir, cloudPos) / cloudR), 0.0).rgb;
            float3 air = saturate(toTop / max(cloudToTop, 1.0e-6));
            float3 front = saturate((1.0 - air) / max(1.0 - toTop, 1.0e-4));
            return sky * (front + cloud.transmittance * (1.0 - front)) + air * cloud.inScattered + cloud.transmittance * sun;
        }
    }
#endif
    return sky + sun;
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
// single scattering is shadowed by a ray towards the sun, the LUT's multiple scattering by one in a
// random direction.
float3 AtmosphereInScattering(float3 origin, float3 dir, float segmentLength, uint shadowRayFlags, inout uint rng)
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
    if (g_Frame.atmosphereSkyVisibility != 0)
    {
        // The LUT's multiple scattering arrives from the whole of an open sky, so only the fraction of
        // directions that reach it lights the point, estimated by one uniform direction on the sphere
        // (a cone of 1 - cos = 2).
        float2 v = Rand3(rng).xy;
        inScattered *= TraceShadowRay(origin + dir * s, SampleUniformCone(float3(0.0, 1.0, 0.0), 2.0, v), FLT_MAX, shadowRayFlags);
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
            visibility = TraceShadowRay(origin + dir * s, sunDir, FLT_MAX, shadowRayFlags);
        }
        inScattered += sunTransmittance * visibility * CloudShadow(origin + dir * s) * phaseScattering;
    }
    return SunIlluminance() * inScattered * weight;
}
#endif
