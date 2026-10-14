// The sun, the sky and the scene's air (single scattering with volumetric shadows).
import SkyAtmosphere;
import PathTracer.Math;
import PathTracer.Random;
import PathTracer.RayTracing;
import PathTracer.Resources;

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
// single scattering is shadowed by a ray towards the sun, the LUT's multiple scattering is not.
float3 AtmosphereInScattering(float3 origin, float3 dir, float segmentLength, inout uint rng)
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
        inScattered += sunTransmittance * visibility * phaseScattering;
    }
    return SunIlluminance() * inScattered * weight;
}
#endif
