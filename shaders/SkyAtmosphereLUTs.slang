// LUT kernels of "A Scalable and Production Ready Sky and Atmosphere Rendering Technique" (Hillaire
// 2020), run by SkyAtmospherePass. Every LUT is rendered for a sun of unit illuminance. Distances are
// in km, with the planet centre at the origin and +Y up.

import SkyAtmosphere;

ConstantBuffer<AtmosphereLUTConstants> g_PushConstants : register(b0);
ConstantBuffer<AtmosphereParameters> g_Atmosphere : register(b1);
Texture2D<float4> g_TransmittanceLUT : register(t0);
Texture2D<float4> g_MultiScatteringLUT : register(t1);
RWTexture2D<float4> g_Output : register(u0);
RWStructuredBuffer<float4> g_SceneLighting : register(u1);
SamplerState g_LinearClamp : register(s2);

float3 SampleTransmittance(float viewHeight, float viewZenithCosAngle)
{
    float2 uv = TransmittanceLutUv(g_Atmosphere, viewHeight, viewZenithCosAngle);
    return g_TransmittanceLUT.SampleLevel(g_LinearClamp, uv, 0).rgb;
}

float3 SampleMultiScattering(float viewHeight, float sunZenithCosAngle)
{
    float2 uv = MultiScatteringLutUv(g_Atmosphere, viewHeight, sunZenithCosAngle);
    return g_MultiScatteringLUT.SampleLevel(g_LinearClamp, uv, 0).rgb;
}

// Optical depth from worldPos along worldDir to the ground or the top of the atmosphere.
float3 IntegrateOpticalDepth(float3 worldPos, float3 worldDir, float sampleCount)
{
    float tMax = DistanceToAtmosphereBoundary(worldPos, worldDir);
    float3 opticalDepth = 0.0;
    float t = 0.0;
    for (float s = 0.0; s < sampleCount; s += 1.0)
    {
        // The exact step length matters for the accuracy of the multiple scattering.
        float newT = tMax * (s + 0.3) / sampleCount;
        float dt = newT - t;
        t = newT;
        float3 P = worldPos + t * worldDir;
        opticalDepth += SampleAtmosphereMedium(g_Atmosphere, length(P) - g_Atmosphere.bottomRadius).extinction * dt;
    }
    return opticalDepth;
}

// Distance to the ground if the ray hits it, otherwise to the top of the atmosphere, 0 if it misses both.
float DistanceToAtmosphereBoundary(float3 worldPos, float3 worldDir)
{
    float3 earthO = float3(0.0, 0.0, 0.0);
    float tBottom = RaySphereIntersectNearest(worldPos, worldDir, earthO, g_Atmosphere.bottomRadius);
    float tTop = RaySphereIntersectNearest(worldPos, worldDir, earthO, g_Atmosphere.topRadius);
    if (tBottom < 0.0)
        return max(tTop, 0.0);
    return tTop > 0.0 ? min(tTop, tBottom) : 0.0;
}

struct SingleScatteringResult
{
    float3 L; // scattered luminance
    float3 transmittance;
    float3 multiScatAs1; // light scattered once along the ray per unit of isotropic radiance, f_ms in the paper
};

// Ray marches the luminance scattered towards worldPos along worldDir from the sun towards sunDir.
// With mieRayPhase the Mie and Rayleigh phase functions apply, otherwise an isotropic one (as for
// the multiple scattering LUT). With multiScattering the multiple scattering LUT is added at each
// step, and with ground the sunlit ground where the ray ends.
SingleScatteringResult IntegrateScatteredLuminance(float3 worldPos, float3 worldDir, float3 sunDir, bool ground,
                                                   float sampleCount, bool mieRayPhase, bool multiScattering)
{
    SingleScatteringResult result = (SingleScatteringResult)0;
    result.transmittance = 1.0;

    float3 earthO = float3(0.0, 0.0, 0.0);
    float tBottom = RaySphereIntersectNearest(worldPos, worldDir, earthO, g_Atmosphere.bottomRadius);
    float tMax = DistanceToAtmosphereBoundary(worldPos, worldDir);
    if (tMax <= 0.0)
        return result;

    const float uniformPhase = 1.0 / (4.0 * kAtmospherePi);
    float cosTheta = dot(sunDir, worldDir);
    float miePhaseValue = CornetteShanksMiePhase(g_Atmosphere.miePhaseG, cosTheta);
    float rayleighPhaseValue = RayleighPhase(cosTheta);

    float3 L = 0.0;
    float3 throughput = 1.0;
    float t = 0.0;
    for (float s = 0.0; s < sampleCount; s += 1.0)
    {
        // The exact step length matters for the accuracy of the multiple scattering.
        float newT = tMax * (s + 0.3) / sampleCount;
        float dt = newT - t;
        t = newT;
        float3 P = worldPos + t * worldDir;

        float pHeight = length(P);
        AtmosphereMedium medium = SampleAtmosphereMedium(g_Atmosphere, pHeight - g_Atmosphere.bottomRadius);
        const float3 sampleTransmittance = exp(-medium.extinction * dt);

        const float3 upVector = P / pHeight;
        float sunZenithCosAngle = dot(sunDir, upVector);
        float3 transmittanceToSun = SampleTransmittance(pHeight, sunZenithCosAngle);

        float3 phaseTimesScattering = mieRayPhase ? medium.scatteringMie * miePhaseValue + medium.scatteringRay * rayleighPhaseValue
                                                  : medium.scattering * uniformPhase;

        // Earth shadow
        float tEarth = RaySphereIntersectNearest(P, sunDir, earthO + kPlanetRadiusOffset * upVector, g_Atmosphere.bottomRadius);
        float earthShadow = tEarth >= 0.0 ? 0.0 : 1.0;

        float3 multiScatteredLuminance = 0.0;
        if (multiScattering)
            multiScatteredLuminance = SampleMultiScattering(pHeight, sunZenithCosAngle);

        float3 S = earthShadow * transmittanceToSun * phaseTimesScattering + multiScatteredLuminance * medium.scattering;

        // Analytic integration over the step, slide 28 of "Physically Based and Unified Volumetric
        // Rendering in Frostbite" (Hillaire 2015). Rayleigh-only vacuum has no extinction, so guard it.
        float3 invExtinction = 1.0 / max(medium.extinction, 1.0e-9);
        result.multiScatAs1 += throughput * (medium.scattering - medium.scattering * sampleTransmittance) * invExtinction;
        L += throughput * (S - S * sampleTransmittance) * invExtinction;
        throughput *= sampleTransmittance;
    }

    if (ground && tBottom > 0.0 && tMax == tBottom)
    {
        // Sunlight bounced off the ground
        float3 P = worldPos + tBottom * worldDir;
        float pHeight = length(P);
        const float3 upVector = P / pHeight;
        float sunZenithCosAngle = dot(sunDir, upVector);
        float3 transmittanceToSun = SampleTransmittance(pHeight, sunZenithCosAngle);
        L += transmittanceToSun * throughput * saturate(sunZenithCosAngle) * g_Atmosphere.groundAlbedo / kAtmospherePi;
    }

    result.L = L;
    result.transmittance = throughput;
    return result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void TransmittanceLUT(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    if (any(dispatchThreadID.xy >= uint2(kTransmittanceLutWidth, kTransmittanceLutHeight)))
        return;

    float2 uv = (float2(dispatchThreadID.xy) + 0.5) / float2(kTransmittanceLutWidth, kTransmittanceLutHeight);
    float viewHeight;
    float viewZenithCosAngle;
    TransmittanceLutParams(g_Atmosphere, uv, viewHeight, viewZenithCosAngle);

    float3 worldPos = float3(0.0, viewHeight, 0.0);
    float3 worldDir = float3(sqrt(1.0 - viewZenithCosAngle * viewZenithCosAngle), viewZenithCosAngle, 0.0);
    float3 transmittance = exp(-IntegrateOpticalDepth(worldPos, worldDir, 40.0));
    g_Output[dispatchThreadID.xy] = float4(transmittance, 1.0);
}

groupshared float3 g_MultiScatAs1Shared[kMultiScatteringSamples];
groupshared float3 g_LShared[kMultiScatteringSamples];

// One group per texel: each thread integrates the second order scattering along one direction of an
// 8 x 8 grid over the sphere, and the group sums them (Section 5.5 of the paper).
[shader("compute")]
[numthreads(1, 1, kMultiScatteringSamples)]
void MultiScatteringLUT(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    uint sampleIndex = groupThreadID.z;
    float2 uv = (float2(groupID.xy) + 0.5) / float(kMultiScatteringLutSize);
    uv = float2(FromSubUvsToUnit(uv.x, kMultiScatteringLutSize), FromSubUvsToUnit(uv.y, kMultiScatteringLutSize));

    float cosSunZenithAngle = uv.x * 2.0 - 1.0;
    float3 sunDir = float3(0.0, cosSunZenithAngle, sqrt(saturate(1.0 - cosSunZenithAngle * cosSunZenithAngle)));
    float viewHeight = g_Atmosphere.bottomRadius +
                       saturate(uv.y + kPlanetRadiusOffset) * (g_Atmosphere.topRadius - g_Atmosphere.bottomRadius - kPlanetRadiusOffset);
    float3 worldPos = float3(0.0, viewHeight, 0.0);

    const float sphereSolidAngle = 4.0 * kAtmospherePi;
    const float isotropicPhase = 1.0 / sphereSolidAngle;
    const uint sqrtSampleCount = 8;
    {
        float randA = (0.5 + float(sampleIndex / sqrtSampleCount)) / float(sqrtSampleCount);
        float randB = (0.5 + float(sampleIndex % sqrtSampleCount)) / float(sqrtSampleCount);
        float theta = 2.0 * kAtmospherePi * randA;
        float phi = acos(1.0 - 2.0 * randB);
        float3 worldDir = float3(cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi));

        SingleScatteringResult result = IntegrateScatteredLuminance(worldPos, worldDir, sunDir, true, 20.0, false, false);
        float sampleSolidAngle = sphereSolidAngle / float(kMultiScatteringSamples);
        g_MultiScatAs1Shared[sampleIndex] = result.multiScatAs1 * sampleSolidAngle;
        g_LShared[sampleIndex] = result.L * sampleSolidAngle;
    }

    for (uint stride = kMultiScatteringSamples / 2; stride > 0; stride >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (sampleIndex < stride)
        {
            g_MultiScatAs1Shared[sampleIndex] += g_MultiScatAs1Shared[sampleIndex + stride];
            g_LShared[sampleIndex] += g_LShared[sampleIndex + stride];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (sampleIndex > 0)
        return;

    float3 multiScatAs1 = g_MultiScatAs1Shared[0] * isotropicPhase;
    float3 inScatteredLuminance = g_LShared[0] * isotropicPhase;

    // Every further order of scattering is the last one times f_ms, so their sum is the geometric
    // series 1 / (1 - f_ms) (equation 9).
    float3 L = inScatteredLuminance / (1.0 - multiScatAs1);
    g_Output[groupID.xy] = float4(g_Atmosphere.multipleScatteringFactor * L, 1.0);
}

// The sky seen from g_PushConstants.viewHeight, without the sun disc, in the parameterisation of
// SkyViewLutParams. The sun lies in the XY plane at sunZenithCos.
[shader("compute")]
[numthreads(8, 8, 1)]
void SkyViewLUT(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    if (any(dispatchThreadID.xy >= uint2(kSkyViewLutWidth, kSkyViewLutHeight)))
        return;

    float2 uv = (float2(dispatchThreadID.xy) + 0.5) / float2(kSkyViewLutWidth, kSkyViewLutHeight);
    float viewHeight = g_PushConstants.viewHeight;

    float viewZenithCosAngle;
    float lightViewCosAngle;
    SkyViewLutParams(g_Atmosphere.bottomRadius, viewHeight, uv, viewZenithCosAngle, lightViewCosAngle);

    float sunZenithCosAngle = g_PushConstants.sunZenithCos;
    float3 sunDir = float3(sqrt(saturate(1.0 - sunZenithCosAngle * sunZenithCosAngle)), sunZenithCosAngle, 0.0);

    float viewZenithSinAngle = sqrt(saturate(1.0 - viewZenithCosAngle * viewZenithCosAngle));
    float3 worldDir = float3(viewZenithSinAngle * lightViewCosAngle,
                             viewZenithCosAngle,
                             viewZenithSinAngle * sqrt(saturate(1.0 - lightViewCosAngle * lightViewCosAngle)));
    float3 worldPos = float3(0.0, viewHeight, 0.0);

    float3 L = 0.0;
    if (MoveToTopAtmosphere(worldPos, worldDir, g_Atmosphere.topRadius))
    {
        // The path tracer's environment, so unlike the paper's this LUT includes the sunlit ground
        // below the horizon.
        L = IntegrateScatteredLuminance(worldPos, worldDir, sunDir, true, 30.0, true, true).L;
    }
    g_Output[dispatchThreadID.xy] = float4(L, 1.0);
}

// One thread: the sun's transmittance to the scene (zero once the planet hides it) and the multiple
// scattering there, which the path tracer treats as constant over the scene.
[shader("compute")]
[numthreads(1, 1, 1)]
void SceneLighting()
{
    float viewHeight = g_PushConstants.viewHeight;
    float sunZenithCosAngle = g_PushConstants.sunZenithCos;
    float3 sunDir = float3(sqrt(saturate(1.0 - sunZenithCosAngle * sunZenithCosAngle)), sunZenithCosAngle, 0.0);
    float3 worldPos = float3(0.0, viewHeight, 0.0);

    bool earthShadow = RaySphereIntersectNearest(worldPos, sunDir, float3(0.0, 0.0, 0.0), g_Atmosphere.bottomRadius) >= 0.0;
    float3 sunTransmittance = earthShadow ? 0.0 : SampleTransmittance(viewHeight, sunZenithCosAngle);

    g_SceneLighting[kAtmosphereLightingSunTransmittance] = float4(sunTransmittance, 0.0);
    g_SceneLighting[kAtmosphereLightingMultiScattering] = float4(SampleMultiScattering(viewHeight, sunZenithCosAngle), 0.0);
}
