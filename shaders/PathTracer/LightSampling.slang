// Next event estimation: light selection (light tree or uniform), sun and emissive triangle sampling.
import LightTree;
import OpenPBRBaseSubstrate;
import PathTracer.Math;
import PathTracer.Random;
import PathTracer.RayTracing;
import PathTracer.Resources;
import PathTracer.Sky;

struct LightSample
{
    float3 direction; // world space, towards the light
    float distance;
    float3 radiance;
    float pdf; // solid angle PDF
    // Delta light, or one not in the scene geometry (sphere and area lights). BSDF sampling cannot hit
    // either, so light sampling takes the full MIS weight.
    bool lightSamplingOnly;
};

struct LightSelectVertex
{
    float3 position;
    float3 normal;  // the shading frame (mat.geometryNormal), the bitangent is cross(normal, tangent)
    float3 tangent;
    float3 wo;
    float diffuseWeight;
    float specularWeight;
    float2 alpha;
    bool tree; // the light tree picks the light, otherwise uniform selection
};

LightSample SampleSun(float2 u)
{
    LightSample ls;
    ls.distance = FLT_MAX;
    ls.radiance = g_Frame.sunRadiance * SunTransmittance();
    ls.pdf = g_Frame.sunPdf;
    ls.lightSamplingOnly = SunIsDelta();
    ls.direction = SunIsDelta() ? g_Frame.sunDirection : SampleUniformCone(g_Frame.sunDirection, g_Frame.sunOneMinusCosMax, u);
    return ls;
}

// Uniform area sample of an emissive triangle's front face (counter-clockwise v1, v2, v3) as seen
// from position, as a solid angle PDF that excludes the selection probability.
LightSample SampleBufferLight(GpuLight light, float3 position, float2 u)
{
    LightSample ls;
    float3 p0 = light.v1.xyz;
    float3 p1 = light.v2.xyz;
    float3 p2 = light.v3.xyz;
    float su = sqrt(u.x);
    float b0 = 1.0 - su;
    float b1 = u.y * su;
    float b2 = 1.0 - b0 - b1;
    float3 p = b0 * p0 + b1 * p1 + b2 * p2;

    float3 crossE = cross(p1 - p0, p2 - p0);
    float twiceArea = length(crossE);
    float3 toLight = p - position;
    float d2 = max(dot(toLight, toLight), 1.0e-8);
    float d = sqrt(d2);
    ls.direction = toLight / d;
    float cosLight = -dot(crossE, ls.direction) / twiceArea;

    bool front = cosLight > 0.0;
    float3 radiance = light.radiance.xyz;
    uint radianceMap = asuint(light.radiance.w);
    if (front && radianceMap != INVALIDDESCRIPTORHANDLE)
    {
        float2 uv = b0 * UnpackHalf2(light.v1.w) + b1 * UnpackHalf2(light.v2.w) + b2 * UnpackHalf2(light.v3.w);
        radiance *= SampleTexture(radianceMap, uv).rgb;
    }

    ls.radiance = front ? radiance : float3(0.0, 0.0, 0.0);
    ls.pdf = front ? 2.0 * d2 / (twiceArea * cosLight) : 0.0;
    // Stop short of the emitter so it does not shadow itself.
    ls.distance = d * 0.999;
    ls.lightSamplingOnly = false;
    return ls;
}

// The light tree only handles reflecting surfaces, others fall back to uniform selection. From bounce
// lightTreeGlossyBounces on, the specular weight folds into the diffuse one to cheapen traversal.
LightSelectVertex MakeLightSelectVertex(float3 position, OpenPBRBaseParams mat, OpenPBRContext ctx, float3 wo, bool reflectOnly,
                                        uint bounce)
{
    LightSelectVertex v;
    v.position = position;
    v.normal = mat.geometryNormal;
    v.tangent = mat.geometryTangent;
    v.wo = wo;
    v.diffuseWeight = ctx.probDiffuse;
    v.specularWeight = ctx.probSpecR;
    if (bounce >= g_Frame.lightTreeGlossyBounces)
    {
        v.diffuseWeight += dot(wo, v.normal) > 0.0 ? v.specularWeight : 0.0;
        v.specularWeight = 0.0;
    }
    v.alpha = ctx.alpha;
    v.tree = g_Frame.lightSampling == LIGHT_SAMPLING_TREE && reflectOnly;
    return v;
}

LightTreeShadingContext MakeLightTreeShadingContext(LightSelectVertex v)
{
    return MakeLightTreeShadingContext(v.position, v.tangent, cross(v.normal, v.tangent), v.normal, v.wo,
                                       v.diffuseWeight, v.specularWeight, v.alpha.x, v.alpha.y,
                                       g_Frame.lightTreeDiffuseOnlyAlpha, g_Frame.lightTreeFarFieldSharpness);
}

// Probability of SampleLight at v picking lightIndex from the light buffer, sun selection included.
float LightSelectPmf(LightSelectVertex v, uint lightIndex)
{
    float pmf = v.tree ? LightTreePmf(g_LightTreeNodes, g_LightTreeParents, g_LightLeaves, MakeLightTreeShadingContext(v), lightIndex)
                       : 1.0 / g_Frame.lightCount;
    return pmf * (1.0 - g_Frame.sunSelectProbability);
}

// Solid angle PDF of SampleLight at v producing position on the emissive triangle lightIndex,
// reached along dir.
float EmitterLightPdf(LightSelectVertex v, uint lightIndex, float3 dir, float3 position)
{
    GpuLight light = g_Lights[lightIndex];
    float3 crossE = cross(light.v2.xyz - light.v1.xyz, light.v3.xyz - light.v1.xyz);
    float twiceArea = length(crossE);
    float cosLight = -dot(crossE, dir) / twiceArea;
    if (!(cosLight > 0.0))
        return 0.0;

    float3 toLight = position - v.position;
    float d2 = max(dot(toLight, toLight), 1.0e-8);
    return LightSelectPmf(v, lightIndex) * 2.0 * d2 / (twiceArea * cosLight);
}

// Picks the sun with sunSelectProbability, otherwise a light from the light buffer, and samples it.
// The PDF includes the selection probability and is 0 when there is nothing to sample.
LightSample SampleLight(LightSelectVertex v, float2 u, inout uint rng)
{
    float pSun = g_Frame.sunSelectProbability;
    if (u.x < pSun)
    {
        LightSample ls = SampleSun(float2(u.x / pSun, u.y));
        ls.pdf *= pSun;
        return ls;
    }

    LightSample ls = (LightSample)0;
    if (g_Frame.lightCount == 0)
        return ls;

    float uSelect = min((u.x - pSun) / (1.0 - pSun), 0.99999994);
    uint lightIndex;
    float pmf;
    if (v.tree)
    {
        if (!SampleLightTree(g_LightTreeNodes, g_LightTreeLeafLights, MakeLightTreeShadingContext(v), uSelect, lightIndex, pmf))
            return ls;
    }
    else
    {
        lightIndex = min(uint(uSelect * g_Frame.lightCount), g_Frame.lightCount - 1);
        pmf = 1.0 / g_Frame.lightCount;
    }

    ls = SampleBufferLight(g_Lights[lightIndex], v.position, Rand3(rng).xy);
    ls.pdf *= pmf * (1.0 - pSun);
    return ls;
}

// Solid angle PDF of SampleLight producing dir towards the sky, which only the sun disc can.
float LightPdf(float3 origin, float3 dir, HitInfo hit)
{
    if (hit.hit || SunIsDelta())
        return 0.0;
    return InSunDisc(dir) ? g_Frame.sunPdf * g_Frame.sunSelectProbability : 0.0;
}
