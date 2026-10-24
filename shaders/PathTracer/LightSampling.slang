// Next event estimation: light selection (light tree or uniform), sun and emissive triangle sampling.
import LightTree;
import OpenPBRBSDF;
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
    uint selection; // LIGHT_SELECT_*, how the light is picked from the light buffer
};

static const uint LIGHT_SELECT_UNIFORM = 0;
static const uint LIGHT_SELECT_TREE = 1;
static const uint LIGHT_SELECT_POWER = 2; // in proportion to flux, g_LightPower

// The selection for a vertex the light tree may serve (treeAllowed) or not, under the frame's sampling mode.
uint LightSelection(bool treeAllowed)
{
    if (g_Frame.lightSampling == LIGHT_SAMPLING_UNIFORM)
        return LIGHT_SELECT_UNIFORM;
    return g_Frame.lightSampling == LIGHT_SAMPLING_TREE && treeAllowed ? LIGHT_SELECT_TREE : LIGHT_SELECT_POWER;
}

// Sunlight reaching position, through the clouds' shadow.
LightSample SampleSun(float3 position, float2 u)
{
    LightSample ls;
    ls.distance = FLT_MAX;
    ls.radiance = g_Frame.sunRadiance * SunTransmittance() * CloudShadow(position);
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

// The light tree only handles reflecting surfaces before bounce lightTreeBounces, others select by power.
// From bounce lightTreeGlossyBounces on, the specular weight folds into the diffuse one to cheapen traversal.
LightSelectVertex MakeLightSelectVertex(float3 position, OpenPBRBaseParams mat, OpenPBRContext ctx, float3 wo, bool reflectOnly,
                                        uint bounce)
{
    LightSelectVertex v;
    v.position = position;
    v.normal = mat.geometryNormal;
    v.tangent = mat.geometryTangent;
    v.wo = wo;
    // Fuzz is broad enough to count as diffuse. The coat joins the specular lobe, which takes the
    // roughness of whichever of the two reflects more.
    v.diffuseWeight = ctx.probDiffuse + ctx.probFuzz;
    v.specularWeight = ctx.probSpecR + ctx.probCoat;
    if (bounce >= g_Frame.lightTreeGlossyBounces)
    {
        v.diffuseWeight += dot(wo, v.normal) > 0.0 ? v.specularWeight : 0.0;
        v.specularWeight = 0.0;
    }
    v.alpha = ctx.probCoat > ctx.probSpecR ? ctx.coatAlpha.xx : ctx.alpha;
    v.selection = LightSelection(reflectOnly && bounce < g_Frame.lightTreeBounces);
    return v;
}

// A LightSelectVertex as the path carries it to the next hit, half the registers: directions
// octahedral, weights and alpha as halves.
struct PackedLightSelectVertex
{
    float3 position;
    uint normal;
    uint tangent;
    uint wo;
    uint weights; // half2: diffuseWeight, specularWeight
    uint alpha;   // half2
    uint selection;
};

PackedLightSelectVertex PackLightSelectVertex(LightSelectVertex v)
{
    PackedLightSelectVertex p;
    p.position = v.position;
    p.normal = OctEncode(v.normal);
    p.tangent = OctEncode(v.tangent);
    p.wo = OctEncode(v.wo);
    p.weights = PackHalf2(float2(v.diffuseWeight, v.specularWeight));
    p.alpha = PackHalf2(v.alpha);
    p.selection = v.selection;
    return p;
}

LightSelectVertex UnpackLightSelectVertex(PackedLightSelectVertex p)
{
    LightSelectVertex v;
    v.position = p.position;
    v.normal = OctDecode(p.normal);
    v.tangent = OctDecode(p.tangent);
    v.wo = OctDecode(p.wo);
    float2 weights = UnpackHalf2(p.weights);
    v.diffuseWeight = weights.x;
    v.specularWeight = weights.y;
    v.alpha = UnpackHalf2(p.alpha);
    v.selection = p.selection;
    return v;
}

LightTreeShadingContext MakeLightTreeShadingContext(LightSelectVertex v)
{
    return MakeLightTreeShadingContext(v.position, v.tangent, cross(v.normal, v.tangent), v.normal, v.wo,
                                       v.diffuseWeight, v.specularWeight, v.alpha.x, v.alpha.y,
                                       g_Frame.lightTreeDiffuseOnlyAlpha, g_Frame.lightTreeFarFieldSharpness);
}

// Share of the flux of the part of the light buffer (static or dynamic emitters) that lightIndex is in.
float LightPartPowerShare(uint lightIndex)
{
    return lightIndex < g_Frame.staticLightCount ? g_Frame.staticLightPowerShare : 1.0 - g_Frame.staticLightPowerShare;
}

// Probability of SampleLight at v picking lightIndex from the light buffer, sun selection included.
float LightSelectPmf(LightSelectVertex v, uint lightIndex)
{
    float pmf;
    if (v.selection == LIGHT_SELECT_TREE)
        pmf = LightTreePmf(g_LightTreeNodes, g_LightTreeParents, g_LightLeaves, MakeLightTreeShadingContext(v), lightIndex);
    else if (v.selection == LIGHT_SELECT_POWER)
        pmf = g_LightPower[lightIndex].pmf * LightPartPowerShare(lightIndex);
    else
        pmf = 1.0 / g_Frame.lightCount;
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
        LightSample ls = SampleSun(v.position, float2(u.x / pSun, u.y));
        ls.pdf *= pSun;
        return ls;
    }

    LightSample ls = (LightSample)0;
    if (g_Frame.lightCount == 0)
        return ls;

    float uSelect = min((u.x - pSun) / (1.0 - pSun), 0.99999994);
    uint lightIndex;
    float pmf;
    if (v.selection == LIGHT_SELECT_TREE)
    {
        if (!SampleLightTree(g_LightTreeNodes, g_LightTreeLeafLights, MakeLightTreeShadingContext(v), uSelect, lightIndex, pmf))
            return ls;
    }
    else if (v.selection == LIGHT_SELECT_POWER)
    {
        // The static or the dynamic part by its share of the flux, then an entry of the part's alias
        // table. u.y, which only the sun uses, decides between the entry's light and its alias.
        float staticShare = g_Frame.staticLightPowerShare;
        bool inStatic = uSelect < staticShare;
        uint first = inStatic ? 0 : g_Frame.staticLightCount;
        uint count = inStatic ? g_Frame.staticLightCount : g_Frame.lightCount - g_Frame.staticLightCount;
        float share = inStatic ? staticShare : 1.0 - staticShare;
        float uPart = min((inStatic ? uSelect : uSelect - staticShare) / share, 0.99999994);
        uint entry = first + min(uint(uPart * count), count - 1);
        GpuLightAliasEntry alias = g_LightPower[entry];
        bool keep = u.y < alias.threshold;
        lightIndex = keep ? entry : alias.alias;
        pmf = (keep ? alias.pmf : alias.aliasPmf) * share;
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
