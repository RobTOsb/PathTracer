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
    // mat.geometryNormal turned to the side of the glossy lobe, which for rough glass may be the transmitted side
    // (see MakeLightSelectVertex).
    float3 normal;
    float3 wo; // the view direction whose mirror image about normal is the glossy lobe's axis
    float diffuseWeight; // on normal's side
    float specularWeight;
    float backDiffuseWeight; // on the far side of a transmissive surface
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

// A KHR_lights_punctual point light (intensity in candela, v1 its position and range): I / d^2, inside a
// finite range windowed as the extension recommends, (1 - (d / range)^4)^2 clamped to [0, 1]. A delta
// light, so the PDF is 1 and only light sampling reaches it.
LightSample SamplePointLight(GpuLight light, float3 position)
{
    LightSample ls;
    float3 toLight = light.v1.xyz - position;
    float d2 = max(dot(toLight, toLight), 1.0e-8);
    float d = sqrt(d2);
    float range = light.v1.w;
    float window = 1.0;
    if (range > 0.0)
    {
        float r2 = d2 / (range * range);
        window = saturate(1.0 - r2 * r2);
        window *= window;
    }
    ls.direction = toLight / d;
    ls.distance = d;
    ls.radiance = light.radiance.xyz * (window / d2);
    // Past the range there is nothing to trace a shadow ray for.
    ls.pdf = window > 0.0 ? 1.0 : 0.0;
    ls.lightSamplingOnly = true;
    return ls;
}

// Uniform area sample of an emissive triangle's front face (counter-clockwise v1, v2, v3) as seen
// from position, as a solid angle PDF that excludes the selection probability. Point lights go to
// SamplePointLight.
LightSample SampleBufferLight(GpuLight light, float3 position, float2 u)
{
    if (light.GetLightType() == kLight_Point)
        return SamplePointLight(light, position);

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

// The light tree handles vertices before bounce lightTreeBounces, later ones select by power. From bounce
// lightTreeGlossyBounces on, the specular weight folds into the diffuse one to cheapen traversal.
// A transmissive surface ranks the lights on both of its sides. Its one glossy lobe is whichever of the
// reflection and the rough transmission is the more likely: the transmission as the reflection of a virtual
// view about the flipped normal, so that its axis is the refracted direction. The other lobe counts as
// diffuse on its side. Smooth transmission, which NEE cannot reach, counts on neither.
LightSelectVertex MakeLightSelectVertex(float3 position, OpenPBRBaseParams mat, OpenPBRContext ctx, float3 wo, uint bounce)
{
    LightSelectVertex v;
    v.position = position;
    // Inside a transmissive object the reflection is on the inner side. An opaque surface's frame already faces
    // the viewer, and wo below its shading normal is the normal map's doing.
    bool transmissive = mat.transmissionWeight > 0.0 || HasDiffuseTransmission(mat);
    v.normal = transmissive && dot(wo, mat.geometryNormal) < 0.0 ? -mat.geometryNormal : mat.geometryNormal;
    v.wo = wo;
    // Fuzz is broad enough to count as diffuse. The coat joins the specular lobe, which takes the
    // roughness of whichever of the two reflects more.
    float diffuseTransmitted = ctx.probDiffuse * DiffuseTransmittedShare(mat);
    v.diffuseWeight = ctx.probDiffuse - diffuseTransmitted + ctx.probFuzz;
    v.specularWeight = ctx.probSpecR + ctx.probCoat;
    v.backDiffuseWeight = diffuseTransmitted;
    v.alpha = ctx.probCoat > ctx.probSpecR ? ctx.coatAlpha.xx : ctx.alpha;
    float transmitted = HasDeltaTransmission(mat) || max(ctx.alpha.x, ctx.alpha.y) <= 1.0e-4 ? 0.0 : ctx.probSpecT;
    float3 refracted = mat.geometryThinWalled ? -wo : refract(-wo, v.normal, 1.0 / ctx.eta);
    if (transmitted > v.specularWeight && dot(refracted, refracted) > 0.0)
    {
        v.normal = -v.normal;
        v.wo = 2.0 * dot(refracted, v.normal) * v.normal - refracted;
        float swap = v.diffuseWeight + v.specularWeight;
        v.diffuseWeight = v.backDiffuseWeight;
        v.specularWeight = transmitted;
        v.backDiffuseWeight = swap;
        v.alpha = ctx.alpha;
    }
    else
    {
        v.backDiffuseWeight += transmitted;
    }
    if (bounce >= g_Frame.lightTreeGlossyBounces)
    {
        v.diffuseWeight += dot(v.wo, v.normal) > 0.0 ? v.specularWeight : 0.0;
        v.specularWeight = 0.0;
    }
    v.selection = LightSelection(bounce < g_Frame.lightTreeBounces);
    return v;
}

// A LightSelectVertex as the path carries it to the next hit, half the registers: directions
// octahedral, weights and alpha as halves.
struct PackedLightSelectVertex
{
    float3 position;
    uint normal;
    uint wo;
    uint weights; // half2: diffuseWeight, specularWeight
    uint alpha;   // half2
    uint selection; // low half the selection, high half backDiffuseWeight as a half
};

PackedLightSelectVertex PackLightSelectVertex(LightSelectVertex v)
{
    PackedLightSelectVertex p;
    p.position = v.position;
    p.normal = OctEncode(v.normal);
    p.wo = OctEncode(v.wo);
    p.weights = PackHalf2(float2(v.diffuseWeight, v.specularWeight));
    p.alpha = PackHalf2(v.alpha);
    p.selection = v.selection | (f32tof16(v.backDiffuseWeight) << 16);
    return p;
}

LightSelectVertex UnpackLightSelectVertex(PackedLightSelectVertex p)
{
    LightSelectVertex v;
    v.position = p.position;
    v.normal = OctDecode(p.normal);
    v.wo = OctDecode(p.wo);
    float2 weights = UnpackHalf2(p.weights);
    v.diffuseWeight = weights.x;
    v.specularWeight = weights.y;
    v.alpha = UnpackHalf2(p.alpha);
    v.selection = p.selection & 0xFFFF;
    v.backDiffuseWeight = f16tof32(p.selection >> 16);
    return v;
}

LightTreeShadingContext MakeLightTreeShadingContext(LightSelectVertex v)
{
    return MakeLightTreeShadingContext(v.position, v.normal, v.wo, v.diffuseWeight, v.specularWeight,
                                       v.backDiffuseWeight, v.alpha.x, v.alpha.y, g_Frame.lightTreeDiffuseOnlyAlpha);
}

// Probability of SampleLight at v picking lightIndex from the light buffer, sun selection included.
float LightSelectPmf(LightSelectVertex v, uint lightIndex)
{
    float pmf;
    if (v.selection == LIGHT_SELECT_TREE)
    {
        float mix = g_Frame.lightTreePowerMix;
        pmf = (1.0 - mix) * LightTreePmf(g_LightTreeNodes, g_LightTreeParents, g_LightLeaves, MakeLightTreeShadingContext(v), lightIndex) +
              mix * g_LightPower[lightIndex].pmf;
    }
    else if (v.selection == LIGHT_SELECT_POWER)
        pmf = g_LightPower[lightIndex].pmf;
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

// Power selection: the light owning a point of the flux CDF (see GpuLightPowerEntry) at 31 bits, uSelect giving the
// top 24 and uLow the rest. The guide table's bucket holding it bounds the search to the lights between the owners
// of its start and of the next bucket's. False past the last light's interval, which the quantised flux leaves a
// little short of 2^31.
bool SamplePowerLight(float uSelect, float uLow, out uint lightIndex, out float pmf)
{
    uint target = (min(uint(uSelect * 16777216.0), 16777215u) << 7) | min(uint(uLow * 128.0), 127u);
    uint guideBits = firstbithigh(g_Frame.lightCount);
    uint bucket = target >> (31 - guideBits);
    uint lo = g_LightPower[bucket].guide;
    uint hi = bucket + 1 < (1u << guideBits) ? g_LightPower[bucket + 1].guide : g_Frame.lightCount - 1;
    [loop]
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        if (g_LightPower[mid].cdfEnd > target)
            hi = mid;
        else
            lo = mid + 1;
    }
    GpuLightPowerEntry entry = g_LightPower[lo];
    lightIndex = lo;
    pmf = entry.pmf;
    return entry.cdfEnd > target;
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
        // Defensive sampling: a share of the picks by power, and every pick weighed by the mix of both PMFs (the
        // balance heuristic over the two strategies), which costs the other strategy's PMF: one load after a tree
        // walk, a PMF walk after a power pick.
        float mix = g_Frame.lightTreePowerMix;
        LightTreeShadingContext treeCtx = MakeLightTreeShadingContext(v);
        float treePmf;
        float powerPmf;
        if (uSelect < mix)
        {
            if (!SamplePowerLight(min(uSelect / mix, 0.99999994), u.y, lightIndex, powerPmf))
                return ls;
            treePmf = LightTreePmf(g_LightTreeNodes, g_LightTreeParents, g_LightLeaves, treeCtx, lightIndex);
        }
        else
        {
            float uTree = min((uSelect - mix) / (1.0 - mix), 0.99999994);
            if (!SampleLightTree(g_LightTreeNodes, g_LightTreeLeafLights, treeCtx, uTree, lightIndex, treePmf))
                return ls;
            powerPmf = mix > 0.0 ? g_LightPower[lightIndex].pmf : 0.0;
        }
        pmf = (1.0 - mix) * treePmf + mix * powerPmf;
    }
    else if (v.selection == LIGHT_SELECT_POWER)
    {
        if (!SamplePowerLight(uSelect, u.y, lightIndex, pmf))
            return ls;
    }
    else
    {
        lightIndex = min(uint(uSelect * g_Frame.lightCount), g_Frame.lightCount - 1);
        pmf = 1.0 / g_Frame.lightCount;
    }

    ls = SampleBufferLight(g_Lights[lightIndex], v.position, Rand3(rng).xy);
    ls.radiance *= LightEmissionScale(lightIndex);
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
