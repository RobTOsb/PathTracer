#include "PathTracerInterop.h"
#include "DLSSUtil.h"

import OpenPBRBaseSubstrate;
import LightTree;
import SkyAtmosphere;

// Resources

ConstantBuffer<FrameConstants> g_Frame : register(b0);
RaytracingAccelerationStructure g_Scene : register(t0);
RWTexture2D<float4> g_Output : register(u0);
SamplerState g_LinearWrap : register(s3);
SamplerState g_AnisotropicWrap : register(s5);

// DLSS Ray Reconstruction guide buffers, all at render resolution. PathTracerPass compiles this
// permutation (and binds these) only while DLSS-RR is enabled.
#ifdef DLSS_RR
RWTexture2D<float4> g_DiffuseAlbedo : register(u1);
RWTexture2D<float4> g_SpecularAlbedo : register(u2);
RWTexture2D<float4> g_NormalRoughness : register(u3); // world-space shading normal, .w = linear roughness
RWTexture2D<float2> g_MotionVectors : register(u4);   // render pixels, current to previous
RWTexture2D<float> g_Depth : register(u5);            // hardware depth, reversed Z
RWTexture2D<float> g_SpecularHitDistance : register(u6); // kept across frames for the hold, see InitialSpecularHitDistance
RWTexture2D<float2> g_SpecularMotionVectors : register(u7); // render pixels, current to previous, of the reflection
#endif

struct PrimitiveDesc
{
    uint meshletOffset;
    uint meshletVerticesOffset;
    uint meshletTrianglesOffset;
    uint meshletCount;
    uint materialIndex;
    uint vertexOffset;
    uint vertexCount;
    uint indexOffset;
    uint indexCount;
};

StructuredBuffer<PrimitiveDesc> g_Primitives : register(t1);
StructuredBuffer<uint> g_Indices : register(t2);
StructuredBuffer<GpuVertex> g_Vertices : register(t3);
StructuredBuffer<GpuMaterial> g_Materials : register(t4);
StructuredBuffer<GpuLight> g_Lights : register(t5);
StructuredBuffer<GpuLightTreeNode> g_LightTreeNodes : register(t6);
StructuredBuffer<GpuLightTreeLeafEntry> g_LightLeaves : register(t7);
StructuredBuffer<uint> g_Emitters : register(t8);
StructuredBuffer<uint> g_LightTreeParents : register(t9);
StructuredBuffer<GpuLightTreeLeafEntry> g_LightTreeLeafLights : register(t10);

#ifdef ATMOSPHERE
Texture2D<float4> g_SkyViewLUT : register(t11);
StructuredBuffer<float4> g_AtmosphereLighting : register(t12);
SamplerState g_LinearClamp : register(s2);
#endif

PrimitiveDesc LoadPrimitive(uint index)
{
    return g_Primitives[index];
}

GpuMaterial LoadGpuMaterial(uint index)
{
    return g_Materials[index];
}

float4 SampleTexture(uint textureIndex, float2 uv)
{
    Texture2D<float4> texture = ResourceDescriptorHeap[NonUniformResourceIndex(textureIndex)];
    return texture.SampleLevel(g_LinearWrap, uv, 0.0);
}

float4 SampleTextureGrad(uint textureIndex, float2 uv, float2 ddx, float2 ddy, bool anisotropic)
{
    Texture2D<float4> texture = ResourceDescriptorHeap[NonUniformResourceIndex(textureIndex)];
    if (anisotropic)
        return texture.SampleGrad(g_AnisotropicWrap, uv, ddx, ddy);
    return texture.SampleGrad(g_LinearWrap, uv, ddx, ddy);
}

// A ray's cone: its width where it last met a surface (0 at a pinhole camera) and its full apex
// angle in radians, which grows the width by spread per unit of distance.
struct RayCone
{
    float width;
    float spread;
};

// Apex angle of a narrow cone (solid angle about pi (angle / 2)^2) covering the 1 / pdf of solid
// angle that a BSDF sample stands for.
float SampledConeSpread(float pdf)
{
    return 2.0 * rsqrt(PI * pdf);
}

// Scene interface

struct HitInfo
{
    bool hit;
    float t;
    float3 position;
    float3 Ng; // geometric normal, oriented outwards
    float3 T;  // interpolated vertex tangent
    float3 B;  // interpolated vertex bitangent
    float3 N;  // interpolated vertex normal
    float3 Ns; // shading normal from the normal map in world space, equal to N without a map
    float2 uv;
    float4 uvGradients; // uv across the two ellipse axes of a ray cone of unit width, see RayConeUvGradients
    float3 emission; // emitted radiance (both faces), set by LoadMaterial
    uint materialId;
    uint emitter; // entry of the hit triangle in g_Emitters, INVALID_LIGHT_INDEX if its instance does not emit
    float curvature; // radians the vertex normals turn per unit length, DLSS_RR permutation only
    float meanCurvature; // signed, positive where the surface bulges towards N, DLSS_RR permutation only
};

struct LightSample
{
    float3 direction; // world space, towards the light
    float distance;
    float3 radiance;
    float pdf; // solid angle PDF
    // Delta light, or one that is not in the scene geometry (sphere and area lights). BSDF sampling cannot
    // hit either, so light sampling takes the full MIS weight.
    bool lightSamplingOnly;
};

// Ray types. Each one indexes both the hit group (every TLAS instance contributes 0 and the
// geometry multiplier is 0) and the miss shader in the shader table, so PathTracerPass must add
// its records in this order.
static const uint STANDARD_RAY_INDEX = 0;
static const uint SHADOW_RAY_INDEX = 1;

struct ShadowRayPayload
{
    float3 visibility;
};

struct HitPayload
{
    float hitT;
    float3 Ng;
    uint materialId; // on the way in, the seed AnyHit draws its blend decisions from
    float3 N;
    float tangentSign;
    float3 T;
    float2 uv;
    uint emitter;
    float4 uvGradients;
    uint curvature; // half2: HitInfo::curvature, HitInfo::meanCurvature
};

// First bounce whose hits the SER pipeline reorders. Camera rays leave raygen in screen order, so
// their hits are already about as coherent as reordering would make them.
static const uint SER_START_BOUNCE = 1;

HitInfo TraceClosest(float3 origin, float3 dir, bool reorder, inout uint rng)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = FLT_MAX;

    HitPayload payload = (HitPayload)0;
    payload.hitT = -1.0;
    payload.materialId = rng++;
    // Masked and blended geometry runs AnyHit; everything else is flagged opaque in its BLAS.
#ifdef SER
    // DXR 1.2 shader execution reordering: traverse without shading, regroup the threads by what
    // they hit (the hit object sorts misses from hits, the hint sorts hits by material), then run
    // the closest hit or miss shader. The threads stay in the new order for the shading in
    // TracePath, which is where the material divergence is.
    HitObject hitObject =
        HitObject::TraceRay(g_Scene, RAY_FLAG_NONE, 0xFF, STANDARD_RAY_INDEX, 0, STANDARD_RAY_INDEX, ray, payload);
    if (reorder)
    {
        uint materialHint = hitObject.IsHit() ? LoadPrimitive(hitObject.GetInstanceID()).materialIndex : 0;
        ReorderThread(hitObject, materialHint, g_Frame.serCoherenceHintBits);
    }
    HitObject::Invoke(hitObject, payload);
#else
    TraceRay(g_Scene, RAY_FLAG_NONE, 0xFF, STANDARD_RAY_INDEX, 0, STANDARD_RAY_INDEX, ray, payload);
#endif

    HitInfo hit = (HitInfo)0;
    hit.hit = payload.hitT >= 0.0;
    if (!hit.hit)
        return hit;

    hit.t = payload.hitT;
    hit.position = origin + dir * payload.hitT;
    hit.Ng = payload.Ng;
    hit.N = payload.N;
    hit.T = normalize(payload.T - hit.N * dot(hit.N, payload.T));
    hit.B = cross(hit.N, hit.T) * payload.tangentSign;
    hit.uv = payload.uv;
    hit.uvGradients = payload.uvGradients;
    hit.materialId = payload.materialId;
    hit.emitter = payload.emitter;
    float2 curvature = f16tof32(uint2(payload.curvature, payload.curvature >> 16));
    hit.curvature = curvature.x;
    hit.meanCurvature = curvature.y;
    hit.emission = float3(0.0, 0.0, 0.0);
    hit.Ns = hit.N; // LoadMaterial applies the normal map
    return hit;
}

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

void BuildOrthonormalBasis(float3 n, out float3 t, out float3 b)
{
    // Duff et al. 2017
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// Uniform direction in the cone of 1 - cos(half angle) = oneMinusCosMax around axis, whose solid
// angle PDF is 1 / (2 pi oneMinusCosMax). An oneMinusCosMax of 2 covers the whole sphere.
float3 SampleUniformCone(float3 axis, float oneMinusCosMax, float2 u)
{
    float oneMinusCos = u.x * oneMinusCosMax;
    float cosTheta = 1.0 - oneMinusCos;
    float sinTheta = sqrt(max(0.0, oneMinusCos * (2.0 - oneMinusCos)));
    float phi = TWO_PI * u.y;

    float3 t, b;
    BuildOrthonormalBasis(axis, t, b);
    return normalize(t * (sinTheta * cos(phi)) + b * (sinTheta * sin(phi)) + axis * cosTheta);
}

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

// Samples one emissive triangle from the light buffer as seen from position, by uniform area sampling
// of the triangle, emitting from the front face (counter-clockwise v1, v2, v3), converted to a solid
// angle PDF. The PDF excludes the selection probability.
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

// The two halves of a float's bits as float16 values, low half in x (PackUVsToFloat).
float2 UnpackHalf2(float packed)
{
    uint bits = asuint(packed);
    return float2(f16tof32(bits & 0xFFFF), f16tof32(bits >> 16));
}

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

// The light tree only handles surfaces that reflect (lights below the tangent plane get no importance),
// so ones that NEE can light from behind fall back to uniform selection. A delta transmission lobe
// does not count, NEE cannot sample it. From bounce lightTreeGlossyBounces on, the
// specular weight is folded into the diffuse one, as for rough lobes in MakeLightTreeShadingContext:
// the glossy importance is the expensive half of the traversal, and those vertices carry little
// throughput.
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

// Next event estimation: picks the sun with sunSelectProbability, otherwise one light from the
// light buffer, and samples it. The returned PDF includes the selection probability and is 0 when
// there is nothing to sample.
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

// Traces a shadow ray and returns the visibility of the light, from 0 (fully occluded) to 1 (unoccluded) per channel
// Note that we use dedicated hit group with simpler shaders for shadow rays
float3 TraceShadowRay(float3 origin, float3 directionToLight, float tracingDistance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = directionToLight;
    ray.TMin = 0.0f;
    ray.TMax = tracingDistance;

    ShadowRayPayload payload;
    payload.visibility = float3(1.0f, 1.0f, 1.0f);

    uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    TraceRay(g_Scene, rayFlags, 0xFF, SHADOW_RAY_INDEX, 0, SHADOW_RAY_INDEX, ray, payload);

    return payload.visibility;
}

// Shadow ray from a surface. The caller skips back-facing light directions on opaque surfaces before
// it gets here, so the offset goes to whichever side the light is on, which is below the surface for
// transmission.
float3 CastShadowRay(float3 hitPosition, float3 surfaceNormal, float3 directionToLight, float tracingDistance)
{
    return TraceShadowRay(OffsetRay(hitPosition, surfaceNormal, directionToLight), directionToLight, tracingDistance);
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
// Sunlight that the scene's air scatters towards origin along the segment of segmentLength towards
// dir, estimated at one distance along it. The single scattering traces a shadow ray towards the sun,
// which gives the volumetric shadows; the multiple scattering from the LUT is not shadowed.
float3 AtmosphereInScattering(float3 origin, float3 dir, float segmentLength, inout uint rng)
{
    float3 sigmaT = g_Frame.atmosphereExtinction;
    float sigmaMean = (sigmaT.x + sigmaT.y + sigmaT.z) * (1.0 / 3.0);
    float tauMean = sigmaMean * segmentLength;
    if (!(tauMean > 0.0))
        return float3(0.0, 0.0, 0.0);

    // Distance s with pdf sigmaMean exp(-sigmaMean s) / (1 - exp(-tauMean)), the mean transmittance
    // normalised over the segment, or uniform where the segment is too thin for that to be accurate in
    // float. weight is the transmittance to s over its pdf.
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

// glTF transmission (KHR_materials_transmission and _ior) in OpenPBR terms. glTF tints what
// crosses the surface by the base colour, which is transmissionColour at zero depth. A volume's
// attenuation becomes the medium (colour at depth) instead, which drops the base colour tint; assets
// rarely use both.
void SetTransmission(inout OpenPBRBaseParams p, GpuMaterial material, float transmission, bool thinWalled)
{
    p.transmissionWeight = saturate(transmission);
    p.specularIor = material.ior;
    p.geometryThinWalled = thinWalled;
    if (!thinWalled && material.attenuationDistance < FLT_MAX)
    {
        p.transmissionColour = material.attenuationColour;
        p.transmissionDepth = material.attenuationDistance;
    }
    else
    {
        p.transmissionColour = p.baseColour;
    }
}

// The Rendering panel's debug switches (MaterialOverrideFlags), applied to every material.
void ApplyMaterialOverrides(inout OpenPBRBaseParams p)
{
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_SPECULAR) != 0)
    {
        p.baseMetalness = 0.0;
        p.specularColour = float3(0.0, 0.0, 0.0);
    }
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_TRANSMISSION) != 0)
        p.transmissionWeight = 0.0;
}

// Reads the material row once, samples all of its textures and applies the normal map to hit.Ns.
// The textures are filtered over the footprint of a ray cone coneWidth wide where it meets the surface.
OpenPBRBaseParams LoadMaterial(inout HitInfo hit, float coneWidth, bool anisotropic)
{
    GpuMaterial material = LoadGpuMaterial(hit.materialId);
    float2 uv = hit.uv;
    float2 ddx = hit.uvGradients.xy * coneWidth;
    float2 ddy = hit.uvGradients.zw * coneWidth;

    // Tangent-space normal map: +X along the tangent, +Y along the bitangent, +Z along N.
    if (material.normalTexture != INVALIDDESCRIPTORHANDLE)
    {
        float3 tn = SampleTextureGrad(material.normalTexture, uv, ddx, ddy, anisotropic).xyz * 2.0 - 1.0;
        tn.y = -tn.y; // flip green channel (DirectX-style normal map convention)
        tn.xy *= material.normalScale;
        hit.Ns = normalize(hit.T * tn.x + hit.B * tn.y + hit.N * tn.z);
    }

    float3 baseColour = material.baseColour.rgb;
    if (material.albedoTexture != INVALIDDESCRIPTORHANDLE)
        baseColour *= SampleTextureGrad(material.albedoTexture, uv, ddx, ddy, anisotropic).rgb;

    float roughness = material.roughness;
    float metallic = material.metallic;
    if (material.roughMetalTexture != INVALIDDESCRIPTORHANDLE)
    {
        float4 roughMetal = SampleTextureGrad(material.roughMetalTexture, uv, ddx, ddy, anisotropic);
        roughness *= roughMetal.g;
        metallic *= roughMetal.b;
    }

    float transmission = material.transmission;
    if (transmission > 0.0 && material.transmissionTexture != INVALIDDESCRIPTORHANDLE)
        transmission *= SampleTextureGrad(material.transmissionTexture, uv, ddx, ddy, anisotropic).r;
    bool thinWalled = (material.transmissionFlags & TRANSMISSION_THIN_WALLED) != 0;
    if (thinWalled && material.transmission > 0.0 && (material.transmissionFlags & TRANSMISSION_PASS_THROUGH) == 0)
        roughness = max(roughness, 1.0e-4);

    if (any(material.emissive > 0.0))
    {
        hit.emission = material.emissive;
        if (material.emissiveTexture != INVALIDDESCRIPTORHANDLE)
            hit.emission *= SampleTextureGrad(material.emissiveTexture, uv, ddx, ddy, anisotropic).rgb;
    }

    OpenPBRBaseParams p = OpenPBRDefaults();
    p.baseColour = baseColour;
    p.baseMetalness = saturate(metallic);
    p.specularRoughness = saturate(roughness);
    SetTransmission(p, material, transmission, thinWalled);
    ApplyMaterialOverrides(p);
    return p;
}

void GenerateCameraRay(uint2 pixel, float2 u, out float3 origin, out float3 dir)
{
    uint2 dims = DispatchRaysDimensions().xy;
    float2 ndc = (float2(pixel) + u) / float2(dims) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    // Unproject a point on the near plane (z = 1 with reversed Z) into view space, then rotate into
    // world space. Kept separate from the translation to avoid precision loss with a tiny near plane.
    float4 viewPoint = mul(float4(ndc, 1.0, 1.0), g_Frame.invProj);
    float3 viewDir = viewPoint.xyz / viewPoint.w;

    origin = g_Frame.invView[3].xyz;
    dir = normalize(mul(float4(viewDir, 0.0), g_Frame.invView).xyz);
}

// Random numbers

static const uint RNG_CALLS_PER_FRAME = 256;

uint3 Pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

uint RngInit(uint frameIndex)
{
    return frameIndex * RNG_CALLS_PER_FRAME;
}

float3 Rand3(inout uint rng)
{
    uint3 h = Pcg3d(uint3(DispatchRaysIndex().xy, rng++));
    return float3(h >> 8) * (1.0 / 16777216.0);
}

// Material helpers

// Spec default values, with an identity basis that the caller overwrites per hit.
OpenPBRBaseParams OpenPBRDefaults()
{
    OpenPBRBaseParams p;
    p.baseWeight = 1.0;
    p.baseColour = float3(0.8, 0.8, 0.8);
    p.baseMetalness = 0.0;
    p.baseDiffuseRoughness = 0.0;
    p.specularWeight = 1.0;
    p.specularColour = float3(1.0, 1.0, 1.0);
    p.specularRoughness = 0.3;
    p.specularRoughnessAnisotropy = 0.0;
    p.specularIor = 1.5;
    p.transmissionWeight = 0.0;
    p.transmissionColour = float3(1.0, 1.0, 1.0);
    p.transmissionDepth = 0.0;
    p.subsurfaceWeight = 0.0;
    p.subsurfaceColour = float3(0.8, 0.8, 0.8);
    p.geometryThinWalled = false;
    p.geometryTangent = float3(1.0, 0.0, 0.0);
    p.geometryBitangent = float3(0.0, 1.0, 0.0);
    p.geometryNormal = float3(0.0, 0.0, 1.0);
    return p;
}

// Beer-Lambert extinction so that light reaching transmissionDepth has transmissionColor.
float3 MediumExtinction(OpenPBRBaseParams p)
{
    if (p.transmissionWeight <= 0.0 || p.transmissionDepth <= 0.0)
        return float3(0.0, 0.0, 0.0);
    return -log(max(p.transmissionColour, 1.0e-4)) / p.transmissionDepth;
}

// Path tracing utilities

float PowerHeuristic(float a, float b)
{
    float a2 = a * a;
    float b2 = b * b;
    return a2 + b2 > 0.0 ? a2 / (a2 + b2) : 0.0;
}

// OffsetRay function from Ray Tracing Gems chapter 6
// Offsets the ray origin from current position p, along normal n (which must be geometric normal)
// so that no self-intersection can occur.
float3 OffsetRay(const float3 p, const float3 n)
{
    static const float origin = 1.0f / 32.0f;
    static const float float_scale = 1.0f / 65536.0f;
    static const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n);

    float3 p_i =
        float3(asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)), asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)), asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x, abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y, abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

// Offsets along whichever side of the geometric normal the new ray leaves from, so transmitted
// rays start below the surface and reflected ones above it.
float3 OffsetRay(float3 p, float3 Ng, float3 dir)
{
    return OffsetRay(p, dot(dir, Ng) < 0.0 ? -Ng : Ng);
}

static const uint RR_START_BOUNCE = 1;

static const float HALF_MAX = 65504.0;

// FP16_MAX, what DLSS-RR expects as the specular hit distance of sky pixels.
static const float SKY_HIT_DISTANCE = HALF_MAX;

// What DLSS-RR needs to know about the primary surface. Defaults are the sky values from the RR
// integration guide, so a primary miss leaves them as they are.
struct PrimaryGuides
{
    float4 position; // homogeneous world position, w = 0 for the sky (a direction at infinity)
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    float3 normal;
    float roughness;
    float specularHitDistance; // world-space length of the primary specular reflection ray
    // The surface whose reflection the specular guides describe, normally the guides' own surface but
    // a pane of glass in front of it when that reflects more (see PaneReflection).
    float4 reflector;
    float reflectorRoughness;
    float curvature; // the reflector's signed mean curvature as the camera sees it, positive for a convex mirror
};

PrimaryGuides DefaultGuides(float3 dir)
{
    PrimaryGuides g;
    g.position = float4(dir, 0.0);
    g.diffuseAlbedo = float3(0.5, 0.5, 0.5);
    g.specularAlbedo = float3(0.0, 0.0, 0.0);
    g.normal = float3(0.0, 0.0, 0.0);
    g.roughness = 0.0;
    g.specularHitDistance = SKY_HIT_DISTANCE;
    g.reflector = g.position;
    g.reflectorRoughness = 0.0;
    g.curvature = 0.0;
    return g;
}

// The hit's mean curvature as a viewer looking along -wo sees it, positive for a bulge towards them.
float ViewedCurvature(HitInfo hit, float3 wo)
{
    return dot(hit.Ng, wo) < 0.0 ? -hit.meanCurvature : hit.meanCurvature;
}

// Where the reflection on the guides' reflector appears, which its specular motion vector follows. A flat
// mirror shows the reflected point hitDistance beyond the surface along the view ray, at its mirror
// image, which stays put while the camera moves. A curved mirror shows it where the thin mirror
// equation 1/i = 1/o + 2 curvature puts it, behind the surface for positive i and in front of it, a real
// image, for negative i. A glossy surface shows a blend of the reflection and its own texture, so the
// image only moves the dominant direction's share of the way out from the surface (Lagarde and de
// Rousiers 2014, listing 22). The point is homogeneous, so a reflected sky (o infinite) or an image at
// a concave mirror's focus (i infinite) is a direction at infinity, and a real image behind the camera
// still projects to where it is seen.
float4 SpecularVirtualPosition(PrimaryGuides g, float3 viewDir, float hitDistance)
{
    float alpha = g.reflectorRoughness * g.reflectorRoughness;
    float smoothness = 1.0 - alpha;
    float dominantFactor = smoothness * (sqrt(smoothness) + alpha);
    if (g.reflector.w == 0.0 || hitDistance <= 0.0 || dominantFactor <= 0.0)
        return g.reflector;

    float inverseImageDistance = (hitDistance >= SKY_HIT_DISTANCE ? 0.0 : 1.0 / hitDistance) + 2.0 * g.curvature;
    return float4(g.reflector.xyz * inverseImageDistance + viewDir * dominantFactor, inverseImageDistance);
}

// Directional reflectance of an OpenPBR base, split into its diffuse and specular parts.
void ReflectanceGuides(OpenPBRBaseParams p, float3 wo, inout PrimaryGuides g)
{
    float M = p.baseMetalness;
    float3 baseColour = p.baseWeight * p.baseColour;

    float etaS = ModulatedIORRatio(p.specularIor, p.specularWeight);
    float dielectricF0 = (1.0 - etaS) / (1.0 + etaS);
    dielectricF0 *= dielectricF0;
    float3 F0 = lerp(dielectricF0 * p.specularColour, p.specularWeight * baseColour, M);

    float NdotV = dot(p.geometryNormal, wo);
    float roughness = p.specularRoughness;
    g.specularAlbedo = EnvBRDFApprox2(F0, roughness * roughness, NdotV);
    g.diffuseAlbedo = (1.0 - M) * (1.0 - p.transmissionWeight) * lerp(baseColour, p.subsurfaceColour, p.subsurfaceWeight);
}

float InitialSpecularHitDistance(float roughness)
{
    return roughness <= g_Frame.specularHitDistanceHoldRoughness ? -1.0 : 0.0;
}

static const float PSR_MIN_SPECULAR_FRACTION = 0.9;
static const float PSR_MAX_VIRTUAL_DEPTH_JITTER = 0.002;
float2 SpecularFractions(OpenPBRBaseParams p)
{
    float M = p.baseMetalness;
    float T = p.transmissionWeight;
    float3 baseColour = p.baseWeight * p.baseColour;

    float etaS = ModulatedIORRatio(p.specularIor, p.specularWeight);
    float F0 = (1.0 - etaS) / (1.0 + etaS);
    F0 *= F0;

    // F82-tint reflects baseColour head on.
    float specular = (1.0 - M) * F0 * Luminance(p.specularColour) + M * p.specularWeight * Luminance(baseColour);
    float diffuse = (1.0 - M) * (1.0 - F0) * (1.0 - T) * Luminance(lerp(baseColour, p.subsurfaceColour, p.subsurfaceWeight));
    float transmission = (1.0 - M) * (1.0 - F0) * T * Luminance(TransmissionTint(p));
    return float2(specular, transmission) / max(specular + transmission + diffuse, 1.0e-4);
}

bool IsPsrEvent(OpenPBRBaseParams p, OpenPBRContext ctx, uint lobe, out float fraction)
{
    fraction = 1.0;
    bool reflection = lobe == LOBE_SPECULAR_REFLECTION;
    if (!(reflection || lobe == LOBE_SPECULAR_TRANSMISSION) || p.specularRoughness > g_Frame.psrMaxRoughness)
        return false;
    if (ctx.inside)
        return reflection == (FresnelDielectric(ctx.wo.z, ctx.eta) >= 0.5);
    float2 fractions = SpecularFractions(p);
    fraction = reflection ? fractions.x : fractions.y;
    return fraction >= PSR_MIN_SPECULAR_FRACTION;
}

// Directional reflectance (reflection) or transmittance of a chain event, as wo sees it.
float3 PsrEventThroughput(OpenPBRBaseParams p, OpenPBRContext ctx, float3 wo, bool reflection)
{
    if (ctx.inside)
    {
        float F = FresnelDielectric(ctx.wo.z, ctx.eta);
        return (reflection ? F : 1.0 - F).xxx;
    }
    if (reflection)
    {
        PrimaryGuides reflectance;
        ReflectanceGuides(p, wo, reflectance);
        return reflectance.specularAlbedo;
    }
    return (1.0 - p.baseMetalness) * p.transmissionWeight * (1.0 - ctx.E) * TransmissionTint(p);
}

// A chain of mirrors and glass in front of the camera, primary first, while the path follows it.
struct SpecularChain
{
    bool active;
    // The chain's transform from real to virtual space, the product of one transform per event in
    // chain order: the Householder reflection of a mirror, or for a refraction the rotation taking the
    // refracted direction back to the incident one. A product with k reflections is a rotation when k
    // is even and a rotation after the fixed z reflection when k is odd, so it is kept as that
    // rotation's quaternion and the parity instead of a 3x3 matrix.
    bool parityOdd;
    float4 rotor;
    float3 throughput; // the events' directional reflectance or transmittance, and any absorption between them
    float primaryFraction; // share of the primary surface's reflectance in the lobe its path took
    float primaryDistance; // camera to primary surface
    float length; // primary surface to the latest hit, through the events in between
    float pixelDeflection; // largest angle an event turns the path by across one pixel's footprint
    bool throughGlass; // starts with a refraction into solid glass, whose transmittance it carries
};

float4 QuaternionMultiply(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// The rotation half of H(n) F at even parity, or of F H(n) at odd, where H(n) is the Householder
// reflection about the plane with normal n and F the reflection about z = 0. Either is a rotation by
// twice the angle between n and z, about their common perpendicular.
float4 MirrorRotor(float3 n, bool parityOdd)
{
    return parityOdd ? float4(n.y, -n.x, 0.0, n.z) : float4(-n.y, n.x, 0.0, n.z);
}

// The rotation taking unit vector a to unit vector b along the shorter arc, which is well defined
// for a refraction since it turns the ray by less than 90 degrees.
float4 ArcRotor(float3 a, float3 b)
{
    return normalize(float4(cross(a, b), 1.0 + dot(a, b)));
}

// Appends the event where the path arrived along dirIn and left along dirOut, at a surface whose
// normal turns by curvature radians per unit length.
void AddPsrEvent(inout SpecularChain chain, OpenPBRBaseParams p, OpenPBRContext ctx, float3 wo, float3 dirIn, float3 dirOut,
                 bool reflection, float pixelFootprint, float curvature)
{
    chain.throughput *= PsrEventThroughput(p, ctx, wo, reflection);

    // How much a turn of the normal turns the path: twice for a reflection or for the mirrored
    // reflection that a thin wall transmits, 1 - 1/eta for a refraction.
    float deflection;
    if (reflection)
    {
        chain.rotor = QuaternionMultiply(chain.rotor, MirrorRotor(p.geometryNormal, chain.parityOdd));
        chain.parityOdd = !chain.parityOdd;
        deflection = 2.0;
    }
    else
    {
        // Behind an odd number of reflections the rotation is conjugated by F, which negates the x and
        // y of its quaternion.
        float4 q = ArcRotor(dirOut, dirIn);
        chain.rotor = QuaternionMultiply(chain.rotor, chain.parityOdd ? float4(-q.xy, q.zw) : q);
        deflection = p.geometryThinWalled ? 2.0 : abs(1.0 - 1.0 / ctx.eta);
    }
    chain.pixelDeflection = max(chain.pixelDeflection, deflection * curvature * pixelFootprint);
}

SpecularChain StartSpecularChain(OpenPBRBaseParams p, OpenPBRContext ctx, float3 wo, float3 dirIn, float3 dirOut, bool reflection,
                                 float fraction, float primaryDistance, float pixelFootprint, float curvature)
{
    SpecularChain chain;
    chain.active = true;
    chain.parityOdd = false;
    chain.rotor = float4(0.0, 0.0, 0.0, 1.0);
    chain.throughput = float3(1.0, 1.0, 1.0);
    chain.primaryFraction = fraction;
    chain.primaryDistance = primaryDistance;
    chain.length = 0.0;
    chain.pixelDeflection = 0.0;
    chain.throughGlass = !reflection && !p.geometryThinWalled;
    AddPsrEvent(chain, p, ctx, wo, dirIn, dirOut, reflection, pixelFootprint, curvature);
    return chain;
}

float3 ApplySpecularChain(SpecularChain chain, float3 v)
{
    if (chain.parityOdd)
        v.z = -v.z;
    float3 t = 2.0 * cross(chain.rotor.xyz, v);
    return v + chain.rotor.w * t + cross(chain.rotor.xyz, t);
}

// Swap the primary surface's guides for the surface the chain ends on, unless an event on the way is
// curved enough to smear the virtual image. Returns whether it did.
bool ReplacePrimarySurface(SpecularChain chain, OpenPBRBaseParams surface, float curvature, float3 wo, float3 cameraDir,
                           inout PrimaryGuides guides)
{
    float pathLength = chain.primaryDistance + chain.length;
    if (chain.pixelDeflection * chain.length > PSR_MAX_VIRTUAL_DEPTH_JITTER * pathLength)
        return false;

    // The chain's transform is a rotation or reflection, which preserves the angle between wo and the
    // normal, so the reflectance seen in virtual space is the real one.
    PrimaryGuides seen;
    ReflectanceGuides(surface, wo, seen);

    // What the primary surface shows of the surface comes scaled by the chain's throughput, on top of
    // the primary's own (small) diffuse part.
    guides.position = float4(guides.position.xyz + cameraDir * chain.length, 1.0);
    guides.normal = normalize(ApplySpecularChain(chain, surface.geometryNormal));
    guides.roughness = surface.specularRoughness;
    guides.diffuseAlbedo += chain.throughput * seen.diffuseAlbedo;
    guides.specularAlbedo = lerp(guides.specularAlbedo, chain.throughput * seen.specularAlbedo, chain.primaryFraction);
    guides.specularHitDistance = InitialSpecularHitDistance(surface.specularRoughness);
    // The chain's rotations and reflections keep the viewer on the same side of the surface.
    guides.curvature = curvature;
    return true;
}

struct PaneReflection
{
    float3 albedo; // the split panes' directional reflectance, through the panes in front of them
    float luminance; // the brightest pane's share of albedo, 0 until a pane splits
    float distance; // camera to that pane
    float curvature; // its curvature as the camera sees it
    float hitDistance; // length of its reflection ray, SKY_HIT_DISTANCE for the sky
};

bool AddPaneReflection(inout PaneReflection panes, float3 albedo, float distance, float curvature)
{
    panes.albedo += albedo;
    float luminance = Luminance(albedo);
    if (luminance <= panes.luminance)
        return false;
    panes.luminance = luminance;
    panes.distance = distance;
    panes.curvature = curvature;
    panes.hitDistance = 0.0;
    return true;
}

static const uint DEFERRED_PENDING = 1;
static const uint DEFERRED_REFLECTION = 2; // solid glass's reflection, otherwise what is behind a pane
static const uint DEFERRED_BRIGHTEST_PANE = 4; // the reflection's first hit gives the brightest pane's hit distance
static const uint DEFERRED_SMOOTH_SEGMENT = 8; // set while the resumed reflection is on its first segment
static const uint DEFERRED_BOUNCE_SHIFT = 8; // the bounce the branch resumes at

struct DeferredBranch
{
    uint flags;
    float3 origin;
    float3 throughput;
    float3 dirOrCameraTransmittance; // the reflection's direction (behind a pane is along the camera ray)
    float coneWidth;
    float coneSpreadOrCameraDistance;
};

float MaxComponent(float3 v)
{
    return max(v.x, max(v.y, v.z));
}

float3 TracePath(float3 origin, float3 dir, inout uint rng, out PrimaryGuides guides)
{
    guides = DefaultGuides(dir);
    // The guides' surface continued the path with a specular reflection, so the next hit gives its
    // specular hit distance.
    bool hitDistancePending = false;
    SpecularChain chain = (SpecularChain)0;
    const float3 cameraOrigin = origin;
    const float3 cameraDir = dir;

    PaneReflection panes = (PaneReflection)0;
    bool paneHitPending = false; // the brightest pane's reflection ray ends at the next hit
    DeferredBranch deferred = (DeferredBranch)0;
    uint splits = 0;
    // Russian roulette compares the throughput with this, the weight of the branch it is on, so a
    // pane's reflection survives as long as a path of its own would.
    float rrInvScale = 1.0;

    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float3 segmentExtinction = float3(0.0, 0.0, 0.0);
    float prevBsdfPdf = 0.0;
    LightSelectVertex prevVertex = (LightSelectVertex)0; // where the current segment started, once a surface scattered
    RayCone cone = { 0.0, g_Frame.rayConePixelSpread };

    bool fromCamera = true;
    float3 cameraTransmittance = float3(1.0, 1.0, 1.0); // of the walls in front of the primary surface
    float cameraDistance = 0.0;

    // maxBounces counts indirect bounces, so the primary segment comes on top.
    const uint maxSegments = g_Frame.maxBounces + 1;

    uint bounce = 0;
    for (;;)
    {
        for (; bounce < maxSegments; ++bounce)
        {
            HitInfo hit = TraceClosest(origin, dir, bounce >= SER_START_BOUNCE, rng);

            if (hitDistancePending)
            {
                guides.specularHitDistance = hit.hit ? hit.t : SKY_HIT_DISTANCE;
                hitDistancePending = false;
            }
            if (paneHitPending)
            {
                panes.hitDistance = hit.hit ? hit.t : SKY_HIT_DISTANCE;
                paneHitPending = false;
            }
            if (chain.active)
                chain.length += hit.t;
            if (fromCamera)
                cameraDistance += hit.t;

            // Absorption along the segment just travelled, if it was inside a medium.
            if (any(segmentExtinction > 0.0))
            {
                if (!hit.hit)
                    break;
                float3 absorption = exp(-segmentExtinction * hit.t);
                throughput *= absorption;
                if (chain.active)
                    chain.throughput *= absorption;
            }
    #ifdef ATMOSPHERE
            else
            {
                // Otherwise it crossed the scene's air, which scatters sunlight into the path and dims
                // whatever lies beyond. Rays that leave the scene cross atmosphereMissDistance of it.
                float segmentLength = hit.hit ? hit.t : g_Frame.atmosphereMissDistance;
                if (bounce < g_Frame.atmosphereInScatterSegments)
                    radiance += throughput * AtmosphereInScattering(origin, dir, segmentLength, rng);
                throughput *= AtmosphereTransmittance(segmentLength);
            }
    #endif

            if (!hit.hit)
            {
                // Sky seen through a window: the guides describe the sky, as for a camera ray.
                if (fromCamera)
                    guides = DefaultGuides(dir);

                // The sky is never light sampled, so BSDF sampling gets its full contribution.
                radiance += throughput * EnvironmentRadiance(dir);

                // The sun disc is also reached by NEE, so weight it against light sampling.
                float3 Lsun = SunRadiance(dir);
                if (any(Lsun > 0.0))
                {
                    float w = fromCamera || (deferred.flags & DEFERRED_SMOOTH_SEGMENT) != 0 ? 1.0 : PowerHeuristic(prevBsdfPdf, LightPdf(origin, dir, hit));
                    radiance += throughput * Lsun * w;
                }
                break;
            }

            // Material and shading basis for this hit, its textures filtered over the cone's footprint.
            cone.width += cone.spread * hit.t;
            OpenPBRBaseParams mat = LoadMaterial(hit, cone.width * g_Frame.textureLodScale, bounce < g_Frame.anisotropicSegments);
            float3 wo = -dir;

            // Emission found by BSDF sampling, from front faces only like the area lights NEE samples.
            // NEE at the previous vertex could have picked this triangle too, so weight it against that,
            // unless the triangle has no light (degenerate, or black by the CPU texture estimate) or NEE
            // of the light buffer is off.
            if (any(hit.emission > 0.0) && dot(hit.Ng, wo) > 0.0)
            {
                float w = 1.0;
                if (!fromCamera && (deferred.flags & DEFERRED_SMOOTH_SEGMENT) == 0 && g_Frame.lightCount > 0 && hit.emitter != INVALID_LIGHT_INDEX)
                {
                    uint lightIndex = g_Emitters[hit.emitter];
                    if (lightIndex != INVALID_LIGHT_INDEX)
                        w = PowerHeuristic(prevBsdfPdf, EmitterLightPdf(prevVertex, lightIndex, dir, hit.position));
                }
                radiance += throughput * hit.emission * w;
            }
            deferred.flags &= ~DEFERRED_SMOOTH_SEGMENT;

            // Opaque and thin-walled surfaces are treated as two-sided: face the frame towards the viewer
            // so a back face is not mistaken for the inside of a transmissive object.
            bool opaque = mat.transmissionWeight <= 0.0;
            if ((opaque || mat.geometryThinWalled) && dot(hit.Ng, wo) < 0.0)
            {
                hit.Ng = -hit.Ng;
                hit.N = -hit.N;
                hit.Ns = -hit.Ns;
                hit.meanCurvature = -hit.meanCurvature;
            }

            SetShadingBasis(mat, hit.T, hit.B, hit.N, hit.Ns, wo);
            OpenPBRContext ctx = OpenPBRSetup(mat, wo);
            float3 extinction = MediumExtinction(mat);
            bool deltaTransmission = HasDeltaTransmission(mat);

            if (fromCamera)
            {
                guides.position = float4(hit.position, 1.0);
                guides.normal = mat.geometryNormal;
                guides.roughness = mat.specularRoughness;
                // Until the path continues with a specular reflection: no reflection data, or last frame's.
                guides.specularHitDistance = InitialSpecularHitDistance(mat.specularRoughness);
                guides.curvature = ViewedCurvature(hit, wo);
                ReflectanceGuides(mat, wo, guides);
                guides.diffuseAlbedo *= cameraTransmittance;
                guides.specularAlbedo *= cameraTransmittance;
            }

            // One hash covers the light sample (xy) and the Russian roulette decision (z).
            float3 lightU = Rand3(rng);

            LightSelectVertex vertex = MakeLightSelectVertex(hit.position, mat, ctx, wo, opaque || deltaTransmission, bounce);
            LightSample ls = SampleLight(vertex, lightU.xy, rng);
            bool lastSegment = bounce + 1 >= maxSegments;
            float3 neeContribution = float3(0.0, 0.0, 0.0);
            bool traceShadowRay = false;
            if (ls.pdf > 0.0 && (!opaque || dot(ls.direction, hit.Ng) > 0.0))
            {
                float bsdfPdf;
                float3 fCos = OpenPBREvalPdf(mat, ctx, ls.direction, bsdfPdf);
                if (any(fCos > 0.0))
                {
                    // The last segment takes no BSDF sample, which leaves NEE as the only strategy there.
                    float w = ls.lightSamplingOnly || lastSegment ? 1.0 : PowerHeuristic(ls.pdf, bsdfPdf);
                    bool intoMedium = dot(ls.direction, hit.Ng) < 0.0 && !mat.geometryThinWalled;
                    float3 shadowTransmittance = intoMedium ? exp(-extinction * ls.distance) : AtmosphereTransmittance(ls.distance);
                    neeContribution = throughput * fCos * ls.radiance * shadowTransmittance * (w / ls.pdf);
                    traceShadowRay = true;
                }
            }

            // Continue the path by sampling the BSDF, unless this is the last segment
            bool continuePath = false;
            bool specularReflection = false;
            bool passThrough = false;
            uint lobe = LOBE_NONE;
            const float3 dirIn = dir;
            float3 passTransmittance = float3(1.0, 1.0, 1.0);
            bool splittable = fromCamera && !lastSegment && splits < g_Frame.glassReflectionSplits && ctx.probDiffuse <= 0.0 &&
                              ctx.probSpecR > 0.0 && ctx.probSpecR < 1.0;
            // A thin pane: its reflection now, what is behind it deferred.
            bool paneSplit = splittable && deltaTransmission;
            // Solid glass: the refraction now, which the guides' chain follows, its reflection deferred.
            bool glassSplit = splittable && !mat.geometryThinWalled && !ctx.inside && mat.specularRoughness <= 0.0;
            if (paneSplit || glassSplit)
                ++splits;

            // Solid glass takes two samples, its deferred reflection's first. One call site in a loop keeps
            // the sampler inlined once. A split sets u.x to land in its lobe, so a sample is in that lobe or
            // none.
            BSDFSample bs = (BSDFSample)0;
            BSDFSample reflected = (BSDFSample)0;
            if (!lastSegment)
            {
                [loop]
                for (uint s = glassSplit ? 0 : 1; s < 2; ++s)
                {
                    float3 u = Rand3(rng);
                    if (s == 0 || paneSplit)
                        u.x = 0.0;
                    else if (glassSplit)
                        u.x = ctx.probSpecR; // the first value past the reflection's share
                    BSDFSample sample = OpenPBRSample(mat, ctx, u);
                    if (s == 0)
                        reflected = sample;
                    else
                        bs = sample;
                }
            }

            if (paneSplit)
            {
                float3 transmittance = ThinDeltaTransmittance(mat, ctx.wo.z);
                deferred.flags = DEFERRED_PENDING | ((bounce + 1) << DEFERRED_BOUNCE_SHIFT);
                deferred.origin = OffsetRay(hit.position, hit.Ng, dir);
                deferred.throughput = throughput * transmittance;
                deferred.dirOrCameraTransmittance = cameraTransmittance * transmittance;
                deferred.coneSpreadOrCameraDistance = cameraDistance;
                deferred.coneWidth = cone.width;

                PrimaryGuides seen;
                ReflectanceGuides(mat, wo, seen);
                paneHitPending = AddPaneReflection(panes, cameraTransmittance * seen.specularAlbedo, cameraDistance,
                                                   ViewedCurvature(hit, wo)) &&
                                 bs.lobe != LOBE_NONE;
            }
            else if (glassSplit && reflected.lobe != LOBE_NONE)
            {
                deferred.flags = DEFERRED_PENDING | DEFERRED_REFLECTION | ((bounce + 1) << DEFERRED_BOUNCE_SHIFT);
                deferred.origin = OffsetRay(hit.position, hit.Ng, reflected.wi);
                deferred.dirOrCameraTransmittance = reflected.wi;
                deferred.throughput = throughput * reflected.weight * ctx.probSpecR;
                deferred.coneWidth = cone.width;
                deferred.coneSpreadOrCameraDistance =
                        min(cone.spread + g_Frame.rayConeBsdfSpread * SampledConeSpread(reflected.pdf / ctx.probSpecR), PI);
                // The guides' specular albedo is the glass's reflection, which moves to the panes.
                if (AddPaneReflection(panes, guides.specularAlbedo, cameraDistance, ViewedCurvature(hit, wo)))
                    deferred.flags |= DEFERRED_BRIGHTEST_PANE;
                guides.specularAlbedo = float3(0.0, 0.0, 0.0);
            }

            lobe = bs.lobe;
            passThrough = bs.lobe == LOBE_DELTA_TRANSMISSION;
            continuePath = bs.lobe != LOBE_NONE;
            if (passThrough)
            {
                // Same direction, same cone. NEE where the segment started reaches the next hit through
                // this wall with its transmittance, so the MIS there only gains the probability of
                // picking the lobe, which bs.pdf is.
                throughput *= bs.weight;
                prevBsdfPdf *= bs.pdf;
                passTransmittance = bs.weight * bs.pdf;
                segmentExtinction = float3(0.0, 0.0, 0.0);
            }
            else if (continuePath)
            {
                // A pane's reflection is not its guides' (what is behind it is), see PaneReflection.
                specularReflection = !paneSplit && bs.lobe == LOBE_SPECULAR_REFLECTION;
                float lobeProbability = paneSplit ? ctx.probSpecR : glassSplit ? ctx.probSpecT : 1.0;
                throughput *= bs.weight * lobeProbability;
                // Russian roulette on a pane's reflection is relative to the branch, as for a path of its own.
                if (paneSplit)
                    rrInvScale = 1.0 / max(MaxComponent(throughput), 1.0e-8);
                prevBsdfPdf = bs.pdf;
                prevVertex = vertex;
                // Surface curvature is ignored. The cap stops a tiny pdf from making the width infinite.
                cone.spread = min(cone.spread + g_Frame.rayConeBsdfSpread * SampledConeSpread(bs.pdf / lobeProbability), PI);

                // Heading into the object means travelling through its medium until the next hit.
                segmentExtinction = dot(bs.wi, hit.Ng) < 0.0 ? extinction : float3(0.0, 0.0, 0.0);
                dir = bs.wi;
            }

            // DLSS-RR guides: the primary surface, or the first surface past the mirrors and glass in front
            // of it. A surface that takes another lobe, or ends the path, ends the chain on itself.
            float psrFraction = 1.0;
            bool psrEvent = continuePath && !paneSplit && (fromCamera || chain.active) &&
                            (glassSplit ? mat.specularRoughness <= g_Frame.psrMaxRoughness : IsPsrEvent(mat, ctx, lobe, psrFraction));
            if (fromCamera)
            {
                hitDistancePending = specularReflection;
                if (psrEvent)
                    chain = StartSpecularChain(mat, ctx, wo, dirIn, dir, specularReflection, psrFraction, cameraDistance, cone.width,
                                               hit.curvature);
                if (!opaque && !mat.geometryThinWalled && !(psrEvent && chain.throughGlass))
                    guides.diffuseAlbedo += cameraTransmittance * PsrEventThroughput(mat, ctx, wo, false);
                if (passThrough)
                    cameraTransmittance *= passTransmittance;
            }
            else if (chain.active)
            {
                if (passThrough)
                {
                    chain.throughput *= passTransmittance;
                }
                else if (psrEvent)
                {
                    AddPsrEvent(chain, mat, ctx, wo, dirIn, dir, specularReflection, cone.width, hit.curvature);
                }
                else
                {
                    chain.active = false;
                    if (!ctx.inside && ReplacePrimarySurface(chain, mat, ViewedCurvature(hit, wo), wo, cameraDir, guides))
                        hitDistancePending = specularReflection;
                    else if (chain.throughGlass)
                        guides.diffuseAlbedo += cameraTransmittance * chain.throughput;
                }
            }

            if (continuePath)
                origin = OffsetRay(hit.position, hit.Ng, dir);

            if (traceShadowRay)
                radiance += neeContribution * CastShadowRay(hit.position, hit.Ng, ls.direction, ls.distance);

            if (!continuePath)
                break;

            fromCamera = fromCamera && passThrough;

            // Russian roulette.
            if (bounce >= RR_START_BOUNCE)
            {
                float q = saturate(MaxComponent(throughput) * rrInvScale);
                if (lightU.z >= q)
                    break;
                throughput /= q;
            }
        }

        // A refraction chain the path ended on leaves the glass's transmittance as its diffuse albedo.
        if (chain.active && chain.throughGlass)
            guides.diffuseAlbedo += cameraTransmittance * chain.throughput;
        chain.active = false;

        if ((deferred.flags & DEFERRED_PENDING) == 0)
            break;

        origin = deferred.origin;
        throughput = deferred.throughput;
        cone.width = deferred.coneWidth;
        segmentExtinction = float3(0.0, 0.0, 0.0);
        bounce = deferred.flags >> DEFERRED_BOUNCE_SHIFT;
        hitDistancePending = false;
        if ((deferred.flags & DEFERRED_REFLECTION) != 0)
        {
            dir = deferred.dirOrCameraTransmittance;
            cone.spread = deferred.coneSpreadOrCameraDistance;
            rrInvScale = 1.0;
            paneHitPending = (deferred.flags & DEFERRED_BRIGHTEST_PANE) != 0;
            fromCamera = false;
            deferred.flags = DEFERRED_SMOOTH_SEGMENT;
        }
        else
        {
            dir = cameraDir;
            cameraTransmittance = deferred.dirOrCameraTransmittance;
            cameraDistance = deferred.coneSpreadOrCameraDistance;
            cone.spread = g_Frame.rayConePixelSpread;
            prevBsdfPdf = 0.0;
            rrInvScale = 1.0;
            fromCamera = true;
            deferred.flags = 0;
        }
    }

    // The specular guides follow the reflection on the guides' surface, or on the brightest split pane
    // in front of it if that reflects more.
    guides.reflector = guides.position;
    guides.reflectorRoughness = guides.roughness;
    if (panes.luminance > 0.0)
    {
        bool paneReflects = panes.luminance >= Luminance(guides.specularAlbedo);
        guides.specularAlbedo += panes.albedo;
        if (paneReflects)
        {
            guides.reflector = float4(cameraOrigin + cameraDir * panes.distance, 1.0);
            guides.reflectorRoughness = 0.0;
            guides.roughness = 0.0;
            guides.curvature = panes.curvature;
            guides.specularHitDistance = panes.hitDistance;
        }
    }

    return radiance;
}

// Entry points

[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
    uint rng = RngInit(g_Frame.frameIndex);

#ifdef DLSS_RR
    // Every pixel shares this frame's jitter, which DLSS is told about separately.
    float2 subpixel = 0.5 + g_Frame.jitter;
#else
    float2 subpixel = Rand3(rng).xy;
#endif

    float3 origin, dir;
    GenerateCameraRay(pixel, subpixel, origin, dir);
    PrimaryGuides guides;
    float3 L = TracePath(origin, dir, rng, guides);

    // A single bad sample would otherwise stick in any temporal filter downstream.
    if (any(isnan(L)) || any(isinf(L)))
        L = float3(0.0, 0.0, 0.0);

    g_Output[pixel] = float4(L, 1.0);

#ifdef DLSS_RR
    // Last frame's specular hit distance, loaded first so its latency hides behind the guides below.
    float heldHitDistance = g_SpecularHitDistance[pixel];

    // Static scene, so the primary point was in the same place last frame and only the camera
    // moved. CalculateMotionVector works in NDC, whose +y is up in D3D, while DLSS wants +y down.
    float2 motion = CalculateMotionVector(guides.position, guides.position, g_Frame.prevViewProj, g_Frame.viewProj, float2(dims));
    motion.y = -motion.y;

    // Reversed Z: the sky sits on the far plane at 0.
    float4 clip = mul(guides.position, g_Frame.viewProj);
    float depth = guides.position.w > 0.0 ? clip.z / clip.w : 0.0;

    g_DiffuseAlbedo[pixel] = float4(guides.diffuseAlbedo, 1.0);
    g_SpecularAlbedo[pixel] = float4(guides.specularAlbedo, 1.0);
    g_NormalRoughness[pixel] = float4(guides.normal, guides.roughness);
    g_MotionVectors[pixel] = motion;
    g_Depth[pixel] = depth;

    // Negative: keep last frame's, which is whatever the texture was created with until a frame writes it.
    float hitDistance = guides.specularHitDistance;
    if (hitDistance >= 0.0)
        g_SpecularHitDistance[pixel] = hitDistance;
    else
        hitDistance = heldHitDistance;
    if (!isfinite(hitDistance))
        hitDistance = 0.0;

    // The reflected point is static too, so only the camera moves its image.
    float4 reflection = SpecularVirtualPosition(guides, dir, hitDistance);
    float2 specularMotion = CalculateMotionVector(reflection, reflection, g_Frame.prevViewProj, g_Frame.viewProj, float2(dims));
    specularMotion.y = -specularMotion.y;
    g_SpecularMotionVectors[pixel] = specularMotion;
#endif
}

[shader("miss")]
void Miss(inout HitPayload payload)
{
    payload.hitT = -1.0;
}

// Vertex indices of the triangle being hit, already rebased into the shared vertex buffer.
uint3 LoadTriangle(PrimitiveDesc prim)
{
    uint first = prim.indexOffset + PrimitiveIndex() * 3;
    return uint3(g_Indices[first], g_Indices[first + 1], g_Indices[first + 2]) + prim.vertexOffset;
}

float3 Barycentrics(BuiltInTriangleIntersectionAttributes attr)
{
    return float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
}

float2 InterpolateTexCoord(uint3 tri, float3 bary)
{
    return g_Vertices[tri.x].texCoord * bary.x + g_Vertices[tri.y].texCoord * bary.y + g_Vertices[tri.z].texCoord * bary.z;
}

// uv across the ellipse where a ray cone of unit width meets the triangle's plane, as SampleGrad's
// ddx (.xy) and ddy (.zw), from "Improved Shader and Texture Level of Detail Using Ray Cones"
// (Akenine-Moller et al. 2021). The axis along the ray's projection onto the plane is stretched by
// 1 / cos, the one across it is not. Both scale linearly with the cone width, which raygen knows.
// e1, e2 are the world-space edges from the first corner and n = cross(e1, e2).
float4 RayConeUvGradients(float3 e1, float3 e2, float3 n, float2 uv0, float2 uv1, float2 uv2, float3 dir)
{
    float invN2 = 1.0 / max(dot(n, n), 1.0e-30);
    float3 nHat = n * sqrt(invN2);
    float cosTheta = dot(nHat, dir);
    float3 major = dir - nHat * cosTheta;
    float sin2 = dot(major, major);
    // Head on, the footprint is a circle and any axis in the plane will do.
    major = sin2 > 1.0e-8 ? major * rsqrt(sin2) : normalize(e1);
    float3 minor = cross(nHat, major);
    major /= max(abs(cosTheta), 1.0e-3);

    // Barycentric gradients in the plane: an offset a = b1 e1 + b2 e2 has b1 = dot(a, cross(e2, n)) / |n|^2
    // and b2 = dot(a, cross(n, e1)) / |n|^2.
    float3 db1 = cross(e2, n) * invN2;
    float3 db2 = cross(n, e1) * invN2;
    float2 duv1 = uv1 - uv0;
    float2 duv2 = uv2 - uv0;
    return float4(duv1 * dot(major, db1) + duv2 * dot(major, db2), duv1 * dot(minor, db1) + duv2 * dot(minor, db2));
}

[shader("closesthit")]
void ClosestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
    uint3 tri = LoadTriangle(prim);
    float3 bary = Barycentrics(attr);

    GpuVertex v0 = g_Vertices[tri.x];
    GpuVertex v1 = g_Vertices[tri.y];
    GpuVertex v2 = g_Vertices[tri.z];
    float3 p0 = v0.position;
    float3 p1 = v1.position;
    float3 p2 = v2.position;
    float3 objectNormal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    float4 objectTangent = v0.tangent * bary.x + v1.tangent * bary.y + v2.tangent * bary.z;

    float3x3 linearToWorld = (float3x3)ObjectToWorld3x4();
    float3x3 cofactor = float3x3(cross(linearToWorld[1], linearToWorld[2]),
                                 cross(linearToWorld[2], linearToWorld[0]),
                                 cross(linearToWorld[0], linearToWorld[1]));
    float det = dot(linearToWorld[0], cofactor[0]);
    float handedness = det < 0.0 ? -1.0 : 1.0;

    // cross(M a, M b) = cofactor(M) cross(a, b), so this is the transformed object-space face normal.
    float3 e1 = mul(linearToWorld, p1 - p0);
    float3 e2 = mul(linearToWorld, p2 - p0);
    float3 faceCross = cross(e1, e2);
    float3 Ng = normalize(faceCross * handedness);

    payload.hitT = RayTCurrent();
    payload.Ng = Ng;
    payload.materialId = prim.materialIndex;
    payload.N = normalize(mul(cofactor, objectNormal) * handedness);
    payload.T = normalize(mul(linearToWorld, objectTangent.xyz));
    // A mirroring transform flips cross(N, T), so fold its handedness into the bitangent sign.
    payload.tangentSign = (objectTangent.w < 0.0 ? -1.0 : 1.0) * handedness;
    float2 uv0 = v0.texCoord;
    float2 uv1 = v1.texCoord;
    float2 uv2 = v2.texCoord;
    payload.uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    payload.uvGradients = RayConeUvGradients(e1, e2, faceCross, uv0, uv1, uv2, WorldRayDirection());

#ifdef DLSS_RR
    // Largest turn of the vertex normals per unit length along an edge, which tells a flat mirror from
    // a curved one for primary surface replacement. The chord between unit normals stands in for
    // the angle, and the normals are taken in object space against world-space edges.
    float3 dn1 = v1.normal - v0.normal;
    float3 dn2 = v2.normal - v0.normal;
    float3 dn3 = dn2 - dn1;
    float3 e3 = e2 - e1;
    float k1 = dot(dn1, dn1) / max(dot(e1, e1), 1.0e-20);
    float k2 = dot(dn2, dn2) / max(dot(e2, e2), 1.0e-20);
    float k3 = dot(dn3, dn3) / max(dot(e3, e3), 1.0e-20);
    float curvature = sqrt(max(k1, max(k2, k3)));
    float3 o1 = p1 - p0;
    float3 o2 = p2 - p0;
    float3 o3 = o2 - o1;
    float m1 = dot(dn1, o1) * rsqrt(max(dot(o1, o1) * dot(e1, e1), 1.0e-30));
    float m2 = dot(dn2, o2) * rsqrt(max(dot(o2, o2) * dot(e2, e2), 1.0e-30));
    float m3 = dot(dn3, o3) * rsqrt(max(dot(o3, o3) * dot(e3, e3), 1.0e-30));
    float meanCurvature = (m1 + m2 + m3) * (1.0 / 3.0);
    payload.curvature = f32tof16(min(curvature, HALF_MAX)) | (f32tof16(clamp(meanCurvature, -HALF_MAX, HALF_MAX)) << 16);
#else
    payload.curvature = 0;
#endif

    uint emitterTable = g_Emitters[InstanceIndex()];
    payload.emitter = emitterTable != INVALID_LIGHT_INDEX ? emitterTable + PrimitiveIndex() : INVALID_LIGHT_INDEX;
}

// Base colour and alpha at the candidate hit, for the any-hit shaders.
float4 LoadBaseColourAlpha(GpuMaterial material, PrimitiveDesc prim, BuiltInTriangleIntersectionAttributes attr)
{
    float4 albedo = material.baseColour;
    if (material.albedoTexture != INVALIDDESCRIPTORHANDLE)
        albedo *= SampleTexture(material.albedoTexture, InterpolateTexCoord(LoadTriangle(prim), Barycentrics(attr)));
    return albedo;
}

// Uniform random number for the candidate hit. Hashing the instance and primitive in gives every
// surface along the ray its own number, and a repeated any-hit call for the same triangle the same one.
float SurfaceRandom(uint seed)
{
    uint3 h = Pcg3d(uint3(DispatchRaysIndex().xy, seed));
    h = Pcg3d(uint3(h.x, InstanceID(), PrimitiveIndex()));
    return float(h.x >> 8) * (1.0 / 16777216.0);
}

// Alpha for camera and bounce rays. Masked surfaces are cut at alphaCutoff. Blended ones use
// stochastic transparency: the ray stops at each with probability alpha, so on average a surface
// contributes alpha times its shading and lets 1 - alpha of what is behind it through. 
[shader("anyhit")]
void AnyHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
    GpuMaterial material = LoadGpuMaterial(prim.materialIndex);
    if (material.alphaMode == ALPHA_MODE_OPAQUE)
        return;
    float alpha = LoadBaseColourAlpha(material, prim, attr).a;

    bool cutOut = material.alphaMode == ALPHA_MODE_MASK ? alpha < material.alphaCutoff
                                                        : SurfaceRandom(payload.materialId) >= alpha;
    if (cutOut)
        IgnoreHit();
}

[shader("miss")]
void ShadowMiss(inout ShadowRayPayload payload)
{
}

[shader("closesthit")]
void ShadowClosestHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.visibility = float3(0.0, 0.0, 0.0);
}

// Cosine between the shadow ray and the candidate triangle's plane.
float FaceCosine(PrimitiveDesc prim)
{
    uint3 tri = LoadTriangle(prim);
    float3x3 linearToWorld = (float3x3)ObjectToWorld3x4();
    float3 e1 = mul(linearToWorld, g_Vertices[tri.y].position - g_Vertices[tri.x].position);
    float3 e2 = mul(linearToWorld, g_Vertices[tri.z].position - g_Vertices[tri.x].position);
    float3 n = cross(e1, e2);
    return abs(dot(n, WorldRayDirection())) * rsqrt(max(dot(n, n), 1.0e-30));
}

float3 PassThroughTransmittance(GpuMaterial material, PrimitiveDesc prim, BuiltInTriangleIntersectionAttributes attr,
                                float3 baseColour)
{
    float2 uv = InterpolateTexCoord(LoadTriangle(prim), Barycentrics(attr));
    float transmission = material.transmission;
    if (material.transmissionTexture != INVALIDDESCRIPTORHANDLE)
        transmission *= SampleTexture(material.transmissionTexture, uv).r;
    float metallic = material.metallic;
    if (material.roughMetalTexture != INVALIDDESCRIPTORHANDLE)
        metallic *= SampleTexture(material.roughMetalTexture, uv).b;

    OpenPBRBaseParams p = OpenPBRDefaults();
    p.baseColour = baseColour;
    p.baseMetalness = saturate(metallic);
    p.specularRoughness = 0.0;
    SetTransmission(p, material, transmission, true);
    ApplyMaterialOverrides(p);
    return ThinDeltaTransmittance(p, FaceCosine(prim));
}

// Shadow rays stop at the first surface that blocks them, multiplying in what the blended (1 - alpha)
// and pass-through surfaces before it let through.
[shader("anyhit")]
void ShadowAnyHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
    GpuMaterial material = LoadGpuMaterial(prim.materialIndex);
    float4 albedo = LoadBaseColourAlpha(material, prim, attr);

    // Of the light reaching the surface, the share that meets it rather than passing a cut-out or blend.
    float coverage = 1.0;
    if (material.alphaMode == ALPHA_MODE_MASK && albedo.a < material.alphaCutoff)
        IgnoreHit();
    if (material.alphaMode == ALPHA_MODE_BLEND)
        coverage = albedo.a;

    float3 transmitted = float3(1.0 - coverage, 1.0 - coverage, 1.0 - coverage);
    if ((material.transmissionFlags & TRANSMISSION_PASS_THROUGH) != 0)
        transmitted += coverage * PassThroughTransmittance(material, prim, attr, albedo.rgb);

    payload.visibility *= transmitted;
    if (dot(payload.visibility, 0.333) > 0.001)
        IgnoreHit();
    AcceptHitAndEndSearch();
}
