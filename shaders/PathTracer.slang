#include "PathTracerInterop.h"
#include "DLSSUtil.h"

import OpenPBRBaseSubstrate;
import LightTree;

// Resources

ConstantBuffer<FrameConstants> g_Frame : register(b0);
RaytracingAccelerationStructure g_Scene : register(t0);
RWTexture2D<float4> g_Output : register(u0);
SamplerState g_LinearWrap : register(s3);

// DLSS Ray Reconstruction guide buffers, all at render resolution. PathTracerPass compiles this
// permutation (and binds these) only while DLSS-RR is enabled.
#ifdef DLSS_RR
RWTexture2D<float4> g_DiffuseAlbedo : register(u1);
RWTexture2D<float4> g_SpecularAlbedo : register(u2);
RWTexture2D<float4> g_NormalRoughness : register(u3); // world-space shading normal, .w = linear roughness
RWTexture2D<float2> g_MotionVectors : register(u4);   // render pixels, current to previous
RWTexture2D<float> g_Depth : register(u5);            // hardware depth, reversed Z
RWTexture2D<float> g_SpecularHitDistance : register(u6);
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
StructuredBuffer<float3> g_Positions : register(t3);
StructuredBuffer<float3> g_Normals : register(t4);
StructuredBuffer<float2> g_TexCoords : register(t5);
StructuredBuffer<float4> g_Tangents : register(t6);
StructuredBuffer<GpuMaterial> g_Materials : register(t7);
// Point and area lights, indexed by the light tree leaves. Never empty, so these stay bound.
StructuredBuffer<GpuLight> g_Lights : register(t8);
StructuredBuffer<GpuLightTreeNode> g_LightTreeNodes : register(t9);

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
    // No ray differentials yet, so always the top mip.
    Texture2D<float4> texture = ResourceDescriptorHeap[NonUniformResourceIndex(textureIndex)];
    return texture.SampleLevel(g_LinearWrap, uv, 0.0);
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
    float3 emission;
    uint materialId;
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
    uint materialId;
    float3 N;
    float tangentSign;
    float3 T;
    float2 uv;
};

HitInfo TraceClosest(float3 origin, float3 dir)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = FLT_MAX;

    HitPayload payload = (HitPayload)0;
    payload.hitT = -1.0;
    TraceRay(g_Scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, STANDARD_RAY_INDEX, 0, STANDARD_RAY_INDEX, ray, payload);

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
    hit.materialId = payload.materialId;
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

// Radiance of the sun disc seen along dir, zero outside it.
float3 SunRadiance(float3 dir)
{
    return !SunIsDelta() && InSunDisc(dir) ? g_Frame.sunRadiance : float3(0.0, 0.0, 0.0);
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
    ls.radiance = g_Frame.sunRadiance;
    ls.pdf = g_Frame.sunPdf;
    ls.lightSamplingOnly = SunIsDelta();
    ls.direction = SunIsDelta() ? g_Frame.sunDirection : SampleUniformCone(g_Frame.sunDirection, g_Frame.sunOneMinusCosMax, u);
    return ls;
}

// Samples one light from the light buffer as seen from position. The PDF excludes the selection
// probability. The radiance map of a textured area light is not applied.
LightSample SampleBufferLight(GpuLight light, float3 position, float2 u)
{
    LightSample ls;
    if (light.GetLightType() == kLight_Point)
    {
        float3 toLight = light.v1.xyz - position;
        float d2 = max(dot(toLight, toLight), 1.0e-8);
        float d = sqrt(d2);
        float radius = light.v2.x;
        if (radius <= 0.0)
        {
            // Isotropic point light: radiance.xyz is the intensity, so it delivers intensity / d^2.
            ls.distance = d;
            ls.direction = toLight / d;
            ls.radiance = light.radiance.xyz / d2;
            ls.pdf = 1.0;
            ls.lightSamplingOnly = true;
            return ls;
        }

        float r2 = radius * radius;
        bool inside = d2 <= r2;
        float sin2Max = r2 / d2;
        float oneMinusCosMax = inside ? 2.0 : sin2Max / (1.0 + sqrt(1.0 - sin2Max));
        ls.direction = SampleUniformCone(toLight / d, oneMinusCosMax, u);

        // Distance to the near side of the sphere along the sample, or the far side from inside.
        float b = dot(toLight, ls.direction);
        float h = sqrt(max(r2 - (d2 - b * b), 0.0));
        ls.distance = max(inside ? b + h : b - h, 0.0);
        ls.radiance = light.radiance.xyz / (PI * r2);
        ls.pdf = 1.0 / (TWO_PI * oneMinusCosMax);
        ls.lightSamplingOnly = true;
        return ls;
    }

    // Area light: uniform area sampling of the triangle, emitting from the front face (counter-
    // clockwise v1, v2, v3), converted to a solid angle PDF.
    float3 p0 = light.v1.xyz;
    float3 p1 = light.v2.xyz;
    float3 p2 = light.v3.xyz;
    float su = sqrt(u.x);
    float b0 = 1.0 - su;
    float b1 = u.y * su;
    float3 p = b0 * p0 + b1 * p1 + (1.0 - b0 - b1) * p2;

    float3 crossE = cross(p1 - p0, p2 - p0);
    float twiceArea = length(crossE);
    float3 toLight = p - position;
    float d2 = max(dot(toLight, toLight), 1.0e-8);
    float d = sqrt(d2);
    ls.direction = toLight / d;
    float cosLight = -dot(crossE, ls.direction) / twiceArea;

    ls.radiance = cosLight > 0.0 ? light.radiance.xyz : float3(0.0, 0.0, 0.0);
    ls.pdf = cosLight > 0.0 ? 2.0 * d2 / (twiceArea * cosLight) : 0.0;
    // Stop short of the emitter so it does not shadow itself.
    ls.distance = d * 0.999;
    // Not in the scene geometry either, so BSDF sampling never finds its emission. Switch this
    // back to MIS once emissive hits are weighted against NEE with LightTreePmf.
    ls.lightSamplingOnly = true;
    return ls;
}

// Next event estimation: picks the sun with sunSelectProbability, otherwise one light from the
// light buffer, and samples it. The returned PDF includes the selection probability and is 0 when
// there is nothing to sample. The light tree only handles opaque surfaces (lights below the
// tangent plane get no importance), so transmissive ones fall back to uniform selection.
LightSample SampleLight(float3 position, OpenPBRBaseParams mat, OpenPBRContext ctx, float3 wo, bool opaque,
                        float2 u, inout uint rng)
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
    if (g_Frame.lightSampling == LIGHT_SAMPLING_TREE && opaque)
    {
        LightTreeShadingContext treeCtx = MakeLightTreeShadingContext(
            position, mat.geometryTangent, mat.geometryBitangent, mat.geometryNormal, wo,
            ctx.probDiffuse, ctx.probSpecR, ctx.alpha.x, ctx.alpha.y);
        if (!SampleLightTree(g_LightTreeNodes, treeCtx, uSelect, lightIndex, pmf))
            return ls;
    }
    else
    {
        lightIndex = min(uint(uSelect * g_Frame.lightCount), g_Frame.lightCount - 1);
        pmf = 1.0 / g_Frame.lightCount;
    }

    GpuLight light = g_Lights[lightIndex];
    // Only area and sphere lights need a position on the light.
    GpuLightType type = light.GetLightType();
    bool hasArea = type == kLight_Area || (type == kLight_Point && light.v2.x > 0.0);
    float2 uLight = hasArea ? Rand3(rng).xy : float2(0.0, 0.0);
    ls = SampleBufferLight(light, position, uLight);
    ls.radiance *= g_Frame.lightIntensityScale;
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

// Constant sky; the sun disc is handled separately so it can be MIS weighted.
float3 EnvironmentRadiance(float3 dir)
{
    return g_Frame.skyRadiance;
}

// Casts a shadow ray and returns the visibility of the light, from 0 (fully occluded) to 1 (unoccluded) per channel
// Note that we use dedicated hit group with simpler shaders for shadow rays
// The caller skips back-facing light directions on opaque surfaces before it gets here, so the
// offset goes to whichever side the light is on, which is below the surface for transmission.
float3 CastShadowRay(float3 hitPosition, float3 surfaceNormal, float3 directionToLight, float tracingDistance)
{
    RayDesc ray;
    ray.Origin = OffsetRay(hitPosition, surfaceNormal, directionToLight);
    ray.Direction = directionToLight;
    ray.TMin = 0.0f;
    ray.TMax = tracingDistance;

    ShadowRayPayload payload;
    payload.visibility = float3(1.0f, 1.0f, 1.0f);

    uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    TraceRay(g_Scene, rayFlags, 0xFF, SHADOW_RAY_INDEX, 0, SHADOW_RAY_INDEX, ray, payload);

    return payload.visibility;
}

// Reads the material row once, samples all of its textures and applies the normal map to hit.Ns.
OpenPBRBaseParams LoadMaterial(inout HitInfo hit)
{
    GpuMaterial material = LoadGpuMaterial(hit.materialId);
    float2 uv = hit.uv;

    // Tangent-space normal map: +X along the tangent, +Y along the bitangent, +Z along N.
    if (material.normalTexture != INVALIDDESCRIPTORHANDLE)
    {
        float3 tn = SampleTexture(material.normalTexture, uv).xyz * 2.0 - 1.0;
        tn.xy *= material.normalScale;
        hit.Ns = normalize(hit.T * tn.x + hit.B * tn.y + hit.N * tn.z);
    }

    float3 baseColour = material.baseColour.rgb;
    if (material.albedoTexture != INVALIDDESCRIPTORHANDLE)
        baseColour *= SampleTexture(material.albedoTexture, uv).rgb;

    float roughness = material.roughness;
    float metallic = material.metallic;
    if (material.roughMetalTexture != INVALIDDESCRIPTORHANDLE)
    {
        float4 roughMetal = SampleTexture(material.roughMetalTexture, uv);
        roughness *= roughMetal.g;
        metallic *= roughMetal.b;
    }

    OpenPBRBaseParams p = OpenPBRDefaults();
    p.baseColour = baseColour;
    p.baseMetalness = saturate(metallic);
    p.specularRoughness = saturate(roughness);
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

// FP16_MAX, what DLSS-RR expects as the specular hit distance of sky pixels.
static const float SKY_HIT_DISTANCE = 65504.0;

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
    return g;
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

float3 TracePath(float3 origin, float3 dir, inout uint rng, out PrimaryGuides guides)
{
    guides = DefaultGuides(dir);
    bool primarySpecular = false;

    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float3 segmentExtinction = float3(0.0, 0.0, 0.0);
    float prevBsdfPdf = 0.0;

    // maxBounces counts indirect bounces, so the primary segment comes on top.
    const uint maxSegments = g_Frame.maxBounces + 1;

    for (uint bounce = 0; bounce < maxSegments; ++bounce)
    {
        HitInfo hit = TraceClosest(origin, dir);

        if (bounce == 1 && primarySpecular)
            guides.specularHitDistance = hit.hit ? hit.t : SKY_HIT_DISTANCE;

        // Absorption along the segment just travelled, if it was inside a medium.
        if (any(segmentExtinction > 0.0))
        {
            if (!hit.hit)
                break;
            throughput *= exp(-segmentExtinction * hit.t);
        }

        if (!hit.hit)
        {
            // The sky is never light sampled, so BSDF sampling gets its full contribution.
            radiance += throughput * EnvironmentRadiance(dir);

            // The sun disc is also reached by NEE, so weight it against light sampling.
            float3 Lsun = SunRadiance(dir);
            if (any(Lsun > 0.0))
            {
                float w = bounce == 0 ? 1.0 : PowerHeuristic(prevBsdfPdf, LightPdf(origin, dir, hit));
                radiance += throughput * Lsun * w;
            }
            break;
        }

        // Emission found by BSDF sampling. Emissive surfaces are not in the light buffer yet; once
        // they are, weight this against NEE with LightTreePmf (needs a triangle to light map).
        if (any(hit.emission > 0.0))
            radiance += throughput * hit.emission;

        // Material and shading basis for this hit.
        OpenPBRBaseParams mat = LoadMaterial(hit);
        float3 wo = -dir;

        // Opaque surfaces are treated as two-sided: face the frame towards the viewer so a back
        // face is not mistaken for the inside of a transmissive object.
        bool opaque = mat.transmissionWeight <= 0.0;
        if (opaque && dot(hit.Ng, wo) < 0.0)
        {
            hit.Ng = -hit.Ng;
            hit.N = -hit.N;
            hit.Ns = -hit.Ns;
        }

        SetShadingBasis(mat, hit.T, hit.B, hit.N, hit.Ns, wo);
        OpenPBRContext ctx = OpenPBRSetup(mat, wo);
        float3 extinction = MediumExtinction(mat);

        if (bounce == 0)
        {
            guides.position = float4(hit.position, 1.0);
            guides.normal = mat.geometryNormal;
            guides.roughness = mat.specularRoughness;
            // Stays 0 (no reflection data) unless the path continues with a specular reflection.
            guides.specularHitDistance = 0.0;
            ReflectanceGuides(mat, wo, guides);
        }

        // One hash covers the light sample (xy) and the Russian roulette decision (z).
        float3 lightU = Rand3(rng);

        LightSample ls = SampleLight(hit.position, mat, ctx, wo, opaque, lightU.xy, rng);
        float3 neeContribution = float3(0.0, 0.0, 0.0);
        bool traceShadowRay = false;
        if (ls.pdf > 0.0 && (!opaque || dot(ls.direction, hit.Ng) > 0.0))
        {
            float bsdfPdf;
            float3 fCos = OpenPBREvalPdf(mat, ctx, ls.direction, bsdfPdf);
            if (any(fCos > 0.0))
            {
                float w = ls.lightSamplingOnly ? 1.0 : PowerHeuristic(ls.pdf, bsdfPdf);
                float3 shadowTransmittance =
                    dot(ls.direction, hit.Ng) < 0.0 ? exp(-extinction * ls.distance) : float3(1.0, 1.0, 1.0);
                neeContribution = throughput * fCos * ls.radiance * shadowTransmittance * (w / ls.pdf);
                traceShadowRay = true;
            }
        }

        // Continue the path by sampling the BSDF, unless this is the last segment.
        bool continuePath = false;
        if (bounce + 1 < maxSegments)
        {
            BSDFSample bs = OpenPBRSample(mat, ctx, Rand3(rng));
            continuePath = bs.lobe != LOBE_NONE;
            if (continuePath)
            {
                primarySpecular = bounce == 0 && bs.lobe == LOBE_SPECULAR_REFLECTION;
                throughput *= bs.weight;
                prevBsdfPdf = bs.pdf;

                // Heading into the object means travelling through its medium until the next hit.
                segmentExtinction = dot(bs.wi, hit.Ng) < 0.0 ? extinction : float3(0.0, 0.0, 0.0);
                dir = bs.wi;
            }
        }

        if (traceShadowRay)
            radiance += neeContribution * CastShadowRay(hit.position, hit.Ng, ls.direction, ls.distance);

        if (!continuePath)
            break;

        origin = OffsetRay(hit.position, hit.Ng, dir);

        // Russian roulette.
        if (bounce >= RR_START_BOUNCE)
        {
            float q = saturate(max(throughput.x, max(throughput.y, throughput.z)));
            if (lightU.z >= q)
                break;
            throughput /= q;
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
    g_SpecularHitDistance[pixel] = guides.specularHitDistance;
#endif
}

[shader("miss")]
void Miss(inout HitPayload payload)
{
    payload.hitT = -1.0;
}

// Vertex indices of the triangle being hit, already rebased into the shared streams.
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
    return g_TexCoords[tri.x] * bary.x + g_TexCoords[tri.y] * bary.y + g_TexCoords[tri.z] * bary.z;
}

[shader("closesthit")]
void ClosestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
    uint3 tri = LoadTriangle(prim);
    float3 bary = Barycentrics(attr);

    float3 p0 = g_Positions[tri.x];
    float3 p1 = g_Positions[tri.y];
    float3 p2 = g_Positions[tri.z];
    float3 objectNormal = g_Normals[tri.x] * bary.x + g_Normals[tri.y] * bary.y + g_Normals[tri.z] * bary.z;
    float4 objectTangent = g_Tangents[tri.x] * bary.x + g_Tangents[tri.y] * bary.y + g_Tangents[tri.z] * bary.z;

    float3x3 linearToWorld = (float3x3)ObjectToWorld3x4();
    float3x3 cofactor = float3x3(cross(linearToWorld[1], linearToWorld[2]),
                                 cross(linearToWorld[2], linearToWorld[0]),
                                 cross(linearToWorld[0], linearToWorld[1]));
    float det = dot(linearToWorld[0], cofactor[0]);
    float handedness = det < 0.0 ? -1.0 : 1.0;

    float3 Ng = normalize(mul(cofactor, cross(p1 - p0, p2 - p0)) * handedness);

    payload.hitT = RayTCurrent();
    payload.Ng = Ng;
    payload.materialId = prim.materialIndex;
    payload.N = normalize(mul(cofactor, objectNormal) * handedness);
    payload.T = normalize(mul(linearToWorld, objectTangent.xyz));
    // A mirroring transform flips cross(N, T), so fold its handedness into the bitangent sign.
    payload.tangentSign = (objectTangent.w < 0.0 ? -1.0 : 1.0) * handedness;
    payload.uv = InterpolateTexCoord(tri, bary);
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

[shader("anyhit")]
void ShadowAnyHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
    GpuMaterial material = LoadGpuMaterial(prim.materialIndex);

    float4 albedo = material.baseColour;
    if (material.albedoTexture != INVALIDDESCRIPTORHANDLE)
        albedo *= SampleTexture(material.albedoTexture, InterpolateTexCoord(LoadTriangle(prim), Barycentrics(attr)));

    if (material.alphaMode == ALPHA_MODE_MASK)
    {
        if (albedo.a < material.alphaCutoff)
            IgnoreHit();
        AcceptHitAndEndSearch();
    }

    if (material.alphaMode == ALPHA_MODE_BLEND)
    {
        payload.visibility *= (1.0 - albedo.a) * lerp(float3(1.0, 1.0, 1.0), albedo.rgb, material.transmission);
        if (dot(payload.visibility, 0.333) > 0.001)
            IgnoreHit();
    }

    AcceptHitAndEndSearch();
}
