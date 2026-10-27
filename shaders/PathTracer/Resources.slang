// Path tracer bindings and the loads that go through them. Registers must match PathTracerPass.
// The only module that includes the interop headers, so their types are declared once.
// FrameConstants holds the SkyAtmosphere module's CloudParameters.
import SkyAtmosphere;
#define SKY_ATMOSPHERE_IMPORTED
#include "PathTracerInterop.h"
#include "DLSSUtil.h"

import LightTree;

// Macros do not cross module boundaries, so the ones other modules use become exported constants
// (PI comes from OpenPBRBSDF).
static const float kTwoPi = TWO_PI;
static const float kFltMax = FLT_MAX;
static const uint kInvalidDescriptorHandle = INVALIDDESCRIPTORHANDLE;
static const uint kInvalidLightIndex = INVALID_LIGHT_INDEX;
#undef TWO_PI
#undef FLT_MAX
#undef INVALIDDESCRIPTORHANDLE
#undef INVALID_LIGHT_INDEX
static const float TWO_PI = kTwoPi;
static const float FLT_MAX = kFltMax;
static const uint INVALIDDESCRIPTORHANDLE = kInvalidDescriptorHandle;
static const uint INVALID_LIGHT_INDEX = kInvalidLightIndex;

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
RWStructuredBuffer<GpuCameraState> g_CameraState : register(u8); // per pixel, see PathTracer/CameraState.slang
// Per TLAS instance (InstanceIndex()), last frame's ObjectToWorld3x4() as three rows, for the motion of moving
// geometry (RenderGPUScene::GetPreviousTransformBuffer).
StructuredBuffer<float4> g_PreviousTransforms : register(t16);
#endif
// The reference's sum of every path so far per pixel (RGBA32F); PathTracerPass binds it only under REFERENCE.
#ifdef REFERENCE
RWTexture2D<float4> g_Accumulation : register(u1);
#endif
RWStructuredBuffer<GpuDeferredBranch> g_DeferredBranches : register(u9); // per pixel, see PathTracer/GlassSplit.slang

// Index of this pixel's row in the per-pixel buffers.
uint PixelIndex()
{
    uint2 pixel = DispatchRaysIndex().xy;
    return pixel.y * DispatchRaysDimensions().x + pixel.x;
}

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
StructuredBuffer<GpuLightAliasEntry> g_LightPower : register(t13);
StructuredBuffer<uint> g_LightEmitters : register(t14);    // per light, its emissive instance
StructuredBuffer<float> g_EmissionScales : register(t15); // per emissive instance, see RenderGPUScene::SetEmissionScale

#ifdef ATMOSPHERE
Texture2D<float4> g_SkyViewLUT : register(t11);
StructuredBuffer<float4> g_AtmosphereLighting : register(t12);
SamplerState g_LinearClamp : register(s2);
#endif

PrimitiveDesc LoadPrimitive(uint index)
{
    return g_Primitives[index];
}

// What the light's instance emits now, relative to the material (animated lights dim and switch off).
float LightEmissionScale(uint lightIndex)
{
    return g_EmissionScales[g_LightEmitters[lightIndex]];
}

// The two float16 values of a GpuMaterial field, low half in x (Math's UnpackHalf2, which imports this).
float2 UnpackMaterialHalves(uint bits)
{
    return float2(f16tof32(bits & 0xFFFF), f16tof32(bits >> 16));
}

// The row's GPU_MATERIAL_SIMPLE bit, which picks TracePath's shading specialisation.
bool IsSimpleMaterial(uint index)
{
    return (g_Materials[index].emissiveFlags.y & GPU_MATERIAL_SIMPLE) != 0;
}

// TEMPORARY test: every blended material is read as a mask cut at 0.5, the cut its micromaps' unknown
// micro-triangles already lean by (kBlendShadowOpacity), so the baked scene cache is unchanged.
static const bool FORCE_BLEND_AS_MASK = false;

// UnpackGpuMaterial's shader side. Fields the caller does not read are never loaded or converted.
MaterialData LoadGpuMaterial(uint index)
{
    GpuMaterial p = g_Materials[index];
    uint flags = p.emissiveFlags.y;
    float2 baseColour0 = UnpackMaterialHalves(p.baseColour.x);
    float2 baseColour1 = UnpackMaterialHalves(p.baseColour.y);
    float2 roughnessMetallic = UnpackMaterialHalves(p.roughnessMetallic);
    float2 normalScaleAlphaCutoff = UnpackMaterialHalves(p.normalScaleAlphaCutoff);
    float2 transmissionIor = UnpackMaterialHalves(p.transmissionIor);
    float2 emissive = UnpackMaterialHalves(p.emissiveFlags.x);
    float2 attenuation0 = UnpackMaterialHalves(p.attenuation.x);
    float2 attenuation1 = UnpackMaterialHalves(p.attenuation.y);
    float2 subsurfaceColour0 = UnpackMaterialHalves(p.subsurfaceColourWeight.x);
    float2 subsurfaceColour1 = UnpackMaterialHalves(p.subsurfaceColourWeight.y);
    float2 subsurfaceRadius0 = UnpackMaterialHalves(p.subsurfaceRadiusAnisotropy.x);
    float2 subsurfaceRadius1 = UnpackMaterialHalves(p.subsurfaceRadiusAnisotropy.y);
    float2 sheenColourRoughness0 = UnpackMaterialHalves(p.sheenColourRoughness.x);
    float2 sheenColourRoughness1 = UnpackMaterialHalves(p.sheenColourRoughness.y);
    float2 specularColourWeight0 = UnpackMaterialHalves(p.specularColourWeight.x);
    float2 specularColourWeight1 = UnpackMaterialHalves(p.specularColourWeight.y);

    MaterialData m;
    m.baseColour = float4(baseColour0, baseColour1);
    m.roughness = roughnessMetallic.x;
    m.metallic = roughnessMetallic.y;
    m.normalScale = normalScaleAlphaCutoff.x;
    m.alphaCutoff = normalScaleAlphaCutoff.y;
    m.albedoTexture = p.albedoTexture;
    m.roughMetalTexture = p.roughMetalTexture;
    m.normalTexture = p.normalTexture;
    m.alphaMode = (flags >> GPU_MATERIAL_ALPHA_MODE_SHIFT) & 3;
    m.transmission = transmissionIor.x;
    m.ior = transmissionIor.y;
    m.transmissionFlags = (flags >> GPU_MATERIAL_TRANSMISSION_FLAGS_SHIFT) & 3;
    m.transmissionTexture = p.transmissionTexture;
    m.emissive = float3(emissive, f16tof32(flags & 0xFFFF));
    m.emissiveTexture = p.emissiveTexture;
    m.attenuationColour = float3(attenuation0, attenuation1.x);
    m.attenuationDistance = attenuation1.y;
    m.subsurfaceColour = float3(subsurfaceColour0, subsurfaceColour1.x);
    m.subsurfaceWeight = subsurfaceColour1.y;
    m.subsurfaceWeightTexture = p.subsurfaceWeightTexture;
    m.subsurfaceColourTexture = p.subsurfaceColourTexture;
    m.anisotropy = UnpackMaterialHalves(p.anisotropy);
    m.anisotropyTexture = p.anisotropyTexture;
    m.subsurfaceRadius = float3(subsurfaceRadius0, subsurfaceRadius1.x);
    m.subsurfaceScatterAnisotropy = subsurfaceRadius1.y;
    m.subsurfaceRadiusTexture = p.subsurfaceRadiusTexture;
    m.subsurfaceScaleTexture = p.subsurfaceScaleTexture;
    m.subsurfaceWeightChannel = (flags >> GPU_MATERIAL_WEIGHT_CHANNEL_SHIFT) & 3;
    m.subsurfaceScaleChannel = (flags >> GPU_MATERIAL_SCALE_CHANNEL_SHIFT) & 3;
    m.sheenColour = float3(sheenColourRoughness0, sheenColourRoughness1.x);
    m.sheenRoughness = sheenColourRoughness1.y;
    m.sheenColourTexture = p.sheenColourTexture;
    m.sheenRoughnessTexture = p.sheenRoughnessTexture;
    m.coatWeight = (p.coatWeightRoughnessNormalScale & 0xFF) / 255.0;
    m.coatRoughness = ((p.coatWeightRoughnessNormalScale >> 8) & 0xFF) / 255.0;
    m.coatNormalScale = f16tof32(p.coatWeightRoughnessNormalScale >> 16);
    m.coatWeightTexture = p.coatWeightTexture;
    m.coatRoughnessTexture = p.coatRoughnessTexture;
    m.coatNormalTexture = p.coatNormalTexture;
    m.specularColour = float3(specularColourWeight0, specularColourWeight1.x);
    m.specularWeight = specularColourWeight1.y;
    m.specularTexture = p.specularTexture;
    m.specularColourTexture = p.specularColourTexture;
    if (FORCE_BLEND_AS_MASK && m.alphaMode == ALPHA_MODE_BLEND)
    {
        m.alphaMode = uint(ALPHA_MODE_MASK);
        m.alphaCutoff = 0.5;
    }
    return m;
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
