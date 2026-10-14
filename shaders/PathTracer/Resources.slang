// Path tracer bindings and the loads that go through them. Registers must match PathTracerPass.
// The only module that includes the interop headers, so their types are declared once.
#include "PathTracerInterop.h"
#include "DLSSUtil.h"

import LightTree;

// Macros do not cross module boundaries, so the ones other modules use become exported constants
// (PI comes from OpenPBRBaseSubstrate).
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
