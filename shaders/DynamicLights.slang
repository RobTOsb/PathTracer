#include "ShaderInterop.h"
#include "LightTreeInterop.h"

ConstantBuffer<DynamicLightConstants> g_PushConstants : register(b0);

float3 TransformDynamicLightPoint(DynamicLightTransform t, float3 p)
{
    return p.x * t.rows[0].xyz + p.y * t.rows[1].xyz + p.z * t.rows[2].xyz + float3(t.rows[0].w, t.rows[1].w, t.rows[2].w);
}

[shader("compute")]
[numthreads(DYNAMIC_LIGHTS_GROUP_SIZE, 1, 1)]
void TransformDynamicLights(uint3 dispatchThread : SV_DispatchThreadID)
{
    uint i = g_PushConstants.firstThread + dispatchThread.x;
    uint dynamicLights = g_PushConstants.dynamicLightCount;
    if (i >= dynamicLights + g_PushConstants.pointLightCount)
        return;

    RWStructuredBuffer<GpuLight> lights = ResourceDescriptorHeap[g_PushConstants.lightsHandle];
    uint target = g_PushConstants.staticLightCount + i;
    if (i >= dynamicLights)
    {
        StructuredBuffer<GpuLight> pointLights = ResourceDescriptorHeap[g_PushConstants.pointLightsHandle];
        lights[target] = pointLights[i - dynamicLights];
        return;
    }

    StructuredBuffer<GpuLight> objectLights = ResourceDescriptorHeap[g_PushConstants.objectLightsHandle];
    StructuredBuffer<uint> lightInstances = ResourceDescriptorHeap[g_PushConstants.lightInstancesHandle];
    StructuredBuffer<DynamicLightTransform> transforms = ResourceDescriptorHeap[g_PushConstants.transformsHandle];
    GpuLight light = objectLights[i];
    DynamicLightTransform t = transforms[lightInstances[i]];
    // The UVs in w stay with their corners.
    light.v1.xyz = TransformDynamicLightPoint(t, light.v1.xyz);
    light.v2.xyz = TransformDynamicLightPoint(t, light.v2.xyz);
    light.v3.xyz = TransformDynamicLightPoint(t, light.v3.xyz);
    // A mirroring transform flips the winding, and so the side the one-sided light emits from.
    if (t.swapCorners != 0)
    {
        float4 v2 = light.v2;
        light.v2 = light.v3;
        light.v3 = v2;
    }
    lights[target] = light;
}
