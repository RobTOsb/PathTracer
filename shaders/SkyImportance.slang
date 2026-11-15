// The sky importance map, run by SkyAtmospherePass each frame after the sky-view LUT and the cloud panorama: the
// luminance of the sky without the sun disc, as the path tracer's rays that leave the scene see it (the cloud panorama
// where it is rendered, otherwise the clear sky of the sky-view LUT), over SkyImportanceUv's equal-area texels, and the
// pyramid of its sums the path tracer descends to sample the sky in proportion (SampleSkyImportance in
// PathTracer/Sky.slang).

import Clouds;
import SkyAtmosphere;

ConstantBuffer<SkyImportanceConstants> g_PushConstants : register(b0);
Texture2D<float4> g_SkyViewLUT : register(t0);
Texture2D<float4> g_Panorama : register(t1);
RWStructuredBuffer<float> g_ImportanceOutput : register(u0);
SamplerState g_LinearClamp : register(s2);

// A 16 x 16 tile of one level, row-major.
groupshared float gs_Sums[256];

// Sums the size x size tile of level - 1 in gs_Sums (the group's tile, at group) count times by 2 x 2 into the levels
// from level on, each written to the pyramid and kept in gs_Sums for the next.
void ReduceTile(uint2 local, uint2 group, uint size, uint level, uint count)
{
    for (uint i = 0; i < count; ++i)
    {
        uint halfSize = size >> 1;
        bool active = all(local < halfSize);
        float sum = 0.0;
        if (active)
        {
            uint corner = local.y * 32u + local.x * 2u;
            sum = gs_Sums[corner] + gs_Sums[corner + 1u] + gs_Sums[corner + 16u] + gs_Sums[corner + 17u];
        }
        GroupMemoryBarrierWithGroupSync();
        if (active)
        {
            gs_Sums[local.y * 16u + local.x] = sum;
            g_ImportanceOutput[SkyImportanceIndex(level + i, group * halfSize + local)] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
        size = halfSize;
    }
}

float3 SkyRadianceForUnitSun(float3 dir)
{
    if (g_PushConstants.usePanorama != 0)
        return g_Panorama.SampleLevel(g_LinearClamp, CloudPanoramaUv(dir), 0.0).rgb;
    float2 uv = SkyViewLutUvFromDirection(g_PushConstants.planetRadius, g_PushConstants.viewHeight, dir,
                                          g_PushConstants.sunDirection);
    return g_SkyViewLUT.SampleLevel(g_LinearClamp, uv, 0.0).rgb;
}

// Level 0, the mean luminance of 2 x 2 points in each texel, and the 16 x 16 tiles' sums up to level 4 (8 x 8).
// Dispatched as (SKY_IMPORTANCE_SIZE / 16)^2 groups.
[shader("compute")]
[numthreads(16, 16, 1)]
void SkyImportanceTexels(uint3 id: SV_DispatchThreadID, uint3 local: SV_GroupThreadID, uint3 group: SV_GroupID)
{
    float luminance = 0.0;
    for (uint s = 0; s < 4u; ++s)
    {
        float2 uv = (float2(id.xy) + float2(0.25 + 0.5 * float(s & 1u), 0.25 + 0.5 * float(s >> 1))) / float(kSkyImportanceSize);
        float3 radiance = SkyRadianceForUnitSun(SkyImportanceDirection(uv)) * g_PushConstants.sunIlluminance;
        luminance += dot(radiance, float3(0.2126, 0.7152, 0.0722));
    }
    // Finite and positive, so no NaN or infinity reaches the sums.
    luminance = luminance < 3.0e38 ? max(luminance * 0.25, 0.0) : 0.0;
    g_ImportanceOutput[SkyImportanceIndex(0u, id.xy)] = luminance;
    gs_Sums[local.y * 16u + local.x] = luminance;
    GroupMemoryBarrierWithGroupSync();
    ReduceTile(local.xy, group.xy, 16u, 1u, 4u);
}

// Levels 5 to 7 from the 8 x 8 of level 4, in one group.
[shader("compute")]
[numthreads(16, 16, 1)]
void SkyImportanceReduce(uint3 local: SV_GroupThreadID)
{
    gs_Sums[local.y * 16u + local.x] = all(local.xy < 8u) ? g_ImportanceOutput[SkyImportanceIndex(4u, local.xy)] : 0.0;
    GroupMemoryBarrierWithGroupSync();
    ReduceTile(local.xy, uint2(0u, 0u), 8u, 5u, 3u);
}
