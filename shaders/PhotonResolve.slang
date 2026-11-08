// Turns each frame's photon deposits (PathTracer/Photons.slang) into irradiance over each cell's face and blends it,
// and the frame's photon count, into the cell's running averages, then empties the frame's sums for the next. A new
// cell starts from zero, the frames before it existed having had no photons there: starting from its first frame alone
// made every lone photon a bright speckle. Cells that get photons less often blend their irradiance more slowly, so
// that they average over about as many photons as the rest. Cells nothing was deposited in for evictFrames frames are emptied, so the
// grid frees the rows of what the camera has moved away from.
#include "ShaderInterop.h"
#include "PhotonInterop.h"

ConstantBuffer<PhotonResolveConstants> g_PushConstants : register(b0);

[shader("compute")]
[numthreads(PHOTON_RESOLVE_GROUP_SIZE, 1, 1)]
void ResolvePhotons(uint3 id : SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= PHOTON_GRID_CAPACITY)
        return;
    RWStructuredBuffer<uint> keys = ResourceDescriptorHeap[g_PushConstants.gridKeys];
    RWStructuredBuffer<uint4> accum = ResourceDescriptorHeap[g_PushConstants.gridAccum];
    RWStructuredBuffer<float4> cache = ResourceDescriptorHeap[g_PushConstants.gridCache];

    if (g_PushConstants.reset != 0)
    {
        keys[slot] = 0u;
        accum[slot] = uint4(0u, 0u, 0u, 0u);
        cache[slot] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    if (keys[slot] == 0u)
        return;

    uint4 frame = accum[slot];
    float4 cell = cache[slot];
    uint meta = asuint(cell.w);
    float averageCount = f16tof32(meta & 0xFFFFu);
    uint idle = meta >> 16;
    uint count = frame.w & PHOTON_COUNT_MASK;
    uint levelBits = frame.w & ~PHOTON_COUNT_MASK;
    bool deposited = count != 0u;
    idle = deposited ? 0u : idle + 1u;
    if (idle > g_PushConstants.evictFrames)
    {
        keys[slot] = 0u;
        accum[slot] = uint4(0u, 0u, 0u, 0u);
        cache[slot] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float size = exp2(float(int(levelBits >> PHOTON_LEVEL_SHIFT) - PHOTON_LEVEL_BIAS));
    float3 irradiance = float3(frame.xyz) * (g_PushConstants.unitScale / (size * size));
    // A sparse cell averages over more frames, about as many photons as a cell with minPhotons a frame. The weight is
    // from the photon count before this frame's, which a lucky frame would otherwise weigh itself up by.
    float blend = g_PushConstants.blend;
    float irradianceBlend = max(blend * saturate(averageCount / g_PushConstants.minPhotons), g_PushConstants.minBlend);
    float3 average = lerp(cell.rgb, irradiance, irradianceBlend);
    averageCount = lerp(averageCount, float(count), blend);
    cache[slot] = float4(average, asfloat(f32tof16(averageCount) | (min(idle, 0xFFFFu) << 16)));
    accum[slot] = uint4(0u, 0u, 0u, levelBits);
}
