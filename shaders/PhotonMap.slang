// Caustic photons' screen and photon map passes (PhotonInterop.h), after Yang and Ouyang's Adaptive Anisotropic Photon
// Scattering (Ray Tracing Gems II, ch. 30). Each frame, in order:
//   UpdatePhotonMap: each photon map texel's photon share from last frame's feedback (their eq. 30.2 and 30.3).
//   ScanPhotonBlocks, ScanPhotonTotals: the shares' running sums, which PhotonRayGen spreads the photons by.
//   (PhotonRayGen traces and records the photons, the path tracer marks the receivers.)
//   SplatPhotons: every record as an anisotropic Gaussian footprint onto the receivers it covers, and its feedback.
//   CompositeCaustics: the splatted irradiance into the path tracer's radiance, and how much it changed since last
//   frame, which the next frame's feedback reads.
#include "PathTracerInterop.h"

ConstantBuffer<PhotonPassConstants> g_PushConstants : register(b0);
ConstantBuffer<FrameConstants> g_Frame : register(b1); // the path tracer's

static const float kPi = 3.14159265358979;

// The cell at distance d along a Hilbert curve over a 2^order square of cells (as PathTracer/Photons.slang).
uint2 HilbertCell(uint d, uint order)
{
    uint2 c = uint2(0u, 0u);
    for (uint s = 1u; s < (1u << order); s <<= 1u)
    {
        uint rx = 1u & (d >> 1u);
        uint ry = 1u & (d ^ rx);
        if (ry == 0u)
        {
            if (rx == 1u)
                c = s - 1u - c;
            c = c.yx;
        }
        c += s * uint2(rx, ry);
        d >>= 2u;
    }
    return c;
}

// The distance along the Hilbert curve of cell c, the inverse of HilbertCell.
uint HilbertIndex(uint2 c, uint order)
{
    uint n = 1u << order;
    uint d = 0u;
    for (uint s = n >> 1u; s > 0u; s >>= 1u)
    {
        uint rx = (c.x & s) != 0u ? 1u : 0u;
        uint ry = (c.y & s) != 0u ? 1u : 0u;
        d += s * s * ((3u * rx) ^ ry);
        if (ry == 0u)
        {
            if (rx == 1u)
                c = n - 1u - c;
            c = c.yx;
        }
    }
    return d;
}

// As PathTracer/Math.slang.
float3 OctDecode(uint bits)
{
    float2 e = max(float2(int2(int(bits << 16) >> 16, int(bits) >> 16)) * (1.0 / 32767.0), -1.0);
    float3 v = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-v.z);
    v.x += v.x >= 0.0 ? -t : t;
    v.y += v.y >= 0.0 ? -t : t;
    return normalize(v);
}

void BuildOrthonormalBasis(float3 n, out float3 t, out float3 b)
{
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

uint Hash(uint3 v)
{
    uint h = v.x * 0x8da6b343u ^ v.y * 0xd8163841u ^ v.z * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return h;
}

float Luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// The pixel a world point projects to through a view-projection, false behind the camera or off screen.
bool ProjectToPixel(float3 position, float4x4 viewProj, out int2 pixel)
{
    float4 clip = mul(float4(position, 1.0), viewProj);
    pixel = int2(0, 0);
    if (clip.w <= 1.0e-6)
        return false;
    float2 ndc = clip.xy / clip.w;
    float2 p = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * float2(g_Frame.photons.width, g_Frame.photons.height);
    pixel = int2(floor(p));
    return all(pixel >= 0) && pixel.x < int(g_Frame.photons.width) && pixel.y < int(g_Frame.photons.height);
}

// Each texel's share of the photons. Texels over no caster's rectangle get none. Otherwise, from the photons it sent
// last frame that landed where the camera sees (the feedback): their average footprint area on screen a and the
// caustics' change v where they landed suggest last frame's photons times a / targetArea, so that every photon covers
// about targetArea pixels, plus varianceGain v where the caustics flicker; at least minDensity, so that every caster
// keeps being found. That is blended with the 3x3 neighbourhood's last shares by temporalWeight so that small features
// change smoothly. Shares count photons per texel, last frame's scaled to the photons traced, so minDensity and the
// gain mean photons; the scan then spreads the frame's photons in proportion, whatever their total.
[shader("compute")]
[numthreads(PHOTON_GROUP_SIZE, 1, 1)]
void UpdatePhotonMap(uint3 id : SV_DispatchThreadID)
{
    uint texel = id.x;
    if (texel >= PHOTON_MAP_TEXELS)
        return;
    PhotonParameters ph = g_Frame.photons;
    RWStructuredBuffer<float> previous = ResourceDescriptorHeap[g_PushConstants.densityPrevious];
    RWStructuredBuffer<float> density = ResourceDescriptorHeap[ph.density];
    RWStructuredBuffer<uint4> feedback = ResourceDescriptorHeap[ph.feedback];
    RWStructuredBuffer<float> blockPrefix = ResourceDescriptorHeap[ph.blockPrefix];
    StructuredBuffer<GpuPhotonRect> rects = ResourceDescriptorHeap[ph.rects];

    uint4 f = feedback[texel];
    feedback[texel] = uint4(0u, 0u, 0u, 0u);

    uint2 xy = HilbertCell(texel, PHOTON_MAP_ORDER);
    float texelSize = ph.mapScale / float(PHOTON_MAP_SIZE);
    float2 lo = ph.mapOrigin + float2(xy) * texelSize;
    float2 hi = lo + texelSize;
    bool covered = false;
    for (uint r = 0; r < ph.rectCount && !covered; ++r)
    {
        GpuPhotonRect rect = rects[r];
        covered = all(lo <= rect.hi) && all(hi >= rect.lo);
    }
    if (!covered)
    {
        density[texel] = 0.0;
        return;
    }
    float previousTotal = blockPrefix[PHOTON_MAP_BLOCKS];
    if (g_PushConstants.reset != 0 || !(previousTotal > 0.0))
    {
        density[texel] = 1.0;
        return;
    }
    float toPhotons = float(ph.photonCount) / previousTotal;

    float suggested = 0.0;
    if (f.z > 0u)
    {
        float area = float(f.x) / (PHOTON_AREA_SCALE * float(f.z));
        float variance = float(f.y) / (PHOTON_VARIANCE_SCALE * float(f.z));
        suggested = previous[texel] * toPhotons * area / g_PushConstants.targetArea + g_PushConstants.varianceGain * variance;
    }
    suggested = max(suggested, g_PushConstants.minDensity);

    // The covered neighbours' last shares, weighted 4 / 2 / 1 from the centre out.
    float sum = 0.0;
    float weights = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int2 n = int2(xy) + int2(dx, dy);
            if (any(n < 0) || any(n >= int(PHOTON_MAP_SIZE)))
                continue;
            float last = previous[HilbertIndex(uint2(n), PHOTON_MAP_ORDER)];
            if (!(last > 0.0))
                continue;
            float w = (dx == 0 ? 2.0 : 1.0) * (dy == 0 ? 2.0 : 1.0);
            sum += w * last;
            weights += w;
        }
    }
    float filtered = weights > 0.0 ? sum / weights * toPhotons : suggested;
    float share = lerp(filtered, suggested, g_PushConstants.temporalWeight);
    density[texel] = clamp(share, g_PushConstants.minDensity, g_PushConstants.maxDensity);
}

groupshared float s_WaveSums[PHOTON_MAP_BLOCK];

// The inclusive running sum of a group's values, one per thread, the group's total in total.
float GroupInclusiveSum(float value, uint index, out float total)
{
    uint lanes = WaveGetLaneCount();
    uint wave = index / lanes;
    float inclusive = WavePrefixSum(value) + value;
    if (WaveGetLaneIndex() == lanes - 1u)
        s_WaveSums[wave] = inclusive;
    GroupMemoryBarrierWithGroupSync();
    if (index == 0u)
    {
        float run = 0.0;
        for (uint w = 0; w < PHOTON_MAP_BLOCK / lanes; ++w)
        {
            float waveSum = s_WaveSums[w];
            s_WaveSums[w] = run;
            run += waveSum;
        }
        s_WaveSums[PHOTON_MAP_BLOCK - 1u] = run;
    }
    GroupMemoryBarrierWithGroupSync();
    total = s_WaveSums[PHOTON_MAP_BLOCK - 1u];
    return inclusive + s_WaveSums[wave];
}

// One group per block: the inclusive running sums of its texels' shares, and the block's total into blockPrefix.
[shader("compute")]
[numthreads(PHOTON_MAP_BLOCK, 1, 1)]
void ScanPhotonBlocks(uint3 group : SV_GroupID, uint index : SV_GroupIndex)
{
    PhotonParameters ph = g_Frame.photons;
    RWStructuredBuffer<float> density = ResourceDescriptorHeap[ph.density];
    RWStructuredBuffer<float> texelPrefix = ResourceDescriptorHeap[ph.texelPrefix];
    RWStructuredBuffer<float> blockPrefix = ResourceDescriptorHeap[ph.blockPrefix];
    uint texel = group.x * PHOTON_MAP_BLOCK + index;
    float total;
    texelPrefix[texel] = GroupInclusiveSum(density[texel], index, total);
    if (index == 0u)
        blockPrefix[group.x] = total;
}

// One group: the blocks' totals into the sums of those before each, the whole at PHOTON_MAP_BLOCKS. Also empties the
// photon records for this frame's photons.
[shader("compute")]
[numthreads(PHOTON_MAP_BLOCKS, 1, 1)]
void ScanPhotonTotals(uint index : SV_GroupIndex)
{
    PhotonParameters ph = g_Frame.photons;
    RWStructuredBuffer<float> blockPrefix = ResourceDescriptorHeap[ph.blockPrefix];
    RWStructuredBuffer<uint> recordCount = ResourceDescriptorHeap[ph.recordCount];
    float value = blockPrefix[index];
    float total;
    float inclusive = GroupInclusiveSum(value, index, total);
    blockPrefix[index] = inclusive - value;
    if (index == 0u)
    {
        blockPrefix[PHOTON_MAP_BLOCKS] = total;
        recordCount[0] = 0u;
    }
}

// One thread per photon record: its footprint is a Gaussian on its surface of its covariance, widened by half a
// pixel's footprint at its distance so that a focus stays a few pixels across rather than aliasing, cut off at three
// standard deviations. It visits the pixels its cut-off ellipse's bounds project to (every few past
// PHOTON_MAX_SPLAT_PIXELS, weighted up to match), and adds its flux times the Gaussian at every receiver on its surface
// that faces its way, as fixed point irradiance, rounded at random so that dim photons add up. Then the feedback, for
// a photon that lands where the camera sees: its area in pixels and last frame's caustics change there, into the
// texel it came from.
[shader("compute")]
[numthreads(PHOTON_GROUP_SIZE, 1, 1)]
void SplatPhotons(uint3 id : SV_DispatchThreadID)
{
    PhotonParameters ph = g_Frame.photons;
    RWStructuredBuffer<uint> recordCount = ResourceDescriptorHeap[ph.recordCount];
    uint i = id.x;
    if (i >= min(recordCount[0], PHOTON_RECORD_CAPACITY))
        return;
    RWStructuredBuffer<GpuPhotonRecord> records = ResourceDescriptorHeap[ph.records];
    RWStructuredBuffer<float4> receivers = ResourceDescriptorHeap[ph.receivers];
    RWStructuredBuffer<uint4> caustics = ResourceDescriptorHeap[g_PushConstants.caustics];
    GpuPhotonRecord record = records[i];

    float3 n = OctDecode(record.normal);
    float3 toCamera = g_Frame.invView[3].xyz - record.position;
    float distance = length(toCamera);
    float cosView = dot(n, toCamera) / max(distance, 1.0e-30);
    // The camera sees only the other side.
    if (cosView <= 0.0)
        return;
    float3 t, b;
    BuildOrthonormalBasis(n, t, b);
    float pixelSize = distance * g_Frame.rayConePixelSpread;
    float3 c = record.covariance + float3(0.25 * pixelSize * pixelSize, 0.0, 0.25 * pixelSize * pixelSize);
    float det = c.x * c.z - c.y * c.y;
    if (!(det > 0.0))
        return;
    float3 inverse = float3(c.z, -c.y, c.x) / det;
    float sigma = sqrt(max(c.x, c.z));
    float planeTolerance = 1.5 * sigma + 2.0 * pixelSize;

    // The cut-off ellipse's world bounds, through the camera.
    float3 extent = 3.0 * sqrt(max(c.x * t * t + 2.0 * c.y * t * b + c.z * b * b, 0.0));
    float2 lo = float2(1.0e30, 1.0e30);
    float2 hi = float2(-1.0e30, -1.0e30);
    float2 size = float2(ph.width, ph.height);
    for (uint k = 0; k < 8u; ++k)
    {
        float3 corner = record.position + extent * float3((k & 1u) != 0u ? 1.0 : -1.0, (k & 2u) != 0u ? 1.0 : -1.0,
                                                          (k & 4u) != 0u ? 1.0 : -1.0);
        float4 clip = mul(float4(corner, 1.0), g_Frame.viewProj);
        if (clip.w <= 1.0e-6)
            return;
        float2 ndc = clip.xy / clip.w;
        float2 p = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size;
        lo = min(lo, p);
        hi = max(hi, p);
    }
    int2 p0 = max(int2(floor(lo)), int2(0, 0));
    int2 p1 = min(int2(floor(hi)), int2(ph.width, ph.height) - 1);
    if (any(p1 < p0))
        return;
    uint2 box = uint2(p1 - p0 + 1);
    uint pixels = box.x * box.y;
    uint stride = pixels > PHOTON_MAX_SPLAT_PIXELS ? uint(ceil(sqrt(float(pixels) / float(PHOTON_MAX_SPLAT_PIXELS)))) : 1u;
    uint2 offset = stride > 1u ? uint2(Hash(uint3(i, g_PushConstants.frameIndex, 1u)), Hash(uint3(i, g_PushConstants.frameIndex, 2u))) % stride
                               : uint2(0u, 0u);
    float3 scale = record.power * (float(stride * stride) / (2.0 * kPi * sqrt(det) * ph.irradianceUnit));

    for (int y = p0.y + int(offset.y); y <= p1.y; y += int(stride))
    {
        for (int x = p0.x + int(offset.x); x <= p1.x; x += int(stride))
        {
            uint pixel = uint(y) * ph.width + uint(x);
            float4 position = receivers[2u * pixel];
            if (position.w == 0.0)
                continue;
            float3 delta = position.xyz - record.position;
            if (abs(dot(delta, n)) > planeTolerance)
                continue;
            float2 uv = float2(dot(delta, t), dot(delta, b));
            float q = inverse.x * uv.x * uv.x + 2.0 * inverse.y * uv.x * uv.y + inverse.z * uv.y * uv.y;
            if (q > 9.0)
                continue;
            if (dot(OctDecode(asuint(receivers[2u * pixel + 1u].w)), n) < 0.5)
                continue;
            float rounding = float(Hash(uint3(pixel, i, g_PushConstants.frameIndex)) >> 8) * (1.0 / 16777216.0);
            uint3 units = uint3(min(scale * exp(-0.5 * q), 16777216.0) + rounding);
            if (units.x != 0u)
                InterlockedAdd(caustics[pixel].x, units.x);
            if (units.y != 0u)
                InterlockedAdd(caustics[pixel].y, units.y);
            if (units.z != 0u)
                InterlockedAdd(caustics[pixel].z, units.z);
        }
    }

    // Feedback from a photon whose centre the camera sees.
    int2 centre;
    if (!ProjectToPixel(record.position, g_Frame.viewProj, centre))
        return;
    uint pixel = uint(centre.y) * ph.width + uint(centre.x);
    float4 position = receivers[2u * pixel];
    if (position.w == 0.0 || abs(dot(position.xyz - record.position, n)) > planeTolerance ||
        dot(OctDecode(asuint(receivers[2u * pixel + 1u].w)), n) < 0.5)
        return;
    RWStructuredBuffer<float2> historyPrevious = ResourceDescriptorHeap[g_PushConstants.historyPrevious];
    RWStructuredBuffer<uint4> feedback = ResourceDescriptorHeap[ph.feedback];
    float area = min(record.area * cosView / (pixelSize * pixelSize), 65536.0);
    float variance = saturate(historyPrevious[pixel].y);
    InterlockedAdd(feedback[record.texel].x, uint(area * PHOTON_AREA_SCALE + 0.5));
    InterlockedAdd(feedback[record.texel].y, uint(variance * PHOTON_VARIANCE_SCALE + 0.5));
    InterlockedAdd(feedback[record.texel].z, 1u);
}

// Per pixel: the splatted irradiance times the receiver's reflectance into the radiance, the sums emptied for the next
// frame. The caustics' change since last frame, relative to them, where the receiver was last frame: the variance the
// next frame's feedback reads.
[shader("compute")]
[numthreads(PHOTON_SCREEN_GROUP_SIZE, PHOTON_SCREEN_GROUP_SIZE, 1)]
void CompositeCaustics(uint3 id : SV_DispatchThreadID)
{
    PhotonParameters ph = g_Frame.photons;
    if (id.x >= ph.width || id.y >= ph.height)
        return;
    uint pixel = id.y * ph.width + id.x;
    RWStructuredBuffer<uint4> caustics = ResourceDescriptorHeap[g_PushConstants.caustics];
    RWStructuredBuffer<float4> receivers = ResourceDescriptorHeap[ph.receivers];
    RWStructuredBuffer<float2> historyPrevious = ResourceDescriptorHeap[g_PushConstants.historyPrevious];
    RWStructuredBuffer<float2> history = ResourceDescriptorHeap[g_PushConstants.history];

    uint4 sum = caustics[pixel];
    caustics[pixel] = uint4(0u, 0u, 0u, 0u);
    float3 irradiance = float3(sum.xyz) * ph.irradianceUnit;
    float4 position = receivers[2u * pixel];
    float luminance = 0.0;
    float variance = 0.0;
    if (position.w != 0.0)
    {
        RWTexture2D<float4> output = ResourceDescriptorHeap[g_PushConstants.output];
        float4 radiance = output[id.xy];
        radiance.rgb += receivers[2u * pixel + 1u].rgb * irradiance;
        output[id.xy] = radiance;
        luminance = Luminance(irradiance);
        int2 previousPixel;
        if (g_PushConstants.reset == 0 && ProjectToPixel(position.xyz, g_Frame.prevViewProj, previousPixel))
        {
            float previous = historyPrevious[uint(previousPixel.y) * ph.width + uint(previousPixel.x)].x;
            float both = luminance + previous;
            variance = both > 0.0 ? abs(luminance - previous) / both : 0.0;
        }
    }
    history[pixel] = float2(luminance, variance);
}
