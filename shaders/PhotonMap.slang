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

// A photon's footprint is a Gaussian on its surface, cut off at q = kSplatCutoff (three standard deviations). Its screen
// bounds and the test of each pixel's ray against its plane, before the pixel's receiver is read, cut off a little
// further out at kScreenCutoff (3.5), for receivers just off the plane, whose own q then decides.
static const float kSplatCutoff = 9.0;
static const float kScreenCutoff = 12.25;

// A photon record's splat, from SetupSplat.
struct Splat
{
    float3 position;
    float3 n;
    float3 t; // the footprint's frame on its surface, BuildOrthonormalBasis(n)
    float3 b;
    float3 inverse; // the inverse of the footprint's covariance, xx, xy, yy
    float planeTolerance; // how far off the footprint's plane a receiver may be
    float3 scale; // flux times the Gaussian's normalisation and the stride's weight, in fixed point irradiance units
    // The footprint plane's (t, b) coordinates of the pixel point (x, y) are g.xy / g.z, g = g0 + x gx + y gy, where
    // that pixel's ray meets the plane in front of the camera just when g.z > 0.
    float3 g0;
    float3 gx;
    float3 gy;
    int2 start; // the first pixel visited
    uint stride; // the pixels visited are every stride-th across and down
    uint2 count; // the pixels visited across and down
    uint record;
};

float QuadraticForm(float3 m, float2 v)
{
    return m.x * v.x * v.x + 2.0 * m.y * v.x * v.y + m.z * v.y * v.y;
}

// A point (w = 1) or direction (w = 0) through the view-projection to homogeneous pixel coordinates (x w, y w, w).
float3 ToScreen(float4 p)
{
    float4 clip = mul(p, g_Frame.viewProj);
    return float3((clip.x + clip.w) * (0.5 * float(g_Frame.photons.width)),
                  (clip.w - clip.y) * (0.5 * float(g_Frame.photons.height)), clip.w);
}

// The range along one screen axis of an ellipse given by its dual conic in homogeneous pixel coordinates: where the
// lines across that axis touch it, ww x^2 - 2 aw x + aa = 0 for the dual's entries aa (the axis's), aw and ww.
float2 ConicRange(float aa, float aw, float ww)
{
    float root = sqrt(max(aw * aw - aa * ww, 0.0));
    float a = (aw + root) / ww;
    float b = (aw - root) / ww;
    return float2(min(a, b), max(a, b));
}

// A photon record's splat, false if it has none: the camera sees the other side of its surface, or its footprint is
// degenerate, reaches behind the camera or misses the screen. The footprint is the Gaussian of the record's covariance
// widened by half a pixel's footprint at its distance, so that a focus stays a few pixels across rather than aliasing.
// Its cut-off ellipse projects to an ellipse on screen (the plane to the screen is a homography), whose bounds are the
// pixels visited, every few past PHOTON_MAX_SPLAT_PIXELS (weighted up to match, from a random first one).
bool SetupSplat(GpuPhotonRecord record, uint index, out Splat s, out float cosView, out float pixelSize)
{
    s = (Splat)0;
    s.record = index;
    s.position = record.position;
    s.n = OctDecode(record.normal);
    float3 toCamera = g_Frame.invView[3].xyz - record.position;
    float distance = length(toCamera);
    cosView = dot(s.n, toCamera) / max(distance, 1.0e-30);
    pixelSize = distance * g_Frame.rayConePixelSpread;
    // The camera sees only the other side.
    if (!(cosView > 0.0))
        return false;
    BuildOrthonormalBasis(s.n, s.t, s.b);
    float3 c = record.covariance + float3(0.25 * pixelSize * pixelSize, 0.0, 0.25 * pixelSize * pixelSize);
    float det = c.x * c.z - c.y * c.y;
    if (!(det > 0.0))
        return false;
    s.inverse = float3(c.z, -c.y, c.x) / det;
    s.planeTolerance = 1.5 * sqrt(max(c.x, c.z)) + 2.0 * pixelSize;

    // The footprint's plane to homogeneous pixel coordinates: h = h0 + u hu + v hv, a 3x3 matrix H of columns hu, hv, h0.
    float3 h0 = ToScreen(float4(record.position, 1.0));
    float3 hu = ToScreen(float4(s.t, 0.0));
    float3 hv = ToScreen(float4(s.b, 0.0));
    // The cut-off ellipse's dual conic on screen, H diag(kScreenCutoff c, -1) H^T; its ww entry is negative just when
    // the whole ellipse is in front of the camera.
    float3 cu = kScreenCutoff * (c.x * hu + c.y * hv);
    float3 cv = kScreenCutoff * (c.y * hu + c.z * hv);
    float xx = hu.x * cu.x + hv.x * cv.x - h0.x * h0.x;
    float yy = hu.y * cu.y + hv.y * cv.y - h0.y * h0.y;
    float ww = hu.z * cu.z + hv.z * cv.z - h0.z * h0.z;
    float xw = hu.x * cu.z + hv.x * cv.z - h0.x * h0.z;
    float yw = hu.y * cu.z + hv.y * cv.z - h0.y * h0.z;
    if (!(h0.z > 0.0) || !(ww < -1.0e-6 * h0.z * h0.z))
        return false;
    float2 size = float2(g_Frame.photons.width, g_Frame.photons.height);
    float2 xRange = clamp(ConicRange(xx, xw, ww), -1.0, size.x + 1.0);
    float2 yRange = clamp(ConicRange(yy, yw, ww), -1.0, size.y + 1.0);
    int2 p0 = max(int2(floor(float2(xRange.x, yRange.x))), int2(0, 0));
    int2 p1 = min(int2(floor(float2(xRange.y, yRange.y))), int2(size) - 1);
    if (any(p1 < p0))
        return false;

    uint2 box = uint2(p1 - p0 + 1);
    uint pixels = box.x * box.y;
    s.stride = pixels > PHOTON_MAX_SPLAT_PIXELS ? uint(ceil(sqrt(float(pixels) / float(PHOTON_MAX_SPLAT_PIXELS)))) : 1u;
    uint2 offset = s.stride > 1u ? uint2(Hash(uint3(index, g_PushConstants.frameIndex, 1u)),
                                         Hash(uint3(index, g_PushConstants.frameIndex, 2u))) % s.stride
                                 : uint2(0u, 0u);
    s.start = p0 + int2(offset);
    s.count.x = box.x > offset.x ? (box.x - offset.x + s.stride - 1u) / s.stride : 0u;
    s.count.y = box.y > offset.y ? (box.y - offset.y + s.stride - 1u) / s.stride : 0u;
    s.scale = record.power * (float(s.stride * s.stride) / (2.0 * kPi * sqrt(det) * g_Frame.photons.irradianceUnit));

    // Pixel points to the plane: H^-1 (x, y, 1). H^-1's rows are the cross products of H's columns over its
    // determinant, whose sign alone matters here: it keeps g.z positive in front of the camera.
    float3 r0 = cross(hv, h0);
    float3 r1 = cross(h0, hu);
    float3 r2 = cross(hu, hv);
    float sign = dot(hu, r0) >= 0.0 ? 1.0 : -1.0;
    s.gx = sign * float3(r0.x, r1.x, r2.x);
    s.gy = sign * float3(r0.y, r1.y, r2.y);
    s.g0 = sign * float3(r0.z, r1.z, r2.z);
    return true;
}

// A splat's visit (kx, ky)'s pixel.
uint2 SplatPixelAt(Splat s, uint kx, uint ky)
{
    return uint2(s.start + int2(kx, ky) * int(s.stride));
}

// A splat's flux times its Gaussian onto the receiver of a pixel, if that is on its surface and faces its way, as fixed
// point irradiance, rounded at random so that dim photons add up. The pixel's ray (through this frame's jitter, as the
// path tracer's) against the footprint's plane rules out most pixels of the bounds before their receivers are read.
void SplatPixel(Splat s, uint2 pixel, RWStructuredBuffer<uint4> receivers, RWStructuredBuffer<uint4> caustics)
{
    float2 screen = float2(pixel) + 0.5 + g_Frame.jitter;
    float3 g = s.g0 + screen.x * s.gx + screen.y * s.gy;
    if (!(g.z > 0.0) || QuadraticForm(s.inverse, g.xy) > kScreenCutoff * g.z * g.z)
        return;
    uint index = pixel.y * g_Frame.photons.width + pixel.x;
    uint4 receiver = receivers[index];
    if (receiver.w == PHOTON_NO_RECEIVER)
        return;
    float3 delta = asfloat(receiver.xyz) - s.position;
    if (abs(dot(delta, s.n)) > s.planeTolerance)
        return;
    float q = QuadraticForm(s.inverse, float2(dot(delta, s.t), dot(delta, s.b)));
    if (q > kSplatCutoff || dot(OctDecode(receiver.w), s.n) < 0.5)
        return;
    float rounding = float(Hash(uint3(index, s.record, g_PushConstants.frameIndex)) >> 8) * (1.0 / 16777216.0);
    uint3 units = uint3(min(s.scale * exp(-0.5 * q), 16777216.0) + rounding);
    if (units.x != 0u)
        InterlockedAdd(caustics[index].x, units.x);
    if (units.y != 0u)
        InterlockedAdd(caustics[index].y, units.y);
    if (units.z != 0u)
        InterlockedAdd(caustics[index].z, units.z);
}

// The feedback from a photon whose centre the camera sees: its area in pixels and last frame's caustics change there,
// into the photon map texel it came from.
void SendFeedback(GpuPhotonRecord record, Splat s, float cosView, float pixelSize, RWStructuredBuffer<uint4> receivers)
{
    int2 centre;
    if (!ProjectToPixel(record.position, g_Frame.viewProj, centre))
        return;
    uint pixel = uint(centre.y) * g_Frame.photons.width + uint(centre.x);
    uint4 receiver = receivers[pixel];
    if (receiver.w == PHOTON_NO_RECEIVER || abs(dot(asfloat(receiver.xyz) - record.position, s.n)) > s.planeTolerance ||
        dot(OctDecode(receiver.w), s.n) < 0.5)
        return;
    RWStructuredBuffer<float2> historyPrevious = ResourceDescriptorHeap[g_PushConstants.historyPrevious];
    RWStructuredBuffer<uint4> feedback = ResourceDescriptorHeap[g_Frame.photons.feedback];
    float area = min(record.area * cosView / (pixelSize * pixelSize), 65536.0);
    float variance = saturate(historyPrevious[pixel].y);
    InterlockedAdd(feedback[record.texel].x, uint(area * PHOTON_AREA_SCALE + 0.5));
    InterlockedAdd(feedback[record.texel].y, uint(variance * PHOTON_VARIANCE_SCALE + 0.5));
    InterlockedAdd(feedback[record.texel].z, 1u);
}

// One thread per photon record: its feedback, then its splat (SetupSplat, SplatPixel). A splat of up to
// PHOTON_SPLAT_SHARED_PIXELS visits stays with its thread; wider ones are taken one at a time by the whole wave, every
// lane setting it up again from the record and visiting every lanes-th pixel, so that the lanes read neighbouring
// receivers together and no lane waits idle on another's wide splat.
[shader("compute")]
[numthreads(PHOTON_GROUP_SIZE, 1, 1)]
void SplatPhotons(uint3 id : SV_DispatchThreadID)
{
    PhotonParameters ph = g_Frame.photons;
    RWStructuredBuffer<uint> recordCount = ResourceDescriptorHeap[ph.recordCount];
    uint count = min(recordCount[0], PHOTON_RECORD_CAPACITY);
    uint i = id.x;
    // A wave with any record stays whole: the shared splats below count its lanes.
    if (WaveActiveAllTrue(i >= count))
        return;
    RWStructuredBuffer<GpuPhotonRecord> records = ResourceDescriptorHeap[ph.records];
    RWStructuredBuffer<uint4> receivers = ResourceDescriptorHeap[ph.receivers];
    RWStructuredBuffer<uint4> caustics = ResourceDescriptorHeap[g_PushConstants.caustics];

    GpuPhotonRecord record = records[min(i, count - 1u)];
    Splat s;
    float cosView;
    float pixelSize;
    bool valid = SetupSplat(record, i, s, cosView, pixelSize) && i < count;
    if (valid)
        SendFeedback(record, s, cosView, pixelSize, receivers);

    uint visits = valid ? s.count.x * s.count.y : 0u;
    if (visits <= PHOTON_SPLAT_SHARED_PIXELS)
    {
        uint kx = 0u;
        uint ky = 0u;
        for (uint k = 0u; k < visits; ++k)
        {
            SplatPixel(s, SplatPixelAt(s, kx, ky), receivers, caustics);
            if (++kx == s.count.x)
            {
                kx = 0u;
                ++ky;
            }
        }
    }

    bool pending = visits > PHOTON_SPLAT_SHARED_PIXELS;
    uint lanes = WaveActiveCountBits(true);
    uint rank = WavePrefixCountBits(true);
    while (WaveActiveAnyTrue(pending))
    {
        uint owner = WaveActiveMin(pending ? WaveGetLaneIndex() : 0xFFFFFFFFu);
        uint shared = WaveReadLaneAt(i, owner);
        if (WaveGetLaneIndex() == owner)
            pending = false;
        Splat w;
        float unusedCosView;
        float unusedPixelSize;
        SetupSplat(records[shared], shared, w, unusedCosView, unusedPixelSize);
        uint2 step = uint2(lanes % w.count.x, lanes / w.count.x);
        uint kx = rank % w.count.x;
        uint ky = rank / w.count.x;
        while (ky < w.count.y)
        {
            SplatPixel(w, SplatPixelAt(w, kx, ky), receivers, caustics);
            kx += step.x;
            ky += step.y;
            if (kx >= w.count.x)
            {
                kx -= w.count.x;
                ++ky;
            }
        }
    }
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
    RWStructuredBuffer<uint4> receivers = ResourceDescriptorHeap[ph.receivers];
    RWStructuredBuffer<float2> historyPrevious = ResourceDescriptorHeap[g_PushConstants.historyPrevious];
    RWStructuredBuffer<float2> history = ResourceDescriptorHeap[g_PushConstants.history];

    uint4 sum = caustics[pixel];
    caustics[pixel] = uint4(0u, 0u, 0u, 0u);
    float3 irradiance = float3(sum.xyz) * ph.irradianceUnit;
    uint4 receiver = receivers[pixel];
    float luminance = 0.0;
    float variance = 0.0;
    if (receiver.w != PHOTON_NO_RECEIVER)
    {
        RWTexture2D<float4> output = ResourceDescriptorHeap[g_PushConstants.output];
        float4 radiance = output[id.xy];
        radiance.rgb += asfloat(receivers[ph.width * ph.height + pixel].rgb) * irradiance;
        output[id.xy] = radiance;
        luminance = Luminance(irradiance);
        int2 previousPixel;
        if (g_PushConstants.reset == 0 && ProjectToPixel(asfloat(receiver.xyz), g_Frame.prevViewProj, previousPixel))
        {
            float previous = historyPrevious[uint(previousPixel.y) * ph.width + uint(previousPixel.x)].x;
            float both = luminance + previous;
            variance = both > 0.0 ? abs(luminance - previous) / both : 0.0;
        }
    }
    history[pixel] = float2(luminance, variance);
}
