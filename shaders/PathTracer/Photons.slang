// Caustic photons (PhotonInterop.h), after Yang and Ouyang's Adaptive Anisotropic Photon Scattering: sun photons
// emitted from the photon map, traced through smooth glass and mirrors carrying photon differentials, and recorded
// where they land on a surface with a diffuse part, which PhotonMap.slang splats onto the surfaces the camera sees
// (WritePhotonReceiver).
import OpenPBRBSDF;
import PathTracer.Resources;
import PathTracer.Math;
import PathTracer.Random;
import PathTracer.RayTracing;
import PathTracer.Materials;
import PathTracer.Sky;
import PathTracer.Psr;

// A lobe the photons and the camera paths both pass as smooth: specular reflection, refraction or the coat, no rougher
// than photons.maxAlpha.
bool PhotonSmoothLobe(OpenPBRContext ctx, uint lobe)
{
    if ((lobe & (LOBE_SPECULAR_REFLECTION | LOBE_SPECULAR_TRANSMISSION | LOBE_COAT)) == 0)
        return false;
    float lobeAlpha = lobe == LOBE_COAT ? ctx.coatAlpha : max(ctx.alpha.x, ctx.alpha.y);
    return lobeAlpha <= g_Frame.photons.maxAlpha;
}

// Receivers: the surface with a diffuse part the camera ray of this pixel meets first (also through pass-through
// panes, which leave it straight), on which PhotonMap.slang splats the photons and adds their light by reflectance,
// its diffuse reflectance over pi times the path's throughput.
uint PhotonReceiverIndex()
{
    return 2u * (DispatchRaysIndex().y * g_Frame.photons.width + DispatchRaysIndex().x);
}

void ClearPhotonReceiver()
{
    RWStructuredBuffer<float4> receivers = ResourceDescriptorHeap[g_Frame.photons.receivers];
    receivers[PhotonReceiverIndex()] = float4(0.0, 0.0, 0.0, 0.0);
}

void WritePhotonReceiver(float3 position, float3 n, float3 reflectance)
{
    RWStructuredBuffer<float4> receivers = ResourceDescriptorHeap[g_Frame.photons.receivers];
    uint i = PhotonReceiverIndex();
    receivers[i] = float4(position, 1.0);
    receivers[i + 1u] = float4(reflectance, asfloat(OctEncode(n)));
}

// The cell at distance d along a Hilbert curve over a 2^order square of cells.
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

// Owen scrambling of a 0.32 fixed point value (Burley 2020, "Practical Hash-based Owen Scrambling"): each bit flipped
// by a hash of the bits above it and the seed.
uint OwenScramble(uint x, uint seed)
{
    x = reversebits(x);
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return reversebits(x);
}

float FixedToUnit(uint x)
{
    return float(x >> 8) * (1.0 / 16777216.0);
}

// Where a photon starts on the plane perpendicular to the sun, and what it stands for.
struct PhotonStart
{
    float3 origin;
    float3 power; // flux: the sunlight over its area
    float3 dPu; // positional differentials: the sides of the area it stands for on the plane
    float3 dPv;
    uint texel; // the photon map texel it came from, in Hilbert order
    float lobeU; // its first surface's lobe choice
};

bool EmitSunPhoton(uint index, inout uint rng, out PhotonStart start)
{
    PhotonParameters ph = g_Frame.photons;
    RWStructuredBuffer<float> density = ResourceDescriptorHeap[ph.density];
    RWStructuredBuffer<float> blockPrefix = ResourceDescriptorHeap[ph.blockPrefix];
    RWStructuredBuffer<float> texelPrefix = ResourceDescriptorHeap[ph.texelPrefix];
    start = (PhotonStart)0;
    float total = blockPrefix[PHOTON_MAP_BLOCKS];
    if (!(total > 0.0))
        return false;

    float3 u = Rand3(rng);
    float target = (float(index) + u.x) / float(ph.photonCount) * total;
    // The last block whose shares start at or before the target: blocks without any are passed over.
    uint lo = 0u;
    uint hi = PHOTON_MAP_BLOCKS - 1u;
    while (lo < hi)
    {
        uint mid = (lo + hi + 1u) / 2u;
        if (blockPrefix[mid] <= target)
            lo = mid;
        else
            hi = mid - 1u;
    }
    uint first = lo * PHOTON_MAP_BLOCK;
    float local = target - blockPrefix[lo];
    // The first texel of the block whose shares up to it pass the target.
    lo = 0u;
    hi = PHOTON_MAP_BLOCK - 1u;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2u;
        if (texelPrefix[first + mid] <= local)
            lo = mid + 1u;
        else
            hi = mid;
    }
    uint texel = first + lo;
    float share = density[texel];
    if (!(share > 0.0))
        return false;
    float before = lo > 0u ? texelPrefix[texel - 1u] : 0.0;
    float t = saturate((local - before) / share);

    // A curve over at least as many cells as the texel gets photons, so that a photon's part of it is a cell or a few.
    float expected = share / total * float(ph.photonCount);
    uint order = expected <= 1.0 ? 0u : min((firstbithigh(uint(ceil(expected)) - 1u) + 2u) / 2u, 8u);
    uint cells = 1u << (2u * order);
    uint d = min(uint(t * float(cells)), cells - 1u);
    float2 inTexel = (float2(HilbertCell(d, order)) + u.yz) * exp2(-float(order));
    float texelSize = ph.mapScale / float(PHOTON_MAP_SIZE);
    float2 ab = ph.mapOrigin + (float2(HilbertCell(texel, PHOTON_MAP_ORDER)) + inTexel) * texelSize;

    float area = texelSize * texelSize / expected;
    float spacing = sqrt(area);
    start.origin = ph.basisU * ab.x + ph.basisV * ab.y + g_Frame.sunDirection * ph.planeHeight;
    start.power = SunIlluminance() * SunTransmittance() * area;
    start.dPu = ph.basisU * spacing;
    start.dPv = ph.basisV * spacing;
    start.texel = texel;
    uint seed = Pcg3d(uint3(g_Frame.frameIndex, 0x9E3779B9u, 0x7F4A7C15u)).x;
    start.lobeU = FixedToUnit(OwenScramble(reversebits(index), seed));
    return any(start.power > 0.0);
}

// Records a photon's power where it landed on a surface of normal n facing the light it came from: its footprint's
// covariance from its differentials, photons.footprintScale times the positional ones across (one standard deviation
// is half of that) and the sun disc's one-sigma directional ones as they are. One atomic per wave claims the slots.
void RecordPhoton(float3 position, float3 n, float3 power, uint texel, RayDifferential du, RayDifferential dv,
                  RayDifferential da, RayDifferential db)
{
    PhotonParameters ph = g_Frame.photons;
    GpuPhotonRecord record;
    record.position = position;
    record.normal = OctEncode(n);
    record.power = power;
    record.texel = texel;
    float3 t, b;
    BuildOrthonormalBasis(OctDecode(record.normal), t, b);
    float s = 0.5 * ph.footprintScale;
    float2 pu = float2(dot(du.dP, t), dot(du.dP, b));
    float2 pv = float2(dot(dv.dP, t), dot(dv.dP, b));
    float2 pa = float2(dot(da.dP, t), dot(da.dP, b));
    float2 pb = float2(dot(db.dP, t), dot(db.dP, b));
    record.covariance = s * s * float3(pu.x * pu.x + pv.x * pv.x, pu.x * pu.y + pv.x * pv.y, pu.y * pu.y + pv.y * pv.y) +
                        float3(pa.x * pa.x + pb.x * pb.x, pa.x * pa.y + pb.x * pb.y, pa.y * pa.y + pb.y * pb.y);
    record.area = abs(pu.x * pv.y - pu.y * pv.x);

    RWStructuredBuffer<uint> recordCount = ResourceDescriptorHeap[ph.recordCount];
    uint laneOffset = WavePrefixCountBits(true);
    uint waveCount = WaveActiveCountBits(true);
    uint base = 0u;
    if (WaveIsFirstLane())
        InterlockedAdd(recordCount[0], waveCount, base);
    uint slot = WaveReadLaneFirst(base) + laneOffset;
    if (slot >= PHOTON_RECORD_CAPACITY)
        return;
    RWStructuredBuffer<GpuPhotonRecord> records = ResourceDescriptorHeap[ph.records];
    records[slot] = record;
}

void TracePhoton(PhotonStart start, inout uint rng)
{
    PhotonParameters ph = g_Frame.photons;
    float3 origin = start.origin;
    float3 dir = -g_Frame.sunDirection;
    float3 throughput = start.power;
    float startPower = MaxComponent(start.power);
    RayDifferential du = { start.dPu, float3(0.0, 0.0, 0.0) };
    RayDifferential dv = { start.dPv, float3(0.0, 0.0, 0.0) };
    RayDifferential da = { float3(0.0, 0.0, 0.0), ph.basisU * ph.sunSpread };
    RayDifferential db = { float3(0.0, 0.0, 0.0), ph.basisV * ph.sunSpread };
    float3 extinction = float3(0.0, 0.0, 0.0);
    uint smoothBounces = 0;
    const uint rayFlags = OpacityMicromapRayFlags(1u);
    [loop]
    for (uint segment = 0; segment <= ph.maxBounces + 4u; ++segment)
    {
        HitPayload payload = TraceClosest(origin, dir, false, true, rayFlags);
        HitInfo hit = HitDistance(payload);
        if (!hit.hit)
            return;
        throughput *= exp(-extinction * hit.t);
        hit = UnpackHit(payload, origin, dir);
        if (segment == 0)
            throughput *= CloudShadow(hit.position);
        if (abs(dot(dir, hit.Ng)) < 1.0e-4)
            return;
        TransferDifferential(du, dir, hit.t, hit.Ng);
        TransferDifferential(dv, dir, hit.t, hit.Ng);
        TransferDifferential(da, dir, hit.t, hit.Ng);
        TransferDifferential(db, dir, hit.t, hit.Ng);

        float3 wo = -dir;
        OpenPBRBaseParams mat = LoadMaterial(hit, 0.0, false, IsSimpleMaterial(hit.materialId), float2(0.0, 0.0));
        // Faced to the incoming light as ShadeSurface faces surfaces to the viewer.
        bool opaque = mat.transmissionWeight <= 0.0 && !HasDiffuseTransmission(mat);
        if (opaque || mat.geometryThinWalled)
        {
            if (dot(hit.Ng, wo) < 0.0)
                hit.Ng = -hit.Ng;
            if (dot(hit.N, hit.Ng) < 0.0)
            {
                hit.N = -hit.N;
                hit.Ns = -hit.Ns;
                FlipShape(hit);
            }
        }
        SetShadingBasis(mat, hit.T, hit.B, hit.N, hit.Ns, hit.Ng, wo);
        OpenPBRContext ctx = OpenPBRSetup(mat, wo);

        if (smoothBounces > 0 && ctx.probDiffuse > 0.0)
            RecordPhoton(hit.position, dot(hit.Ng, wo) >= 0.0 ? hit.Ng : -hit.Ng, throughput, start.texel, du, dv, da, db);

        // The first surface's lobe choice is stratified across neighbouring photons (EmitSunPhoton).
        float3 su = Rand3(rng);
        if (segment == 0u)
            su.x = start.lobeU;
        BSDFSample bs = OpenPBRSample(mat, ctx, su);
        float roulette = Rand3(rng).x;
        if (bs.lobe == LOBE_DELTA_TRANSMISSION)
        {
            throughput *= bs.weight;
            origin = OffsetRay(hit.position, hit.Ng, dir);
            extinction = float3(0.0, 0.0, 0.0);
            continue;
        }
        // Thin-walled materials cast no caustics: past a smooth pane's pass-through, a photon they scatter is lost.
        if (mat.geometryThinWalled || !PhotonSmoothLobe(ctx, bs.lobe) || smoothBounces >= ph.maxBounces)
            return;
        throughput *= bs.weight;

        bool reflection = bs.lobe != LOBE_SPECULAR_TRANSMISSION;
        if (!reflection && abs(dot(bs.wi, hit.N)) < 1.0e-4)
            return;
        float eta = 1.0 / ctx.eta;
        TurnDifferential(du, hit, hit.N, dir, bs.wi, reflection, eta);
        TurnDifferential(dv, hit, hit.N, dir, bs.wi, reflection, eta);
        TurnDifferential(da, hit, hit.N, dir, bs.wi, reflection, eta);
        TurnDifferential(db, hit, hit.N, dir, bs.wi, reflection, eta);
        ++smoothBounces;

        // Russian roulette past the first few bounces, against the power the photon started with.
        if (smoothBounces >= 3u)
        {
            float q = saturate(MaxComponent(throughput) / startPower);
            if (roulette >= q)
                return;
            throughput /= q;
        }

        // Into the object through its medium, out of it through the medium outside (an interface inside one).
        if (dot(bs.wi, hit.Ng) < 0.0)
            extinction = MediumExtinction(mat);
        else
            extinction = opaque || mat.geometryThinWalled ? float3(0.0, 0.0, 0.0) : ExteriorExtinction(hit.materialId);
        origin = OffsetRay(hit.position, hit.Ng, bs.wi);
        dir = bs.wi;
    }
}
