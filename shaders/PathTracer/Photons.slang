// Caustic photons (PhotonInterop.h): the sun's light through smooth glass and mirrors, traced from the sun by
// PhotonRayGen into a world-space hash grid of irradiance, which the path tracer reads at the first diffuse surface of
// a camera path that has met only smooth surfaces (light -> smooth+ -> diffuse -> smooth* -> camera).
import OpenPBRBSDF;
import PathTracer.Resources;
import PathTracer.Math;
import PathTracer.Random;
import PathTracer.RayTracing;
import PathTracer.Materials;
import PathTracer.Sky;

// The camera position the cell levels are measured from.
float3 PhotonCameraPosition()
{
    return g_Frame.invView[3].xyz;
}

// A lobe the photons and the camera paths both pass as smooth: specular reflection, refraction or the coat, no rougher
// than photons.maxAlpha.
bool PhotonSmoothLobe(OpenPBRContext ctx, uint lobe)
{
    if ((lobe & (LOBE_SPECULAR_REFLECTION | LOBE_SPECULAR_TRANSMISSION | LOBE_COAT)) == 0)
        return false;
    float lobeAlpha = lobe == LOBE_COAT ? ctx.coatAlpha : max(ctx.alpha.x, ctx.alpha.y);
    return lobeAlpha <= g_Frame.photons.maxAlpha;
}

struct PhotonCell
{
    uint key;  // never 0, which marks an empty row
    uint slot; // the row probing starts at
};

// log2 of the cell size for a point: about cellPixels pixels across at its distance from the camera.
float PhotonLevel(float3 position)
{
    float footprint = length(position - PhotonCameraPosition()) * g_Frame.rayConePixelSpread * g_Frame.photons.cellPixels;
    return clamp(log2(max(footprint, 1.0e-30)), float(-PHOTON_LEVEL_BIAS + 1), float(PHOTON_LEVEL_BIAS - PHOTON_DEPOSIT_LEVELS - 1));
}

// The cell of the given level of a point on a surface of normal n (either side), the point first moved by up to half a
// cell along the surface by u, a stochastic box filter that blurs the cell edges into noise.
PhotonCell FindPhotonCell(float3 position, float3 n, float2 u, int level)
{
    float size = exp2(float(level));
    float3 t, b;
    BuildOrthonormalBasis(n, t, b);
    float3 p = position + (t * (u.x - 0.5) + b * (u.y - 0.5)) * size;

    float3 a = abs(n);
    uint axis = a.x >= a.y && a.x >= a.z ? 0u : a.y >= a.z ? 1u : 2u;
    uint face = axis * 2u + (n[axis] < 0.0 ? 1u : 0u);

    uint3 c = asuint(int3(floor(p / size)));
    uint3 h = Pcg3d(uint3(c.x ^ (uint(level + PHOTON_LEVEL_BIAS) << 24), c.y ^ (face << 29), c.z));
    h = Pcg3d(h ^ c.zxy);
    PhotonCell cell;
    cell.slot = h.x & (PHOTON_GRID_CAPACITY - 1u);
    cell.key = h.y | 1u;
    return cell;
}

// Adds fixed point power to a cell of the given level.
void DepositInCell(PhotonCell cell, int level, uint3 fixedPower)
{
    RWStructuredBuffer<uint> keys = ResourceDescriptorHeap[g_Frame.photons.gridKeys];
    RWStructuredBuffer<uint4> accum = ResourceDescriptorHeap[g_Frame.photons.gridAccum];
    for (uint i = 0; i < PHOTON_GRID_PROBES; ++i)
    {
        uint slot = (cell.slot + i) & (PHOTON_GRID_CAPACITY - 1u);
        uint previous;
        InterlockedCompareExchange(keys[slot], 0u, cell.key, previous);
        if (previous == 0u || previous == cell.key)
        {
            // Every photon of the cell sets the same level bits, and counts itself below them.
            InterlockedOr(accum[slot].w, uint(level + PHOTON_LEVEL_BIAS) << PHOTON_LEVEL_SHIFT);
            InterlockedAdd(accum[slot].w, 1u);
            InterlockedAdd(accum[slot].x, fixedPower.x);
            InterlockedAdd(accum[slot].y, fixedPower.y);
            InterlockedAdd(accum[slot].z, fixedPower.z);
            return;
        }
    }
    // The rows around a full part of the grid are taken: the photon is lost there.
}

// Adds a photon's power where it landed on a surface of normal n, to the cells of the PHOTON_DEPOSIT_LEVELS levels from
// the one below the point's level up.
void DepositPhoton(float3 position, float3 n, float3 power, float2 u)
{
    float3 units = min(power * (PHOTON_FIXED_POINT_SCALE / g_Frame.photons.unitPower), PHOTON_MAX_DEPOSIT_UNITS);
    uint3 fixedPower = uint3(units + 0.5);
    if (all(fixedPower == 0u))
        return;
    int level = int(floor(PhotonLevel(position)));
    for (int k = 0; k < PHOTON_DEPOSIT_LEVELS; ++k)
        DepositInCell(FindPhotonCell(position, n, u, level + k), level + k, fixedPower);
}

// A cell's running average irradiance (rgb) and photons per frame (w), zero for a cell not in the grid.
float4 CellEstimate(PhotonCell cell, out bool found)
{
    RWStructuredBuffer<uint> keys = ResourceDescriptorHeap[g_Frame.photons.gridKeys];
    RWStructuredBuffer<float4> cache = ResourceDescriptorHeap[g_Frame.photons.gridCache];
    // Every probe is read, since an emptied row may sit between a key's hashed row and its own.
    for (uint i = 0; i < PHOTON_GRID_PROBES; ++i)
    {
        uint slot = (cell.slot + i) & (PHOTON_GRID_CAPACITY - 1u);
        if (keys[slot] == cell.key)
        {
            float4 c = cache[slot];
            found = true;
            return float4(c.rgb, f16tof32(asuint(c.w) & 0xFFFFu));
        }
    }
    found = false;
    return float4(0.0, 0.0, 0.0, 0.0);
}

// The caustic irradiance at a point on a surface of normal n facing the viewer: the two levels around the point's,
// blended by where it falls between them, giving way to the two levels above as their photons get fewer than
// photons.minPhotons.
float3 LookUpPhotonIrradiance(float3 position, float3 n, float2 u)
{
    float level = PhotonLevel(position);
    int lower = int(floor(level));
    float upperWeight = level - float(lower);

    // Every photon of the finer cells is in the widest, so without it there are none.
    bool found;
    float4 widest = CellEstimate(FindPhotonCell(position, n, u, lower + 3), found);
    if (!found)
        return float3(0.0, 0.0, 0.0);
    float4 wide = CellEstimate(FindPhotonCell(position, n, u, lower + 2), found);
    float4 coarse = CellEstimate(FindPhotonCell(position, n, u, lower + 1), found);
    float4 fine = CellEstimate(FindPhotonCell(position, n, u, lower), found);

    float4 own = lerp(fine, coarse, upperWeight);
    float4 sparse = lerp(wide, widest, upperWeight);
    float3 irradiance = lerp(sparse.rgb, own.rgb, saturate(own.w / g_Frame.photons.minPhotons));
    // Where even the wide cells get too few photons (light spread thin far from its caster), the estimate sparkles
    // however long it is averaged: it fades out instead.
    float minWide = g_Frame.photons.minWidePhotons;
    return minWide > 0.0 ? irradiance * saturate(sparse.w / minWide) : irradiance;
}

// A sun photon aimed at the casters: a point on one of the rectangles their bounds make on the plane perpendicular
// to the sun, picked in proportion to the rectangles' areas, so that the plane is covered with a density of the
// rectangles over it per total area. Its power is the sunlight over the area it stands for. False for none.
bool EmitSunPhoton(inout uint rng, out float3 origin, out float3 dir, out float3 power)
{
    PhotonParameters ph = g_Frame.photons;
    StructuredBuffer<GpuPhotonRect> rects = ResourceDescriptorHeap[ph.rects];
    float3 u = Rand3(rng);

    // The first rectangle whose CDF passes u.x.
    uint lo = 0;
    uint hi = ph.rectCount - 1u;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2u;
        if (rects[mid].cdf <= u.x)
            lo = mid + 1u;
        else
            hi = mid;
    }
    GpuPhotonRect rect = rects[lo];
    float2 ab = lerp(rect.lo, rect.hi, u.yz);

    // The other rectangles over the point raise its density.
    uint covering = 0;
    for (uint r = 0; r < ph.rectCount; ++r)
    {
        GpuPhotonRect other = rects[r];
        if (all(ab >= other.lo) && all(ab <= other.hi))
            ++covering;
    }
    covering = max(covering, 1u);

    origin = ph.basisU * ab.x + ph.basisV * ab.y + g_Frame.sunDirection * ph.planeHeight;
    float3 toSun = SunIsDelta() ? g_Frame.sunDirection : SampleUniformCone(g_Frame.sunDirection, g_Frame.sunOneMinusCosMax, Rand3(rng).xy);
    dir = -toSun;
    power = SunIlluminance() * SunTransmittance() * (ph.totalArea / (float(covering) * float(ph.photonCount)));
    return any(power > 0.0);
}

// Follows a photon through smooth surfaces, leaving its power at every surface with a diffuse part it reaches after
// at least one of them. Pass-through panes let it by without counting: light through a window alone is direct light,
// which the path tracer's own light sampling finds. Thin-walled surfaces are no smooth surface to it: what else they
// scatter it to is lost, as camera paths count the sunlight they find past one (PHOTON_COVERED).
void TracePhoton(float3 origin, float3 dir, float3 power, inout uint rng)
{
    float3 throughput = power;
    float startPower = MaxComponent(power);
    float3 extinction = float3(0.0, 0.0, 0.0);
    uint smoothBounces = 0;
    const uint rayFlags = OpacityMicromapRayFlags(1u);
    [loop]
    for (uint segment = 0; segment <= g_Frame.photons.maxBounces + 4u; ++segment)
    {
        HitPayload payload = TraceClosest(origin, dir, false, false, rayFlags);
        HitInfo hit = HitDistance(payload);
        if (!hit.hit)
            return;
        throughput *= exp(-extinction * hit.t);
        hit = UnpackHit(payload, origin, dir);
        if (segment == 0)
            throughput *= CloudShadow(hit.position);

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
            }
        }
        SetShadingBasis(mat, hit.T, hit.B, hit.N, hit.Ns, hit.Ng, wo);
        OpenPBRContext ctx = OpenPBRSetup(mat, wo);

        float3 u = Rand3(rng);
        if (smoothBounces > 0 && ctx.probDiffuse > 0.0)
            DepositPhoton(hit.position, dot(hit.Ng, wo) >= 0.0 ? hit.Ng : -hit.Ng, throughput, u.yz);

        BSDFSample bs = OpenPBRSample(mat, ctx, Rand3(rng));
        if (bs.lobe == LOBE_DELTA_TRANSMISSION)
        {
            throughput *= bs.weight;
            origin = OffsetRay(hit.position, hit.Ng, dir);
            extinction = float3(0.0, 0.0, 0.0);
            continue;
        }
        // Thin-walled materials cast no caustics: past a smooth pane's pass-through, a photon they scatter is lost.
        if (mat.geometryThinWalled || !PhotonSmoothLobe(ctx, bs.lobe) || smoothBounces >= g_Frame.photons.maxBounces)
            return;
        throughput *= bs.weight;
        ++smoothBounces;

        // Russian roulette past the first few bounces, against the power the photon started with.
        if (smoothBounces >= 3u)
        {
            float q = saturate(MaxComponent(throughput) / startPower);
            if (u.x >= q)
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
