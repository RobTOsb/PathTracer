#include "PathTracerInterop.h"

ConstantBuffer<LensFlareConstants> g_Constants : register(b0);

// Select and Trace
RWStructuredBuffer<GpuFlareResult> g_Results : register(u0);
RWStructuredBuffer<float> g_Exposure : register(u1); // [0] the exposure, see AutoExposurePass
RWStructuredBuffer<GpuFlareLight> g_Lights : register(u2);
// [0..3] the ghost quads' draw arguments (vertex count grows in Trace), [4..7] the starbursts' (an instance per
// light), [8] the light count.
RWStructuredBuffer<uint> g_Args : register(u3);
RWStructuredBuffer<float4> g_GhostVertices : register(u4); // per ghost instance and grid vertex: tangent xy, radiance, rim
RWStructuredBuffer<uint> g_GhostApertures : register(u5); // the aperture position as two halves
RWStructuredBuffer<uint> g_Quads : register(u6); // instance << 12 | y << 6 | x
StructuredBuffer<GpuFlareSurface> g_Surfaces : register(t0); // FLARE_WAVELENGTHS copies, one per channel's indices
StructuredBuffer<GpuFlareGhost> g_Ghosts : register(t1);

// The draws
StructuredBuffer<GpuFlareLight> g_FlareLights : register(t2);
StructuredBuffer<float4> g_Vertices : register(t3);
StructuredBuffer<uint> g_Apertures : register(t4);
StructuredBuffer<uint> g_QuadList : register(t5);
Texture2D<float4> g_Starburst : register(t6);

// Composite
Texture2D<float4> g_Flare : register(t7);
RWTexture2D<float4> g_Output : register(u7);

SamplerState g_LinearClamp : register(s2);

#if FLARE_GRID > 63
#error "FLARE_GRID must be <= 63 to fit into 6-bit packed quad coordinates."
#endif

static const uint kArgsGhostVertices = 0;
static const uint kArgsStarburst = 4;
static const uint kArgsLightCount = 8;
static const uint kNoCandidate = 0xFFFFFFFF;
// The wavelength (nm) each colour channel's ghosts are traced at.
static const float kWavelengths[FLARE_WAVELENGTHS] = { 650.0, 540.0, 450.0 };
// Below fp16's largest value, so that a few overlapping peaks cannot overflow the target.
static const float kMaxFlareValue = 16384.0;
// How far the iris's sides bow out towards a circle (0 = straight blades). Keep in step with LensFlarePass.cpp.
static const float kBladeCurvature = 0.2;
// The diffraction rings inside a ghost's iris edge, in the iris's units (its circumradius is 1).
static const float kRingPeriod = 0.06;
static const float kRingDecay = 0.1;
static const float kRingStrength = 0.3;
// The share of a channel's ghost light that stays in its own channel; the rest spreads evenly over all three.
static const float kChannelPurity = 0.75;
// Coarse rays per side in the pass that finds the part of the entrance a ghost's light comes through.
static const uint kCoarseGrid = 16;
// How far inside the rims and the iris (in their units) a ray takes its full weight in a ghost's centroid.
static const float kCentroidEdge = 0.05;

// Chromatic aberration tuning constants
static const float kRefWavelength = 540.0;
// Radial tangent magnification shift per 100 nm offset from kRefWavelength.
static const float kGhostLateralCa = 0.018;
// Iris/edge scale shift per 100 nm offset from kRefWavelength.
static const float kGhostEdgeCa = 0.025;
// Starburst UV radial dispersion across R, G, B.
static const float kStarburstRadialCa = 0.035;
// Subtle screen-edge radial dispersion in Composite (scales with distance from screen centre).
static const float kCompositeEdgeCa = 0.004;

float Luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float WavelengthScale(uint channel, float strength)
{
    float w = kWavelengths[min(channel, FLARE_WAVELENGTHS - 1)];
    return 1.0 + strength * ((w - kRefWavelength) / 100.0);
}

groupshared uint2 gs_keys[FLARE_SELECT_GROUP]; // (exposed luminance bits, candidate), unused = (0, kNoCandidate)
groupshared uint gs_count;

// Whether the candidate can flare this frame: lit, in front of the camera, at most half a screen off its edges, and
// at least as bright as the threshold once exposed.
bool MeasureCandidate(uint index, float exposure, out float score, out float3 view)
{
    GpuFlareResult result = g_Results[index];
    score = Luminance(result.irradiance.rgb) * exposure;
    view = float3(0.0, 0.0, 0.0);
    float3 dir = result.direction.xyz;
    if (!(score > 0.0) || !isfinite(score) || dot(dir, dir) <= 0.0)
        return false;
    dir = normalize(dir);
    view = float3(dot(dir, g_Constants.cameraRight), dot(dir, g_Constants.cameraUp), dot(dir, g_Constants.cameraFront));
    if (view.z <= 0.1)
        return false;
    float2 ndc = view.xy / view.z / float2(g_Constants.tanHalfFovX, g_Constants.tanHalfFovY);
    return all(abs(ndc) <= 1.5) && score >= g_Constants.threshold;
}

// a before b: brighter first, then the lower candidate index, so the order never depends on the append order.
bool Before(uint2 a, uint2 b)
{
    return a.x > b.x || (a.x == b.x && a.y < b.y);
}

[shader("compute")]
[numthreads(FLARE_SELECT_GROUP, 1, 1)]
void Select(uint thread : SV_GroupIndex)
{
    if (thread == 0)
        gs_count = 0;
    gs_keys[thread] = uint2(0, kNoCandidate);
    GroupMemoryBarrierWithGroupSync();

    float exposure = g_Exposure[0];
    for (uint i = thread; i < g_Constants.candidateCount; i += FLARE_SELECT_GROUP)
    {
        float score;
        float3 view;
        if (MeasureCandidate(i, exposure, score, view))
        {
            uint slot;
            InterlockedAdd(gs_count, 1, slot);
            if (slot < FLARE_SELECT_GROUP)
                gs_keys[slot] = uint2(asuint(score), i);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort into Before order.
    for (uint k = 2; k <= FLARE_SELECT_GROUP; k <<= 1)
    {
        for (uint j = k >> 1; j > 0; j >>= 1)
        {
            uint partner = thread ^ j;
            if (partner > thread)
            {
                uint2 a = gs_keys[thread];
                uint2 b = gs_keys[partner];
                bool swap = (thread & k) == 0 ? Before(b, a) : Before(a, b);
                if (swap)
                {
                    gs_keys[thread] = b;
                    gs_keys[partner] = a;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    uint lightCount = min(min(gs_count, (uint)FLARE_SELECT_GROUP), min(g_Constants.maxLights, (uint)FLARE_MAX_LIGHTS));
    if (thread < lightCount)
    {
        uint index = gs_keys[thread].y;
        float score;
        float3 view;
        MeasureCandidate(index, exposure, score, view);
        float fade = g_Constants.threshold > 0.0 ? saturate(score / g_Constants.threshold - 1.0) : 1.0;
        GpuFlareLight light;
        light.screenTan = view.xy / view.z;
        light.exposure = exposure;
        light.cosAngle = view.z;
        // Flare compression: past the threshold a light's flare grows only as (score / threshold)^compression, so
        // the sun no longer outshines every lamp's flare by the whole of its brightness. Nothing is brightened.
        float compressed = g_Constants.threshold > 0.0
                               ? pow(max(score / g_Constants.threshold, 1.0), g_Constants.compression - 1.0)
                               : 1.0;
        light.irradiance = g_Results[index].irradiance.rgb * fade * compressed;
        light.exposedLuminance = score * fade * compressed;
        g_Lights[thread] = light;
    }
    if (thread == 0)
    {
        g_Args[kArgsGhostVertices + 0] = 0;
        g_Args[kArgsGhostVertices + 1] = 1;
        g_Args[kArgsGhostVertices + 2] = 0;
        g_Args[kArgsGhostVertices + 3] = 0;
        g_Args[kArgsStarburst + 0] = 6;
        g_Args[kArgsStarburst + 1] = lightCount;
        g_Args[kArgsStarburst + 2] = 0;
        g_Args[kArgsStarburst + 3] = 0;
        g_Args[kArgsLightCount] = lightCount;
    }
}

float IrisDistance(float2 p)
{
    float r = length(p);
    if (g_Constants.blades < 3)
        return r;
    float sector = 2.0 * 3.14159265 / g_Constants.blades;
    float angle = atan2(p.y, p.x) + g_Constants.bladeRotation;
    float local = angle - sector * floor(angle / sector) - 0.5 * sector;
    float polygon = r * cos(local) / cos(0.5 * sector);
    return lerp(polygon, r, kBladeCurvature);
}

struct LensRay
{
    float3 position;
    float3 direction;
};

bool IntersectSurface(GpuFlareSurface surface, inout LensRay ray, out float3 normal)
{
    if (surface.radius == 0.0)
    {
        ray.position += ray.direction * ((surface.z - ray.position.z) / ray.direction.z);
        normal = float3(0.0, 0.0, ray.direction.z > 0.0 ? -1.0 : 1.0);
        return true;
    }
    float3 centre = float3(0.0, 0.0, surface.z + surface.radius);
    float3 oc = ray.position - centre;
    float b = dot(ray.direction, oc);
    float c = dot(oc, oc) - surface.radius * surface.radius;
    float discriminant = b * b - c;
    normal = float3(0.0, 0.0, 0.0);
    if (discriminant < 0.0)
        return false;
    float root = sqrt(discriminant);
    float t = surface.radius * ray.direction.z > 0.0 ? -b - root : -b + root;
    ray.position += ray.direction * t;
    normal = normalize(ray.position - centre);
    if (dot(normal, ray.direction) > 0.0)
        normal = -normal;
    return true;
}

float CoatedReflectance(float cosTheta0, float wavelength, float thickness, float n0, float n1, float n2)
{
    float sin2Theta0 = saturate(1.0 - cosTheta0 * cosTheta0);
    float cos1Squared = 1.0 - sin2Theta0 * (n0 / n1) * (n0 / n1);
    float cos2Squared = 1.0 - sin2Theta0 * (n0 / n2) * (n0 / n2);
    if (cos2Squared <= 0.0 || cos1Squared <= 0.0)
        return 1.0;
    float cos1 = sqrt(cos1Squared);
    float cos2 = sqrt(cos2Squared);

    float rs01 = (n0 * cosTheta0 - n1 * cos1) / (n0 * cosTheta0 + n1 * cos1);
    float rp01 = (n1 * cosTheta0 - n0 * cos1) / (n1 * cosTheta0 + n0 * cos1);
    float rs12 = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2);
    float rp12 = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2);

    float cosPhase = cos(4.0 * 3.14159265 * n1 * thickness * cos1 / wavelength);
    float rs = (rs01 * rs01 + rs12 * rs12 + 2.0 * rs01 * rs12 * cosPhase) /
               (1.0 + rs01 * rs01 * rs12 * rs12 + 2.0 * rs01 * rs12 * cosPhase);
    float rp = (rp01 * rp01 + rp12 * rp12 + 2.0 * rp01 * rp12 * cosPhase) /
               (1.0 + rp01 * rp01 * rp12 * rp12 + 2.0 * rp01 * rp12 * cosPhase);
    return saturate(0.5 * (rs + rp));
}

struct GhostRay
{
    float2 tangent;
    float2 aperture;
    float rim;
    float transmission;
};

GhostRay TraceGhost(LensRay ray, GpuFlareGhost ghost, uint surfaceBase, float wavelength)
{
    GhostRay result;
    result.tangent = float2(0.0, 0.0);
    result.aperture = float2(0.0, 0.0);
    result.rim = 0.0;
    result.transmission = 1.0;

    uint surfaceCount = g_Constants.surfaceCount;
    uint steps = (ghost.first + 1) + (ghost.first - ghost.second) + (surfaceCount - 1 - ghost.second);
    int index = 0;
    int step = 1;
    uint reflections = 0;
    [loop]
    for (uint s = 0; s < steps; ++s)
    {
        GpuFlareSurface surface = g_Surfaces[surfaceBase + index];
        float3 normal;
        if (!IntersectSurface(surface, ray, normal))
        {
            result.transmission = 0.0;
            return result;
        }
        if ((surface.flags & FLARE_SURFACE_STOP) != 0)
        {
            result.aperture = ray.position.xy / g_Constants.stopRadius;
        }
        else
        {
            result.rim = max(result.rim, length(ray.position.xy) / surface.semiAperture);
            bool forward = ray.direction.z > 0.0;
            float n0 = forward ? surface.nBefore : surface.nAfter;
            float n2 = forward ? surface.nAfter : surface.nBefore;
            float cosTheta = saturate(-dot(ray.direction, normal));
            float reflectance = CoatedReflectance(cosTheta, wavelength, surface.coatingThickness, n0, surface.coatingIor, n2);
            bool reflects = (reflections == 0 && index == (int)ghost.first) || (reflections == 1 && index == (int)ghost.second);
            if (reflects)
            {
                ray.direction = reflect(ray.direction, normal);
                result.transmission *= reflectance;
                ++reflections;
                step = -step;
            }
            else
            {
                float3 refracted = refract(ray.direction, normal, n0 / n2);
                if (all(refracted == 0.0))
                {
                    result.transmission = 0.0;
                    return result;
                }
                ray.direction = refracted;
                result.transmission *= 1.0 - reflectance;
            }
        }
        index += step;
    }

    if (ray.direction.z <= 0.0)
    {
        result.transmission = 0.0;
        return result;
    }
    float2 sensor = ray.position.xy + ray.direction.xy * ((g_Constants.sensorZ - ray.position.z) / ray.direction.z);
    result.tangent = -sensor / g_Constants.focalLength;
    return result;
}

groupshared float2 gs_tangent[FLARE_GRID_VERTICES];
groupshared float gs_transmission[FLARE_GRID_VERTICES];
groupshared float gs_rim[FLARE_GRID_VERTICES];
groupshared float2 gs_aperture[FLARE_GRID_VERTICES];
groupshared float gs_radiance[FLARE_GRID_VERTICES];

uint GridIndex(uint x, uint y)
{
    return y * (FLARE_GRID + 1) + x;
}

groupshared uint gs_bounds[4];
// The group's kept quads, and where its share of the quad list starts.
groupshared uint gs_quadCount;
groupshared uint gs_quadBase;
static const uint kQuadsPerThread = (FLARE_GRID * FLARE_GRID + FLARE_TRACE_GROUP - 1) / FLARE_TRACE_GROUP;
static const uint kNoSlot = 0xFFFFFFFFu;

uint OrderedBits(float value)
{
    uint bits = asuint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

float FromOrderedBits(uint bits)
{
    return asfloat((bits & 0x80000000u) != 0 ? bits & 0x7FFFFFFFu : ~bits);
}

LensRay EntranceRay(float2 grid, float3 direction, float startZ)
{
    LensRay ray;
    ray.position = float3(grid, g_Constants.entranceZ) + direction * ((startZ - g_Constants.entranceZ) / direction.z);
    ray.direction = direction;
    return ray;
}

float QuadDensity(uint x, uint y, float cellFlux, float pixel)
{
    uint i00 = GridIndex(x, y);
    uint i10 = GridIndex(x + 1, y);
    uint i11 = GridIndex(x + 1, y + 1);
    uint i01 = GridIndex(x, y + 1);
    float t00 = gs_transmission[i00];
    float t10 = gs_transmission[i10];
    float t11 = gs_transmission[i11];
    float t01 = gs_transmission[i01];
    if (t00 <= 0.0 || t10 <= 0.0 || t11 <= 0.0 || t01 <= 0.0)
        return 0.0;
    float2 d0 = gs_tangent[i11] - gs_tangent[i00];
    float2 d1 = gs_tangent[i01] - gs_tangent[i10];
    float area = 0.5 * abs(d0.x * d1.y - d0.y * d1.x);
    float minArea = pixel * max(pixel, 0.5 * max(length(d0), length(d1)));
    return 0.25 * (t00 + t10 + t11 + t01) * cellFlux / max(area, minArea);
}

[shader("compute")]
[numthreads(FLARE_TRACE_GROUP, 1, 1)]
void Trace(uint3 groupId : SV_GroupID, uint thread : SV_GroupIndex)
{
    uint instance = groupId.x;
    uint channel = instance % FLARE_WAVELENGTHS;
    uint ghostIndex = (instance / FLARE_WAVELENGTHS) % g_Constants.ghostCount;
    uint lightIndex = instance / (FLARE_WAVELENGTHS * g_Constants.ghostCount);
    if (lightIndex >= g_Args[kArgsLightCount])
        return;

    GpuFlareLight light = g_Lights[lightIndex];
    GpuFlareGhost ghost = g_Ghosts[ghostIndex];
    float wavelength = kWavelengths[channel];
    uint surfaceBase = channel * g_Constants.surfaceCount;
    float irradiance = light.irradiance[channel] * g_Constants.ghostIntensity;

    float3 direction = normalize(float3(-light.screenTan * g_Constants.fieldScale, 1.0));
    float radius = g_Constants.entranceRadius;
    float startZ = g_Surfaces[surfaceBase].z - 1.0;

    if (thread == 0)
    {
        gs_bounds[0] = 0xFFFFFFFFu;
        gs_bounds[1] = 0xFFFFFFFFu;
        gs_bounds[2] = 0;
        gs_bounds[3] = 0;
        gs_quadCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    {
        float2 grid = ((float2(thread % kCoarseGrid, thread / kCoarseGrid) + 0.5) / kCoarseGrid * 2.0 - 1.0) * radius;
        GhostRay traced = TraceGhost(EntranceRay(grid, direction, startZ), ghost, surfaceBase, wavelength);
        bool reached = traced.transmission > 0.0 && all(isfinite(traced.tangent));
        float irisValue = IrisDistance(traced.aperture);
        bool shows = reached && traced.rim <= 1.0 && irisValue <= 1.0;
        // Each lane adds itself: wave reductions here were moved by the compiler into the branches on shows, so the
        // wave's first lane saw only the lanes on its own side and the bounds stayed empty.
        if (shows)
        {
            InterlockedMin(gs_bounds[0], OrderedBits(grid.x));
            InterlockedMin(gs_bounds[1], OrderedBits(grid.y));
            InterlockedMax(gs_bounds[2], OrderedBits(grid.x));
            InterlockedMax(gs_bounds[3], OrderedBits(grid.y));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gs_bounds[0] > gs_bounds[2])
        return;

    float coarseCell = 2.0 * radius / kCoarseGrid;
    float2 boundsMin = max(float2(FromOrderedBits(gs_bounds[0]), FromOrderedBits(gs_bounds[1])) - coarseCell, -radius);
    float2 boundsMax = min(float2(FromOrderedBits(gs_bounds[2]), FromOrderedBits(gs_bounds[3])) + coarseCell, radius);
    float2 cell = (boundsMax - boundsMin) / FLARE_GRID;
    float cellFlux = cell.x * cell.y * direction.z / g_Constants.pupilArea;
    // Quads are measured in the lens's tangent plane, so the ghosts' radiance is the lens's own; only the vertices
    // are scaled to the screen.
    // ghostScale shrinks each ghost about its centre (the image of the middle of its bounds) and concentrates its
    // light to match, so a drawn pixel covers 1 / ghostScale as much of the lens's tangent plane.
    float ghostScale = g_Constants.ghostScale;
    float pixel = 2.0 * g_Constants.tanHalfFovY * g_Constants.flareTexelSize.y * g_Constants.fieldScale / ghostScale;

    for (uint v = thread; v < FLARE_GRID_VERTICES; v += FLARE_TRACE_GROUP)
    {
        uint x = v % (FLARE_GRID + 1);
        uint y = v / (FLARE_GRID + 1);
        float2 grid = boundsMin + float2(x, y) * cell;
        GhostRay traced = TraceGhost(EntranceRay(grid, direction, startZ), ghost, surfaceBase, wavelength);
        bool valid = traced.transmission > 0.0 && all(isfinite(traced.tangent));
        gs_tangent[v] = valid ? traced.tangent : float2(0.0, 0.0);
        gs_transmission[v] = valid ? traced.transmission : 0.0;
        gs_rim[v] = traced.rim;
        gs_aperture[v] = traced.aperture;
    }
    GroupMemoryBarrierWithGroupSync();

    // The ghost shrinks about the centroid of its light on the sensor: the transmission-weighted mean of the rays
    // inside the rims and the iris (each entrance cell carries the same flux). It moves with the rays rather than
    // with the grid, whose bounds step a coarse cell at a time, and the weights fade in over kCentroidEdge so rays
    // crossing an edge do not make it jump. gs_radiance, filled only below, holds the partial sums.
    {
        float3 partial = float3(0.0, 0.0, 0.0);
        for (uint v = thread; v < FLARE_GRID_VERTICES; v += FLARE_TRACE_GROUP)
        {
            float inside = saturate((1.0 - gs_rim[v]) / kCentroidEdge) *
                           saturate((1.0 - IrisDistance(gs_aperture[v])) / kCentroidEdge);
            float weight = gs_transmission[v] * inside;
            partial += float3(gs_tangent[v] * weight, weight);
        }
        gs_radiance[thread] = partial.x;
        gs_radiance[FLARE_TRACE_GROUP + thread] = partial.y;
        gs_radiance[2 * FLARE_TRACE_GROUP + thread] = partial.z;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = FLARE_TRACE_GROUP / 2; stride > 0; stride >>= 1)
    {
        if (thread < stride)
        {
            gs_radiance[thread] += gs_radiance[thread + stride];
            gs_radiance[FLARE_TRACE_GROUP + thread] += gs_radiance[FLARE_TRACE_GROUP + thread + stride];
            gs_radiance[2 * FLARE_TRACE_GROUP + thread] += gs_radiance[2 * FLARE_TRACE_GROUP + thread + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float totalWeight = gs_radiance[2 * FLARE_TRACE_GROUP];
    bool scaled = totalWeight > 0.0;
    float2 centre = scaled ? float2(gs_radiance[0], gs_radiance[FLARE_TRACE_GROUP]) / totalWeight : float2(0.0, 0.0);
    // Every thread has read the sums before the vertex loop overwrites them.
    GroupMemoryBarrierWithGroupSync();
    float scale = scaled ? ghostScale : 1.0;
    float concentration = 1.0 / (scale * scale);
    uint vertexBase = instance * FLARE_GRID_VERTICES;
    for (uint v = thread; v < FLARE_GRID_VERTICES; v += FLARE_TRACE_GROUP)
    {
        uint x = v % (FLARE_GRID + 1);
        uint y = v / (FLARE_GRID + 1);
        float sum = 0.0;
        float count = 0.0;
        for (uint q = 0; q < 4; ++q)
        {
            int qx = int(x) - int(q & 1);
            int qy = int(y) - int(q >> 1);
            if (qx < 0 || qy < 0 || qx >= FLARE_GRID || qy >= FLARE_GRID)
                continue;
            float density = QuadDensity(uint(qx), uint(qy), cellFlux, pixel);
            if (density > 0.0)
            {
                sum += density;
                count += 1.0;
            }
        }
        float radiance = count > 0.0 ? irradiance * concentration * sum / count : 0.0;
        gs_radiance[v] = radiance;
        float2 tangent = centre + (gs_tangent[v] - centre) * scale;
        g_GhostVertices[vertexBase + v] = float4(tangent / g_Constants.fieldScale, radiance, gs_rim[v]);
        g_GhostApertures[vertexBase + v] = f32tof16(gs_aperture[v].x) | (f32tof16(gs_aperture[v].y) << 16);
    }
    GroupMemoryBarrierWithGroupSync();

    uint localSlots[kQuadsPerThread];
    uint localQuads[kQuadsPerThread];
    for (uint k = 0; k < kQuadsPerThread; ++k)
    {
        uint quad = thread + k * FLARE_TRACE_GROUP;
        uint x = quad % FLARE_GRID;
        uint y = quad / FLARE_GRID;
        uint corners[4] = { GridIndex(x, y), GridIndex(x + 1, y), GridIndex(x + 1, y + 1), GridIndex(x, y + 1) };
        bool alive = true;
        float minRim = 1.0e30;
        float2 apertureMin = float2(1.0e30, 1.0e30);
        float2 apertureMax = float2(-1.0e30, -1.0e30);
        float brightest = 0.0;
        for (uint c = 0; c < 4; ++c)
        {
            uint i = corners[c];
            alive = alive && gs_transmission[i] > 0.0;
            minRim = min(minRim, gs_rim[i]);
            apertureMin = min(apertureMin, gs_aperture[i]);
            apertureMax = max(apertureMax, gs_aperture[i]);
            brightest = max(brightest, gs_radiance[i]);
        }
        bool keep = alive && minRim <= 1.0 && all(apertureMin <= 1.0) && all(apertureMax >= -1.0) && brightest * light.exposure >= g_Constants.minVisible;
        // A slot in the group's share of the quad list (no wave intrinsics, for the reason above).
        localSlots[k] = kNoSlot;
        if (keep)
        {
            uint local;
            InterlockedAdd(gs_quadCount, 1, local);
            localSlots[k] = local;
            localQuads[k] = (instance << 12) | (y << 6) | x;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (thread == 0)
    {
        uint vertexOffset;
        InterlockedAdd(g_Args[kArgsGhostVertices], gs_quadCount * 6, vertexOffset);
        gs_quadBase = vertexOffset / 6;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 0; k < kQuadsPerThread; ++k)
    {
        uint slot = gs_quadBase + localSlots[k];
        if (localSlots[k] != kNoSlot && slot < g_Constants.quadCapacity)
            g_Quads[slot] = localQuads[k];
    }
}

struct GhostVaryings
{
    float4 position : SV_Position;
    float2 aperture : TEXCOORD0;
    float rim : TEXCOORD1;
    float radiance : TEXCOORD2;
    nointerpolation uint channel : TEXCOORD3;
};

float4 TangentToClip(float2 tangent)
{
    return float4(tangent / float2(g_Constants.tanHalfFovX, g_Constants.tanHalfFovY), 0.5, 1.0);
}

[shader("vertex")]
GhostVaryings GhostVS(uint vertexId : SV_VertexID)
{
    static const uint2 kCorners[6] = { uint2(0, 0), uint2(1, 0), uint2(1, 1), uint2(0, 0), uint2(1, 1), uint2(0, 1) };
    GhostVaryings output;
    uint slot = vertexId / 6;
    if (slot >= g_Constants.quadCapacity)
    {
        output.position = float4(0.0, 0.0, 0.0, 0.0) / 0.0;
        output.aperture = float2(0.0, 0.0);
        output.rim = 2.0;
        output.radiance = 0.0;
        output.channel = 0;
        return output;
    }
    uint packed = g_QuadList[slot];
    uint instance = packed >> 12;
    uint2 corner = uint2(packed & 63, (packed >> 6) & 63) + kCorners[vertexId % 6];
    uint vertex = instance * FLARE_GRID_VERTICES + GridIndex(corner.x, corner.y);
    float4 data = g_Vertices[vertex];
    uint aperture = g_Apertures[vertex];
    uint channel = instance % FLARE_WAVELENGTHS;

    // Lateral chromatic shift in tangent space around the optical axis
    float2 dispersedTangent = data.xy * WavelengthScale(channel, kGhostLateralCa);
    output.position = TangentToClip(dispersedTangent);
    output.aperture = float2(f16tof32(aperture & 0xFFFF), f16tof32(aperture >> 16));
    output.rim = data.w;
    output.radiance = data.z;
    output.channel = channel;
    return output;
}

float FadeInside(float value, float softness)
{
    float pixel = max(fwidth(value), 1.0e-4);
    return smoothstep(1.0 + 0.5 * pixel, 1.0 - softness - 0.5 * pixel, value);
}

[shader("pixel")]
float4 GhostPS(GhostVaryings input) : SV_Target
{
    float softness = g_Constants.ghostSoftness;
    // Slight wavelength-dependent iris/edge scaling creates chromatic fringing near ghost borders
    float edgeScale = WavelengthScale(input.channel, kGhostEdgeCa);
    float iris = IrisDistance(input.aperture * edgeScale);
    float rim = input.rim * edgeScale;
    float shape = FadeInside(iris, softness) * FadeInside(rim, 0.5 * softness);

    float depth = max(1.0 - iris - 0.5 * softness, 0.0);
    float ringFade = saturate(1.0 - 2.0 * fwidth(iris) / kRingPeriod);
    float ring = exp(-depth / kRingDecay) * (0.5 + 0.5 * cos(depth * (2.0 * 3.14159265 / kRingPeriod)));
    float profile = lerp(0.7, 1.0, iris * iris) + kRingStrength * ringFade * ring;

    float value = min(input.radiance * shape * profile, kMaxFlareValue);
    float3 colour = float3(1.0, 1.0, 1.0) * ((1.0 - kChannelPurity) / 3.0);
    colour[input.channel] += kChannelPurity;
    return float4(colour * value, 0.0);
}

struct StarburstVaryings
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    nointerpolation float3 scale : TEXCOORD1;
};

[shader("vertex")]
StarburstVaryings StarburstVS(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    static const float2 kCorners[6] = { float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 0), float2(1, 1), float2(0, 1) };
    GpuFlareLight light = g_FlareLights[instanceId];
    float2 uv = kCorners[vertexId % 6];
    float halfSize = g_Constants.starburstHalfSize;
    StarburstVaryings output;
    output.position = TangentToClip(light.screenTan + (uv * 2.0 - 1.0) * float2(halfSize, -halfSize));
    output.uv = uv;
    output.scale = light.irradiance * g_Constants.starburstIntensity / (4.0 * halfSize * halfSize);
    return output;
}

[shader("pixel")]
float4 StarburstPS(StarburstVaryings input) : SV_Target
{
    // Sample R, G, and B at slightly different radial scales around the starburst centre (0.5, 0.5)
    float2 centred = input.uv - 0.5;
    float2 uvR = 0.5 + centred * (1.0 - kStarburstRadialCa);
    float2 uvG = input.uv;
    float2 uvB = 0.5 + centred * (1.0 + kStarburstRadialCa);

    float3 sampled = float3(
        g_Starburst.SampleLevel(g_LinearClamp, uvR, 0.0).r,
        g_Starburst.SampleLevel(g_LinearClamp, uvG, 0.0).g,
        g_Starburst.SampleLevel(g_LinearClamp, uvB, 0.0).b);

    float3 value = sampled * input.scale;
    return float4(min(value, kMaxFlareValue), 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void Composite(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.outputWidth || id.y >= g_Constants.outputHeight)
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(g_Constants.outputWidth, g_Constants.outputHeight);
    float2 offset = (uv - 0.5) * kCompositeEdgeCa;

    float3 flare = float3(
        g_Flare.SampleLevel(g_LinearClamp, uv - offset, 0.0).r,
        g_Flare.SampleLevel(g_LinearClamp, uv, 0.0).g,
        g_Flare.SampleLevel(g_LinearClamp, uv + offset, 0.0).b);

    if (any(!isfinite(flare)))
        return;
    g_Output[id.xy] = g_Output[id.xy] + float4(flare, 0.0);
}