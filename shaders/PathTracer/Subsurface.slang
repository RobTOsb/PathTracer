// Path-traced subsurface scattering for solid OpenPBR subsurface materials, after Zhang and d'Eon, "Real-Time
// Subsurface Scattering via Hybrid ReSTIR-Path-Tracing and Diffusion" (SIGGRAPH 2025 Advances course).
//
// The medium under the surface is homogeneous, set by the material where the light enters it. A path that
// takes LOBE_SUBSURFACE gathers three parts of the light leaving the entry, each at a point on the object's
// surface where that light came in: crossing the medium without scattering (the far side along the
// refracted ray), scattering once (the far side after a phase function sample) and scattering more (a
// probe hit near the entry, weighted by Burley's diffusion profile less its single scattering, R - R_1).
// Each of these exits takes a light sample as a white Lambertian surface facing out, and the path goes on
// out of one of them, picked by its weight, so BSDF sampling there completes their MIS.
//
// Simplifications: the interface is crossed by diffuse transmission both ways, the camera side weighted by
// OpenPBR's (1 - E(wo)) like the diffuse lobe, the other side by nothing. Probes go along the normal only
// and the profile ignores the medium's anisotropy. Distances are sampled in one channel, weighted by the
// pdf of all three.
import OpenPBRBSDF;
import PathTracer.Geometry;
import PathTracer.LightSampling;
import PathTracer.Math;
import PathTracer.Random;
import PathTracer.RayTracing;
import PathTracer.Resources;
import PathTracer.Shadows;
import PathTracer.Sky;

// Burley's single scattering profile is alpha R_1 (first order of R in the single-scattering albedo, with
// the surface albedo fitted to it), R_1(r) = R1_SCALE (exp(-R1_SHORT r / l) + exp(-R1_LONG r / l)) / (l r).
static const float R1_SCALE = 0.0423281;
static const float R1_SHORT = 5.434;
static const float R1_LONG = 1.8113;

struct SubsurfaceMedium
{
    float3 colour; // multiple-scattering albedo, the material's subsurface colour
    float3 albedo; // single-scattering albedo
    float3 meanFreePath;
    float3 sigmaT; // extinction, 1 / mean free path
    float anisotropy; // Henyey-Greenstein g
};

SubsurfaceMedium GetSubsurfaceMedium(OpenPBRBaseParams p)
{
    SubsurfaceMedium m;
    m.colour = clamp(p.subsurfaceColour, 0.0, 0.999);
    m.meanFreePath = max(p.subsurfaceRadius, 1.0e-6);
    m.sigmaT = 1.0 / m.meanFreePath;
    m.anisotropy = p.subsurfaceScatterAnisotropy;
    // The single-scattering albedo that multiple scattering takes to the colour (van de Hulst, as fitted
    // by Kulla and Conty 2017).
    float3 C = m.colour;
    float3 s = 4.09712 + 4.20863 * C - sqrt(9.59217 + 41.6808 * C + 17.7126 * C * C);
    float3 s2 = s * s;
    m.albedo = (1.0 - s2) / (1.0 - m.anisotropy * s2);
    return m;
}

// 1 - exp(-x) for x > 0, kept from rounding to 0 for tiny x by its second-order series (below it elsewhere).
float3 OneMinusExp(float3 x)
{
    return max(1.0 - exp(-x), x * (1.0 - 0.5 * x));
}

// Burley's s for a surface albedo A, which scales the mean free path into the profile's falloff.
float3 BurleyScale(float3 A)
{
    float3 a = abs(A - 0.8);
    return 1.85 - A + 7.0 * a * a * a;
}

// Direction after scattering off a medium particle while travelling along dir, which the Henyey-Greenstein
// phase function of anisotropy g gives exactly, so the sample's weight is 1.
float3 SampleHenyeyGreenstein(float3 dir, float g, float2 u)
{
    float cosTheta;
    if (abs(g) < 1.0e-3)
    {
        cosTheta = 1.0 - 2.0 * u.x;
    }
    else
    {
        float q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        cosTheta = clamp((1.0 + g * g - q * q) / (2.0 * g), -1.0, 1.0);
    }
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = TWO_PI * u.y;
    float3 t, b;
    BuildOrthonormalBasis(dir, t, b);
    return normalize(t * (sinTheta * cos(phi)) + b * (sinTheta * sin(phi)) + dir * cosTheta);
}

// Radius the probes reach out to, past which every channel's profile keeps under 0.5% of its light:
// (3 / 4) exp(-s r / (3 l)) = 0.005.
float SubsurfaceProbeRadius(SubsurfaceMedium m)
{
    return 5.0 * MaxComponent(3.0 * m.meanFreePath / BurleyScale(m.colour));
}

// Weight of a probe (disk radius and height) that met the medium's surface t along it, at a cosine
// cosProbe between the surface and the probe's axis: R_n at the distance from the entry over the area
// pdf, the disk's projected onto the surface.
float3 SubsurfaceProbeWeight(SubsurfaceMedium m, float2 probe, float t, float cosProbe)
{
    float r = probe.x;
    float height = probe.y - t;
    float d = sqrt(r * r + height * height);
    float3 s = BurleyScale(m.colour);
    float3 falloff = s * m.sigmaT; // s / l

    // R(d) - R_1(d) and the mixture's area pdf at r, each without its 1 / (8 pi l d) or 1 / (8 pi l r).
    float3 profile = m.colour * s * (exp(-falloff * d) + exp(falloff * (-d / 3.0)));
    float3 single = (8.0 * PI * R1_SCALE) * m.albedo * (exp(-R1_SHORT * m.sigmaT * d) + exp(-R1_LONG * m.sigmaT * d));
    float3 pdfs = falloff * (exp(-falloff * r) + exp(falloff * (-r / 3.0)));
    float pdf = (pdfs.x + pdfs.y + pdfs.z) * (1.0 / 3.0);
    // Grazing hits would need probes along the tangents too; the clamp trades their fireflies for bias.
    return max(profile - single, 0.0) * m.sigmaT * (r / (d * pdf * max(abs(cosProbe), 0.25)));
}

struct SubsurfaceHit
{
    bool hit;
    float t;
    float3 Ng; // outwards
    float3 N;  // interpolated vertex normal, outwards
};

// Closest hit on the medium's material along a ray, other surfaces ignored. An inline query, so nothing is
// saved around it.
SubsurfaceHit TraceSubsurfaceRay(float3 origin, float3 dir, float tMax, uint materialIndex)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed())
    {
        if (LoadPrimitive(query.CandidateInstanceID()).materialIndex == materialIndex)
            query.CommitNonOpaqueTriangleHit();
    }

    SubsurfaceHit hit = (SubsurfaceHit)0;
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return hit;

    // The normals as ClosestHit finds them.
    uint3 tri = LoadTriangle(LoadPrimitive(query.CommittedInstanceID()), query.CommittedPrimitiveIndex());
    float3 bary = Barycentrics(query.CommittedTriangleBarycentrics());
    GpuVertex v0 = g_Vertices[tri.x];
    GpuVertex v1 = g_Vertices[tri.y];
    GpuVertex v2 = g_Vertices[tri.z];
    float3x3 linearToWorld = (float3x3)query.CommittedObjectToWorld3x4();
    float3x3 cofactor = float3x3(cross(linearToWorld[1], linearToWorld[2]),
                                 cross(linearToWorld[2], linearToWorld[0]),
                                 cross(linearToWorld[0], linearToWorld[1]));
    float handedness = dot(linearToWorld[0], cofactor[0]) < 0.0 ? -1.0 : 1.0;
    float3 faceCross = cross(mul(linearToWorld, v1.position - v0.position), mul(linearToWorld, v2.position - v0.position));

    hit.hit = true;
    hit.t = query.CommittedRayT();
    hit.Ng = normalize(faceCross * handedness);
    hit.N = normalize(mul(cofactor, v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z) * handedness);
    return hit;
}

// Light selection at an exit, which reflects like a white Lambertian surface facing along normal.
LightSelectVertex ExitLightSelectVertex(float3 position, float3 normal)
{
    LightSelectVertex v;
    v.position = position;
    v.normal = normal;
    float3 bitangent;
    BuildOrthonormalBasis(normal, v.tangent, bitangent);
    v.wo = normal;
    v.diffuseWeight = 1.0;
    v.specularWeight = 0.0;
    v.alpha = float2(1.0, 1.0);
    v.tree = g_Frame.lightSampling == LIGHT_SAMPLING_TREE;
    return v;
}

// Where the path goes on after a subsurface event: out of one of the exits, cosine distributed.
struct SubsurfaceExit
{
    bool valid;
    float3 origin;
    float3 dir;
    float3 throughput;
    float pdf; // of dir, for the MIS of what it hits
    PackedLightSelectVertex vertex; // the exit's, for the same MIS
};

// The light leaving through the surface at position (geometric normal Ng and vertex normal N facing the
// viewer, the object's material materialIndex) by way of the medium under it, for a path that took
// LOBE_SUBSURFACE there into dirIn with the given throughput. exit is how the path goes on.
float3 SubsurfaceScatter(OpenPBRBaseParams mat, float3 position, float3 Ng, float3 N, uint materialIndex, float3 dirIn,
                         float3 throughput, uint shadowRayFlags, inout uint rng, out SubsurfaceExit exit)
{
    SubsurfaceMedium m = GetSubsurfaceMedium(mat);
    uint components = g_Frame.subsurfaceComponents;
    float3 entry = OffsetRay(position, Ng, dirIn);
    float3 radiance = float3(0.0, 0.0, 0.0);

    exit = (SubsurfaceExit)0;
    float3 exitPosition = float3(0.0, 0.0, 0.0);
    float3 exitNg = float3(0.0, 0.0, 0.0);
    float3 exitN = float3(0.0, 0.0, 0.0);
    float pickSum = 0.0; // luminance of the exits' weights so far

    float d0 = -1.0; // how far the medium goes along dirIn, once the first part has found its far side
    float3 singleWeight = float3(0.0, 0.0, 0.0);
    float2 probe = float2(0.0, 0.0);

    // 0: no scattering, 1: single scattering, 2: the diffusion profile. One loop so that the ray query,
    // light sample and shadow ray are each inlined once.
    [loop]
    for (uint part = 0; part < 3; ++part)
    {
        float3 u = Rand3(rng);
        float3 origin = entry;
        float3 dir = dirIn;
        float tMax = FLT_MAX;
        if (part == 1)
        {
            if (d0 < 0.0 || (components & SUBSURFACE_SINGLE_SCATTERING) == 0)
                continue;
            // Somewhere in (0, d0), from one channel's exponential truncated there, weighted by the pdf of
            // picking any of the three; then the phase function's direction.
            float sigma = m.sigmaT[min(uint(u.x * 3.0), 2u)];
            float t = -log(1.0 - u.y * OneMinusExp(sigma * d0).x) / sigma;
            float3 transmittance = exp(-m.sigmaT * t);
            float3 pdfs = m.sigmaT * transmittance / OneMinusExp(m.sigmaT * d0);
            singleWeight = m.albedo * m.sigmaT * transmittance * (3.0 / (pdfs.x + pdfs.y + pdfs.z));
            origin = entry + dirIn * t;
            dir = SampleHenyeyGreenstein(dirIn, m.anisotropy, float2(u.z, Rand3(rng).x));
        }
        else if (part == 2)
        {
            if ((components & SUBSURFACE_MULTIPLE_SCATTERING) == 0)
                continue;
            // One channel's profile, in r a mixture of exponentials with scales l / s and 3 l / s, weighted
            // 1/4 and 3/4, then down along the normal through that point of the tangent disk.
            uint channel = min(uint(u.x * 3.0), 2u);
            float scale = m.meanFreePath[channel] / BurleyScale(m.colour)[channel];
            bool shortLobe = u.y < 0.25;
            float v = shortLobe ? u.y * 4.0 : (u.y - 0.25) * (4.0 / 3.0);
            float r = max(-log(1.0 - v) * (shortLobe ? scale : 3.0 * scale), 1.0e-6 * scale);
            float reach = SubsurfaceProbeRadius(m);
            if (r >= reach)
                continue;
            float h = sqrt(reach * reach - r * r);
            float3 t, b;
            BuildOrthonormalBasis(N, t, b);
            float phi = TWO_PI * u.z;
            origin = position + (t * cos(phi) + b * sin(phi)) * r + N * h;
            dir = -N;
            tMax = 2.0 * h;
            probe = float2(r, h);
        }

        // The medium's far side seen from inside, or for the probe its near side seen from outside.
        SubsurfaceHit hit = TraceSubsurfaceRay(origin, dir, tMax, materialIndex);
        if (!hit.hit || (dot(hit.Ng, dir) < 0.0) != (part == 2))
            continue;
        float3 weight;
        if (part == 0)
        {
            d0 = hit.t;
            if ((components & SUBSURFACE_ZERO_SCATTERING) == 0)
                continue;
            weight = exp(-m.sigmaT * hit.t);
        }
        else if (part == 1)
        {
            weight = singleWeight * exp(-m.sigmaT * hit.t);
        }
        else
        {
            weight = SubsurfaceProbeWeight(m, probe, hit.t, dot(hit.Ng, dir));
        }
        float3 hitPosition = origin + dir * hit.t;
        float3 normal = dot(hit.N, hit.Ng) > 0.0 ? hit.N : hit.Ng;

        // The light entering here, as a white Lambertian surface would reflect it, MIS weighted against the
        // cosine sample the path takes if it leaves through here.
        float3 lightU = Rand3(rng);
        PackedLightSelectVertex vertex = PackLightSelectVertex(ExitLightSelectVertex(hitPosition, normal));
        LightSample ls = SampleLight(UnpackLightSelectVertex(vertex), lightU.xy, rng);
        float cosLight = dot(ls.direction, normal);
        if (ls.pdf > 0.0 && cosLight > 0.0 && dot(ls.direction, hit.Ng) > 0.0)
        {
            float w = ls.lightSamplingOnly ? 1.0 : PowerHeuristic(ls.pdf, cosLight / PI);
            float3 contribution = weight * ls.radiance * AtmosphereTransmittance(ls.distance) * (cosLight / PI * w / ls.pdf);
            radiance += contribution * CastShadowRay(hitPosition, hit.Ng, ls.direction, ls.distance, shadowRayFlags);
        }

        // Weighted reservoir pick of the exit the path leaves by, in proportion to its weight's luminance.
        float pick = Luminance(weight);
        pickSum += pick;
        if (pick > 0.0 && lightU.z * pickSum < pick)
        {
            exitPosition = hitPosition;
            exitNg = hit.Ng;
            exitN = normal;
            exit.throughput = weight / pick;
            exit.vertex = vertex;
            exit.valid = true;
        }
    }

    if (exit.valid)
    {
        float3 t, b;
        BuildOrthonormalBasis(exitN, t, b);
        float3 local = SampleCosineHemisphere(Rand3(rng).xy);
        exit.dir = t * local.x + b * local.y + exitN * local.z;
        exit.pdf = local.z / PI;
        exit.origin = OffsetRay(exitPosition, exitNg, exit.dir);
        exit.throughput *= throughput * pickSum;
        exit.valid = dot(exit.dir, exitNg) > 0.0 && exit.pdf > 0.0;
    }
    return throughput * radiance;
}
