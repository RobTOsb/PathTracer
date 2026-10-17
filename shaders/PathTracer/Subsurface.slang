// Path-traced subsurface scattering for solid OpenPBR subsurface materials, after Zhang and d'Eon, "Real-Time
// Subsurface Scattering via Hybrid ReSTIR-Path-Tracing and Diffusion" (SIGGRAPH 2025 Advances course).
//
// The medium under the surface is homogeneous, set by the material where the light meets it. Light that
// crosses it without scattering, or scattering once, is path traced; the rest comes from Burley's diffusion
// profile with its single scattering taken out, R_n = R - R_1. Either way the path finds a point on the
// object's surface, an exit or a probe hit near the entry, which then stands in as a white Lambertian
// surface facing out: next event estimation there gathers the light entering the object, and the path
// carries on outwards from it. A subsurface event counts as one bounce.
//
// Simplifications: the interface is crossed by diffuse transmission both ways, the camera side weighted by
// OpenPBR's (1 - E(wo)) like the diffuse lobe, the other side by nothing. Probes go along the normal only
// and the profile ignores the medium's anisotropy. The two parts are picked between per path rather than
// both traced, and the single scattering event picks one channel's distance (the others weighted by
// the mixture pdf).
import OpenPBRBaseSubstrate;
import PathTracer.Geometry;
import PathTracer.Materials;
import PathTracer.Math;
import PathTracer.RayTracing;
import PathTracer.Resources;

// Where a path is within a subsurface event, the low bits of TracePath's subsurfaceState; the material
// index of the medium sits above them.
static const uint SUBSURFACE_STAGE_NONE = 0;
static const uint SUBSURFACE_STAGE_ENTERED = 1; // first segment through the medium: crosses it, or scatters once
static const uint SUBSURFACE_STAGE_SCATTERED = 2; // from the single scattering event to where its light leaves
static const uint SUBSURFACE_STAGE_PROBE = 3; // diffusion profile probe, down onto the surface from outside
static const uint SUBSURFACE_STAGE_BITS = 2;
static const uint SUBSURFACE_STAGE_MASK = 3;

// Burley's single scattering profile is alpha R_1 (first order of R in the single-scattering albedo, with
// the surface albedo fitted to it), R_1(r) = R1_SCALE (exp(-R1_SHORT r / l) + exp(-R1_LONG r / l)) / (l r),
// which holds R1_ENERGY of the light over the plane.
static const float R1_SCALE = 0.0423281;
static const float R1_SHORT = 5.434;
static const float R1_LONG = 1.8113;
static const float R1_ENERGY = 0.195781;

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

// A light path arrives at the far side of the medium, d from where it entered. Picks whether its light
// crossed without scattering (false: the path leaves the object here) or scattered once on the way (true:
// at distance t), with the throughput weighted for the pick. u: pick, channel, distance.
bool SubsurfaceScatters(SubsurfaceMedium m, float d, float3 u, inout float3 throughput, out float t)
{
    t = 0.0;
    float3 crossed = exp(-m.sigmaT * d);
    // What each part brings in expectation: the single scattering integral is albedo (1 - T(d)).
    float3 scattered = m.albedo * OneMinusExp(m.sigmaT * d);
    bool zero = (g_Frame.subsurfaceComponents & SUBSURFACE_ZERO_SCATTERING) != 0;
    bool single = (g_Frame.subsurfaceComponents & SUBSURFACE_SINGLE_SCATTERING) != 0;
    float crossedLuminance = Luminance(crossed);
    float pCross = !single ? 1.0
                 : !zero   ? 0.0
                           : 0.1 + 0.8 * crossedLuminance / max(crossedLuminance + Luminance(scattered), 1.0e-8);
    if (u.x < pCross)
    {
        throughput *= crossed / pCross;
        return false;
    }

    // One channel's exponential truncated to (0, d), weighted by the pdf of picking any of the three.
    float sigma = m.sigmaT[min(uint(u.y * 3.0), 2u)];
    t = -log(1.0 - u.z * OneMinusExp(sigma * d).x) / sigma;
    float3 transmittance = exp(-m.sigmaT * t);
    float3 pdfs = m.sigmaT * transmittance / OneMinusExp(m.sigmaT * d);
    float pdf = (pdfs.x + pdfs.y + pdfs.z) * (1.0 / 3.0);
    throughput *= m.albedo * m.sigmaT * transmittance / (pdf * (1.0 - pCross));
    return true;
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

// Carries on a path that took LOBE_SUBSURFACE at position, whose outward normal is normal, by its medium's
// mat. Either through the medium (origin and dir left as they are, pointing into it; extinction set) or as
// a diffusion profile probe (origin and dir replaced; probe gets its disk radius and height). Returns the
// stage, NONE when the probe radius falls outside the reach. u0: pick, channel, radius; u1.x: angle.
uint EnterSubsurface(OpenPBRBaseParams mat, float3 position, float3 normal, float3 u0, float3 u1, inout float3 origin,
                     inout float3 dir, inout float3 throughput, out float3 extinction, out float2 probe)
{
    SubsurfaceMedium m = GetSubsurfaceMedium(mat);
    extinction = float3(0.0, 0.0, 0.0);
    probe = float2(0.0, 0.0);

    // Picked by what each part holds over a thick flat slab: the whole profile less R_1's alpha R1_ENERGY.
    bool traced = (g_Frame.subsurfaceComponents & (SUBSURFACE_ZERO_SCATTERING | SUBSURFACE_SINGLE_SCATTERING)) != 0;
    bool diffusion = (g_Frame.subsurfaceComponents & SUBSURFACE_MULTIPLE_SCATTERING) != 0;
    float multiple = Luminance(max(m.colour - R1_ENERGY * m.albedo, 0.0)) / max(Luminance(m.colour), 1.0e-6);
    float pProbe = !diffusion ? 0.0 : !traced ? 1.0 : clamp(multiple, 0.2, 0.8);
    if (u0.x >= pProbe)
    {
        throughput /= 1.0 - pProbe;
        extinction = m.sigmaT;
        return SUBSURFACE_STAGE_ENTERED;
    }
    throughput /= pProbe;

    // One channel's profile, in r a mixture of exponentials with scales l / s and 3 l / s, weighted 1/4 and 3/4.
    uint channel = min(uint(u0.y * 3.0), 2u);
    float scale = m.meanFreePath[channel] / BurleyScale(m.colour)[channel];
    bool shortLobe = u0.z < 0.25;
    float v = shortLobe ? u0.z * 4.0 : (u0.z - 0.25) * (4.0 / 3.0);
    float r = max(-log(1.0 - v) * (shortLobe ? scale : 3.0 * scale), 1.0e-6 * scale);
    float reach = SubsurfaceProbeRadius(m);
    if (r >= reach)
        return SUBSURFACE_STAGE_NONE;

    // Down along the normal through the point r away on the tangent disk, over the sphere of the reach.
    float h = sqrt(reach * reach - r * r);
    float3 t, b;
    BuildOrthonormalBasis(normal, t, b);
    float phi = TWO_PI * u1.x;
    origin = position + (t * cos(phi) + b * sin(phi)) * r + normal * h;
    dir = -normal;
    probe = float2(r, h);
    return SUBSURFACE_STAGE_PROBE;
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

// The surface an exit or probe hit stands in as: white Lambertian, so what it gathers from outside is the
// light entering the medium there, which the subsurface event's weight has already carried to its exit.
OpenPBRBaseParams SubsurfaceExitMaterial()
{
    OpenPBRBaseParams p = OpenPBRDefaults();
    p.baseColour = float3(1.0, 1.0, 1.0);
    p.specularWeight = 0.0;
    return p;
}

// Closest hit on the medium's material along a ray, other surfaces ignored, packed as TraceClosest's is.
// An inline query, so nothing is saved around it, and texture filtering is left at the top mip.
HitPayload TraceSubsurfaceRay(float3 origin, float3 dir, float tMax, uint materialIndex)
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

    HitPayload payload = (HitPayload)0;
    payload.hitT = -1.0;
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return payload;

    // As ClosestHit, less the ray cone, emitter and curvature.
    PrimitiveDesc prim = LoadPrimitive(query.CommittedInstanceID());
    uint3 tri = LoadTriangle(prim, query.CommittedPrimitiveIndex());
    float3 bary = Barycentrics(query.CommittedTriangleBarycentrics());
    GpuVertex v0 = g_Vertices[tri.x];
    GpuVertex v1 = g_Vertices[tri.y];
    GpuVertex v2 = g_Vertices[tri.z];
    float3 objectNormal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    float4 objectTangent = v0.tangent * bary.x + v1.tangent * bary.y + v2.tangent * bary.z;

    float3x3 linearToWorld = (float3x3)query.CommittedObjectToWorld3x4();
    float3x3 cofactor = float3x3(cross(linearToWorld[1], linearToWorld[2]),
                                 cross(linearToWorld[2], linearToWorld[0]),
                                 cross(linearToWorld[0], linearToWorld[1]));
    float handedness = dot(linearToWorld[0], cofactor[0]) < 0.0 ? -1.0 : 1.0;
    float3 faceCross = cross(mul(linearToWorld, v1.position - v0.position), mul(linearToWorld, v2.position - v0.position));

    payload.hitT = query.CommittedRayT();
    payload.Ng = OctEncode(normalize(faceCross * handedness));
    payload.N = OctEncode(mul(cofactor, objectNormal) * handedness);
    payload.T = OctEncode(mul(linearToWorld, objectTangent.xyz));
    bool tangentNegative = (objectTangent.w < 0.0) != (handedness < 0.0);
    payload.materialId = prim.materialIndex | (tangentNegative ? TANGENT_SIGN_BIT : 0u);
    payload.uv = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
    payload.emitter = INVALID_LIGHT_INDEX;
    return payload;
}
