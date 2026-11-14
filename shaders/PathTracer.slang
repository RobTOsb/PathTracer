import OpenPBRBSDF;
import PathTracer.Resources;
import PathTracer.Math;
import PathTracer.Random;
import PathTracer.Geometry;
import PathTracer.RayTracing;
import PathTracer.Shadows;
import PathTracer.Sky;
import PathTracer.LightSampling;
import PathTracer.Materials;
import PathTracer.Guides;
import PathTracer.Psr;
import PathTracer.CameraState;
import PathTracer.GlassSplit;
import PathTracer.Subsurface;
import PathTracer.Photons;

static const uint RR_START_BOUNCE = 1;
// Longest path through a medium that absorption is evaluated over
static const float MAX_MEDIUM_DISTANCE = 1.0e30;
// How much a light's visibility must have changed since last frame, in some channel, to mark g_ShadowChanges.
static const float SHADOW_CHANGE_THRESHOLD = 0.25;

// TracePath's flags, one word so that a single register is held across each TraceRay. The DEFERRED_*
// flags (GlassSplit.slang) share it.
static const uint PATH_FROM_CAMERA = 0x10;
static const uint PATH_CHAIN_ACTIVE = 0x20; // the camera state's chain follows mirrors and glass in front of the camera
// The guides' surface continued the path with a specular reflection, so the next hit gives its
// specular hit distance.
static const uint PATH_HIT_DISTANCE_PENDING = 0x40;
static const uint PATH_PANE_HIT_PENDING = 0x80; // the brightest pane's reflection ray ends at the next hit
static const uint PATH_DIRECT_SEGMENT = 0x100;
// Diffuse bounces so far, 4 bits; the rest of the bounce count are specular ones.
static const uint PATH_DIFFUSE_BOUNCES_SHIFT = 9;
static const uint PATH_DIFFUSE_BOUNCES_MASK = 0xF;
// The segment left the surface the camera sees by a mirror-like lobe, so the light it finds is that surface's
// image, which the firefly clamp leaves alone.
static const uint PATH_MIRROR_SEGMENT = 0x2000;
// A chain ended on a specular reflection, so the next hit gives the hit distance of the surface it ended on.
static const uint PATH_END_HIT_PENDING = 0x8000;
static const uint PATH_SPLITS_SHIFT = 16; // glass splits so far, 4 bits (bits 16..19)
static const uint PATH_SPLITS_MASK = 0xF;
// The path is inside the solid object the camera sees: it refracted into it at the camera's surface and has only
// been reflected inside it since. Like the camera's surface itself, it takes no specular antialiasing.
static const uint PATH_IN_SEEN_OBJECT = 0x400000;
// Caustic photons (PathTracer/Photons.slang): where the path stands towards them, two bits (14 and 23) of PHOTON_*.
static const uint PATH_PHOTON_LOW = 0x4000;
static const uint PATH_PHOTON_HIGH = 0x800000;
static const uint PHOTON_NONE = 0;
static const uint PHOTON_CAMERA = 1;
// The surface that read them scattered diffusely, and only pass-throughs since.
static const uint PHOTON_ARMED = 2;
// And smooth surfaces since: the sunlight the path finds now is what the photons brought, so it is left out.
static const uint PHOTON_COVERED = 3;

uint PhotonState(uint flags)
{
    return ((flags >> 14) & 1u) | ((flags >> 22) & 2u);
}

uint SetPhotonState(uint flags, uint state)
{
    return (flags & ~(PATH_PHOTON_LOW | PATH_PHOTON_HIGH)) | ((state & 1u) << 14) | ((state & 2u) << 22);
}

// The lobes whose samples spend the diffuse bounce budget, the others spend the specular one.
static const uint DIFFUSE_BOUNCE_LOBES = LOBE_DIFFUSE | LOBE_DIFFUSE_TRANSMISSION | LOBE_SUBSURFACE | LOBE_FUZZ;

uint SetFlag(uint flags, uint flag, bool set)
{
    return set ? flags | flag : flags & ~flag;
}

// PATH_CHAIN_ACTIVE, which only primary surface replacement (the PSR permutation) sets, so without it
// the chain's code compiles away.
bool ChainActive(uint flags)
{
#ifdef PSR
    return (flags & PATH_CHAIN_ACTIVE) != 0;
#else
    return false;
#endif
}


// The firefly clamp: a contribution is scaled down to at most directClamp in its largest channel if the surface
// the camera sees scattered it (direct), to indirectClamp if it came further, and not at all if the camera sees
// it (seen), also in a mirror-like surface.
float3 ClampFirefly(float3 contribution, bool seen, bool direct)
{
    float limit = direct ? g_Frame.directClamp : g_Frame.indirectClamp;
    float m = MaxComponent(contribution);
    return seen || m <= limit ? contribution : contribution * (limit / m);
}

// RTXPT's firefly filter (PathTracerHelpers.hlsli). Each path carries k, 1 from the camera, which every scatter
// shrinks by a probability heuristic of the sample (ScatterFireflyK). Light a vertex finds (its emission or the sky with
// their MIS weights, a light sample's f cos L / pdf, the air's in-scattering) is scaled down to at most k times the
// threshold in the average of its channels (the brightest channel would shift hues), before the path's throughput.
// The threshold follows last frame's exposure: fireflyFilterThreshold times the radiance it maps to middle grey.
static const float FIREFLY_MIDDLE_GREY = 0.18;
static const float FIREFLY_MIN_K = 1.0e-5;

float FireflyFilterBase()
{
    if (!(g_Frame.fireflyFilterThreshold > 0.0) || g_Frame.exposureBuffer == INVALIDDESCRIPTORHANDLE)
        return FLT_MAX;
    RWStructuredBuffer<float> exposure = ResourceDescriptorHeap[g_Frame.exposureBuffer];
    float e = exposure[0];
    return e > 0.0 ? g_Frame.fireflyFilterThreshold * (FIREFLY_MIDDLE_GREY / e) : FLT_MAX;
}

// scale multiplies the limit, for light that already carries part of the throughput.
float3 FireflyFilter(float3 light, bool seen, float k, float scale = 1.0)
{
    if (seen || !(g_Frame.fireflyFilterThreshold > 0.0))
        return light;
    float limit = FireflyFilterBase() * k * scale;
    float average = (light.x + light.y + light.z) * (1.0 / 3.0);
    return average > limit ? light * (limit / average) : light;
}

// RTXPT's ComputeNewScatterFireflyFilterK, an ad hoc heuristic found by experiment: the wider the cone of directions
// the sample stands for (pdf, 0 for a delta event, which has none) and the less likely its lobe was picked, the
// smaller k gets.
float ScatterFireflyK(float k, float pdf, float lobeProbability)
{
    // The cone's apex angle from its solid angle, 1 / pdf (RTXPT's ComputeRayConeSpreadAngleExpansionByScatterPDF).
    float angle = pdf > 0.0 ? 2.0 * acos(max(-1.0, 1.0 - (1.0 / pdf) / TWO_PI)) : 0.0;
    float p = 32.0 / (32.0 + angle * angle) * sqrt(saturate(lobeProbability));
    return max(FIREFLY_MIN_K, k * p);
}

// The share of the mixture a BSDF sample's lobe was picked with.
float SampledLobeProbability(OpenPBRContext ctx, uint lobe)
{
    if (lobe == LOBE_SPECULAR_REFLECTION)
        return ctx.probSpecR;
    if (lobe == LOBE_SPECULAR_TRANSMISSION || lobe == LOBE_DELTA_TRANSMISSION)
        return ctx.probSpecT;
    if (lobe == LOBE_COAT)
        return ctx.probCoat;
    if (lobe == LOBE_FUZZ)
        return ctx.probFuzz;
    return ctx.probDiffuse; // diffuse, diffuse transmission and subsurface share it
}

// Whether light along the segment just traced reaches the surface the camera sees or a surface it left.
bool DirectSegment(uint flags, bool fromCamera, bool chainActive)
{
    return fromCamera || chainActive || (flags & PATH_DIRECT_SEGMENT) != 0;
}

// Whether the camera sees light along the segment just traced, also in a mirror-like surface it left.
bool SeenSegment(uint flags, bool fromCamera, bool chainActive)
{
    return fromCamera || chainActive || (flags & PATH_MIRROR_SEGMENT) != 0;
}

bool LeavesLightsToBsdf(OpenPBRContext ctx)
{
    return !SunIsDelta() && ctx.probDiffuse <= 0.0 && ctx.probFuzz <= 0.0 && max(ctx.alpha.x, ctx.alpha.y) <= 1.0e-4 &&
           (ctx.probCoat <= 0.0 || ctx.coatAlpha <= 1.0e-4);
}

// A sharp reflection lobe (GGX alpha at most reflectionMinProbabilityMaxAlpha) is picked with at least
// reflectionMinProbability, the other lobes scaled down to make room. The weights and the MIS PDFs take the whole
// mixture, so this only moves noise: off the reflection, whose small bright images (an emitter in a polished floor)
// would otherwise show in just the share of frames its albedo earns, onto the diffuse lobe, which denoises better.
// A surface whose coat leads its guides keeps its choice, so the guides follow the same lobe.
void FloorReflectionProbability(inout OpenPBRContext ctx)
{
    float minProbability = g_Frame.reflectionMinProbability;
    if (!(ctx.probSpecR > 0.0 && ctx.probSpecR < minProbability) ||
        max(ctx.alpha.x, ctx.alpha.y) > g_Frame.reflectionMinProbabilityMaxAlpha || CoatLeadsGuides(ctx))
        return;
    float scale = (1.0 - minProbability) / (1.0 - ctx.probSpecR);
    ctx.probSpecR = minProbability;
    ctx.probSpecT *= scale;
    ctx.probDiffuse *= scale;
    ctx.probCoat *= scale;
    ctx.probFuzz *= scale;
}

// The extinction of the medium the next segment crosses and the Russian roulette scale, as halves
// for fewer registers across TraceRay.
uint2 PackSegmentMedium(float3 extinction, float rrInvScale)
{
    // A medium too thin for a half still counts as one.
    extinction = select(extinction > 0.0, max(extinction, 6.2e-5), 0.0);
    return uint2(PackHalf2(extinction.xy), PackHalf2(float2(extinction.z, rrInvScale)));
}

float4 UnpackSegmentMedium(uint2 packed)
{
    return float4(UnpackHalf2(packed.x), UnpackHalf2(packed.y));
}

// No medium and a scale of 1, as a literal since f32tof16 is not folded.
static const uint2 SEGMENT_MEDIUM_AIR = uint2(0, 0x3C000000);

// Applies the segment just traced to the camera state: t is its length, 0 for a miss, through a medium
// of the given extinction. The CAMERA_* parts in loaded are in registers; the hit distances for parts
// that are not go straight to g_CameraState.
void ArriveCameraState(inout CameraState camera, uint loaded, bool fromCamera, bool chainActive, bool hitDistancePending,
                       bool paneHitPending, bool endHitPending, bool hit, float t, float3 extinction)
{
    float hitDistance = hit ? t : SKY_HIT_DISTANCE;
    if (endHitPending)
        StoreChainEndHitDistance(hitDistance);
    if (hitDistancePending)
    {
        if ((loaded & CAMERA_GUIDES) != 0)
            camera.guides.specularHitDistance = hitDistance;
        else
            StoreGuideHitDistance(hitDistance);
    }
    if (paneHitPending)
    {
        if ((loaded & CAMERA_PANES) != 0)
            camera.panes.hitDistance = hitDistance;
        else
            StorePaneHitDistance(hitDistance);
    }
    if (chainActive)
    {
        camera.chain.length += t;
        camera.chain.throughput *= exp(-extinction * t);
    }
    if (fromCamera)
        camera.distance += t;
}

// What ShadeSurface leaves TracePath to decide: whether and how the path goes on, and the number Russian
// roulette compares with.
struct SurfaceOutcome
{
    bool continuePath;
    bool passThrough;
    float rrU;
};

// The weight of the light a BSDF sample found on an emitter's front face, MIS weighted against NEE at the previous
// vertex unless the triangle has no light (degenerate or black) or light buffer NEE is off, with the light's emission
// scale as NEE's radiance has. Its light tree walk needs only the packed hit, so it comes before either shading
// specialisation, outside of which the walk is compiled once.
float EmissionWeight(HitPayload payload, float3 origin, float3 dir, bool fromCamera, bool smoothSegment, float prevBsdfPdf,
                     PackedLightSelectVertex prevVertex)
{
    float emissionWeight = 1.0;
    if (dot(OctDecode(payload.Ng), dir) < 0.0 && payload.emitter != INVALID_LIGHT_INDEX)
    {
        uint lightIndex = g_Emitters[payload.emitter];
        if (lightIndex != INVALID_LIGHT_INDEX)
        {
            emissionWeight = LightEmissionScale(lightIndex);
            if (!fromCamera && !smoothSegment && g_Frame.lightCount > 0)
                emissionWeight *= PowerHeuristic(prevBsdfPdf, EmitterLightPdf(UnpackLightSelectVertex(prevVertex), lightIndex,
                                                                              dir, origin + dir * payload.hitT));
        }
    }
    return emissionWeight;
}

// What ShadeSurfaceBegin hands ShadeSurfaceEnd across the light sample, which TracePath takes between them so that
// SampleLight, whose light tree walk is the largest code of all, is compiled once rather than in both specialisations.
// The walk needs registers of its own, so across it the material is held packed and the BSDF context not at all;
// ShadeSurfaceEnd rebuilds both, and everything from there on, sampling and evaluation alike, uses that pair.
struct SurfaceShading
{
    HitInfo hit;
    PackedMaterial packedMat;
    PackedLightSelectVertex vertex; // for SampleLight, and the next segment's MIS
    float3 lightU; // the light sample (xy) and the Russian roulette decision (z), one hash
    bool sampleLights;
    bool lastSegment;
    bool opaque;
};

// The hit's shading up to the light sample: the material, emission (by emissionWeight, see EmissionWeight) and the
// light select vertex, for the segment that ended at the payload's hit. SIMPLE: a GPU_MATERIAL_SIMPLE material, whose
// specialisation drops every lobe but the base's and so holds far fewer registers.
SurfaceShading ShadeSurfaceBegin<let SIMPLE : bool>(HitPayload payload, float3 origin, float3 dir, uint bounce,
                                                    bool fromCamera, bool chainActive, float emissionWeight, uint flags,
                                                    inout uint rng, inout float3 radiance, float3 throughput,
                                                    inout RayCone cone, float fireflyK)
{
    float3 wo = -dir;
    bool frontFace = dot(OctDecode(payload.Ng), wo) > 0.0;
    // Light found here is direct for a surface the camera sees, or for the one the segment left.
    bool visible = fromCamera || chainActive;
    bool directArrival = DirectSegment(flags, fromCamera, chainActive);
    bool seenArrival = SeenSegment(flags, fromCamera, chainActive);

    HitInfo hit = UnpackHit(payload, origin, dir);

    // Material and shading basis for this hit, its textures filtered over the cone's footprint.
    cone.width += cone.spread * hit.t;

    // A segment whose sample went past its kind's bounce budget only picks up the light it found, the
    // MIS partner of the previous vertex's NEE.
    uint diffuseBounces = (flags >> PATH_DIFFUSE_BOUNCES_SHIFT) & PATH_DIFFUSE_BOUNCES_MASK;
    uint specularBounces = bounce - diffuseBounces;
    bool emissionOnly = diffuseBounces > g_Frame.maxDiffuseBounces || specularBounces > g_Frame.maxSpecularBounces;
    // With both budgets spent a BSDF sample could only find light, so the last segment takes none and
    // samples the lights whatever the surface.
    bool lastSegment = emissionOnly ||
                       (diffuseBounces >= g_Frame.maxDiffuseBounces && specularBounces >= g_Frame.maxSpecularBounces);

    // Blended surfaces were already taken or passed by chance in traversal (AnyHit), so this one is there.

    // Specular antialiasing where the camera sees the surface past mirrors and glass (PSR's chain), never on the
    // camera's own surface or inside the object it belongs to: how the vertex normals turn across the cone's
    // footprint, stretched by 1 / cos along the view, and the normal maps' own spread.
    SpecularAA specularAA = NoSpecularAA();
    if (chainActive && !fromCamera && (flags & PATH_IN_SEEN_OBJECT) == 0)
        specularAA = FootprintSpecularAA(hit, wo, cone.width);
    OpenPBRBaseParams mat = LoadMaterial(hit, cone.width * g_Frame.textureLodScale, bounce < g_Frame.anisotropicSegments,
                                         SIMPLE, specularAA, diffuseBounces >= g_Frame.reducedMaterialDiffuseBounces);
    if (frontFace && any(hit.emission > 0.0))
        radiance += ClampFirefly(throughput * FireflyFilter(hit.emission * emissionWeight, seenArrival, fireflyK), seenArrival,
                                 directArrival);

    // Opaque and thin-walled surfaces are treated as two-sided: face the frame towards the viewer
    // so a back face is not mistaken for the inside of a transmissive object. The vertex normal follows
    // Ng to the viewer's side, also where the asset has it against the winding (whole leaves of
    // Bistro's ivy), which would otherwise leave the viewer below the shading normal.
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

    SurfaceShading s;
    s.lightU = Rand3(rng);
    s.sampleLights = !emissionOnly && (lastSegment || !LeavesLightsToBsdf(ctx));
    // Sampled with the rounding prevVertex keeps, so both ends of an emitter's MIS see the same pmf. A
    // vertex that samples no lights needs none, since the segment it starts takes no MIS. A surface the camera
    // sees, also past PSR's mirrors and glass, selects lights as the primary hit does.
    s.vertex = (PackedLightSelectVertex)0;
    if (s.sampleLights)
        s.vertex = PackLightSelectVertex(MakeLightSelectVertex(hit.position, mat, ctx, wo, visible ? 0u : bounce));
    s.hit = hit;
    s.packedMat = PackMaterial(mat);
    s.lastSegment = lastSegment;
    s.opaque = opaque;
    return s;
}

// The rest of the hit's shading, after the light sample ls (zero where shading.sampleLights is false): NEE, the BSDF
// sample (and the glass splits), the DLSS-RR camera state and subsurface scattering. arrivedExtinction is the medium
// the segment crossed, segmentExtinction comes back as the next one's.
SurfaceOutcome ShadeSurfaceEnd<let SIMPLE : bool>(SurfaceShading shading, LightSample ls, inout float3 origin,
                                                  inout float3 dir, uint bounce, bool fromCamera, bool chainActive,
                                                  bool hitDistanceArrived, bool paneHitArrived, bool endHitArrived,
                                                  bool smoothSegment, bool firstSegment, bool cameraSegment,
                                                  float3 arrivedExtinction, inout uint flags, inout uint rng,
                                                  inout float3 radiance, inout float3 throughput, inout float prevBsdfPdf,
                                                  inout PackedLightSelectVertex prevVertex, inout RayCone cone,
                                                  inout float3 segmentExtinction, inout float rrInvScale,
                                                  inout float fireflyK)
{
    float3 wo = -dir;
    bool visible = fromCamera || chainActive;
    HitInfo hit = shading.hit;
    PackedLightSelectVertex vertex = shading.vertex;
    float3 lightU = shading.lightU;
    bool sampleLights = shading.sampleLights;
    bool lastSegment = shading.lastSegment;
    bool opaque = shading.opaque;
    OpenPBRBaseParams mat = UnpackMaterial(shading.packedMat, SIMPLE);
    OpenPBRContext ctx = OpenPBRSetup(mat, wo);
    // Before NEE, so that its MIS sees the mixture the BSDF sample is drawn from. The light select vertex above keeps
    // the material's own shares, at both ends of an emitter's MIS.
    if (visible && opaque && !lastSegment)
        FloorReflectionProbability(ctx);
    float3 extinction = MediumExtinction(mat);

    // Caustics: the surface with a diffuse part the camera ray meets is the pixel's photon receiver, on which
    // PhotonMap.slang splats the photons and adds their irradiance by its diffuse albedo, under its specular reflection.
    const uint photonState = PhotonState(flags);
    if (photonState == PHOTON_CAMERA && ctx.probDiffuse > 0.0)
    {
        PrimaryGuides reflectance = (PrimaryGuides)0;
        ReflectanceGuides(mat, ctx, wo, reflectance);
        float3 albedo = reflectance.diffuseAlbedo * saturate(1.0 - reflectance.specularAlbedo);
        float3 n = dot(hit.Ng, wo) >= 0.0 ? hit.Ng : -hit.Ng;
        WritePhotonReceiver(hit.position, n, throughput * albedo * (1.0 / PI));
    }
    bool deltaTransmission = HasDeltaTransmission(mat);
    float3 neeContribution = float3(0.0, 0.0, 0.0);
    bool traceShadowRay = false;
    // An opaque surface reflects whatever is above its shading normal's horizon, also below its triangle's
    // plane, where the shadow ray leaves from the smooth surface (SmoothReflectionOrigin).
    if (ls.pdf > 0.0 && (!opaque || dot(ls.direction, mat.geometryNormal) > 0.0))
    {
        float bsdfPdf;
        float3 fCos = OpenPBREvalPdf(mat, ctx, ls.direction, bsdfPdf);
        if (any(fCos > 0.0))
        {
            // The last segment takes no BSDF sample, which leaves NEE as the only strategy there.
            float w = ls.lightSamplingOnly || lastSegment ? 1.0 : PowerHeuristic(ls.pdf, bsdfPdf);
            // A solid's own medium is on the far side of Ng, and an interface inside a medium has one on the near side.
            bool solid = !opaque && !mat.geometryThinWalled;
            float3 shadowExtinction = !solid ? float3(0.0, 0.0, 0.0)
                                      : dot(ls.direction, hit.Ng) < 0.0 ? extinction
                                                                         : ExteriorExtinction(hit.materialId);
            // The sun is FLT_MAX away. Capped, so a zero extinction cannot meet an overflowed distance
            // (0 * inf) once the compiler folds exp's log2(e) into it; any medium is far shorter.
            float3 shadowTransmittance = any(shadowExtinction > 0.0)
                                             ? exp(-shadowExtinction * min(ls.distance, MAX_MEDIUM_DISTANCE))
                                             : AtmosphereTransmittance(ls.distance);
            // Clamped unshadowed, which the shadow ray's visibility (at most 1) only scales down.
            float3 light = FireflyFilter(fCos * ls.radiance * shadowTransmittance * (w / ls.pdf), false,
                                         ScatterFireflyK(fireflyK, ls.pdf, 1.0));
            neeContribution = ClampFirefly(throughput * light, false, visible);
            traceShadowRay = true;
        }
    }
    // Continue the path by sampling the BSDF, unless this is the last segment
    bool continuePath = false;
    bool specularReflection = false;
    bool passThrough = false;
    uint lobe = LOBE_NONE;
    const float3 dirIn = dir;
    float3 passTransmittance = float3(1.0, 1.0, 1.0);
    // The two branches cover the specular lobes only, so nothing with a diffuse, coat or fuzz lobe splits.
    uint splits = (flags >> PATH_SPLITS_SHIFT) & PATH_SPLITS_MASK;
    bool splittable = !SIMPLE && fromCamera && !lastSegment && splits < g_Frame.glassReflectionSplits && ctx.probDiffuse <= 0.0 &&
                      ctx.probCoat <= 0.0 && ctx.probFuzz <= 0.0 && ctx.probSpecR > 0.0 && ctx.probSpecR < 1.0;
    // A thin pane: its reflection now, what is behind it deferred.
    bool paneSplit = splittable && deltaTransmission;
    // Solid glass: the refraction now, which the guides' chain follows, its reflection deferred.
    bool glassSplit = splittable && !mat.geometryThinWalled && !ctx.inside && CoatedSpecularRoughness(mat) <= 0.0;
    if (paneSplit || glassSplit)
        flags += 1u << PATH_SPLITS_SHIFT;

    if (paneSplit)
    {
        // What is behind the pane is along the camera ray, with the cone as it reached the pane.
        flags = (flags & ~DEFERRED_MASK) | DEFERRED_PENDING | ((bounce + 1) << DEFERRED_BOUNCE_SHIFT);
        DeferredBranch deferred;
        deferred.origin = OffsetRay(hit.position, hit.Ng, dir);
        deferred.throughput = throughput * ThinDeltaTransmittance(mat, ctx.wo.z);
        deferred.dir = dir;
        deferred.coneWidth = cone.width;
        deferred.coneSpread = cone.spread;
        StoreDeferredBranch(deferred);
    }

    // Solid glass takes two samples, its deferred reflection's first, from one call site so the
    // sampler is inlined once. A split sets u.x to land in its lobe, so a sample is that or none.
    // The reflection goes to g_DeferredBranches as soon as it is sampled.
    BSDFSample bs = (BSDFSample)0;
    bool reflectionDeferred = false;
    // A chain past the camera's first surface takes the lobe it follows, so its guides do not fall back to last
    // frame's (PsrChainBrokenByChance) whenever the path would have taken the other.
    uint forcedLobe = chainActive && !lastSegment && g_Frame.psrStrongerLobe != 0 ? PsrStrongerLobe(mat, ctx) : uint(LOBE_NONE);
    if (!lastSegment)
    {
        [loop]
        for (uint s = glassSplit ? 0 : 1; s < 2; ++s)
        {
            float3 u = Rand3(rng);
            if (s == 0 || paneSplit)
                u.x = 0.0;
            else if (glassSplit)
                u.x = ctx.probSpecR; // the first value past the reflection's share
            else if (forcedLobe != LOBE_NONE)
                u.x = forcedLobe == LOBE_SPECULAR_REFLECTION ? 0.0 : ctx.probSpecR;
            BSDFSample sample = OpenPBRSample(mat, ctx, u);
            if (s != 0)
            {
                bs = sample;
            }
            else if (sample.lobe != LOBE_NONE)
            {
                flags = (flags & ~DEFERRED_MASK) | DEFERRED_PENDING | DEFERRED_REFLECTION | ((bounce + 1) << DEFERRED_BOUNCE_SHIFT);
                DeferredBranch deferred;
                deferred.origin = OffsetRay(hit.position, hit.Ng, sample.wi);
                deferred.dir = sample.wi;
                deferred.throughput = throughput * sample.weight * ctx.probSpecR;
                deferred.coneWidth = cone.width;
                deferred.coneSpread = min(cone.spread + g_Frame.rayConeBsdfSpread * SampledConeSpread(sample.pdf / ctx.probSpecR), PI);
                StoreDeferredBranch(deferred);
                reflectionDeferred = true;
            }
        }
    }

    lobe = bs.lobe;
    passThrough = bs.lobe == LOBE_DELTA_TRANSMISSION;
    continuePath = bs.lobe != LOBE_NONE;
    if (passThrough)
    {
        // Same direction, same cone. NEE at the segment's start already sees through this wall, so
        // its MIS only gains the probability of picking the lobe, which bs.pdf is.
        throughput *= bs.weight;
        prevBsdfPdf *= bs.pdf;
        passTransmittance = bs.weight * bs.pdf;
        segmentExtinction = float3(0.0, 0.0, 0.0);
        // A delta event, picked with bs.pdf.
        fireflyK = ScatterFireflyK(fireflyK, 0.0, bs.pdf);
    }
    else if (continuePath)
    {
        // A pane's reflection is not its guides' (what is behind it is), see PaneReflection. The
        // guides' reflection is the coat's or the base's, whichever reflects more.
        specularReflection = !paneSplit && bs.lobe == (CoatLeadsGuides(ctx) ? LOBE_COAT : LOBE_SPECULAR_REFLECTION);
        float lobeProbability = paneSplit || forcedLobe == LOBE_SPECULAR_REFLECTION ? ctx.probSpecR
                                : glassSplit || forcedLobe == LOBE_SPECULAR_TRANSMISSION ? ctx.probSpecT
                                : 1.0;
        if ((bs.lobe & DIFFUSE_BOUNCE_LOBES) != 0)
            flags += 1u << PATH_DIFFUSE_BOUNCES_SHIFT;
        float lobeAlpha = bs.lobe == LOBE_COAT ? ctx.coatAlpha : max(ctx.alpha.x, ctx.alpha.y);
        bool mirrorLike = (bs.lobe & (LOBE_SPECULAR_REFLECTION | LOBE_SPECULAR_TRANSMISSION | LOBE_COAT)) != 0 && lobeAlpha <= 1.0e-4;
        flags = SetFlag(flags, PATH_MIRROR_SEGMENT, visible && mirrorLike);
        throughput *= bs.weight * lobeProbability;
        prevBsdfPdf = bs.pdf;
        prevVertex = vertex;
        // Surface curvature is ignored. The cap stops a tiny pdf from making the width infinite.
        cone.spread = min(cone.spread + g_Frame.rayConeBsdfSpread * SampledConeSpread(bs.pdf / lobeProbability), PI);
        // A lobe a split or the chain took was not left to chance.
        fireflyK = ScatterFireflyK(fireflyK, bs.pdf / lobeProbability, lobeProbability < 1.0 ? 1.0 : SampledLobeProbability(ctx, bs.lobe));

        // Heading into the object means travelling through its medium until the next hit, and leaving it
        // through the medium outside, air unless the surface is an interface inside one (the tea against a
        // teapot's glass). Opaque and thin-walled frames face the viewer, so only solid glass has an outside.
        if (dot(bs.wi, hit.Ng) < 0.0)
            segmentExtinction = extinction;
        else
            segmentExtinction = opaque || mat.geometryThinWalled ? float3(0.0, 0.0, 0.0) : ExteriorExtinction(hit.materialId);
        bool intoSolid = !opaque && !mat.geometryThinWalled && dot(bs.wi, hit.Ng) < 0.0;
        flags = SetFlag(flags, PATH_IN_SEEN_OBJECT, intoSolid && (fromCamera || (flags & PATH_IN_SEEN_OBJECT) != 0));
        dir = bs.wi;
    }
    // Lights the next segment finds come at full weight if this vertex left them to its BSDF sample, or,
    // through a pass-through wall, if the vertex the segment continues from did.
    // From the camera, which sampled none, lights come at full weight too.
    flags = SetFlag(flags, DEFERRED_SMOOTH_SEGMENT, passThrough ? smoothSegment || fromCamera : !sampleLights);
    // The photons' part of the path: a pass-through keeps it; a diffuse bounce from the receiver arms it, and a smooth
    // lobe after that covers what it goes on to find; anything else ends it. Thin-walled surfaces cast no photons
    // (TracePhoton), so they cover nothing: the sunlight found past one is the path tracer's to count.
    if (photonState != PHOTON_NONE && !passThrough)
    {
        bool smooth = continuePath && PhotonSmoothLobe(ctx, lobe);
        uint next = PHOTON_NONE;
        if (photonState == PHOTON_CAMERA)
            next = continuePath && lobe == LOBE_DIFFUSE && ctx.probDiffuse > 0.0 ? PHOTON_ARMED : PHOTON_NONE;
        else
            next = smooth && !mat.geometryThinWalled ? PHOTON_COVERED : PHOTON_NONE;
        flags = SetPhotonState(flags, next);
    }

    // DLSS-RR guides: the primary surface, or the first surface past the mirrors and glass in front
    // of it. A surface that takes another lobe, or ends the path, ends the chain on itself.
    float psrFraction = 1.0;
#ifdef PSR
    bool psrEvent = continuePath && !paneSplit && (fromCamera || chainActive) &&
                    (glassSplit ? CoatedSpecularRoughness(mat) <= g_Frame.psrMaxRoughness : IsPsrEvent(mat, ctx, lobe, psrFraction));
#else
    const bool psrEvent = false;
#endif
    bool chainEnds = chainActive && !passThrough && !psrEvent;
    if (chainEnds)
    {
        CameraState none = (CameraState)0;
        ArriveCameraState(none, 0, false, false, hitDistanceArrived, paneHitArrived, endHitArrived, true, hit.t, arrivedExtinction);
        flags &= ~PATH_CHAIN_ACTIVE;
        // A chain the path left by chance keeps last frame's guides, which followed it or its like.
        if (continuePath && PsrChainBrokenByChance(mat, ctx, lobe))
        {
            HoldGuides();
        }
        else
        {
            PrimaryGuides seen;
            ReflectanceGuides(mat, ctx, wo, seen);
            PsrChainEnd end;
            end.diffuseAlbedo = seen.diffuseAlbedo;
            end.specularAlbedo = seen.specularAlbedo;
            end.normal = mat.geometryNormal;
            end.roughness = GuideRoughness(mat, ctx);
            end.curvature = ViewedCurvature(hit, wo);
            end.dir = dirIn;
            end.Ng = hit.Ng;
            end.t = hit.t;
            end.throughputScale = exp(-arrivedExtinction * hit.t);
            end.hitDistance = InitialSpecularHitDistance(end.roughness);
            end.replaceable = !ctx.inside;
            end.sky = false;
            StoreChainEnd(end);
            flags = SetFlag(flags, PATH_END_HIT_PENDING, end.replaceable && specularReflection);
        }
    }
    else if (fromCamera)
    {
        // The surface the camera sees: only the parts it changes come in, its guides written afresh, and they go
        // out again before a chain starts here, which is stored on its own.
        bool panesChange = paneSplit || reflectionDeferred;
        uint parts = (panesChange ? CAMERA_PANES : 0) | CAMERA_VIEW;
        CameraState camera = firstSegment ? DefaultCameraState(dirIn) : LoadCameraState(parts);
        ArriveCameraState(camera, firstSegment ? CAMERA_ALL : parts, true, false, hitDistanceArrived, paneHitArrived, endHitArrived,
                          true, hit.t, arrivedExtinction);

        // What the camera sees of the surface: all that comes through to it.
        float3 seenScale = camera.transmittance;
        camera.guides.position = float4(hit.position, 1.0);
        camera.guides.normal = mat.geometryNormal;
        camera.guides.roughness = GuideRoughness(mat, ctx);
        // Until the path continues with a specular reflection: no reflection data, or last frame's.
        camera.guides.specularHitDistance = InitialSpecularHitDistance(camera.guides.roughness);
        camera.guides.curvature = ViewedCurvature(hit, wo);
        // A moving object's point was elsewhere last frame, which RayGen's motion vectors take in.
        camera.guides.hasMotion = false;
        camera.guides.displacement = hit.motion;
        camera.guides.replaced = false;
        // A blended surface the camera ray took by chance: RayGen mixes it with what is behind it.
        camera.guides.blendOpacity = g_Frame.blendGuides != 0 ? LoadOpacity(hit, cone.width * g_Frame.textureLodScale, bounce < g_Frame.anisotropicSegments) : 1.0;
        ReflectanceGuides(mat, ctx, wo, camera.guides);
        camera.guides.diffuseAlbedo *= seenScale;
        camera.guides.specularAlbedo *= seenScale;

        if (paneSplit)
        {
            PrimaryGuides seen;
            ReflectanceGuides(mat, ctx, wo, seen);
            if (AddPaneReflection(camera.panes, camera.transmittance * seen.specularAlbedo, camera.distance,
                                  ViewedCurvature(hit, wo)) &&
                continuePath)
                flags |= PATH_PANE_HIT_PENDING;
            // The reflection branch leaves the camera's transmittance and distance alone, so they take on
            // the pane's now, for what is behind it.
            camera.transmittance *= ThinDeltaTransmittance(mat, ctx.wo.z);
        }
        else if (reflectionDeferred)
        {
            // The guides' specular albedo is the glass's reflection, which moves to the panes.
            if (AddPaneReflection(camera.panes, camera.guides.specularAlbedo, camera.distance, ViewedCurvature(hit, wo)))
                flags |= DEFERRED_BRIGHTEST_PANE;
            camera.guides.specularAlbedo = float3(0.0, 0.0, 0.0);
        }

        if (specularReflection)
            flags |= PATH_HIT_DISTANCE_PENDING;
        // A chain that refracts into solid glass carries the glass's transmittance itself (SpecularChain::throughGlass).
        bool chainThroughGlass = psrEvent && !specularReflection && !mat.geometryThinWalled;
        if (!opaque && !mat.geometryThinWalled && !chainThroughGlass)
            camera.guides.diffuseAlbedo += seenScale * PsrEventThroughput(mat, ctx, wo, false);
        if (passThrough)
            camera.transmittance *= passTransmittance;
        StoreCameraState(camera, firstSegment ? CAMERA_ALL : CAMERA_GUIDES | parts);

        if (psrEvent)
        {
            CameraState started = (CameraState)0;
            started.chain = StartSpecularChain(mat, ctx, hit, wo, dirIn, dir, specularReflection, psrFraction, camera.distance,
                                               cone.width);
            StoreCameraState(started, CAMERA_CHAIN);
            flags |= PATH_CHAIN_ACTIVE;
        }
    }
    else if (chainActive)
    {
        // A chain that goes on through this surface: the event's share of the light and its geometry first,
        // then only the chain comes in.
        float3 eventThroughput = passThrough ? passTransmittance : PsrEventThroughput(mat, ctx, wo, specularReflection);
        float3 eventNormal = mat.geometryNormal;
        bool eventThinWalled = mat.geometryThinWalled;
        float eventEta = ctx.eta;
        CameraState camera = LoadCameraState(CAMERA_CHAIN);
        ArriveCameraState(camera, CAMERA_CHAIN, false, true, hitDistanceArrived, paneHitArrived, endHitArrived, true, hit.t,
                          arrivedExtinction);
        ChainArrive(camera.chain, dirIn, hit);
        camera.chain.throughput *= eventThroughput;
        if (!passThrough)
            AddPsrEvent(camera.chain, hit, eventNormal, eventThinWalled, eventEta, dirIn, dir, specularReflection, cone.width);
        StoreCameraState(camera, CAMERA_CHAIN);
    }
    else if (cameraSegment)
    {
        // Only hit distances arrive, which go straight to g_CameraState.
        CameraState none = (CameraState)0;
        ArriveCameraState(none, 0, false, false, hitDistanceArrived, paneHitArrived, endHitArrived, true, hit.t, arrivedExtinction);
    }

    if (continuePath)
    {
        // What an opaque surface reflects leaves from the smooth surface, not into the mesh, however its
        // triangle faces. Subsurface paths go into it.
        origin = opaque && lobe != LOBE_SUBSURFACE ? SmoothReflectionOrigin(hit, wo, dir) : OffsetRay(hit.position, hit.Ng, dir);
        // Through the medium under the surface: what leaves here comes in at other points of the
        // object, and the path goes on out of one of them.
        if (!SIMPLE && lobe == LOBE_SUBSURFACE)
        {
            SubsurfaceExit exit = (SubsurfaceExit)0;
            // Its light carries the throughput already, so the firefly filter's limit takes on the throughput's average.
            if (dot(dir, hit.Ng) < 0.0)
                radiance += ClampFirefly(FireflyFilter(SubsurfaceScatter(mat, hit.position, hit.Ng, hit.N, hit.materialId, dir, throughput,
                                                                         bounce + 1, rng, exit),
                                                       false, fireflyK, (throughput.x + throughput.y + throughput.z) * (1.0 / 3.0)),
                                         false, visible);
            continuePath = exit.valid;
            origin = exit.origin;
            dir = exit.dir;
            throughput = exit.throughput;
            prevBsdfPdf = exit.pdf;
            prevVertex = exit.vertex;
            segmentExtinction = float3(0.0, 0.0, 0.0);
        }
    }

    if (traceShadowRay)
    {
        // As far along as the light was from the hit point.
        float3 shadowOrigin = opaque ? SmoothReflectionOrigin(hit, wo, ls.direction) : OffsetRay(hit.position, hit.Ng, ls.direction);
        float shadowDistance = ls.distance - dot(shadowOrigin - hit.position, ls.direction);
        float3 visibility = TraceShadowRay(shadowOrigin, ls.direction, shadowDistance);
        radiance += neeContribution * visibility;
#ifdef DLSS_RR
        if (fromCamera && g_Frame.shadowChanges != 0)
        {
            float3 previousVisibility = TraceShadowRayIn(ResourceDescriptorHeap[g_Frame.previousScene], shadowOrigin,
                                                         ls.direction, shadowDistance);
            float3 change = abs(visibility - previousVisibility);
            if (max(change.x, max(change.y, change.z)) > SHADOW_CHANGE_THRESHOLD)
                g_ShadowChanges[DispatchRaysIndex().xy] = 1.0;
        }
#endif
    }

    SurfaceOutcome outcome;
    outcome.continuePath = continuePath;
    outcome.passThrough = passThrough;
    outcome.rrU = lightU.z;
    return outcome;
}

// The DLSS-RR guides come out in g_CameraState, see FinishGuides.
float3 TracePath(float3 origin, float3 dir, inout uint rng)
{
    uint flags = SetPhotonState(PATH_FROM_CAMERA, g_Frame.photons.enabled != 0 ? PHOTON_CAMERA : PHOTON_NONE);
    uint2 segmentMedium = SEGMENT_MEDIUM_AIR;

    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float prevBsdfPdf = 0.0;
    PackedLightSelectVertex prevVertex = (PackedLightSelectVertex)0; // where the current segment started, once a surface scattered
    RayCone cone = { 0.0, g_Frame.rayConePixelSpread };
    float fireflyK = 1.0; // the firefly filter's probability heuristic, see FireflyFilter

    uint bounce = 0;
    for (;;)
    {
        // ShadeSurface ends the path by the vertex both budgets together reach; the bounces are indirect,
        // so the primary segment comes on top.
        for (; bounce <= g_Frame.maxDiffuseBounces + g_Frame.maxSpecularBounces; ++bounce)
        {
            uint rayFlags = OpacityMicromapRayFlags(bounce);
            // Deep indirect rays skip the alpha work: blended instances are left out, alpha-tested ones are opaque.
            uint instanceMask = 0xFF;
            if (((flags >> PATH_DIFFUSE_BOUNCES_SHIFT) & PATH_DIFFUSE_BOUNCES_MASK) >= g_Frame.opaqueAlphaDiffuseBounces)
            {
                rayFlags |= RAY_FLAG_FORCE_OPAQUE;
                instanceMask = uint(INSTANCE_MASK_SOLID);
            }
            // Only the DLSS-RR guides read the shape of camera and chain hits.
#ifdef DLSS_RR
            const bool needCurvature = (flags & (PATH_FROM_CAMERA | PATH_CHAIN_ACTIVE)) != 0;
#else
            const bool needCurvature = false;
#endif
            HitPayload payload = TraceClosest(origin, dir, bounce >= SER_START_BOUNCE, needCurvature, rayFlags, instanceMask);
            // Unpacked after the in-scatter's shadow ray, so only the packed payload is held across it.
            HitInfo hit = HitDistance(payload);

            // Decoded after the trace, so only the packed forms were held across it.
            const bool fromCamera = (flags & PATH_FROM_CAMERA) != 0;
            const bool chainActive = ChainActive(flags);
            const bool hitDistanceArrived = (flags & PATH_HIT_DISTANCE_PENDING) != 0;
            const bool paneHitArrived = (flags & PATH_PANE_HIT_PENDING) != 0;
            const bool endHitArrived = (flags & PATH_END_HIT_PENDING) != 0;
            const bool smoothSegment = (flags & DEFERRED_SMOOTH_SEGMENT) != 0;
            // Nothing is stored before the camera ray's segment, which starts from the defaults.
            const bool firstSegment = bounce == 0;
            flags &= ~(PATH_HIT_DISTANCE_PENDING | PATH_PANE_HIT_PENDING | PATH_END_HIT_PENDING | DEFERRED_SMOOTH_SEGMENT);
            float4 medium = UnpackSegmentMedium(segmentMedium);
            float3 segmentExtinction = medium.xyz;
            // Russian roulette compares the throughput with this, the weight of the branch it is on, so a
            // pane's reflection survives as long as a path of its own would.
            float rrInvScale = medium.w;

            // Absorption along the segment just travelled, if it was inside a medium.
            bool inMedium = any(segmentExtinction > 0.0);
            if (inMedium)
            {
                if (hit.hit)
                    throughput *= exp(-segmentExtinction * hit.t);
            }
    #ifdef ATMOSPHERE
            else
            {
                // Otherwise it crossed the scene's air, which scatters sunlight into the path and dims
                // whatever lies beyond. Rays that leave the scene cross atmosphereMissDistance of it.
                float segmentLength = hit.hit ? hit.t : g_Frame.atmosphereMissDistance;
                if (bounce < g_Frame.atmosphereInScatterSegments)
                {
                    bool seen = SeenSegment(flags, fromCamera, chainActive);
                    radiance += ClampFirefly(throughput * FireflyFilter(AtmosphereInScattering(origin, dir, segmentLength, fromCamera || chainActive, rng), seen, fireflyK),
                                             seen, DirectSegment(flags, fromCamera, chainActive));
                }
                throughput *= AtmosphereTransmittance(segmentLength);
            }
    #endif

            // Segments that touch the camera state load the parts they change late, and store them again
            // before the next TraceRay, so it stays out of the registers through the shading.
            bool cameraSegment = fromCamera || chainActive || hitDistanceArrived || paneHitArrived || endHitArrived;

            if (!hit.hit)
            {
                // A miss adds nothing to a chain, which the chain's end after the loop takes care of.
                if (cameraSegment)
                {
                    CameraState camera = firstSegment ? DefaultCameraState(dir) : (CameraState)0;
                    ArriveCameraState(camera, firstSegment ? CAMERA_ALL : 0, fromCamera, false, hitDistanceArrived, paneHitArrived,
                                      endHitArrived, false, 0.0, segmentExtinction);
                    // Sky seen through a window: the guides describe the sky, as for a camera ray.
                    if (fromCamera)
                        camera.guides = DefaultGuides(dir);
#ifdef DLSS_RR
                    if (chainActive)
                        g_PsrOutcome[DispatchRaysIndex().xy] = float(uint(PSR_OUTCOME_SKY));
#endif
#ifdef PSR
                    // A chain that leaves into the sky ends on it, for RayGen to put in the guides' place.
                    if (chainActive && !inMedium && g_Frame.psrSky != 0)
                    {
                        StoreChainEnd(SkyChainEnd(dir));
                        flags &= ~PATH_CHAIN_ACTIVE;
                    }
#endif
                    StoreCameraState(camera, firstSegment ? CAMERA_ALL & ~CAMERA_CHAIN : fromCamera ? CAMERA_GUIDES : 0);
                }

                // A ray inside a medium that finds no way out ends there.
                if (!inMedium)
                {
                    // The sky is never light sampled, so BSDF sampling gets its full contribution. The sun disc is
                    // also reached by NEE, so weight it against light sampling.
                    // The sun through glass after a diffuse bounce from where the photons were read is theirs.
                    float3 Lsun = PhotonState(flags) == PHOTON_COVERED ? float3(0.0, 0.0, 0.0) : SunRadiance(dir);
                    if (any(Lsun > 0.0))
                        Lsun *= fromCamera || smoothSegment ? 1.0 : PowerHeuristic(prevBsdfPdf, LightPdf(origin, dir, hit));
                    bool seen = SeenSegment(flags, fromCamera, chainActive);
                    float3 sky = SkyRadiance(origin, dir, Lsun, fromCamera || chainActive, seen, rng);
                    radiance += ClampFirefly(throughput * FireflyFilter(sky, seen, fireflyK), seen,
                                             DirectSegment(flags, fromCamera, chainActive));
                }
                break;
            }

            // Most surfaces have only the base's lobes and shade in a specialisation without the others, in far
            // fewer registers. SER orders the hits by material, so a wave mostly takes just one of the two.
            // The emitter's MIS weight and the light sample come between the halves of the shading, outside the
            // specialisations, so their light tree walks are compiled once and a wave of both kinds walks together.
            const float emissionWeight = EmissionWeight(payload, origin, dir, fromCamera, smoothSegment, prevBsdfPdf, prevVertex);
            const bool simple = IsSimpleMaterial(PayloadMaterialIndex(payload.materialId));
            SurfaceShading shading;
            if (simple)
                shading = ShadeSurfaceBegin<true>(payload, origin, dir, bounce, fromCamera, chainActive, emissionWeight, flags, rng,
                                                  radiance, throughput, cone, fireflyK);
            else
                shading = ShadeSurfaceBegin<false>(payload, origin, dir, bounce, fromCamera, chainActive, emissionWeight, flags,
                                                   rng, radiance, throughput, cone, fireflyK);
            LightSample ls = (LightSample)0;
            if (shading.sampleLights)
                ls = SampleLight(UnpackLightSelectVertex(shading.vertex), shading.lightU.xy, rng);
            SurfaceOutcome outcome;
            if (simple)
                outcome = ShadeSurfaceEnd<true>(shading, ls, origin, dir, bounce, fromCamera, chainActive, hitDistanceArrived,
                                                paneHitArrived, endHitArrived, smoothSegment, firstSegment, cameraSegment,
                                                medium.xyz, flags, rng, radiance, throughput, prevBsdfPdf, prevVertex, cone,
                                                segmentExtinction, rrInvScale, fireflyK);
            else
                outcome = ShadeSurfaceEnd<false>(shading, ls, origin, dir, bounce, fromCamera, chainActive, hitDistanceArrived,
                                                 paneHitArrived, endHitArrived, smoothSegment, firstSegment, cameraSegment,
                                                 medium.xyz, flags, rng, radiance, throughput, prevBsdfPdf, prevVertex, cone,
                                                 segmentExtinction, rrInvScale, fireflyK);

            if (!outcome.continuePath)
                break;

            if (!outcome.passThrough)
                flags = SetFlag(flags & ~PATH_FROM_CAMERA, PATH_DIRECT_SEGMENT, fromCamera || chainActive);

            // Russian roulette, but not while a PSR chain is open: ending the path there would end the
            // chain short, so a random scatter of pixels each frame would fall back to the glass's guides.
            if (bounce >= RR_START_BOUNCE && !ChainActive(flags))
            {
                float q = saturate(MaxComponent(throughput) * rrInvScale);
                if (outcome.rrU >= q)
                    break;
                throughput /= q;
            }
            segmentMedium = PackSegmentMedium(segmentExtinction, rrInvScale);
        }

        // A refraction chain the path ended on leaves the glass's transmittance as its diffuse albedo.
        if (ChainActive(flags))
        {
#ifdef DLSS_RR
            uint2 outcomePixel = DispatchRaysIndex().xy;
            if (g_PsrOutcome[outcomePixel] != float(uint(PSR_OUTCOME_SKY)))
                g_PsrOutcome[outcomePixel] = float(uint(PSR_OUTCOME_CUT_SHORT));
#endif
            CameraState camera = LoadCameraState(CAMERA_CHAIN | CAMERA_GUIDES | CAMERA_VIEW);
            if (camera.chain.throughGlass)
            {
                camera.guides.diffuseAlbedo += camera.transmittance * camera.chain.throughput;
                StoreCameraState(camera, CAMERA_GUIDES);
            }
        }
        flags &= ~PATH_CHAIN_ACTIVE;

        if ((flags & DEFERRED_PENDING) == 0)
            break;

        DeferredBranch deferred = LoadDeferredBranch();
        origin = deferred.origin;
        dir = deferred.dir;
        throughput = deferred.throughput;
        cone.width = deferred.coneWidth;
        cone.spread = deferred.coneSpread;
        segmentMedium = SEGMENT_MEDIUM_AIR;
        // Both branches leave the surface the camera sees.
        fireflyK = 1.0;
        bounce = flags >> DEFERRED_BOUNCE_SHIFT;
        uint deferredFlags = flags & DEFERRED_MASK;
        // Glass splits only where the camera sees, so every bounce before the branch is specular.
        flags &= ~(DEFERRED_MASK | PATH_HIT_DISTANCE_PENDING | PATH_END_HIT_PENDING | PATH_IN_SEEN_OBJECT |
                   (PATH_DIFFUSE_BOUNCES_MASK << PATH_DIFFUSE_BOUNCES_SHIFT));
        // Both branches leave the glass the camera sees by a smooth lobe, so neither is the camera ray any more.
        flags = SetPhotonState(flags, PHOTON_NONE);
        if ((deferredFlags & DEFERRED_REFLECTION) != 0)
        {
            // Solid glass's reflection leaves the surface the camera saw.
            flags |= PATH_DIRECT_SEGMENT | PATH_MIRROR_SEGMENT;
            flags = SetFlag(flags, PATH_PANE_HIT_PENDING, (deferredFlags & DEFERRED_BRIGHTEST_PANE) != 0);
            flags = (flags & ~PATH_FROM_CAMERA) | DEFERRED_SMOOTH_SEGMENT;
        }
        else
        {
            prevBsdfPdf = 0.0;
            flags |= PATH_FROM_CAMERA;
        }
    }

    return radiance;
}

#ifdef DLSS_RR
float TraceGuideHitDistance(float3 position, float3 N, float3 viewDir, float roughness)
{
    N = dot(N, viewDir) > 0.0 ? -N : N;
    float3 mirror = reflect(viewDir, N);
    RayDesc ray;
    ray.Direction = normalize(lerp(N, mirror, SpecularDominantFactor(roughness)));
    ray.Origin = OffsetRay(position, N, ray.Direction);
    ray.TMin = 0.0;
    ray.TMax = FLT_MAX;

    const uint rayFlags = OpacityMicromapRayFlags(1u);
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(SceneFor(rayFlags), RayFlags(rayFlags), 0xFF, ray);
    while (query.Proceed())
    {
        PrimitiveDesc prim = LoadPrimitive(query.CandidateInstanceID() + query.CandidateGeometryIndex());
        MaterialData material = LoadGpuMaterial(prim.materialIndex);
        float2 uv = InterpolateTexCoord(LoadTriangle(prim, query.CandidatePrimitiveIndex()),
                                        Barycentrics(query.CandidateTriangleBarycentrics()));
        // A guide wants no noise of its own, so blended surfaces count from half there rather than by chance.
        float cutoff = material.alphaMode == ALPHA_MODE_BLEND ? 0.5 : AlphaCutoff(material);
        if (material.alphaMode == ALPHA_MODE_OPAQUE || LoadBaseColourAlpha(material, uv).a >= cutoff)
            query.CommitNonOpaqueTriangleHit();
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : SKY_HIT_DISTANCE;
}

// The guides of a camera ray's hit, as ShadeSurface makes them for the surface the camera sees, behind walls of
// the given transmittance; opacity is how much of it is there (LoadOpacity).
// The material of a guide ray's hit as ShadeSurface sees it, a ray cone coneWidth wide there: opaque and thin-walled
// surfaces faced to the viewer, solid glass left as it is, so its context knows the inside from the outside.
// specularAA: as ShadeSurface widens it, past PSR's mirrors and glass but not on the camera's surface or inside its object.
void GuideMaterial(inout HitInfo hit, float3 wo, float coneWidth, bool specularAA, out OpenPBRBaseParams mat,
                   out OpenPBRContext ctx)
{
    bool anisotropic = 0 < g_Frame.anisotropicSegments;
    mat = LoadMaterial(hit, coneWidth * g_Frame.textureLodScale, anisotropic, false,
                       specularAA ? FootprintSpecularAA(hit, wo, coneWidth) : NoSpecularAA());

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
    ctx = OpenPBRSetup(mat, wo);
}

PrimaryGuides SurfaceGuides(HitPayload payload, float3 origin, float3 dir, float3 transmittance, out float opacity)
{
    HitInfo hit = UnpackHit(payload, origin, dir);
    float3 wo = -dir;
    float coneWidth = g_Frame.rayConePixelSpread * hit.t;
    opacity = LoadOpacity(hit, coneWidth * g_Frame.textureLodScale, 0 < g_Frame.anisotropicSegments);
    OpenPBRBaseParams mat;
    OpenPBRContext ctx;
    GuideMaterial(hit, wo, coneWidth, false, mat, ctx);

    PrimaryGuides g = DefaultGuides(dir);
    g.position = float4(hit.position, 1.0);
    g.normal = mat.geometryNormal;
    g.roughness = GuideRoughness(mat, ctx);
    g.specularHitDistance = InitialSpecularHitDistance(g.roughness);
    g.curvature = ViewedCurvature(hit, wo);
    g.displacement = hit.motion;
    ReflectanceGuides(mat, ctx, wo, g);
    g.diffuseAlbedo *= transmittance;
    g.specularAlbedo *= transmittance;
    return g;
}

// How far either side of the guides' surface a coplanar layer (a decal on its wall) is looked for, relative to
// the distance from the camera.
static const float BLEND_GUIDE_SLACK = 1.0e-3;

// The guides of a camera ray that met a blended surface, deterministic rather than of whichever layer the path
// took by chance: its albedos the layer's and those of what is behind it mixed by the layer's opacity, the rest
// the layer's if it is at least half there, else what is behind it's. One guide ray finds the layer the path did
// not take: behind a blended surface it took, the first solid surface; in front of a surface it reached, the
// nearest blended one. Only the first blended layer counts. Guides a PSR chain replaced are left alone.
// s needs CAMERA_GUIDES, CAMERA_VIEW and CAMERA_DIRECTION.
void ResolveBlendGuides(inout CameraState s, float3 origin)
{
    if (s.guides.replaced)
        return;
    float3 dir = s.cameraDir;
    bool sky = s.guides.position.w == 0.0;
    PrimaryGuides front, behind;
    float opacity;
    HitPayload payload;
    if (s.guides.blendOpacity < 1.0)
    {
        front = s.guides;
        opacity = s.guides.blendOpacity;
        float unused;
        behind = TraceGuideSurface(origin, dir, s.distance * (1.0 - BLEND_GUIDE_SLACK), FLT_MAX, uint(INSTANCE_MASK_SOLID), payload)
                 ? SurfaceGuides(payload, origin, dir, s.transmittance, unused)
                 : DefaultGuides(dir);
    }
    else
    {
        float tMax = sky ? FLT_MAX : s.distance * (1.0 + BLEND_GUIDE_SLACK);
        if (!TraceGuideSurface(origin, dir, 0.0, tMax, uint(INSTANCE_MASK_BLENDED), payload))
            return;
        front = SurfaceGuides(payload, origin, dir, s.transmittance, opacity);
        behind = s.guides;
    }

    PrimaryGuides g = opacity >= 0.5 ? front : behind;
    g.diffuseAlbedo = lerp(behind.diffuseAlbedo, front.diffuseAlbedo, opacity);
    g.specularAlbedo = lerp(behind.specularAlbedo, front.specularAlbedo, opacity);
    g.blendOpacity = 1.0;
    s.guides = g;
}

// FrameConstants::guideSamples is at most this.
static const uint GUIDE_SAMPLE_MAX = 8;
// The extra guide samples' offsets from the jittered sample, in pixels: Halton (2, 3) less 1/2, so any first few
// spread over a pixel-wide box around it.
static const float2 GUIDE_SAMPLE_OFFSETS[GUIDE_SAMPLE_MAX] = {
    float2(0.0, -0.1666667), float2(-0.25, 0.1666667), float2(0.25, -0.3888889), float2(-0.375, -0.0555556),
    float2(0.125, 0.2777778), float2(-0.125, -0.2777778), float2(0.375, 0.0555556), float2(-0.4375, 0.3888889)
};

void SupersampleGuides(inout PrimaryGuides guides, uint2 pixel)
{
    uint count = min(g_Frame.guideSamples, GUIDE_SAMPLE_MAX);
    float3 diffuse = guides.diffuseAlbedo;
    float3 specular = guides.specularAlbedo;
    float3 normal = guides.normal;
    float roughness = guides.roughness;
    [loop]
    for (uint i = 0; i < count; ++i)
    {
        float3 origin, dir;
        GenerateCameraRay(pixel, 0.5 + g_Frame.jitter + GUIDE_SAMPLE_OFFSETS[i], origin, dir);
        HitPayload payload;
        PrimaryGuides g;
        if (TraceGuideSurface(origin, dir, 0.0, FLT_MAX, uint(INSTANCE_MASK_SOLID), payload))
        {
            float unused;
            g = SurfaceGuides(payload, origin, dir, float3(1.0, 1.0, 1.0), unused);
        }
        else
        {
            g = DefaultGuides(dir);
        }
        diffuse += g.diffuseAlbedo;
        specular += g.specularAlbedo;
        normal += g.normal;
        roughness += g.roughness;
    }
    float weight = 1.0 / float(count + 1);
    guides.diffuseAlbedo = diffuse * weight;
    guides.specularAlbedo = specular * weight;
    guides.roughness = roughness * weight;
    // The sky's normal is zero, so a mix with it stays a direction only while some surface is in it.
    float length2 = dot(normal, normal);
    guides.normal = length2 > 1.0e-6 ? normal * rsqrt(length2) : float3(0.0, 0.0, 0.0);
}

#ifdef PSR
bool WalkPsrEvent(OpenPBRBaseParams mat, OpenPBRContext ctx, out bool reflection, out float fraction)
{
    fraction = 1.0;
    uint lobe = PsrStrongerLobe(mat, ctx);
    if (lobe == LOBE_NONE)
    {
        reflection = true;
        return IsPsrEvent(mat, ctx, LOBE_SPECULAR_REFLECTION, fraction);
    }
    reflection = lobe == LOBE_SPECULAR_REFLECTION;
    if (!ctx.inside)
    {
        float2 fractions = SpecularFractions(mat);
        fraction = reflection ? fractions.x : fractions.y;
    }
    return true;
}

// The chain's walk is at most this many events past the camera's surface, or the specular bounce budget if less.
static const uint PSR_WALK_MAX_EVENTS = 8;

PrimaryGuides WalkPsrGuideSample(float3 origin, float3 dir)
{
    HitPayload payload;
    if (!TraceGuideSurface(origin, dir, 0.0, FLT_MAX, uint(INSTANCE_MASK_SOLID), payload))
        return DefaultGuides(dir);
    HitInfo hit = UnpackHit(payload, origin, dir);
    float travelled = hit.t;
    OpenPBRBaseParams mat;
    OpenPBRContext ctx;
    GuideMaterial(hit, -dir, g_Frame.rayConePixelSpread * travelled, false, mat, ctx);

    // The camera's surface, whose own guides these are unless a chain starts here.
    PrimaryGuides own = DefaultGuides(dir);
    own.position = float4(hit.position, 1.0);
    own.normal = mat.geometryNormal;
    own.roughness = GuideRoughness(mat, ctx);
    ReflectanceGuides(mat, ctx, -dir, own);
    bool reflection;
    float primaryFraction;
    if (!WalkPsrEvent(mat, ctx, reflection, primaryFraction))
        return own;

    SpecularChain chain = (SpecularChain)0;
    chain.rotor = float4(0.0, 0.0, 0.0, 1.0);
    chain.parityOdd = false;
    bool throughGlass = !reflection && !mat.geometryThinWalled;
    float3 throughput = PsrEventThroughput(mat, ctx, -dir, reflection);
    float3 extinction = float3(0.0, 0.0, 0.0);
    uint budget = min(g_Frame.maxSpecularBounces, PSR_WALK_MAX_EVENTS);
    PrimaryGuides g = own;
    bool inSeenObject = true;
    [loop]
    for (uint event = 0;; ++event)
    {
        // The event's turn about the shading normal faced to the incoming ray, as the path's smooth lobes turn it;
        // its chain transform about the geometry normal, as AddPsrEvent's.
        float3 N = ToWorld(mat, float3(0.0, 0.0, 1.0));
        float3 n = dot(N, dir) < 0.0 ? N : -N;
        float3 dirOut = reflection ? reflect(dir, n) : refract(dir, n, 1.0 / ctx.eta);
        if (dot(dirOut, dirOut) == 0.0)
        {
            reflection = true;
            dirOut = reflect(dir, n);
        }
        dirOut = normalize(dirOut);
        if (reflection)
        {
            chain.rotor = QuaternionMultiply(chain.rotor, MirrorRotor(mat.geometryNormal, chain.parityOdd));
            chain.parityOdd = !chain.parityOdd;
        }
        else
        {
            float4 q = ArcRotor(dirOut, dir);
            chain.rotor = QuaternionMultiply(chain.rotor, chain.parityOdd ? float4(-q.xy, q.zw) : q);
            // Into solid glass its medium, out of it the air.
            if (!mat.geometryThinWalled)
                extinction = ctx.inside ? float3(0.0, 0.0, 0.0) : MediumExtinction(mat);
        }

        // Past the budget the path's chain is cut short: through glass, the glass's transmittance as its content.
        if (event >= budget)
        {
            if (throughGlass)
                g.diffuseAlbedo += throughput;
            return g;
        }

        // Still inside the object the camera sees (PATH_IN_SEEN_OBJECT).
        inSeenObject = inSeenObject && !mat.geometryThinWalled && dot(dirOut, hit.Ng) < 0.0;
        origin = OffsetRay(hit.position, hit.Ng, dirOut);
        dir = dirOut;
        if (!TraceGuideSurface(origin, dir, 0.0, FLT_MAX, uint(INSTANCE_MASK_SOLID), payload))
        {
            // Into the sky, which takes the guides' place (ReplacePrimarySurfaceWithSky), or without psrSky the
            // content of a chain through glass.
            if (g_Frame.psrSky == 0)
            {
                if (throughGlass)
                    g.diffuseAlbedo += throughput;
                return g;
            }
            PrimaryGuides sky = DefaultGuides(dir);
            g.normal = float3(0.0, 0.0, 0.0);
            g.roughness = 0.0;
            g.diffuseAlbedo += throughput * sky.diffuseAlbedo;
            g.specularAlbedo = lerp(g.specularAlbedo, throughput * sky.specularAlbedo, primaryFraction);
            return g;
        }
        hit = UnpackHit(payload, origin, dir);
        travelled += hit.t;
        throughput *= exp(-extinction * hit.t);
        GuideMaterial(hit, -dir, g_Frame.rayConePixelSpread * travelled, !inSeenObject, mat, ctx);
        float unused;
        if (!WalkPsrEvent(mat, ctx, reflection, unused))
        {
            // The chain's end. Inside glass the path's chain is refused, which leaves the glass's content.
            if (ctx.inside)
            {
                if (throughGlass)
                    g.diffuseAlbedo += throughput;
                return g;
            }
            PrimaryGuides seen;
            ReflectanceGuides(mat, ctx, -dir, seen);
            g.normal = normalize(ApplySpecularChain(chain, mat.geometryNormal));
            g.roughness = GuideRoughness(mat, ctx);
            g.diffuseAlbedo += throughput * seen.diffuseAlbedo;
            g.specularAlbedo = lerp(g.specularAlbedo, throughput * seen.specularAlbedo, primaryFraction);
            return g;
        }
        throughput *= PsrEventThroughput(mat, ctx, -dir, reflection);
    }
    return g;
}

// Fixed points of the pixel, from its centre, that PsrDeterministicGuides walks chains from: the centre, then
// Halton (2, 3) less 1/2.
static const float2 PSR_GUIDE_FIXED_OFFSETS[GUIDE_SAMPLE_MAX] = {
    float2(0.0, 0.0), float2(0.0, -0.1666667), float2(-0.25, 0.1666667), float2(0.25, -0.3888889),
    float2(-0.375, -0.0555556), float2(0.125, 0.2777778), float2(-0.125, -0.2777778), float2(0.375, 0.0555556)
};

void PsrDeterministicGuides(inout PrimaryGuides guides, uint2 pixel)
{
    uint count = min(g_Frame.psrGuideSamples, GUIDE_SAMPLE_MAX);
    float3 diffuse = float3(0.0, 0.0, 0.0);
    float3 specular = float3(0.0, 0.0, 0.0);
    float3 normal = float3(0.0, 0.0, 0.0);
    float roughness = 0.0;
    [loop]
    for (uint i = 0; i < count; ++i)
    {
        float3 origin, dir;
        GenerateCameraRay(pixel, 0.5 + PSR_GUIDE_FIXED_OFFSETS[i], origin, dir);
        PrimaryGuides g = WalkPsrGuideSample(origin, dir);
        diffuse += g.diffuseAlbedo;
        specular += g.specularAlbedo;
        normal += g.normal;
        roughness += g.roughness;
    }
    float weight = 1.0 / float(count);
    guides.diffuseAlbedo = diffuse * weight;
    guides.specularAlbedo = specular * weight;
    if (guides.replaced)
    {
        guides.roughness = roughness * weight;
        float length2 = dot(normal, normal);
        guides.normal = length2 > 1.0e-6 ? normal * rsqrt(length2) : float3(0.0, 0.0, 0.0);
    }
}
#endif
#endif

// Entry points

// Caustic photons from the sun's photon map into the photon records (PathTracer/Photons.slang), one per thread, rows of
// PHOTON_DISPATCH_WIDTH. Neighbouring indices start next to each other (EmitSunPhoton), so a wave's rays are coherent.
[shader("raygeneration")]
void PhotonRayGen()
{
#ifdef OPACITY_MICROMAPS
    AllowOpacityMicromapsInRayQueries();
#endif
    uint index = DispatchRaysIndex().y * DispatchRaysDimensions().x + DispatchRaysIndex().x;
    if (index >= g_Frame.photons.photonCount)
        return;
    uint rng = RngInit(g_Frame.frameIndex);
    PhotonStart start;
    if (EmitSunPhoton(index, rng, start))
        TracePhoton(start, rng);
}

[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
#ifdef OPACITY_MICROMAPS
    // The shadow and subsurface ray queries meet the alpha-tested geometry's micromaps.
    AllowOpacityMicromapsInRayQueries();
#endif

#ifdef REFERENCE
    // The paths add to the sum of those before them, each numbered by its sample so that it has its own random
    // sequence, and the output is the mean so far.
    float3 sum = g_Frame.referenceSampleCount > 0 ? g_Accumulation[pixel].rgb : float3(0.0, 0.0, 0.0);
    [loop]
    for (uint s = 0; s < g_Frame.samplesPerFrame; ++s)
    {
        uint rng = RngInit(g_Frame.referenceSampleCount + s);
        float3 origin, dir;
        GenerateCameraRay(pixel, Rand3(rng).xy, origin, dir);
        float3 L = TracePath(origin, dir, rng);
        // A bad sample would otherwise stay in the sum for good.
        if (all(isfinite(L)))
            sum += L;
    }
    g_Accumulation[pixel] = float4(sum, 1.0);
    g_Output[pixel] = float4(sum / float(g_Frame.referenceSampleCount + g_Frame.samplesPerFrame), 1.0);
#else
#ifdef DLSS_RR
    // Before the paths, any of which may mark it.
    g_ShadowChanges[pixel] = 0.0;
    g_PsrOutcome[pixel] = float(uint(PSR_OUTCOME_NONE));
#endif
    // No photon receiver until a path's camera ray meets one.
    if (g_Frame.photons.enabled != 0)
        ClearPhotonReceiver();
    // The mean of the frame's paths, each with its own random sequence. Under DLSS-RR they all start along the
    // same camera ray, and the guides are the last one's.
    float3 origin, dir;
    float3 sum = float3(0.0, 0.0, 0.0);
    [loop]
    for (uint s = 0; s < g_Frame.samplesPerFrame; ++s)
    {
        uint rng = RngInit(g_Frame.frameIndex * g_Frame.samplesPerFrame + s);
#ifdef DLSS_RR
        // Every pixel shares this frame's jitter, which DLSS is told about separately.
        float2 subpixel = 0.5 + g_Frame.jitter;
#else
        float2 subpixel = Rand3(rng).xy;
#endif
        GenerateCameraRay(pixel, subpixel, origin, dir);
        float3 L = TracePath(origin, dir, rng);

        // A single bad sample would otherwise stick in any temporal filter downstream.
        if (all(isfinite(L)))
            sum += L;
    }

    g_Output[pixel] = float4(sum / float(g_Frame.samplesPerFrame), 1.0);

#ifdef DLSS_RR
    CameraState camera = LoadCameraState(CAMERA_GUIDES | CAMERA_PANES | CAMERA_VIEW | CAMERA_DIRECTION);
    // A PSR chain that broke by chance keeps last frame's guides, so that a random few pixels of a glass
    // object do not flip to its own surface each frame, which DLSS-RR takes for disocclusion in motion.
    if (camera.guides.hold)
    {
        g_PsrOutcome[pixel] = float(uint(PSR_OUTCOME_HELD));
        return;
    }
    // Whether the path followed a PSR chain: one that ended (on a surface or the sky), or one the budget cut short.
    bool followedChain = camera.guides.chainEnd || g_PsrOutcome[pixel] == float(uint(PSR_OUTCOME_CUT_SHORT));
#ifdef PSR
    if (camera.guides.chainEnd)
        FinishChainEnd(camera);
#endif
    if (g_Frame.blendGuides != 0)
        ResolveBlendGuides(camera, origin);
    PrimaryGuides guides = FinishGuides(camera, origin);
#ifdef PSR
    // A chain's guides from chains walked from fixed points of the pixel.
    if (followedChain && g_Frame.psrGuideSamples != 0 && !guides.reflectorIsPane)
        PsrDeterministicGuides(guides, pixel);
    else
#endif
    // Only guides of the surface the camera sees in the open: not PSR's virtual ones, nor through or off glass.
    if (g_Frame.guideSamples != 0 && !guides.replaced && !guides.reflectorIsPane && all(camera.transmittance == 1.0))
        SupersampleGuides(guides, pixel);

    // Last frame's specular hit distance, loaded first so its latency hides behind the guides below.
    float heldHitDistance = g_SpecularHitDistance[pixel];

    // Unless the guides have a motion vector of their own (a curved PSR chain), the primary point was last frame
    // where its displacement says, which is where it is but on a moving object. CalculateMotionVector works in
    // NDC, whose +y is up in D3D, while DLSS wants +y down. The sky (w = 0) has no displacement.
    float4 previousPosition = guides.position + float4(guides.displacement, 0.0) * guides.position.w;
    float2 motion = guides.hasMotion
                    ? guides.motion
                    : CalculateMotionVector(guides.position, previousPosition, g_Frame.prevViewProj, g_Frame.viewProj, float2(dims));
    motion.y = -motion.y;

    // Reversed Z: the sky sits on the far plane at 0.
    float4 clip = mul(guides.position, g_Frame.viewProj);
    float depth = guides.position.w > 0.0 ? clip.z / clip.w : 0.0;

    g_DiffuseAlbedo[pixel] = float4(guides.diffuseAlbedo, 1.0);
    g_SpecularAlbedo[pixel] = float4(guides.specularAlbedo, 1.0);
    g_NormalRoughness[pixel] = float4(guides.normal, guides.roughness);
    g_MotionVectors[pixel] = motion;
    g_Depth[pixel] = depth;

    // A smooth reflector's reflection is told to keep less history, most at a mirror. Not PSR's virtual surfaces,
    // whose image is the reflection itself, nor a pane's reflection, nor the sky.
    float responsivity = 0.0;
    if (g_Frame.reflectionResponsivity != 0.0 && guides.position.w > 0.0 && !guides.replaced && !guides.reflectorIsPane)
        responsivity = g_Frame.reflectionResponsivity *
                       saturate(1.0 - guides.reflectorRoughness / g_Frame.reflectionResponsivityMaxRoughness);
    g_Responsivity[pixel] = responsivity;

    // A rough reflector's reflection is a blur of things at many depths that one hit distance cannot place: its
    // specular motion vectors are the surface's own, so the field stays smooth across a highlight (Bevy Solari).
    // A pane's reflection is never rough, and otherwise the reflector is the guides' surface, moving with motion.
    // The sky a chain put in the guides' place has no reflection of its own, so it moves as the sky does.
    if (guides.reflectorRoughness > g_Frame.specularMotionMaxRoughness || (guides.replaced && guides.position.w == 0.0))
    {
        g_SpecularMotionVectors[pixel] = motion;
        return;
    }

    // The guides' own surface traces its hit distance. A virtual one (PSR) and a pane's reflection take the
    // path's, where negative keeps last frame's, which is whatever the texture was created with until a frame
    // writes it.
    float hitDistance = guides.specularHitDistance;
    if (g_Frame.specularHitDistanceRay != 0 && guides.position.w > 0.0 && !guides.replaced && !guides.reflectorIsPane)
        hitDistance = TraceGuideHitDistance(guides.position.xyz, guides.normal, camera.cameraDir, guides.roughness);
    if (hitDistance >= 0.0)
        g_SpecularHitDistance[pixel] = hitDistance;
    else
        hitDistance = heldHitDistance;
    if (!isfinite(hitDistance))
        hitDistance = 0.0;

    // The reflected point is static, so its image moves with the camera, and with a moving reflector as far as
    // the image is the reflector's own surface: all of it where it sits on the surface, less the sharper the
    // lobe, as a mirror shows the world behind it, which the reflector's move barely shifts. A pane does not move.
    float4 reflection = SpecularVirtualPosition(guides, camera.cameraDir, hitDistance);
    bool onReflector = all(reflection == guides.reflector);
    float carried = guides.reflectorIsPane ? 0.0 : onReflector ? 1.0 : 1.0 - SpecularDominantFactor(guides.reflectorRoughness);
    float4 previousReflection = reflection + float4(guides.displacement, 0.0) * (reflection.w * carried);
    float2 specularMotion = CalculateMotionVector(reflection, previousReflection, g_Frame.prevViewProj, g_Frame.viewProj, float2(dims));
    specularMotion.y = -specularMotion.y;
    g_SpecularMotionVectors[pixel] = specularMotion;
#endif
#endif
}

[shader("miss")]
void Miss(inout HitPayload payload)
{
    payload.hitT = -1.0;
}


[shader("closesthit")]
void ClosestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID() + GeometryIndex());
    uint3 tri = LoadTriangle(prim, PrimitiveIndex());
    float3 bary = Barycentrics(attr.barycentrics);

    GpuVertex v0 = g_Vertices[tri.x];
    GpuVertex v1 = g_Vertices[tri.y];
    GpuVertex v2 = g_Vertices[tri.z];
    float3 p0 = v0.position;
    float3 p1 = v1.position;
    float3 p2 = v2.position;
    float3 objectNormal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    float4 objectTangent = v0.tangent * bary.x + v1.tangent * bary.y + v2.tangent * bary.z;

    float3x3 linearToWorld = (float3x3)ObjectToWorld3x4();
    float3x3 cofactor = float3x3(cross(linearToWorld[1], linearToWorld[2]),
                                 cross(linearToWorld[2], linearToWorld[0]),
                                 cross(linearToWorld[0], linearToWorld[1]));
    float det = dot(linearToWorld[0], cofactor[0]);
    float handedness = det < 0.0 ? -1.0 : 1.0;

    // cross(M a, M b) = cofactor(M) cross(a, b), so this is the transformed object-space face normal.
    float3 e1 = mul(linearToWorld, p1 - p0);
    float3 e2 = mul(linearToWorld, p2 - p0);
    float3 faceCross = cross(e1, e2);
    float3 Ng = normalize(faceCross * handedness);

    payload.hitT = RayTCurrent();
    payload.Ng = OctEncode(Ng);
    payload.N = OctEncode(mul(cofactor, objectNormal) * handedness);
    payload.T = OctEncode(mul(linearToWorld, objectTangent.xyz));
    // A mirroring transform flips cross(N, T), so fold its handedness into the bitangent sign.
    bool tangentNegative = (objectTangent.w < 0.0) != (handedness < 0.0);
    // The smooth surface's height above the triangle on the side the ray came from.
    float3 viewNg = dot(Ng, WorldRayDirection()) < 0.0 ? Ng : -Ng;
    float3 N0 = mul(cofactor, v0.normal) * handedness;
    float3 N1 = mul(cofactor, v1.normal) * handedness;
    float3 N2 = mul(cofactor, v2.normal) * handedness;
    float lift = SmoothSurfaceLift(bary, e1, e2, N0 * rsqrt(max(dot(N0, N0), 1.0e-30)),
                                   N1 * rsqrt(max(dot(N1, N1), 1.0e-30)), N2 * rsqrt(max(dot(N2, N2), 1.0e-30)),
                                   mul(cofactor, objectNormal) * handedness, viewNg);
    payload.materialId = prim.materialIndex | ((f32tof16(lift) & 0x7FFFu) << PAYLOAD_LIFT_SHIFT) |
                         (tangentNegative ? TANGENT_SIGN_BIT : 0u);
    float2 uv0 = v0.texCoord;
    float2 uv1 = v1.texCoord;
    float2 uv2 = v2.texCoord;
    payload.uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    float4 uvGradients = RayConeUvGradients(e1, e2, faceCross, uv0, uv1, uv2, WorldRayDirection());
    payload.uvGradients = uint2(PackHalf2(uvGradients.xy), PackHalf2(uvGradients.zw));

    uint emitterTable = g_Emitters[InstanceIndex()];
    payload.emitter = emitterTable != INVALID_LIGHT_INDEX ? emitterTable + PrimitiveIndex() : INVALID_LIGHT_INDEX;

    // How the shading normal turns over the triangle, telling flat mirrors from curved ones for PSR and
    // carrying ray differentials through them. Only the guides and the photons' ray cones use it, so other hits
    // skip it.
    // N = n / |n| for n = cofactor (n0 + b1 (n1 - n0) + b2 (n2 - n0)) at P = p0 + b1 e1 + b2 e2, so
    // dN = (I - N N^T) / |n| (w1 db1 + w2 db2), w the cofactor's normal differences and db_i = g_i . dP.
    if (payload.shape.x == 0)
        return;
    float3 n = mul(cofactor, objectNormal) * handedness;
    float invLength = rsqrt(max(dot(n, n), 1.0e-30));
    float3 N = n * invLength;
    float3 w1 = mul(cofactor, v1.normal - v0.normal) * handedness;
    float3 w2 = mul(cofactor, v2.normal - v0.normal) * handedness;
    w1 = (w1 - N * dot(N, w1)) * invLength;
    w2 = (w2 - N * dot(N, w2)) * invLength;
    float invArea2 = 1.0 / max(dot(faceCross, faceCross), 1.0e-30);
    float3 g1 = cross(e2, faceCross) * invArea2;
    float3 g2 = cross(faceCross, e1) * invArea2;
    // In the frame UnpackHit gives NormalDerivative.
    float3 T = mul(linearToWorld, objectTangent.xyz);
    T = normalize(T - N * dot(N, T));
    float3 B = cross(N, T);
    float3 dNdT = w1 * dot(g1, T) + w2 * dot(g2, T);
    float3 dNdB = w1 * dot(g1, B) + w2 * dot(g2, B);
    float4 shape = clamp(float4(dot(T, dNdT), dot(T, dNdB), dot(B, dNdT), dot(B, dNdB)), -HALF_MAX, HALF_MAX);
    payload.shape = uint2(PackHalf2(shape.xy), PackHalf2(shape.zw));

#ifdef DLSS_RR

    // The same point of the object under last frame's transform. Both transforms take the same steps, so a
    // static instance's motion is zero.
    float4 objectPosition = float4(p0 + (p1 - p0) * bary.y + (p2 - p0) * bary.z, 1.0);
    uint row = 3 * InstanceIndex();
    float3x4 previousToWorld = float3x4(g_PreviousTransforms[row], g_PreviousTransforms[row + 1], g_PreviousTransforms[row + 2]);
    float3 motion = clamp(mul(previousToWorld, objectPosition) - mul(ObjectToWorld3x4(), objectPosition), -HALF_MAX, HALF_MAX);
    payload.motion = uint2(PackHalf2(motion.xy), PackHalf2(float2(motion.z, 0.0)));
#endif
}

// Alpha for camera and bounce rays: masked surfaces are cut at alphaCutoff, blended ones are there by chance
// (CandidateThere), so the closest hit is already one the path takes and ShadeSurface never sees opacity.
[shader("anyhit")]
void AnyHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID() + GeometryIndex());
    MaterialData material = LoadGpuMaterial(prim.materialIndex);
    if (material.alphaMode == ALPHA_MODE_OPAQUE)
        return;
    float2 uv = InterpolateTexCoord(LoadTriangle(prim, PrimitiveIndex()), Barycentrics(attr.barycentrics));
    // A guide ray takes every blended surface that is there at all (u = 0).
    bool byChance = material.alphaMode == ALPHA_MODE_BLEND && (payload.shape.y & PAYLOAD_DETERMINISTIC_ALPHA) == 0;
    float u = byChance ? CandidateRand(WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), PrimitiveIndex()) : 0.0;
    if (!CandidateThere(material, LoadBaseColourAlpha(material, uv).a, u))
        IgnoreHit();
}

// The shadow hit group, which inline shadow rays leave unused (nor would DXC accept a payload no
// TraceRay carries, since Slang only annotates traced ones as [raypayload]).
#ifndef INLINE_SHADOWS
[shader("miss")]
void ShadowMiss(inout ShadowRayPayload payload)
{
}

[shader("closesthit")]
void ShadowClosestHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.visibility = float3(0.0, 0.0, 0.0);
}

// Shadow rays stop at the first surface that blocks them, multiplying in what the pass-through
// surfaces before it let through.
[shader("anyhit")]
void ShadowAnyHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    float3 transmitted;
    if (!ShadowCandidateTransmittance(LoadPrimitive(InstanceID() + GeometryIndex()), PrimitiveIndex(), attr.barycentrics,
                                      (float3x3)ObjectToWorld3x4(), WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), transmitted))
        IgnoreHit();
    payload.visibility *= transmitted;
    if (dot(payload.visibility, 0.333) > 0.001)
        IgnoreHit();
    AcceptHitAndEndSearch();
}
#endif
