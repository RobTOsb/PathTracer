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

static const uint RR_START_BOUNCE = 1;
// Longest path through a medium that absorption is evaluated over
static const float MAX_MEDIUM_DISTANCE = 1.0e30;

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
static const uint PATH_SPLITS_SHIFT = 16; // glass splits so far, 8 bits
static const uint PATH_SPLITS_MASK = 0xFF;

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


float3 ClampIndirect(float3 contribution, bool direct)
{
    float m = MaxComponent(contribution);
    return direct || m <= g_Frame.indirectClamp ? contribution : contribution * (g_Frame.indirectClamp / m);
}

bool LeavesLightsToBsdf(OpenPBRContext ctx)
{
    return !SunIsDelta() && ctx.probDiffuse <= 0.0 && ctx.probFuzz <= 0.0 && max(ctx.alpha.x, ctx.alpha.y) <= 1.0e-4 &&
           (ctx.probCoat <= 0.0 || ctx.coatAlpha <= 1.0e-4);
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
                       bool paneHitPending, bool hit, float t, float3 extinction)
{
    float hitDistance = hit ? t : SKY_HIT_DISTANCE;
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

// The hit's shading: emission, NEE, the BSDF sample (and the glass splits), the DLSS-RR camera state and
// subsurface scattering, for the segment that ended at the payload's hit. SIMPLE: a GPU_MATERIAL_SIMPLE
// material, whose specialisation drops every lobe but the base's and so holds far fewer registers.
// arrivedExtinction is the medium the segment crossed, segmentExtinction comes back as the next one's.
SurfaceOutcome ShadeSurface<let SIMPLE : bool>(HitPayload payload, inout float3 origin, inout float3 dir, uint bounce,
                                               uint rayFlags, bool fromCamera, bool chainActive, bool hitDistanceArrived,
                                               bool paneHitArrived, bool smoothSegment, bool firstSegment, bool cameraSegment,
                                               float3 arrivedExtinction, inout uint flags, inout uint rng, inout float3 radiance,
                                               inout float3 throughput, inout float prevBsdfPdf,
                                               inout PackedLightSelectVertex prevVertex, inout RayCone cone,
                                               inout float3 segmentExtinction, inout float rrInvScale)
{
    // Front-face emission found by BSDF sampling is MIS weighted against NEE at the previous vertex
    // unless the triangle has no light (degenerate or black) or light buffer NEE is off. The weight's
    // light tree walk needs only the packed hit, so it comes before the material, which is then not
    // held across it.
    float3 wo = -dir;
    bool frontFace = dot(OctDecode(payload.Ng), wo) > 0.0;
    // Light found here is direct for a surface the camera sees, or for the one the segment left.
    bool visible = fromCamera || chainActive;
    bool directArrival = visible || (flags & PATH_DIRECT_SEGMENT) != 0;
    float emissionWeight = 1.0;
    if (frontFace && !fromCamera && !smoothSegment && g_Frame.lightCount > 0 && payload.emitter != INVALID_LIGHT_INDEX)
    {
        uint lightIndex = g_Emitters[payload.emitter];
        if (lightIndex != INVALID_LIGHT_INDEX)
            emissionWeight = PowerHeuristic(prevBsdfPdf, EmitterLightPdf(UnpackLightSelectVertex(prevVertex), lightIndex, dir,
                                                                         origin + dir * payload.hitT));
    }

    HitInfo hit = UnpackHit(payload, origin, dir);

    // Material and shading basis for this hit, its textures filtered over the cone's footprint.
    cone.width += cone.spread * hit.t;
    // Specular antialiasing where the camera sees the surface: how far the vertex normals turn across
    // the pixel's footprint, stretched by 1 / cos along the slope, and the normal maps' own spread.
    float2 specularAA = float2(0.0, 0.0);
    if (visible)
    {
        float turn = HitCurvature(hit) * cone.width;
        float cosView = dot(hit.N, wo);
        float stretch = 1.0 + 1.0 / max(cosView * cosView, 0.01);
        specularAA = float2(g_Frame.specularAAVariance * turn * turn * stretch, g_Frame.specularAANormalMapScale);
    }
    OpenPBRBaseParams mat = LoadMaterial(hit, cone.width * g_Frame.textureLodScale, bounce < g_Frame.anisotropicSegments,
                                         SIMPLE, specularAA);
    if (frontFace && any(hit.emission > 0.0))
        radiance += ClampIndirect(throughput * hit.emission * emissionWeight, directArrival);

    // Opaque and thin-walled surfaces are treated as two-sided: face the frame towards the viewer
    // so a back face is not mistaken for the inside of a transmissive object
    bool opaque = mat.transmissionWeight <= 0.0 && !HasDiffuseTransmission(mat);
    if ((opaque || mat.geometryThinWalled) && dot(hit.Ng, wo) < 0.0)
    {
        hit.Ng = -hit.Ng;
        hit.N = -hit.N;
        hit.Ns = -hit.Ns;
        FlipShape(hit);
    }

    SetShadingBasis(mat, hit.T, hit.B, hit.N, hit.Ns, wo);
    OpenPBRContext ctx = OpenPBRSetup(mat, wo);

    // One hash covers the light sample (xy) and the Russian roulette decision (z).
    float3 lightU = Rand3(rng);

    // A segment whose sample went past its kind's bounce budget only picks up the light it found, the
    // MIS partner of the previous vertex's NEE.
    uint diffuseBounces = (flags >> PATH_DIFFUSE_BOUNCES_SHIFT) & PATH_DIFFUSE_BOUNCES_MASK;
    uint specularBounces = bounce - diffuseBounces;
    bool emissionOnly = diffuseBounces > g_Frame.maxDiffuseBounces || specularBounces > g_Frame.maxSpecularBounces;
    // With both budgets spent a BSDF sample could only find light, so the last segment takes none and
    // samples the lights whatever the surface.
    bool lastSegment = emissionOnly ||
                       (diffuseBounces >= g_Frame.maxDiffuseBounces && specularBounces >= g_Frame.maxSpecularBounces);
    bool sampleLights = !emissionOnly && (lastSegment || !LeavesLightsToBsdf(ctx));
    // Sampled with the rounding prevVertex keeps, so both ends of an emitter's MIS see the same pmf. A
    // vertex that samples no lights needs none, since the segment it starts takes no MIS.
    PackedLightSelectVertex vertex = (PackedLightSelectVertex)0;
    if (sampleLights)
        vertex = PackLightSelectVertex(
            MakeLightSelectVertex(hit.position, mat, ctx, wo, opaque || HasDeltaTransmission(mat), bounce));
    // The light tree walk needs registers of its own, so across it the material is held packed and
    // the BSDF context not at all. Both are rebuilt after it, and everything from here on, sampling
    // and evaluation alike, uses the rebuilt pair.
    PackedMaterial packedMat = PackMaterial(mat);
    LightSample ls = (LightSample)0;
    if (sampleLights)
        ls = SampleLight(UnpackLightSelectVertex(vertex), lightU.xy, rng);
    mat = UnpackMaterial(packedMat, SIMPLE);
    ctx = OpenPBRSetup(mat, wo);
    float3 extinction = MediumExtinction(mat);
    bool deltaTransmission = HasDeltaTransmission(mat);
    float3 neeContribution = float3(0.0, 0.0, 0.0);
    bool traceShadowRay = false;
    if (ls.pdf > 0.0 && (!opaque || dot(ls.direction, hit.Ng) > 0.0))
    {
        float bsdfPdf;
        float3 fCos = OpenPBREvalPdf(mat, ctx, ls.direction, bsdfPdf);
        if (any(fCos > 0.0))
        {
            // The last segment takes no BSDF sample, which leaves NEE as the only strategy there.
            float w = ls.lightSamplingOnly || lastSegment ? 1.0 : PowerHeuristic(ls.pdf, bsdfPdf);
            bool intoMedium = dot(ls.direction, hit.Ng) < 0.0 && !mat.geometryThinWalled;
            // The sun is FLT_MAX away. Capped, so a zero extinction cannot meet an overflowed distance
            // (0 * inf) once the compiler folds exp's log2(e) into it; any medium is far shorter.
            float3 shadowTransmittance = intoMedium ? exp(-extinction * min(ls.distance, MAX_MEDIUM_DISTANCE))
                                                    : AtmosphereTransmittance(ls.distance);
            // Clamped unshadowed, which the shadow ray's visibility (at most 1) only scales down.
            neeContribution = ClampIndirect(throughput * fCos * ls.radiance * shadowTransmittance * (w / ls.pdf), visible);
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
    }
    else if (continuePath)
    {
        // A pane's reflection is not its guides' (what is behind it is), see PaneReflection. The
        // guides' reflection is the coat's or the base's, whichever reflects more.
        specularReflection = !paneSplit && bs.lobe == (CoatLeadsGuides(ctx) ? LOBE_COAT : LOBE_SPECULAR_REFLECTION);
        float lobeProbability = paneSplit ? ctx.probSpecR : glassSplit ? ctx.probSpecT : 1.0;
        if ((bs.lobe & DIFFUSE_BOUNCE_LOBES) != 0)
            flags += 1u << PATH_DIFFUSE_BOUNCES_SHIFT;
        throughput *= bs.weight * lobeProbability;
        prevBsdfPdf = bs.pdf;
        prevVertex = vertex;
        // Surface curvature is ignored. The cap stops a tiny pdf from making the width infinite.
        cone.spread = min(cone.spread + g_Frame.rayConeBsdfSpread * SampledConeSpread(bs.pdf / lobeProbability), PI);

        // Heading into the object means travelling through its medium until the next hit.
        segmentExtinction = dot(bs.wi, hit.Ng) < 0.0 ? extinction : float3(0.0, 0.0, 0.0);
        dir = bs.wi;
    }
    // Lights the next segment finds come at full weight if this vertex left them to its BSDF sample, or,
    // through a pass-through wall, if the vertex the segment continues from did.
    flags = SetFlag(flags, DEFERRED_SMOOTH_SEGMENT, passThrough ? smoothSegment : !sampleLights);

    // DLSS-RR guides: the primary surface, or the first surface past the mirrors and glass in front
    // of it. A surface that takes another lobe, or ends the path, ends the chain on itself.
    float psrFraction = 1.0;
#ifdef PSR
    bool psrEvent = continuePath && !paneSplit && (fromCamera || chainActive) &&
                    (glassSplit ? CoatedSpecularRoughness(mat) <= g_Frame.psrMaxRoughness : IsPsrEvent(mat, ctx, lobe, psrFraction));
#else
    const bool psrEvent = false;
#endif
    if (cameraSegment)
    {
        // Only the parts this segment changes come in. The primary surface's guides are written afresh,
        // and a chain needs the guides only where it ends.
        bool chainEnds = chainActive && !passThrough && !psrEvent;
        bool panesChange = paneSplit || reflectionDeferred;
        uint parts = (chainActive ? CAMERA_CHAIN : 0) | (chainEnds ? CAMERA_GUIDES | CAMERA_DIRECTION : 0) |
                     (panesChange ? CAMERA_PANES : 0) | (fromCamera || chainEnds ? CAMERA_VIEW : 0);
        CameraState camera = firstSegment ? DefaultCameraState(dirIn) : LoadCameraState(parts);
        uint loaded = firstSegment ? CAMERA_ALL : parts;
        ArriveCameraState(camera, loaded, fromCamera, chainActive, hitDistanceArrived, paneHitArrived, true, hit.t, arrivedExtinction);

        if (fromCamera)
        {
            camera.guides.position = float4(hit.position, 1.0);
            camera.guides.normal = mat.geometryNormal;
            camera.guides.roughness = GuideRoughness(mat, ctx);
            // Until the path continues with a specular reflection: no reflection data, or last frame's.
            camera.guides.specularHitDistance = InitialSpecularHitDistance(camera.guides.roughness);
            camera.guides.curvature = ViewedCurvature(hit, wo);
            camera.guides.hasChainMotion = false;
            ReflectanceGuides(mat, ctx, wo, camera.guides);
            camera.guides.diffuseAlbedo *= camera.transmittance;
            camera.guides.specularAlbedo *= camera.transmittance;
        }

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

        if (fromCamera)
        {
            if (specularReflection)
                flags |= PATH_HIT_DISTANCE_PENDING;
            if (psrEvent)
                camera.chain = StartSpecularChain(mat, ctx, hit, wo, dirIn, dir, specularReflection, psrFraction, camera.distance,
                                                  cone.width);
            if (!opaque && !mat.geometryThinWalled && !(psrEvent && camera.chain.throughGlass))
                camera.guides.diffuseAlbedo += camera.transmittance * PsrEventThroughput(mat, ctx, wo, false);
            if (passThrough)
                camera.transmittance *= passTransmittance;
        }
        else if (chainActive)
        {
            ChainArrive(camera.chain, dirIn, hit);
            if (passThrough)
            {
                camera.chain.throughput *= passTransmittance;
            }
            else if (psrEvent)
            {
                AddPsrEvent(camera.chain, mat, ctx, hit, wo, dirIn, dir, specularReflection, cone.width);
            }
            else
            {
                camera.chain.active = false;
                if (!ctx.inside && ReplacePrimarySurface(camera.chain, mat, ctx, ViewedCurvature(hit, wo), dirIn, hit.Ng, wo,
                                                         camera.cameraDir, camera.guides))
                    flags = SetFlag(flags, PATH_HIT_DISTANCE_PENDING, specularReflection);
                else if (camera.chain.throughGlass)
                    camera.guides.diffuseAlbedo += camera.transmittance * camera.chain.throughput;
            }
        }

        uint changed = (fromCamera || chainEnds ? CAMERA_GUIDES : 0) | CAMERA_CHAIN | (panesChange ? CAMERA_PANES : 0) |
                       (fromCamera ? CAMERA_VIEW : 0) | (firstSegment ? CAMERA_ALL : 0);
        StoreCameraState(camera, changed);
        flags = SetFlag(flags, PATH_CHAIN_ACTIVE, camera.chain.active);
    }

    if (continuePath)
    {
        origin = OffsetRay(hit.position, hit.Ng, dir);
        // Through the medium under the surface: what leaves here comes in at other points of the
        // object, and the path goes on out of one of them.
        if (!SIMPLE && lobe == LOBE_SUBSURFACE)
        {
            SubsurfaceExit exit = (SubsurfaceExit)0;
            if (dot(dir, hit.Ng) < 0.0)
                radiance += ClampIndirect(SubsurfaceScatter(mat, hit.position, hit.Ng, hit.N, hit.materialId, dir, throughput,
                                                            rayFlags, bounce + 1, rng, exit),
                                          visible);
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
        radiance += neeContribution * CastShadowRay(hit.position, hit.Ng, ls.direction, ls.distance, rayFlags);

    SurfaceOutcome outcome;
    outcome.continuePath = continuePath;
    outcome.passThrough = passThrough;
    outcome.rrU = lightU.z;
    return outcome;
}

// The DLSS-RR guides come out in g_CameraState, see FinishGuides.
float3 TracePath(float3 origin, float3 dir, inout uint rng)
{
    uint flags = PATH_FROM_CAMERA;
    uint2 segmentMedium = SEGMENT_MEDIUM_AIR;

    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float prevBsdfPdf = 0.0;
    PackedLightSelectVertex prevVertex = (PackedLightSelectVertex)0; // where the current segment started, once a surface scattered
    RayCone cone = { 0.0, g_Frame.rayConePixelSpread };

    uint bounce = 0;
    for (;;)
    {
        // ShadeSurface ends the path by the vertex both budgets together reach; the bounces are indirect,
        // so the primary segment comes on top.
        for (; bounce <= g_Frame.maxDiffuseBounces + g_Frame.maxSpecularBounces; ++bounce)
        {
            const uint rayFlags = OpacityMicromapRayFlags(bounce);
            HitPayload payload =
                    TraceClosest(origin, dir, bounce >= SER_START_BOUNCE, (flags & (PATH_FROM_CAMERA | PATH_CHAIN_ACTIVE)) != 0, rayFlags);
            // Unpacked after the in-scatter's shadow ray, so only the packed payload is held across it.
            HitInfo hit = HitDistance(payload);

            // Decoded after the trace, so only the packed forms were held across it.
            const bool fromCamera = (flags & PATH_FROM_CAMERA) != 0;
            const bool chainActive = ChainActive(flags);
            const bool hitDistanceArrived = (flags & PATH_HIT_DISTANCE_PENDING) != 0;
            const bool paneHitArrived = (flags & PATH_PANE_HIT_PENDING) != 0;
            const bool smoothSegment = (flags & DEFERRED_SMOOTH_SEGMENT) != 0;
            // Nothing is stored before the camera ray's segment, which starts from the defaults.
            const bool firstSegment = bounce == 0;
            flags &= ~(PATH_HIT_DISTANCE_PENDING | PATH_PANE_HIT_PENDING | DEFERRED_SMOOTH_SEGMENT);
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
                    radiance += ClampIndirect(throughput * AtmosphereInScattering(origin, dir, segmentLength, rayFlags, rng),
                                              fromCamera || chainActive || (flags & PATH_DIRECT_SEGMENT) != 0);
                throughput *= AtmosphereTransmittance(segmentLength);
            }
    #endif

            // Segments that touch the camera state load the parts they change late, and store them again
            // before the next TraceRay, so it stays out of the registers through the shading.
            bool cameraSegment = fromCamera || chainActive || hitDistanceArrived || paneHitArrived;

            if (!hit.hit)
            {
                // A miss adds nothing to a chain, which the chain's end after the loop takes care of.
                if (cameraSegment)
                {
                    CameraState camera = firstSegment ? DefaultCameraState(dir) : (CameraState)0;
                    ArriveCameraState(camera, firstSegment ? CAMERA_ALL : 0, fromCamera, false, hitDistanceArrived, paneHitArrived,
                                      false, 0.0, segmentExtinction);
                    // Sky seen through a window: the guides describe the sky, as for a camera ray.
                    if (fromCamera)
                        camera.guides = DefaultGuides(dir);
                    StoreCameraState(camera, firstSegment ? CAMERA_ALL & ~CAMERA_CHAIN : fromCamera ? CAMERA_GUIDES : 0);
                }

                // A ray inside a medium that finds no way out ends there.
                if (!inMedium)
                {
                    // The sky is never light sampled, so BSDF sampling gets its full contribution.
                    float3 sky = EnvironmentRadiance(dir);

                    // The sun disc is also reached by NEE, so weight it against light sampling.
                    float3 Lsun = SunRadiance(dir);
                    if (any(Lsun > 0.0))
                    {
                        float w = fromCamera || smoothSegment ? 1.0 : PowerHeuristic(prevBsdfPdf, LightPdf(origin, dir, hit));
                        sky += Lsun * w;
                    }
                    radiance += ClampIndirect(throughput * sky, fromCamera || chainActive || (flags & PATH_DIRECT_SEGMENT) != 0);
                }
                break;
            }

            // Most surfaces have only the base's lobes and shade in a specialisation without the others, in far
            // fewer registers. SER orders the hits by material, so a wave mostly takes just one of the two.
            SurfaceOutcome outcome;
            if (IsSimpleMaterial(payload.materialId & ~TANGENT_SIGN_BIT))
                outcome = ShadeSurface<true>(payload, origin, dir, bounce, rayFlags, fromCamera, chainActive, hitDistanceArrived,
                                             paneHitArrived, smoothSegment, firstSegment, cameraSegment, medium.xyz, flags, rng,
                                             radiance, throughput, prevBsdfPdf, prevVertex, cone, segmentExtinction, rrInvScale);
            else
                outcome = ShadeSurface<false>(payload, origin, dir, bounce, rayFlags, fromCamera, chainActive, hitDistanceArrived,
                                              paneHitArrived, smoothSegment, firstSegment, cameraSegment, medium.xyz, flags, rng,
                                              radiance, throughput, prevBsdfPdf, prevVertex, cone, segmentExtinction, rrInvScale);

            if (!outcome.continuePath)
                break;

            if (!outcome.passThrough)
                flags = SetFlag(flags & ~PATH_FROM_CAMERA, PATH_DIRECT_SEGMENT, fromCamera || chainActive);

            // Russian roulette.
            if (bounce >= RR_START_BOUNCE)
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
        bounce = flags >> DEFERRED_BOUNCE_SHIFT;
        uint deferredFlags = flags & DEFERRED_MASK;
        // Glass splits only where the camera sees, so every bounce before the branch is specular.
        flags &= ~(DEFERRED_MASK | PATH_HIT_DISTANCE_PENDING | (PATH_DIFFUSE_BOUNCES_MASK << PATH_DIFFUSE_BOUNCES_SHIFT));
        if ((deferredFlags & DEFERRED_REFLECTION) != 0)
        {
            // Solid glass's reflection leaves the surface the camera saw.
            flags |= PATH_DIRECT_SEGMENT;
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

// Entry points

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
    CameraState camera = LoadCameraState(CAMERA_GUIDES | CAMERA_PANES | CAMERA_DIRECTION);
    PrimaryGuides guides = FinishGuides(camera, origin);

    // Last frame's specular hit distance, loaded first so its latency hides behind the guides below.
    float heldHitDistance = g_SpecularHitDistance[pixel];

    // Static scene, so the primary point was in the same place last frame and only the camera
    // moved. CalculateMotionVector works in NDC, whose +y is up in D3D, while DLSS wants +y down.
    float2 motion = guides.hasChainMotion
                    ? guides.chainMotion
                    : CalculateMotionVector(guides.position, guides.position, g_Frame.prevViewProj, g_Frame.viewProj, float2(dims));
    motion.y = -motion.y;

    // Reversed Z: the sky sits on the far plane at 0.
    float4 clip = mul(guides.position, g_Frame.viewProj);
    float depth = guides.position.w > 0.0 ? clip.z / clip.w : 0.0;

    g_DiffuseAlbedo[pixel] = float4(guides.diffuseAlbedo, 1.0);
    g_SpecularAlbedo[pixel] = float4(guides.specularAlbedo, 1.0);
    g_NormalRoughness[pixel] = float4(guides.normal, guides.roughness);
    g_MotionVectors[pixel] = motion;
    g_Depth[pixel] = depth;

    // Negative: keep last frame's, which is whatever the texture was created with until a frame writes it.
    float hitDistance = guides.specularHitDistance;
    if (hitDistance >= 0.0)
        g_SpecularHitDistance[pixel] = hitDistance;
    else
        hitDistance = heldHitDistance;
    if (!isfinite(hitDistance))
        hitDistance = 0.0;

    // The reflected point is static too, so only the camera moves its image.
    float4 reflection = SpecularVirtualPosition(guides, camera.cameraDir, hitDistance);
    float2 specularMotion = CalculateMotionVector(reflection, reflection, g_Frame.prevViewProj, g_Frame.viewProj, float2(dims));
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
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
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
    payload.materialId = prim.materialIndex | (tangentNegative ? TANGENT_SIGN_BIT : 0u);
    float2 uv0 = v0.texCoord;
    float2 uv1 = v1.texCoord;
    float2 uv2 = v2.texCoord;
    payload.uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    float4 uvGradients = RayConeUvGradients(e1, e2, faceCross, uv0, uv1, uv2, WorldRayDirection());
    payload.uvGradients = uint2(PackHalf2(uvGradients.xy), PackHalf2(uvGradients.zw));

    uint emitterTable = g_Emitters[InstanceIndex()];
    payload.emitter = emitterTable != INVALID_LIGHT_INDEX ? emitterTable + PrimitiveIndex() : INVALID_LIGHT_INDEX;

#ifdef DLSS_RR
    // How the shading normal turns over the triangle, telling flat mirrors from curved ones for PSR and
    // carrying ray differentials through them. Only the guides use it, so other hits skip it.
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
#else
    payload.shape = uint2(0, 0);
#endif
}

// Alpha for camera and bounce rays: masked surfaces are cut at alphaCutoff.
[shader("anyhit")]
void AnyHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
    MaterialData material = LoadGpuMaterial(prim.materialIndex);
    if (material.alphaMode != ALPHA_MODE_MASK)
        return;
    float2 uv = InterpolateTexCoord(LoadTriangle(prim, PrimitiveIndex()), Barycentrics(attr.barycentrics));
    if (LoadBaseColourAlpha(material, uv).a < material.alphaCutoff)
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
    if (!ShadowCandidateTransmittance(LoadPrimitive(InstanceID()), PrimitiveIndex(), attr.barycentrics,
                                      (float3x3)ObjectToWorld3x4(), WorldRayDirection(), transmitted))
        IgnoreHit();
    payload.visibility *= transmitted;
    if (dot(payload.visibility, 0.333) > 0.001)
        IgnoreHit();
    AcceptHitAndEndSearch();
}
#endif
