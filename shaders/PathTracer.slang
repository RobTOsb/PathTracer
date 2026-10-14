import OpenPBRBaseSubstrate;
import PathTracer.Resources;
import PathTracer.Math;
import PathTracer.Random;
import PathTracer.Geometry;
import PathTracer.RayTracing;
import PathTracer.Sky;
import PathTracer.LightSampling;
import PathTracer.Materials;
import PathTracer.Guides;
import PathTracer.Psr;
import PathTracer.GlassSplit;

static const uint RR_START_BOUNCE = 1;

float3 TracePath(float3 origin, float3 dir, inout uint rng, out PrimaryGuides guides)
{
    guides = DefaultGuides(dir);
    // The guides' surface continued the path with a specular reflection, so the next hit gives its
    // specular hit distance.
    bool hitDistancePending = false;
    SpecularChain chain = (SpecularChain)0;
    const float3 cameraOrigin = origin;
    const float3 cameraDir = dir;

    PaneReflection panes = (PaneReflection)0;
    bool paneHitPending = false; // the brightest pane's reflection ray ends at the next hit
    DeferredBranch deferred = (DeferredBranch)0;
    uint splits = 0;
    // Russian roulette compares the throughput with this, the weight of the branch it is on, so a
    // pane's reflection survives as long as a path of its own would.
    float rrInvScale = 1.0;

    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float3 segmentExtinction = float3(0.0, 0.0, 0.0);
    float prevBsdfPdf = 0.0;
    LightSelectVertex prevVertex = (LightSelectVertex)0; // where the current segment started, once a surface scattered
    RayCone cone = { 0.0, g_Frame.rayConePixelSpread };

    bool fromCamera = true;
    float3 cameraTransmittance = float3(1.0, 1.0, 1.0); // of the walls in front of the primary surface
    float cameraDistance = 0.0;

    // maxBounces counts indirect bounces, so the primary segment comes on top.
    const uint maxSegments = g_Frame.maxBounces + 1;

    uint bounce = 0;
    for (;;)
    {
        for (; bounce < maxSegments; ++bounce)
        {
            HitInfo hit = TraceClosest(origin, dir, bounce >= SER_START_BOUNCE, rng);

            if (hitDistancePending)
            {
                guides.specularHitDistance = hit.hit ? hit.t : SKY_HIT_DISTANCE;
                hitDistancePending = false;
            }
            if (paneHitPending)
            {
                panes.hitDistance = hit.hit ? hit.t : SKY_HIT_DISTANCE;
                paneHitPending = false;
            }
            if (chain.active)
                chain.length += hit.t;
            if (fromCamera)
                cameraDistance += hit.t;

            // Absorption along the segment just travelled, if it was inside a medium.
            if (any(segmentExtinction > 0.0))
            {
                if (!hit.hit)
                    break;
                float3 absorption = exp(-segmentExtinction * hit.t);
                throughput *= absorption;
                if (chain.active)
                    chain.throughput *= absorption;
            }
    #ifdef ATMOSPHERE
            else
            {
                // Otherwise it crossed the scene's air, which scatters sunlight into the path and dims
                // whatever lies beyond. Rays that leave the scene cross atmosphereMissDistance of it.
                float segmentLength = hit.hit ? hit.t : g_Frame.atmosphereMissDistance;
                if (bounce < g_Frame.atmosphereInScatterSegments)
                    radiance += throughput * AtmosphereInScattering(origin, dir, segmentLength, rng);
                throughput *= AtmosphereTransmittance(segmentLength);
            }
    #endif

            if (!hit.hit)
            {
                // Sky seen through a window: the guides describe the sky, as for a camera ray.
                if (fromCamera)
                    guides = DefaultGuides(dir);

                // The sky is never light sampled, so BSDF sampling gets its full contribution.
                radiance += throughput * EnvironmentRadiance(dir);

                // The sun disc is also reached by NEE, so weight it against light sampling.
                float3 Lsun = SunRadiance(dir);
                if (any(Lsun > 0.0))
                {
                    float w = fromCamera || (deferred.flags & DEFERRED_SMOOTH_SEGMENT) != 0 ? 1.0 : PowerHeuristic(prevBsdfPdf, LightPdf(origin, dir, hit));
                    radiance += throughput * Lsun * w;
                }
                break;
            }

            // Material and shading basis for this hit, its textures filtered over the cone's footprint.
            cone.width += cone.spread * hit.t;
            OpenPBRBaseParams mat = LoadMaterial(hit, cone.width * g_Frame.textureLodScale, bounce < g_Frame.anisotropicSegments);
            float3 wo = -dir;

            // Front-face emission found by BSDF sampling, MIS weighted against NEE at the previous vertex
            // unless the triangle has no light (degenerate or black) or light buffer NEE is off.
            if (any(hit.emission > 0.0) && dot(hit.Ng, wo) > 0.0)
            {
                float w = 1.0;
                if (!fromCamera && (deferred.flags & DEFERRED_SMOOTH_SEGMENT) == 0 && g_Frame.lightCount > 0 && hit.emitter != INVALID_LIGHT_INDEX)
                {
                    uint lightIndex = g_Emitters[hit.emitter];
                    if (lightIndex != INVALID_LIGHT_INDEX)
                        w = PowerHeuristic(prevBsdfPdf, EmitterLightPdf(prevVertex, lightIndex, dir, hit.position));
                }
                radiance += throughput * hit.emission * w;
            }
            deferred.flags &= ~DEFERRED_SMOOTH_SEGMENT;

            // Opaque and thin-walled surfaces are treated as two-sided: face the frame towards the viewer
            // so a back face is not mistaken for the inside of a transmissive object.
            bool opaque = mat.transmissionWeight <= 0.0;
            if ((opaque || mat.geometryThinWalled) && dot(hit.Ng, wo) < 0.0)
            {
                hit.Ng = -hit.Ng;
                hit.N = -hit.N;
                hit.Ns = -hit.Ns;
                hit.meanCurvature = -hit.meanCurvature;
            }

            SetShadingBasis(mat, hit.T, hit.B, hit.N, hit.Ns, wo);
            OpenPBRContext ctx = OpenPBRSetup(mat, wo);
            float3 extinction = MediumExtinction(mat);
            bool deltaTransmission = HasDeltaTransmission(mat);

            if (fromCamera)
            {
                guides.position = float4(hit.position, 1.0);
                guides.normal = mat.geometryNormal;
                guides.roughness = mat.specularRoughness;
                // Until the path continues with a specular reflection: no reflection data, or last frame's.
                guides.specularHitDistance = InitialSpecularHitDistance(mat.specularRoughness);
                guides.curvature = ViewedCurvature(hit, wo);
                ReflectanceGuides(mat, wo, guides);
                guides.diffuseAlbedo *= cameraTransmittance;
                guides.specularAlbedo *= cameraTransmittance;
            }

            // One hash covers the light sample (xy) and the Russian roulette decision (z).
            float3 lightU = Rand3(rng);

            LightSelectVertex vertex = MakeLightSelectVertex(hit.position, mat, ctx, wo, opaque || deltaTransmission, bounce);
            LightSample ls = SampleLight(vertex, lightU.xy, rng);
            bool lastSegment = bounce + 1 >= maxSegments;
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
                    float3 shadowTransmittance = intoMedium ? exp(-extinction * ls.distance) : AtmosphereTransmittance(ls.distance);
                    neeContribution = throughput * fCos * ls.radiance * shadowTransmittance * (w / ls.pdf);
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
            bool splittable = fromCamera && !lastSegment && splits < g_Frame.glassReflectionSplits && ctx.probDiffuse <= 0.0 &&
                              ctx.probSpecR > 0.0 && ctx.probSpecR < 1.0;
            // A thin pane: its reflection now, what is behind it deferred.
            bool paneSplit = splittable && deltaTransmission;
            // Solid glass: the refraction now, which the guides' chain follows, its reflection deferred.
            bool glassSplit = splittable && !mat.geometryThinWalled && !ctx.inside && mat.specularRoughness <= 0.0;
            if (paneSplit || glassSplit)
                ++splits;

            // Solid glass takes two samples, its deferred reflection's first, from one call site so the
            // sampler is inlined once. A split sets u.x to land in its lobe, so a sample is that or none.
            BSDFSample bs = (BSDFSample)0;
            BSDFSample reflected = (BSDFSample)0;
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
                    if (s == 0)
                        reflected = sample;
                    else
                        bs = sample;
                }
            }

            if (paneSplit)
            {
                float3 transmittance = ThinDeltaTransmittance(mat, ctx.wo.z);
                deferred.flags = DEFERRED_PENDING | ((bounce + 1) << DEFERRED_BOUNCE_SHIFT);
                deferred.origin = OffsetRay(hit.position, hit.Ng, dir);
                deferred.throughput = throughput * transmittance;
                deferred.dirOrCameraTransmittance = cameraTransmittance * transmittance;
                deferred.coneSpreadOrCameraDistance = cameraDistance;
                deferred.coneWidth = cone.width;

                PrimaryGuides seen;
                ReflectanceGuides(mat, wo, seen);
                paneHitPending = AddPaneReflection(panes, cameraTransmittance * seen.specularAlbedo, cameraDistance,
                                                   ViewedCurvature(hit, wo)) &&
                                 bs.lobe != LOBE_NONE;
            }
            else if (glassSplit && reflected.lobe != LOBE_NONE)
            {
                deferred.flags = DEFERRED_PENDING | DEFERRED_REFLECTION | ((bounce + 1) << DEFERRED_BOUNCE_SHIFT);
                deferred.origin = OffsetRay(hit.position, hit.Ng, reflected.wi);
                deferred.dirOrCameraTransmittance = reflected.wi;
                deferred.throughput = throughput * reflected.weight * ctx.probSpecR;
                deferred.coneWidth = cone.width;
                deferred.coneSpreadOrCameraDistance =
                        min(cone.spread + g_Frame.rayConeBsdfSpread * SampledConeSpread(reflected.pdf / ctx.probSpecR), PI);
                // The guides' specular albedo is the glass's reflection, which moves to the panes.
                if (AddPaneReflection(panes, guides.specularAlbedo, cameraDistance, ViewedCurvature(hit, wo)))
                    deferred.flags |= DEFERRED_BRIGHTEST_PANE;
                guides.specularAlbedo = float3(0.0, 0.0, 0.0);
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
                // A pane's reflection is not its guides' (what is behind it is), see PaneReflection.
                specularReflection = !paneSplit && bs.lobe == LOBE_SPECULAR_REFLECTION;
                float lobeProbability = paneSplit ? ctx.probSpecR : glassSplit ? ctx.probSpecT : 1.0;
                throughput *= bs.weight * lobeProbability;
                // Russian roulette on a pane's reflection is relative to the branch, as for a path of its own.
                if (paneSplit)
                    rrInvScale = 1.0 / max(MaxComponent(throughput), 1.0e-8);
                prevBsdfPdf = bs.pdf;
                prevVertex = vertex;
                // Surface curvature is ignored. The cap stops a tiny pdf from making the width infinite.
                cone.spread = min(cone.spread + g_Frame.rayConeBsdfSpread * SampledConeSpread(bs.pdf / lobeProbability), PI);

                // Heading into the object means travelling through its medium until the next hit.
                segmentExtinction = dot(bs.wi, hit.Ng) < 0.0 ? extinction : float3(0.0, 0.0, 0.0);
                dir = bs.wi;
            }

            // DLSS-RR guides: the primary surface, or the first surface past the mirrors and glass in front
            // of it. A surface that takes another lobe, or ends the path, ends the chain on itself.
            float psrFraction = 1.0;
            bool psrEvent = continuePath && !paneSplit && (fromCamera || chain.active) &&
                            (glassSplit ? mat.specularRoughness <= g_Frame.psrMaxRoughness : IsPsrEvent(mat, ctx, lobe, psrFraction));
            if (fromCamera)
            {
                hitDistancePending = specularReflection;
                if (psrEvent)
                    chain = StartSpecularChain(mat, ctx, wo, dirIn, dir, specularReflection, psrFraction, cameraDistance, cone.width,
                                               hit.curvature);
                if (!opaque && !mat.geometryThinWalled && !(psrEvent && chain.throughGlass))
                    guides.diffuseAlbedo += cameraTransmittance * PsrEventThroughput(mat, ctx, wo, false);
                if (passThrough)
                    cameraTransmittance *= passTransmittance;
            }
            else if (chain.active)
            {
                if (passThrough)
                {
                    chain.throughput *= passTransmittance;
                }
                else if (psrEvent)
                {
                    AddPsrEvent(chain, mat, ctx, wo, dirIn, dir, specularReflection, cone.width, hit.curvature);
                }
                else
                {
                    chain.active = false;
                    if (!ctx.inside && ReplacePrimarySurface(chain, mat, ViewedCurvature(hit, wo), wo, cameraDir, guides))
                        hitDistancePending = specularReflection;
                    else if (chain.throughGlass)
                        guides.diffuseAlbedo += cameraTransmittance * chain.throughput;
                }
            }

            if (continuePath)
                origin = OffsetRay(hit.position, hit.Ng, dir);

            if (traceShadowRay)
                radiance += neeContribution * CastShadowRay(hit.position, hit.Ng, ls.direction, ls.distance);

            if (!continuePath)
                break;

            fromCamera = fromCamera && passThrough;

            // Russian roulette.
            if (bounce >= RR_START_BOUNCE)
            {
                float q = saturate(MaxComponent(throughput) * rrInvScale);
                if (lightU.z >= q)
                    break;
                throughput /= q;
            }
        }

        // A refraction chain the path ended on leaves the glass's transmittance as its diffuse albedo.
        if (chain.active && chain.throughGlass)
            guides.diffuseAlbedo += cameraTransmittance * chain.throughput;
        chain.active = false;

        if ((deferred.flags & DEFERRED_PENDING) == 0)
            break;

        origin = deferred.origin;
        throughput = deferred.throughput;
        cone.width = deferred.coneWidth;
        segmentExtinction = float3(0.0, 0.0, 0.0);
        bounce = deferred.flags >> DEFERRED_BOUNCE_SHIFT;
        hitDistancePending = false;
        if ((deferred.flags & DEFERRED_REFLECTION) != 0)
        {
            dir = deferred.dirOrCameraTransmittance;
            cone.spread = deferred.coneSpreadOrCameraDistance;
            rrInvScale = 1.0;
            paneHitPending = (deferred.flags & DEFERRED_BRIGHTEST_PANE) != 0;
            fromCamera = false;
            deferred.flags = DEFERRED_SMOOTH_SEGMENT;
        }
        else
        {
            dir = cameraDir;
            cameraTransmittance = deferred.dirOrCameraTransmittance;
            cameraDistance = deferred.coneSpreadOrCameraDistance;
            cone.spread = g_Frame.rayConePixelSpread;
            prevBsdfPdf = 0.0;
            rrInvScale = 1.0;
            fromCamera = true;
            deferred.flags = 0;
        }
    }

    // The specular guides follow the reflection on the guides' surface, or on the brightest split pane
    // in front of it if that reflects more.
    guides.reflector = guides.position;
    guides.reflectorRoughness = guides.roughness;
    if (panes.luminance > 0.0)
    {
        bool paneReflects = panes.luminance >= Luminance(guides.specularAlbedo);
        guides.specularAlbedo += panes.albedo;
        if (paneReflects)
        {
            guides.reflector = float4(cameraOrigin + cameraDir * panes.distance, 1.0);
            guides.reflectorRoughness = 0.0;
            guides.roughness = 0.0;
            guides.curvature = panes.curvature;
            guides.specularHitDistance = panes.hitDistance;
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
    uint rng = RngInit(g_Frame.frameIndex);

#ifdef DLSS_RR
    // Every pixel shares this frame's jitter, which DLSS is told about separately.
    float2 subpixel = 0.5 + g_Frame.jitter;
#else
    float2 subpixel = Rand3(rng).xy;
#endif

    float3 origin, dir;
    GenerateCameraRay(pixel, subpixel, origin, dir);
    PrimaryGuides guides;
    float3 L = TracePath(origin, dir, rng, guides);

    // A single bad sample would otherwise stick in any temporal filter downstream.
    if (any(isnan(L)) || any(isinf(L)))
        L = float3(0.0, 0.0, 0.0);

    g_Output[pixel] = float4(L, 1.0);

#ifdef DLSS_RR
    // Last frame's specular hit distance, loaded first so its latency hides behind the guides below.
    float heldHitDistance = g_SpecularHitDistance[pixel];

    // Static scene, so the primary point was in the same place last frame and only the camera
    // moved. CalculateMotionVector works in NDC, whose +y is up in D3D, while DLSS wants +y down.
    float2 motion = CalculateMotionVector(guides.position, guides.position, g_Frame.prevViewProj, g_Frame.viewProj, float2(dims));
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
    float4 reflection = SpecularVirtualPosition(guides, dir, hitDistance);
    float2 specularMotion = CalculateMotionVector(reflection, reflection, g_Frame.prevViewProj, g_Frame.viewProj, float2(dims));
    specularMotion.y = -specularMotion.y;
    g_SpecularMotionVectors[pixel] = specularMotion;
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
    uint3 tri = LoadTriangle(prim);
    float3 bary = Barycentrics(attr);

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
    payload.Ng = Ng;
    payload.materialId = prim.materialIndex;
    payload.N = normalize(mul(cofactor, objectNormal) * handedness);
    payload.T = normalize(mul(linearToWorld, objectTangent.xyz));
    // A mirroring transform flips cross(N, T), so fold its handedness into the bitangent sign.
    payload.tangentSign = (objectTangent.w < 0.0 ? -1.0 : 1.0) * handedness;
    float2 uv0 = v0.texCoord;
    float2 uv1 = v1.texCoord;
    float2 uv2 = v2.texCoord;
    payload.uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    payload.uvGradients = RayConeUvGradients(e1, e2, faceCross, uv0, uv1, uv2, WorldRayDirection());

#ifdef DLSS_RR
    // Largest turn of the vertex normals per unit length along an edge, telling flat mirrors from curved
    // ones for PSR. Chords between object-space normals stand in for angles, over world-space edges.
    float3 dn1 = v1.normal - v0.normal;
    float3 dn2 = v2.normal - v0.normal;
    float3 dn3 = dn2 - dn1;
    float3 e3 = e2 - e1;
    float k1 = dot(dn1, dn1) / max(dot(e1, e1), 1.0e-20);
    float k2 = dot(dn2, dn2) / max(dot(e2, e2), 1.0e-20);
    float k3 = dot(dn3, dn3) / max(dot(e3, e3), 1.0e-20);
    float curvature = sqrt(max(k1, max(k2, k3)));
    float3 o1 = p1 - p0;
    float3 o2 = p2 - p0;
    float3 o3 = o2 - o1;
    float m1 = dot(dn1, o1) * rsqrt(max(dot(o1, o1) * dot(e1, e1), 1.0e-30));
    float m2 = dot(dn2, o2) * rsqrt(max(dot(o2, o2) * dot(e2, e2), 1.0e-30));
    float m3 = dot(dn3, o3) * rsqrt(max(dot(o3, o3) * dot(e3, e3), 1.0e-30));
    float meanCurvature = (m1 + m2 + m3) * (1.0 / 3.0);
    payload.curvature = f32tof16(min(curvature, HALF_MAX)) | (f32tof16(clamp(meanCurvature, -HALF_MAX, HALF_MAX)) << 16);
#else
    payload.curvature = 0;
#endif

    uint emitterTable = g_Emitters[InstanceIndex()];
    payload.emitter = emitterTable != INVALID_LIGHT_INDEX ? emitterTable + PrimitiveIndex() : INVALID_LIGHT_INDEX;
}

// Alpha for camera and bounce rays: masked surfaces are cut at alphaCutoff, blended ones use
// stochastic transparency, stopping the ray with probability alpha.
[shader("anyhit")]
void AnyHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
    GpuMaterial material = LoadGpuMaterial(prim.materialIndex);
    if (material.alphaMode == ALPHA_MODE_OPAQUE)
        return;
    float alpha = LoadBaseColourAlpha(material, prim, attr).a;

    bool cutOut = material.alphaMode == ALPHA_MODE_MASK ? alpha < material.alphaCutoff
                                                        : SurfaceRandom(payload.materialId) >= alpha;
    if (cutOut)
        IgnoreHit();
}

[shader("miss")]
void ShadowMiss(inout ShadowRayPayload payload)
{
}

[shader("closesthit")]
void ShadowClosestHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.visibility = float3(0.0, 0.0, 0.0);
}

// Shadow rays stop at the first surface that blocks them, multiplying in what the blended (1 - alpha)
// and pass-through surfaces before it let through.
[shader("anyhit")]
void ShadowAnyHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PrimitiveDesc prim = LoadPrimitive(InstanceID());
    GpuMaterial material = LoadGpuMaterial(prim.materialIndex);
    float4 albedo = LoadBaseColourAlpha(material, prim, attr);

    // Of the light reaching the surface, the share that meets it rather than passing a cut-out or blend.
    float coverage = 1.0;
    if (material.alphaMode == ALPHA_MODE_MASK && albedo.a < material.alphaCutoff)
        IgnoreHit();
    if (material.alphaMode == ALPHA_MODE_BLEND)
        coverage = albedo.a;

    float3 transmitted = float3(1.0 - coverage, 1.0 - coverage, 1.0 - coverage);
    if ((material.transmissionFlags & TRANSMISSION_PASS_THROUGH) != 0)
        transmitted += coverage * PassThroughTransmittance(material, prim, attr, albedo.rgb);

    payload.visibility *= transmitted;
    if (dot(payload.visibility, 0.333) > 0.001)
        IgnoreHit();
    AcceptHitAndEndSearch();
}
