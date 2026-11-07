// Shadow rays: TraceRay with the shadow hit group, or with INLINE_SHADOWS an inline ray query, which
// keeps the path's state in registers instead of saving and restoring it around a TraceRay.
import PathTracer.Materials;
import PathTracer.Math;
import PathTracer.RayTracing;
import PathTracer.Resources;

// Blended surfaces block a shadow ray by chance (CandidateThere), so their unknown micro-triangles must reach
// the any-hit: no RAY_FLAG_FORCE_OMM_2_STATE, which would resolve them by its fixed cut. Shadow rays trace the
// TLAS whose masks alone read their micromaps in 2-state mode instead (ShadowScene).
static const uint SHADOW_RAY_FLAGS = RAY_FLAG_NONE;

// Visibility of the light from 0 (occluded) to 1 (unoccluded) per channel, in the given TLAS.
float3 TraceShadowRayIn(RaytracingAccelerationStructure scene, float3 origin, float3 directionToLight, float tracingDistance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = directionToLight;
    ray.TMin = 0.0f;
    ray.TMax = tracingDistance;

#ifdef INLINE_SHADOWS
    // Opaque triangles commit themselves and end the search; the rest come here as ShadowAnyHit would.
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, SHADOW_RAY_FLAGS, g_Frame.shadowInstanceMask, ray);
    float3 visibility = float3(1.0, 1.0, 1.0);
    while (query.Proceed())
    {
        float3 transmitted;
        if (!ShadowCandidateTransmittance(LoadPrimitive(query.CandidateInstanceID() + query.CandidateGeometryIndex()), query.CandidatePrimitiveIndex(),
                                          query.CandidateTriangleBarycentrics(), (float3x3)query.CandidateObjectToWorld3x4(), origin,
                                          directionToLight, query.CandidateTriangleRayT(), transmitted))
            continue;
        visibility *= transmitted;
        if (dot(visibility, 0.333) <= 0.001)
            query.CommitNonOpaqueTriangleHit();
    }
    return query.CommittedStatus() == COMMITTED_NOTHING ? visibility : float3(0.0, 0.0, 0.0);
#else
    ShadowRayPayload payload;
    payload.visibility = float3(1.0f, 1.0f, 1.0f);

    TraceRay(scene, SHADOW_RAY_FLAGS | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, g_Frame.shadowInstanceMask, SHADOW_RAY_INDEX, 0, SHADOW_RAY_INDEX,
             ray, payload);

    return payload.visibility;
#endif
}

// The TLAS shadow rays trace: with micromaps, the one whose masks read them in 2-state mode (blended surfaces
// keep 4-state there).
RaytracingAccelerationStructure ShadowScene()
{
    return SceneFor(RAY_SCENE_OMM_2_STATE);
}

// Visibility of the light from 0 (occluded) to 1 (unoccluded) per channel.
float3 TraceShadowRay(float3 origin, float3 directionToLight, float tracingDistance)
{
    return TraceShadowRayIn(ShadowScene(), origin, directionToLight, tracingDistance);
}

// Shadow ray from a surface, offset to whichever side the light is on (below it for transmission).
// The caller already skips back-facing light directions on opaque surfaces.
float3 CastShadowRay(float3 hitPosition, float3 surfaceNormal, float3 directionToLight, float tracingDistance)
{
    return TraceShadowRay(OffsetRay(hitPosition, surfaceNormal, directionToLight), directionToLight, tracingDistance);
}
