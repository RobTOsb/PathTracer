// Shadow rays: TraceRay with the shadow hit group, or with INLINE_SHADOWS an inline ray query, which
// keeps the path's state in registers instead of saving and restoring it around a TraceRay.
import PathTracer.Materials;
import PathTracer.Math;
import PathTracer.RayTracing;
import PathTracer.Resources;

// Visibility of the light from 0 (occluded) to 1 (unoccluded) per channel. rayFlags: OpacityMicromapRayFlags.
float3 TraceShadowRay(float3 origin, float3 directionToLight, float tracingDistance, uint rayFlags)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = directionToLight;
    ray.TMin = 0.0f;
    ray.TMax = tracingDistance;

#ifdef INLINE_SHADOWS
    // Opaque triangles commit themselves and end the search; the rest come here as ShadowAnyHit would.
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(g_Scene, rayFlags, 0xFF, ray);
    float3 visibility = float3(1.0, 1.0, 1.0);
    while (query.Proceed())
    {
        float3 transmitted;
        if (!ShadowCandidateTransmittance(LoadPrimitive(query.CandidateInstanceID()), query.CandidatePrimitiveIndex(),
                                          query.CandidateTriangleBarycentrics(),
                                          (float3x3)query.CandidateObjectToWorld3x4(), directionToLight, transmitted))
            continue;
        visibility *= transmitted;
        if (dot(visibility, 0.333) <= 0.001)
            query.CommitNonOpaqueTriangleHit();
    }
    return query.CommittedStatus() == COMMITTED_NOTHING ? visibility : float3(0.0, 0.0, 0.0);
#else
    ShadowRayPayload payload;
    payload.visibility = float3(1.0f, 1.0f, 1.0f);

    TraceRay(g_Scene, rayFlags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, SHADOW_RAY_INDEX, 0, SHADOW_RAY_INDEX,
             ray, payload);

    return payload.visibility;
#endif
}

// Shadow ray from a surface, offset to whichever side the light is on (below it for transmission).
// The caller already skips back-facing light directions on opaque surfaces.
float3 CastShadowRay(float3 hitPosition, float3 surfaceNormal, float3 directionToLight, float tracingDistance,
                     uint rayFlags)
{
    return TraceShadowRay(OffsetRay(hitPosition, surfaceNormal, directionToLight), directionToLight, tracingDistance,
                          rayFlags);
}
