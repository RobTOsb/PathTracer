// Ray payloads, camera rays and the TraceRay wrappers for standard and shadow rays.
import PathTracer.Math;
import PathTracer.Resources;

struct HitInfo
{
    bool hit;
    float t;
    float3 position;
    float3 Ng; // geometric normal, oriented outwards
    float3 T;  // interpolated vertex tangent
    float3 B;  // interpolated vertex bitangent
    float3 N;  // interpolated vertex normal
    float3 Ns; // shading normal from the normal map in world space, equal to N without a map
    float2 uv;
    float4 uvGradients; // uv across the two ellipse axes of a ray cone of unit width, see RayConeUvGradients
    float3 emission; // emitted radiance (both faces), set by LoadMaterial
    uint materialId;
    uint emitter; // entry of the hit triangle in g_Emitters, INVALID_LIGHT_INDEX if its instance does not emit
    float curvature; // radians the vertex normals turn per unit length, DLSS_RR permutation only
    float meanCurvature; // signed, positive where the surface bulges towards N, DLSS_RR permutation only
};

// Ray types, indexing both the hit group and the miss shader in the shader table, so PathTracerPass
// must add its records in this order.
static const uint STANDARD_RAY_INDEX = 0;
static const uint SHADOW_RAY_INDEX = 1;

struct ShadowRayPayload
{
    float3 visibility;
};

// Packed like a G-buffer to keep it in registers: 44 bytes.
struct HitPayload
{
    float hitT;
    uint materialId; // on the way in, the seed AnyHit draws its blend decisions from; bit 31 is the tangent sign
    uint emitter;
    uint Ng; // octahedral snorm16x2
    uint N;  // octahedral snorm16x2
    uint T;  // octahedral snorm16x2
    float2 uv;
    uint2 uvGradients; // half4
    uint curvature; // half2: HitInfo::curvature, HitInfo::meanCurvature; on the way in, nonzero to compute them
};

static const uint TANGENT_SIGN_BIT = 0x80000000u;

// First bounce whose hits the SER pipeline reorders. Camera rays leave raygen in screen order, so
// their hits are already about as coherent as reordering would make them.
static const uint SER_START_BOUNCE = 1;

void GenerateCameraRay(uint2 pixel, float2 u, out float3 origin, out float3 dir)
{
    uint2 dims = DispatchRaysDimensions().xy;
    float2 ndc = (float2(pixel) + u) / float2(dims) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    // Unproject a near plane point (z = 1, reversed Z) into view space, then rotate into world space,
    // separately from the translation to avoid precision loss with a tiny near plane.
    float4 viewPoint = mul(float4(ndc, 1.0, 1.0), g_Frame.invProj);
    float3 viewDir = viewPoint.xyz / viewPoint.w;

    origin = g_Frame.invView[3].xyz;
    dir = normalize(mul(float4(viewDir, 0.0), g_Frame.invView).xyz);
}

// needCurvature: the hit feeds the DLSS-RR guides, the only users of its curvature. Returns the
// payload still packed (see HitDistance and UnpackHit), the smaller form to keep across other rays.
HitPayload TraceClosest(float3 origin, float3 dir, bool reorder, bool needCurvature, inout uint rng)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = FLT_MAX;

    HitPayload payload = (HitPayload)0;
    payload.hitT = -1.0;
    payload.materialId = rng++;
    payload.curvature = needCurvature ? 1u : 0u;
    // Masked and blended geometry runs AnyHit; everything else is flagged opaque in its BLAS.
#ifdef SER
    // DXR 1.2 SER: traverse, regroup threads by hit/miss and material, then shade. The threads stay in
    // the new order for the shading in TracePath, which is where the material divergence is.
    HitObject hitObject =
        HitObject::TraceRay(g_Scene, RAY_FLAG_NONE, 0xFF, STANDARD_RAY_INDEX, 0, STANDARD_RAY_INDEX, ray, payload);
    if (reorder)
    {
        uint materialHint = hitObject.IsHit() ? LoadPrimitive(hitObject.GetInstanceID()).materialIndex : 0;
        ReorderThread(hitObject, materialHint, g_Frame.serCoherenceHintBits);
    }
    // A fresh payload: only AnyHit read the traced one, so it need not survive the reorder.
    payload = (HitPayload)0;
    payload.hitT = -1.0;
    payload.curvature = needCurvature ? 1u : 0u;
    HitObject::Invoke(hitObject, payload);
#else
    TraceRay(g_Scene, RAY_FLAG_NONE, 0xFF, STANDARD_RAY_INDEX, 0, STANDARD_RAY_INDEX, ray, payload);
#endif
    return payload;
}

// Whether and how far the ray went, the rest of the hit left at zero.
HitInfo HitDistance(HitPayload payload)
{
    HitInfo hit = (HitInfo)0;
    hit.hit = payload.hitT >= 0.0;
    hit.t = hit.hit ? payload.hitT : 0.0;
    return hit;
}

// The whole hit of a ray from origin along dir that met a surface.
HitInfo UnpackHit(HitPayload payload, float3 origin, float3 dir)
{
    HitInfo hit;
    hit.hit = true;
    hit.t = payload.hitT;
    hit.position = origin + dir * payload.hitT;
    hit.Ng = OctDecode(payload.Ng);
    hit.N = OctDecode(payload.N);
    float3 T = OctDecode(payload.T);
    hit.T = normalize(T - hit.N * dot(hit.N, T));
    hit.B = cross(hit.N, hit.T) * ((payload.materialId & TANGENT_SIGN_BIT) != 0 ? -1.0 : 1.0);
    hit.uv = payload.uv;
    hit.uvGradients = float4(UnpackHalf2(payload.uvGradients.x), UnpackHalf2(payload.uvGradients.y));
    hit.materialId = payload.materialId & ~TANGENT_SIGN_BIT;
    hit.emitter = payload.emitter;
    float2 curvature = UnpackHalf2(payload.curvature);
    hit.curvature = curvature.x;
    hit.meanCurvature = curvature.y;
    hit.emission = float3(0.0, 0.0, 0.0);
    hit.Ns = hit.N; // LoadMaterial applies the normal map
    return hit;
}
