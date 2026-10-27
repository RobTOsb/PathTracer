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
    // How N turns as the hit point moves over the triangle (the derivative of the normalised interpolated
    // vertex normal), as a 2x2 matrix in the frame (T, cross(N, T)): see NormalDerivative. DLSS_RR only.
    float4 shape;
    float smoothLift;
};

// The change of N for a small move dP of the hit point along its triangle.
float3 NormalDerivative(HitInfo hit, float3 dP)
{
    float3 B = cross(hit.N, hit.T);
    float a = dot(dP, hit.T);
    float b = dot(dP, B);
    return hit.T * (hit.shape.x * a + hit.shape.y * b) + B * (hit.shape.z * a + hit.shape.w * b);
}

// Radians the vertex normals turn per unit length, at most: the shape matrix's larger singular value.
float HitCurvature(HitInfo hit)
{
    float4 s = hit.shape;
    return 0.5 * (length(float2(s.x + s.w, s.z - s.y)) + length(float2(s.x - s.w, s.y + s.z)));
}

// Signed mean curvature, positive where the surface bulges towards N.
float HitMeanCurvature(HitInfo hit)
{
    return 0.5 * (hit.shape.x + hit.shape.w);
}

// The shape of the hit with N turned around, which turns dN and cross(N, T) around with it.
void FlipShape(inout HitInfo hit)
{
    hit.shape.xw = -hit.shape.xw;
}

// Ray types, indexing both the hit group and the miss shader in the shader table, so PathTracerPass
// must add its records in this order.
static const uint STANDARD_RAY_INDEX = 0;
static const uint SHADOW_RAY_INDEX = 1;

struct ShadowRayPayload
{
    float3 visibility;
};

// Packed like a G-buffer to keep it in registers: 48 bytes.
struct HitPayload
{
    float hitT;
    uint materialId;
    uint emitter;
    uint Ng; // octahedral snorm16x2
    uint N;  // octahedral snorm16x2
    uint T;  // octahedral snorm16x2
    float2 uv;
    uint2 uvGradients; // half4
    uint2 shape; // half4: HitInfo::shape; on the way in, x nonzero to compute it
};

static const uint TANGENT_SIGN_BIT = 0x80000000u;
static const uint PAYLOAD_MATERIAL_MASK = 0xFFFFu;
static const uint PAYLOAD_LIFT_SHIFT = 16u;

uint PayloadMaterialIndex(uint materialId)
{
    return materialId & PAYLOAD_MATERIAL_MASK;
}

float SmoothSurfaceLift(float3 bary, float3 e1, float3 e2, float3 N0, float3 N1, float3 N2, float3 n, float3 Ng)
{
    // Vertex 0 at the origin.
    float3 P = e1 * bary.y + e2 * bary.z;
    float u = bary.x;
    float v = bary.y;
    float a = dot(N2 - N0, -e2);
    float b = dot(N2 - N1, e1 - e2);
    float c = dot(N1 - N0, e1);
    float h = a * u * (u - 1.0) + (a + b + c) * u * v + b * v * (v - 1.0);
    if (dot(n, Ng) > 0.0)
    {
        float h0 = max(max(dot(e1, N0), dot(e2, N0)), 0.0);
        float h1 = max(max(dot(-e1, N1), dot(e2 - e1, N1)), 0.0);
        float h2 = max(max(dot(-e2, N2), dot(e1 - e2, N2)), 0.0);
        h0 = max(dot(-P, N0) + h0, 0.0);
        h1 = max(dot(e1 - P, N1) + h1, 0.0);
        h2 = max(dot(e2 - P, N2) + h2, 0.0);
        h = max(min(min(h0, h1), h2), h * 0.5);
    }
    else
    {
        float h0 = max(max(dot(-e1, N0), dot(-e2, N0)), 0.0);
        float h1 = max(max(dot(e1, N1), dot(e1 - e2, N1)), 0.0);
        float h2 = max(max(dot(e2, N2), dot(e2 - e1, N2)), 0.0);
        h0 = max(dot(P, N0) + h0, 0.0);
        h1 = max(dot(P - e1, N1) + h1, 0.0);
        h2 = max(dot(P - e2, N2) + h2, 0.0);
        h = min(-min(min(h0, h1), h2), h * 0.5);
    }
    // Along n on Ng's side: h >= 0 above, h <= 0 when n faces away. NaN from degenerate normals gives 0.
    float lift = abs(h);
    return lift < HALF_MAX ? lift : 0.0;
}

static const float SMOOTH_LIFT_CUTOFF = 0.1;

float3 SmoothReflectionOrigin(HitInfo hit, float3 wo, float3 L)
{
    float3 Ng = dot(hit.Ng, wo) >= 0.0 ? hit.Ng : -hit.Ng;
    float3 n = dot(hit.N, Ng) >= 0.0 ? hit.N : -hit.N;
    float NgL = dot(Ng, L);
    float NL = abs(dot(n, L));
    float amount = NL < SMOOTH_LIFT_CUTOFF ? saturate(2.0 - (NgL + NL) / SMOOTH_LIFT_CUTOFF)
                                           : saturate(1.0 - NgL / SMOOTH_LIFT_CUTOFF);
    return OffsetRay(hit.position + n * (hit.smoothLift * amount), Ng);
}

// First bounce whose hits the SER pipeline reorders. Camera rays leave raygen in screen order, so
// their hits are already about as coherent as reordering would make them.
static const uint SER_START_BOUNCE = 1;

#ifdef OPACITY_MICROMAPS
// Ray queries that may meet opacity micromaps need RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS, which Slang's
// RayQuery has no template parameter for. Calling this (anywhere in the program) puts a prelude in the
// generated HLSL that turns every RayQuery<flags> into RayQuery<flags, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS>.
// Shader model 6.9.
void AllowOpacityMicromapsInRayQueries()
{
    __target_switch
    {
    case hlsl:
        __requirePrelude(R"(
template<uint RayFlags> using RayQueryAllowingOpacityMicromaps = RayQuery<RayFlags, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS>;
#define RayQuery RayQueryAllowingOpacityMicromaps
)");
        __intrinsic_asm "";
    }
}
#endif

// Not a ray flag, but carried with them: the ray traces the TLAS whose masks read their micromaps in 2-state
// mode (FrameConstants::opacityMicromap2StateScene). SceneFor picks the TLAS and RayFlags clears it.
static const uint RAY_SCENE_OMM_2_STATE = 0x80000000u;

// Flags for the rays from a path vertex at this bounce: past g_Frame.opacityMicromap2StateBounce the masks
// read their micromaps in 2-state mode and skip the alpha test.
uint OpacityMicromapRayFlags(uint bounce)
{
#ifdef OPACITY_MICROMAPS
    return bounce >= g_Frame.opacityMicromap2StateBounce ? RAY_SCENE_OMM_2_STATE : RAY_FLAG_NONE;
#else
    return RAY_FLAG_NONE;
#endif
}

RaytracingAccelerationStructure SceneFor(uint rayFlags)
{
#ifdef OPACITY_MICROMAPS
    if ((rayFlags & RAY_SCENE_OMM_2_STATE) != 0)
        return ResourceDescriptorHeap[g_Frame.opacityMicromap2StateScene];
#endif
    return g_Scene;
}

uint RayFlags(uint rayFlags)
{
    return rayFlags & ~RAY_SCENE_OMM_2_STATE;
}

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

// needCurvature: the hit feeds the DLSS-RR guides, the only users of its shape. Returns the
// payload still packed (see HitDistance and UnpackHit), the smaller form to keep across other rays.
HitPayload TraceClosest(float3 origin, float3 dir, bool reorder, bool needCurvature, uint rayFlags)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = FLT_MAX;

    HitPayload payload = (HitPayload)0;
    payload.hitT = -1.0;
    payload.shape = uint2(needCurvature ? 1u : 0u, 0u);
    // Masked and pass-through geometry runs AnyHit; everything else is flagged opaque in its BLAS.
#ifdef SER
    // DXR 1.2 SER: traverse, regroup threads by hit/miss and material, then shade. The threads stay in
    // the new order for the shading in TracePath, which is where the material divergence is.
    HitObject hitObject = HitObject::TraceRay(SceneFor(rayFlags), RayFlags(rayFlags), 0xFF, STANDARD_RAY_INDEX, 0,
                                              STANDARD_RAY_INDEX, ray, payload);
    if (reorder)
    {
        uint materialHint = hitObject.IsHit() ? LoadPrimitive(hitObject.GetInstanceID()).materialIndex : 0;
        ReorderThread(hitObject, materialHint, g_Frame.serCoherenceHintBits);
    }
    // A fresh payload: only AnyHit read the traced one, so it need not survive the reorder.
    payload = (HitPayload)0;
    payload.hitT = -1.0;
    payload.shape = uint2(needCurvature ? 1u : 0u, 0u);
    HitObject::Invoke(hitObject, payload);
#else
    TraceRay(SceneFor(rayFlags), RayFlags(rayFlags), 0xFF, STANDARD_RAY_INDEX, 0, STANDARD_RAY_INDEX, ray, payload);
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
    hit.materialId = PayloadMaterialIndex(payload.materialId);
    hit.smoothLift = f16tof32((payload.materialId >> PAYLOAD_LIFT_SHIFT) & 0x7FFFu);
    hit.emitter = payload.emitter;
#ifdef DLSS_RR
    hit.shape = float4(UnpackHalf2(payload.shape.x), UnpackHalf2(payload.shape.y));
#else
    hit.shape = float4(0.0, 0.0, 0.0, 0.0);
#endif
    hit.emission = float3(0.0, 0.0, 0.0);
    hit.Ns = hit.N; // LoadMaterial applies the normal map
    return hit;
}
