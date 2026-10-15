// Triangle attribute fetches and ray cone texture filtering for the hit shaders.
import OpenPBRBaseSubstrate;
import PathTracer.Resources;

// A ray's cone: its width where it last met a surface (0 at a pinhole camera) and its full apex
// angle in radians, which grows the width by spread per unit of distance.
struct RayCone
{
    float width;
    float spread;
};

// Apex angle of a narrow cone (solid angle about pi (angle / 2)^2) covering the 1 / pdf of solid
// angle that a BSDF sample stands for.
float SampledConeSpread(float pdf)
{
    return 2.0 * rsqrt(PI * pdf);
}

// Vertex indices of a triangle of prim, already rebased into the shared vertex buffer.
uint3 LoadTriangle(PrimitiveDesc prim, uint primitiveIndex)
{
    uint first = prim.indexOffset + primitiveIndex * 3;
    return uint3(g_Indices[first], g_Indices[first + 1], g_Indices[first + 2]) + prim.vertexOffset;
}

float3 Barycentrics(float2 barycentrics)
{
    return float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
}

float2 InterpolateTexCoord(uint3 tri, float3 bary)
{
    return g_Vertices[tri.x].texCoord * bary.x + g_Vertices[tri.y].texCoord * bary.y + g_Vertices[tri.z].texCoord * bary.z;
}

// Cosine between a ray along dir and the plane of triangle tri, placed in the world by linearToWorld.
float FaceCosine(uint3 tri, float3x3 linearToWorld, float3 dir)
{
    float3 e1 = mul(linearToWorld, g_Vertices[tri.y].position - g_Vertices[tri.x].position);
    float3 e2 = mul(linearToWorld, g_Vertices[tri.z].position - g_Vertices[tri.x].position);
    float3 n = cross(e1, e2);
    return abs(dot(n, dir)) * rsqrt(max(dot(n, n), 1.0e-30));
}

// uv across the ellipse a unit-width ray cone makes on the triangle's plane, as SampleGrad's ddx (.xy)
// and ddy (.zw) (Akenine-Moller et al. 2021). e1, e2 are world-space edges and n = cross(e1, e2).
float4 RayConeUvGradients(float3 e1, float3 e2, float3 n, float2 uv0, float2 uv1, float2 uv2, float3 dir)
{
    float invN2 = 1.0 / max(dot(n, n), 1.0e-30);
    float3 nHat = n * sqrt(invN2);
    float cosTheta = dot(nHat, dir);
    float3 major = dir - nHat * cosTheta;
    float sin2 = dot(major, major);
    // Head on, the footprint is a circle and any axis in the plane will do.
    major = sin2 > 1.0e-8 ? major * rsqrt(sin2) : normalize(e1);
    float3 minor = cross(nHat, major);
    major /= max(abs(cosTheta), 1.0e-3);

    // Barycentric gradients in the plane: an offset a = b1 e1 + b2 e2 has b1 = dot(a, cross(e2, n)) / |n|^2
    // and b2 = dot(a, cross(n, e1)) / |n|^2.
    float3 db1 = cross(e2, n) * invN2;
    float3 db2 = cross(n, e1) * invN2;
    float2 duv1 = uv1 - uv0;
    float2 duv2 = uv2 - uv0;
    return float4(duv1 * dot(major, db1) + duv2 * dot(major, db2), duv1 * dot(minor, db1) + duv2 * dot(minor, db2));
}
