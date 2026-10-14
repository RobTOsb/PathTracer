// General maths and sampling helpers.
import PathTracer.Resources;

static const float HALF_MAX = 65504.0;

float MaxComponent(float3 v)
{
    return max(v.x, max(v.y, v.z));
}

// The two halves of a float's bits as float16 values, low half in x (PackUVsToFloat).
float2 UnpackHalf2(float packed)
{
    uint bits = asuint(packed);
    return float2(f16tof32(bits & 0xFFFF), f16tof32(bits >> 16));
}

void BuildOrthonormalBasis(float3 n, out float3 t, out float3 b)
{
    // Duff et al. 2017
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// Uniform direction in the cone of 1 - cos(half angle) = oneMinusCosMax around axis, whose solid
// angle PDF is 1 / (2 pi oneMinusCosMax). An oneMinusCosMax of 2 covers the whole sphere.
float3 SampleUniformCone(float3 axis, float oneMinusCosMax, float2 u)
{
    float oneMinusCos = u.x * oneMinusCosMax;
    float cosTheta = 1.0 - oneMinusCos;
    float sinTheta = sqrt(max(0.0, oneMinusCos * (2.0 - oneMinusCos)));
    float phi = TWO_PI * u.y;

    float3 t, b;
    BuildOrthonormalBasis(axis, t, b);
    return normalize(t * (sinTheta * cos(phi)) + b * (sinTheta * sin(phi)) + axis * cosTheta);
}

float PowerHeuristic(float a, float b)
{
    float a2 = a * a;
    float b2 = b * b;
    return a2 + b2 > 0.0 ? a2 / (a2 + b2) : 0.0;
}

// Ray Tracing Gems chapter 6: offsets p along the geometric normal n so the new ray cannot
// self-intersect.
float3 OffsetRay(const float3 p, const float3 n)
{
    static const float origin = 1.0f / 32.0f;
    static const float float_scale = 1.0f / 65536.0f;
    static const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n);

    float3 p_i =
        float3(asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)), asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)), asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x, abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y, abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

// Offsets along whichever side of the geometric normal the new ray leaves from, so transmitted
// rays start below the surface and reflected ones above it.
float3 OffsetRay(float3 p, float3 Ng, float3 dir)
{
    return OffsetRay(p, dot(dir, Ng) < 0.0 ? -Ng : Ng);
}
