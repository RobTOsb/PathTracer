// Random numbers, hashed from the pixel and a per-frame counter.
static const uint RNG_CALLS_PER_FRAME = 256;

uint3 Pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

uint RngInit(uint frameIndex)
{
    return frameIndex * RNG_CALLS_PER_FRAME;
}

float3 Rand3(inout uint rng)
{
    uint3 h = Pcg3d(uint3(DispatchRaysIndex().xy, rng++));
    return float3(h >> 8) * (1.0 / 16777216.0);
}
