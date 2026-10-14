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

// Uniform random number for an any-hit candidate, hashed with the instance and primitive so each
// surface along the ray gets its own, and a repeated call for the same triangle the same one.
float SurfaceRandom(uint seed)
{
    uint3 h = Pcg3d(uint3(DispatchRaysIndex().xy, seed));
    h = Pcg3d(uint3(h.x, InstanceID(), PrimitiveIndex()));
    return float(h.x >> 8) * (1.0 / 16777216.0);
}
