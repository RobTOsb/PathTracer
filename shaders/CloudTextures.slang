// The cloud textures of Clouds.slang, run by SkyAtmospherePass: three tiling noise textures made once, the shadow map
// of the clouds over the scene whenever the clouds or the sun change, and the panorama of the cloudy sky every frame.

import Clouds;
import SkyAtmosphere;

ConstantBuffer<CloudParameters> g_Clouds : register(b0);
RWTexture3D<float4> g_NoiseOutput : register(u0);
RWTexture2D<float2> g_WeatherOutput : register(u1);
RWTexture2D<float> g_ShadowMapOutput : register(u2);
RWTexture3D<float> g_ShapeOutput : register(u3);
RWTexture2D<float4> g_PanoramaOutput : register(u4);
Texture2D<float4> g_SkyViewLUT : register(t0);
ConstantBuffer<CloudPanoramaConstants> g_PushConstants : register(b1);
SamplerState g_LinearClamp : register(s2);
SamplerState g_LinearWrap : register(s3);

// The cells of the noises below repeat every period cells, so the textures tile.
int3 WrapCell(int3 cell, int3 period)
{
    return ((cell % period) + period) % period;
}

uint CellHash(int3 cell)
{
    uint3 v = uint3(cell) * uint3(1597334673u, 3812015801u, 2798796415u);
    return (v.x ^ v.y ^ v.z) * 1597334673u;
}

float3 CellPoint(int3 cell)
{
    uint s = CellHash(cell);
    uint3 v = uint3(s, s ^ 3812015801u, s ^ 2798796415u) * uint3(1597334673u, 3812015801u, 2798796415u);
    return float3(v & 0xFFFFFFu) / float(0x1000000u);
}

// 1 at the cells' feature points, falling off with the distance to the nearest.
float Worley(float3 p, int3 period)
{
    int3 cell = int3(floor(p));
    float minDistance = 1.0e9;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                int3 neighbour = cell + int3(x, y, z);
                float3 featurePoint = float3(neighbour) + CellPoint(WrapCell(neighbour, period));
                minDistance = min(minDistance, length(p - featurePoint));
            }
    return saturate(1.0 - minDistance);
}

// Three octaves, each at twice the frequency and so twice the cells per tile.
float WorleyFbm(float3 p, int3 period)
{
    return Worley(p, period) * 0.625 + Worley(p * 2.0, period * 2) * 0.25 + Worley(p * 4.0, period * 4) * 0.125;
}

// Gradient noise remapped to 0 to 1.
float Perlin(float3 p, int3 period)
{
    static const float3 kGradients[12] = { float3(1, 1, 0), float3(-1, 1, 0), float3(1, -1, 0), float3(-1, -1, 0),
                                           float3(1, 0, 1), float3(-1, 0, 1), float3(1, 0, -1), float3(-1, 0, -1),
                                           float3(0, 1, 1), float3(0, -1, 1), float3(0, 1, -1), float3(0, -1, -1) };
    int3 cell = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);

    float corners[8];
    for (uint i = 0; i < 8; ++i)
    {
        int3 offset = int3(i & 1, (i >> 1) & 1, i >> 2);
        corners[i] = dot(kGradients[CellHash(WrapCell(cell + offset, period)) % 12u], f - float3(offset));
    }
    float y0 = lerp(lerp(corners[0], corners[1], u.x), lerp(corners[2], corners[3], u.x), u.y);
    float y1 = lerp(lerp(corners[4], corners[5], u.x), lerp(corners[6], corners[7], u.x), u.y);
    return lerp(y0, y1, u.z) * 0.5 + 0.5;
}

// The curl of a field of three differently scaled Perlin noises, by central differences, mapped to 0 to 1.
float3 CurlNoise(float3 p, int3 period)
{
    const float eps = 0.01;
    float3 dx = float3(eps, 0.0, 0.0);
    float3 dy = float3(0.0, eps, 0.0);
    float3 dz = float3(0.0, 0.0, eps);
    float3 py = p * 1.3;
    float3 pz = p * 1.7;
    float3 curl = float3((Perlin(p + dy, period) - Perlin(p - dy, period)) - (Perlin(p + dz, period) - Perlin(p - dz, period)),
                         (Perlin(py + dz, period) - Perlin(py - dz, period)) - (Perlin(py + dx, period) - Perlin(py - dx, period)),
                         (Perlin(pz + dx, period) - Perlin(pz - dx, period)) - (Perlin(pz + dy, period) - Perlin(pz - dy, period))) /
                  (2.0 * eps);
    return saturate(curl * 0.05 + 0.5);
}

// Noise coordinates of a texel of a size^3 texture holding four cells per side.
static const int kNoiseCells = 4;

float3 NoisePosition(uint3 texel, uint size)
{
    return (float3(texel) + 0.5) / float(size) * kNoiseCells;
}

// CLOUD_SHAPE_NOISE_SIZE^3, one channel: Perlin-Worley (Perlin noise remapped by Worley noise, billowing), 0.625 of
// it, and Worley noise at 1, 2 and 4 times the frequency, 0.375 of it in octaves of 0.625, 0.25 and 0.125. Only that
// sum is ever read, so a quarter of the bytes a texel of all four would take.
[shader("compute")]
[numthreads(4, 4, 4)]
void ShapeNoise(uint3 id: SV_DispatchThreadID)
{
    if (any(id >= kCloudShapeNoiseSize))
        return;
    int3 period = int3(kNoiseCells, kNoiseCells, kNoiseCells);
    float3 p = NoisePosition(id, kCloudShapeNoiseSize);
    float worley = WorleyFbm(p, period);
    float perlinWorley = Remap(Perlin(p, period), 1.0 - worley, 1.0, 0.0, 1.0);
    float worleyFbm = dot(float3(worley, WorleyFbm(p * 2.0, period * 2), WorleyFbm(p * 4.0, period * 4)), float3(0.625, 0.25, 0.125));
    g_ShapeOutput[id] = perlinWorley * 0.625 + worleyFbm * 0.375;
}

// CLOUD_DETAIL_NOISE_SIZE^3: .rgb curl noise, .a Worley noise.
[shader("compute")]
[numthreads(4, 4, 4)]
void DetailNoise(uint3 id: SV_DispatchThreadID)
{
    if (any(id >= kCloudDetailNoiseSize))
        return;
    int3 period = int3(kNoiseCells, kNoiseCells, kNoiseCells);
    float3 p = NoisePosition(id, kCloudDetailNoiseSize);
    g_NoiseOutput[id] = float4(CurlNoise(p, period), WorleyFbm(p, period));
}

// CLOUD_WEATHER_MAP_SIZE^2: .r coverage in large blobs, .g cloud type (0 stratus to 1 cumulus) in larger regions.
// Slices of 3D noise between cell corners, which only tile in x and y.
[shader("compute")]
[numthreads(8, 8, 1)]
void WeatherMap(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= kCloudWeatherMapSize))
        return;
    float2 p = (float2(id.xy) + 0.5) / float(kCloudWeatherMapSize);
    float coverage = saturate(Perlin(float3(p * 4.0, 0.37), int3(4, 4, 4)) * 1.4 - 0.2);
    float cloudType = saturate(Perlin(float3(p * 2.0, 0.71), int3(2, 2, 2)));
    g_WeatherOutput[id.xy] = float2(coverage, cloudType);
}

// Marched at this many points from each texel towards the sun.
static const uint kShadowMapSteps = 64;

// CLOUD_SHADOW_MAP_SIZE^2: the transmittance of the clouds towards the sun from the shadow map's plane.
[shader("compute")]
[numthreads(8, 8, 1)]
void ShadowMap(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= kCloudShadowMapSize))
        return;
    CloudLayer layer = { g_Clouds, g_LinearWrap, g_LinearClamp };
    float2 uv = (float2(id.xy) + 0.5) / float(kCloudShadowMapSize);
    float3 pos = CloudSpacePosition(g_Clouds, CloudShadowMapPosition(g_Clouds, uv));
    g_ShadowMapOutput[id.xy] = CloudTransmittance(layer, pos, g_Clouds.sunDirection, kShadowMapSteps);
}

// CLOUD_PANORAMA_SIZE^2, octahedral (CloudPanoramaDirection): the cloudy sky seen from the camera, as the path tracer's
// SkyRadiance would see it along a ray the camera sees through but for a sun of unit illuminance and without the sun
// disc, whose share the clouds let through is in .a. Each frame's march is jittered and blended in.
[shader("compute")]
[numthreads(8, 8, 1)]
void CloudPanorama(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= kCloudPanoramaSize))
        return;
    CloudLayer layer = { g_Clouds, g_LinearWrap, g_LinearClamp };
    float3 dir = CloudPanoramaDirection((float2(id.xy) + 0.5) / float(kCloudPanoramaSize));
    float viewHeight = g_PushConstants.viewHeight;
    float3 sky = g_SkyViewLUT.SampleLevel(g_LinearClamp, SkyViewLutUvFromDirection(g_Clouds.planetRadius, viewHeight, dir,
                                                                                      g_Clouds.sunDirection), 0.0).rgb;
    float3 ambient = g_SkyViewLUT.SampleLevel(g_LinearClamp, SkyViewLutUvFromDirection(g_Clouds.planetRadius, viewHeight,
                                                                                          float3(0.0, 1.0, 0.0), g_Clouds.sunDirection), 0.0).rgb;
    uint3 h = uint3(id.xy, g_PushConstants.frame) * uint3(1664525u, 1013904223u, 2654435761u);
    h ^= h.yzx >> 16;
    h *= uint3(2246822519u, 3266489917u, 668265263u);
    h ^= h.zxy >> 13;
    float3 u = float3(h >> 8) * (1.0 / 16777216.0);
    float4 cloudy = CloudySky(layer, sky, CloudSpacePosition(g_Clouds, g_PushConstants.origin), dir, g_Clouds.viewSteps,
                              float3(1.0, 1.0, 1.0), ambient, u, true);
    float blend = g_PushConstants.blend;
    g_PanoramaOutput[id.xy] = blend >= 1.0 ? cloudy : lerp(g_PanoramaOutput[id.xy], cloudy, blend);
}
