// Volumetric clouds: a layer of noise-shaped cloud inside the atmosphere of SkyAtmosphere.slang, shared by the
// kernels in CloudTextures.slang and the path tracer. Distances are in km with the planet centre at the origin
// and +Y up, as in SkyAtmosphere.slang. Nothing here binds resources: the textures are the descriptor heap
// indices in CloudParameters and the samplers are the caller's.

import SkyAtmosphere;

// The scene's units are metres.
static const float kKmPerSceneUnit = 0.001;

// Noise frequencies per km. The weather map tiles every 20 km, the shape noise every 10 km, blended with a
// copy at an unrelated frequency to hide the repeats.
static const float kWeatherFrequency = 0.05;
static const float kShapeFrequency = 0.1;
static const float kShapeFrequency2 = 0.173;
static const float kCurlFrequency = 0.5;
static const float kDetailFrequency = 1.2;
// How far the curl noise pushes the detail noise around, in detail noise tiles per unit of curl.
static const float kCurlDistortion = 20.0;

// Distance between the steps of MarchClouds, which takes fewer where the layer is crossed in less.
static const float kCloudStepLength = 0.2;
// Steps towards the sun from a lit point, each kSunStepGrowth times longer, which reaches ~8 km.
static const uint kSunSteps = 6;
static const float kSunFirstStep = 0.3;
static const float kSunStepGrowth = 1.6;

// The cloud parameters and the samplers its textures are read with.
struct CloudLayer
{
    CloudParameters c;
    SamplerState linearWrap;
    SamplerState linearClamp;
};

// The planet-centred position of a scene position, lifted onto the ground from below it, where rays would
// otherwise cross the planet to the clouds on its far side.
float3 CloudSpacePosition(CloudParameters c, float3 scenePosition)
{
    float3 pos = scenePosition * kKmPerSceneUnit + float3(0.0, c.sceneHeight, 0.0);
    float r = length(pos);
    float minR = c.planetRadius + kPlanetRadiusOffset;
    return r < minR ? pos * (minR / r) : pos;
}

// Shadow map coordinates of a scene position: where the ray towards the sun crosses the shadow map's plane.
float2 CloudShadowMapUv(CloudParameters c, float3 scenePosition)
{
    float2 xz = scenePosition.xz - c.sunDirection.xz * ((scenePosition.y - c.shadowMapPlane) / max(c.sunDirection.y, 0.01));
    return (xz - c.shadowMapOrigin) * c.shadowMapInvSize;
}

// The scene position of shadow map coordinates, on the plane.
float3 CloudShadowMapPosition(CloudParameters c, float2 uv)
{
    float2 xz = c.shadowMapOrigin + uv / c.shadowMapInvSize;
    return float3(xz.x, c.shadowMapPlane, xz.y);
}

// Distances along a ray from ro to the near and far intersections with the sphere of radius r around the
// planet centre. False if it misses the sphere or the sphere is behind it.
bool RaySphere(float3 ro, float3 rd, float r, out float t0, out float t1)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float discriminant = b * b - c;
    t0 = t1 = -1.0;
    if (discriminant < 0.0)
        return false;
    float s = sqrt(discriminant);
    t0 = -b - s;
    t1 = -b + s;
    return t1 > 0.0;
}

// The stretch of a ray from origin that lies in the cloud layer: from below, from the base up to the top; from
// inside, up to the base or the top; from above, from the top down to the base or out of the top again. Further
// crossings are left out, and so is any ray that meets the ground first.
bool CloudLayerInterval(CloudParameters c, float3 origin, float3 dir, out float tEnter, out float tExit)
{
    tEnter = tExit = 0.0;
    float i0, i1, o0, o1;
    bool hitBase = RaySphere(origin, dir, c.bottomRadius, i0, i1);
    if (!RaySphere(origin, dir, c.topRadius, o0, o1))
        return false;

    float r = length(origin);
    if (r < c.bottomRadius)
    {
        tEnter = hitBase ? max(i1, 0.0) : 0.0;
        tExit = o1;
    }
    else
    {
        tEnter = r <= c.topRadius ? 0.0 : o0;
        tExit = hitBase && i0 > 0.0 ? i0 : o1;
    }

    float g0, g1;
    if (RaySphere(origin, dir, c.planetRadius, g0, g1) && g0 > 0.0)
        tExit = min(tExit, g0);
    return tExit > tEnter;
}

float Remap(float value, float inLow, float inHigh, float outLow, float outHigh)
{
    return outLow + saturate((value - inLow) / (inHigh - inLow)) * (outHigh - outLow);
}

// 0 at the cloud base, 1 at the top of the layer.
float CloudHeightFraction(CloudParameters c, float3 pos)
{
    return saturate((length(pos) - c.bottomRadius) / (c.topRadius - c.bottomRadius));
}

// How the density varies with the height fraction for a cloud type: from flat-topped stratus in the lowest
// 30% of the layer (0) to cumulus up to its top (1), both with a flat base.
float CloudHeightProfile(float h, float cloudType)
{
    float top = lerp(0.3, 1.0, cloudType);
    return saturate(Remap(h, 0.0, 0.07, 0.0, 1.0)) * saturate(Remap(h, 0.6 * top, top, 1.0, 0.0));
}

// Density at pos (0 to 1), h being its height fraction. The base shape is the weather map's coverage and cloud
// type carving the shape noise, and with detail its edges are eroded by the detail noise, swirled by curl noise:
// wispy at the base of the layer, billowing above.
float CloudDensity(CloudLayer layer, float3 pos, float h, bool detail)
{
    CloudParameters c = layer.c;
    Texture2D<float2> weatherMap = ResourceDescriptorHeap[c.weatherMap];
    Texture3D<float4> shapeNoise = ResourceDescriptorHeap[c.shapeNoise];
    Texture3D<float4> detailNoise = ResourceDescriptorHeap[c.detailNoise];

    // Relative to the ground below the planet centre, and carried along by the wind.
    float3 p = float3(pos.x - c.windOffset.x, pos.y - c.planetRadius, pos.z - c.windOffset.y);

    float2 weather = weatherMap.SampleLevel(layer.linearWrap, p.xz * kWeatherFrequency, 0.0);
    float coverage = saturate(weather.r + c.coverage);

    // .r Perlin-Worley, .gba three octaves of Worley noise.
    float4 shape = 0.5 * (shapeNoise.SampleLevel(layer.linearWrap, p * kShapeFrequency, 0.0) +
                          shapeNoise.SampleLevel(layer.linearWrap, p * kShapeFrequency2 + float3(13.7, 5.3, 9.1), 0.0));
    float worleyFbm = dot(shape.gba, float3(0.625, 0.25, 0.125));
    float fbm = shape.r * 0.625 + worleyFbm * 0.375;

    float base = saturate(Remap(fbm * CloudHeightProfile(h, weather.g), 1.0 - coverage, 1.0, 0.0, 1.0)) * coverage;
    if (base <= 0.0 || !detail)
        return base;

    // .rgb curl noise, .a Worley erosion.
    float3 curl = detailNoise.SampleLevel(layer.linearWrap, p * kCurlFrequency, 0.0).rgb - 0.5;
    float erosion = detailNoise.SampleLevel(layer.linearWrap, p * kDetailFrequency + curl * kCurlDistortion, 0.0).a;
    erosion = lerp(erosion, 1.0 - erosion, saturate(h * 3.0));
    return saturate(Remap(base, erosion * 0.2, 1.0, 0.0, 1.0));
}

float HenyeyGreenstein(float g, float cosTheta)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * kAtmospherePi * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

// A strong forward lobe for the silver lining with some back scattering, both scaled by gScale. cosTheta is
// 1 looking straight into the sun.
float CloudPhase(float cosTheta, float gScale)
{
    return lerp(HenyeyGreenstein(-0.3 * gScale, cosTheta), HenyeyGreenstein(0.8 * gScale, cosTheta), 0.7);
}

// Transmittance of the cloud along a ray from pos, from steps evenly spaced through the layer.
float CloudTransmittance(CloudLayer layer, float3 pos, float3 dir, uint steps)
{
    float t0, t1;
    if (!CloudLayerInterval(layer.c, pos, dir, t0, t1))
        return 1.0;
    float dt = (t1 - t0) / steps;
    float opticalDepth = 0.0;
    for (uint i = 0; i < steps; ++i)
    {
        float3 p = pos + dir * (t0 + (i + 0.5) * dt);
        opticalDepth += CloudDensity(layer, p, CloudHeightFraction(layer.c, p), true);
    }
    return exp(-opticalDepth * layer.c.extinction * dt);
}

// Sunlight scattered at pos towards a viewer looking at cosTheta to the sun, per unit of scattering coefficient
// and of sun illuminance at the top of the atmosphere. The air dims it as the transmittance LUT says (and the
// planet may hide the sun), the cloud by a march of kSunSteps towards the sun, jittered by u. The cloud's
// multiple scattering is three octaves of Wrenninge et al. 2013, each scattering half as much, seeing half the
// optical depth with half the anisotropy.
float3 CloudSunlight(CloudLayer layer, float3 pos, float cosTheta, float u)
{
    CloudParameters c = layer.c;
    float r = length(pos);
    float sunZenithCos = dot(c.sunDirection, pos) / r;
    if (RaySphereIntersectNearest(pos, c.sunDirection, float3(0.0, 0.0, 0.0), c.planetRadius) >= 0.0)
        return float3(0.0, 0.0, 0.0);

    AtmosphereParameters atmosphere = (AtmosphereParameters)0;
    atmosphere.bottomRadius = c.planetRadius;
    atmosphere.topRadius = c.atmosphereTopRadius;
    Texture2D<float4> transmittanceLut = ResourceDescriptorHeap[c.transmittanceLut];
    float3 air = transmittanceLut.SampleLevel(layer.linearClamp, TransmittanceLutUv(atmosphere, r, sunZenithCos), 0.0).rgb;

    float opticalDepth = 0.0;
    float t = 0.0;
    float dt = kSunFirstStep;
    for (uint i = 0; i < kSunSteps; ++i)
    {
        float3 p = pos + c.sunDirection * (t + u * dt);
        if (length(p) > c.topRadius)
            break;
        opticalDepth += CloudDensity(layer, p, CloudHeightFraction(c, p), true) * dt;
        t += dt;
        dt *= kSunStepGrowth;
    }
    opticalDepth *= c.extinction;

    float scattered = 0.0;
    float a = 1.0;
    [unroll]
    for (uint octave = 0; octave < 3; ++octave)
    {
        scattered += a * CloudPhase(cosTheta, a) * exp(-a * opticalDepth);
        a *= 0.5;
    }
    return air * scattered;
}

// What the clouds along a ray do to the light from beyond them.
struct CloudSegment
{
    float3 inScattered; // radiance scattered towards the ray's origin, before the air in front dims it
    float transmittance;
    float distance; // where along the ray the scattered light comes from on average, km
};

// Marches the cloud layer along a ray from pos in at most maxSteps steps, jittered by u.x. Each step scatters the
// sky's light (skyAmbient: its radiance from straight above) and the sun's (sunIlluminance at the top of the
// atmosphere). The sunlight takes a march of its own, so it is estimated at one step only, picked in proportion
// to what each step passes on to the origin with u.y, and u.z jitters its march.
CloudSegment MarchClouds(CloudLayer layer, float3 pos, float3 dir, uint maxSteps, float3 sunIlluminance,
                         float3 skyAmbient, float3 u)
{
    CloudSegment segment = { float3(0.0, 0.0, 0.0), 1.0, 0.0 };
    float t0, t1;
    if (!CloudLayerInterval(layer.c, pos, dir, t0, t1))
        return segment;

    uint steps = clamp(uint(ceil((t1 - t0) / kCloudStepLength)), 1u, max(maxSteps, 1u));
    float dt = (t1 - t0) / steps;
    // Weighted reservoir sampling of the step that gets the sunlight, with one random number rescaled after
    // each decision.
    float weightSum = 0.0;
    float uPick = u.y;
    float3 picked = pos;
    float distanceSum = 0.0;
    for (uint i = 0; i < steps; ++i)
    {
        float t = t0 + (i + u.x) * dt;
        float3 p = pos + dir * t;
        float h = CloudHeightFraction(layer.c, p);
        float density = CloudDensity(layer, p, h, true);
        if (density <= 0.0)
            continue;

        // The step's opacity, which is also the light it scatters per unit of incident radiance.
        float opacity = 1.0 - exp(-density * layer.c.extinction * dt);
        float weight = segment.transmittance * opacity;
        // About half the sphere around the point is sky, less of it deep below the cloud tops.
        segment.inScattered += weight * skyAmbient * (0.5 * lerp(0.25, 1.0, h));
        distanceSum += weight * t;

        weightSum += weight;
        float pickProbability = weight / weightSum;
        if (uPick < pickProbability)
        {
            picked = p;
            uPick /= pickProbability;
        }
        else
        {
            uPick = (uPick - pickProbability) / (1.0 - pickProbability);
        }

        segment.transmittance *= 1.0 - opacity;
        if (segment.transmittance < 0.01)
            break;
    }

    if (weightSum > 0.0)
    {
        segment.inScattered += weightSum * sunIlluminance * CloudSunlight(layer, picked, dot(dir, layer.c.sunDirection), u.z);
        segment.distance = distanceSum / weightSum;
    }
    return segment;
}
