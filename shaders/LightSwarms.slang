// Light swarms: millions of moving, flashing point lights to stress the two-level light tree (see
// LightSwarmConstants), written straight into the light buffer every frame. No geometry moves, so the scene's
// acceleration structures stay as they are.

#include "ShaderInterop.h"
#include "LightTreeInterop.h"

ConstantBuffer<LightSwarmConstants> g_PushConstants : register(b0);

static const float SWARM_TWO_PI = 6.28318530717959;

uint SwarmHash(uint x)
{
    // PCG output permutation of one LCG step.
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float SwarmRandom(inout uint state)
{
    state = SwarmHash(state);
    return float(state >> 8) * (1.0 / 16777216.0);
}

float3 SwarmRandom3(inout uint state)
{
    float x = SwarmRandom(state);
    float y = SwarmRandom(state);
    return float3(x, y, SwarmRandom(state));
}

float3 SwarmHue(float h)
{
    return saturate(abs(frac(h + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
}

// Light i of the swarms: swarm i / lightsPerSwarm wanders the region on a Lissajous path of its own, slow and
// smooth, and its lights orbit home positions spread through a ball of the swarm's radius, never more than a third of
// it away, so that each swarm's refit light BLAS keeps a good topology however long they move. Each light flashes on
// a clock of its own, a short pulse a few times a second, in a colour near its swarm's. No range: the light tree sees
// every light.
[shader("compute")]
[numthreads(LIGHT_SWARM_GROUP_SIZE, 1, 1)]
void AnimateLightSwarms(uint3 dispatchThread : SV_DispatchThreadID)
{
    LightSwarmConstants c = g_PushConstants;
    uint i = dispatchThread.x;
    if (i >= c.lightCount)
        return;
    uint swarm = i / c.lightsPerSwarm;
    float t = c.time;

    uint swarmState = SwarmHash(swarm ^ c.seed);
    float3 frequency = lerp(0.015, 0.05, SwarmRandom3(swarmState)) * c.speed;
    float3 phase = SwarmRandom3(swarmState) * SWARM_TWO_PI;
    float hue = SwarmRandom(swarmState);
    // In the region, so that its lights' orbits stay within the radius and a third of it beyond.
    float3 centre = c.regionMin + c.regionSize * (0.5 + 0.5 * sin(SWARM_TWO_PI * frequency * t + phase));

    uint lightState = SwarmHash(i * 0x9E3779B9u ^ SwarmHash(c.seed + 1u));
    float3 r = SwarmRandom3(lightState);
    // A uniform point in the ball.
    float cosTheta = 2.0 * r.x - 1.0;
    float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    float azimuth = SWARM_TWO_PI * r.y;
    float3 home = float3(sinTheta * cos(azimuth), cosTheta, sinTheta * sin(azimuth)) * (c.radius * pow(r.z, 1.0 / 3.0));
    float3 orbitRate = lerp(0.3, 1.2, SwarmRandom3(lightState)) * c.speed;
    float3 orbitPhase = SwarmRandom3(lightState) * SWARM_TWO_PI;
    float3 orbit = (c.radius / 3.0 / sqrt(3.0)) * sin(orbitRate * t + orbitPhase);

    // A pulse cos^12 (mean about a fifth) on the light's own clock.
    float rate = c.flashRate * lerp(0.5, 1.5, SwarmRandom(lightState));
    float flashPhase = SwarmRandom(lightState);
    float pulse = saturate(0.5 + 0.5 * cos(SWARM_TWO_PI * (rate * t + flashPhase)));
    pulse *= pulse;
    pulse *= pulse * pulse;
    pulse *= pulse;
    float3 colour = lerp(SwarmHue(frac(hue + 0.08 * (SwarmRandom(lightState) - 0.5))), 1.0, 0.25);

    GpuLight light;
    light.radiance = float4(colour * (c.intensity * pulse), asfloat(0xFFFFFFFFu));
    light.v1 = float4(centre + home + orbit, 0.0);
    light.v2 = float4(0.0, 0.0, 0.0, 0.0);
    light.v3 = float4(0.0, 0.0, 0.0, asfloat(uint(kLight_Point)));
    RWStructuredBuffer<GpuLight> lights = ResourceDescriptorHeap[c.lightsHandle];
    lights[c.firstLight + i] = light;
}
