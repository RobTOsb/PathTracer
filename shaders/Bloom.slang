#include "PathTracerInterop.h"

// Bloom on the HDR image before exposure and tone mapping, after Jimenez, "Next Generation Post
// Processing in Call of Duty: Advanced Warfare" (SIGGRAPH 2014). Downsample filters the image into
// a chain of half-resolution levels, Upsample adds each level into the one above it, and Composite
// blends the sum of all the levels into the image.

ConstantBuffer<BloomConstants> g_PushConstants : register(b0);
Texture2D<float4> g_Input : register(t0);
Texture2D<float4> g_Bloom : register(t1); // Composite: bloom level 0
RWTexture2D<float4> g_Output : register(u0);
RWStructuredBuffer<float> g_Exposure : register(u1); // Downsample: [0] the exposure, see AutoExposurePass
SamplerState g_LinearClamp : register(s2);

float Luma(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

bool OutsideOutput(uint2 pixel)
{
    return pixel.x >= g_PushConstants.width || pixel.y >= g_PushConstants.height;
}

float2 OutputUV(uint2 pixel)
{
    return (float2(pixel) + 0.5) * g_PushConstants.outputTexelSize;
}

// Whether the camera looks at the sun disc (or the margin around it) through uv.
bool InSunMask(float2 uv)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float3 dir = normalize(float3(ndc * g_PushConstants.tanHalfFov, 1.0));
    return 1.0 - dot(dir, g_PushConstants.sunDirectionView) <= g_PushConstants.sunMaskOneMinusCos;
}

// One bright NaN or infinity from the path tracer would otherwise spread over the whole chain.
float3 SampleInput(float2 uv)
{
    if (g_PushConstants.sunMaskOneMinusCos > 0.0 && InSunMask(uv))
        return 0.0;

    float3 c = g_Input.SampleLevel(g_LinearClamp, uv, 0).rgb;
    return all(isfinite(c)) ? max(c, 0.0) : 0.0;
}

// Karis's weighted average, which keeps a single firefly from blooming into a flickering blob.
float KarisWeight(float3 box)
{
    return 1.0 / (1.0 + Luma(box) * g_Exposure[0]);
}

// 3x3 tent around uv, t apart (one texel of the texture it reads).
float3 Tent(Texture2D<float4> source, float2 uv, float2 t)
{
    float3 sum = source.SampleLevel(g_LinearClamp, uv, 0).rgb * 4.0;
    sum += (source.SampleLevel(g_LinearClamp, uv + float2(-t.x, 0.0), 0).rgb +
            source.SampleLevel(g_LinearClamp, uv + float2(t.x, 0.0), 0).rgb +
            source.SampleLevel(g_LinearClamp, uv + float2(0.0, -t.y), 0).rgb +
            source.SampleLevel(g_LinearClamp, uv + float2(0.0, t.y), 0).rgb) * 2.0;
    sum += source.SampleLevel(g_LinearClamp, uv + float2(-t.x, -t.y), 0).rgb +
           source.SampleLevel(g_LinearClamp, uv + float2(t.x, -t.y), 0).rgb +
           source.SampleLevel(g_LinearClamp, uv + float2(-t.x, t.y), 0).rgb +
           source.SampleLevel(g_LinearClamp, uv + float2(t.x, t.y), 0).rgb;
    return sum * (1.0 / 16.0);
}

// Halves g_Input into g_Output with Jimenez's 13 bilinear taps: five overlapping 4x4 boxes, the
// centre one weighing 0.5 and the four corner ones 0.125 each.
[shader("compute")]
[numthreads(8, 8, 1)]
void Downsample(uint3 id : SV_DispatchThreadID)
{
    if (OutsideOutput(id.xy))
        return;

    float2 uv = OutputUV(id.xy);
    float2 t = g_PushConstants.inputTexelSize;

    float3 a = SampleInput(uv + t * float2(-2.0, -2.0));
    float3 b = SampleInput(uv + t * float2(0.0, -2.0));
    float3 c = SampleInput(uv + t * float2(2.0, -2.0));
    float3 d = SampleInput(uv + t * float2(-2.0, 0.0));
    float3 e = SampleInput(uv);
    float3 f = SampleInput(uv + t * float2(2.0, 0.0));
    float3 g = SampleInput(uv + t * float2(-2.0, 2.0));
    float3 h = SampleInput(uv + t * float2(0.0, 2.0));
    float3 i = SampleInput(uv + t * float2(2.0, 2.0));
    float3 j = SampleInput(uv + t * float2(-1.0, -1.0));
    float3 k = SampleInput(uv + t * float2(1.0, -1.0));
    float3 l = SampleInput(uv + t * float2(-1.0, 1.0));
    float3 m = SampleInput(uv + t * float2(1.0, 1.0));

    float3 centre = (j + k + l + m) * 0.25;
    float3 topLeft = (a + b + d + e) * 0.25;
    float3 topRight = (b + c + e + f) * 0.25;
    float3 bottomLeft = (d + e + g + h) * 0.25;
    float3 bottomRight = (e + f + h + i) * 0.25;

    float3 colour;
    if (g_PushConstants.karisAverage != 0)
    {
        float wCentre = 0.5 * KarisWeight(centre);
        float wTopLeft = 0.125 * KarisWeight(topLeft);
        float wTopRight = 0.125 * KarisWeight(topRight);
        float wBottomLeft = 0.125 * KarisWeight(bottomLeft);
        float wBottomRight = 0.125 * KarisWeight(bottomRight);
        colour = (centre * wCentre + topLeft * wTopLeft + topRight * wTopRight + bottomLeft * wBottomLeft +
                  bottomRight * wBottomRight) /
                 (wCentre + wTopLeft + wTopRight + wBottomLeft + wBottomRight);
    }
    else
    {
        colour = centre * 0.5 + (topLeft + topRight + bottomLeft + bottomRight) * 0.125;
    }
    g_Output[id.xy] = float4(colour, 1.0);
}

// Adds the level below (g_Input, half the size) into g_Output, blurred by a tent.
[shader("compute")]
[numthreads(8, 8, 1)]
void Upsample(uint3 id : SV_DispatchThreadID)
{
    if (OutsideOutput(id.xy))
        return;

    float3 below = Tent(g_Input, OutputUV(id.xy), g_PushConstants.inputTexelSize);
    g_Output[id.xy] = float4(g_Output[id.xy].rgb + below, 1.0);
}

// Replaces strength of the image (g_Input, the output's size) with the mean of the bloom levels,
// so the bloom moves light around without adding any.
[shader("compute")]
[numthreads(8, 8, 1)]
void Composite(uint3 id : SV_DispatchThreadID)
{
    if (OutsideOutput(id.xy))
        return;

    float3 colour = g_Input[id.xy].rgb;
    float strength = g_PushConstants.strength;
    if (strength > 0.0)
    {
        float3 bloom = Tent(g_Bloom, OutputUV(id.xy), g_PushConstants.inputTexelSize) / g_PushConstants.levelCount;
        colour = lerp(colour, bloom, strength);
    }
    g_Output[id.xy] = float4(colour, 1.0);
}
