// Exposure + tone mapping + sRGB encode of the path tracer's HDR radiance into the LDR image that
// gets copied to the (UNORM, non-sRGB) back buffer.

#include "PathTracerInterop.h"

ConstantBuffer<TonemapConstants> g_PushConstants : register(b0);
RWTexture2D<float4> g_Input : register(u0);
RWTexture2D<float4> g_Output : register(u1);

float3 Reinhard(float3 x)
{
    return x / (1.0 + x);
}

// Stephen Hill's fit of the ACES RRT + ODT.
float3 RRTAndODTFit(float3 v)
{
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

float3 ACESFitted(float3 colour)
{
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    const float3x3 inputMat = float3x3(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777);
    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    const float3x3 outputMat = float3x3(
         1.60475, -0.53108, -0.07367,
        -0.10208,  1.10813, -0.00605,
        -0.00327, -0.07276,  1.07602);

    colour = mul(inputMat, colour);
    colour = RRTAndODTFit(colour);
    colour = mul(outputMat, colour);
    return saturate(colour);
}

float3 KhronosPBRNeutral(float3 colour)
{
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(colour.r, min(colour.g, colour.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    colour -= offset;

    float peak = max(colour.r, max(colour.g, colour.b));
    if (peak < startCompression)
        return colour;

    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    colour *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return lerp(colour, newPeak.xxx, g);
}

float3 LinearToSRGB(float3 c)
{
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_PushConstants.width || id.y >= g_PushConstants.height)
        return;

    float3 colour = max(g_Input[id.xy].rgb, 0.0) * g_PushConstants.exposure;

    switch ((ToneMappingOperator)g_PushConstants.toneMapOperator)
    {
        case TONEMAP_REINHARD:
            colour = Reinhard(colour);
            break;
        case TONEMAP_KHRONOS_PBR_NEUTRAL:
            colour = KhronosPBRNeutral(colour);
            break;
        case TONEMAP_OFF:
            break;
        case TONEMAP_ACES_FITTED:
        default:
            colour = ACESFitted(colour);
            break;
    }

    g_Output[id.xy] = float4(LinearToSRGB(colour), 1.0);
}
