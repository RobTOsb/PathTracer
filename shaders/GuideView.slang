// Debug view of one DLSS-RR guide in place of the image (GuideViewPass), or of how much it changed since last
// frame along the motion vectors, which is where DLSS-RR finds its history does not match.
#include "PathTracerInterop.h"

ConstantBuffer<GuideViewConstants> g_PushConstants : register(b0);
ConstantBuffer<FrameConstants> g_Frame : register(b1); // the path tracer's, for this frame's jitter and cameras
RWTexture2D<float4> g_Guide : register(u0);
RWTexture2D<float2> g_MotionVectors : register(u1); // render pixels, current to previous, +y down
RWTexture2D<float4> g_History : register(u2); // last frame's GuideValue, render resolution
RWTexture2D<float4> g_NewHistory : register(u3); // this frame's
RWTexture2D<float4> g_Output : register(u4);
RWStructuredBuffer<GpuCameraState> g_CameraState : register(u5); // the path tracer's, per render pixel

static const uint CAMERA_GUIDES_HOLD = 4; // as PathTracer/CameraState.slang

// The selected guide as the history keeps it.
float4 GuideValue(uint2 pixel)
{
    float4 g = g_Guide[pixel];
    switch ((GuideView)g_PushConstants.guide)
    {
    case GUIDE_VIEW_ROUGHNESS:
        return float4(g.w, 0.0, 0.0, 0.0);
    case GUIDE_VIEW_DEPTH:
        return float4(g.x, 0.0, 0.0, 0.0);
    case GUIDE_VIEW_MOTION_VECTORS:
    case GUIDE_VIEW_SPECULAR_MOTION_VECTORS:
        return float4(g.xy, 0.0, 0.0);
    default:
        return float4(g.xyz, 0.0);
    }
}

// The history between texel centres, q in texels.
float4 LoadHistory(float2 q)
{
    int2 base = int2(floor(q));
    float2 f = q - float2(base);
    int2 last = int2(g_PushConstants.renderSize) - 1;
    float4 a = g_History[clamp(base, 0, last)];
    float4 b = g_History[clamp(base + int2(1, 0), 0, last)];
    float4 c = g_History[clamp(base + int2(0, 1), 0, last)];
    float4 d = g_History[clamp(base + int2(1, 1), 0, last)];
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// The guide's point in the world, from the hardware depth of the jittered sample it was traced through.
float3 GuideWorldPosition(float2 sample, float depth)
{
    float2 ndc = float2(sample.x / g_PushConstants.renderSize.x * 2.0 - 1.0, 1.0 - sample.y / g_PushConstants.renderSize.y * 2.0);
    float4 view = mul(float4(ndc, depth, 1.0), g_Frame.invProj);
    float4 world = mul(view, g_Frame.invView);
    return world.xyz / world.w;
}

// How far apart two values of the guide are, 1 drawing at full heat with a gain of 1.
float GuideChange(float4 current, float4 previous, float expectedDepth)
{
    switch ((GuideView)g_PushConstants.guide)
    {
    case GUIDE_VIEW_NORMAL:
        return length(current.xyz - previous.xyz) * 2.0;
    case GUIDE_VIEW_ROUGHNESS:
        return abs(current.x - previous.x) * 4.0;
    case GUIDE_VIEW_DEPTH:
        // Reversed Z with an infinite far plane is proportional to 1 / distance, so this is the relative error of
        // last frame's distance there, 10% at full heat.
        if (current.x <= 0.0 || previous.x <= 0.0)
            return current.x == previous.x ? 0.0 : 1.0;
        return abs(expectedDepth / previous.x - 1.0) * 10.0;
    case GUIDE_VIEW_MOTION_VECTORS:
    case GUIDE_VIEW_SPECULAR_MOTION_VECTORS:
        return length(current.xy - previous.xy) * 0.5; // 2 render pixels at full heat
    default:
        float3 d = abs(current.xyz - previous.xyz);
        return max(d.x, max(d.y, d.z)) * 4.0;
    }
}

// Black through red and yellow to white.
float3 Heat(float t)
{
    t = saturate(t);
    return saturate(float3(3.0 * t, 3.0 * t - 1.0, 3.0 * t - 2.0));
}

// The guide as a display colour, sRGB encoded.
float3 GuideColour(float4 value)
{
    float gain = g_PushConstants.gain;
    switch ((GuideView)g_PushConstants.guide)
    {
    case GUIDE_VIEW_NORMAL:
        return value.xyz * 0.5 + 0.5;
    case GUIDE_VIEW_ROUGHNESS:
        return value.xxx;
    case GUIDE_VIEW_DEPTH:
        // Proportional to 1 / distance, so near is bright.
        return saturate(sqrt(value.x * gain * 10.0)).xxx;
    case GUIDE_VIEW_MOTION_VECTORS:
    case GUIDE_VIEW_SPECULAR_MOTION_VECTORS:
        return float3(saturate(0.5 + value.xy * gain * 0.05), 0.5); // 10 render pixels to either edge
    default:
        return pow(saturate(value.xyz), 1.0 / 2.2);
    }
}

float SRGBToLinear(float c)
{
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

// Tonemap.slang's HDR10 encoding of a display colour at paper white.
float3 EncodeHDR10(float3 encoded)
{
    float3 colour = float3(SRGBToLinear(encoded.r), SRGBToLinear(encoded.g), SRGBToLinear(encoded.b));
    const float3x3 rec709ToRec2020 = float3x3(
        0.6274040, 0.3292820, 0.0433136,
        0.0690970, 0.9195400, 0.0113612,
        0.0163916, 0.0880132, 0.8955950);
    float3 nits = mul(rec709ToRec2020, colour * g_PushConstants.paperWhiteNits);
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    float3 y = pow(saturate(nits / 10000.0), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint2 outputSize = g_PushConstants.outputSize;
    if (id.x >= outputSize.x || id.y >= outputSize.y)
        return;

    // Nearest, so single render pixels stay visible. Every render pixel is some output pixel's, and those that
    // share one all write the same history.
    uint2 renderSize = g_PushConstants.renderSize;
    uint2 pixel = min(id.xy * renderSize / outputSize, renderSize - 1);
    float4 value = GuideValue(pixel);
    g_NewHistory[pixel] = value;

    float3 colour;
    if (g_PushConstants.change == 0)
    {
        colour = GuideColour(value);
    }
    else if (g_PushConstants.historyValid == 0)
    {
        colour = 0.0;
    }
    else
    {
        // This pixel's sample sat at pixel + 0.5 + jitter, the unjittered motion vector takes it to where it was
        // last frame, and last frame's samples sat at their texel + 0.5 + their jitter.
        float2 sample = float2(pixel) + 0.5 + g_Frame.jitter;
        float2 previous = sample + g_MotionVectors[pixel] - 0.5 - g_PushConstants.prevJitter;
        if (any(previous < -0.5) || any(previous > float2(renderSize) - 0.5))
        {
            colour = float3(0.0, 0.0, 0.25); // came from off screen
        }
        else
        {
            // Depth changes with the camera too, so it is compared with where last frame's camera would have
            // seen this point.
            float expectedDepth = 0.0;
            if ((GuideView)g_PushConstants.guide == GUIDE_VIEW_DEPTH && value.x > 0.0)
            {
                float4 clip = mul(float4(GuideWorldPosition(sample, value.x), 1.0), g_Frame.prevViewProj);
                expectedDepth = clip.z / clip.w;
            }
            colour = Heat(GuideChange(value, LoadHistory(previous), expectedDepth) * g_PushConstants.gain);
        }
    }

    if (g_PushConstants.showHeld != 0 && (g_CameraState[pixel.y * renderSize.x + pixel.x].guides.flags & CAMERA_GUIDES_HOLD) != 0)
        colour = lerp(colour, float3(1.0, 0.0, 1.0), 0.75);

    g_Output[id.xy] = float4(g_PushConstants.hdr10 != 0 ? EncodeHDR10(colour) : colour, 1.0);
}
