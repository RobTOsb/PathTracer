
#include "PathTracerInterop.h"

ConstantBuffer<TonemapConstants> g_PushConstants : register(b0);
RWTexture2D<float4> g_Input : register(u0);
RWTexture2D<float4> g_Output : register(u1);
RWStructuredBuffer<float> g_Exposure : register(u2); // [0] the exposure, see AutoExposurePass

float Luma(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

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

// Not saturated here so it can be used as a look that the HDR path extends.
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
    return max(colour, 0.0);
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

// RenoDX's Neutwo curve: f(x) = c*p*x / sqrt(x*x*(c*c - p*p) + c*c*p*p).
// Slope 1 at black, close to linear through the mid-tones, and f(clip) = peak exactly.
float Neutwo(float x, float peak, float clip)
{
    clip = max(clip, peak);
    float cc = clip * clip;
    float pp = peak * peak;
    return min(clip * peak * x * rsqrt(x * x * (cc - pp) + cc * pp), peak);
}

// Tonemaps luminance only and rescales RGB, so hue and saturation survive the curve.
// A colour whose luminance fits but whose brightest channel does not is then desaturated
// towards white at constant luminance, which is how bright coloured lights blow out.
float3 ToneMapNeutral(float3 colour, float peak, float clip)
{
    float y = Luma(colour);
    if (y <= 0.0)
        return 0.0;

    float yOut = Neutwo(y, peak, clip);
    colour *= yOut / y;

    float maxChannel = max(colour.r, max(colour.g, colour.b));
    if (maxChannel > peak)
    {
        float t = (peak - yOut) / max(maxChannel - yOut, 1e-6);
        colour = lerp(yOut.xxx, colour, saturate(t));
    }
    return colour;
}

float3 ApplySDRLook(float3 colour, ToneMappingOperator op, float clip)
{
    switch (op)
    {
        case TONEMAP_REINHARD:
            return Reinhard(colour);
        case TONEMAP_KHRONOS_PBR_NEUTRAL:
            return KhronosPBRNeutral(colour);
        case TONEMAP_ACES_FITTED:
            return ACESFitted(colour);
        case TONEMAP_NEUTWO:
            return ToneMapNeutral(colour, 1.0, clip);
        case TONEMAP_OFF:
        default:
            return colour;
    }
}

// RenoDX's UpgradeToneMap bridge. Mid-tones take the SDR look's luminance and colour, the
// luminance the neutral SDR curve compressed away is added back on top, and the result is
// scene-referred again so the HDR curve can map it to the display.
float3 ApplySDRLookToScene(float3 scene, float3 sdrNeutral, float3 sdrLook)
{
    float yLook = Luma(sdrLook);
    if (yLook <= 0.0)
        return 0.0;

    float yNew = yLook + max(Luma(scene) - Luma(sdrNeutral), 0.0);
    return sdrLook * (yNew / yLook);
}

// Input must be non-negative; values above 1 follow the same curve.
float3 LinearToSRGBUnclamped(float3 c)
{
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

float SRGBToLinear(float c)
{
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float3 SRGBToLinearUnclamped(float3 c)
{
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}

float3 LinearToSRGB(float3 c)
{
    return LinearToSRGBUnclamped(saturate(c));
}

// Jarzynski and Olano, "Hash Functions for GPU Rendering".
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

// Zero-mean, unit-variance uniform value per channel at a lattice point.
float3 GrainLattice(int2 cell, uint seed)
{
    uint3 h = Pcg3d(uint3(asuint(cell), seed));
    return (float3(h >> 8u) * (2.0 / 16777216.0) - 1.0) * 1.7320508;
}

float4 CubicBSplineWeights(float t)
{
    float t2 = t * t;
    float t3 = t2 * t;
    float s = 1.0 - t;
    return float4(s * s * s, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
}

// White noise on a lattice, smoothed with a cubic B-spline into soft, round grains about one
// lattice cell across. The weighted sum of 16 uniforms is close to Gaussian.
float3 GrainOctave(float2 p, uint seed)
{
    float2 cell = floor(p);
    float4 wx = CubicBSplineWeights(p.x - cell.x);
    float4 wy = CubicBSplineWeights(p.y - cell.y);
    int2 base = int2(cell) - 1;

    float3 sum = 0.0;
    [unroll]
    for (int y = 0; y < 4; ++y)
    {
        [unroll]
        for (int x = 0; x < 4; ++x)
            sum += (wx[x] * wy[y]) * GrainLattice(base + int2(x, y), seed);
    }
    // The variance of the sum depends on where p sits in its cell; dividing it out stops the grain
    // strength from following the lattice.
    return sum * rsqrt(dot(wx, wx) * dot(wy, wy));
}

// Unit-variance grain per channel. Fine grain plus a coarser clumping octave, each on a rotated
// lattice so neither shows the screen axes.
float3 FilmGrain(uint2 pixel)
{
    const float2x2 rotateFine = float2x2(0.8776, -0.4794, 0.4794, 0.8776); // 0.5 rad
    const float2x2 rotateCoarse = float2x2(-0.3233, -0.9463, 0.9463, -0.3233); // 1.9 rad
    float2 p = float2(pixel) + 0.5;
    float size = max(g_PushConstants.grainSize, 1.0);
    uint seed = g_PushConstants.grainSeed;

    // 0.8^2 + 0.6^2 = 1 keeps unit variance.
    float3 n = 0.8 * GrainOctave(mul(rotateFine, p) / size, seed) +
               0.6 * GrainOctave(mul(rotateCoarse, p) / (size * 2.3), seed ^ 0x9E3779B9u);

    // Blend from one shared (luminance) grain to independent dye-layer grain, then renormalise:
    // the channels are correlated with the shared grain by 1/sqrt(3).
    float c = saturate(g_PushConstants.grainColour);
    float mono = (n.r + n.g + n.b) * 0.57735027;
    float variance = (1.0 - c) * (1.0 - c) + c * c + 2.0 * c * (1.0 - c) * 0.57735027;
    return lerp(mono.xxx, n, c) * rsqrt(variance);
}

// Added to the display-encoded value, where equal steps look equally strong, as grain in a print
// does. The mid-tone bell fades it out at black and white so blacks are not lifted and whites stay
// clean; values above paper white (HDR) get none.
float3 ApplyFilmGrain(float3 encoded, uint2 pixel)
{
    if (g_PushConstants.grainStrength <= 0.0)
        return encoded;

    float v = saturate(Luma(encoded));
    float weight = pow(max(4.0 * v * (1.0 - v), 1e-6), g_PushConstants.grainResponse);
    return max(encoded + FilmGrain(pixel) * (g_PushConstants.grainStrength * weight), 0.0);
}

// SDR monitors decode with a pure 2.2 power rather than the piecewise sRGB curve, so that is
// what people are used to seeing. Re-encoding with sRGB and decoding with 2.2 reproduces it in
// HDR; without it shadows look lifted compared with the SDR output. This is the inverse of that
// re-encode and decode.
float GammaCorrectInverse(float c)
{
    return SRGBToLinear(pow(c, 1.0 / 2.2));
}

// Rec.709 primaries => Rec.2020 primaries (linear).
float3 Rec709ToRec2020(float3 c)
{
    const float3x3 m = float3x3(
        0.6274040, 0.3292820, 0.0433136,
        0.0690970, 0.9195400, 0.0113612,
        0.0163916, 0.0880132, 0.8955950);
    return mul(m, c);
}

// SMPTE ST 2084 inverse EOTF. Input is absolute luminance in nits.
float3 LinearToPQ(float3 nits)
{
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
    if (id.x >= g_PushConstants.width || id.y >= g_PushConstants.height)
        return;

    float3 colour = max(g_Input[id.xy].rgb, 0.0) * g_Exposure[0];
    ToneMappingOperator op = (ToneMappingOperator)g_PushConstants.toneMapOperator;
    float clip = g_PushConstants.whiteClip;

    if (g_PushConstants.hdr10 != 0)
    {
        // Scene 1.0 is paper white, so the peak is expressed relative to it.
        float peak = g_PushConstants.peakNits / g_PushConstants.paperWhiteNits;
        bool gammaCorrect = g_PushConstants.hdrGammaCorrection != 0;

        // Tonemap to the value that lands exactly on peakNits once gamma correction is applied.
        float tonemapPeak = gammaCorrect ? GammaCorrectInverse(peak) : peak;

        if (op == TONEMAP_OFF)
        {
            colour = min(colour, tonemapPeak);
        }
        else
        {
            if (op != TONEMAP_NEUTWO)
            {
                float3 sdrNeutral = ToneMapNeutral(colour, 1.0, clip);
                float3 sdrLook = ApplySDRLook(colour, op, clip);
                colour = ApplySDRLookToScene(colour, sdrNeutral, sdrLook);
            }
            colour = ToneMapNeutral(colour, tonemapPeak, clip);
        }

        // The sRGB encoding of the HDR colour is what SDR would have output below paper white,
        // so the grain matches SDR there.
        float3 encoded = ApplyFilmGrain(LinearToSRGBUnclamped(colour), id.xy);
        colour = gammaCorrect ? pow(encoded, 2.2) : SRGBToLinearUnclamped(encoded);

        float3 nits = Rec709ToRec2020(colour * g_PushConstants.paperWhiteNits);
        g_Output[id.xy] = float4(LinearToPQ(nits), 1.0);
        return;
    }

    colour = ApplySDRLook(colour, op, clip);
    float3 encoded = ApplyFilmGrain(LinearToSRGB(colour), id.xy);
    g_Output[id.xy] = float4(saturate(encoded), 1.0);
}
