// OpenPBR base substrate for path tracing: evaluation, sampling and PDF.
//
// API
//   void           SetShadingBasis(p, T, B, N, Ns, wo)  fills the shading basis in the params
//   OpenPBRContext OpenPBRSetup(p, wo)                  per shading point setup, call once per hit
//   float3         OpenPBREval(p, ctx, wi)              returns f(wi, wo) * |dot(wi, geometryNormal)|
//   float          OpenPBRPdf(p, ctx, wi)               returns the solid angle PDF of sampling wi given wo
//   float3         OpenPBREvalPdf(p, ctx, wi, out pdf)  both of the above at once, sharing the common terms
//   BSDFSample     OpenPBRSample(p, ctx, u)             samples wi given wo, u holds three uniform numbers in [0, 1)
//
// Conventions
//   All directions passed in and returned are in world space.
//   wi points towards the light (the sampled direction), wo towards the viewer.
//   Both point away from the surface.
//   The shading basis comes in through the parameters. geometryTangent, geometryBitangent
//   and geometryNormal must be orthonormal, with normal mapping already applied and the
//   tangent already pointing along the anisotropy direction. Internally everything is
//   evaluated in the local frame where the normal is +Z and the tangent is +X.
//   The side of geometryNormal that wo lies on decides inside versus outside:
//     outside: the full base substrate applies.
//     inside:  the viewer is inside a transmissive object, only the dielectric interface applies.
//   If a normal map can tilt geometryNormal away from the viewer, bend it back when building
//   the basis, otherwise an outside hit can be treated as inside.
//   Thin-walled surfaces have no inside: face the basis towards the viewer, as for opaque ones.
//
// Sampling
//   One-sample mixture over lobes, selected with probabilities from approximate albedos.
//   The returned weight is Eval / Pdf using the full mixture PDF, so it is unbiased and can be
//   combined with light sampling through MIS using OpenPBRPdf (or OpenPBREvalPdf).
//   Specular reflection uses bounded VNDF sampling (Eto and Tokuyoshi 2023).
//   Specular transmission uses spherical cap VNDF sampling (Dupuy and Benyoub 2023), because
//   the bounded cap is only valid for reflections.
//   Diffuse and subsurface use cosine-weighted hemisphere sampling.
//   Thin-walled transmission mirrors the specular reflection lobe through the tangent plane, so it
//   samples like the reflection. At zero roughness it is a delta lobe straight through
//   (LOBE_DELTA_TRANSMISSION), which Eval and Pdf leave out; the sample's pdf is then the discrete
//   probability of picking it, and its weight ThinDeltaTransmittance / pdf.
//
// Simplifications
//   No coat, fuzz or thin film, ambient IOR is 1.
//   Thin-walled surfaces have no diffuse or subsurface transmission, and their specular transmission
//   is the single-interface lobe above (as in glTF), not a slab with two interfaces.
//   Subsurface is approximated by a Lambertian lobe under the dielectric interface.
//   transmissionColour with zero depth is applied once, on entry. With depth > 0 the renderer
//   should convert it to medium extinction instead.
//   No multiple-scattering compensation for the GGX lobes.

static const float PI = 3.14159265358979;

static const uint LOBE_NONE = 0;
static const uint LOBE_DIFFUSE = 1;
static const uint LOBE_SPECULAR_REFLECTION = 2;
static const uint LOBE_SPECULAR_TRANSMISSION = 4;
static const uint LOBE_DELTA_TRANSMISSION = 8;

struct OpenPBRBaseParams
{
    float baseWeight;
    float3 baseColour;
    float baseMetalness;
    float baseDiffuseRoughness;

    float specularWeight;
    float3 specularColour;
    float specularRoughness;
    float specularRoughnessAnisotropy;
    float specularIor;

    float transmissionWeight;
    float3 transmissionColour;
    float transmissionDepth;

    float subsurfaceWeight;
    float3 subsurfaceColour;

    bool geometryThinWalled;
    float3 geometryTangent;
    float3 geometryBitangent;
    float3 geometryNormal;
};

struct BSDFSample
{
    float3 wi;
    float3 weight;
    float pdf;
    uint lobe;
};

struct OpenPBRContext
{
    bool inside;
    bool deltaTransmission; // thin-walled and smooth, see LOBE_DELTA_TRANSMISSION
    float3 wo;
    float2 alpha;
    float eta;
    float E;
    float probSpecR;
    float probSpecT;
    float probDiffuse;

    float lambdaO;      // GGXLambda(wo, alpha)
    float boundedB;     // lower bound of the bounded cap, b in Eto and Tokuyoshi 2023, Listing 1
    float reflPdfScale; // bounded VNDF reflection PDF divided by D(h), Listing 2
    float vndfPdfScale; // spherical cap VNDF PDF divided by D(m) * dot(wo, m), equation 7
};

// Utilities

// Rec.709 luminance, only used as a heuristic for lobe selection.
float Luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float3 ToLocal(OpenPBRBaseParams p, float3 v)
{
    return float3(dot(v, p.geometryTangent), dot(v, p.geometryBitangent), dot(v, p.geometryNormal));
}

float3 ToWorld(OpenPBRBaseParams p, float3 v)
{
    return v.x * p.geometryTangent + v.y * p.geometryBitangent + v.z * p.geometryNormal;
}

float3 TransmissionTint(OpenPBRBaseParams p)
{
    return p.transmissionDepth > 0.0 ? float3(1.0, 1.0, 1.0) : p.transmissionColour;
}

// A thin wall that transmits as a delta lobe.
bool HasDeltaTransmission(OpenPBRBaseParams p)
{
    return p.geometryThinWalled && p.specularRoughness <= 0.0 && p.transmissionWeight > 0.0;
}

// GGX microfacet model

float2 GGXAlpha(float roughness, float anisotropy)
{
    float r2 = roughness * roughness;
    float oneMinusA = 1.0 - anisotropy;
    float at = r2 * sqrt(2.0 / (1.0 + oneMinusA * oneMinusA));
    float ab = oneMinusA * at;
    return max(float2(at, ab), 1.0e-4);
}

float GGXD(float3 m, float2 alpha)
{
    if (m.z <= 0.0)
        return 0.0;
    // 1 / (pi ax ay (x^2 / ax^2 + y^2 / ay^2 + z^2)^2), multiplied through by (ax ay)^4 so it
    // needs one division instead of three. alpha >= 1e-4 keeps every term in float range.
    float a = alpha.x * alpha.y;
    float q = m.x * m.x * alpha.y * alpha.y + m.y * m.y * alpha.x * alpha.x + m.z * m.z * a * a;
    return a * a * a / (PI * q * q);
}

float GGXLambda(float3 w, float2 alpha)
{
    float z2 = max(w.z * w.z, 1.0e-8);
    float a2 = (alpha.x * alpha.x * w.x * w.x + alpha.y * alpha.y * w.y * w.y) / z2;
    return 0.5 * (-1.0 + sqrt(1.0 + a2));
}

// lambdaO is GGXLambda(wo, alpha), constant per hit, so it comes from the context.
float GGXG2(float3 wi, float lambdaO, float2 alpha)
{
    return 1.0 / (1.0 + GGXLambda(wi, alpha) + lambdaO);
}

// Fresnel

// eta = transmitted side IOR / incident side IOR, cosI > 0
float FresnelDielectric(float cosI, float eta)
{
    float sin2T = (1.0 - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0)
        return 1.0;
    float cosT = sqrt(1.0 - sin2T);
    // Both ratios share one reciprocal of the product of their denominators.
    float sDenom = cosI + eta * cosT;
    float pDenom = eta * cosI + cosT;
    float invDenoms = 1.0 / (sDenom * pDenom);
    float rs = (cosI - eta * cosT) * pDenom * invDenoms;
    float rp = (eta * cosI - cosT) * sDenom * invDenoms;
    return 0.5 * (rs * rs + rp * rp);
}

// Spec equation [modulated_ior]: specularWeight scales F0 by altering the IOR ratio.
float ModulatedIORRatio(float etaS, float specularWeight)
{
    float Fs = (1.0 - etaS) / (1.0 + etaS);
    Fs *= Fs;
    if (Fs < 1.0e-8)
        return 1.0;
    float xi = clamp(specularWeight, 0.0, 1.0 / Fs);
    float e = sign(etaS - 1.0) * sqrt(xi * Fs);
    e = clamp(e, -0.9999, 0.9999);
    return (1.0 + e) / (1.0 - e);
}

// F82-tint metallic Fresnel (Kutz et al. 2021).
float3 FresnelF82Tint(float mu, float3 F0, float3 tint)
{
    const float muBar = 1.0 / 7.0;
    const float invDenom = 1.0 / (muBar * pow(1.0 - muBar, 6.0));
    // Integer powers by multiplication; pow would be a log and an exp per call.
    float m1 = 1.0 - mu;
    float m2 = m1 * m1;
    float m5 = m2 * m2 * m1;
    float3 FSchlick = F0 + (1.0 - F0) * m5;
    float3 FSchlickBar = F0 + (1.0 - F0) * pow(1.0 - muBar, 5.0);
    float3 correction = (mu * m5 * m1 * invDenom) * (1.0 - tint) * FSchlickBar;
    return saturate(FSchlick - correction);
}

// Directional albedo of the dielectric specular lobe, analytic fit from Karis (2014).
float DielectricAlbedo(float NdotV, float roughness, float F0)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return saturate(F0 * ab.x + saturate(50.0 * F0) * ab.y);
}

static const float FON_C1 = 0.5 - 2.0 / (3.0 * PI);
static const float FON_C2 = 2.0 / 3.0 - 28.0 / (15.0 * PI);

float EFONApprox(float mu, float r)
{
    float mc = 1.0 - mu;
    const float g1 = 0.0571085289;
    const float g2 = 0.491881867;
    const float g3 = -0.332181442;
    const float g4 = 0.0714429953;
    float GoverPi = mc * (g1 + mc * (g2 + mc * (g3 + mc * g4)));
    return (1.0 + r * GoverPi) / (1.0 + FON_C1 * r);
}

float3 EvalEON(float3 rho, float r, float3 wi, float3 wo)
{
    float muI = wi.z;
    float muO = wo.z;
    float s = dot(wi, wo) - muI * muO;
    float sOverT = s > 0.0 ? s / max(muI, muO) : s;
    float AF = 1.0 / (1.0 + FON_C1 * r);
    float3 fSS = (rho / PI) * AF * (1.0 + r * sOverT);

    float EFo = EFONApprox(muO, r);
    float EFi = EFONApprox(muI, r);
    float avgEF = AF * (1.0 + FON_C2 * r);
    float3 rhoMS = (rho * rho) * avgEF / (1.0 - rho * (1.0 - avgEF));
    const float eps = 1.0e-7;
    float3 fMS = (rhoMS / PI) * max(eps, 1.0 - EFo) * max(eps, 1.0 - EFi) / max(eps, 1.0 - avgEF);
    return fSS + fMS;
}

// Sampling routines

float BoundedCapK(float3 v, float2 alpha)
{
    float a = saturate(min(alpha.x, alpha.y));
    float s = 1.0 + length(v.xy);
    float a2 = a * a;
    float s2 = s * s;
    return (1.0 - a2) * s2 / (s2 + a2 * v.z * v.z);
}

float3 SampleGGXReflectionBounded(float3 v, float2 alpha, float b, float2 u)
{
    float3 vStd = normalize(float3(v.xy * alpha, v.z));
    float phi = 2.0 * PI * u.x;
    float z = mad(1.0 - u.y, 1.0 + b, -b);
    float sinTheta = sqrt(saturate(1.0 - z * z));
    float3 oStd = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
    float3 mStd = vStd + oStd;
    float3 m = normalize(float3(mStd.xy * alpha, mStd.z));
    return 2.0 * dot(v, m) * m - v;
}

float3 SampleGGXVNDF(float3 v, float2 alpha, float2 u)
{
    float3 vStd = normalize(float3(v.xy * alpha, v.z));
    float phi = 2.0 * PI * u.x;
    float z = mad(1.0 - u.y, 1.0 + vStd.z, -vStd.z);
    float sinTheta = sqrt(saturate(1.0 - z * z));
    float3 oStd = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
    float3 mStd = vStd + oStd;
    return normalize(float3(mStd.xy * alpha, mStd.z));
}

float3 SampleCosineHemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(saturate(1.0 - u.x)));
}

bool TransmissionHalfVector(float3 wi, float3 wo, float eta, out float3 h, out float wiDotH, out float woDotH)
{
    h = -(eta * wi + wo);
    wiDotH = 0.0;
    woDotH = 0.0;
    float len = length(h);
    if (len < 1.0e-6)
        return false;
    h /= len;
    if (h.z < 0.0)
        h = -h;
    wiDotH = dot(wi, h);
    woDotH = dot(wo, h);
    return woDotH > 0.0 && wiDotH < 0.0;
}

float EvalPdfGGXTransmission(float3 wi, OpenPBRContext c, out float pdf)
{
    pdf = 0.0;
    float3 wo = c.wo;
    float eta = c.eta;
    float3 h;
    float wiDotH, woDotH;
    if (!TransmissionHalfVector(wi, wo, eta, h, wiDotH, woDotH))
        return 0.0;
    float ndf = GGXD(h, c.alpha);
    float denom = eta * wiDotH + woDotH;
    float invDenom2 = 1.0 / (denom * denom);
    // Spherical cap VNDF density of h (woDotH > 0 here) times the refraction Jacobian.
    float jacobian = eta * eta * abs(wiDotH) * invDenom2;
    pdf = ndf * woDotH * c.vndfPdfScale * jacobian;
    float F = FresnelDielectric(woDotH, eta);
    return abs(wiDotH * woDotH) / abs(wi.z * wo.z) * (1.0 - F) * GGXG2(wi, c.lambdaO, c.alpha) * ndf * invDenom2;
}

// Thin-walled transmission: the reflection lobe at the mirror image of wi through the tangent plane,
// with 1 - F in place of F. pdf is that of sampling the reflection lobe towards the mirror image.
float EvalPdfGGXThinTransmission(float3 wi, OpenPBRContext c, out float pdf)
{
    float3 wr = float3(wi.xy, -wi.z);
    float3 wo = c.wo;
    float3 h = normalize(wr + wo);
    float ndf = GGXD(h, c.alpha);
    pdf = ndf * c.reflPdfScale;
    float F = FresnelDielectric(saturate(dot(wo, h)), c.eta);
    return (1.0 - F) * ndf * GGXG2(wr, c.lambdaO, c.alpha) / (4.0 * wr.z * wo.z);
}

// What a smooth thin wall lets through straight along -wo, cosTheta from its normal: the delta
// lobe's integral, so its sample weight times the probability of picking it.
float3 ThinDeltaTransmittance(OpenPBRBaseParams p, float cosTheta)
{
    float eta = ModulatedIORRatio(p.specularIor, p.specularWeight);
    float F = FresnelDielectric(saturate(cosTheta), eta);
    return (1.0 - p.baseMetalness) * p.transmissionWeight * (1.0 - F) * TransmissionTint(p);
}

OpenPBRContext OpenPBRSetup(OpenPBRBaseParams p, float3 woWorld)
{
    float3 wo = ToLocal(p, woWorld);
    OpenPBRContext c;
    c.inside = wo.z < 0.0 && !p.geometryThinWalled;
    c.deltaTransmission = p.geometryThinWalled && p.specularRoughness <= 0.0;
    c.wo = c.inside ? -wo : wo;
    c.alpha = GGXAlpha(p.specularRoughness, p.specularRoughnessAnisotropy);

    float etaS = ModulatedIORRatio(p.specularIor, p.specularWeight);
    c.eta = c.inside ? 1.0 / etaS : etaS;

    float t = length(float3(c.alpha * c.wo.xy, c.wo.z)); // length of the stretched view vector
    float k = BoundedCapK(c.wo, c.alpha);
    c.lambdaO = GGXLambda(c.wo, c.alpha);
    c.boundedB = k * c.wo.z / t;
    c.reflPdfScale = 1.0 / (2.0 * (k * c.wo.z + t));
    c.vndfPdfScale = 2.0 / (c.wo.z + t);

    if (c.inside)
    {
        c.E = 0.0;
        c.probSpecR = clamp(FresnelDielectric(c.wo.z, c.eta), 0.05, 0.95);
        c.probSpecT = 1.0 - c.probSpecR;
        c.probDiffuse = 0.0;
        return c;
    }

    float F0 = (1.0 - etaS) / (1.0 + etaS);
    F0 *= F0;
    c.E = DielectricAlbedo(c.wo.z, p.specularRoughness, F0);

    float M = p.baseMetalness;
    float T = p.transmissionWeight;
    float S = p.subsurfaceWeight;

    float3 metalF = p.specularWeight * FresnelF82Tint(c.wo.z, p.baseWeight * p.baseColour, p.specularColour);
    float3 diffuseAlbedo = lerp(p.baseWeight * p.baseColour, p.subsurfaceColour, S);

    float wSpecR = (1.0 - M) * c.E * Luminance(p.specularColour) + M * Luminance(metalF);
    float wSpecT = (1.0 - M) * T * (1.0 - c.E) * Luminance(TransmissionTint(p));
    float wDiffuse = (1.0 - M) * (1.0 - T) * (1.0 - c.E) * Luminance(diffuseAlbedo);

    float total = wSpecR + wSpecT + wDiffuse;
    float invTotal = total > 0.0 ? 1.0 / total : 0.0;
    c.probSpecR = wSpecR * invTotal;
    c.probSpecT = wSpecT * invTotal;
    c.probDiffuse = wDiffuse * invTotal;
    return c;
}

float3 OpenPBREvalPdfLocal(OpenPBRBaseParams p, OpenPBRContext c, float3 wi, out float pdf)
{
    pdf = 0.0;
    float3 wo = c.wo;
    if (c.inside)
        wi = -wi;
    if (wo.z <= 0.0 || wi.z == 0.0)
        return float3(0.0, 0.0, 0.0);

    float M = p.baseMetalness;
    float T = p.transmissionWeight;
    float S = p.subsurfaceWeight;

    if (wi.z < 0.0)
    {
        if (c.deltaTransmission)
            return float3(0.0, 0.0, 0.0);
        float transmissionPdf;
        float fT = (p.geometryThinWalled ? EvalPdfGGXThinTransmission(wi, c, transmissionPdf)
                                        : EvalPdfGGXTransmission(wi, c, transmissionPdf)) * abs(wi.z);
        pdf = c.probSpecT * transmissionPdf;
        if (c.inside)
            return fT.xxx;
        return (1.0 - M) * T * TransmissionTint(p) * fT;
    }

    float3 h = normalize(wi + wo);
    float ndf = GGXD(h, c.alpha);

    pdf = c.probSpecR * ndf * c.reflPdfScale + c.probDiffuse * wi.z / PI;

    float cosH = saturate(dot(wo, h));
    float microfacet = ndf * GGXG2(wi, c.lambdaO, c.alpha) / (4.0 * wi.z * wo.z);
    float Fd = FresnelDielectric(cosH, c.eta);

    if (c.inside)
        return (Fd * microfacet * wi.z).xxx;

    float3 fMetal = p.specularWeight * FresnelF82Tint(cosH, p.baseWeight * p.baseColour, p.specularColour) * microfacet;
    float3 fSpecR = p.specularColour * Fd * microfacet;

    // Albedo scaling with the untinted albedo, since specularColour does not affect transmission.
    float transmitted = 1.0 - c.E;
    float3 fDiffuse = transmitted * p.baseWeight * EvalEON(p.baseColour, p.baseDiffuseRoughness, wi, wo);
    float3 fSubsurface = transmitted * p.subsurfaceColour / PI;

    // Spec mix tree. The translucent slab only contributes fSpecR in the upper hemisphere.
    float3 fOpaqueBase = fSpecR + lerp(fDiffuse, fSubsurface, S);
    float3 fDielectricBase = lerp(fOpaqueBase, fSpecR, T);
    return lerp(fDielectricBase, fMetal, M) * wi.z;
}

BSDFSample OpenPBRSample(OpenPBRBaseParams p, OpenPBRContext c, float3 u)
{
    BSDFSample s;
    s.wi = float3(0.0, 0.0, 0.0);
    s.weight = float3(0.0, 0.0, 0.0);
    s.pdf = 0.0;
    s.lobe = LOBE_NONE;

    float3 wo = c.wo;
    if (wo.z <= 0.0)
        return s;

    float3 wi;
    uint lobe;

    if (u.x < c.probSpecR)
    {
        wi = SampleGGXReflectionBounded(wo, c.alpha, c.boundedB, u.yz);
        if (wi.z <= 0.0)
            return s;
        lobe = LOBE_SPECULAR_REFLECTION;
    }
    else if (u.x < c.probSpecR + c.probSpecT && c.deltaTransmission)
    {
        // Straight on, which leaves the local frame alone. Never inside, being thin-walled.
        s.wi = ToWorld(p, -wo);
        s.pdf = c.probSpecT;
        s.weight = ThinDeltaTransmittance(p, wo.z) / c.probSpecT;
        s.lobe = LOBE_DELTA_TRANSMISSION;
        return s;
    }
    else if (u.x < c.probSpecR + c.probSpecT && p.geometryThinWalled)
    {
        wi = SampleGGXReflectionBounded(wo, c.alpha, c.boundedB, u.yz);
        if (wi.z <= 0.0)
            return s;
        wi.z = -wi.z;
        lobe = LOBE_SPECULAR_TRANSMISSION;
    }
    else if (u.x < c.probSpecR + c.probSpecT)
    {
        float3 m = SampleGGXVNDF(wo, c.alpha, u.yz);
        wi = refract(-wo, m, 1.0 / c.eta);
        if (dot(wi, wi) == 0.0 || wi.z >= 0.0)
            return s;
        wi = normalize(wi);
        lobe = LOBE_SPECULAR_TRANSMISSION;
    }
    else if (c.probDiffuse > 0.0)
    {
        wi = SampleCosineHemisphere(u.yz);
        lobe = LOBE_DIFFUSE;
    }
    else
    {
        return s;
    }

    // Back to the caller's frame, then weight with the full mixture.
    if (c.inside)
        wi = -wi;

    float pdf;
    float3 fCos = OpenPBREvalPdfLocal(p, c, wi, pdf);
    if (!(pdf > 0.0))
        return s;

    s.wi = ToWorld(p, wi);
    s.pdf = pdf;
    s.weight = fCos / pdf;
    s.lobe = lobe;
    return s;
}

// World-space entry points. Eval and Pdf alone are wrappers; the unused half is removed when inlined.

float3 OpenPBREvalPdf(OpenPBRBaseParams p, OpenPBRContext c, float3 wi, out float pdf)
{
    return OpenPBREvalPdfLocal(p, c, ToLocal(p, wi), pdf);
}

float3 OpenPBREval(OpenPBRBaseParams p, OpenPBRContext c, float3 wi)
{
    float pdf;
    return OpenPBREvalPdf(p, c, wi, pdf);
}

float OpenPBRPdf(OpenPBRBaseParams p, OpenPBRContext c, float3 wi)
{
    float pdf;
    OpenPBREvalPdf(p, c, wi, pdf);
    return pdf;
}

// Optional leak check. Ng is the geometric normal of the triangle, oriented outwards.
// True when the shading basis and the geometry agree on reflection versus transmission.
// Use it on sampled directions and on light samples before evaluating.
bool GeometricallyConsistent(OpenPBRBaseParams p, float3 Ng, float3 wi, float3 wo)
{
    float gi = dot(Ng, wi);
    float go = dot(Ng, wo);
    float si = dot(p.geometryNormal, wi);
    float so = dot(p.geometryNormal, wo);
    return gi != 0.0 && ((gi * go > 0.0) == (si * so > 0.0));
}

// Builds the shading basis in the params from the interpolated vertex frame and the
// normal-mapped shading normal. Call once per hit, before OpenPBRSetup.
//
// T, B, N  interpolated vertex tangent, bitangent and normal (world space)
// Ns       shading normal from the normal map (world space), pass N if there is no map
// wo       direction towards the viewer, normally -rayDir
void SetShadingBasis(inout OpenPBRBaseParams p, float3 T, float3 B, float3 N, float3 Ns, float3 wo)
{
    // Keep the viewer on the same side of Ns as of the vertex normal.
    const float eps = 0.01;
    Ns = normalize(Ns);
    float side = dot(N, wo) >= 0.0 ? 1.0 : -1.0;
    float d = dot(Ns, wo) * side;
    if (d < eps)
        Ns = normalize(Ns - side * wo * (d - eps));

    // Make the tangent perpendicular to Ns, falling back to the bitangent if they are parallel.
    float3 t = T - Ns * dot(Ns, T);
    if (dot(t, t) < 1.0e-12)
        t = cross(B, Ns);

    p.geometryNormal = Ns;
    p.geometryTangent = normalize(t);
    p.geometryBitangent = cross(Ns, p.geometryTangent);
}
