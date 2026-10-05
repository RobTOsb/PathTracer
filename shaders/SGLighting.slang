// Spherical Gaussian lighting approximations from
// "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"
// (Tokuyoshi, Ikeda, Kulkarni, Harada, SIGGRAPH Asia 2024).
//
// SG convention: g(o; xi, kappa) = exp(kappa * (dot(o, xi) - 1)).

static const float SG_PI = 3.14159265358979;
static const float SG_TWO_PI = 6.28318530717959;
static const float SG_INV_PI = 0.318309886183791;
static const float SG_INV_SQRT_PI = 0.564189583547756;

// Complementary error function, fractional error < 1.2e-7 (Numerical Recipes erfcc).
float SGErfc(float x)
{
    float z = abs(x);
    float t = 1.0 / (1.0 + 0.5 * z);
    float r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
              t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
              t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0.0 ? r : 2.0 - r;
}

// Error function. Uses a Taylor series near zero to avoid cancellation in 1 - erfc(x).
float SGErf(float x)
{
    if (abs(x) < 0.5)
    {
        float x2 = x * x;
        return 1.12837916709551 * x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (0.1 + x2 * (-1.0 / 42.0 + x2 * (1.0 / 216.0)))));
    }
    float r = 1.0 - SGErfc(abs(x));
    return x >= 0.0 ? r : -r;
}

// Full spherical integral of an SG: 2 * pi * (1 - exp(-2 * kappa)) / kappa.
float SGIntegral(float kappa)
{
    if (kappa < 0.05)
    {
        float k = kappa;
        return SG_TWO_PI * (2.0 + k * (-2.0 + k * (4.0 / 3.0 + k * (-2.0 / 3.0))));
    }
    return SG_TWO_PI * (1.0 - exp(-2.0 * kappa)) / kappa;
}

struct SGLight
{
    float3 axis;      // xi
    float sharpness;  // kappa
    float amplitude;  // W
};

// Product of two SGs, returned as (axis, sharpness) plus the log of the scale factor.
// exp(logScale) * g(o; axis, sharpness) = g(o; a0, k0) * g(o; a1, k1).
// logScale = |k0 a0 + k1 a1| - k0 - k1, rewritten to avoid cancellation.
void SGProduct(float3 a0, float k0, float3 a1, float k1, out float3 axis, out float sharpness, out float logScale)
{
    float3 v = k0 * a0 + k1 * a1;
    sharpness = length(v);
    axis = sharpness > 0.0 ? v / sharpness : a1;
    float denom = sharpness + k0 + k1;
    logScale = denom > 0.0 ? -2.0 * k0 * k1 * (1.0 - dot(a0, a1)) / denom : 0.0;
}

// Section 4: interpolation-based diffuse SG lighting.
// Approximates the integral over S^2 of g(o; xi, kappa) * max(dot(o, n), 0), given cosine = dot(xi, n).
float SGClampedCosineProductIntegral(float cosine, float kappa)
{
    float k = kappa;
    float c = clamp(cosine, -1.0, 1.0);

    // Fitted t(kappa): inverse standard deviation * sqrt(0.5) of the planar Gaussian.
    float t = k * sqrt(0.5 * ((k + 2.7360833) * k + 17.021297) / (((k + 4.0100827) * k + 15.219156) * k + 76.087896));

    // Interpolation factor u = (q(c) - q(-1)) / (q(1) - q(-1)), where q(1) - q(-1) = 2t.
    float u;
    if (t > 1e-6)
    {
        float tc = t * c;
        float qc = tc * SGErfc(-tc) + exp(-tc * tc) * SG_INV_SQRT_PI;
        float qm = -t * SGErfc(t) + exp(-t * t) * SG_INV_SQRT_PI;
        u = saturate((qc - qm) / (2.0 * t));
    }
    else
    {
        u = 0.5 * (c + 1.0);
    }

    // Closed forms for xi = +n (upper) and xi = -n (lower), both divided by 2 pi.
    float e = exp(-k);
    float upper, lower;
    if (k < 0.25)
    {
        upper = 0.5 + k * (-1.0 / 6.0 + k * (1.0 / 24.0 + k * (-1.0 / 120.0 + k * (1.0 / 720.0))));
        lower = e * (0.5 + k * (-1.0 / 3.0 + k * (1.0 / 8.0 + k * (-1.0 / 30.0 + k * (1.0 / 144.0)))));
    }
    else
    {
        float invK2 = 1.0 / (k * k);
        upper = (e - 1.0 + k) * invK2;
        lower = e * (1.0 - e - k * e) * invK2;
    }

    return SG_TWO_PI * lerp(lower, upper, u);
}

// Eq. 7. Diffuse importance for a Lambertian lobe (albedo factored out).
float SGDiffuseImportance(SGLight sg, float3 normal)
{
    return sg.amplitude * SG_INV_PI * SGClampedCosineProductIntegral(dot(sg.axis, normal), sg.sharpness);
}

// Normalised hemispherical integral of an SG [Tokuyoshi 2022]:
// (integral over the upper hemisphere of g) / (integral over S^2 of g), given cosine = dot(xi, n).
// Used for the filtered visibility V in Section 5.2.
float SGNormalisedHemisphericalIntegral(float cosine, float kappa)
{
    float c = clamp(cosine, -1.0, 1.0);
    if (kappa < 1e-4)
    {
        return 0.5 + 0.25 * kappa * c;
    }
    const float A = 0.6517328826907056;
    const float B = 1.3418280033141288;
    const float C = 7.2216687798956709;
    float steepness = kappa * sqrt((0.5 * kappa + A) / ((kappa + B) * kappa + C));
    float f = 0.5 + 0.5 * (SGErf(steepness * c) / SGErf(steepness));
    float e = exp(-kappa);
    return lerp(e, 1.0, f) / (1.0 + e);
}

// Per shading point data for the glossy importance. Everything here is constant during tree traversal.
struct GlossyContext
{
    float3 tangent;
    float3 bitangent;
    float3 normal;
    float3 viewTS;      // i in tangent space
    float2 alpha2;      // diagonal of the roughness matrix A = diag(alpha_x^2, alpha_y^2)
    float3 jjt;         // symmetric J J^T at h = n, stored as (xx, xy, yy)
    float3 reflectDir;  // xi_p, perfect specular reflection in world space
    float kappaP;       // sharpness of the conservative SG reflection lobe
};

GlossyContext MakeGlossyContext(float3 T, float3 B, float3 N, float3 V, float alphaX, float alphaY)
{
    GlossyContext g;
    g.tangent = T;
    g.bitangent = B;
    g.normal = N;
    g.viewTS = float3(dot(V, T), dot(V, B), dot(V, N));

    float ax = clamp(alphaX, 1e-3, 1.0);
    float ay = clamp(alphaY, 1e-3, 1.0);
    g.alpha2 = float2(ax * ax, ay * ay);

    // Jacobian of the light direction to projected halfvector mapping, evaluated at the NDF peak h = n.
    // With o = reflect(i, n): J J^T = (E - i_xy i_xy^T) / (4 i_z^2).
    float iz = max(g.viewTS.z, 1e-3);
    float2 ixy = g.viewTS.xy;
    float s = 1.0 / (4.0 * iz * iz);
    g.jjt = float3(1.0 - ixy.x * ixy.x, -ixy.x * ixy.y, 1.0 - ixy.y * ixy.y) * s;

    g.reflectDir = 2.0 * dot(V, N) * N - V;
    float a2max = max(g.alpha2.x, g.alpha2.y);
    g.kappaP = (1.0 - a2max) / (2.0 * a2max);
    return g;
}

float2x2 SGInverse2x2(float2x2 m)
{
    float det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    float invDet = 1.0 / det;
    return float2x2(m[1][1], -m[0][1], -m[1][0], m[0][0]) * invDet;
}

// Section 5.1: NDF filtering with an SG light.
// Sigma_D = 0.5 (A^-1 - E)^-1, Sigma_L = J J^T / kappa, A_bar = ((2 (Sigma_D + Sigma_L))^-1 + E)^-1.
// Rewritten as A_bar = (A + 2K(E - A)) (E + 2K(E - A))^-1 with K = J J^T / kappa,
// which has no singularity at alpha = 1. Returns the symmetric result as (xx, xy, yy).
float3 SGFilteredRoughness(GlossyContext g, float kappa)
{
    float invKappa = 1.0 / max(kappa, 1e-6);
    float2x2 K2 = float2x2(g.jjt.x, g.jjt.y, g.jjt.y, g.jjt.z) * (2.0 * invKappa);
    float2x2 C = float2x2(1.0 - g.alpha2.x, 0.0, 0.0, 1.0 - g.alpha2.y);
    float2x2 KC = mul(K2, C);
    float2x2 M1 = float2x2(g.alpha2.x, 0.0, 0.0, g.alpha2.y) + KC;
    float2x2 M2 = float2x2(1.0, 0.0, 0.0, 1.0) + KC;
    float2x2 R = mul(M1, SGInverse2x2(M2));
    return float3(R[0][0], 0.5 * (R[0][1] + R[1][0]), R[1][1]);
}

// Section 5.2: VNDF based reflection lobe p(o; i, A) for the generalised GGX NDF [Heitz 2014],
// with A a symmetric 2x2 roughness matrix (xx, xy, yy). All vectors in tangent space.
// p(o; i, A) = D(m; A) / (4 sqrt(i_xy^T A i_xy + i_z^2)), m = sgn(h.z) h.
float SGGGXReflectionLobe(float3 o, float3 i, float3 A)
{
    float3 h = i + o;
    float len2 = dot(h, h);
    h = len2 > 1e-12 ? h * rsqrt(len2) : float3(0.0, 0.0, 1.0);
    float3 m = h.z < 0.0 ? -h : h;

    float det = max(A.x * A.z - A.y * A.y, 1e-20);
    float quad = A.z * m.x * m.x - 2.0 * A.y * m.x * m.y + A.x * m.y * m.y; // m_xy^T adj(A) m_xy
    float denom = quad + det * m.z * m.z;
    float D = det * sqrt(det) / (SG_PI * denom * denom);

    float norm = sqrt(max(A.x * i.x * i.x + 2.0 * A.y * i.x * i.y + A.z * i.y * i.y + i.z * i.z, 1e-12));
    return D / (4.0 * norm);
}

// Eq. 12. Glossy importance (specular reflectance factored out).
// I ~ W * V * p(xi; i, A_bar) * integral of g(o; xi, kappa).
float SGGlossyImportance(SGLight sg, GlossyContext g)
{
    float3 xiTS = float3(dot(sg.axis, g.tangent), dot(sg.axis, g.bitangent), dot(sg.axis, g.normal));
    float3 Abar = SGFilteredRoughness(g, sg.sharpness);
    float lobe = SGGGXReflectionLobe(xiTS, g.viewTS, Abar);

    // Filtered visibility using the lower frequency kernel g(o; xi_p, kappa_p) g(o; xi, kappa).
    float3 vAxis;
    float vSharpness, unusedLogScale;
    SGProduct(g.reflectDir, g.kappaP, sg.axis, sg.sharpness, vAxis, vSharpness, unusedLogScale);
    float visibility = SGNormalisedHemisphericalIntegral(dot(vAxis, g.normal), vSharpness);

    return sg.amplitude * visibility * lobe * SGIntegral(sg.sharpness);
}
