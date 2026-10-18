// Primary surface replacement: follows mirrors and glass in front of the camera so the DLSS-RR
// guides describe the surface seen through them.
import OpenPBRBaseSubstrate;
import PathTracer.Guides;
import PathTracer.Resources;

static const float PSR_MIN_SPECULAR_FRACTION = 0.9;
static const float PSR_MAX_VIRTUAL_DEPTH_JITTER = 0.002;

float2 SpecularFractions(OpenPBRBaseParams p)
{
    float M = p.baseMetalness;
    float T = p.transmissionWeight;
    float3 baseColour = p.baseWeight * p.baseColour;

    float etaS = ModulatedIORRatio(p.specularIor, p.specularWeight);
    float F0 = (1.0 - etaS) / (1.0 + etaS);
    F0 *= F0;

    // F82-tint reflects baseColour head on.
    float specular = (1.0 - M) * F0 * Luminance(p.specularColour) + M * p.specularWeight * Luminance(baseColour);
    float diffuse = (1.0 - M) * (1.0 - F0) * (1.0 - T) * Luminance(lerp(baseColour, p.subsurfaceColour, p.subsurfaceWeight));
    float transmission = (1.0 - M) * (1.0 - F0) * T * Luminance(TransmissionTint(p));
    // Head-on fuzz over all of it.
    float fuzzAlbedo = p.fuzzWeight > 0.0 ? p.fuzzWeight * FuzzAlbedo(1.0, FuzzRoughness(p)) : 0.0;
    float fuzz = fuzzAlbedo * Luminance(p.fuzzColour);
    float under = 1.0 - fuzzAlbedo;
    return under * float2(specular, transmission) / max(under * (specular + transmission + diffuse) + fuzz, 1.0e-4);
}

bool IsPsrEvent(OpenPBRBaseParams p, OpenPBRContext ctx, uint lobe, out float fraction)
{
    fraction = 1.0;
    bool reflection = lobe == LOBE_SPECULAR_REFLECTION;
    if (!(reflection || lobe == LOBE_SPECULAR_TRANSMISSION) || p.specularRoughness > g_Frame.psrMaxRoughness)
        return false;
    if (ctx.inside)
        return reflection == (FresnelDielectric(ctx.wo.z, ctx.eta) >= 0.5);
    float2 fractions = SpecularFractions(p);
    fraction = reflection ? fractions.x : fractions.y;
    return fraction >= PSR_MIN_SPECULAR_FRACTION;
}

// Directional reflectance (reflection) or transmittance of a chain event, as wo sees it.
float3 PsrEventThroughput(OpenPBRBaseParams p, OpenPBRContext ctx, float3 wo, bool reflection)
{
    if (ctx.inside)
    {
        float F = FresnelDielectric(ctx.wo.z, ctx.eta);
        return (reflection ? F : 1.0 - F).xxx;
    }
    if (reflection)
    {
        PrimaryGuides reflectance;
        ReflectanceGuides(p, wo, reflectance);
        return reflectance.specularAlbedo;
    }
    return (1.0 - p.fuzzWeight * ctx.fuzzE) * (1.0 - p.baseMetalness) * p.transmissionWeight * (1.0 - ctx.E) *
           TransmissionTint(p);
}

// A chain of mirrors and glass in front of the camera, primary first, while the path follows it.
struct SpecularChain
{
    bool active;
    // Real to virtual space transform, a product of reflections and refraction rotations, kept as a
    // rotation quaternion plus parity (odd adds a fixed z reflection) instead of a 3x3 matrix.
    bool parityOdd;
    float4 rotor;
    float3 throughput; // the events' directional reflectance or transmittance, and any absorption between them
    float primaryFraction; // share of the primary surface's reflectance in the lobe its path took
    float primaryDistance; // camera to primary surface
    float length; // primary surface to the latest hit, through the events in between
    float pixelDeflection; // largest angle an event turns the path by across one pixel's footprint
    bool throughGlass; // starts with a refraction into solid glass, whose transmittance it carries
};

float4 QuaternionMultiply(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// Rotation half of H(n) F (even parity) or F H(n) (odd), H(n) the reflection about normal n and F
// about z = 0: twice the angle between n and z, about their common perpendicular.
float4 MirrorRotor(float3 n, bool parityOdd)
{
    return parityOdd ? float4(n.y, -n.x, 0.0, n.z) : float4(-n.y, n.x, 0.0, n.z);
}

// The rotation taking unit vector a to unit vector b along the shorter arc, which is well defined
// for a refraction since it turns the ray by less than 90 degrees.
float4 ArcRotor(float3 a, float3 b)
{
    return normalize(float4(cross(a, b), 1.0 + dot(a, b)));
}

// Appends the event where the path arrived along dirIn and left along dirOut, at a surface whose
// normal turns by curvature radians per unit length.
void AddPsrEvent(inout SpecularChain chain, OpenPBRBaseParams p, OpenPBRContext ctx, float3 wo, float3 dirIn, float3 dirOut,
                 bool reflection, float pixelFootprint, float curvature)
{
    chain.throughput *= PsrEventThroughput(p, ctx, wo, reflection);

    // How much a turn of the normal turns the path: twice for a reflection (or a thin wall's mirrored
    // transmission), 1 - 1/eta for a refraction.
    float deflection;
    if (reflection)
    {
        chain.rotor = QuaternionMultiply(chain.rotor, MirrorRotor(p.geometryNormal, chain.parityOdd));
        chain.parityOdd = !chain.parityOdd;
        deflection = 2.0;
    }
    else
    {
        // Behind an odd number of reflections the rotation is conjugated by F, which negates the x and
        // y of its quaternion.
        float4 q = ArcRotor(dirOut, dirIn);
        chain.rotor = QuaternionMultiply(chain.rotor, chain.parityOdd ? float4(-q.xy, q.zw) : q);
        deflection = p.geometryThinWalled ? 2.0 : abs(1.0 - 1.0 / ctx.eta);
    }
    chain.pixelDeflection = max(chain.pixelDeflection, deflection * curvature * pixelFootprint);
}

SpecularChain StartSpecularChain(OpenPBRBaseParams p, OpenPBRContext ctx, float3 wo, float3 dirIn, float3 dirOut, bool reflection,
                                 float fraction, float primaryDistance, float pixelFootprint, float curvature)
{
    SpecularChain chain;
    chain.active = true;
    chain.parityOdd = false;
    chain.rotor = float4(0.0, 0.0, 0.0, 1.0);
    chain.throughput = float3(1.0, 1.0, 1.0);
    chain.primaryFraction = fraction;
    chain.primaryDistance = primaryDistance;
    chain.length = 0.0;
    chain.pixelDeflection = 0.0;
    chain.throughGlass = !reflection && !p.geometryThinWalled;
    AddPsrEvent(chain, p, ctx, wo, dirIn, dirOut, reflection, pixelFootprint, curvature);
    return chain;
}

float3 ApplySpecularChain(SpecularChain chain, float3 v)
{
    if (chain.parityOdd)
        v.z = -v.z;
    float3 t = 2.0 * cross(chain.rotor.xyz, v);
    return v + chain.rotor.w * t + cross(chain.rotor.xyz, t);
}

// Swap the primary surface's guides for the surface the chain ends on, unless an event on the way is
// curved enough to smear the virtual image. Returns whether it did.
bool ReplacePrimarySurface(SpecularChain chain, OpenPBRBaseParams surface, float curvature, float3 wo, float3 cameraDir,
                           inout PrimaryGuides guides)
{
    float pathLength = chain.primaryDistance + chain.length;
    if (chain.pixelDeflection * chain.length > PSR_MAX_VIRTUAL_DEPTH_JITTER * pathLength)
        return false;

    // The chain's transform is a rotation or reflection, which preserves the angle between wo and the
    // normal, so the reflectance seen in virtual space is the real one.
    PrimaryGuides seen;
    ReflectanceGuides(surface, wo, seen);

    // What the primary surface shows of the surface comes scaled by the chain's throughput, on top of
    // the primary's own (small) diffuse part.
    guides.position = float4(guides.position.xyz + cameraDir * chain.length, 1.0);
    guides.normal = normalize(ApplySpecularChain(chain, surface.geometryNormal));
    guides.roughness = surface.specularRoughness;
    guides.diffuseAlbedo += chain.throughput * seen.diffuseAlbedo;
    guides.specularAlbedo = lerp(guides.specularAlbedo, chain.throughput * seen.specularAlbedo, chain.primaryFraction);
    guides.specularHitDistance = InitialSpecularHitDistance(surface.specularRoughness);
    // The chain's rotations and reflections keep the viewer on the same side of the surface.
    guides.curvature = curvature;
    return true;
}
