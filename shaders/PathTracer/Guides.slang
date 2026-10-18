// DLSS Ray Reconstruction guides for the primary surface: albedos, normal, hit distance, motion.
import OpenPBRBSDF;
import PathTracer.Math;
import PathTracer.RayTracing;
import PathTracer.Resources;

// FP16_MAX, what DLSS-RR expects as the specular hit distance of sky pixels.
static const float SKY_HIT_DISTANCE = HALF_MAX;

// What DLSS-RR needs to know about the primary surface. Defaults are the sky values from the RR
// integration guide, so a primary miss leaves them as they are.
struct PrimaryGuides
{
    float4 position; // homogeneous world position, w = 0 for the sky (a direction at infinity)
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    float3 normal;
    float roughness;
    float specularHitDistance; // world-space length of the primary specular reflection ray
    // The surface whose reflection the specular guides describe, normally the guides' own surface but
    // a pane of glass in front of it when that reflects more (see PaneReflection).
    float4 reflector;
    float reflectorRoughness;
    float curvature; // the reflector's signed mean curvature as the camera sees it, positive for a convex mirror
    // Motion vector of the surface a curved chain of mirrors and glass ends on (see ChainMotionVector),
    // for the virtual position's own reprojection, which only flat mirrors move like.
    float2 chainMotion;
    bool hasChainMotion;
};

PrimaryGuides DefaultGuides(float3 dir)
{
    PrimaryGuides g;
    g.position = float4(dir, 0.0);
    g.diffuseAlbedo = float3(0.5, 0.5, 0.5);
    g.specularAlbedo = float3(0.0, 0.0, 0.0);
    g.normal = float3(0.0, 0.0, 0.0);
    g.roughness = 0.0;
    g.specularHitDistance = SKY_HIT_DISTANCE;
    g.reflector = g.position;
    g.reflectorRoughness = 0.0;
    g.curvature = 0.0;
    g.chainMotion = float2(0.0, 0.0);
    g.hasChainMotion = false;
    return g;
}

// The hit's mean curvature as a viewer looking along -wo sees it, positive for a bulge towards them.
float ViewedCurvature(HitInfo hit, float3 wo)
{
    float meanCurvature = HitMeanCurvature(hit);
    return dot(hit.Ng, wo) < 0.0 ? -meanCurvature : meanCurvature;
}

// Where the reflector's reflection appears, by the thin mirror equation 1/i = 1/o + 2 curvature, moved
// out only by the dominant direction's share for glossy surfaces (Lagarde and de Rousiers 2014).
float4 SpecularVirtualPosition(PrimaryGuides g, float3 viewDir, float hitDistance)
{
    float alpha = g.reflectorRoughness * g.reflectorRoughness;
    float smoothness = 1.0 - alpha;
    float dominantFactor = smoothness * (sqrt(smoothness) + alpha);
    if (g.reflector.w == 0.0 || hitDistance <= 0.0 || dominantFactor <= 0.0)
        return g.reflector;

    float inverseImageDistance = (hitDistance >= SKY_HIT_DISTANCE ? 0.0 : 1.0 / hitDistance) + 2.0 * g.curvature;
    return float4(g.reflector.xyz * inverseImageDistance + viewDir * dominantFactor, inverseImageDistance);
}

// The specular guides follow the coat where it reflects more than the base's specular lobe.
bool CoatLeadsGuides(OpenPBRContext ctx)
{
    return ctx.probCoat > ctx.probSpecR;
}

// The roughness of the lobe the specular guides follow.
float GuideRoughness(OpenPBRBaseParams p, OpenPBRContext ctx)
{
    return CoatLeadsGuides(ctx) ? p.coatRoughness : CoatedSpecularRoughness(p);
}

// Directional reflectance of an OpenPBR surface, split into its diffuse and specular parts. Fuzz is broad
// enough to count as diffuse, the coat is specular and takes its terms from the context.
void ReflectanceGuides(OpenPBRBaseParams p, OpenPBRContext ctx, float3 wo, inout PrimaryGuides g)
{
    float M = p.baseMetalness;
    float3 baseColour = p.baseWeight * p.baseColour;

    float dielectricF0 = FresnelF0(SpecularIorRatio(p));
    float3 F0 = lerp(dielectricF0 * p.specularColour, p.specularWeight * baseColour, M);

    float NdotV = dot(p.geometryNormal, wo);
    float roughness = CoatedSpecularRoughness(p);
    g.specularAlbedo = EnvBRDFApprox2(F0, roughness * roughness, NdotV);
    g.diffuseAlbedo = (1.0 - M) * (1.0 - p.transmissionWeight) * lerp(baseColour, p.subsurfaceColour, p.subsurfaceWeight);
    if (p.coatWeight > 0.0)
    {
        float3 under = CoatBaseScale(p, ctx, true, ctx.coatCos);
        g.specularAlbedo = p.coatWeight * ctx.coatE + under * g.specularAlbedo;
        g.diffuseAlbedo *= under;
    }
    
    if (p.fuzzWeight > 0.0)
    {
        float fuzzAlbedo = p.fuzzWeight * FuzzAlbedo(saturate(NdotV), FuzzRoughness(p));
        g.specularAlbedo *= 1.0 - fuzzAlbedo;
        g.diffuseAlbedo = (1.0 - fuzzAlbedo) * g.diffuseAlbedo + fuzzAlbedo * p.fuzzColour;
    }
}

float InitialSpecularHitDistance(float roughness)
{
    return roughness <= g_Frame.specularHitDistanceHoldRoughness ? -1.0 : 0.0;
}
