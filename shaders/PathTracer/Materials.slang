// glTF material rows to OpenPBR parameters, for the integrator and the any-hit shaders.
import OpenPBRBaseSubstrate;
import PathTracer.Geometry;
import PathTracer.Math;
import PathTracer.RayTracing;
import PathTracer.Resources;

// Spec default values, with an identity basis that the caller overwrites per hit.
OpenPBRBaseParams OpenPBRDefaults()
{
    OpenPBRBaseParams p;
    p.baseWeight = 1.0;
    p.baseColour = float3(0.8, 0.8, 0.8);
    p.baseMetalness = 0.0;
    p.baseDiffuseRoughness = 0.0;
    p.specularWeight = 1.0;
    p.specularColour = float3(1.0, 1.0, 1.0);
    p.specularRoughness = 0.3;
    p.specularRoughnessAnisotropy = 0.0;
    p.specularIor = 1.5;
    p.transmissionWeight = 0.0;
    p.transmissionColour = float3(1.0, 1.0, 1.0);
    p.transmissionDepth = 0.0;
    p.subsurfaceWeight = 0.0;
    p.subsurfaceColour = float3(0.8, 0.8, 0.8);
    p.subsurfaceScatterAnisotropy = 0.0;
    p.subsurfaceRadius = float3(0.0, 0.0, 0.0);
    p.geometryThinWalled = false;
    p.geometryTangent = float3(1.0, 0.0, 0.0);
    p.geometryBitangent = float3(0.0, 1.0, 0.0);
    p.geometryNormal = float3(0.0, 0.0, 1.0);
    return p;
}

// glTF transmission in OpenPBR terms: the base colour tints what crosses the surface, unless a
// volume's attenuation sets the medium instead (assets rarely use both).
void SetTransmission(inout OpenPBRBaseParams p, GpuMaterial material, float transmission, bool thinWalled)
{
    p.transmissionWeight = saturate(transmission);
    p.specularIor = material.ior;
    p.geometryThinWalled = thinWalled;
    if (!thinWalled && material.attenuationDistance < FLT_MAX)
    {
        p.transmissionColour = material.attenuationColour;
        p.transmissionDepth = material.attenuationDistance;
    }
    else
    {
        p.transmissionColour = p.baseColour;
    }
}

// glTF anisotropic GGX in OpenPBR terms. glTF stretches alpha along the anisotropy direction to
// mix(r^2, 1, s^2) and keeps r^2 across it. OpenPBR (GGXAlpha) has alphaT = r'^2 sqrt(2 / (1 + (1 - a)^2))
// and alphaB = (1 - a) alphaT, so 1 - a = alphaB / alphaT and r'^2 is the RMS of the two alphas, which
// gives the same lobe exactly. The tangent must already point along the anisotropy direction.
void SetGltfAnisotropy(inout OpenPBRBaseParams p, float strength)
{
    float alphaB = p.specularRoughness * p.specularRoughness;
    float alphaT = lerp(alphaB, 1.0, strength * strength);
    if (alphaT <= 0.0)
        return;
    p.specularRoughnessAnisotropy = 1.0 - alphaB / alphaT;
    p.specularRoughness = sqrt(sqrt(0.5 * (alphaT * alphaT + alphaB * alphaB)));
}

// The Rendering panel's debug switches (MaterialOverrideFlags), applied to every material.
void ApplyMaterialOverrides(inout OpenPBRBaseParams p)
{
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_SPECULAR) != 0)
    {
        p.baseMetalness = 0.0;
        p.specularColour = float3(0.0, 0.0, 0.0);
    }
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_TRANSMISSION) != 0)
    {
        p.transmissionWeight = 0.0;
        if (p.geometryThinWalled)
            p.subsurfaceWeight = 0.0;
    }
}

// Reads the material row once, samples all of its textures and applies the normal map to hit.Ns.
// The textures are filtered over the footprint of a ray cone coneWidth wide where it meets the surface.
OpenPBRBaseParams LoadMaterial(inout HitInfo hit, float coneWidth, bool anisotropic)
{
    GpuMaterial material = LoadGpuMaterial(hit.materialId);
    float2 uv = hit.uv;
    float2 ddx = hit.uvGradients.xy * coneWidth;
    float2 ddy = hit.uvGradients.zw * coneWidth;

    // Tangent-space normal map: +X along the tangent, +Y along the bitangent, +Z along N.
    if (material.normalTexture != INVALIDDESCRIPTORHANDLE)
    {
        float3 tn = SampleTextureGrad(material.normalTexture, uv, ddx, ddy, anisotropic).xyz * 2.0 - 1.0;
        tn.y = -tn.y; // flip green channel (DirectX-style normal map convention)
        tn.xy *= material.normalScale;
        hit.Ns = normalize(hit.T * tn.x + hit.B * tn.y + hit.N * tn.z);
    }

    float3 baseColour = material.baseColour.rgb;
    if (material.albedoTexture != INVALIDDESCRIPTORHANDLE)
        baseColour *= SampleTextureGrad(material.albedoTexture, uv, ddx, ddy, anisotropic).rgb;

    float roughness = material.roughness;
    float metallic = material.metallic;
    if (material.roughMetalTexture != INVALIDDESCRIPTORHANDLE)
    {
        float4 roughMetal = SampleTextureGrad(material.roughMetalTexture, uv, ddx, ddy, anisotropic);
        roughness *= roughMetal.g;
        metallic *= roughMetal.b;
    }

    float transmission = material.transmission;
    if (transmission > 0.0 && material.transmissionTexture != INVALIDDESCRIPTORHANDLE)
        transmission *= SampleTextureGrad(material.transmissionTexture, uv, ddx, ddy, anisotropic).r;
    bool thinWalled = (material.transmissionFlags & TRANSMISSION_THIN_WALLED) != 0;
    if (thinWalled && material.transmission > 0.0 && (material.transmissionFlags & TRANSMISSION_PASS_THROUGH) == 0)
        roughness = max(roughness, 1.0e-4);

    float anisotropyStrength = 0.0;
    float2 anisotropy = UnpackHalf2(material.anisotropy);
    if (any(anisotropy != 0.0))
    {
        float2 direction = anisotropy;
        anisotropyStrength = length(anisotropy);
        if (material.anisotropyTexture != INVALIDDESCRIPTORHANDLE)
        {
            float3 texel = SampleTextureGrad(material.anisotropyTexture, uv, ddx, ddy, anisotropic).rgb;
            float2 texDirection = texel.rg * 2.0 - 1.0;
            direction = float2(anisotropy.x * texDirection.x - anisotropy.y * texDirection.y,
                               anisotropy.y * texDirection.x + anisotropy.x * texDirection.y);
            anisotropyStrength *= texel.b;
        }
        // Mips average opposite directions towards zero; the rotation alone is the best guess there.
        float lengthSq = dot(direction, direction);
        direction = lengthSq > 1.0e-8 ? direction * rsqrt(lengthSq) : anisotropy * rsqrt(dot(anisotropy, anisotropy));
        // The normal map's tangent space: glTF's +Y is -hit.B.
        float3 T = hit.T * direction.x - hit.B * direction.y;
        hit.B = hit.T * direction.y + hit.B * direction.x;
        hit.T = T;
    }

    if (any(material.emissive > 0.0))
    {
        hit.emission = material.emissive;
        if (material.emissiveTexture != INVALIDDESCRIPTORHANDLE)
            hit.emission *= SampleTextureGrad(material.emissiveTexture, uv, ddx, ddy, anisotropic).rgb;
    }

    OpenPBRBaseParams p = OpenPBRDefaults();
    p.baseColour = baseColour;
    p.baseMetalness = saturate(metallic);
    p.specularRoughness = saturate(roughness);
    SetTransmission(p, material, transmission, thinWalled);
    if (anisotropyStrength > 0.0)
        SetGltfAnisotropy(p, saturate(anisotropyStrength));

    // glTF diffuse transmission mixes the base's diffuse BRDF with a diffuse BTDF, which is OpenPBR's
    // thin-walled subsurface with all of it transmitted (anisotropy 1). In a solid object the subsurface
    // is a scattering medium instead, unless the Rendering panel turned all of it off.
    if (material.subsurfaceWeight > 0.0)
    {
        float weight = material.subsurfaceWeight;
        if (material.subsurfaceWeightTexture != INVALIDDESCRIPTORHANDLE)
            weight *= SampleTextureGrad(material.subsurfaceWeightTexture, uv, ddx, ddy, anisotropic)
                          [material.subsurfaceWeightChannel];
        float3 colour = material.subsurfaceColour;
        if (material.subsurfaceColourTexture != INVALIDDESCRIPTORHANDLE)
            colour *= SampleTextureGrad(material.subsurfaceColourTexture, uv, ddx, ddy, anisotropic).rgb;
        p.subsurfaceWeight = saturate(weight);
        p.subsurfaceColour = colour;
        p.subsurfaceScatterAnisotropy = material.subsurfaceScatterAnisotropy;
        if (!thinWalled && g_Frame.subsurfaceComponents != 0)
        {
            float3 radius = material.subsurfaceRadius;
            if (material.subsurfaceRadiusTexture != INVALIDDESCRIPTORHANDLE)
                radius *= SampleTextureGrad(material.subsurfaceRadiusTexture, uv, ddx, ddy, anisotropic).rgb;
            if (material.subsurfaceScaleTexture != INVALIDDESCRIPTORHANDLE)
                radius *= SampleTextureGrad(material.subsurfaceScaleTexture, uv, ddx, ddy, anisotropic)
                              [material.subsurfaceScaleChannel];
            p.subsurfaceRadius = radius * g_Frame.subsurfaceRadiusScale;
        }
    }

    ApplyMaterialOverrides(p);
    return p;
}

// Beer-Lambert extinction so that light reaching transmissionDepth has transmissionColor.
float3 MediumExtinction(OpenPBRBaseParams p)
{
    if (p.transmissionWeight <= 0.0 || p.transmissionDepth <= 0.0)
        return float3(0.0, 0.0, 0.0);
    return -log(max(p.transmissionColour, 1.0e-4)) / p.transmissionDepth;
}

// Base colour and alpha at a candidate hit, for the any-hit shaders and the inline shadow query.
float4 LoadBaseColourAlpha(GpuMaterial material, float2 uv)
{
    float4 albedo = material.baseColour;
    if (material.albedoTexture != INVALIDDESCRIPTORHANDLE)
        albedo *= SampleTexture(material.albedoTexture, uv);
    return albedo;
}

// What a pass-through surface lets a shadow ray through.
float3 PassThroughTransmittance(GpuMaterial material, float2 uv, float3 baseColour, float faceCosine)
{
    float transmission = material.transmission;
    if (material.transmissionTexture != INVALIDDESCRIPTORHANDLE)
        transmission *= SampleTexture(material.transmissionTexture, uv).r;
    float metallic = material.metallic;
    if (material.roughMetalTexture != INVALIDDESCRIPTORHANDLE)
        metallic *= SampleTexture(material.roughMetalTexture, uv).b;

    OpenPBRBaseParams p = OpenPBRDefaults();
    p.baseColour = baseColour;
    p.baseMetalness = saturate(metallic);
    p.specularRoughness = 0.0;
    SetTransmission(p, material, transmission, true);
    ApplyMaterialOverrides(p);
    return ThinDeltaTransmittance(p, faceCosine);
}

// What a candidate triangle lets a shadow ray along dir through: pass-through surfaces some, anything
// else nothing. False for a cut-out, which the ray ignores.
bool ShadowCandidateTransmittance(PrimitiveDesc prim, uint primitiveIndex, float2 barycentrics, float3x3 linearToWorld,
                                  float3 dir, out float3 transmitted)
{
    transmitted = float3(0.0, 0.0, 0.0);
    GpuMaterial material = LoadGpuMaterial(prim.materialIndex);
    uint3 tri = LoadTriangle(prim, primitiveIndex);
    float2 uv = InterpolateTexCoord(tri, Barycentrics(barycentrics));
    float4 albedo = LoadBaseColourAlpha(material, uv);

    if (material.alphaMode == ALPHA_MODE_MASK && albedo.a < material.alphaCutoff)
        return false;

    if ((material.transmissionFlags & TRANSMISSION_PASS_THROUGH) != 0)
        transmitted = PassThroughTransmittance(material, uv, albedo.rgb, FaceCosine(tri, linearToWorld, dir));
    return true;
}
