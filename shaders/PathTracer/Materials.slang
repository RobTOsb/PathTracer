// glTF material rows to OpenPBR parameters, for the integrator and the any-hit shaders.
import OpenPBRBaseSubstrate;
import PathTracer.Geometry;
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

// The Rendering panel's debug switches (MaterialOverrideFlags), applied to every material.
void ApplyMaterialOverrides(inout OpenPBRBaseParams p)
{
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_SPECULAR) != 0)
    {
        p.baseMetalness = 0.0;
        p.specularColour = float3(0.0, 0.0, 0.0);
    }
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_TRANSMISSION) != 0)
        p.transmissionWeight = 0.0;
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

// Base colour and alpha at the candidate hit, for the any-hit shaders.
float4 LoadBaseColourAlpha(GpuMaterial material, PrimitiveDesc prim, BuiltInTriangleIntersectionAttributes attr)
{
    float4 albedo = material.baseColour;
    if (material.albedoTexture != INVALIDDESCRIPTORHANDLE)
        albedo *= SampleTexture(material.albedoTexture, InterpolateTexCoord(LoadTriangle(prim), Barycentrics(attr)));
    return albedo;
}

// What a pass-through surface lets a shadow ray through, for ShadowAnyHit.
float3 PassThroughTransmittance(GpuMaterial material, PrimitiveDesc prim, BuiltInTriangleIntersectionAttributes attr,
                                float3 baseColour)
{
    float2 uv = InterpolateTexCoord(LoadTriangle(prim), Barycentrics(attr));
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
    return ThinDeltaTransmittance(p, FaceCosine(prim));
}
