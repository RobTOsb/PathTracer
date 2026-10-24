// glTF material rows to OpenPBR parameters, for the integrator and the any-hit shaders.
import OpenPBRBSDF;
import PathTracer.Geometry;
import PathTracer.Math;
import PathTracer.RayTracing;
import PathTracer.Resources;

// kMinIor in RenderGPUScene.cpp: an interface that hardly reflects or bends light, which the BSDF's
// Fresnel and refraction handle badly.
static const float MIN_IOR = 1.05;

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
    p.coatWeight = 0.0;
    p.coatColour = float3(1.0, 1.0, 1.0);
    p.coatRoughness = 0.0;
    p.coatIor = 1.6;
    p.coatDarkening = 1.0;
    p.fuzzWeight = 0.0;
    p.fuzzColour = float3(1.0, 1.0, 1.0);
    p.fuzzRoughness = 0.5;
    p.geometryThinWalled = false;
    p.geometryTangent = float3(1.0, 0.0, 0.0);
    p.geometryBitangent = float3(0.0, 1.0, 0.0);
    p.geometryNormal = float3(0.0, 0.0, 1.0);
    p.geometryCoatNormal = float3(0.0, 0.0, 1.0);
    return p;
}

// glTF clearcoat as OpenPBR coat: white, of IOR 1.5. glTF has no darkening; this keeps OpenPBR's default
// of 1, so coated base colours come out a little darker than in glTF viewers. Set on every material, coat
// or not, so the IOR is a constant the compiler folds the coat's Fresnel terms with.
void SetGltfClearcoat(inout OpenPBRBaseParams p, float weight, float roughness, float3 normal)
{
    p.coatWeight = saturate(weight);
    p.coatRoughness = saturate(roughness);
    p.coatIor = 1.5;
    p.geometryCoatNormal = normal;
}

// glTF transmission in OpenPBR terms: the base colour tints what crosses the surface, unless a
// volume's attenuation sets the medium instead (assets rarely use both).
void SetTransmission(inout OpenPBRBaseParams p, MaterialData material, float transmission, bool thinWalled)
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

// glTF specular (KHR_materials_specular) in OpenPBR terms. glTF scales the dielectric F0 by the weight and
// tints it by the colour, F0 = min(F0(ior) colour, 1) weight, and leaves metals alone. OpenPBR's
// specularWeight would scale metals too, so the weight goes into the IOR as its [modulated_ior] would
// put it, which also bends refraction a little where glTF does not. A colour above 1 moves its excess
// into the weight, and metals keep a white F82 tint. glTF's grazing reflectance (the weight, untinted)
// becomes OpenPBR's (1, tinted). Call after SetTransmission, which sets the IOR.
void SetGltfSpecular(inout OpenPBRBaseParams p, float weight, float3 colour)
{
    float excess = MaxComponent(colour);
    if (excess > 1.0)
    {
        weight *= excess;
        colour /= excess;
    }
    // Kept off 1 like the material's own IOR (kMinIor in RenderGPUScene.cpp), which a zero weight would
    // otherwise reach exactly.
    p.specularIor = max(ModulatedIORRatio(p.specularIor, max(weight, 0.0)), MIN_IOR);
    p.specularColour = lerp(saturate(colour), float3(1.0, 1.0, 1.0), p.baseMetalness);
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

// glTF sheen as OpenPBR fuzz: fuzzWeight * fuzzColour is the sheen colour, which makes the lobe the
// sheen colour times the LTC and leaves 1 - max(sheen colour) E_F(wo) to the layers below, as in
// KHR_materials_sheen's albedo scaling. Only the lobe's shape differs, Zeltner's LTC in place of Charlie.
void SetGltfSheen(inout OpenPBRBaseParams p, float3 sheen, float roughness)
{
    sheen = saturate(sheen);
    p.fuzzWeight = MaxComponent(sheen);
    p.fuzzColour = p.fuzzWeight > 0.0 ? sheen / p.fuzzWeight : float3(1.0, 1.0, 1.0);
    p.fuzzRoughness = saturate(roughness);
}

// The Rendering panel's debug switches (MaterialOverrideFlags), applied to every material.
void ApplyMaterialOverrides(inout OpenPBRBaseParams p)
{
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_SPECULAR) != 0)
    {
        p.baseMetalness = 0.0;
        p.specularColour = float3(0.0, 0.0, 0.0);
        p.coatWeight = 0.0;
    }
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_TRANSMISSION) != 0)
    {
        p.transmissionWeight = 0.0;
        if (p.geometryThinWalled)
            p.subsurfaceWeight = 0.0;
    }
}

// A GPU_MATERIAL_SIMPLE material's layers as the constants they are, so the compiler drops every lobe
// but the base's diffuse and specular from the code that follows.
void SimplifyMaterial(inout OpenPBRBaseParams p)
{
    OpenPBRBaseParams d = OpenPBRDefaults();
    p.specularColour = d.specularColour;
    p.specularRoughnessAnisotropy = 0.0;
    p.transmissionWeight = 0.0;
    p.transmissionColour = d.transmissionColour;
    p.transmissionDepth = 0.0;
    p.subsurfaceWeight = 0.0;
    p.subsurfaceColour = d.subsurfaceColour;
    p.subsurfaceScatterAnisotropy = 0.0;
    p.subsurfaceRadius = float3(0.0, 0.0, 0.0);
    p.coatWeight = 0.0;
    p.coatRoughness = 0.0;
    p.geometryCoatNormal = d.geometryCoatNormal;
    p.fuzzWeight = 0.0;
    p.fuzzColour = d.fuzzColour;
    p.fuzzRoughness = d.fuzzRoughness;
    // Without transmission or subsurface, a thin wall shades as a solid opaque surface.
    p.geometryThinWalled = false;
}

// Geometric specular antialiasing (Tokuyoshi and Kaplanyan 2019, "Improved Geometric Specular
// Antialiasing"): the lobe widened by the variance of the normals across the pixel, added to alpha^2.
float WidenRoughness(float roughness, float variance)
{
    float r2 = roughness * roughness;
    return sqrt(sqrt(min(r2 * r2 + variance, 1.0)));
}

// Toksvig: a normal map's mips average its normals, which come out the shorter the more they spread.
// Their variance as a vMF lobe of the averaged length, 2 * 0.25 / kappa for kappa = (3r - r^3) / (1 - r^2)
// as in Filament, slopes scaled by the map's scale. The few thousandths an 8-bit, BC7-encoded normal
// falls short of unit length at the top mip do not count as spread.
float NormalMapVariance(float3 averaged, float scale)
{
    float r = min(length(averaged) + 1.0 / 128.0, 1.0);
    return 0.5 * (1.0 - r * r) / max(3.0 * r - r * r * r, 1.0e-4) * scale * scale;
}

// Reads the material row once, samples all of its textures and applies the normal map to hit.Ns.
// The textures are filtered over the footprint of a ray cone coneWidth wide where it meets the surface.
// simple: the row is GPU_MATERIAL_SIMPLE, so only the base's textures are read.
// specularAA: geometric specular antialiasing of opaque surfaces, (variance of the vertex normals across
// the pixel, scale of the normal maps' variance), zero for none.
OpenPBRBaseParams LoadMaterial(inout HitInfo hit, float coneWidth, bool anisotropic, bool simple, float2 specularAA)
{
    MaterialData material = LoadGpuMaterial(hit.materialId);
    float2 uv = hit.uv;
    float2 ddx = hit.uvGradients.xy * coneWidth;
    float2 ddy = hit.uvGradients.zw * coneWidth;

    // Tangent-space normal map: +X along the tangent, +Y along the bitangent, +Z along N.
    float normalVariance = specularAA.x;
    if (material.normalTexture != INVALIDDESCRIPTORHANDLE)
    {
        float3 tn = SampleTextureGrad(material.normalTexture, uv, ddx, ddy, anisotropic).xyz * 2.0 - 1.0;
        if (specularAA.y > 0.0)
            normalVariance += specularAA.y * NormalMapVariance(tn, material.normalScale);
        tn.y = -tn.y; // flip green channel (DirectX-style normal map convention)
        tn.xy *= material.normalScale;
        hit.Ns = normalize(hit.T * tn.x + hit.B * tn.y + hit.N * tn.z);
    }

    // The coat has a normal map of its own, and without one follows the vertex normal (glTF), in the
    // tangent space before anisotropy turns it.
    float coatWeight = material.coatWeight;
    float coatRoughness = material.coatRoughness;
    float3 coatNormal = hit.N;
    float coatNormalVariance = specularAA.x;
    if (!simple && coatWeight > 0.0)
    {
        if (material.coatWeightTexture != INVALIDDESCRIPTORHANDLE)
            coatWeight *= SampleTextureGrad(material.coatWeightTexture, uv, ddx, ddy, anisotropic).r;
        if (material.coatRoughnessTexture != INVALIDDESCRIPTORHANDLE)
            coatRoughness *= SampleTextureGrad(material.coatRoughnessTexture, uv, ddx, ddy, anisotropic).g;
        if (material.coatNormalTexture != INVALIDDESCRIPTORHANDLE)
        {
            float3 tn = SampleTextureGrad(material.coatNormalTexture, uv, ddx, ddy, anisotropic).xyz * 2.0 - 1.0;
            if (specularAA.y > 0.0)
                coatNormalVariance += specularAA.y * NormalMapVariance(tn, material.coatNormalScale);
            tn.y = -tn.y;
            tn.xy *= material.coatNormalScale;
            coatNormal = normalize(hit.T * tn.x + hit.B * tn.y + hit.N * tn.z);
        }
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
    if (!simple && transmission > 0.0 && material.transmissionTexture != INVALIDDESCRIPTORHANDLE)
        transmission *= SampleTextureGrad(material.transmissionTexture, uv, ddx, ddy, anisotropic).r;
    bool thinWalled = (material.transmissionFlags & TRANSMISSION_THIN_WALLED) != 0;
    if (thinWalled && material.transmission > 0.0 && (material.transmissionFlags & TRANSMISSION_PASS_THROUGH) == 0)
        roughness = max(roughness, 1.0e-4);
    // Glass is left alone: its splits, pass-throughs and PSR depend on the material's own roughness.
    if (transmission <= 0.0 && any(specularAA > 0.0))
    {
        roughness = WidenRoughness(roughness, min(normalVariance, g_Frame.specularAAMaxVariance));
        coatRoughness = WidenRoughness(coatRoughness, min(coatNormalVariance, g_Frame.specularAAMaxVariance));
    }

    float anisotropyStrength = 0.0;
    float2 anisotropy = material.anisotropy;
    if (!simple && any(anisotropy != 0.0))
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
    float specularWeight = material.specularWeight;
    float3 specularColour = material.specularColour;
    if (!simple && material.specularTexture != INVALIDDESCRIPTORHANDLE)
        specularWeight *= SampleTextureGrad(material.specularTexture, uv, ddx, ddy, anisotropic).a;
    if (!simple && material.specularColourTexture != INVALIDDESCRIPTORHANDLE)
        specularColour *= SampleTextureGrad(material.specularColourTexture, uv, ddx, ddy, anisotropic).rgb;
    SetGltfSpecular(p, specularWeight, specularColour);
    if (anisotropyStrength > 0.0)
        SetGltfAnisotropy(p, saturate(anisotropyStrength));
    if (!simple && any(material.sheenColour > 0.0))
    {
        float3 sheen = material.sheenColour;
        if (material.sheenColourTexture != INVALIDDESCRIPTORHANDLE)
            sheen *= SampleTextureGrad(material.sheenColourTexture, uv, ddx, ddy, anisotropic).rgb;
        float sheenRoughness = material.sheenRoughness;
        if (material.sheenRoughnessTexture != INVALIDDESCRIPTORHANDLE)
            sheenRoughness *= SampleTextureGrad(material.sheenRoughnessTexture, uv, ddx, ddy, anisotropic).a;
        SetGltfSheen(p, sheen, sheenRoughness);
    }
    SetGltfClearcoat(p, coatWeight, coatRoughness, coatNormal);

    // On a thin wall the subsurface is split between diffuse reflection and diffuse transmission by its
    // anisotropy. In a solid object it is a scattering medium instead, unless the Rendering panel
    // turned all of it off.
    if (!simple && material.subsurfaceWeight > 0.0)
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

    if (simple)
        SimplifyMaterial(p);
    ApplyMaterialOverrides(p);
    return p;
}

// What LoadMaterial sets per hit, in fewer registers: halves, the shading frame octahedral. The rest
// keeps the constants of OpenPBRDefaults and SetGltfClearcoat, which the compiler folds into the BSDF.
struct PackedMaterial
{
    uint4 base;     // half: baseColour rgb, baseMetalness, specularRoughness, specularRoughnessAnisotropy, specularIor, transmissionWeight
    uint4 layers;   // half: transmissionColour rgb, subsurfaceWeight, subsurfaceColour rgb, subsurfaceScatterAnisotropy
    uint4 coatFuzz; // half: coatWeight, coatRoughness, fuzzWeight, fuzzRoughness; unorm8: fuzzColour, specularColour
    float4 medium;  // transmissionDepth, subsurfaceRadius
    uint3 basis;    // octahedral: geometryNormal, geometryTangent, geometryCoatNormal
    bool thinWalled;
};

// A weight too small for a half still counts as nonzero, so the lobes it switches on stay on.
float KeepNonzero(float weight)
{
    return weight > 0.0 ? max(weight, 6.0e-8) : 0.0;
}

PackedMaterial PackMaterial(OpenPBRBaseParams p)
{
    PackedMaterial m;
    m.base = uint4(PackHalf2(p.baseColour.rg), PackHalf2(float2(p.baseColour.b, p.baseMetalness)),
                   PackHalf2(float2(p.specularRoughness, p.specularRoughnessAnisotropy)),
                   PackHalf2(float2(p.specularIor, KeepNonzero(p.transmissionWeight))));
    m.layers = uint4(PackHalf2(p.transmissionColour.rg), PackHalf2(float2(p.transmissionColour.b, KeepNonzero(p.subsurfaceWeight))),
                     PackHalf2(p.subsurfaceColour.rg), PackHalf2(float2(p.subsurfaceColour.b, p.subsurfaceScatterAnisotropy)));
    m.coatFuzz = uint4(PackHalf2(float2(KeepNonzero(p.coatWeight), p.coatRoughness)),
                       PackHalf2(float2(KeepNonzero(p.fuzzWeight), p.fuzzRoughness)), PackUnorm3x8(p.fuzzColour),
                       PackUnorm3x8(p.specularColour));
    m.medium = float4(p.transmissionDepth, p.subsurfaceRadius);
    m.basis = uint3(OctEncode(p.geometryNormal), OctEncode(p.geometryTangent), OctEncode(p.geometryCoatNormal));
    m.thinWalled = p.geometryThinWalled;
    return m;
}

OpenPBRBaseParams UnpackMaterial(PackedMaterial m, bool simple)
{
    OpenPBRBaseParams p = OpenPBRDefaults();
    float2 h;
    h = UnpackHalf2(m.base.x);
    p.baseColour.rg = h;
    h = UnpackHalf2(m.base.y);
    p.baseColour.b = h.x;
    p.baseMetalness = h.y;
    h = UnpackHalf2(m.base.z);
    p.specularRoughness = h.x;
    p.specularRoughnessAnisotropy = h.y;
    h = UnpackHalf2(m.base.w);
    p.specularIor = h.x;
    p.transmissionWeight = h.y;

    h = UnpackHalf2(m.layers.x);
    p.transmissionColour.rg = h;
    h = UnpackHalf2(m.layers.y);
    p.transmissionColour.b = h.x;
    p.subsurfaceWeight = h.y;
    h = UnpackHalf2(m.layers.z);
    p.subsurfaceColour.rg = h;
    h = UnpackHalf2(m.layers.w);
    p.subsurfaceColour.b = h.x;
    p.subsurfaceScatterAnisotropy = h.y;

    h = UnpackHalf2(m.coatFuzz.y);
    p.fuzzWeight = h.x;
    p.fuzzRoughness = h.y;
    p.fuzzColour = UnpackUnorm3x8(m.coatFuzz.z);
    p.specularColour = UnpackUnorm3x8(m.coatFuzz.w);

    p.transmissionDepth = m.medium.x;
    p.subsurfaceRadius = m.medium.yzw;
    p.geometryThinWalled = m.thinWalled;

    // The tangent comes back perpendicular to the normal, which the quantization does not keep exactly.
    float3 N = OctDecode(m.basis.x);
    float3 T = OctDecode(m.basis.y);
    p.geometryNormal = N;
    p.geometryTangent = normalize(T - N * dot(N, T));
    p.geometryBitangent = cross(N, p.geometryTangent);
    h = UnpackHalf2(m.coatFuzz.x);
    SetGltfClearcoat(p, h.x, h.y, OctDecode(m.basis.z));
    if (simple)
        SimplifyMaterial(p);
    // The only override that sets a value SimplifyMaterial puts back to its default.
    if ((g_Frame.materialOverrides & MATERIAL_OVERRIDE_NO_SPECULAR) != 0)
        p.specularColour = float3(0.0, 0.0, 0.0);
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
float4 LoadBaseColourAlpha(MaterialData material, float2 uv)
{
    float4 albedo = material.baseColour;
    if (material.albedoTexture != INVALIDDESCRIPTORHANDLE)
        albedo *= SampleTexture(material.albedoTexture, uv);
    return albedo;
}

// What a pass-through surface lets a shadow ray through.
float3 PassThroughTransmittance(MaterialData material, float2 uv, float3 baseColour, float faceCosine)
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
    // The specular weight changes how much the Fresnel reflects back; its colour does not tint transmission.
    float specularWeight = material.specularWeight;
    if (material.specularTexture != INVALIDDESCRIPTORHANDLE)
        specularWeight *= SampleTexture(material.specularTexture, uv).a;
    SetGltfSpecular(p, specularWeight, material.specularColour);
    // The fuzz on top keeps some of it back too.
    if (any(material.sheenColour > 0.0))
    {
        float3 sheen = material.sheenColour;
        if (material.sheenColourTexture != INVALIDDESCRIPTORHANDLE)
            sheen *= SampleTexture(material.sheenColourTexture, uv).rgb;
        float sheenRoughness = material.sheenRoughness;
        if (material.sheenRoughnessTexture != INVALIDDESCRIPTORHANDLE)
            sheenRoughness *= SampleTexture(material.sheenRoughnessTexture, uv).a;
        SetGltfSheen(p, sheen, sheenRoughness);
    }
    // And so does the coat. A pass-through surface's coat is smooth, so only its weight matters.
    float coatWeight = material.coatWeight;
    if (coatWeight > 0.0 && material.coatWeightTexture != INVALIDDESCRIPTORHANDLE)
        coatWeight *= SampleTexture(material.coatWeightTexture, uv).r;
    SetGltfClearcoat(p, coatWeight, 0.0, p.geometryNormal);
    ApplyMaterialOverrides(p);
    return ThinDeltaTransmittance(p, faceCosine);
}

// What a candidate triangle lets a shadow ray along dir through: pass-through surfaces some, anything
// else nothing. False for a cut-out, which the ray ignores.
bool ShadowCandidateTransmittance(PrimitiveDesc prim, uint primitiveIndex, float2 barycentrics, float3x3 linearToWorld,
                                  float3 dir, out float3 transmitted)
{
    transmitted = float3(0.0, 0.0, 0.0);
    MaterialData material = LoadGpuMaterial(prim.materialIndex);
    uint3 tri = LoadTriangle(prim, primitiveIndex);
    float2 uv = InterpolateTexCoord(tri, Barycentrics(barycentrics));
    float4 albedo = LoadBaseColourAlpha(material, uv);

    if (material.alphaMode == ALPHA_MODE_MASK && albedo.a < material.alphaCutoff)
        return false;

    if ((material.transmissionFlags & TRANSMISSION_PASS_THROUGH) != 0)
        transmitted = PassThroughTransmittance(material, uv, albedo.rgb, FaceCosine(tri, linearToWorld, dir));
    return true;
}
