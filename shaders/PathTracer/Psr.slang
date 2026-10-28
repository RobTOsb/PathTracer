// Primary surface replacement: follows mirrors and glass in front of the camera so the DLSS-RR
// guides describe the surface seen through them.
import OpenPBRBSDF;
import PathTracer.Guides;
import PathTracer.RayTracing;
import PathTracer.Resources;

static const float PSR_MIN_SPECULAR_FRACTION = 0.9;
static const float PSR_MAX_VIRTUAL_DEPTH_JITTER = 0.002;

float2 SpecularFractions(OpenPBRBaseParams p)
{
    float M = p.baseMetalness;
    float T = p.transmissionWeight;
    float3 baseColour = p.baseWeight * p.baseColour;

    float F0 = FresnelF0(SpecularIorRatio(p));

    // F82-tint reflects baseColour head on.
    float specular = (1.0 - M) * F0 * Luminance(p.specularColour) + M * p.specularWeight * Luminance(baseColour);
    float diffuse = (1.0 - M) * (1.0 - F0) * (1.0 - T) * Luminance(lerp(baseColour, p.subsurfaceColour, p.subsurfaceWeight));
    float transmission = (1.0 - M) * (1.0 - F0) * T * Luminance(TransmissionTint(p));
    // Head-on coat over the base, then fuzz over all of it.
    float coat = 0.0;
    if (p.coatWeight > 0.0)
    {
        float coatAlbedo = CoatAlbedo(p, 1.0);
        float exponent = CoatAbsorptionExponent(p, 1.0);
        float3 transmitted = CoatTransmittance(p, coatAlbedo, exponent, 0.0);
        float reflected =
                Luminance(CoatTransmittance(p, coatAlbedo, exponent, 1.0) * CoatDarkening(p, 1.0, SpecularIorRatio(p), CoatedSpecularRoughness(p)));
        specular *= reflected;
        diffuse *= reflected;
        transmission *= Luminance(transmitted);
        coat = p.coatWeight * coatAlbedo;
    }
    float fuzzAlbedo = p.fuzzWeight > 0.0 ? p.fuzzWeight * FuzzAlbedo(1.0, FuzzRoughness(p)) : 0.0;
    float fuzz = fuzzAlbedo * Luminance(p.fuzzColour);
    float under = 1.0 - fuzzAlbedo;
    return under * float2(specular, transmission) / max(under * (specular + transmission + diffuse + coat) + fuzz, 1.0e-4);
}

bool IsPsrEvent(OpenPBRBaseParams p, OpenPBRContext ctx, uint lobe, out float fraction)
{
    fraction = 1.0;
    bool reflection = lobe == LOBE_SPECULAR_REFLECTION;
    if (!(reflection || lobe == LOBE_SPECULAR_TRANSMISSION) || CoatedSpecularRoughness(p) > g_Frame.psrMaxRoughness)
        return false;
    if (ctx.inside)
        return reflection == (FresnelDielectric(ctx.wo.z, ctx.eta) >= 0.5);
    float2 fractions = SpecularFractions(p);
    fraction = reflection ? fractions.x : fractions.y;
    return fraction >= PSR_MIN_SPECULAR_FRACTION;
}

// Whether a chain that ends at this surface ended only because the path sampled the weaker of its specular
// lobes, where the stronger one would have been an event: smooth enough, and inside glass the lobe the chain
// follows, outside one that takes at least PSR_MIN_SPECULAR_FRACTION of the light. Which lobe a path takes
// is random each frame, so such a frame's guides would flip between the chain's end and this surface.
bool PsrChainBrokenByChance(OpenPBRBaseParams p, OpenPBRContext ctx, uint lobe)
{
    bool reflection = lobe == LOBE_SPECULAR_REFLECTION;
    if (!(reflection || lobe == LOBE_SPECULAR_TRANSMISSION) || CoatedSpecularRoughness(p) > g_Frame.psrMaxRoughness)
        return false;
    if (ctx.inside)
        return reflection != (FresnelDielectric(ctx.wo.z, ctx.eta) >= 0.5);
    float2 fractions = SpecularFractions(p);
    return (reflection ? fractions.y : fractions.x) >= PSR_MIN_SPECULAR_FRACTION;
}

// The lobe of smooth solid glass a chain follows where the path could take either: inside the one Fresnel
// favours, outside the one IsPsrEvent takes. LOBE_NONE where the glass has other lobes or is too rough for PSR.
uint PsrStrongerLobe(OpenPBRBaseParams p, OpenPBRContext ctx)
{
    if (p.geometryThinWalled || ctx.deltaTransmission || CoatedSpecularRoughness(p) > g_Frame.psrMaxRoughness ||
        ctx.probDiffuse > 0.0 || ctx.probCoat > 0.0 || ctx.probFuzz > 0.0 || ctx.probSpecR <= 0.0 || ctx.probSpecT <= 0.0)
        return LOBE_NONE;
    if (ctx.inside)
        return FresnelDielectric(ctx.wo.z, ctx.eta) >= 0.5 ? LOBE_SPECULAR_REFLECTION : LOBE_SPECULAR_TRANSMISSION;
    float2 fractions = SpecularFractions(p);
    if (fractions.y >= PSR_MIN_SPECULAR_FRACTION)
        return LOBE_SPECULAR_TRANSMISSION;
    return fractions.x >= PSR_MIN_SPECULAR_FRACTION ? LOBE_SPECULAR_REFLECTION : LOBE_NONE;
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
        ReflectanceGuides(p, ctx, wo, reflectance);
        return reflectance.specularAlbedo;
    }
    return (1.0 - p.fuzzWeight * ctx.fuzzE) * (1.0 - p.baseMetalness) * p.transmissionWeight * (1.0 - ctx.E) *
           TransmissionTint(p) * CoatBaseScale(p, ctx, false, 0.0);
}

// First-order change of a ray (Igehy 1999): of its point on the surface it last met, and of its direction.
struct RayDifferential
{
    float3 dP;
    float3 dD;
};

// Moves the differential along a ray from its surface to the next, of normal Ng, t further along dir.
// The point's change keeps to that surface's plane.
void TransferDifferential(inout RayDifferential r, float3 dir, float t, float3 Ng)
{
    float3 dP = r.dP + t * r.dD;
    r.dP = dP - dir * (dot(dP, Ng) / dot(dir, Ng));
}

// Turns the differential with a ray that arrived along dirIn and left along dirOut, about shading normal
// N, by reflection or by refraction with eta = incident / transmitted IOR. The normal's change comes
// from the hit's shape (the normal map's is left out).
void TurnDifferential(inout RayDifferential r, HitInfo hit, float3 N, float3 dirIn, float3 dirOut, bool reflection, float eta)
{
    float3 dN = NormalDerivative(hit, r.dP);
    float DN = dot(dirIn, N);
    float dDN = dot(r.dD, N) + dot(dirIn, dN);
    if (reflection)
    {
        r.dD -= 2.0 * (dDN * N + DN * dN);
    }
    else
    {
        // dirOut = eta dirIn - mu N
        float TN = dot(dirOut, N);
        float mu = eta * DN - TN;
        float dMu = (eta - eta * eta * DN / TN) * dDN;
        r.dD = eta * r.dD - (dMu * N + mu * dN);
    }
}

// The turn of the camera ray along dir by one pixel's angle, across and up the screen, at right angles
// to dir and to each other.
void CameraPixelAxes(float3 dir, out float3 u, out float3 v)
{
    u = normalize(cross(g_Frame.invView[1].xyz, dir)) * g_Frame.rayConePixelSpread;
    v = cross(dir, u);
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
    // The chain's path as the camera ray turns by pixelU or pixelV (CameraPixelAxes), or starts from last
    // frame's camera (motion), up to the latest hit. Only while differentialsValid, which the chain loses
    // to a grazing hit or refraction.
    RayDifferential pixelU;
    RayDifferential pixelV;
    RayDifferential motion;
    bool differentialsValid;
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

// Carries the differentials along the segment that arrived at hit along dir.
void ChainArrive(inout SpecularChain chain, float3 dir, HitInfo hit)
{
    if (!chain.differentialsValid)
        return;
    chain.differentialsValid = abs(dot(dir, hit.Ng)) > 1.0e-4;
    TransferDifferential(chain.pixelU, dir, hit.t, hit.Ng);
    TransferDifferential(chain.pixelV, dir, hit.t, hit.Ng);
    TransferDifferential(chain.motion, dir, hit.t, hit.Ng);
}

// Appends the turn of the event where the path arrived at hit along dirIn and left along dirOut, about the
// surface's geometric normal, by reflection or by refraction with ctxEta (OpenPBRContext::eta). The caller
// multiplies in its throughput (PsrEventThroughput).
void AddPsrEvent(inout SpecularChain chain, HitInfo hit, float3 normal, bool thinWalled, float ctxEta, float3 dirIn,
                 float3 dirOut, bool reflection, float pixelFootprint)
{
    // A thin wall's transmission mirrors its reflection back through the wall, which leaves the
    // direction alone however the surface turns.
    if (chain.differentialsValid && (reflection || !thinWalled))
    {
        float eta = 1.0 / ctxEta;
        chain.differentialsValid = reflection || abs(dot(dirOut, normal)) > 1.0e-4;
        TurnDifferential(chain.pixelU, hit, normal, dirIn, dirOut, reflection, eta);
        TurnDifferential(chain.pixelV, hit, normal, dirIn, dirOut, reflection, eta);
        TurnDifferential(chain.motion, hit, normal, dirIn, dirOut, reflection, eta);
    }

    // How much a turn of the normal turns the path: twice for a reflection (or a thin wall's mirrored
    // transmission), 1 - 1/eta for a refraction.
    float deflection;
    if (reflection)
    {
        chain.rotor = QuaternionMultiply(chain.rotor, MirrorRotor(normal, chain.parityOdd));
        chain.parityOdd = !chain.parityOdd;
        deflection = 2.0;
    }
    else
    {
        // Behind an odd number of reflections the rotation is conjugated by F, which negates the x and
        // y of its quaternion.
        float4 q = ArcRotor(dirOut, dirIn);
        chain.rotor = QuaternionMultiply(chain.rotor, chain.parityOdd ? float4(-q.xy, q.zw) : q);
        deflection = thinWalled ? 2.0 : abs(1.0 - 1.0 / ctxEta);
    }
    chain.pixelDeflection = max(chain.pixelDeflection, deflection * HitCurvature(hit) * pixelFootprint);
}

// Starts the chain at the primary surface, hit primaryDistance from the camera along dirIn.
SpecularChain StartSpecularChain(OpenPBRBaseParams p, OpenPBRContext ctx, HitInfo hit, float3 wo, float3 dirIn, float3 dirOut,
                                 bool reflection, float fraction, float primaryDistance, float pixelFootprint)
{
    SpecularChain chain;
    chain.active = true;
    chain.parityOdd = false;
    chain.rotor = float4(0.0, 0.0, 0.0, 1.0);
    chain.throughput = PsrEventThroughput(p, ctx, wo, reflection);
    chain.primaryFraction = fraction;
    chain.primaryDistance = primaryDistance;
    chain.length = 0.0;
    chain.pixelDeflection = 0.0;
    chain.throughGlass = !reflection && !p.geometryThinWalled;

    // From the camera, through any panes on the way, which leave the direction alone.
    float3 u, v;
    CameraPixelAxes(dirIn, u, v);
    chain.pixelU.dP = float3(0.0, 0.0, 0.0);
    chain.pixelU.dD = u;
    chain.pixelV.dP = float3(0.0, 0.0, 0.0);
    chain.pixelV.dD = v;
    chain.motion.dP = g_Frame.prevCameraPosition - g_Frame.invView[3].xyz;
    chain.motion.dD = float3(0.0, 0.0, 0.0);
#ifdef DLSS_RR
    chain.differentialsValid = g_Frame.psrMaxMinification > 0.0;
#else
    chain.differentialsValid = false;
#endif
    HitInfo arrival = hit;
    arrival.t = primaryDistance;
    ChainArrive(chain, dirIn, arrival);

    AddPsrEvent(chain, hit, p.geometryNormal, p.geometryThinWalled, ctx.eta, dirIn, dirOut, reflection, pixelFootprint);
    return chain;
}

// The motion vector of the point the chain ends on, which it reached along endDir at a surface of
// normal endNg after pathLength, and how much the chain shrinks that surface's image (linear, against
// seeing it straight on from the same distance).
// Last frame's camera saw the point along cameraDir + a u + b v (CameraPixelAxes), where a and b undo
// the camera's move at the end surface: motion.dP + a pixelU.dP + b pixelV.dP = 0, all three in its
// plane, solved in the least squares sense.
bool ChainMotionVector(SpecularChain chain, float3 cameraDir, float3 endDir, float3 endNg, float pathLength, out float2 motion,
                       out float minification)
{
    motion = float2(0.0, 0.0);
    minification = 0.0;
    if (!chain.differentialsValid)
        return false;
    float3 du = chain.pixelU.dP;
    float3 dv = chain.pixelV.dP;
    float3 dm = chain.motion.dP;
    float uu = dot(du, du);
    float uv = dot(du, dv);
    float vv = dot(dv, dv);
    float det = uu * vv - uv * uv;
    // A fold of the image (a caustic, a glass's silhouette) maps the pixel onto a line. NaN fails here too.
    if (!(det > 1.0e-4 * uu * vv))
        return false;
    float um = dot(du, dm);
    float vm = dot(dv, dm);
    float a = (uv * vm - vv * um) / det;
    float b = (uv * um - uu * vm) / det;

    float3 u, v;
    CameraPixelAxes(cameraDir, u, v);
    // Every point on a camera ray projects where its direction does.
    float4 currClip = mul(float4(cameraDir, 0.0), g_Frame.viewProj);
    float4 prevClip = mul(float4(cameraDir + a * u + b * v, 0.0), g_Frame.prevViewProj);
    if (currClip.w <= 0.0 || prevClip.w <= 0.0)
        return false;
    motion = CalculateMotionVector(currClip, prevClip, float2(DispatchRaysDimensions().xy));

    float pixelWidth = g_Frame.rayConePixelSpread * pathLength;
    minification = sqrt(length(cross(du, dv)) * abs(dot(endDir, endNg))) / pixelWidth;
    return all(isfinite(motion)) && isfinite(minification);
}

float3 ApplySpecularChain(SpecularChain chain, float3 v)
{
    if (chain.parityOdd)
        v.z = -v.z;
    float3 t = 2.0 * cross(chain.rotor.xyz, v);
    return v + chain.rotor.w * t + cross(chain.rotor.xyz, t);
}

// The surface a chain ended on and the ray it arrived along, all that ReplacePrimarySurface needs of them.
struct PsrChainEnd
{
    float3 diffuseAlbedo; // the surface's reflectance guides, as the chain's last ray sees it
    float3 specularAlbedo;
    float3 normal; // geometric
    float roughness; // GuideRoughness
    float curvature; // ViewedCurvature
    float3 dir; // the chain's last ray
    float3 Ng;
    float t; // its length
    float3 throughputScale; // what the chain's throughput takes on along that ray and at the surface
    // The surface's specular hit distance, InitialSpecularHitDistance until the path's next hit gives it.
    float hitDistance;
    bool replaceable; // not inside glass
};

// Swap the primary surface's guides for the surface the chain ends on. A chain with a motion vector
// through it stands unless it shrinks the surface's image past psrMaxMinification; one without, unless
// an event on the way is curved enough to smear the virtual image. Returns whether it did.
bool ReplacePrimarySurface(SpecularChain chain, PsrChainEnd end, float3 cameraDir, inout PrimaryGuides guides)
{
    float pathLength = chain.primaryDistance + chain.length;
    float2 motion;
    float minification;
    bool hasMotion = ChainMotionVector(chain, cameraDir, end.dir, end.Ng, pathLength, motion, minification);
    if (hasMotion ? minification > g_Frame.psrMaxMinification
                  : chain.pixelDeflection * chain.length > PSR_MAX_VIRTUAL_DEPTH_JITTER * pathLength)
        return false;
    guides.hasMotion = hasMotion;
    guides.motion = motion;
    guides.displacement = float3(0.0, 0.0, 0.0);
    guides.replaced = true;

    // The chain's transform is a rotation or reflection, which preserves the angle between the view and
    // the normal, so the reflectance seen in virtual space is the real one. What the primary surface shows
    // of the surface comes scaled by the chain's throughput, on top of the primary's own (small) diffuse part.
    guides.position = float4(guides.position.xyz + cameraDir * chain.length, 1.0);
    guides.normal = normalize(ApplySpecularChain(chain, end.normal));
    guides.roughness = end.roughness;
    guides.diffuseAlbedo += chain.throughput * end.diffuseAlbedo;
    guides.specularAlbedo = lerp(guides.specularAlbedo, chain.throughput * end.specularAlbedo, chain.primaryFraction);
    guides.specularHitDistance = end.hitDistance;
    // The chain's rotations and reflections keep the viewer on the same side of the surface.
    guides.curvature = end.curvature;
    return true;
}
