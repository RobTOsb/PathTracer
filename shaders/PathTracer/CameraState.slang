// What a path keeps for the camera: the DLSS-RR guides and what feeds them. Only the segments the
// camera sees through touch it, so between them it waits in g_CameraState rather than in registers,
// where it would be saved and restored around every TraceRay of the path. Without DLSS_RR nothing
// reads it, so it is neither stored nor loaded.
import OpenPBRBSDF;
import PathTracer.GlassSplit;
import PathTracer.Guides;
import PathTracer.Math;
import PathTracer.Psr;
import PathTracer.Resources;

struct CameraState
{
    PrimaryGuides guides; // all but the reflector, which FinishGuides sets
    SpecularChain chain;
    PaneReflection panes;
    float3 transmittance; // of the walls in front of the primary surface
    float distance; // camera to the latest hit through those walls
    float3 cameraDir; // the camera ray's direction
};

// Parts of the camera state, loaded and stored on their own so a segment only moves what it touches.
static const uint CAMERA_GUIDES = 1;
static const uint CAMERA_CHAIN = 2;
static const uint CAMERA_PANES = 4;
static const uint CAMERA_VIEW = 8; // transmittance and distance
static const uint CAMERA_DIRECTION = 16;
static const uint CAMERA_ALL = 31;

CameraState DefaultCameraState(float3 cameraDir)
{
    CameraState s = (CameraState)0;
    s.guides = DefaultGuides(cameraDir);
    s.transmittance = float3(1.0, 1.0, 1.0);
    s.cameraDir = cameraDir;
    return s;
}

// GpuCameraGuides::flags
static const uint CAMERA_GUIDES_SKY = 1; // position is a direction (w = 0)
static const uint CAMERA_GUIDES_CHAIN_MOTION = 2; // PrimaryGuides::hasChainMotion
// GpuSpecularChain::flags
static const uint CAMERA_CHAIN_PARITY_ODD = 1;
static const uint CAMERA_CHAIN_THROUGH_GLASS = 2;
static const uint CAMERA_CHAIN_DIFFERENTIALS = 4; // SpecularChain::differentialsValid

uint3 PackDifferential(RayDifferential r)
{
    return uint3(PackHalf2(r.dP.xy), PackHalf2(float2(r.dP.z, r.dD.x)), PackHalf2(r.dD.yz));
}

RayDifferential UnpackDifferential(uint3 p)
{
    float2 a = UnpackHalf2(p.x);
    float2 b = UnpackHalf2(p.y);
    float2 c = UnpackHalf2(p.z);
    RayDifferential r;
    r.dP = float3(a, b.x);
    r.dD = float3(b.y, c);
    return r;
}

GpuCameraGuides PackGuides(PrimaryGuides g)
{
    bool sky = g.position.w == 0.0;
    GpuCameraGuides p;
    p.position = g.position.xyz;
    p.normal = sky ? 0 : OctEncode(g.normal); // the sky's normal is zero
    p.albedos = uint3(PackHalf2(g.diffuseAlbedo.xy), PackHalf2(float2(g.diffuseAlbedo.z, g.specularAlbedo.x)),
                      PackHalf2(g.specularAlbedo.yz));
    p.roughnessCurvature = PackHalf2(float2(g.roughness, g.curvature));
    p.specularHitDistance = g.specularHitDistance;
    p.flags = (sky ? CAMERA_GUIDES_SKY : 0) | (g.hasChainMotion ? CAMERA_GUIDES_CHAIN_MOTION : 0);
    p.chainMotion = g.chainMotion;
    return p;
}

PrimaryGuides UnpackGuides(GpuCameraGuides p)
{
    PrimaryGuides g = (PrimaryGuides)0;
    bool sky = (p.flags & CAMERA_GUIDES_SKY) != 0;
    g.position = float4(p.position, sky ? 0.0 : 1.0);
    g.normal = sky ? float3(0.0, 0.0, 0.0) : OctDecode(p.normal);
    float2 albedos0 = UnpackHalf2(p.albedos.x);
    float2 albedos1 = UnpackHalf2(p.albedos.y);
    float2 albedos2 = UnpackHalf2(p.albedos.z);
    g.diffuseAlbedo = float3(albedos0, albedos1.x);
    g.specularAlbedo = float3(albedos1.y, albedos2);
    float2 roughnessCurvature = UnpackHalf2(p.roughnessCurvature);
    g.roughness = roughnessCurvature.x;
    g.curvature = roughnessCurvature.y;
    g.specularHitDistance = p.specularHitDistance;
    g.hasChainMotion = (p.flags & CAMERA_GUIDES_CHAIN_MOTION) != 0;
    g.chainMotion = p.chainMotion;
    return g;
}

GpuSpecularChain PackChain(SpecularChain c)
{
    GpuSpecularChain p;
    p.rotor = c.rotor;
    p.throughputFraction = uint2(PackHalf2(c.throughput.xy), PackHalf2(float2(c.throughput.z, c.primaryFraction)));
    p.primaryDistance = c.primaryDistance;
    p.length = c.length;
    p.pixelDeflection = c.pixelDeflection;
    p.flags = (c.parityOdd ? CAMERA_CHAIN_PARITY_ODD : 0) | (c.throughGlass ? CAMERA_CHAIN_THROUGH_GLASS : 0) |
              (c.differentialsValid ? CAMERA_CHAIN_DIFFERENTIALS : 0);
    p.pixelU = PackDifferential(c.pixelU);
    p.pixelV = PackDifferential(c.pixelV);
    p.motion = PackDifferential(c.motion);
    return p;
}

SpecularChain UnpackChain(GpuSpecularChain p)
{
    SpecularChain c;
    c.active = true;
    c.parityOdd = (p.flags & CAMERA_CHAIN_PARITY_ODD) != 0;
    c.throughGlass = (p.flags & CAMERA_CHAIN_THROUGH_GLASS) != 0;
    c.differentialsValid = (p.flags & CAMERA_CHAIN_DIFFERENTIALS) != 0;
    c.rotor = p.rotor;
    float2 throughput0 = UnpackHalf2(p.throughputFraction.x);
    float2 throughput1 = UnpackHalf2(p.throughputFraction.y);
    c.throughput = float3(throughput0, throughput1.x);
    c.primaryFraction = throughput1.y;
    c.primaryDistance = p.primaryDistance;
    c.length = p.length;
    c.pixelDeflection = p.pixelDeflection;
    c.pixelU = UnpackDifferential(p.pixelU);
    c.pixelV = UnpackDifferential(p.pixelV);
    c.motion = UnpackDifferential(p.motion);
    return c;
}

GpuPaneReflection PackPanes(PaneReflection r)
{
    GpuPaneReflection p;
    p.albedoLuminance = uint2(PackHalf2(r.albedo.xy), PackHalf2(float2(r.albedo.z, r.luminance)));
    p.distance = r.distance;
    p.curvature = r.curvature;
    p.hitDistance = r.hitDistance;
    return p;
}

PaneReflection UnpackPanes(GpuPaneReflection p)
{
    PaneReflection r;
    float2 albedo0 = UnpackHalf2(p.albedoLuminance.x);
    float2 albedo1 = UnpackHalf2(p.albedoLuminance.y);
    r.albedo = float3(albedo0, albedo1.x);
    r.luminance = albedo1.y;
    r.distance = p.distance;
    r.curvature = p.curvature;
    r.hitDistance = p.hitDistance;
    return r;
}

// Stores the given CAMERA_* parts. Only an active chain is stored, since only one is ever loaded.
void StoreCameraState(CameraState s, uint parts)
{
#ifdef DLSS_RR
    uint i = PixelIndex();
    if ((parts & CAMERA_GUIDES) != 0)
        g_CameraState[i].guides = PackGuides(s.guides);
    if ((parts & CAMERA_CHAIN) != 0 && s.chain.active)
        g_CameraState[i].chain = PackChain(s.chain);
    if ((parts & CAMERA_PANES) != 0)
        g_CameraState[i].panes = PackPanes(s.panes);
    if ((parts & CAMERA_VIEW) != 0)
    {
        g_CameraState[i].transmittance = uint2(PackHalf2(s.transmittance.xy), PackHalf2(float2(s.transmittance.z, 0.0)));
        g_CameraState[i].distance = s.distance;
    }
    if ((parts & CAMERA_DIRECTION) != 0)
        g_CameraState[i].cameraDirection = s.cameraDir;
#endif
}

// Loads the given CAMERA_* parts and leaves the rest zero. A chain is loaded as active.
CameraState LoadCameraState(uint parts)
{
    CameraState s = (CameraState)0;
#ifdef DLSS_RR
    uint i = PixelIndex();
    if ((parts & CAMERA_GUIDES) != 0)
        s.guides = UnpackGuides(g_CameraState[i].guides);
    if ((parts & CAMERA_CHAIN) != 0)
        s.chain = UnpackChain(g_CameraState[i].chain);
    if ((parts & CAMERA_PANES) != 0)
        s.panes = UnpackPanes(g_CameraState[i].panes);
    if ((parts & CAMERA_VIEW) != 0)
    {
        uint2 transmittance = g_CameraState[i].transmittance;
        s.transmittance = float3(UnpackHalf2(transmittance.x), UnpackHalf2(transmittance.y).x);
        s.distance = g_CameraState[i].distance;
    }
    if ((parts & CAMERA_DIRECTION) != 0)
        s.cameraDir = g_CameraState[i].cameraDirection;
#endif
    return s;
}

// The hit distances a segment can deliver to parts it does not otherwise load.
void StoreGuideHitDistance(float hitDistance)
{
#ifdef DLSS_RR
    g_CameraState[PixelIndex()].guides.specularHitDistance = hitDistance;
#endif
}

void StorePaneHitDistance(float hitDistance)
{
#ifdef DLSS_RR
    g_CameraState[PixelIndex()].panes.hitDistance = hitDistance;
#endif
}

// The guides once the path is done. The specular guides follow the reflection on the guides' surface,
// or on the brightest split pane in front of it if that reflects more.
PrimaryGuides FinishGuides(CameraState s, float3 cameraOrigin)
{
    PrimaryGuides guides = s.guides;
    guides.reflector = guides.position;
    guides.reflectorRoughness = guides.roughness;
    if (s.panes.luminance > 0.0)
    {
        bool paneReflects = s.panes.luminance >= Luminance(guides.specularAlbedo);
        guides.specularAlbedo += s.panes.albedo;
        if (paneReflects)
        {
            guides.reflector = float4(cameraOrigin + s.cameraDir * s.panes.distance, 1.0);
            guides.reflectorRoughness = 0.0;
            guides.roughness = 0.0;
            guides.curvature = s.panes.curvature;
            guides.specularHitDistance = s.panes.hitDistance;
        }
    }
    return guides;
}
