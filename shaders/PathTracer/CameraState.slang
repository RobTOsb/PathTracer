// What a path keeps for the camera: the DLSS-RR guides and what feeds them. Only the segments the
// camera sees through touch it, so between them it waits in g_CameraState rather than in registers,
// where it would be saved and restored around every TraceRay of the path. Without DLSS_RR nothing
// reads it, so it is neither stored nor loaded.
import OpenPBRBaseSubstrate;
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
};

CameraState DefaultCameraState(float3 cameraDir)
{
    CameraState s = (CameraState)0;
    s.guides = DefaultGuides(cameraDir);
    s.transmittance = float3(1.0, 1.0, 1.0);
    return s;
}

// GpuCameraState::flags
static const uint CAMERA_STATE_SKY = 1; // guides.position is a direction (w = 0)
static const uint CAMERA_STATE_CHAIN_ACTIVE = 2;
static const uint CAMERA_STATE_CHAIN_PARITY_ODD = 4;
static const uint CAMERA_STATE_CHAIN_THROUGH_GLASS = 8;
static const uint CAMERA_STATE_CHAIN_DIFFERENTIALS = 16; // SpecularChain::differentialsValid
static const uint CAMERA_STATE_CHAIN_MOTION = 32; // PrimaryGuides::hasChainMotion

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

uint CameraStateIndex()
{
    uint2 pixel = DispatchRaysIndex().xy;
    return pixel.y * DispatchRaysDimensions().x + pixel.x;
}

void StoreCameraState(CameraState s)
{
#ifdef DLSS_RR
    bool sky = s.guides.position.w == 0.0;
    GpuCameraState p;
    p.guidePosition = s.guides.position.xyz;
    p.guideNormal = sky ? 0 : OctEncode(s.guides.normal); // the sky's normal is zero
    p.guideAlbedos = uint3(PackHalf2(s.guides.diffuseAlbedo.xy),
                           PackHalf2(float2(s.guides.diffuseAlbedo.z, s.guides.specularAlbedo.x)),
                           PackHalf2(s.guides.specularAlbedo.yz));
    p.guideRoughnessCurvature = PackHalf2(float2(s.guides.roughness, s.guides.curvature));
    p.guideSpecularHitDistance = s.guides.specularHitDistance;
    p.chainPrimaryDistance = s.chain.primaryDistance;
    p.chainLength = s.chain.length;
    p.chainPixelDeflection = s.chain.pixelDeflection;
    p.chainRotor = s.chain.rotor;
    p.chainThroughputFraction = uint2(PackHalf2(s.chain.throughput.xy),
                                      PackHalf2(float2(s.chain.throughput.z, s.chain.primaryFraction)));
    p.paneAlbedoLuminance = uint2(PackHalf2(s.panes.albedo.xy), PackHalf2(float2(s.panes.albedo.z, s.panes.luminance)));
    p.paneDistance = s.panes.distance;
    p.paneCurvatureHitDistance = PackHalf2(float2(s.panes.curvature, s.panes.hitDistance));
    p.transmittance = uint2(PackHalf2(s.transmittance.xy), PackHalf2(float2(s.transmittance.z, 0.0)));
    p.distance = s.distance;
    p.flags = (sky ? CAMERA_STATE_SKY : 0) | (s.chain.active ? CAMERA_STATE_CHAIN_ACTIVE : 0) |
              (s.chain.parityOdd ? CAMERA_STATE_CHAIN_PARITY_ODD : 0) |
              (s.chain.throughGlass ? CAMERA_STATE_CHAIN_THROUGH_GLASS : 0) |
              (s.chain.differentialsValid ? CAMERA_STATE_CHAIN_DIFFERENTIALS : 0) |
              (s.guides.hasChainMotion ? CAMERA_STATE_CHAIN_MOTION : 0);
    // The differentials are done with once the chain ends, which is when it leaves its motion vector.
    if (s.chain.active)
    {
        p.chainPixelU = PackDifferential(s.chain.pixelU);
        p.chainPixelV = PackDifferential(s.chain.pixelV);
        p.chainMotion = PackDifferential(s.chain.motion);
    }
    else
    {
        p.chainPixelU = uint3(asuint(s.guides.chainMotion), 0);
        p.chainPixelV = uint3(0, 0, 0);
        p.chainMotion = uint3(0, 0, 0);
    }
    g_CameraState[CameraStateIndex()] = p;
#endif
}

CameraState LoadCameraState()
{
    CameraState s = (CameraState)0;
#ifdef DLSS_RR
    GpuCameraState p = g_CameraState[CameraStateIndex()];
    bool sky = (p.flags & CAMERA_STATE_SKY) != 0;
    s.guides.position = float4(p.guidePosition, sky ? 0.0 : 1.0);
    s.guides.normal = sky ? float3(0.0, 0.0, 0.0) : OctDecode(p.guideNormal);
    float2 albedos0 = UnpackHalf2(p.guideAlbedos.x);
    float2 albedos1 = UnpackHalf2(p.guideAlbedos.y);
    float2 albedos2 = UnpackHalf2(p.guideAlbedos.z);
    s.guides.diffuseAlbedo = float3(albedos0, albedos1.x);
    s.guides.specularAlbedo = float3(albedos1.y, albedos2);
    float2 roughnessCurvature = UnpackHalf2(p.guideRoughnessCurvature);
    s.guides.roughness = roughnessCurvature.x;
    s.guides.curvature = roughnessCurvature.y;
    s.guides.specularHitDistance = p.guideSpecularHitDistance;
    s.chain.active = (p.flags & CAMERA_STATE_CHAIN_ACTIVE) != 0;
    s.chain.parityOdd = (p.flags & CAMERA_STATE_CHAIN_PARITY_ODD) != 0;
    s.chain.throughGlass = (p.flags & CAMERA_STATE_CHAIN_THROUGH_GLASS) != 0;
    s.chain.differentialsValid = (p.flags & CAMERA_STATE_CHAIN_DIFFERENTIALS) != 0;
    s.chain.pixelU = UnpackDifferential(p.chainPixelU);
    s.chain.pixelV = UnpackDifferential(p.chainPixelV);
    s.chain.motion = UnpackDifferential(p.chainMotion);
    s.guides.hasChainMotion = (p.flags & CAMERA_STATE_CHAIN_MOTION) != 0;
    s.guides.chainMotion = asfloat(p.chainPixelU.xy);
    s.chain.primaryDistance = p.chainPrimaryDistance;
    s.chain.length = p.chainLength;
    s.chain.pixelDeflection = p.chainPixelDeflection;
    s.chain.rotor = p.chainRotor;
    float2 throughput0 = UnpackHalf2(p.chainThroughputFraction.x);
    float2 throughput1 = UnpackHalf2(p.chainThroughputFraction.y);
    s.chain.throughput = float3(throughput0, throughput1.x);
    s.chain.primaryFraction = throughput1.y;
    float2 pane0 = UnpackHalf2(p.paneAlbedoLuminance.x);
    float2 pane1 = UnpackHalf2(p.paneAlbedoLuminance.y);
    s.panes.albedo = float3(pane0, pane1.x);
    s.panes.luminance = pane1.y;
    s.panes.distance = p.paneDistance;
    float2 paneCurvatureHitDistance = UnpackHalf2(p.paneCurvatureHitDistance);
    s.panes.curvature = paneCurvatureHitDistance.x;
    s.panes.hitDistance = paneCurvatureHitDistance.y;
    s.transmittance = float3(UnpackHalf2(p.transmittance.x), UnpackHalf2(p.transmittance.y).x);
    s.distance = p.distance;
#endif
    return s;
}

// The guides once the path is done. The specular guides follow the reflection on the guides' surface,
// or on the brightest split pane in front of it if that reflects more.
PrimaryGuides FinishGuides(CameraState s, float3 cameraOrigin, float3 cameraDir)
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
            guides.reflector = float4(cameraOrigin + cameraDir * s.panes.distance, 1.0);
            guides.reflectorRoughness = 0.0;
            guides.roughness = 0.0;
            guides.curvature = s.panes.curvature;
            guides.specularHitDistance = s.panes.hitDistance;
        }
    }
    return guides;
}
