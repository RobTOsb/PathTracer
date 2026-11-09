#ifndef PHOTON_INTEROP_H
#define PHOTON_INTEROP_H

#include "ShaderInterop.h"

// Caustic photons, after Yang and Ouyang's Adaptive Anisotropic Photon Scattering (Ray Tracing Gems II, ch. 30):
// PhotonRayGen traces sun photons from a light-space photon map through smooth glass and mirrors, carrying photon
// differentials, and records where they land (PathTracer/Photons.slang). The path tracer marks the surface each pixel
// sees directly as a receiver; PhotonMap.slang then splats every record as an anisotropic Gaussian footprint onto the
// receivers, adds the caustics to the image, and adapts the photon map's density from feedback: how many pixels the
// photons' footprints cover and how much the caustics changed since last frame.

// Photon tracing threads per DispatchRays row.
#define PHOTON_DISPATCH_WIDTH 1024u
// Threads per group of PhotonMap.slang's one-dimensional passes, and the side of its screen passes' groups.
#define PHOTON_GROUP_SIZE 256u
#define PHOTON_SCREEN_GROUP_SIZE 8u
// Caster rectangles on the sun's plane, at most; more casters share one rectangle around them all.
#define PHOTON_MAX_RECTS 256u
// The photon map: a square of PHOTON_MAP_SIZE^2 texels on the plane perpendicular to the sun, over the casters'
// rectangles, stored in Hilbert curve order and scanned in blocks of PHOTON_MAP_BLOCK texels.
#define PHOTON_MAP_ORDER 9u
#define PHOTON_MAP_SIZE 512u
#define PHOTON_MAP_TEXELS (PHOTON_MAP_SIZE * PHOTON_MAP_SIZE)
#define PHOTON_MAP_BLOCK 512u
#define PHOTON_MAP_BLOCKS (PHOTON_MAP_TEXELS / PHOTON_MAP_BLOCK)
// Photon records a frame, at most; more are lost.
#define PHOTON_RECORD_CAPACITY (1u << 20)
// Splatted irradiance in fixed point: PhotonParameters::irradianceUnit per unit.
#define PHOTON_IRRADIANCE_SCALE 4096.0f
// Pixels one footprint's splat visits at most: a wider one visits every few, weighted up to match.
#define PHOTON_MAX_SPLAT_PIXELS 1024u
// The feedback's sums in fixed point: units per pixel of footprint area, and per unit of caustics variance.
#define PHOTON_AREA_SCALE 16.0f
#define PHOTON_VARIANCE_SCALE 4096.0f

// One caster's bounds projected onto the plane perpendicular to the sun (PhotonParameters basisU, basisV). The photon
// map only emits from texels that overlap one.
struct GpuPhotonRect
{
	float2 lo;
	float2 hi;
};

// Where a photon left power on a surface with a diffuse part, after at least one smooth bounce.
struct GpuPhotonRecord
{
	float3 position;
	uint normal; // OctEncode of the geometric normal, on the side the photon came from
	float3 power; // flux
	uint texel; // the photon map texel it was emitted from, in Hilbert order
	// Its footprint's covariance on the surface, xx, xy, yy, in the frame BuildOrthonormalBasis(OctDecode(normal)):
	// its own spread (photons.footprintScale times its positional differentials) plus the sun disc's blur.
	float3 covariance;
	float area; // the area it stands for on the surface: its positional differentials' parallelogram
};

// FrameConstants::photons.
struct PhotonParameters
{
	float3 basisU; // perpendicular to the sun direction, with basisV
	float planeHeight; // along FrameConstants::sunDirection, above every caster: where the photons start
	float3 basisV;
	float mapScale; // the photon map's side on the plane
	float2 mapOrigin; // the photon map's corner on the plane (basisU, basisV coordinates)
	uint enabled; // 0 = no photons, and the path tracer marks no receivers
	uint photonCount; // traced this frame
	uint rects; // bindless StructuredBuffer<GpuPhotonRect>
	uint rectCount;
	uint maxBounces; // smooth bounces a photon may take
	float maxAlpha; // GGX alpha up to which a lobe counts as smooth, for photons and camera paths alike
	uint density; // bindless RWStructuredBuffer<float>: each texel's share of the photons, in Hilbert order
	// bindless RWStructuredBuffer<float>: the shares of the blocks before each block, the total at PHOTON_MAP_BLOCKS
	uint blockPrefix;
	uint texelPrefix; // bindless RWStructuredBuffer<float>: the shares of a block's texels up to each, inclusive
	uint records; // bindless RWStructuredBuffer<GpuPhotonRecord>
	uint recordCount; // bindless RWStructuredBuffer<uint>: [0] = records written this frame, past the capacity too
	uint feedback; // bindless RWStructuredBuffer<uint4>: per texel, fixed point area and variance sums and photons
	// bindless RWStructuredBuffer<float4>, two per pixel: the receiver's position (w = 1, 0 for none) and its diffuse
	// reflectance over pi times the path's throughput (rgb), w = asfloat(OctEncode of its normal towards the camera)
	uint receivers;
	uint width; // render size
	float footprintScale; // a photon's footprint is this times its positional differentials across (1 sigma = half)
	float sunSpread; // standard deviation of the sun disc's directions about its centre, radians (0 for a delta sun)
	float irradianceUnit; // the splat's fixed point unit, the sun's peak irradiance / PHOTON_IRRADIANCE_SCALE
	uint height;
};

// PhotonMap.slang's root constants.
struct PhotonPassConstants
{
	uint densityPrevious; // bindless RWStructuredBuffer<float>: last frame's shares
	uint historyPrevious; // bindless RWStructuredBuffer<float2>: last frame's caustics luminance and variance per pixel
	uint history; // this frame's
	uint caustics; // bindless RWStructuredBuffer<uint4>: this frame's splatted irradiance per pixel, fixed point
	uint output; // bindless RWTexture2D<float4>: the path tracer's radiance
	uint reset; // 1: every covered texel starts from an even share, and the feedback is dropped
	float targetArea; // pixels a photon's area should cover on screen
	float varianceGain; // photons per texel a caustics variance of 1 adds
	float temporalWeight; // weight of the suggested share against the filtered last one
	float minDensity; // photons per covered texel, at least
	float maxDensity; // photons per texel, at most
	uint frameIndex;
};

#endif
