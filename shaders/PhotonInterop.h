#ifndef PHOTON_INTEROP_H
#define PHOTON_INTEROP_H

#include "ShaderInterop.h"

// Caustic photons (PathTracer/Photons.slang): sun photons traced through smooth glass and mirrors into a world-space
// hash grid of irradiance, which the path tracer reads at the first diffuse surface of a camera path that has only met
// smooth surfaces. PhotonResolve.slang turns each frame's deposits into a running average.

// Rows of the grid, a power of two.
#define PHOTON_GRID_CAPACITY (1u << 21)
// Slots a key may be found in after its hashed one (linear probing).
#define PHOTON_GRID_PROBES 16u
// A deposit's power in fixed point: this many units per FrameConstants::photons.unitPower.
#define PHOTON_FIXED_POINT_SCALE 1024.0f
// Most units one deposit may add to a channel, so a frame's sum stays far from overflowing.
#define PHOTON_MAX_DEPOSIT_UNITS 4194304.0f
// The cell level is stored with this bias in the top bits of the accum row's w, above the frame's photon count.
#define PHOTON_LEVEL_BIAS 64
#define PHOTON_LEVEL_SHIFT 24u
#define PHOTON_COUNT_MASK 0xFFFFFFu
// Levels a photon is deposited in: the two around its point's level, and the two above for sparse light.
#define PHOTON_DEPOSIT_LEVELS 4
// Photon tracing threads per DispatchRays row.
#define PHOTON_DISPATCH_WIDTH 1024u
#define PHOTON_RESOLVE_GROUP_SIZE 256u
// Caster rectangles the sun photons are aimed at, at most; more casters share one rectangle around them all.
#define PHOTON_MAX_RECTS 256u

// One caster's bounds projected onto the plane perpendicular to the sun (FrameConstants::photons basisU, basisV), and
// the CDF of picking the rectangles in proportion to their areas (this one's upper end).
struct GpuPhotonRect
{
	float2 lo;
	float2 hi;
	float cdf;
	float pad0;
	float pad1;
	float pad2;
};

// FrameConstants::photons.
struct PhotonParameters
{
	float3 basisU; // perpendicular to the sun direction, with basisV
	float planeHeight; // along FrameConstants::sunDirection, above every caster: where the photons start
	float3 basisV;
	float totalArea; // of the rectangles, overlaps counted once per rectangle
	uint enabled; // 0 = no photons, and the path tracer reads none
	uint rectCount;
	uint rects; // bindless StructuredBuffer<GpuPhotonRect>
	uint photonCount; // traced this frame
	uint gridKeys; // bindless RWStructuredBuffer<uint>: a cell's key, 0 for an empty row
	// bindless RWStructuredBuffer<uint4>: this frame's fixed point power, w = photon count | (level + PHOTON_LEVEL_BIAS) <<
	// PHOTON_LEVEL_SHIFT
	uint gridAccum;
	// bindless RWStructuredBuffer<float4>: running average irradiance, w = asuint(half average photon count | idle << 16)
	uint gridCache;
	uint maxBounces; // smooth bounces a photon may take
	float unitPower; // the power a photon deposit counts in units of, see PHOTON_FIXED_POINT_SCALE
	float maxAlpha; // GGX alpha up to which a lobe counts as smooth, for photons and camera paths alike
	float cellPixels; // cell size in pixels of footprint at the cell's distance from the camera
	// Average photons per frame a cell needs for its own estimate; sparser cells give way to the cells two levels up.
	float minPhotons;
	// Average photons per frame the cells two levels up need for full brightness; sparser ones fade to black, the light
	// spread too thin far from its caster to be estimated without sparkling. 0 = no fade.
	float minWidePhotons;
	float pad0;
	float pad1;
	float pad2;
};

// PhotonResolve.slang's root constants.
struct PhotonResolveConstants
{
	uint gridKeys;
	uint gridAccum;
	uint gridCache;
	uint reset; // 1: empty the grid instead
	float unitScale; // unitPower / PHOTON_FIXED_POINT_SCALE
	float blend; // the running average's weight of a new frame, for cells averaging minPhotons photons a frame or more
	uint evictFrames; // frames without a deposit before a cell is emptied
	// Sparser cells weigh a new frame's irradiance by blend scaled by their photons per frame over minPhotons, down
	// to minBlend, so that every cell averages over about as many photons.
	float minBlend;
	float minPhotons;
	uint pad0;
	uint pad1;
	uint pad2;
};

#endif
