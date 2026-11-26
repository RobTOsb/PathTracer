#pragma once

#include "AnimationData.h"

namespace Animation
{
	// Linear blend skinning on the CPU: each vertex moved by the sum of its joints' matrices by weight. Positions take
	// the whole matrix; normals and tangents its 3x3, renormalised, which is right for rotations and uniform scales
	// (non-uniform scaling joints tilt the normals slightly). Large runs are split over threads.
	//
	// jointMatrices: per joint of the skin, from the bind pose to the output space (see AnimationSystem).
	void SkinVertices(std::span<const Matrix3x4> jointMatrices,
					  std::span<const VertexInfluence> influences,
					  std::span<const SkinVertex> bindVertices,
					  std::span<SkinVertex> out);
} // namespace Animation
