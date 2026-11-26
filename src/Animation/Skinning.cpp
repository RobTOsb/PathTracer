#include "Skinning.h"

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>

namespace Animation
{
	namespace
	{
		// Vertices per task: below this a run is skinned on the calling thread.
		constexpr size_t kVerticesPerTask = 4096;

		void Normalise3(float v[3])
		{
			const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
			const float scale = length > 0.0f ? 1.0f / length : 0.0f;
			v[0] *= scale;
			v[1] *= scale;
			v[2] *= scale;
		}

		void SkinRange(std::span<const Matrix3x4> jointMatrices,
					   std::span<const VertexInfluence> influences,
					   std::span<const SkinVertex> bindVertices,
					   std::span<SkinVertex> out,
					   size_t begin,
					   size_t end)
		{
			const uint32_t jointCount = static_cast<uint32_t>(jointMatrices.size());
			for (size_t v = begin; v < end; ++v)
			{
				const VertexInfluence &influence = influences[v];
				// The weighted sum of the joints' matrices, which is affine as the weights sum to 1.
				float m[12] = {};
				for (uint32_t i = 0; i < kMaxInfluences; ++i)
				{
					const float weight = influence.weights[i];
					if (weight == 0.0f || influence.joints[i] >= jointCount)
					{
						continue;
					}
					const Matrix3x4 &joint = jointMatrices[influence.joints[i]];
					for (int k = 0; k < 12; ++k)
					{
						m[k] += weight * joint[k];
					}
				}

				const SkinVertex &bind = bindVertices[v];
				SkinVertex &skinned = out[v];
				const float *p = bind.position;
				skinned.position[0] = p[0] * m[0] + p[1] * m[4] + p[2] * m[8] + m[3];
				skinned.position[1] = p[0] * m[1] + p[1] * m[5] + p[2] * m[9] + m[7];
				skinned.position[2] = p[0] * m[2] + p[1] * m[6] + p[2] * m[10] + m[11];
				const float *n = bind.normal;
				skinned.normal[0] = n[0] * m[0] + n[1] * m[4] + n[2] * m[8];
				skinned.normal[1] = n[0] * m[1] + n[1] * m[5] + n[2] * m[9];
				skinned.normal[2] = n[0] * m[2] + n[1] * m[6] + n[2] * m[10];
				Normalise3(skinned.normal);
				const float *t = bind.tangent;
				skinned.tangent[0] = t[0] * m[0] + t[1] * m[4] + t[2] * m[8];
				skinned.tangent[1] = t[0] * m[1] + t[1] * m[5] + t[2] * m[9];
				skinned.tangent[2] = t[0] * m[2] + t[1] * m[6] + t[2] * m[10];
				Normalise3(skinned.tangent);
				skinned.tangent[3] = t[3];
			}
		}
	} // namespace

	void SkinVertices(std::span<const Matrix3x4> jointMatrices,
					  std::span<const VertexInfluence> influences,
					  std::span<const SkinVertex> bindVertices,
					  std::span<SkinVertex> out)
	{
		const size_t count = std::min({ influences.size(), bindVertices.size(), out.size() });
		if (count <= kVerticesPerTask)
		{
			SkinRange(jointMatrices, influences, bindVertices, out, 0, count);
			return;
		}
		std::vector<size_t> tasks((count + kVerticesPerTask - 1) / kVerticesPerTask);
		std::iota(tasks.begin(), tasks.end(), size_t{ 0 });
		std::for_each(std::execution::par,
					  tasks.begin(),
					  tasks.end(),
					  [&](size_t task)
					  {
						  const size_t begin = task * kVerticesPerTask;
						  SkinRange(jointMatrices,
									influences,
									bindVertices,
									out,
									begin,
									std::min(begin + kVerticesPerTask, count));
					  });
	}
} // namespace Animation
