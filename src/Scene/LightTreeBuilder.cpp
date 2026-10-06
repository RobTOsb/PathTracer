#include "LightTreeBuilder.h"

#include <algorithm>
#include <cfloat>
#include <math.h>

namespace
{
	float Length(const float3 &v)
	{
		return sqrtf(Dot(v, v));
	}

	float3 Min(const float3 &a, const float3 &b)
	{
		return { std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) };
	}

	float3 Max(const float3 &a, const float3 &b)
	{
		return { std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) };
	}

	float Component(const float3 &v, int axis)
	{
		return (&v.x)[axis];
	}

	float3 XYZ(const float4 &v)
	{
		return { v.x, v.y, v.z };
	}

	float Luminance(const float3 &c)
	{
		return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
	}

	struct Bounds
	{
		float3 lo = { FLT_MAX, FLT_MAX, FLT_MAX };
		float3 hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

		void Grow(const float3 &p)
		{
			lo = Min(lo, p);
			hi = Max(hi, p);
		}

		void Grow(const Bounds &b)
		{
			lo = Min(lo, b.lo);
			hi = Max(hi, b.hi);
		}

		bool IsValid() const
		{
			return lo.x <= hi.x;
		}

		float SurfaceArea() const
		{
			if (!IsValid())
			{
				return 0.0f;
			}
			const float3 d = hi - lo;
			return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
		}
	};

	struct Primitive
	{
		Bounds bounds;
		float3 centroid;
		float flux;
		uint32_t light;
	};

	// Cluster parameters of a single light, the leaf case of Eqs. 2 to 5.
	struct LeafCluster
	{
		float3 mean;
		float variance;
		float radius;
		float flux;
		float3 averageDirection; // nu bar in Eq. 2, 0 for an isotropic emitter
	};

	struct BuildTask
	{
		uint32_t node;
		uint32_t begin;
		uint32_t end;
		Bounds centroidBounds; // of [begin, end), only filled for ranges large enough to be binned
	};

	struct Bin
	{
		Bounds bounds;
		Bounds centroidBounds;
		float flux = 0.0f;
		uint32_t count = 0;
	};

	constexpr uint32_t kMaxBins = 32;
	// Ranges up to this size get an exact SAH sweep over every split instead of binning.
	constexpr uint32_t kSweepMaxCount = 8;

	uint32_t SplitImbalance(uint32_t leftCount, uint32_t count)
	{
		return leftCount * 2 > count ? leftCount * 2 - count : count - leftCount * 2;
	}

	uint32_t BinIndex(float c, float lo, float scale, uint32_t binCount)
	{
		return std::min(binCount - 1, static_cast<uint32_t>((c - lo) * scale));
	}

	Bounds CentroidBounds(const std::vector<Primitive> &prims, uint32_t begin, uint32_t end)
	{
		Bounds bounds;
		for (uint32_t i = begin; i < end; ++i)
		{
			bounds.Grow(prims[i].centroid);
		}
		return bounds;
	}

	// Fills the leaf cluster and the build primitive of one light. Returns false for light types the
	// tree does not handle and for lights that emit nothing.
	bool MakeLeaf(GpuLight light, LeafCluster &leaf, Primitive &primitive)
	{
		const GpuLightType type = light.GetLightType();
		if (type == kLight_Point)
		{
			const float3 position = XYZ(light.v1);
			const float radius = light.v2.x;
			const float3 extent = { radius, radius, radius };
			leaf.mean = position;
			leaf.variance = radius * radius;
			leaf.radius = radius;
			leaf.flux = 4.0f * Pi * Luminance(XYZ(light.radiance));
			leaf.averageDirection = { 0.0f, 0.0f, 0.0f };
			primitive.bounds.Grow(position - extent);
			primitive.bounds.Grow(position + extent);
		}
		else if (type == kLight_Area)
		{
			// One-sided Lambertian triangle, Phi = pi L A. A radiance map is not accounted for, the
			// radiance factor alone sets the flux.
			const float3 p0 = XYZ(light.v1);
			const float3 p1 = XYZ(light.v2);
			const float3 p2 = XYZ(light.v3);
			const float3 e1 = p1 - p0;
			const float3 e2 = p2 - p0;
			const float3 crossE = Cross(e1, e2);
			const float twiceArea = Length(crossE);
			const float3 centroid = (p0 + p1 + p2) * (1.0f / 3.0f);

			leaf.mean = centroid;
			leaf.variance = (Dot(e1, e1) + Dot(e2, e2) - Dot(e1, e2)) / 18.0f;
			leaf.radius = std::max({ Length(p0 - centroid), Length(p1 - centroid), Length(p2 - centroid) });
			leaf.flux = Pi * Luminance(XYZ(light.radiance)) * 0.5f * twiceArea;
			// Rough vMF fit to Lambert's cosine around the front face normal.
			leaf.averageDirection = twiceArea > 0.0f ? crossE * (0.5f / twiceArea) : float3(0.0f, 0.0f, 0.0f);
			primitive.bounds.Grow(p0);
			primitive.bounds.Grow(p1);
			primitive.bounds.Grow(p2);
		}
		else
		{
			return false;
		}

		primitive.centroid = leaf.mean;
		primitive.flux = leaf.flux;
		return leaf.flux > 0.0f;
	}

	// Exact SAH for small ranges: sorts along each axis and evaluates every split, which is cheaper
	// than binning at this size and never fails on coincident centroids. Leaves the range ordered
	// along the best axis and returns the split position in (begin, end).
	uint32_t SweepSplit(std::vector<Primitive> &prims, uint32_t begin, uint32_t end)
	{
		const uint32_t count = end - begin;
		Primitive local[kSweepMaxCount];
		std::copy(prims.begin() + begin, prims.begin() + end, local);

		uint8_t order[3][kSweepMaxCount];
		float rightCost[kSweepMaxCount];
		float bestCost = FLT_MAX;
		uint32_t bestImbalance = UINT32_MAX;
		int bestAxis = 0;
		uint32_t bestLeftCount = count / 2;

		for (int axis = 0; axis < 3; ++axis)
		{
			uint8_t *o = order[axis];
			for (uint32_t i = 0; i < count; ++i)
			{
				o[i] = static_cast<uint8_t>(i);
			}
			std::sort(o,
					  o + count,
					  [&](uint8_t a, uint8_t b)
					  { return Component(local[a].centroid, axis) < Component(local[b].centroid, axis); });

			Bounds accBounds;
			float accFlux = 0.0f;
			for (uint32_t i = count - 1; i > 0; --i)
			{
				accBounds.Grow(local[o[i]].bounds);
				accFlux += local[o[i]].flux;
				rightCost[i] = accFlux * accBounds.SurfaceArea();
			}

			accBounds = Bounds{};
			accFlux = 0.0f;
			for (uint32_t leftCount = 1; leftCount < count; ++leftCount)
			{
				accBounds.Grow(local[o[leftCount - 1]].bounds);
				accFlux += local[o[leftCount - 1]].flux;

				// Point lights with zero radius have zero-area bounds, so every split can cost zero.
				// Prefer the most even one when costs tie.
				const float cost = accFlux * accBounds.SurfaceArea() + rightCost[leftCount];
				const uint32_t imbalance = SplitImbalance(leftCount, count);
				if (cost < bestCost || (cost == bestCost && imbalance < bestImbalance))
				{
					bestCost = cost;
					bestImbalance = imbalance;
					bestAxis = axis;
					bestLeftCount = leftCount;
				}
			}
		}

		for (uint32_t i = 0; i < count; ++i)
		{
			prims[begin + i] = local[order[bestAxis][i]];
		}
		return begin + bestLeftCount;
	}

	struct BinnedSplitResult
	{
		uint32_t mid = 0;
		Bounds leftCentroids;
		Bounds rightCentroids;
	};

	// Flux weighted binned SAH. All three axes are binned in one pass over the range, and the bins
	// also track centroid bounds so the children's come for free. bins is scratch space for
	// 3 * binCount bins. Returns false if no axis separates the centroids.
	bool BinnedSplit(std::vector<Primitive> &prims,
					 const BuildTask &task,
					 std::span<Bin> bins,
					 uint32_t binCount,
					 BinnedSplitResult &result)
	{
		const uint32_t count = task.end - task.begin;
		float lo[3];
		float scale[3];
		for (int axis = 0; axis < 3; ++axis)
		{
			lo[axis] = Component(task.centroidBounds.lo, axis);
			const float extent = Component(task.centroidBounds.hi, axis) - lo[axis];
			// A flat axis puts everything in bin 0, which the sweep below rejects.
			scale[axis] = extent > 0.0f ? static_cast<float>(binCount) / extent : 0.0f;
		}

		std::fill(bins.begin(), bins.end(), Bin{});
		for (uint32_t i = task.begin; i < task.end; ++i)
		{
			const Primitive &p = prims[i];
			for (int axis = 0; axis < 3; ++axis)
			{
				Bin &bin = bins[axis * binCount + BinIndex(Component(p.centroid, axis), lo[axis], scale[axis], binCount)];
				bin.bounds.Grow(p.bounds);
				bin.centroidBounds.Grow(p.centroid);
				bin.flux += p.flux;
				bin.count++;
			}
		}

		float bestCost = FLT_MAX;
		uint32_t bestImbalance = UINT32_MAX;
		int bestAxis = -1;
		uint32_t bestBin = 0;
		float rightCost[kMaxBins];

		for (int axis = 0; axis < 3; ++axis)
		{
			if (scale[axis] == 0.0f)
			{
				continue;
			}
			const Bin *axisBins = &bins[axis * binCount];

			Bounds accBounds;
			float accFlux = 0.0f;
			for (uint32_t b = binCount - 1; b > 0; --b)
			{
				accBounds.Grow(axisBins[b].bounds);
				accFlux += axisBins[b].flux;
				rightCost[b] = accFlux * accBounds.SurfaceArea();
			}

			accBounds = Bounds{};
			accFlux = 0.0f;
			uint32_t leftCount = 0;
			for (uint32_t b = 0; b + 1 < binCount; ++b)
			{
				accBounds.Grow(axisBins[b].bounds);
				accFlux += axisBins[b].flux;
				leftCount += axisBins[b].count;
				if (leftCount == 0 || leftCount == count)
				{
					continue;
				}

				const float cost = accFlux * accBounds.SurfaceArea() + rightCost[b + 1];
				const uint32_t imbalance = SplitImbalance(leftCount, count);
				if (cost < bestCost || (cost == bestCost && imbalance < bestImbalance))
				{
					bestCost = cost;
					bestImbalance = imbalance;
					bestAxis = axis;
					bestBin = b;
				}
			}
		}

		if (bestAxis < 0)
		{
			return false;
		}

		const Bin *axisBins = &bins[bestAxis * binCount];
		result.leftCentroids = Bounds{};
		result.rightCentroids = Bounds{};
		for (uint32_t b = 0; b < binCount; ++b)
		{
			(b <= bestBin ? result.leftCentroids : result.rightCentroids).Grow(axisBins[b].centroidBounds);
		}

		const float axisLo = lo[bestAxis];
		const float axisScale = scale[bestAxis];
		const auto it = std::partition(prims.begin() + task.begin,
									   prims.begin() + task.end,
									   [&](const Primitive &p)
									   { return BinIndex(Component(p.centroid, bestAxis), axisLo, axisScale, binCount) <= bestBin; });
		result.mid = static_cast<uint32_t>(it - prims.begin());
		return result.mid != task.begin && result.mid != task.end;
	}

	// Fallback for coincident centroids: median split by count on the largest axis.
	uint32_t MedianSplit(std::vector<Primitive> &prims, uint32_t begin, uint32_t end, const Bounds &centroidBounds)
	{
		const float3 d = centroidBounds.hi - centroidBounds.lo;
		const int axis = (d.x >= d.y && d.x >= d.z) ? 0 : (d.y >= d.z ? 1 : 2);
		const uint32_t mid = begin + (end - begin) / 2;
		std::nth_element(prims.begin() + begin,
						 prims.begin() + mid,
						 prims.begin() + end,
						 [axis](const Primitive &a, const Primitive &b)
						 { return Component(a.centroid, axis) < Component(b.centroid, axis); });
		return mid;
	}
} // namespace

LightTree BuildLightTree(std::span<const GpuLight> lights, const LightTreeBuildSettings &settings)
{
	LightTree tree;
	std::vector<Primitive> prims;
	std::vector<LeafCluster> leaves;
	prims.reserve(lights.size());
	leaves.reserve(lights.size());

	for (uint32_t i = 0; i < static_cast<uint32_t>(lights.size()); ++i)
	{
		LeafCluster leaf = {};
		Primitive primitive = {};
		if (!MakeLeaf(lights[i], leaf, primitive))
		{
			continue;
		}

		primitive.light = static_cast<uint32_t>(tree.lights.size());
		tree.lights.push_back(lights[i]);
		tree.sourceIndex.push_back(i);
		prims.push_back(primitive);
		leaves.push_back(leaf);
	}

	const uint32_t lightCount = static_cast<uint32_t>(prims.size());
	if (lightCount == 0)
	{
		return tree;
	}

	tree.leafOfLight.resize(lightCount);
	tree.nodes.reserve(2 * lightCount - 1);
	tree.nodes.push_back(GpuLightTreeNode{});
	tree.nodes[0].parent = 0;

	// Top down topology. Children are allocated as adjacent pairs, so a child always has a larger
	// index than its parent, which lets the bottom-up pass below run in reverse index order.
	std::vector<BuildTask> stack;
	stack.push_back({ 0, 0, lightCount, CentroidBounds(prims, 0, lightCount) });
	const uint32_t binCount = std::clamp(settings.binCount, 2u, kMaxBins);
	std::vector<Bin> bins(3 * binCount);

	while (!stack.empty())
	{
		const BuildTask task = stack.back();
		stack.pop_back();

		const uint32_t count = task.end - task.begin;
		if (count == 1)
		{
			const uint32_t light = prims[task.begin].light;
			tree.nodes[task.node].childOrLight = light | LIGHT_TREE_LEAF_BIT;
			tree.leafOfLight[light] = task.node;
			continue;
		}

		// Children of small ranges are never binned, so their centroid bounds stay empty.
		uint32_t mid;
		Bounds leftCentroids;
		Bounds rightCentroids;
		BinnedSplitResult split;
		if (count == 2)
		{
			mid = task.begin + 1;
		}
		else if (count <= kSweepMaxCount)
		{
			mid = SweepSplit(prims, task.begin, task.end);
		}
		else if (BinnedSplit(prims, task, bins, binCount, split))
		{
			mid = split.mid;
			leftCentroids = split.leftCentroids;
			rightCentroids = split.rightCentroids;
		}
		else
		{
			mid = MedianSplit(prims, task.begin, task.end, task.centroidBounds);
			leftCentroids = CentroidBounds(prims, task.begin, mid);
			rightCentroids = CentroidBounds(prims, mid, task.end);
		}

		const uint32_t left = static_cast<uint32_t>(tree.nodes.size());
		tree.nodes.push_back(GpuLightTreeNode{});
		tree.nodes.push_back(GpuLightTreeNode{});
		tree.nodes[task.node].childOrLight = left;
		tree.nodes[left].parent = task.node;
		tree.nodes[left + 1].parent = task.node;

		stack.push_back({ left + 1, mid, task.end, rightCentroids });
		stack.push_back({ left, task.begin, mid, leftCentroids });
	}

	// Bottom up cluster parameters (Eqs. 2 to 5).
	std::vector<float3> averageDirection(tree.nodes.size());
	for (size_t index = tree.nodes.size(); index-- > 0;)
	{
		GpuLightTreeNode &node = tree.nodes[index];

		if (node.childOrLight & LIGHT_TREE_LEAF_BIT)
		{
			const LeafCluster &leaf = leaves[node.childOrLight & ~LIGHT_TREE_LEAF_BIT];
			node.mean = leaf.mean;
			node.variance = leaf.variance;
			node.radius = leaf.radius;
			node.flux = leaf.flux;
			averageDirection[index] = leaf.averageDirection;
			continue;
		}

		const uint32_t l = node.childOrLight;
		const uint32_t r = l + 1;
		const GpuLightTreeNode &a = tree.nodes[l];
		const GpuLightTreeNode &b = tree.nodes[r];
		const float flux = a.flux + b.flux;
		const float wl = a.flux / flux;
		const float wr = b.flux / flux;

		const float3 mean = a.mean * wl + b.mean * wr;
		const float3 dLR = a.mean - b.mean;

		node.mean = mean;
		node.variance = wl * a.variance + wr * b.variance + wl * wr * Dot(dLR, dLR);
		node.radius = std::max(Length(a.mean - mean) + a.radius, Length(b.mean - mean) + b.radius);
		node.flux = flux;
		averageDirection[index] = averageDirection[l] * wl + averageDirection[r] * wr;
	}

	// Convert average directions to the vMF axis and sharpness (Eq. 3).
	for (size_t index = 0; index < tree.nodes.size(); ++index)
	{
		const float3 v = averageDirection[index];
		float len = Length(v);
		tree.nodes[index].axis = len > 0.0f ? v * (1.0f / len) : float3(0.0f, 0.0f, 1.0f);
		len = std::min(len, 0.9999f);
		tree.nodes[index].sharpness = (3.0f * len - len * len * len) / (1.0f - len * len);
	}

	return tree;
}

std::vector<uint32_t> BuildInputToLightMap(const LightTree &tree, size_t inputLightCount)
{
	std::vector<uint32_t> map(inputLightCount, kInvalidLightIndex);
	for (uint32_t light = 0; light < static_cast<uint32_t>(tree.sourceIndex.size()); ++light)
	{
		map[tree.sourceIndex[light]] = light;
	}
	return map;
}
