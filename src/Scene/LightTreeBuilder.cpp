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

	// Full spherical integral of an SG, 2 pi (1 - exp(-2 kappa)) / kappa. Matches SGIntegral in SGLighting.slang.
	float SGIntegral(float kappa)
	{
		if (kappa < 0.05f)
		{
			return 2.0f * Pi * (2.0f + kappa * (-2.0f + kappa * (4.0f / 3.0f + kappa * (-2.0f / 3.0f))));
		}
		return 2.0f * Pi * (1.0f - expf(-2.0f * kappa)) / kappa;
	}

	// vMF sharpness from the length of a cluster's average direction (Eq. 3). The length is capped so
	// that the sharpness stays below about 1e4.
	float VmfSharpness(float averageDirectionLength)
	{
		const float len = std::min(averageDirectionLength, 0.9999f);
		return (3.0f * len - len * len * len) / (1.0f - len * len);
	}

	// Kept in full 16 byte vectors: binning grows the same bin for consecutive primitives, and
	// 12 byte float3 stores followed by wider loads stall on store forwarding.
	struct Bounds
	{
		SIMDVector lo = XMVectorReplicate(FLT_MAX);
		SIMDVector hi = XMVectorReplicate(-FLT_MAX);

		void Grow(SIMDVector p)
		{
			lo = XMVectorMin(lo, p);
			hi = XMVectorMax(hi, p);
		}

		void Grow(const float3 &p)
		{
			Grow(LoadFloat3(p));
		}

		void Grow(const Bounds &b)
		{
			lo = XMVectorMin(lo, b.lo);
			hi = XMVectorMax(hi, b.hi);
		}

		bool IsValid() const
		{
			return XMVectorGetX(lo) <= XMVectorGetX(hi);
		}

		float SurfaceArea() const
		{
			if (!IsValid())
			{
				return 0.0f;
			}
			const SIMDVector d = XMVectorSubtract(hi, lo);
			const SIMDVector products = XMVectorMultiply(d, XMVectorSwizzle<XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_X, XM_SWIZZLE_W>(d));
			return 2.0f * XMVectorGetX(XMVector3Dot(products, XMVectorSplatOne()));
		}
	};

	struct Primitive
	{
		Bounds bounds;
		// Flux times the average emission direction (nu bar) in xyz, flux in w. Summed over a range it
		// gives the range's flux, and its average direction after dividing by that.
		SIMDVector fluxDirection;
		float3 centroid;
		uint32_t light;
	};

	// Split cost of one side: flux times the surface area of its bounds and, with orientation, times
	// the spherical integral of the normalised vMF fitted to its normals, which is what the node will
	// store. That runs from about pi for triangles that all face one way (|nu bar| = 0.5 each) to
	// 4 pi when their normals cancel, the same range as the cone measure M_Omega of SAOH [Conty and
	// Kulla 2018], but it only needs the summed fluxDirection rather than a cone union per bin.
	float SideCost(const Bounds &bounds, SIMDVector fluxDirection, bool orientation)
	{
		const float flux = XMVectorGetW(fluxDirection);
		if (!(flux > 0.0f))
		{
			return 0.0f;
		}
		float cost = flux * bounds.SurfaceArea();
		if (orientation)
		{
			cost *= SGIntegral(VmfSharpness(XMVectorGetX(XMVector3Length(fluxDirection)) / flux));
		}
		return cost;
	}

	// Cluster parameters (Eqs. 2 to 5) in full precision, of a single light or of a node while
	// building. Packed into GpuLightTreeNode at the end, so the bottom-up pass reads unrounded values.
	struct Cluster
	{
		float3 mean;
		float variance;
		float radius;
		float flux;
		float3 averageDirection; // nu bar in Eq. 2, 0 for an isotropic emitter
	};

	// Merges count clusters, get(i) returning the i-th: flux weighted mean, variance and average
	// direction, and a bounding sphere around the new mean that holds theirs. For two clusters the
	// variance is the paper's w_l var_l + w_r var_r + w_l w_r |mu_l - mu_r|^2.
	template <typename Get>
	Cluster Combine(uint32_t count, Get get)
	{
		Cluster c = {};
		for (uint32_t i = 0; i < count; ++i)
		{
			c.flux += get(i).flux;
		}
		c.mean = { 0.0f, 0.0f, 0.0f };
		c.averageDirection = { 0.0f, 0.0f, 0.0f };
		for (uint32_t i = 0; i < count; ++i)
		{
			const Cluster &child = get(i);
			const float w = child.flux / c.flux;
			c.mean = c.mean + child.mean * w;
			c.averageDirection = c.averageDirection + child.averageDirection * w;
		}
		for (uint32_t i = 0; i < count; ++i)
		{
			const Cluster &child = get(i);
			const float3 d = child.mean - c.mean;
			c.variance += child.flux / c.flux * (child.variance + Dot(d, d));
			c.radius = std::max(c.radius, Length(d) + child.radius);
		}
		return c;
	}

	// Node of the binary top-down build, which is then collapsed into the 4-wide tree.
	struct BinaryNode
	{
		uint32_t left; // the right child is left + 1, 0 for a leaf
		uint32_t begin; // range of the build primitives under it
		uint32_t end;
	};

	constexpr uint32_t kEmptyNode = UINT32_MAX;

	// Octahedral encoding of a unit vector as snorm16x2, x in the low half. LightTreeAxis in
	// LightTree.slang decodes it.
	uint32_t PackOctahedralSnorm16x2(const float3 &v)
	{
		const float l1 = fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
		float x = v.x / l1;
		float y = v.y / l1;
		if (v.z < 0.0f)
		{
			const float ox = x;
			x = (1.0f - fabsf(y)) * (ox >= 0.0f ? 1.0f : -1.0f);
			y = (1.0f - fabsf(ox)) * (y >= 0.0f ? 1.0f : -1.0f);
		}
		auto snorm = [](float f)
		{ return static_cast<uint32_t>(static_cast<int32_t>(lroundf(std::clamp(f, -1.0f, 1.0f) * 32767.0f)) & 0xFFFF); };
		return snorm(x) | (snorm(y) << 16);
	}

	// float16 tops out at 65504, so a larger spatial deviation (a cluster wider than that in world
	// units) is clamped. It only shapes the importance, which stays positive either way.
	constexpr float kMaxHalf = 65504.0f;

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
		SIMDVector fluxDirection = XMVectorZero(); // sum of Primitive::fluxDirection
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

	// Fills the leaf cluster and the build primitive of one light, with its flux multiplied by
	// fluxScale. Returns false for anything but area lights and for lights that emit nothing.
	bool MakeLeaf(GpuLight light, float fluxScale, Cluster &leaf, Primitive &primitive)
	{
		if (light.GetLightType() != kLight_Area)
		{
			return false;
		}

		// One-sided Lambertian triangle, Phi = pi L A. A radiance map is not accounted for here, the
		// caller passes its average over the triangle as fluxScale.
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
		leaf.flux = Pi * Luminance(XYZ(light.radiance)) * 0.5f * twiceArea * fluxScale;
		// Rough vMF fit to Lambert's cosine around the front face normal.
		leaf.averageDirection = twiceArea > 0.0f ? crossE * (0.5f / twiceArea) : float3(0.0f, 0.0f, 0.0f);

		primitive.bounds.Grow(p0);
		primitive.bounds.Grow(p1);
		primitive.bounds.Grow(p2);
		primitive.centroid = centroid;
		primitive.fluxDirection = XMVectorSetW(XMVectorScale(LoadFloat3(leaf.averageDirection), leaf.flux), leaf.flux);
		return leaf.flux > 0.0f;
	}

	// Exact SAH for small ranges: sorts along each axis and evaluates every split, which is cheaper
	// than binning at this size and never fails on coincident centroids. Leaves the range ordered
	// along the best axis and returns the split position in (begin, end).
	uint32_t SweepSplit(std::vector<Primitive> &prims, uint32_t begin, uint32_t end, bool orientation)
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
			SIMDVector accFluxDirection = XMVectorZero();
			for (uint32_t i = count - 1; i > 0; --i)
			{
				accBounds.Grow(local[o[i]].bounds);
				accFluxDirection = XMVectorAdd(accFluxDirection, local[o[i]].fluxDirection);
				rightCost[i] = SideCost(accBounds, accFluxDirection, orientation);
			}

			accBounds = Bounds{};
			accFluxDirection = XMVectorZero();
			for (uint32_t leftCount = 1; leftCount < count; ++leftCount)
			{
				accBounds.Grow(local[o[leftCount - 1]].bounds);
				accFluxDirection = XMVectorAdd(accFluxDirection, local[o[leftCount - 1]].fluxDirection);

				// Prefer the most even split when costs tie.
				const float cost = SideCost(accBounds, accFluxDirection, orientation) + rightCost[leftCount];
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

	// Binned split by SideCost. All three axes are binned in one pass over the range, and the bins
	// also track centroid bounds so the children's come for free. bins is scratch space for
	// 3 * binCount bins. Returns false if no axis separates the centroids.
	bool BinnedSplit(std::vector<Primitive> &prims,
					 const BuildTask &task,
					 std::span<Bin> bins,
					 uint32_t binCount,
					 bool orientation,
					 BinnedSplitResult &result)
	{
		const uint32_t count = task.end - task.begin;
		float3 centroidLo;
		float3 centroidHi;
		StoreFloat3(centroidLo, task.centroidBounds.lo);
		StoreFloat3(centroidHi, task.centroidBounds.hi);
		float lo[3];
		float scale[3];
		for (int axis = 0; axis < 3; ++axis)
		{
			lo[axis] = Component(centroidLo, axis);
			const float extent = Component(centroidHi, axis) - lo[axis];
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
				bin.fluxDirection = XMVectorAdd(bin.fluxDirection, p.fluxDirection);
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
			SIMDVector accFluxDirection = XMVectorZero();
			for (uint32_t b = binCount - 1; b > 0; --b)
			{
				accBounds.Grow(axisBins[b].bounds);
				accFluxDirection = XMVectorAdd(accFluxDirection, axisBins[b].fluxDirection);
				rightCost[b] = SideCost(accBounds, accFluxDirection, orientation);
			}

			accBounds = Bounds{};
			accFluxDirection = XMVectorZero();
			uint32_t leftCount = 0;
			for (uint32_t b = 0; b + 1 < binCount; ++b)
			{
				accBounds.Grow(axisBins[b].bounds);
				accFluxDirection = XMVectorAdd(accFluxDirection, axisBins[b].fluxDirection);
				leftCount += axisBins[b].count;
				if (leftCount == 0 || leftCount == count)
				{
					continue;
				}

				const float cost = SideCost(accBounds, accFluxDirection, orientation) + rightCost[b + 1];
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
		float3 d;
		StoreFloat3(d, XMVectorSubtract(centroidBounds.hi, centroidBounds.lo));
		const int axis = (d.x >= d.y && d.x >= d.z) ? 0 : (d.y >= d.z ? 1 : 2);
		const uint32_t mid = begin + (end - begin) / 2;
		std::nth_element(prims.begin() + begin,
						 prims.begin() + mid,
						 prims.begin() + end,
						 [axis](const Primitive &a, const Primitive &b)
						 { return Component(a.centroid, axis) < Component(b.centroid, axis); });
		return mid;
	}

	// Vose's alias method over the leaves' flux, in light order.
	std::vector<GpuLightAliasEntry> BuildPowerTable(const std::vector<Cluster> &leaves)
	{
		const size_t count = leaves.size();
		double total = 0.0;
		for (const Cluster &leaf: leaves)
		{
			total += leaf.flux;
		}

		// Each entry's flux in units of the mean, split into the ones below it and the ones that cover them.
		std::vector<double> scaled(count);
		std::vector<uint32_t> small;
		std::vector<uint32_t> large;
		for (uint32_t i = 0; i < static_cast<uint32_t>(count); ++i)
		{
			scaled[i] = leaves[i].flux * static_cast<double>(count) / total;
			(scaled[i] < 1.0 ? small : large).push_back(i);
		}

		std::vector<GpuLightAliasEntry> table(count);
		while (!small.empty() && !large.empty())
		{
			const uint32_t s = small.back();
			small.pop_back();
			const uint32_t l = large.back();
			table[s].threshold = static_cast<float>(scaled[s]);
			table[s].alias = l;
			scaled[l] -= 1.0 - scaled[s];
			if (scaled[l] < 1.0)
			{
				large.pop_back();
				small.push_back(l);
			}
		}
		// What is left is 1 up to rounding, and keeps itself.
		for (const std::vector<uint32_t> *rest: { &small, &large })
		{
			for (uint32_t i: *rest)
			{
				table[i].threshold = 1.0f;
				table[i].alias = i;
			}
		}

		for (uint32_t i = 0; i < static_cast<uint32_t>(count); ++i)
		{
			table[i].pmf = static_cast<float>(leaves[i].flux / total);
		}
		for (GpuLightAliasEntry &entry: table)
		{
			entry.aliasPmf = table[entry.alias].pmf;
		}
		return table;
	}
} // namespace

float LightTreeFlux(const GpuLight &light, float fluxScale)
{
	Cluster leaf = {};
	Primitive primitive = {};
	return MakeLeaf(light, fluxScale, leaf, primitive) ? leaf.flux : 0.0f;
}

LightTree BuildLightTree(std::span<const GpuLight> lights,
						 std::span<const float> fluxScales,
						 const LightTreeBuildSettings &settings)
{
	LightTree tree;
	std::vector<Primitive> prims;
	std::vector<Cluster> leaves;
	prims.reserve(lights.size());
	leaves.reserve(lights.size());
	tree.lights.reserve(lights.size());
	tree.sourceIndex.reserve(lights.size());

	for (uint32_t i = 0; i < static_cast<uint32_t>(lights.size()); ++i)
	{
		Cluster leaf = {};
		Primitive primitive = {};
		const float fluxScale = fluxScales.empty() ? 1.0f : fluxScales[i];
		if (!MakeLeaf(lights[i], fluxScale, leaf, primitive))
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

	if (lightCount > LIGHT_TREE_LEAF_FIRST_MASK + 1)
	{
		// Leaves address their range with 29 bits.
		return LightTree{};
	}

	// Still in light order here; the build below reorders prims, not leaves.
	tree.powerTable = BuildPowerTable(leaves);

	// Top down binary topology. Children are allocated as adjacent pairs, so a child always has a
	// larger index than its parent. Ranges of maxLeafLights or fewer become leaves.
	const uint32_t maxLeafLights = std::clamp(settings.maxLeafLights, 1u, LIGHT_TREE_MAX_LEAF_LIGHTS);
	std::vector<BinaryNode> binary;
	binary.reserve(2 * lightCount - 1);
	binary.push_back({ 0, 0, lightCount });
	std::vector<BuildTask> stack;
	stack.push_back({ 0, 0, lightCount, CentroidBounds(prims, 0, lightCount) });
	const uint32_t binCount = std::clamp(settings.binCount, 2u, kMaxBins);
	std::vector<Bin> bins(3 * binCount);

	while (!stack.empty())
	{
		const BuildTask task = stack.back();
		stack.pop_back();

		const uint32_t count = task.end - task.begin;
		if (count <= maxLeafLights)
		{
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
			mid = SweepSplit(prims, task.begin, task.end, settings.orientationSplits);
		}
		else if (BinnedSplit(prims, task, bins, binCount, settings.orientationSplits, split))
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

		const uint32_t left = static_cast<uint32_t>(binary.size());
		binary.push_back({ 0, task.begin, mid });
		binary.push_back({ 0, mid, task.end });
		binary[task.node].left = left;

		stack.push_back({ left + 1, mid, task.end, rightCentroids });
		stack.push_back({ left, task.begin, mid, leftCentroids });
	}

	// Collapse into groups of LIGHT_TREE_WIDTH: a group starts as the two children of a binary node,
	// and the member with the most lights is replaced by its own two children until the group is full
	// or holds only leaves. Opening the largest keeps the tree shallow. Groups are appended while the
	// nodes are visited in index order, so a child always has a larger index than its parent, which
	// lets the bottom-up pass below run in reverse index order.
	std::vector<uint32_t> nodeBinary; // binary node of each node, kEmptyNode for padding
	std::vector<uint32_t> depth; // levels of groups from the root down to the node
	nodeBinary.reserve(2 * lightCount);
	depth.reserve(2 * lightCount);
	auto addGroup = [&](uint32_t binaryNode, uint32_t parent, uint32_t parentDepth)
	{
		uint32_t members[LIGHT_TREE_WIDTH];
		uint32_t memberCount = 0;
		if (binary[binaryNode].left == 0)
		{
			members[memberCount++] = binaryNode; // a single leaf under the root
		}
		else
		{
			members[memberCount++] = binary[binaryNode].left;
			members[memberCount++] = binary[binaryNode].left + 1;
		}
		while (memberCount < LIGHT_TREE_WIDTH)
		{
			uint32_t open = UINT32_MAX;
			uint32_t openCount = 0;
			for (uint32_t i = 0; i < memberCount; ++i)
			{
				const BinaryNode &b = binary[members[i]];
				if (b.left != 0 && b.end - b.begin > openCount)
				{
					open = i;
					openCount = b.end - b.begin;
				}
			}
			if (open == UINT32_MAX)
			{
				break;
			}
			const uint32_t left = binary[members[open]].left;
			members[open] = left;
			members[memberCount++] = left + 1;
		}

		const uint32_t first = static_cast<uint32_t>(nodeBinary.size());
		for (uint32_t i = 0; i < LIGHT_TREE_WIDTH; ++i)
		{
			nodeBinary.push_back(i < memberCount ? members[i] : kEmptyNode);
			depth.push_back(parentDepth + 1);
		}
		tree.parents.push_back(parent);
		return first;
	};

	std::vector<uint32_t> childOrLight;
	childOrLight.reserve(2 * lightCount);
	addGroup(0, 0, 0);
	for (uint32_t index = 0; index < static_cast<uint32_t>(nodeBinary.size()); ++index)
	{
		const uint32_t b = nodeBinary[index];
		if (b == kEmptyNode)
		{
			childOrLight.push_back(0);
		}
		else if (binary[b].left == 0)
		{
			const uint32_t count = binary[b].end - binary[b].begin;
			childOrLight.push_back(LIGHT_TREE_LEAF_BIT | ((count - 1) << LIGHT_TREE_LEAF_COUNT_SHIFT) | binary[b].begin);
			tree.maxDepth = std::max(tree.maxDepth, depth[index]);
			tree.averageDepth += static_cast<float>(depth[index] * count);
		}
		else
		{
			childOrLight.push_back(addGroup(b, index, depth[index]));
		}
	}
	tree.averageDepth /= static_cast<float>(lightCount);

	// Bottom up cluster parameters (Eqs. 2 to 5), and the leaf entries: each light is picked from its
	// leaf in proportion to its flux.
	tree.leafLights.resize(lightCount);
	tree.leafOfLight.resize(lightCount);
	std::vector<Cluster> clusters(nodeBinary.size());
	for (uint32_t index = static_cast<uint32_t>(nodeBinary.size()); index-- > 0;)
	{
		const uint32_t b = nodeBinary[index];
		if (b == kEmptyNode)
		{
			continue;
		}
		if (binary[b].left == 0)
		{
			const uint32_t begin = binary[b].begin;
			const uint32_t count = binary[b].end - begin;
			clusters[index] = Combine(count, [&](uint32_t i) -> const Cluster & { return leaves[prims[begin + i].light]; });
			for (uint32_t i = 0; i < count; ++i)
			{
				const uint32_t light = prims[begin + i].light;
				const float pmf = leaves[light].flux / clusters[index].flux;
				tree.leafLights[begin + i] = { light, pmf };
				tree.leafOfLight[light] = { index, pmf };
			}
			continue;
		}

		const uint32_t first = childOrLight[index];
		uint32_t members[LIGHT_TREE_WIDTH];
		uint32_t memberCount = 0;
		for (uint32_t i = 0; i < LIGHT_TREE_WIDTH; ++i)
		{
			if (nodeBinary[first + i] != kEmptyNode)
			{
				members[memberCount++] = first + i;
			}
		}
		clusters[index] = Combine(memberCount, [&](uint32_t i) -> const Cluster & { return clusters[members[i]]; });
	}

	// Convert average directions to the vMF axis and sharpness (Eq. 3), and fold the flux and the
	// sharpness dependent normalisation into one amplitude so the shader does not recompute it per
	// visit. Then pack each node; sharpness stays below about 1e4 (len <= 0.9999), within float16.
	tree.nodes.resize(nodeBinary.size());
	for (size_t index = 0; index < nodeBinary.size(); ++index)
	{
		GpuLightTreeNode &packed = tree.nodes[index];
		if (nodeBinary[index] == kEmptyNode)
		{
			packed = {};
			packed.radius = -FLT_MAX; // rejected by LightTreeNodeImportance as below the horizon
			continue;
		}

		const Cluster &node = clusters[index];
		const float3 v = node.averageDirection;
		const float len = Length(v);
		const float3 axis = len > 0.0f ? v * (1.0f / len) : float3(0.0f, 0.0f, 1.0f);
		const float sharpness = VmfSharpness(len);

		packed.mean = node.mean;
		packed.amplitude = node.flux / (2.0f * Pi * SGIntegral(sharpness));
		packed.childOrLight = childOrLight[index];
		packed.axis = PackOctahedralSnorm16x2(axis);
		packed.sharpnessStdDev = PackF16Pair(sharpness, std::min(sqrtf(std::max(node.variance, 0.0f)), kMaxHalf));
		packed.radius = node.radius;
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
