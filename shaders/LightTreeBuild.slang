// The light tree, built on the GPU every frame after the single-level light tree of "Real-Time Stochastic Lightcuts"
// (Lin and Yuksel 2020), Section 5.1: the lights are sorted along a Morton curve and a 4-wide tree is laid over the
// sorted order, so its topology is implicit and only the clusters need computing, bottom up. The clusters and the node
// format are those of LightTree.slang, which walks the tree. LightTreeBuilder dispatches, in this order:
//
//   LightKeys                        per light: its cluster and Morton key
//   RadixCount, ScanRadixCounts,     a stable 4-bit LSD radix sort of the (key, light) pairs, LIGHT_TREE_RADIX_PASSES
//   RadixScatter                     times
//   BuildLeaves                      per leaf: leafSize consecutive sorted lights
//   BuildLevel                       per node of one interior level, for the levels wider than LIGHT_TREE_BUILD_TOP_NODES
//   BuildTopLevels                   one group for the remaining levels up to the root group
//   PowerReduce, ScanPowerBlocks,    the flux CDF of power selection, in light order
//   PowerApply, PowerGuide
//
// The tree: the leaf level has ceil(lightCount / leafSize) nodes, and every level above has a quarter as many as the
// one below, rounded up, until one fits a group, the root group. Each level is padded with empty nodes to whole
// groups, so node j of a level has the children 4j to 4j + 3 of the level below. Levels are stored from the root group
// down, so the root group is at node 0 where the traversal starts.

#include "ShaderInterop.h"
#include "LightTreeInterop.h"

ConstantBuffer<LightTreeBuildConstants> g_PushConstants : register(b0);
ConstantBuffer<LightTreeBuildResources> g_Resources : register(b1);

static const float LIGHT_TREE_BUILD_PI = 3.14159265358979;
static const float LIGHT_TREE_BUILD_FLT_MAX = 3.402823466e+38;
// float16 tops out at 65504, so a larger spatial deviation is clamped. It only shapes the importance.
static const float LIGHT_TREE_BUILD_MAX_HALF = 65504.0;
static const uint LIGHT_TREE_INVALID_KEY = 0xFFFFFFFFu;
static const uint LIGHT_TREE_RADIX_DIGITS = 1u << LIGHT_TREE_RADIX_BITS;
static const uint LIGHT_TREE_KEYS_PER_THREAD = LIGHT_TREE_BUILD_BLOCK_SIZE / LIGHT_TREE_BUILD_GROUP_SIZE;
// 2^31 less a margin for the rounding of the flux total, so that the quantised fluxes never sum past 2^31.
static const float LIGHT_TREE_POWER_SCALE = 2147450880.0;
static const float LIGHT_TREE_POWER_INV_SCALE = 1.0 / 2147483648.0;

// Index of the leaf level, 0 when the leaves fit the root group.
uint LightTreeLeafLevel(uint leafCount)
{
    return leafCount <= LIGHT_TREE_WIDTH ? 0 : firstbithigh(leafCount - 1) / 2;
}

struct LightTreeLayout
{
    uint leafCount;
    uint leafLevel;

    // Nodes of level d that hold something: ceil(leafCount / 4^(leafLevel - d)).
    uint Width(uint d)
    {
        uint shift = 2 * (leafLevel - d);
        return (leafCount + (1u << shift) - 1) >> shift;
    }

    // Nodes of level d with the padding to whole groups.
    uint Nodes(uint d)
    {
        return (Width(d) + LIGHT_TREE_WIDTH - 1) & ~(LIGHT_TREE_WIDTH - 1);
    }

    // First node of level d.
    uint Offset(uint d)
    {
        uint offset = 0;
        for (uint e = 0; e < d; ++e)
            offset += Nodes(e);
        return offset;
    }
};

LightTreeLayout MakeLightTreeLayout()
{
    LightTreeLayout layout;
    layout.leafCount = (g_PushConstants.lightCount + g_PushConstants.leafSize - 1) / g_PushConstants.leafSize;
    layout.leafLevel = LightTreeLeafLevel(layout.leafCount);
    return layout;
}

float LightTreeLuminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Full spherical integral of an SG, 2 pi (1 - exp(-2 kappa)) / kappa.
float SGIntegral(float kappa)
{
    if (kappa < 0.05)
        return 2.0 * LIGHT_TREE_BUILD_PI * (2.0 + kappa * (-2.0 + kappa * (4.0 / 3.0 + kappa * (-2.0 / 3.0))));
    return 2.0 * LIGHT_TREE_BUILD_PI * (1.0 - exp(-2.0 * kappa)) / kappa;
}

// vMF sharpness from the length of a cluster's average direction (Eq. 3). The length is capped so that the sharpness
// stays below about 1e4, within float16.
float VmfSharpness(float averageDirectionLength)
{
    float len = min(averageDirectionLength, 0.9999);
    return (3.0 * len - len * len * len) / (1.0 - len * len);
}

// Octahedral encoding of a unit vector as snorm16x2, x in the low half. LightTreeAxis decodes it.
uint PackOctahedralSnorm16x2(float3 v)
{
    float2 e = v.xy / (abs(v.x) + abs(v.y) + abs(v.z));
    if (v.z < 0.0)
        e = (1.0 - abs(e.yx)) * float2(e.x >= 0.0 ? 1.0 : -1.0, e.y >= 0.0 ? 1.0 : -1.0);
    int2 s = int2(round(clamp(e, -1.0, 1.0) * 32767.0));
    return (uint(s.x) & 0xFFFF) | (uint(s.y) << 16);
}

// A one-sided Lambertian triangle, Phi = pi L A, or an isotropic point light, Phi = 4 pi I, times fluxScale.
// Anything else, and a light that emits nothing, gets zero flux, which leaves it without a leaf.
GpuLightTreeCluster MakeLightCluster(GpuLight light, float fluxScale)
{
    GpuLightTreeCluster c = (GpuLightTreeCluster)0;
    GpuLightType type = light.GetLightType();
    if (type == kLight_Point)
    {
        float flux = 4.0 * LIGHT_TREE_BUILD_PI * LightTreeLuminance(light.radiance.xyz) * fluxScale;
        if (!(flux > 0.0) || isinf(flux))
            return c;
        // No average direction, so the vMF is flat (sharpness 0), and no extent.
        c.mean = light.v1.xyz;
        c.flux = flux;
        return c;
    }
    if (type != kLight_Area)
        return c;

    float3 p0 = light.v1.xyz;
    float3 p1 = light.v2.xyz;
    float3 p2 = light.v3.xyz;
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 crossE = cross(e1, e2);
    float twiceArea = length(crossE);
    float3 centroid = (p0 + p1 + p2) * (1.0 / 3.0);

    float flux = LIGHT_TREE_BUILD_PI * LightTreeLuminance(light.radiance.xyz) * 0.5 * twiceArea * fluxScale;
    if (!(flux > 0.0) || isinf(flux))
        return c;
    c.mean = centroid;
    c.flux = flux;
    // Rough vMF fit to Lambert's cosine around the front face normal.
    c.averageDirection = crossE * (0.5 / twiceArea);
    c.variance = (dot(e1, e1) + dot(e2, e2) - dot(e1, e2)) / 18.0;
    c.radius = max(max(length(p0 - centroid), length(p1 - centroid)), length(p2 - centroid));
    return c;
}

// Merges the first count clusters, leaving out those without flux: flux weighted mean, variance and average
// direction, and a bounding sphere around the new mean that holds theirs. For two clusters the variance is the
// paper's w_l var_l + w_r var_r + w_l w_r |mu_l - mu_r|^2. Zero flux when none has any.
GpuLightTreeCluster CombineClusters<let N : int>(GpuLightTreeCluster children[N], uint count)
{
    GpuLightTreeCluster c = (GpuLightTreeCluster)0;
    [unroll]
    for (int i = 0; i < N; ++i)
    {
        if (i < count)
            c.flux += children[i].flux;
    }
    if (!(c.flux > 0.0))
        return (GpuLightTreeCluster)0;

    [unroll]
    for (int i = 0; i < N; ++i)
    {
        if (i < count)
        {
            float w = children[i].flux / c.flux;
            c.mean += children[i].mean * w;
            c.averageDirection += children[i].averageDirection * w;
        }
    }
    [unroll]
    for (int i = 0; i < N; ++i)
    {
        if (i < count && children[i].flux > 0.0)
        {
            float3 d = children[i].mean - c.mean;
            c.variance += children[i].flux / c.flux * (children[i].variance + dot(d, d));
            c.radius = max(c.radius, length(d) + children[i].radius);
        }
    }
    return c;
}

// A node that LightTreeNodeImportance rejects as lying below the horizon, for empty slots.
GpuLightTreeNode EmptyLightTreeNode()
{
    GpuLightTreeNode node;
    node.meanAmplitude = float4(0.0, 0.0, 0.0, 0.0);
    node.packed = uint4(0, 0, 0, asuint(-LIGHT_TREE_BUILD_FLT_MAX));
    return node;
}

// The vMF axis and sharpness from the average direction (Eq. 3), with the flux and the sharpness dependent
// normalisation folded into one amplitude so that the traversal does not recompute it per visit. An empty node for a
// cluster without flux.
GpuLightTreeNode PackLightTreeNode(GpuLightTreeCluster c, uint childOrLight)
{
    if (!(c.flux > 0.0))
        return EmptyLightTreeNode();

    float len = length(c.averageDirection);
    float3 axis = len > 0.0 ? c.averageDirection / len : float3(0.0, 0.0, 1.0);
    float sharpness = VmfSharpness(len);
    float stdDev = min(sqrt(max(c.variance, 0.0)), LIGHT_TREE_BUILD_MAX_HALF);

    GpuLightTreeNode node;
    node.meanAmplitude = float4(c.mean, c.flux / (2.0 * LIGHT_TREE_BUILD_PI * SGIntegral(sharpness)));
    node.packed = uint4(childOrLight, PackOctahedralSnorm16x2(axis), f32tof16(sharpness) | (f32tof16(stdDev) << 16),
                        asuint(c.radius));
    return node;
}

groupshared uint gs_WaveSums[LIGHT_TREE_BUILD_GROUP_SIZE];
groupshared uint gs_GroupTotal;

// Exclusive prefix sum of value over the group's threads in index order, and in total the group's sum. Assumes the
// usual layout of waves in a 1D group: lane i of wave w is thread w * WaveGetLaneCount() + i.
uint GroupExclusiveSum(uint value, uint thread, out uint total)
{
    uint lanes = WaveGetLaneCount();
    uint wave = thread / lanes;
    uint prefix = WavePrefixSum(value);
    if (WaveGetLaneIndex() == lanes - 1)
        gs_WaveSums[wave] = prefix + value;
    GroupMemoryBarrierWithGroupSync();
    if (thread == 0)
    {
        uint running = 0;
        for (uint w = 0; w < LIGHT_TREE_BUILD_GROUP_SIZE / lanes; ++w)
        {
            uint sum = gs_WaveSums[w];
            gs_WaveSums[w] = running;
            running += sum;
        }
        gs_GroupTotal = running;
    }
    GroupMemoryBarrierWithGroupSync();
    uint result = prefix + gs_WaveSums[wave];
    total = gs_GroupTotal;
    // Before another call overwrites them.
    GroupMemoryBarrierWithGroupSync();
    return result;
}

// Exclusive prefix sum of count entries in place, by one group, a block of LIGHT_TREE_BUILD_BLOCK_SIZE at a time.
void ScanInPlace(RWStructuredBuffer<uint> values, uint count, uint thread)
{
    uint carry = 0;
    for (uint base = 0; base < count; base += LIGHT_TREE_BUILD_BLOCK_SIZE)
    {
        uint first = base + thread * LIGHT_TREE_KEYS_PER_THREAD;
        uint local[LIGHT_TREE_KEYS_PER_THREAD];
        uint sum = 0;
        [unroll]
        for (uint i = 0; i < LIGHT_TREE_KEYS_PER_THREAD; ++i)
        {
            uint value = first + i < count ? values[first + i] : 0;
            local[i] = sum;
            sum += value;
        }
        uint total;
        uint prefix = GroupExclusiveSum(sum, thread, total) + carry;
        [unroll]
        for (uint i = 0; i < LIGHT_TREE_KEYS_PER_THREAD; ++i)
        {
            if (first + i < count)
                values[first + i] = prefix + local[i];
        }
        carry += total;
    }
}

uint LightTreeBlockCount()
{
    return (g_PushConstants.lightCount + LIGHT_TREE_BUILD_BLOCK_SIZE - 1) / LIGHT_TREE_BUILD_BLOCK_SIZE;
}

// Spreads the low 10 bits of v to every third bit.
uint ExpandBits10(uint v)
{
    v &= 0x3FF;
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint MortonKey(float3 p)
{
    uint3 cell = uint3(clamp((p - g_PushConstants.boundsMin) * g_PushConstants.boundsScale, 0.0, 1023.0));
    return (ExpandBits10(cell.x) << 2) | (ExpandBits10(cell.y) << 1) | ExpandBits10(cell.z);
}

// Each light's cluster, with its flux at the emission its instance has now so that a light that is off gets no
// samples, and its key; a light without flux gets the invalid key, which sorts after every other.
[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void LightKeys(uint3 dispatchThread : SV_DispatchThreadID)
{
    uint light = dispatchThread.x;
    if (light >= g_PushConstants.lightCount)
        return;

    StructuredBuffer<GpuLight> lights = ResourceDescriptorHeap[g_PushConstants.lightsHandle];
    StructuredBuffer<float> emissionScales = ResourceDescriptorHeap[g_PushConstants.emissionScalesHandle];
    StructuredBuffer<float> fluxScales = ResourceDescriptorHeap[g_Resources.fluxScales];
    StructuredBuffer<uint> lightEmitters = ResourceDescriptorHeap[g_Resources.lightEmitters];
    RWStructuredBuffer<GpuLightTreeCluster> lightClusters = ResourceDescriptorHeap[g_Resources.lightClusters];
    RWStructuredBuffer<uint> keys = ResourceDescriptorHeap[g_Resources.keysA];
    RWStructuredBuffer<uint> values = ResourceDescriptorHeap[g_Resources.valuesA];

    float fluxScale = fluxScales[light] * max(emissionScales[lightEmitters[light]], 0.0);
    GpuLightTreeCluster cluster = MakeLightCluster(lights[light], fluxScale);
    lightClusters[light] = cluster;
    keys[light] = cluster.flux > 0.0 ? MortonKey(cluster.mean) : LIGHT_TREE_INVALID_KEY;
    values[light] = light;
}

// Pass p sorts by bits 4p to 4p + 3, from the A pair to the B pair on even passes and back on odd ones. A block of
// LIGHT_TREE_BUILD_BLOCK_SIZE keys is one group; RadixCount counts each digit per block, ScanRadixCounts turns the
// counts (digit major) into each block's first slot per digit, and RadixScatter ranks its keys stably and moves them.
groupshared uint gs_DigitCounts[LIGHT_TREE_RADIX_DIGITS];

uint RadixDigit(uint key)
{
    return (key >> (g_PushConstants.pass * LIGHT_TREE_RADIX_BITS)) & (LIGHT_TREE_RADIX_DIGITS - 1);
}

[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void RadixCount(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    bool odd = (g_PushConstants.pass & 1) != 0;
    RWStructuredBuffer<uint> keys = ResourceDescriptorHeap[odd ? g_Resources.keysB : g_Resources.keysA];
    RWStructuredBuffer<uint> counts = ResourceDescriptorHeap[g_Resources.radixCounts];

    if (thread < LIGHT_TREE_RADIX_DIGITS)
        gs_DigitCounts[thread] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint first = group.x * LIGHT_TREE_BUILD_BLOCK_SIZE;
    for (uint i = 0; i < LIGHT_TREE_KEYS_PER_THREAD; ++i)
    {
        uint index = first + i * LIGHT_TREE_BUILD_GROUP_SIZE + thread;
        if (index < g_PushConstants.lightCount)
            InterlockedAdd(gs_DigitCounts[RadixDigit(keys[index])], 1);
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread < LIGHT_TREE_RADIX_DIGITS)
        counts[thread * LightTreeBlockCount() + group.x] = gs_DigitCounts[thread];
}

[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void ScanRadixCounts(uint thread : SV_GroupIndex)
{
    RWStructuredBuffer<uint> counts = ResourceDescriptorHeap[g_Resources.radixCounts];
    ScanInPlace(counts, LIGHT_TREE_RADIX_DIGITS * LightTreeBlockCount(), thread);
}

// Per wave and digit: the wave's count, then the slot of the wave's first key with that digit.
groupshared uint gs_WaveDigitSlots[LIGHT_TREE_BUILD_GROUP_SIZE / 4][LIGHT_TREE_RADIX_DIGITS];

[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void RadixScatter(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    bool odd = (g_PushConstants.pass & 1) != 0;
    RWStructuredBuffer<uint> keysIn = ResourceDescriptorHeap[odd ? g_Resources.keysB : g_Resources.keysA];
    RWStructuredBuffer<uint> valuesIn = ResourceDescriptorHeap[odd ? g_Resources.valuesB : g_Resources.valuesA];
    RWStructuredBuffer<uint> keysOut = ResourceDescriptorHeap[odd ? g_Resources.keysA : g_Resources.keysB];
    RWStructuredBuffer<uint> valuesOut = ResourceDescriptorHeap[odd ? g_Resources.valuesA : g_Resources.valuesB];
    RWStructuredBuffer<uint> counts = ResourceDescriptorHeap[g_Resources.radixCounts];

    // The next free slot per digit, which each round of keys advances.
    if (thread < LIGHT_TREE_RADIX_DIGITS)
        gs_DigitCounts[thread] = counts[thread * LightTreeBlockCount() + group.x];
    GroupMemoryBarrierWithGroupSync();

    uint lanes = WaveGetLaneCount();
    uint wave = thread / lanes;
    uint waves = LIGHT_TREE_BUILD_GROUP_SIZE / lanes;
    uint first = group.x * LIGHT_TREE_BUILD_BLOCK_SIZE;
    // Rounds go through the block in order and so do the waves and lanes within a round, which keeps the sort stable.
    for (uint round = 0; round < LIGHT_TREE_KEYS_PER_THREAD; ++round)
    {
        uint index = first + round * LIGHT_TREE_BUILD_GROUP_SIZE + thread;
        bool valid = index < g_PushConstants.lightCount;
        uint key = valid ? keysIn[index] : 0;
        uint digit = valid ? RadixDigit(key) : LIGHT_TREE_RADIX_DIGITS;

        uint rankInWave = 0;
        for (uint d = 0; d < LIGHT_TREE_RADIX_DIGITS; ++d)
        {
            bool match = digit == d;
            uint before = WavePrefixCountBits(match);
            uint count = WaveActiveCountBits(match);
            if (match)
                rankInWave = before;
            if (WaveIsFirstLane())
                gs_WaveDigitSlots[wave][d] = count;
        }
        GroupMemoryBarrierWithGroupSync();

        if (thread < LIGHT_TREE_RADIX_DIGITS)
        {
            uint slot = gs_DigitCounts[thread];
            for (uint w = 0; w < waves; ++w)
            {
                uint count = gs_WaveDigitSlots[w][thread];
                gs_WaveDigitSlots[w][thread] = slot;
                slot += count;
            }
            gs_DigitCounts[thread] = slot;
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid)
        {
            uint slot = gs_WaveDigitSlots[wave][digit] + rankInWave;
            keysOut[slot] = key;
            valuesOut[slot] = valuesIn[index];
        }
        // Before the next round overwrites the slots.
        GroupMemoryBarrierWithGroupSync();
    }
}

// The parent link of the group that node j of level d starts, if it starts one. The root group's is never read.
void WriteParentLink(LightTreeLayout layout, uint d, uint j, uint node)
{
    if (j % LIGHT_TREE_WIDTH != 0)
        return;
    RWStructuredBuffer<uint> parents = ResourceDescriptorHeap[g_Resources.parents];
    parents[node / LIGHT_TREE_WIDTH] = d == 0 ? 0 : layout.Offset(d - 1) + j / LIGHT_TREE_WIDTH;
}

// Leaf j: the sorted lights from j * leafSize on, picked within it in proportion to their flux. The lights without
// flux, sorted to the end, get no entry of the leaf, a leaf slot with a PMF of 0 and a PMF of 0 at node 0, which the
// PMF walk turns into 0 after one group.
[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void BuildLeaves(uint3 dispatchThread : SV_DispatchThreadID)
{
    LightTreeLayout layout = MakeLightTreeLayout();
    uint d = layout.leafLevel;
    uint j = dispatchThread.x;
    if (j >= layout.Nodes(d))
        return;

    RWStructuredBuffer<uint> keys = ResourceDescriptorHeap[g_Resources.keysA];
    RWStructuredBuffer<uint> values = ResourceDescriptorHeap[g_Resources.valuesA];
    RWStructuredBuffer<GpuLightTreeCluster> lightClusters = ResourceDescriptorHeap[g_Resources.lightClusters];
    RWStructuredBuffer<GpuLightTreeCluster> nodeClusters = ResourceDescriptorHeap[g_Resources.nodeClusters];
    RWStructuredBuffer<GpuLightTreeNode> nodes = ResourceDescriptorHeap[g_Resources.nodes];
    RWStructuredBuffer<GpuLightTreeLeafEntry> leafLights = ResourceDescriptorHeap[g_Resources.leafLights];
    RWStructuredBuffer<GpuLightTreeLeafEntry> leafOfLight = ResourceDescriptorHeap[g_Resources.leafOfLight];

    uint node = layout.Offset(d) + j;
    WriteParentLink(layout, d, j, node);

    uint first = j * g_PushConstants.leafSize;
    uint count = j < layout.leafCount ? min(g_PushConstants.leafSize, g_PushConstants.lightCount - first) : 0;
    GpuLightTreeCluster children[LIGHT_TREE_MAX_LEAF_LIGHTS];
    uint lights[LIGHT_TREE_MAX_LEAF_LIGHTS];
    uint valid = 0;
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_MAX_LEAF_LIGHTS; ++i)
    {
        children[i] = (GpuLightTreeCluster)0;
        lights[i] = 0;
        if (i < count)
        {
            lights[i] = values[first + i];
            if (keys[first + i] != LIGHT_TREE_INVALID_KEY)
            {
                children[i] = lightClusters[lights[i]];
                valid = i + 1;
            }
        }
    }

    GpuLightTreeCluster leaf = CombineClusters(children, valid);
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_MAX_LEAF_LIGHTS; ++i)
    {
        if (i < count)
        {
            float pmf = i < valid ? children[i].flux / leaf.flux : 0.0;
            GpuLightTreeLeafEntry slot = { lights[i], pmf };
            leafLights[first + i] = slot;
            GpuLightTreeLeafEntry entry = { i < valid ? node : 0, pmf };
            leafOfLight[lights[i]] = entry;
        }
    }

    uint link = LIGHT_TREE_LEAF_BIT | ((max(valid, 1) - 1) << LIGHT_TREE_LEAF_COUNT_SHIFT) | first;
    nodes[node] = PackLightTreeNode(leaf, link);
    nodeClusters[node] = leaf;
}

// Node j of interior level d from its children's clusters, which the caller loaded only for the nodes that hold
// something (j below the level's width); returns its cluster.
GpuLightTreeCluster BuildInteriorNode(LightTreeLayout layout, uint d, uint j, GpuLightTreeCluster children[LIGHT_TREE_WIDTH])
{
    RWStructuredBuffer<GpuLightTreeCluster> nodeClusters = ResourceDescriptorHeap[g_Resources.nodeClusters];
    RWStructuredBuffer<GpuLightTreeNode> nodes = ResourceDescriptorHeap[g_Resources.nodes];

    uint node = layout.Offset(d) + j;
    WriteParentLink(layout, d, j, node);
    GpuLightTreeCluster cluster = j < layout.Width(d) ? CombineClusters(children, LIGHT_TREE_WIDTH)
                                                       : (GpuLightTreeCluster)0;
    nodes[node] = PackLightTreeNode(cluster, layout.Offset(d + 1) + j * LIGHT_TREE_WIDTH);
    nodeClusters[node] = cluster;
    return cluster;
}

// One interior level, pass, whose children the previous dispatch wrote.
[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void BuildLevel(uint3 dispatchThread : SV_DispatchThreadID)
{
    LightTreeLayout layout = MakeLightTreeLayout();
    uint d = g_PushConstants.pass;
    uint j = dispatchThread.x;
    if (j >= layout.Nodes(d))
        return;

    RWStructuredBuffer<GpuLightTreeCluster> nodeClusters = ResourceDescriptorHeap[g_Resources.nodeClusters];
    GpuLightTreeCluster children[LIGHT_TREE_WIDTH];
    uint firstChild = layout.Offset(d + 1) + j * LIGHT_TREE_WIDTH;
    bool holds = j < layout.Width(d);
    [unroll]
    for (uint c = 0; c < LIGHT_TREE_WIDTH; ++c)
        children[c] = holds ? nodeClusters[firstChild + c] : (GpuLightTreeCluster)0;
    BuildInteriorNode(layout, d, j, children);
}

// The clusters of the last two levels built, by level parity.
groupshared GpuLightTreeCluster gs_LevelClusters[2][LIGHT_TREE_BUILD_TOP_NODES];

// Levels pass to 0, each of at most LIGHT_TREE_BUILD_TOP_NODES nodes, so one thread per node; level pass reads its
// children from memory, the levels above from group shared memory.
[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_TOP_NODES, 1, 1)]
void BuildTopLevels(uint thread : SV_GroupIndex)
{
    LightTreeLayout layout = MakeLightTreeLayout();
    RWStructuredBuffer<GpuLightTreeCluster> nodeClusters = ResourceDescriptorHeap[g_Resources.nodeClusters];
    uint top = g_PushConstants.pass;
    for (int d = int(top); d >= 0; --d)
    {
        if (thread < layout.Nodes(d))
        {
            GpuLightTreeCluster children[LIGHT_TREE_WIDTH];
            uint firstChild = thread * LIGHT_TREE_WIDTH;
            uint memoryOffset = d == int(top) ? layout.Offset(d + 1) : 0;
            bool holds = thread < layout.Width(d);
            [unroll]
            for (uint c = 0; c < LIGHT_TREE_WIDTH; ++c)
            {
                children[c] = !holds                ? (GpuLightTreeCluster)0
                              : d == int(top)       ? nodeClusters[memoryOffset + firstChild + c]
                                                    : gs_LevelClusters[(d + 1) & 1][firstChild + c];
            }
            gs_LevelClusters[d & 1][thread] = BuildInteriorNode(layout, d, thread, children);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// Each light's flux as a share of the total, quantised to an integer, summed over the lights in order. Integers make
// the CDF exact, so it is monotonic and each light's PMF is exactly the share of the draws that land in its interval.

// The total flux, of the root group's clusters.
float LightTreeTotalFlux()
{
    RWStructuredBuffer<GpuLightTreeCluster> nodeClusters = ResourceDescriptorHeap[g_Resources.nodeClusters];
    float total = 0.0;
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_WIDTH; ++i)
        total += nodeClusters[i].flux;
    return total;
}

// Quantises the block's fluxes into keysB and sums them per block.
[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void PowerReduce(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    RWStructuredBuffer<GpuLightTreeCluster> lightClusters = ResourceDescriptorHeap[g_Resources.lightClusters];
    RWStructuredBuffer<uint> quantised = ResourceDescriptorHeap[g_Resources.keysB];
    RWStructuredBuffer<uint> blockSums = ResourceDescriptorHeap[g_Resources.blockSums];

    float total = LightTreeTotalFlux();
    float scale = total > 0.0 ? LIGHT_TREE_POWER_SCALE / total : 0.0;
    uint first = group.x * LIGHT_TREE_BUILD_BLOCK_SIZE + thread * LIGHT_TREE_KEYS_PER_THREAD;
    uint sum = 0;
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_KEYS_PER_THREAD; ++i)
    {
        uint light = first + i;
        if (light < g_PushConstants.lightCount)
        {
            uint q = uint(min(lightClusters[light].flux * scale, LIGHT_TREE_POWER_SCALE));
            quantised[light] = q;
            sum += q;
        }
    }
    uint blockSum;
    GroupExclusiveSum(sum, thread, blockSum);
    if (thread == 0)
        blockSums[group.x] = blockSum;
}

[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void ScanPowerBlocks(uint thread : SV_GroupIndex)
{
    RWStructuredBuffer<uint> blockSums = ResourceDescriptorHeap[g_Resources.blockSums];
    ScanInPlace(blockSums, LightTreeBlockCount(), thread);
}

// Each light's interval end and PMF.
[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void PowerApply(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    RWStructuredBuffer<uint> quantised = ResourceDescriptorHeap[g_Resources.keysB];
    RWStructuredBuffer<uint> blockSums = ResourceDescriptorHeap[g_Resources.blockSums];
    RWStructuredBuffer<GpuLightPowerEntry> power = ResourceDescriptorHeap[g_Resources.power];

    uint first = group.x * LIGHT_TREE_BUILD_BLOCK_SIZE + thread * LIGHT_TREE_KEYS_PER_THREAD;
    uint q[LIGHT_TREE_KEYS_PER_THREAD];
    uint sum = 0;
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_KEYS_PER_THREAD; ++i)
    {
        q[i] = first + i < g_PushConstants.lightCount ? quantised[first + i] : 0;
        sum += q[i];
    }
    uint blockTotal;
    uint end = blockSums[group.x] + GroupExclusiveSum(sum, thread, blockTotal);
    [unroll]
    for (uint i = 0; i < LIGHT_TREE_KEYS_PER_THREAD; ++i)
    {
        end += q[i];
        if (first + i < g_PushConstants.lightCount)
        {
            GpuLightPowerEntry entry = { end, float(q[i]) * LIGHT_TREE_POWER_INV_SCALE, 0, 0 };
            power[first + i] = entry;
        }
    }
}

// Guide bucket k of 2^guideBits covers [k, k + 1) << (31 - guideBits) of the CDF; its entry is the light whose interval
// holds the bucket's start, the last light when no interval does.
[shader("compute")]
[numthreads(LIGHT_TREE_BUILD_GROUP_SIZE, 1, 1)]
void PowerGuide(uint3 dispatchThread : SV_DispatchThreadID)
{
    RWStructuredBuffer<GpuLightPowerEntry> power = ResourceDescriptorHeap[g_Resources.power];
    uint lightCount = g_PushConstants.lightCount;
    uint guideBits = firstbithigh(lightCount);
    uint bucket = dispatchThread.x;
    if (bucket >= (1u << guideBits))
        return;

    // The first light whose interval ends past the bucket's start.
    uint start = bucket << (31 - guideBits);
    uint lo = 0;
    uint hi = lightCount - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        if (power[mid].cdfEnd > start)
            hi = mid;
        else
            lo = mid + 1;
    }
    power[bucket].guide = lo;
}
