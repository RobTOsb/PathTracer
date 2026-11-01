#include "LightTreeBuilder.h"

#include <algorithm>
#include <bit>
#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
	float Luminance(const float3 &c)
	{
		return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
	}

	uint32_t GroupCount(uint32_t threads, uint32_t groupSize)
	{
		return (threads + groupSize - 1) / groupSize;
	}

	// The tree over lightCount lights at leafSize per leaf, as LightTreeLayout in LightTreeBuild.slang lays it out:
	// level 0 the root group, the leaves the deepest level, each level a quarter of the one below rounded up and
	// padded to whole groups.
	struct TreeLayout
	{
		uint32_t leafCount;
		uint32_t leafLevel;

		TreeLayout(uint32_t lightCount, uint32_t leafSize) :
			leafCount((lightCount + leafSize - 1) / leafSize),
			leafLevel(leafCount <= LIGHT_TREE_WIDTH ? 0 : (static_cast<uint32_t>(std::bit_width(leafCount - 1)) - 1) / 2)
		{
		}

		uint32_t Width(uint32_t d) const
		{
			const uint32_t shift = 2 * (leafLevel - d);
			return (leafCount + (1u << shift) - 1) >> shift;
		}

		uint32_t Nodes(uint32_t d) const
		{
			return (Width(d) + LIGHT_TREE_WIDTH - 1) & ~(LIGHT_TREE_WIDTH - 1);
		}

		uint32_t NodeCount() const
		{
			uint32_t count = 0;
			for (uint32_t d = 0; d <= leafLevel; ++d)
			{
				count += Nodes(d);
			}
			return count;
		}
	};
} // namespace

float LightTreeFlux(const GpuLight &light, float fluxScale)
{
	GpuLight copy = light; // GetLightType is not const
	if (copy.GetLightType() != kLight_Area)
	{
		return 0.0f;
	}
	const float3 p0 = { light.v1.x, light.v1.y, light.v1.z };
	const float3 p1 = { light.v2.x, light.v2.y, light.v2.z };
	const float3 p2 = { light.v3.x, light.v3.y, light.v3.z };
	const float3 crossE = Cross(p1 - p0, p2 - p0);
	const float twiceArea = sqrtf(Dot(crossE, crossE));
	const float flux = Pi * Luminance({ light.radiance.x, light.radiance.y, light.radiance.z }) * 0.5f * twiceArea * fluxScale;
	return flux > 0.0f && isfinite(flux) ? flux : 0.0f;
}

LightTreeBuilder::LightTreeBuilder(RenderContext *renderContext) : renderContext_(renderContext)
{
	for (auto [pipeline, entryPoint]: { std::pair{ &keysPipeline_, "LightKeys" },
										std::pair{ &radixCountPipeline_, "RadixCount" },
										std::pair{ &radixScanPipeline_, "ScanRadixCounts" },
										std::pair{ &radixScatterPipeline_, "RadixScatter" },
										std::pair{ &leavesPipeline_, "BuildLeaves" },
										std::pair{ &levelPipeline_, "BuildLevel" },
										std::pair{ &topLevelsPipeline_, "BuildTopLevels" },
										std::pair{ &powerReducePipeline_, "PowerReduce" },
										std::pair{ &powerScanPipeline_, "ScanPowerBlocks" },
										std::pair{ &powerApplyPipeline_, "PowerApply" },
										std::pair{ &powerGuidePipeline_, "PowerGuide" } })
	{
		*pipeline = renderContext->CreateComputePipeline(PATHTRACER_SHADER_DIR "LightTreeBuild.slang", entryPoint);
		if (*pipeline == InvalidPipelineHandle)
		{
			throw std::runtime_error(std::string("Failed to create the light tree build pipeline ") + entryPoint +
									 " (see the shader log above).");
		}
	}
}

LightTreeBuilder::~LightTreeBuilder()
{
	DestroyBuffers();
}

void LightTreeBuilder::DestroyBuffers()
{
	for (ResourceHandle *buffer: { &resources_,
								   &fluxScales_,
								   &lightClusters_,
								   &nodeClusters_,
								   &keys_[0],
								   &keys_[1],
								   &values_[0],
								   &values_[1],
								   &radixCounts_,
								   &blockSums_,
								   &nodes_,
								   &parents_,
								   &leafLights_,
								   &leafOfLight_,
								   &power_ })
	{
		if (*buffer != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(*buffer);
			*buffer = InvalidResourceHandle;
		}
	}
	lightCount_ = 0;
	depth_ = 0;
}

void LightTreeBuilder::Reset(std::span<const float> fluxScales, ResourceHandle lightEmitters)
{
	if (fluxScales.size() > static_cast<size_t>(LIGHT_TREE_LEAF_FIRST_MASK) + 1)
	{
		// Leaves address their first slot with 28 bits.
		throw std::runtime_error("Too many lights for the light tree");
	}

	renderContext_->WaitForGPUIdle();
	DestroyBuffers();
	lightCount_ = static_cast<uint32_t>(fluxScales.size());

	// At least one element each, so the bindings stay valid without lights. The tree is sized for one light per
	// leaf, the most nodes any leaf size needs, so the leaf size can change without new buffers.
	const uint32_t count = std::max(lightCount_, 1u);
	const uint32_t nodeCount = TreeLayout(count, 1).NodeCount();
	const uint32_t blockCount = GroupCount(count, LIGHT_TREE_BUILD_BLOCK_SIZE);
	auto create = [&](uint32_t elements, uint32_t stride, const char *debugName)
	{
		BufferDesc desc = {};
		desc.sizeInBytes = static_cast<uint64_t>(elements) * stride;
		desc.structureByteStride = stride;
		desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		desc.debugName = debugName;
		return renderContext_->CreateBuffer(desc, ResourceViewType::UnorderedAccess);
	};
	lightClusters_ = create(count, sizeof(GpuLightTreeCluster), "LightTree_LightClusters");
	nodeClusters_ = create(nodeCount, sizeof(GpuLightTreeCluster), "LightTree_NodeClusters");
	keys_[0] = create(count, sizeof(uint32_t), "LightTree_KeysA");
	keys_[1] = create(count, sizeof(uint32_t), "LightTree_KeysB");
	values_[0] = create(count, sizeof(uint32_t), "LightTree_ValuesA");
	values_[1] = create(count, sizeof(uint32_t), "LightTree_ValuesB");
	radixCounts_ = create((1u << LIGHT_TREE_RADIX_BITS) * blockCount, sizeof(uint32_t), "LightTree_RadixCounts");
	blockSums_ = create(blockCount, sizeof(uint32_t), "LightTree_BlockSums");
	nodes_ = create(nodeCount, sizeof(GpuLightTreeNode), "LightTree_Nodes");
	parents_ = create(nodeCount / LIGHT_TREE_WIDTH, sizeof(uint32_t), "LightTree_Parents");
	leafLights_ = create(count, sizeof(GpuLightTreeLeafEntry), "LightTree_LeafLights");
	leafOfLight_ = create(count, sizeof(GpuLightTreeLeafEntry), "LightTree_LeafOfLight");
	power_ = create(count, sizeof(GpuLightPowerEntry), "LightTree_Power");

	std::vector<float> scales(fluxScales.begin(), fluxScales.end());
	scales.resize(count, 1.0f);
	BufferDesc desc = {};
	desc.sizeInBytes = count * sizeof(float);
	desc.structureByteStride = sizeof(float);
	desc.debugName = "LightTree_FluxScales";
	const UploadBatchHandle batch = renderContext_->BeginUpload();
	fluxScales_ = renderContext_->CreateBuffer(desc, ResourceViewType::ShaderResource, scales.data(), batch);
	renderContext_->EndUpload(batch);

	auto index = [&](ResourceHandle buffer) { return static_cast<uint>(renderContext_->GetResourceViewInfo(buffer).descriptorIndex); };
	LightTreeBuildResources resources = {};
	resources.fluxScales = index(fluxScales_);
	resources.lightEmitters = index(lightEmitters);
	resources.lightClusters = index(lightClusters_);
	resources.nodeClusters = index(nodeClusters_);
	resources.keysA = index(keys_[0]);
	resources.valuesA = index(values_[0]);
	resources.keysB = index(keys_[1]);
	resources.valuesB = index(values_[1]);
	resources.radixCounts = index(radixCounts_);
	resources.blockSums = index(blockSums_);
	resources.nodes = index(nodes_);
	resources.parents = index(parents_);
	resources.leafLights = index(leafLights_);
	resources.leafOfLight = index(leafOfLight_);
	resources.power = index(power_);
	resources_ = renderContext_->CreateConstantBuffer(sizeof(resources), "LightTree_BuildResources");
	renderContext_->UpdateConstantBuffer(resources_, &resources, sizeof(resources));
}

void LightTreeBuilder::Build(const Inputs &inputs)
{
	if (lightCount_ == 0)
	{
		return;
	}

	const uint32_t leafSize = std::clamp(inputs.maxLeafLights, 1u, LIGHT_TREE_MAX_LEAF_LIGHTS);
	const TreeLayout layout(lightCount_, leafSize);
	depth_ = layout.leafLevel + 1;
	const uint32_t blockCount = GroupCount(lightCount_, LIGHT_TREE_BUILD_BLOCK_SIZE);

	LightTreeBuildConstants constants = {};
	constants.boundsMin = inputs.boundsMin;
	const float extent = std::max({ inputs.boundsMax.x - inputs.boundsMin.x,
									inputs.boundsMax.y - inputs.boundsMin.y,
									inputs.boundsMax.z - inputs.boundsMin.z,
									0.0f });
	constants.boundsScale = extent > 0.0f ? 1024.0f / extent : 0.0f;
	constants.lightCount = lightCount_;
	constants.leafSize = leafSize;
	constants.lightsHandle = renderContext_->GetResourceViewInfo(inputs.lights).descriptorIndex;
	constants.emissionScalesHandle = renderContext_->GetResourceViewInfo(inputs.emissionScales).descriptorIndex;

	renderContext_->BeginEvent("LightTree");
	renderContext_->BeginProfileScope("LightTree");

	// The barriers also order this frame's writes after last frame's reads. Between the kernels, which all read
	// what the one before wrote, global barriers do instead, leaving these states as they are.
	const ResourceHandle outputs[] = { nodes_, parents_, leafLights_, leafOfLight_, power_ };
	for (ResourceHandle buffer: { lightClusters_, nodeClusters_, keys_[0], keys_[1], values_[0], values_[1], radixCounts_, blockSums_ })
	{
		renderContext_->TransitionResource(buffer, ResourceLayout::UnorderedAccess, ResourceSync::ComputeShading, ResourceAccess::UnorderedAccess);
	}
	for (ResourceHandle buffer: outputs)
	{
		renderContext_->TransitionResource(buffer, ResourceLayout::UnorderedAccess, ResourceSync::ComputeShading, ResourceAccess::UnorderedAccess);
	}

	auto dispatch = [&](PipelineHandle pipeline, uint32_t groups, uint32_t pass)
	{
		constants.pass = pass;
		renderContext_->SetComputePipeline(pipeline);
		renderContext_->SetCompute32BitConstants(&constants, sizeof(constants) / sizeof(uint32_t));
		renderContext_->SetComputeConstantBuffer(1, resources_);
		renderContext_->Dispatch(groups, 1, 1);
		renderContext_->GlobalBarrier(ResourceSync::ComputeShading,
									  ResourceAccess::UnorderedAccess,
									  ResourceSync::ComputeShading,
									  ResourceAccess::UnorderedAccess);
	};

	dispatch(keysPipeline_, GroupCount(lightCount_, LIGHT_TREE_BUILD_GROUP_SIZE), 0);
	for (uint32_t pass = 0; pass < LIGHT_TREE_RADIX_PASSES; ++pass)
	{
		dispatch(radixCountPipeline_, blockCount, pass);
		dispatch(radixScanPipeline_, 1, pass);
		dispatch(radixScatterPipeline_, blockCount, pass);
	}

	// Bottom up: the leaves, the wide interior levels one dispatch each, then one group for the rest.
	dispatch(leavesPipeline_, GroupCount(layout.Nodes(layout.leafLevel), LIGHT_TREE_BUILD_GROUP_SIZE), 0);
	int level = static_cast<int>(layout.leafLevel) - 1;
	for (; level >= 0 && layout.Nodes(level) > LIGHT_TREE_BUILD_TOP_NODES; --level)
	{
		dispatch(levelPipeline_, GroupCount(layout.Nodes(level), LIGHT_TREE_BUILD_GROUP_SIZE), level);
	}
	if (level >= 0)
	{
		dispatch(topLevelsPipeline_, 1, level);
	}

	dispatch(powerReducePipeline_, blockCount, 0);
	dispatch(powerScanPipeline_, 1, 0);
	dispatch(powerApplyPipeline_, blockCount, 0);
	dispatch(powerGuidePipeline_, GroupCount(std::bit_floor(lightCount_), LIGHT_TREE_BUILD_GROUP_SIZE), 0);

	for (ResourceHandle buffer: outputs)
	{
		renderContext_->TransitionResource(buffer, ResourceLayout::ShaderResource, ResourceSync::Raytracing, ResourceAccess::ShaderResource);
	}

	renderContext_->EndProfileScope();
	renderContext_->EndEvent();
}
