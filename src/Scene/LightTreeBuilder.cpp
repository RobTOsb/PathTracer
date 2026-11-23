#include "LightTreeBuilder.h"

#include "Renderer/FrameGraph.h"

#include <algorithm>
#include <array>
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
	for (ResourceHandle *buffer: { &fluxScales_, &placeholders_[kNodes], &placeholders_[kLeafLights],
								   &placeholders_[kLeafOfLight], &placeholders_[kPower] })
	{
		if (*buffer != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(*buffer);
			*buffer = InvalidResourceHandle;
		}
	}
	for (ResourceHandle buffer: resources_)
	{
		renderContext_->DestroyResource(buffer);
	}
	resources_.clear();
	outputs_ = {};
	lightCount_ = 0;
	depth_ = 0;
	leafSize_ = 1;
}

void LightTreeBuilder::Reset(std::span<const float> fluxScales, ResourceHandle lightEmitters)
{
	renderContext_->WaitForGPUIdle();
	DestroyBuffers();
	lightCount_ = static_cast<uint32_t>(fluxScales.size());
	lightEmitters_ = lightEmitters;

	// At least one element each, so the bindings stay valid without lights. The tree is sized for one light per
	// leaf, the most nodes any leaf size needs, so the leaf size can change without a new graph.
	const uint32_t count = std::max(lightCount_, 1u);
	lightElements_ = count;
	nodeElements_ = TreeLayout(count, 1).NodeCount();
	blockElements_ = GroupCount(count, LIGHT_TREE_BUILD_BLOCK_SIZE);
	auto create = [&](uint32_t elements, uint32_t stride, const char *debugName)
	{
		BufferDesc desc = {};
		desc.sizeInBytes = static_cast<uint64_t>(elements) * stride;
		desc.structureByteStride = stride;
		desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		desc.debugName = debugName;
		return renderContext_->CreateBuffer(desc, ResourceViewType::UnorderedAccess);
	};
	if (lightCount_ == 0)
	{
		placeholders_[kNodes] = create(nodeElements_, sizeof(GpuLightTreeNode), "LightTree_Nodes");
		placeholders_[kLeafLights] = create(count, sizeof(GpuLightTreeLeafEntry), "LightTree_LeafLights");
		placeholders_[kLeafOfLight] = create(count, sizeof(GpuLightTreeLeafEntry), "LightTree_LeafOfLight");
		placeholders_[kPower] = create(count, sizeof(GpuLightPowerEntry), "LightTree_Power");
	}

	std::vector<float> scales(fluxScales.begin(), fluxScales.end());
	scales.resize(count, 1.0f);
	BufferDesc desc = {};
	desc.sizeInBytes = count * sizeof(float);
	desc.structureByteStride = sizeof(float);
	desc.debugName = "LightTree_FluxScales";
	const UploadBatchHandle batch = renderContext_->BeginUpload();
	fluxScales_ = renderContext_->CreateBuffer(desc, ResourceViewType::ShaderResource, scales.data(), batch);
	renderContext_->EndUpload(batch);

	resources_.resize(renderContext_->GetFramesInFlight());
	for (ResourceHandle &resources: resources_)
	{
		resources = renderContext_->CreateConstantBuffer(sizeof(LightTreeBuildResources), "LightTree_BuildResources");
	}
}

void LightTreeBuilder::AddPasses(FrameGraph &frame, const Inputs &inputs)
{
	outputs_ = {};
	if (resources_.empty())
	{
		return;
	}
	if (lightCount_ == 0)
	{
		outputs_ = { frame.ImportBuffer(placeholders_[kNodes], "LightTree_Nodes"),
					 frame.ImportBuffer(placeholders_[kLeafLights], "LightTree_LeafLights"),
					 frame.ImportBuffer(placeholders_[kLeafOfLight], "LightTree_LeafOfLight"),
					 frame.ImportBuffer(placeholders_[kPower], "LightTree_Power") };
		return;
	}

	const uint32_t leafSize = std::clamp(inputs.maxLeafLights, 1u, LIGHT_TREE_MAX_LEAF_LIGHTS);
	const TreeLayout layout(lightCount_, leafSize);
	depth_ = layout.leafLevel + 1;
	leafSize_ = leafSize;
	const uint32_t lightCount = lightCount_;
	const uint32_t blockCount = GroupCount(lightCount_, LIGHT_TREE_BUILD_BLOCK_SIZE);

	LightTreeBuildConstants constants = {};
	const float extent = std::max({ inputs.boundsMax.x - inputs.boundsMin.x,
									inputs.boundsMax.y - inputs.boundsMin.y,
									inputs.boundsMax.z - inputs.boundsMin.z,
									0.0f });
	constants.boundsScale = extent > 0.0f ? 1024.0f / extent : 0.0f;
	// The node means are 16-bit steps over the same cube (PackLightTreeNode). The bounds are only finite with lights.
	origin_ = isfinite(extent) ? inputs.boundsMin : float3(0.0f, 0.0f, 0.0f);
	step_ = isfinite(extent) && extent > 0.0f ? extent / 65535.0f : 0.0f;
	constants.boundsMin = origin_;
	constants.quantStep = step_;
	constants.lightCount = lightCount_;
	constants.leafSize = leafSize;
	constants.lightsHandle = frame.DescriptorIndex(inputs.lights);
	constants.emissionScalesHandle = frame.DescriptorIndex(inputs.emissionScales);

	// The scratch of one build, and its outputs, which nothing reads after this frame's paths.
	auto transient = [&](uint32_t elements, uint32_t stride, const char *debugName)
	{
		BufferDesc desc = {};
		desc.sizeInBytes = static_cast<uint64_t>(elements) * stride;
		desc.structureByteStride = stride;
		desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		desc.debugName = debugName;
		return frame.graph.CreateBuffer(desc, ResourceViewType::UnorderedAccess);
	};
	const rg::ResourceId lightClusters = transient(lightElements_, sizeof(GpuLightTreeCluster), "LightTree_LightClusters");
	const rg::ResourceId nodeClusters = transient(nodeElements_, sizeof(GpuLightTreeCluster), "LightTree_NodeClusters");
	const rg::ResourceId keysA = transient(lightElements_, sizeof(uint32_t), "LightTree_KeysA");
	const rg::ResourceId keysB = transient(lightElements_, sizeof(uint32_t), "LightTree_KeysB");
	const rg::ResourceId valuesA = transient(lightElements_, sizeof(uint32_t), "LightTree_ValuesA");
	const rg::ResourceId valuesB = transient(lightElements_, sizeof(uint32_t), "LightTree_ValuesB");
	const rg::ResourceId radixCounts =
			transient((1u << LIGHT_TREE_RADIX_BITS) * blockElements_, sizeof(uint32_t), "LightTree_RadixCounts");
	const rg::ResourceId blockSums = transient(blockElements_, sizeof(uint32_t), "LightTree_BlockSums");

	const rg::ResourceId nodes = transient(nodeElements_, sizeof(GpuLightTreeNode), "LightTree_Nodes");
	const rg::ResourceId leafLights = transient(lightElements_, sizeof(GpuLightTreeLeafEntry), "LightTree_LeafLights");
	const rg::ResourceId leafOfLight = transient(lightElements_, sizeof(GpuLightTreeLeafEntry), "LightTree_LeafOfLight");
	const rg::ResourceId power = transient(lightElements_, sizeof(GpuLightPowerEntry), "LightTree_Power");
	outputs_ = { nodes, leafLights, leafOfLight, power };

	const rg::ResourceId lights = frame.ImportBuffer(inputs.lights, "Scene_Lights");

	// Every kernel reads and writes through UAVs. Between the kernels of one pass, which read what the one before
	// wrote, global UAV barriers order them, leaving every buffer in the state the pass declared.
	struct Dispatch
	{
		PipelineHandle pipeline;
		uint32_t groups;
		uint32_t pass;
	};
	const ResourceHandle resourceBuffer = resources_[frame.rc.GetCurrentFrameIndex()];
	auto record = [constants, resourceBuffer](RenderContext &rc, std::span<const Dispatch> dispatches)
	{
		LightTreeBuildConstants c = constants;
		for (size_t i = 0; i < dispatches.size(); ++i)
		{
			if (i > 0)
			{
				rc.GlobalBarrier(ResourceSync::ComputeShading,
								 ResourceAccess::UnorderedAccess,
								 ResourceSync::ComputeShading,
								 ResourceAccess::UnorderedAccess);
			}
			c.pass = dispatches[i].pass;
			rc.SetComputePipeline(dispatches[i].pipeline);
			rc.SetCompute32BitConstants(&c, sizeof(c) / sizeof(uint32_t));
			rc.SetComputeConstantBuffer(1, resourceBuffer);
			rc.Dispatch(dispatches[i].groups, 1, 1);
		}
	};
	const rg::Access rw = rg::Access::ComputeShaderReadWrite;

	// The descriptors of this frame's transients, which the graph has placed by the time the first pass runs.
	const rg::RenderGraph &graph = frame.graph;
	const ResourceHandle fluxScales = fluxScales_;
	const ResourceHandle lightEmitters = lightEmitters_;
	const std::array<rg::ResourceId, 8> scratchIds = { lightClusters, nodeClusters, keysA, valuesA,
													   keysB,		  valuesB,		radixCounts, blockSums };
	const std::array<rg::ResourceId, kOutputCount> outputIds = outputs_;
	const std::array<Dispatch, 1> keys = { { { keysPipeline_, GroupCount(lightCount, LIGHT_TREE_BUILD_GROUP_SIZE), 0 } } };
	frame.AddPass(rg::Pass::Compute("LightTree/Keys")
								.Use(lights, rw)
								.Use(lightClusters, rw)
								.Use(keysA, rw)
								.Use(valuesA, rw)
								.Executes(
										[&graph, record, keys, resourceBuffer, fluxScales, lightEmitters, outputIds,
										 scratchIds](rg::PassContext &ctx)
										{
											auto index = [&](ResourceHandle handle)
											{ return static_cast<uint>(ctx.rc.GetResourceViewInfo(handle).descriptorIndex); };
											auto scratchIndex = [&](size_t i) { return index(graph.GetHandle(scratchIds[i])); };
											auto outputIndex = [&](size_t i) { return index(graph.GetHandle(outputIds[i])); };
											LightTreeBuildResources resources = {};
											resources.fluxScales = index(fluxScales);
											resources.lightEmitters = index(lightEmitters);
											resources.lightClusters = scratchIndex(0);
											resources.nodeClusters = scratchIndex(1);
											resources.keysA = scratchIndex(2);
											resources.valuesA = scratchIndex(3);
											resources.keysB = scratchIndex(4);
											resources.valuesB = scratchIndex(5);
											resources.radixCounts = scratchIndex(6);
											resources.blockSums = scratchIndex(7);
											resources.nodes = outputIndex(kNodes);
											resources.leafLights = outputIndex(kLeafLights);
											resources.leafOfLight = outputIndex(kLeafOfLight);
											resources.power = outputIndex(kPower);
											ctx.rc.UpdateConstantBuffer(resourceBuffer, &resources, sizeof(resources));
											record(ctx.rc, keys);
										}));

	std::vector<Dispatch> sort;
	for (uint32_t pass = 0; pass < LIGHT_TREE_RADIX_PASSES; ++pass)
	{
		sort.push_back({ radixCountPipeline_, blockCount, pass });
		sort.push_back({ radixScanPipeline_, 1, pass });
		sort.push_back({ radixScatterPipeline_, blockCount, pass });
	}
	frame.AddPass(rg::Pass::Compute("LightTree/Sort")
								.Use(keysA, rw)
								.Use(valuesA, rw)
								.Use(keysB, rw)
								.Use(valuesB, rw)
								.Use(radixCounts, rw)
								.Executes([record, sort](rg::PassContext &ctx) { record(ctx.rc, sort); }));

	// Bottom up: the leaves, the wide interior levels one dispatch each, then one group for the rest.
	std::vector<Dispatch> build;
	build.push_back({ leavesPipeline_, GroupCount(layout.Nodes(layout.leafLevel), LIGHT_TREE_BUILD_GROUP_SIZE), 0 });
	int level = static_cast<int>(layout.leafLevel) - 1;
	for (; level >= 0 && layout.Nodes(level) > LIGHT_TREE_BUILD_TOP_NODES; --level)
	{
		build.push_back({ levelPipeline_, GroupCount(layout.Nodes(level), LIGHT_TREE_BUILD_GROUP_SIZE), static_cast<uint32_t>(level) });
	}
	if (level >= 0)
	{
		build.push_back({ topLevelsPipeline_, 1, static_cast<uint32_t>(level) });
	}
	frame.AddPass(rg::Pass::Compute("LightTree/Nodes")
								.Use(keysA, rw)
								.Use(valuesA, rw)
								.Use(lightClusters, rw)
								.Use(nodeClusters, rw)
								.Use(nodes, rw)
								.Use(leafLights, rw)
								.Use(leafOfLight, rw)
								.Executes([record, build](rg::PassContext &ctx) { record(ctx.rc, build); }));

	const std::array<Dispatch, 4> powerList = { {
			{ powerReducePipeline_, blockCount, 0 },
			{ powerScanPipeline_, 1, 0 },
			{ powerApplyPipeline_, blockCount, 0 },
			{ powerGuidePipeline_, GroupCount(std::bit_floor(lightCount), LIGHT_TREE_BUILD_GROUP_SIZE), 0 },
	} };
	frame.AddPass(rg::Pass::Compute("LightTree/Power")
								.Use(lightClusters, rw)
								.Use(nodeClusters, rw)
								.Use(keysB, rw)
								.Use(blockSums, rw)
								.Use(power, rw)
								.Executes([record, powerList](rg::PassContext &ctx) { record(ctx.rc, powerList); }));
}
