#include "LightTreeBuilder.h"

#include "Renderer/FrameGraph.h"

#include "fmtlog.h"

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
	// padded to whole groups. A BLAS is laid out the same way from its node offset.
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

	// Nodes of the two-level hierarchy at a leaf size: the TLAS (one BLAS a leaf), then every BLAS.
	uint64_t TwoLevelNodeCount(std::span<const LightBlasDesc> blases, uint32_t leafSize)
	{
		uint64_t count = TreeLayout(std::max<uint32_t>(static_cast<uint32_t>(blases.size()), 1), 1).NodeCount();
		for (const LightBlasDesc &blas: blases)
		{
			count += TreeLayout(blas.lightCount, leafSize).NodeCount();
		}
		return count;
	}
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
										std::pair{ &powerGuidePipeline_, "PowerGuide" },
										std::pair{ &clearBoundsPipeline_, "ClearBlasBounds" },
										std::pair{ &boundsPipeline_, "BlasBounds" },
										std::pair{ &topologyKeysPipeline_, "TopologyKeys" },
										std::pair{ &topologySlotsPipeline_, "TopologySlots" },
										std::pair{ &refitLeavesPipeline_, "RefitLeaves" },
										std::pair{ &refitInteriorPipeline_, "RefitInterior" },
										std::pair{ &blasClustersPipeline_, "BlasClusters" } })
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

void LightTreeBuilder::DestroyLayout()
{
	for (ResourceHandle *buffer: { &blases_, &refitEntries_ })
	{
		if (*buffer != InvalidResourceHandle)
		{
			renderContext_->DestroyResource(*buffer);
			*buffer = InvalidResourceHandle;
		}
	}
	heights_.clear();
	layoutValid_ = false;
	topologyPending_ = false;
}

void LightTreeBuilder::DestroyBuffers()
{
	DestroyLayout();
	for (ResourceHandle *buffer: { &fluxScales_,
								   &lightBlas_,
								   &lightClusters_,
								   &nodeClusters_,
								   &blasClusters_,
								   &blasBounds_,
								   &placeholderBlases_,
								   &outputBuffers_[kNodes],
								   &outputBuffers_[kLeafLights],
								   &outputBuffers_[kLeafOfLight],
								   &outputBuffers_[kPower] })
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
	blasDescs_.clear();
	outputs_ = {};
	lightCount_ = 0;
	depth_ = 0;
}

void LightTreeBuilder::Reset(std::span<const float> fluxScales,
							 ResourceHandle lightEmitters,
							 std::span<const LightBlasDesc> blases)
{
	renderContext_->WaitForGPUIdle();
	DestroyBuffers();
	lightCount_ = static_cast<uint32_t>(fluxScales.size());
	lightEmitters_ = lightEmitters;

	// The BLASes, one over every light if none were given, each light in exactly one. The sort key holds the BLAS
	// index above the Morton code, so past LIGHT_TREE_MAX_BLAS_BITS of them neighbours share one.
	if (lightCount_ > 0)
	{
		if (blases.empty())
		{
			blasDescs_.push_back({ 0, lightCount_, true });
		}
		else
		{
			uint32_t next = 0;
			for (const LightBlasDesc &blas: blases)
			{
				if (blas.firstLight != next || blas.lightCount == 0)
				{
					throw std::runtime_error("Light BLASes must cover the lights in order, none empty");
				}
				next += blas.lightCount;
				blasDescs_.push_back(blas);
			}
			if (next != lightCount_)
			{
				throw std::runtime_error("Light BLASes must cover every light");
			}
		}
		while (blasDescs_.size() > (size_t(1) << LIGHT_TREE_MAX_BLAS_BITS))
		{
			std::vector<LightBlasDesc> merged;
			for (size_t i = 0; i < blasDescs_.size(); i += 2)
			{
				LightBlasDesc blas = blasDescs_[i];
				if (i + 1 < blasDescs_.size())
				{
					blas.lightCount += blasDescs_[i + 1].lightCount;
					blas.dynamic = blas.dynamic || blasDescs_[i + 1].dynamic;
				}
				merged.push_back(blas);
			}
			blasDescs_ = std::move(merged);
		}
	}

	// At least one element each, so the bindings stay valid without lights. The nodes are sized for one light per
	// leaf, the most nodes any leaf size needs in either form, so the leaf size can change without new buffers.
	const uint32_t count = std::max(lightCount_, 1u);
	const uint32_t blasCount = std::max(static_cast<uint32_t>(blasDescs_.size()), 1u);
	lightElements_ = count;
	blockElements_ = GroupCount(count, LIGHT_TREE_BUILD_BLOCK_SIZE);
	const uint64_t nodeElements =
			std::max<uint64_t>(TreeLayout(count, 1).NodeCount(), TwoLevelNodeCount(blasDescs_, 1));
	if (nodeElements > UINT32_MAX / sizeof(GpuLightTreeCluster))
	{
		throw std::runtime_error("Too many lights for the light tree");
	}
	auto create = [&](uint64_t elements, uint32_t stride, const char *debugName)
	{
		BufferDesc desc = {};
		desc.sizeInBytes = elements * stride;
		desc.structureByteStride = stride;
		desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		desc.debugName = debugName;
		return renderContext_->CreateBuffer(desc, ResourceViewType::UnorderedAccess);
	};
	outputBuffers_[kNodes] = create(nodeElements, sizeof(GpuLightTreeNode), "LightTree_Nodes");
	outputBuffers_[kLeafLights] = create(uint64_t(count) + blasCount, sizeof(GpuLightTreeLeafEntry), "LightTree_LeafLights");
	outputBuffers_[kLeafOfLight] =
			create(uint64_t(count) + blasCount, sizeof(GpuLightTreeLeafEntry), "LightTree_LeafOfLight");
	outputBuffers_[kPower] = create(count, sizeof(GpuLightPowerEntry), "LightTree_Power");
	lightClusters_ = create(count, sizeof(GpuLightTreeCluster), "LightTree_LightClusters");
	nodeClusters_ = create(nodeElements, sizeof(GpuLightTreeCluster), "LightTree_NodeClusters");
	blasClusters_ = create(blasCount, sizeof(GpuLightTreeCluster), "LightTree_BlasClusters");
	blasBounds_ = create(uint64_t(blasCount) * 6, sizeof(uint32_t), "LightTree_BlasBounds");

	std::vector<float> scales(fluxScales.begin(), fluxScales.end());
	scales.resize(count, 1.0f);
	std::vector<uint32_t> lightBlas(count, 0);
	for (uint32_t b = 0; b < static_cast<uint32_t>(blasDescs_.size()); ++b)
	{
		std::fill_n(lightBlas.begin() + blasDescs_[b].firstLight, blasDescs_[b].lightCount, b);
	}
	const GpuLightBlas noBlas = {};
	auto upload = [&](const void *data, uint64_t elements, uint32_t stride, const char *debugName, UploadBatchHandle batch)
	{
		BufferDesc desc = {};
		desc.sizeInBytes = elements * stride;
		desc.structureByteStride = stride;
		desc.debugName = debugName;
		return renderContext_->CreateBuffer(desc, ResourceViewType::ShaderResource, data, batch);
	};
	const UploadBatchHandle batch = renderContext_->BeginUpload();
	fluxScales_ = upload(scales.data(), count, sizeof(float), "LightTree_FluxScales", batch);
	lightBlas_ = upload(lightBlas.data(), count, sizeof(uint32_t), "LightTree_LightBlas", batch);
	placeholderBlases_ = upload(&noBlas, 1, sizeof(GpuLightBlas), "LightTree_NoBlases", batch);
	renderContext_->EndUpload(batch);

	resources_.resize(renderContext_->GetFramesInFlight());
	for (ResourceHandle &resources: resources_)
	{
		resources = renderContext_->CreateConstantBuffer(sizeof(LightTreeBuildResources), "LightTree_BuildResources");
	}
	layoutValid_ = false;
}

void LightTreeBuilder::Prepare(bool twoLevel, uint32_t maxLeafLights)
{
	const uint32_t leafSize = std::clamp(maxLeafLights, 1u, LIGHT_TREE_MAX_LEAF_LIGHTS);
	if (layoutValid_ && twoLevel == twoLevel_ && leafSize == leafSize_)
	{
		return;
	}
	const bool hadLayout = blases_ != InvalidResourceHandle;
	twoLevel_ = twoLevel;
	leafSize_ = leafSize;
	if (!twoLevel || lightCount_ == 0)
	{
		// The single-level tree is built whole every frame; a two-level layout left over is rebuilt when it is next
		// chosen, as the single-level tree overwrites its nodes.
		layoutValid_ = true;
		topologyPending_ = false;
		return;
	}

	// The frames in flight may still read the old tables.
	if (hadLayout)
	{
		renderContext_->WaitForGPUIdle();
	}
	DestroyLayout();

	// Every BLAS from its node offset after the TLAS's nodes; per height above the leaves (0 = the leaves), the nodes
	// of every BLAS that deep, the dynamic BLASes' first so that a frame refitting only them takes a prefix.
	const uint32_t blasCount = static_cast<uint32_t>(blasDescs_.size());
	std::vector<GpuLightBlas> blases(blasCount);
	std::vector<std::vector<uint32_t>> levelOffsets(blasCount);
	uint32_t nodeOffset = TreeLayout(blasCount, 1).NodeCount();
	uint32_t maxLeafLevel = 0;
	for (uint32_t b = 0; b < blasCount; ++b)
	{
		const LightBlasDesc &desc = blasDescs_[b];
		const TreeLayout layout(desc.lightCount, leafSize);
		blases[b] = { nodeOffset, desc.firstLight, desc.lightCount, layout.leafLevel };
		uint32_t offset = nodeOffset;
		for (uint32_t d = 0; d <= layout.leafLevel; ++d)
		{
			levelOffsets[b].push_back(offset);
			offset += layout.Nodes(d);
		}
		nodeOffset = offset;
		maxLeafLevel = std::max(maxLeafLevel, layout.leafLevel);
	}

	std::vector<GpuLightRefitEntry> entries;
	for (uint32_t height = 0; height <= maxLeafLevel; ++height)
	{
		HeightRange range = { static_cast<uint32_t>(entries.size()), 0, 0 };
		for (bool dynamic: { true, false })
		{
			for (uint32_t b = 0; b < blasCount; ++b)
			{
				const LightBlasDesc &desc = blasDescs_[b];
				const TreeLayout layout(desc.lightCount, leafSize);
				if (desc.dynamic != dynamic || layout.leafLevel < height)
				{
					continue;
				}
				const uint32_t d = layout.leafLevel - height;
				for (uint32_t j = 0; j < layout.Nodes(d); ++j)
				{
					GpuLightRefitEntry entry = { levelOffsets[b][d] + j, 0, 0 };
					if (height == 0)
					{
						entry.first = desc.firstLight + j * leafSize;
						entry.count = j < layout.leafCount ? std::min(leafSize, desc.lightCount - j * leafSize) : 0;
					}
					else
					{
						entry.first = levelOffsets[b][d + 1] + j * LIGHT_TREE_WIDTH;
						entry.count = j < layout.Width(d) ? LIGHT_TREE_WIDTH : 0;
					}
					entries.push_back(entry);
				}
			}
			if (dynamic)
			{
				range.dynamicCount = static_cast<uint32_t>(entries.size()) - range.first;
			}
		}
		range.totalCount = static_cast<uint32_t>(entries.size()) - range.first;
		heights_.push_back(range);
	}

	auto upload = [&](const void *data, uint64_t elements, uint32_t stride, const char *debugName, UploadBatchHandle batch)
	{
		BufferDesc desc = {};
		desc.sizeInBytes = elements * stride;
		desc.structureByteStride = stride;
		desc.debugName = debugName;
		return renderContext_->CreateBuffer(desc, ResourceViewType::ShaderResource, data, batch);
	};
	const UploadBatchHandle batch = renderContext_->BeginUpload();
	blases_ = upload(blases.data(), blasCount, sizeof(GpuLightBlas), "LightTree_Blases", batch);
	refitEntries_ = upload(entries.data(), entries.size(), sizeof(GpuLightRefitEntry), "LightTree_RefitEntries", batch);
	renderContext_->EndUpload(batch);

	layoutValid_ = true;
	topologyPending_ = true;
	logi("Light tree: {} BLAS(es) over {} light(s), {} light(s) per leaf, {} node(s), deepest BLAS {} level(s)",
		 blasCount,
		 lightCount_,
		 leafSize,
		 nodeOffset,
		 maxLeafLevel + 1);
}

void LightTreeBuilder::AddPasses(FrameGraph &frame, const Inputs &inputs)
{
	outputs_ = {};
	if (resources_.empty())
	{
		return;
	}
	outputs_ = { frame.ImportBuffer(outputBuffers_[kNodes], "LightTree_Nodes"),
				 frame.ImportBuffer(outputBuffers_[kLeafLights], "LightTree_LeafLights"),
				 frame.ImportBuffer(outputBuffers_[kLeafOfLight], "LightTree_LeafOfLight"),
				 frame.ImportBuffer(outputBuffers_[kPower], "LightTree_Power") };
	if (lightCount_ == 0 || !layoutValid_)
	{
		return;
	}

	const bool twoLevel = IsTwoLevel();
	const uint32_t leafSize = leafSize_;
	const uint32_t lightCount = lightCount_;
	const uint32_t blasCount = static_cast<uint32_t>(blasDescs_.size());
	const uint32_t blockCount = GroupCount(lightCount_, LIGHT_TREE_BUILD_BLOCK_SIZE);
	// The top tree: the single one over the lights, or the TLAS over the BLASes.
	const TreeLayout layout = twoLevel ? TreeLayout(blasCount, 1) : TreeLayout(lightCount, leafSize);
	depth_ = layout.leafLevel + 1;

	LightTreeBuildConstants base = {};
	const float extent = std::max({ inputs.boundsMax.x - inputs.boundsMin.x,
									inputs.boundsMax.y - inputs.boundsMin.y,
									inputs.boundsMax.z - inputs.boundsMin.z,
									0.0f });
	base.boundsScale = extent > 0.0f ? 1024.0f / extent : 0.0f;
	// The node means are 16-bit steps over the same cube (PackLightTreeNode). The bounds are only finite with lights.
	origin_ = isfinite(extent) ? inputs.boundsMin : float3(0.0f, 0.0f, 0.0f);
	step_ = isfinite(extent) && extent > 0.0f ? extent / 65535.0f : 0.0f;
	base.boundsMin = origin_;
	base.quantStep = step_;
	base.lightCount = lightCount;
	base.leafSize = leafSize;
	base.lightsHandle = frame.DescriptorIndex(inputs.lights);
	base.emissionScalesHandle = frame.DescriptorIndex(inputs.emissionScales);

	// The scratch of one build.
	auto transient = [&](uint32_t elements, uint32_t stride, const char *debugName)
	{
		BufferDesc desc = {};
		desc.sizeInBytes = static_cast<uint64_t>(elements) * stride;
		desc.structureByteStride = stride;
		desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		desc.debugName = debugName;
		return frame.graph.CreateBuffer(desc, ResourceViewType::UnorderedAccess);
	};
	const rg::ResourceId keysA = transient(lightElements_, sizeof(uint32_t), "LightTree_KeysA");
	const rg::ResourceId keysB = transient(lightElements_, sizeof(uint32_t), "LightTree_KeysB");
	const rg::ResourceId valuesA = transient(lightElements_, sizeof(uint32_t), "LightTree_ValuesA");
	const rg::ResourceId valuesB = transient(lightElements_, sizeof(uint32_t), "LightTree_ValuesB");
	const rg::ResourceId radixCounts =
			transient((1u << LIGHT_TREE_RADIX_BITS) * blockElements_, sizeof(uint32_t), "LightTree_RadixCounts");
	const rg::ResourceId blockSums = transient(blockElements_, sizeof(uint32_t), "LightTree_BlockSums");

	const rg::ResourceId lights = frame.ImportBuffer(inputs.lights, "Scene_Lights");
	const rg::ResourceId lightClusters = frame.ImportBuffer(lightClusters_, "LightTree_LightClusters");
	const rg::ResourceId nodeClusters = frame.ImportBuffer(nodeClusters_, "LightTree_NodeClusters");
	const rg::ResourceId blasClusters = frame.ImportBuffer(blasClusters_, "LightTree_BlasClusters");
	const rg::ResourceId blasBounds = frame.ImportBuffer(blasBounds_, "LightTree_BlasBounds");
	const rg::ResourceId nodes = outputs_[kNodes];
	const rg::ResourceId leafLights = outputs_[kLeafLights];
	const rg::ResourceId leafOfLight = outputs_[kLeafOfLight];
	const rg::ResourceId power = outputs_[kPower];

	// The descriptors, which for the transients are known once the graph has placed them, so every pass writes them
	// (the same each time) before its dispatches.
	const ResourceHandle resourceBuffer = resources_[frame.rc.GetCurrentFrameIndex()];
	const rg::RenderGraph &graph = frame.graph;
	const std::array<rg::ResourceId, 6> scratchIds = { keysA, valuesA, keysB, valuesB, radixCounts, blockSums };
	const std::array<ResourceHandle, 13> persistent = { fluxScales_,
														lightEmitters_,
														lightClusters_,
														nodeClusters_,
														outputBuffers_[kNodes],
														outputBuffers_[kLeafLights],
														outputBuffers_[kLeafOfLight],
														outputBuffers_[kPower],
														blasClusters_,
														blasBounds_,
														GetBlasBuffer(),
														lightBlas_,
														refitEntries_ != InvalidResourceHandle ? refitEntries_ : lightBlas_ };
	auto writeResources = [&graph, resourceBuffer, scratchIds, persistent](rg::PassContext &ctx)
	{
		auto index = [&](ResourceHandle handle)
		{ return static_cast<uint>(ctx.rc.GetResourceViewInfo(handle).descriptorIndex); };
		auto scratchIndex = [&](size_t i) { return index(graph.GetHandle(scratchIds[i])); };
		LightTreeBuildResources resources = {};
		resources.fluxScales = index(persistent[0]);
		resources.lightEmitters = index(persistent[1]);
		resources.lightClusters = index(persistent[2]);
		resources.nodeClusters = index(persistent[3]);
		resources.nodes = index(persistent[4]);
		resources.leafLights = index(persistent[5]);
		resources.leafOfLight = index(persistent[6]);
		resources.power = index(persistent[7]);
		resources.blasClusters = index(persistent[8]);
		resources.blasBounds = index(persistent[9]);
		resources.blases = index(persistent[10]);
		resources.lightBlas = index(persistent[11]);
		resources.refitEntries = index(persistent[12]);
		resources.keysA = scratchIndex(0);
		resources.valuesA = scratchIndex(1);
		resources.keysB = scratchIndex(2);
		resources.valuesB = scratchIndex(3);
		resources.radixCounts = scratchIndex(4);
		resources.blockSums = scratchIndex(5);
		ctx.rc.UpdateConstantBuffer(resourceBuffer, &resources, sizeof(resources));
	};

	// Every kernel reads and writes through UAVs. Between the kernels of one pass, which read what the one before
	// wrote, global UAV barriers order them, leaving every buffer in the state the pass declared.
	struct Dispatch
	{
		PipelineHandle pipeline;
		uint32_t groups;
		LightTreeBuildConstants constants;
	};
	using Dispatches = std::vector<Dispatch>;
	auto addPass = [&](const char *name, std::initializer_list<rg::ResourceId> uses, Dispatches dispatches)
	{
		rg::Pass pass = rg::Pass::Compute(name);
		for (rg::ResourceId id: uses)
		{
			FrameGraph::Use(pass, id, rg::Access::ComputeShaderReadWrite);
		}
		pass.Executes(
				[writeResources, resourceBuffer, dispatches = std::move(dispatches)](rg::PassContext &ctx)
				{
					writeResources(ctx);
					for (size_t i = 0; i < dispatches.size(); ++i)
					{
						if (i > 0)
						{
							ctx.rc.GlobalBarrier(ResourceSync::ComputeShading,
												 ResourceAccess::UnorderedAccess,
												 ResourceSync::ComputeShading,
												 ResourceAccess::UnorderedAccess);
						}
						const Dispatch &dispatch = dispatches[i];
						ctx.rc.SetComputePipeline(dispatch.pipeline);
						ctx.rc.SetCompute32BitConstants(&dispatch.constants,
														sizeof(dispatch.constants) / sizeof(uint32_t));
						ctx.rc.SetComputeConstantBuffer(1, resourceBuffer);
						ctx.rc.Dispatch(dispatch.groups, 1, 1);
					}
				});
		frame.AddPass(std::move(pass));
	};
	auto dispatch = [](PipelineHandle pipeline, uint32_t groups, LightTreeBuildConstants constants)
	{ return Dispatch{ pipeline, groups, constants }; };
	// The stable radix sort of count (key, value) pairs in the A buffers, back into them.
	auto appendSort = [&](Dispatches &list, uint32_t count, LightTreeBuildConstants c)
	{
		c.lightCount = count;
		const uint32_t blocks = GroupCount(count, LIGHT_TREE_BUILD_BLOCK_SIZE);
		for (uint32_t pass = 0; pass < LIGHT_TREE_RADIX_PASSES; ++pass)
		{
			c.pass = pass;
			list.push_back(dispatch(radixCountPipeline_, blocks, c));
			list.push_back(dispatch(radixScanPipeline_, 1, c));
			list.push_back(dispatch(radixScatterPipeline_, blocks, c));
		}
	};
	// The tree over the sorted pairs in the A buffers, bottom up: the leaves, the wide interior levels one dispatch
	// each, then one group for the rest. c says whose tree (lightCount, leafSize, the TLAS flag).
	auto appendTree = [&](Dispatches &list, const TreeLayout &tree, LightTreeBuildConstants c)
	{
		c.pass = 0;
		list.push_back(dispatch(leavesPipeline_, GroupCount(tree.Nodes(tree.leafLevel), LIGHT_TREE_BUILD_GROUP_SIZE), c));
		int level = static_cast<int>(tree.leafLevel) - 1;
		for (; level >= 0 && tree.Nodes(level) > LIGHT_TREE_BUILD_TOP_NODES; --level)
		{
			c.pass = static_cast<uint32_t>(level);
			list.push_back(dispatch(levelPipeline_, GroupCount(tree.Nodes(level), LIGHT_TREE_BUILD_GROUP_SIZE), c));
		}
		if (level >= 0)
		{
			c.pass = static_cast<uint32_t>(level);
			list.push_back(dispatch(topLevelsPipeline_, 1, c));
		}
	};

	if (!twoLevel)
	{
		addPass("LightTree/Keys",
				{ lights, lightClusters, keysA, valuesA },
				{ dispatch(keysPipeline_, GroupCount(lightCount, LIGHT_TREE_BUILD_GROUP_SIZE), base) });
		Dispatches sort;
		appendSort(sort, lightCount, base);
		addPass("LightTree/Sort", { keysA, valuesA, keysB, valuesB, radixCounts }, std::move(sort));
		Dispatches build;
		appendTree(build, layout, base);
		addPass("LightTree/Nodes",
				{ keysA, valuesA, lightClusters, nodeClusters, nodes, leafLights, leafOfLight },
				std::move(build));
	}
	else
	{
		const bool refitAll = topologyPending_ || inputs.refitStatic;
		if (topologyPending_)
		{
			// Each BLAS's lights along the Morton curve over its own bounds, in its own slots.
			const uint32_t blasBits =
					blasCount > 1 ? static_cast<uint32_t>(std::bit_width(blasCount - 1)) : 0u;
			LightTreeBuildConstants c = base;
			c.first = blasCount;
			c.mortonBits = std::min(30u, 32u - blasBits);
			Dispatches topology = { dispatch(clearBoundsPipeline_, GroupCount(blasCount, LIGHT_TREE_BUILD_GROUP_SIZE), c),
									dispatch(boundsPipeline_, GroupCount(lightCount, LIGHT_TREE_BUILD_GROUP_SIZE), c),
									dispatch(topologyKeysPipeline_, GroupCount(lightCount, LIGHT_TREE_BUILD_GROUP_SIZE), c) };
			appendSort(topology, lightCount, c);
			topology.push_back(dispatch(topologySlotsPipeline_, GroupCount(lightCount, LIGHT_TREE_BUILD_GROUP_SIZE), c));
			addPass("LightTree/Topology",
					{ lights, blasBounds, keysA, valuesA, keysB, valuesB, radixCounts, leafLights, leafOfLight },
					std::move(topology));
			topologyPending_ = false;
		}

		// Bottom up, every BLAS refit at once, a height at a time (Figure 4 of the paper).
		Dispatches refit;
		for (size_t height = 0; height < heights_.size(); ++height)
		{
			const HeightRange &range = heights_[height];
			LightTreeBuildConstants c = base;
			c.first = range.first;
			c.count = refitAll ? range.totalCount : range.dynamicCount;
			if (c.count == 0)
			{
				continue;
			}
			refit.push_back(dispatch(height == 0 ? refitLeavesPipeline_ : refitInteriorPipeline_,
									 GroupCount(c.count, LIGHT_TREE_BUILD_GROUP_SIZE),
									 c));
		}
		if (!refit.empty())
		{
			addPass("LightTree/Refit",
					{ lights, lightClusters, nodeClusters, nodes, leafLights, leafOfLight },
					std::move(refit));
		}

		// The TLAS from scratch over the BLASes' clusters, its entries after the lights' in the leaf buffers.
		LightTreeBuildConstants c = base;
		c.lightCount = blasCount;
		c.leafSize = 1;
		c.flags = LIGHT_TREE_BUILD_TLAS;
		c.entryOffset = lightCount;
		Dispatches tlas = { dispatch(blasClustersPipeline_, GroupCount(blasCount, LIGHT_TREE_BUILD_GROUP_SIZE), c) };
		appendSort(tlas, blasCount, c);
		appendTree(tlas, layout, c);
		addPass("LightTree/TLAS",
				{ nodeClusters, blasClusters, keysA, valuesA, keysB, valuesB, radixCounts, nodes, leafLights, leafOfLight },
				std::move(tlas));
	}

	// Power selection over every light, from their clusters and the top tree's root group's total.
	const Dispatches powerList = {
		dispatch(powerReducePipeline_, blockCount, base),
		dispatch(powerScanPipeline_, 1, base),
		dispatch(powerApplyPipeline_, blockCount, base),
		dispatch(powerGuidePipeline_, GroupCount(std::bit_floor(lightCount), LIGHT_TREE_BUILD_GROUP_SIZE), base),
	};
	addPass("LightTree/Power", { lightClusters, nodeClusters, keysB, blockSums, power }, powerList);
}
