#pragma once

#include "RenderGraph.h"

#include <string>
#include <unordered_map>

inline constexpr const char *kGpuFrameScope = "GPU frame";

class FrameGraph
{
public:
	FrameGraph(rg::RenderGraph &graph, RenderContext &renderContext) : graph(graph), rc(renderContext) {}

	FrameGraph(const FrameGraph &) = delete;
	FrameGraph &operator=(const FrameGraph &) = delete;

	rg::RenderGraph &graph;
	RenderContext &rc;

	// The swapchain image this frame presents.
	rg::ResourceId backBuffer;

	// The queue AddPass puts passes on: the main loop sets it to the compute queue for the sky and clouds when async
	// compute is on, then back to graphics for the rest, with a Submit() after the TLAS and light tree.
	QueueType queue = QueueType::Graphics;

	rg::PassId AddPass(rg::Pass pass)
	{
		pass.queue = queue;
		return graph.AddPass(std::move(pass));
	}

	rg::ResourceId ImportTexture(ResourceHandle handle, const char *name)
	{
		return Import(handle, name, true);
	}

	rg::ResourceId ImportBuffer(ResourceHandle handle, const char *name)
	{
		return Import(handle, name, false);
	}

	// Attaches resource to pass once. Two uses in one pass (a guide shown as itself and read for its motion, say)
	// become one: the write if either writes, since the graph refuses a resource attached twice.
	static void Use(rg::Pass &pass, rg::ResourceId resource, rg::Access access)
	{
		for (rg::Attachment &attachment: pass.attachments)
		{
			if (attachment.resource == resource)
			{
				if (rg::IsWriteAccess(access) && !rg::IsWriteAccess(attachment.access))
				{
					attachment.access = access;
				}
				return;
			}
		}
		pass.Use(resource, access);
	}

	// Imports and attaches a persistent resource in one go.
	void UseTexture(rg::Pass &pass, ResourceHandle handle, const char *name, rg::Access access)
	{
		Use(pass, ImportTexture(handle, name), access);
	}

	void UseBuffer(rg::Pass &pass, ResourceHandle handle, const char *name, rg::Access access)
	{
		Use(pass, ImportBuffer(handle, name), access);
	}

	// The bindless index of a resource's own view, for constants that name resources by descriptor.
	uint32_t DescriptorIndex(ResourceHandle handle) const
	{
		return rc.GetResourceViewInfo(handle).descriptorIndex;
	}

private:
	rg::ResourceId Import(ResourceHandle handle, const char *name, bool texture)
	{
		auto it = imports_.find(handle);
		if (it != imports_.end())
		{
			return it->second;
		}
		const rg::ResourceId id = texture ? graph.ImportTexture(handle, name) : graph.ImportBuffer(handle, name);
		imports_.emplace(handle, id);
		return id;
	}

	std::unordered_map<ResourceHandle, rg::ResourceId> imports_;
};
