#pragma once

#include "RenderContext.h"
#include "RenderGraph.h"

#include <functional>

class FrameGraph;

// The HDR image a pass reads, asked for every frame: a graph transient of the frame (DLSS-RR's output or the path
// tracer's), recorded before the pass.
using HdrSource = std::function<rg::ResourceId()>;

// A unit of GPU work, declared once per frame into the render graph. AddPasses runs while the frame is recorded:
// it does the frame's CPU work (constants, which dispatches are needed) and adds graph passes, each naming the
// resources it uses and how, with a callback that the graph runs later in the frame to record the commands. The
// graph places the barriers between the passes, so a callback leaves every resource it uses in the state its pass
// declared.
class RenderPass
{
public:
	virtual ~RenderPass() = default;

	virtual void AddPasses(FrameGraph &frame) = 0;
};
