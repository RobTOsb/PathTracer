#pragma once

class RenderContext;

// A unit of GPU work recorded once per frame. Passes are stored in order and executed one after
// another on the frame's graphics command list; each pass leaves the resources it writes in
// whatever state the next pass transitions them out of.
class RenderPass
{
public:
	virtual ~RenderPass() = default;

	virtual void Execute(RenderContext *renderContext) = 0;
};
