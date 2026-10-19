// Splitting camera paths at glass: one lobe now, the other deferred, plus the panes' reflections.
import OpenPBRBSDF;
import PathTracer.Resources;

struct PaneReflection
{
    float3 albedo; // the split panes' directional reflectance, through the panes in front of them
    float luminance; // the brightest pane's share of albedo, 0 until a pane splits
    float distance; // camera to that pane
    float curvature; // its curvature as the camera sees it
    float hitDistance; // length of its reflection ray, SKY_HIT_DISTANCE for the sky
};

bool AddPaneReflection(inout PaneReflection panes, float3 albedo, float distance, float curvature)
{
    panes.albedo += albedo;
    float luminance = Luminance(albedo);
    if (luminance <= panes.luminance)
        return false;
    panes.luminance = luminance;
    panes.distance = distance;
    panes.curvature = curvature;
    panes.hitDistance = 0.0;
    return true;
}

// The deferred branch's flags, which share TracePath's flag word: the low bits and the top byte.
static const uint DEFERRED_PENDING = 1;
static const uint DEFERRED_REFLECTION = 2; // solid glass's reflection, otherwise what is behind a pane
static const uint DEFERRED_BRIGHTEST_PANE = 4; // the reflection's first hit gives the brightest pane's hit distance
static const uint DEFERRED_SMOOTH_SEGMENT = 8; // set while the resumed reflection is on its first segment
static const uint DEFERRED_BOUNCE_SHIFT = 24; // the bounce the branch resumes at
static const uint DEFERRED_MASK = 0xFF00000Fu;

// The rest of the branch waits in g_DeferredBranches, since only paths through glass have one.
struct DeferredBranch
{
    float3 origin;
    float3 throughput;
    float3 dir;
    float coneWidth;
    float coneSpread;
};

void StoreDeferredBranch(DeferredBranch b)
{
    GpuDeferredBranch p;
    p.origin = b.origin;
    p.coneWidth = b.coneWidth;
    p.throughput = b.throughput;
    p.coneSpread = b.coneSpread;
    p.direction = b.dir;
    g_DeferredBranches[PixelIndex()] = p;
}

DeferredBranch LoadDeferredBranch()
{
    GpuDeferredBranch p = g_DeferredBranches[PixelIndex()];
    DeferredBranch b;
    b.origin = p.origin;
    b.throughput = p.throughput;
    b.dir = p.direction;
    b.coneWidth = p.coneWidth;
    b.coneSpread = p.coneSpread;
    return b;
}
