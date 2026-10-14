// Splitting camera paths at glass: one lobe now, the other deferred, plus the panes' reflections.
import OpenPBRBaseSubstrate;

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

static const uint DEFERRED_PENDING = 1;
static const uint DEFERRED_REFLECTION = 2; // solid glass's reflection, otherwise what is behind a pane
static const uint DEFERRED_BRIGHTEST_PANE = 4; // the reflection's first hit gives the brightest pane's hit distance
static const uint DEFERRED_SMOOTH_SEGMENT = 8; // set while the resumed reflection is on its first segment
static const uint DEFERRED_BOUNCE_SHIFT = 8; // the bounce the branch resumes at

struct DeferredBranch
{
    uint flags;
    float3 origin;
    float3 throughput;
    float3 dirOrCameraTransmittance; // the reflection's direction (behind a pane is along the camera ray)
    float coneWidth;
    float coneSpreadOrCameraDistance;
};
