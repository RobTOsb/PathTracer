// Lays the UI, drawn on its own over nothing, onto the scene: the blend ImGui would have made drawing straight onto
// it, as its colour comes premultiplied by its alpha. See FrameGenerationPass.

struct UICompositeConstants
{
    uint width;
    uint height;
};

ConstantBuffer<UICompositeConstants> g_PushConstants : register(b0);
Texture2D<float4> g_UI : register(t0);
RWTexture2D<float4> g_Scene : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= g_PushConstants.width || id.y >= g_PushConstants.height)
    {
        return;
    }
    const float4 ui = g_UI[id.xy];
    if (ui.a <= 0.0 && all(ui.rgb <= 0.0))
    {
        return;
    }
    const float4 scene = g_Scene[id.xy];
    g_Scene[id.xy] = float4(ui.rgb + (1.0 - ui.a) * scene.rgb, scene.a);
}
