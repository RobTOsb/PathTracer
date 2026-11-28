add_subdirectory(Common)
add_subdirectory(Animation)

add_executable(PathTracer
    main.cpp
    Streamline.cpp
    Scene/RenderGPUScene.cpp
    Scene/SceneCache.cpp
    Scene/LightTreeBuilder.cpp
    Renderer/PathTracerPass.cpp
    Renderer/SkyAtmospherePass.cpp
    Renderer/DLSSPass.cpp
    Renderer/FrameGenerationPass.cpp
    Renderer/GuideViewPass.cpp
    Renderer/AutoExposurePass.cpp
    Renderer/BloomPass.cpp
    Renderer/LensFlarePass.cpp
    Renderer/TonemapPass.cpp
    Renderer/FilmLut.cpp
    Renderer/ImGuiPass.cpp
)

target_include_directories(PathTracer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/shaders
)

# Shaders are compiled at runtime by Slang straight from the source tree, so edits only need a restart.
target_compile_definitions(PathTracer PRIVATE
    NOMINMAX
    PATHTRACER_SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/"
    PATHTRACER_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets/"
)

# Streamline first, so its interposer resolves the D3D12 and DXGI entry points ahead of d3d12.lib and dxgi.lib, which the
# D3D12 libraries bring in further down the link line (cmake/streamline.cmake).
target_link_libraries(PathTracer PRIVATE
    streamline
    Common
    Animation
    RenderGraph
    GLTFScene
    meshoptimizer
)

d3d12_deploy_runtime_dlls(PathTracer)

streamline_deploy_runtime_dlls(PathTracer)
