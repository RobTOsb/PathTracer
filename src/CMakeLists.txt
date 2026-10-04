add_subdirectory(Common)

add_executable(PathTracer
    main.cpp
    Scene/RenderGPUScene.cpp
    Renderer/PathTracerPass.cpp
    Renderer/TonemapPass.cpp
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
)

target_link_libraries(PathTracer PRIVATE Common D3D12RenderContext GLTFScene)

d3d12_deploy_runtime_dlls(PathTracer)
