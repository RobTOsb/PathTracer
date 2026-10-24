add_subdirectory(Common)

add_executable(PathTracer
    main.cpp
    DLSSCommon.cpp
    Scene/RenderGPUScene.cpp
    Scene/SceneCache.cpp
    Scene/LightTreeBuilder.cpp
    Renderer/PathTracerPass.cpp
    Renderer/SkyAtmospherePass.cpp
    Renderer/DLSSPass.cpp
    Renderer/GuideViewPass.cpp
    Renderer/AutoExposurePass.cpp
    Renderer/BloomPass.cpp
    Renderer/TonemapPass.cpp
    Renderer/ImGuiPass.cpp
)

# NVIDIA NGX SDK (DLSS). The import libraries are built against the dynamic CRT (/MD), with a
# separate one for the debug CRT.
set(DLSS_SDK_DIR "${CMAKE_SOURCE_DIR}/external/DLSS")
set(DLSS_LIB_DIR "${DLSS_SDK_DIR}/lib/Windows_x86_64")

target_include_directories(PathTracer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/shaders
    ${DLSS_SDK_DIR}/include
)

# Shaders are compiled at runtime by Slang straight from the source tree, so edits only need a restart.
target_compile_definitions(PathTracer PRIVATE
    NOMINMAX
    PATHTRACER_SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/"
)

target_link_libraries(PathTracer PRIVATE
    Common
    D3D12RenderContext
    GLTFScene
    meshoptimizer
    $<IF:$<CONFIG:Debug>,${DLSS_LIB_DIR}/x64/nvsdk_ngx_d_dbg.lib,${DLSS_LIB_DIR}/x64/nvsdk_ngx_d.lib>
)

d3d12_deploy_runtime_dlls(PathTracer)

# NGX loads the DLSS feature DLLs from the executable's directory. The dev builds draw the
# on-screen indicator and enable the debug hotkeys (jitter / motion vector checks).
add_custom_command(TARGET PathTracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLSS_LIB_DIR}/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlss.dll"
        "${DLSS_LIB_DIR}/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlssd.dll"
        "$<TARGET_FILE_DIR:PathTracer>"
    COMMENT "Deploying DLSS runtime DLLs next to PathTracer")
