# NVIDIA Streamline SDK, which runs DLSS Ray Reconstruction, DLSS Frame Generation and Reflex. The prebuilt release
# package is downloaded at configure time and checked against its SHA-256; it holds the headers, sl.interposer.lib
# and the NVIDIA-signed DLLs.
#
# Streamline is integrated the standard way, without manual hooking: sl.interposer.lib is linked ahead of d3d12.lib and
# dxgi.lib, so D3D12CreateDevice and CreateDXGIFactory* resolve to sl.interposer.dll and the device, factory, queues,
# command lists and swap chain the render context makes are Streamline's proxies.

include(FetchContent)

set(STREAMLINE_VERSION "2.14.16")
message(STATUS "NVIDIA Streamline SDK: ${STREAMLINE_VERSION}")

FetchContent_Declare(
    streamline_sdk
    URL "https://github.com/NVIDIA-RTX/Streamline/releases/download/v${STREAMLINE_VERSION}/streamline-sdk-v${STREAMLINE_VERSION}.zip"
    URL_HASH SHA256=32041e51db0cab709a271b72a352281b690c2abbd669a827809401a5dd23956b
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # The package's own CMakeLists.txt copies its DLLs into CMAKE_RUNTIME_OUTPUT_DIRECTORY, which this project leaves
    # unset, and takes one DLL folder for every configuration. Pointing at a folder without a CMakeLists.txt keeps it
    # out of the build; the target and the DLL copy are below instead.
    SOURCE_SUBDIR "no-cmake"
)
FetchContent_MakeAvailable(streamline_sdk)

set(STREAMLINE_SDK_DIR "${streamline_sdk_SOURCE_DIR}" CACHE INTERNAL "Streamline SDK root")

# Headers and the interposer's import library. Link it before anything that brings in d3d12.lib or dxgi.lib: the first
# library on the link line that defines a symbol is the one used.
add_library(streamline INTERFACE)
target_include_directories(streamline INTERFACE "${STREAMLINE_SDK_DIR}/include")
target_link_libraries(streamline INTERFACE "${STREAMLINE_SDK_DIR}/lib/x64/sl.interposer.lib")

# Copies the DLLs Streamline needs next to the target: the interposer, sl.common, sl.pcl (which the interposer always
# loads), the DLSS-RR and DLSS-G plugins with their NGX snippets, and Reflex. Debug builds take the development set, which is unsigned and draws
# the on-screen DLSS indicator; the others take the signed production set, whose interposer only loads plugins that
# carry NVIDIA's signature. The two sets are never mixed.
function(streamline_deploy_runtime_dlls target)
    set(_dir "$<IF:$<CONFIG:Debug>,${STREAMLINE_SDK_DIR}/bin/x64/development,${STREAMLINE_SDK_DIR}/bin/x64>")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_dir}/sl.interposer.dll"
            "${_dir}/sl.common.dll"
            "${_dir}/sl.pcl.dll"
            "${_dir}/sl.dlss_d.dll"
            "${_dir}/nvngx_dlssd.dll"
            "${_dir}/sl.dlss_g.dll"
            "${_dir}/nvngx_dlssg.dll"
            "${_dir}/sl.reflex.dll"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Deploying Streamline (DLSS-RR, DLSS-G, Reflex) runtime DLLs next to ${target}")
endfunction()
