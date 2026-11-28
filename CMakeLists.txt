
cmake_minimum_required(VERSION 3.28)

project("PathTracer")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

include(cmake/CPM.cmake)
include(cmake/streamline.cmake)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-march=x86-64-v3)
    endif()
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    add_custom_target(copy_compile_commands ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_SOURCE_DIR}/build/compile_commands.json
        DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
    )
endif()

# The render graph brings its own RenderContext. Only its library is added: its project root also defines
# copy_compile_commands, tests and a sandbox, which stay out of this build.
add_subdirectory("external/RenderGraph/external/D3D12RenderContext")
add_subdirectory("external/GLTFScene")
add_subdirectory("external/RenderGraph/src" EXCLUDE_FROM_ALL)

add_subdirectory("src")
