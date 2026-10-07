
cmake_minimum_required(VERSION 3.28)

project("PathTracer")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

include(cmake/CPM.cmake)

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

add_subdirectory("external/D3D12RenderContext")
add_subdirectory("external/GLTFScene")

add_subdirectory("src")
