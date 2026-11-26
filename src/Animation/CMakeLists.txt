add_library(Animation STATIC
    AnimationSystem.cpp
    Skinning.cpp
)

target_include_directories(Animation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Animation PUBLIC GLTFScene)
target_compile_definitions(Animation PRIVATE NOMINMAX)
