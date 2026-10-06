find_package(Vulkan REQUIRED GLOBAL)
find_package(Threads REQUIRED GLOBAL)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static third-party libraries" FORCE)

#TODO: Replace with submodules

add_library(entt INTERFACE)
target_include_directories(entt INTERFACE "entt/include")

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE "nlohmann/include")

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "stb/include")

add_subdirectory(vma)

set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
foreach(importer FBX GLTF OBJ)
    set(ASSIMP_BUILD_${importer}_IMPORTER ON CACHE BOOL "" FORCE)
endforeach()
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
add_subdirectory(assimp EXCLUDE_FROM_ALL)
target_compile_options(assimp PRIVATE /EHsc)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
set(USE_MSVC_RUNTIME_LIBRARY_DLL ON CACHE BOOL "" FORCE)
add_subdirectory(glfw EXCLUDE_FROM_ALL)

set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
set(GLM_ENABLE_CXX_20 ON CACHE BOOL "" FORCE)
add_subdirectory(glm EXCLUDE_FROM_ALL)

set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(spdlog EXCLUDE_FROM_ALL)


set(TRACY_STATIC ON CACHE BOOL "" FORCE)
set(TRACY_ENABLE OFF CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND OFF CACHE BOOL "" FORCE)
set(TRACY_FIBERS OFF CACHE BOOL "" FORCE)
set(TRACY_LTO OFF CACHE BOOL "" FORCE)
add_subdirectory(tracy EXCLUDE_FROM_ALL)
target_compile_definitions(TracyClient
    PUBLIC "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>,$<NOT:$<BOOL:${LUTH_ENABLE_ASAN}>>>:TRACY_ENABLE;TRACY_ON_DEMAND>"
    PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>,$<NOT:$<BOOL:${LUTH_ENABLE_ASAN}>>>:TRACY_FIBERS>"
)

set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SHARED OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_STATIC ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_GLSL ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
add_subdirectory(spirv-cross EXCLUDE_FROM_ALL)

set(USE_ASSERTS OFF CACHE BOOL "" FORCE)
set(OVERRIDE_CXX_FLAGS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(FLOATING_POINT_EXCEPTIONS_ENABLED OFF CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(PROFILER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(ENABLE_OBJECT_STREAM OFF CACHE BOOL "" FORCE)
set(ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
set(JPH_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
set(USE_SSE4_1 ON CACHE BOOL "" FORCE)
set(USE_SSE4_2 ON CACHE BOOL "" FORCE)
set(USE_AVX ON CACHE BOOL "" FORCE)
set(USE_AVX2 ON CACHE BOOL "" FORCE)
set(USE_LZCNT ON CACHE BOOL "" FORCE)
set(USE_TZCNT ON CACHE BOOL "" FORCE)
set(USE_F16C ON CACHE BOOL "" FORCE)
set(USE_FMADD ON CACHE BOOL "" FORCE)
add_subdirectory(jolt/Build EXCLUDE_FROM_ALL)
target_compile_definitions(Jolt PUBLIC
    "$<$<CONFIG:Debug>:JPH_ENABLE_ASSERTS;JPH_DEBUG_RENDERER>"
    "$<$<CONFIG:Release>:JPH_DEBUG_RENDERER;NDEBUG>"
    "$<$<CONFIG:Dist>:NDEBUG>"
)
# Jolt publishes its C++ compiler flags. Guard them so they are not forwarded to
# the MASM source in Luth when the target is linked transitively.
get_target_property(jolt_interface_options Jolt INTERFACE_COMPILE_OPTIONS)
if(jolt_interface_options)
    set_property(TARGET Jolt PROPERTY INTERFACE_COMPILE_OPTIONS "")
    foreach(option IN LISTS jolt_interface_options)
        target_compile_options(Jolt INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${option}>")
    endforeach()
endif()

add_library(imgui STATIC
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_vulkan.cpp
)
add_library(ImGui::ImGui ALIAS imgui)
target_compile_features(imgui PUBLIC cxx_std_17)
target_include_directories(imgui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

add_library(imguizmo STATIC
    imguizmo/ImGuizmo.cpp
    imguizmo/GraphEditor.cpp
)
add_library(ImGuizmo::ImGuizmo ALIAS imguizmo)
target_compile_features(imguizmo PUBLIC cxx_std_20)
target_compile_definitions(imguizmo PRIVATE IMGUI_DEFINE_MATH_OPERATORS)
target_include_directories(imguizmo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/imguizmo")
target_link_libraries(imguizmo PUBLIC ImGui::ImGui)

add_library(bc7enc STATIC bc7enc/bc7enc.cpp bc7enc/rgbcx.cpp)
add_library(bc7enc::bc7enc ALIAS bc7enc)
target_compile_features(bc7enc PUBLIC cxx_std_20)
target_include_directories(bc7enc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/bc7enc")


add_library(slang_compiler SHARED IMPORTED GLOBAL)
set_target_properties(slang_compiler PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/slang/include"
)
set(slang_dir "${CMAKE_CURRENT_SOURCE_DIR}/slang/lib/")
set_target_properties(slang_compiler PROPERTIES
    IMPORTED_LOCATION "${slang_dir}/slang-compiler.dll"
    IMPORTED_IMPLIB "${slang_dir}/slang-compiler.lib"
)
set(LUTH_SLANG_RUNTIME_FILES
    "${slang_dir}/slang-compiler.dll"
    "${slang_dir}/slang-glslang.dll"
    "${slang_dir}/slang-glsl-module.dll"
    "${slang_dir}/slang-rt.dll"
    CACHE INTERNAL "Pinned Slang runtime files"
)


if(LUTH_BUILD_TESTS)
    add_library(doctest INTERFACE)
    target_include_directories(doctest INTERFACE "doctest/include")
endif()





