cmake_minimum_required(VERSION 3.30)
project(Luth VERSION 3.12)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(find_asm)
include(LuthBuild)

option(LUTH_BUILD_TESTS "Build the Luth test suite" OFF)
option(LUTH_BUILD_SAMPLES "Build Luth sample and proof targets" OFF)
option(LUTH_ENABLE_ASAN "Enable AddressSanitizer for first-party targets" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Dist" CACHE STRING "Available build configurations" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build configuration" FORCE)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release Dist)
endif()

if(MSVC)
    if(LUTH_ENABLE_ASAN)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
        add_compile_definitions(
            _DISABLE_VECTOR_ANNOTATION
            _DISABLE_STRING_ANNOTATION
            _DISABLE_OPTIONAL_ANNOTATION
        )
        add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/U_DEBUG>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

if(MSVC)
    set(CMAKE_C_FLAGS_DIST "${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "C flags for Dist" FORCE)
    set(CMAKE_CXX_FLAGS_DIST "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "C++ flags for Dist" FORCE)
else()
    set(CMAKE_C_FLAGS_DIST "-O3 -DNDEBUG" CACHE STRING "C flags for Dist" FORCE)
    set(CMAKE_CXX_FLAGS_DIST "-O3 -DNDEBUG" CACHE STRING "C++ flags for Dist" FORCE)
endif()
set(CMAKE_EXE_LINKER_FLAGS_DIST "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING "Executable linker flags for Dist" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DIST "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING "Shared linker flags for Dist" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_DIST "${CMAKE_STATIC_LINKER_FLAGS_RELEASE}" CACHE STRING "Static linker flags for Dist" FORCE)


add_subdirectory(third_party)
add_subdirectory(engine)
add_subdirectory(editor)
add_subdirectory(apps/luthien)

if(LUTH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()