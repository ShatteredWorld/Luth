include_guard(GLOBAL)

set(LUTH_PLATFORM_DIR "windows-x86_64")

add_library(luth_build_config INTERFACE)
add_library(Luth::BuildConfig ALIAS luth_build_config)

target_compile_definitions(luth_build_config INTERFACE
    GLFW_INCLUDE_NONE
    FMT_HEADER_ONLY=1
    SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
    "$<$<CONFIG:Debug>:LUTH_BUILD_DEBUG>"
    "$<$<CONFIG:Release>:LUTH_BUILD_RELEASE;NDEBUG>"
    "$<$<CONFIG:Dist>:LUTH_BUILD_DIST;NDEBUG>"
)

target_compile_options(luth_build_config INTERFACE
    "$<$<COMPILE_LANGUAGE:C>:/utf-8;/FS>"
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/FS;/EHsc>"
)

function(luth_set_output_directories target)
    foreach(config Debug Release Dist)
        string(TOUPPER "${config}" config_upper)
        set(output_config "${config}")
        if(LUTH_ENABLE_ASAN AND config STREQUAL "Debug")
            set(output_config "DebugASan")
        endif()
        set(output_dir "${CMAKE_SOURCE_DIR}/bin/${LUTH_PLATFORM_DIR}/${output_config}/${target}")
        set_target_properties(${target} PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${output_dir}"
            LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${output_dir}"
            RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${output_dir}"
            PDB_OUTPUT_DIRECTORY_${config_upper} "${output_dir}"
            COMPILE_PDB_OUTPUT_DIRECTORY_${config_upper} "${output_dir}"
        )
    endforeach()
endfunction()

function(luth_enable_asan target)
    if(NOT LUTH_ENABLE_ASAN)
        return()
    endif()

    target_compile_options(${target} PRIVATE /fsanitize=address /Zi)
    target_link_options(${target} PRIVATE /DEBUG)
    set_property(TARGET ${target} PROPERTY MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)

    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "EXECUTABLE")
        get_filename_component(msvc_tool_dir "${CMAKE_LINKER}" DIRECTORY)
        set(asan_runtime "${msvc_tool_dir}/clang_rt.asan_dynamic-x86_64.dll")
        if(NOT EXISTS "${asan_runtime}")
            message(FATAL_ERROR "MSVC ASan runtime not found: ${asan_runtime}")
        endif()
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${asan_runtime}"
                "$<TARGET_FILE_DIR:${target}>"
            VERBATIM
        )
    endif()
endfunction()

function(luth_enable_tracy_fibers target)
    target_compile_definitions(${target} PRIVATE
        "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>,$<NOT:$<BOOL:${LUTH_ENABLE_ASAN}>>>:TRACY_FIBERS>"
    )
endfunction()
