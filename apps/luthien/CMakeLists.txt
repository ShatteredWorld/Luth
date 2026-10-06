add_executable(Runtime 
    source/LuthienApp.cpp
)
set_target_properties(Runtime PROPERTIES
    OUTPUT_NAME Luthien
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_sources(Runtime PRIVATE resource.h Luthien.rc icons/Luth.ico)
endif()

target_compile_features(Runtime PRIVATE cxx_std_20)
target_include_directories(Runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/source")
target_link_libraries(Runtime PRIVATE Luth::Luthien Luth::Luth)

luth_enable_tracy_fibers(Runtime)
luth_enable_asan(Runtime)
luth_set_output_directories(Runtime)
set_property(TARGET Runtime PROPERTY FOLDER "Luthien")

add_custom_command(TARGET Runtime POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        ${LUTH_SLANG_RUNTIME_FILES}
        "$<TARGET_FILE_DIR:Runtime>"
    COMMAND_EXPAND_LISTS
    VERBATIM
)

if(LUTH_AFTERMATH_DLL)
    add_custom_command(TARGET Runtime POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${LUTH_AFTERMATH_DLL}"
            "$<TARGET_FILE_DIR:Runtime>"
        VERBATIM
    )
endif()
