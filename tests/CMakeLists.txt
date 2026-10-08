add_executable(LuthTests
    main.cpp
    resources/TextureCompressor.cpp
    memory/test_taggedpage.cpp
    memory/test_linearallocator.cpp
    jobs/test_spinlock.cpp
    jobs/test_smoke.cpp
    jobs/test_mpmcqueue.cpp
    jobs/test_atomiccounter.cpp
    jobs/test_v3_isrecording.cpp
    jobs/test_v2_main_isolation.cpp
    jobs/test_v4_wakeup.cpp
    jobs/test_v5_inline.cpp
    jobs/test_workstealing.cpp
    jobs/test_waitcounter.cpp
    renderer/RenderGraphSolver.cpp
    renderer/RenderBlackboard.cpp
    renderer/RenderPipelineCompiler.cpp
    renderer/RenderViewState.cpp
    renderer/SortedTransparencyFeature.cpp
    renderer/BloomFeature.cpp
    renderer/TaaFeature.cpp
    renderer/SlimVizViewState.cpp
    renderer/SlimVizFeature.cpp
    renderer/OutlineFeature.cpp
    renderer/SelectionMaskFeature.cpp
    renderer/GridFeature.cpp
    renderer/CompositeFeature.cpp
    renderer/EditorOverlayViewState.cpp
    renderer/CompositeViewState.cpp
    renderer/BloomViewState.cpp
    renderer/TaaViewState.cpp
    renderer/TransparencyViewState.cpp
    renderer/RefractionBackdropFeature.cpp
    renderer/FogCompositeFeature.cpp
    renderer/FogComputeFeature.cpp
    renderer/FogViewState.cpp
    renderer/GTAOFeature.cpp
    renderer/DeformationFeature.cpp
    renderer/VisibilityFeature.cpp
    renderer/DepthPrepassFeature.cpp
    renderer/SlimGBufferFeature.cpp
    renderer/CsmFeature.cpp
    renderer/ClusteredLightingFeature.cpp
    renderer/SkyFeature.cpp
    renderer/ForwardOpaqueFeature.cpp
)
target_compile_features(LuthTests PRIVATE cxx_std_20)
target_include_directories(LuthTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(LuthTests PRIVATE Luth::Luth doctest)

luth_enable_tracy_fibers(LuthTests)
luth_enable_asan(LuthTests)
luth_set_output_directories(LuthTests)
set_property(TARGET LuthTests PROPERTY FOLDER "Tests")

add_custom_command(TARGET LuthTests POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        ${LUTH_SLANG_RUNTIME_FILES}
        "$<TARGET_FILE_DIR:LuthTests>"
    COMMAND_EXPAND_LISTS
    VERBATIM
)

add_test(NAME luth-smoke COMMAND LuthTests "--test-case=*[smoke]*")
set_tests_properties(luth-smoke PROPERTIES
    LABELS smoke
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(NAME luth-all COMMAND LuthTests)
set_tests_properties(luth-all PROPERTIES
    LABELS full
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)