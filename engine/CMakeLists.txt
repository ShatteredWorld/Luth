add_library(Luth STATIC 

    source/luthpch.cpp

    source/luth/core/App.cpp
    source/luth/core/DebugDraw.cpp
    source/luth/core/EditorHooks.cpp
    source/luth/core/ProjectFile.cpp
    source/luth/core/RenderSnapshot.cpp
    source/luth/core/UUID.cpp
    source/luth/core/Version.cpp

    source/luth/core/diagnostics/Log.cpp
    source/luth/core/diagnostics/StackTrace.cpp

    source/luth/jobs/FiberPrimitiveHelper.cpp
    source/luth/jobs/FiberStackAllocator.cpp
    source/luth/jobs/IOThread.cpp
    source/luth/jobs/JobSystem.cpp
    source/luth/jobs/MainThreadPump.cpp

    source/luth/memory/GPUTaggedPageAllocator.cpp
    source/luth/memory/GlobalNewDelete.cpp
    source/luth/memory/MemoryTracker.cpp
    source/luth/memory/TaggedPageAllocator.cpp

    source/luth/physics/LuthJobSystemForJolt.cpp
    source/luth/physics/PhysicsDebugRenderer.cpp
    source/luth/physics/PhysicsListeners.cpp
    source/luth/physics/PhysicsMaterial.cpp
    source/luth/physics/ShapeBuilder.cpp
    source/luth/physics/ShapeCache.cpp

    source/luth/platform/FileDialog.cpp
    source/luth/platform/Input.cpp
    source/luth/platform/Window.cpp

    source/luth/renderer/DrawListBuilder.cpp
    source/luth/renderer/FrameDebugger.cpp
    source/luth/renderer/FrameTargets.cpp
    source/luth/renderer/RenderBackend.cpp
    source/luth/renderer/RenderPipeline.cpp
    source/luth/renderer/Renderer.cpp
    source/luth/renderer/ViewResources.cpp

    source/luth/renderer/features/RenderBlackboard.cpp
    source/luth/renderer/features/GTAOFeature.cpp
    source/luth/renderer/features/DeformationFeature.cpp
    source/luth/renderer/features/VisibilityFeature.cpp
    source/luth/renderer/features/DepthPrepassFeature.cpp
    source/luth/renderer/features/SlimGBufferFeature.cpp
    source/luth/renderer/features/CsmFeature.cpp
    source/luth/renderer/features/ClusteredLightingFeature.cpp
    source/luth/renderer/features/SkyFeature.cpp
    source/luth/renderer/features/ForwardOpaqueFeature.cpp
    source/luth/renderer/features/TransparencyFeature.cpp
    source/luth/renderer/subsystems/TransparencyOit.cpp
    source/luth/renderer/subsystems/TransparencySorted.cpp
    source/luth/renderer/features/BloomFeature.cpp
    source/luth/renderer/subsystems/PostProcessBloom.cpp
    source/luth/renderer/features/TaaFeature.cpp
    source/luth/renderer/subsystems/PostProcessTaa.cpp
    source/luth/renderer/features/SlimVizViewState.cpp
    source/luth/renderer/subsystems/PostProcessSlimVizState.cpp
    source/luth/renderer/features/SlimVizFeature.cpp
    source/luth/renderer/features/ClusterVizFeature.cpp
    source/luth/renderer/features/FogVizFeature.cpp
    source/luth/renderer/features/rt/GiReservoirVizFeature.cpp
    source/luth/renderer/features/ClusterVizViewState.cpp
    source/luth/renderer/subsystems/PostProcessSlimViz.cpp
    source/luth/renderer/features/OutlineFeature.cpp
    source/luth/renderer/features/DebugDrawFeature.cpp
    source/luth/renderer/subsystems/EditorOverlayOutline.cpp
    source/luth/renderer/features/SelectionMaskFeature.cpp
    source/luth/renderer/subsystems/EditorOverlaySelection.cpp
    source/luth/renderer/features/GridFeature.cpp
    source/luth/renderer/subsystems/EditorOverlayGrid.cpp
    source/luth/renderer/features/CompositeFeature.cpp
    source/luth/renderer/subsystems/PostProcessComposite.cpp
    source/luth/renderer/features/EditorOverlayViewState.cpp
    source/luth/renderer/subsystems/EditorOverlayState.cpp
    source/luth/renderer/features/CompositeViewState.cpp
    source/luth/renderer/subsystems/PostProcessCompositeState.cpp
    source/luth/renderer/features/BloomViewState.cpp
    source/luth/renderer/subsystems/PostProcessBloomState.cpp
    source/luth/renderer/features/TaaViewState.cpp
    source/luth/renderer/subsystems/PostProcessTaaState.cpp
    source/luth/renderer/features/TransparencyViewState.cpp
    source/luth/renderer/features/RefractionBackdropFeature.cpp
    source/luth/renderer/subsystems/TransparencyBackdrop.cpp
    source/luth/renderer/features/FogCompositeFeature.cpp
    source/luth/renderer/subsystems/VolumetricComposite.cpp
    source/luth/renderer/subsystems/VolumetricViz.cpp
    source/luth/renderer/features/FogComputeFeature.cpp
    source/luth/renderer/subsystems/VolumetricCompute.cpp
    source/luth/renderer/features/FogViewState.cpp
    source/luth/renderer/features/GtaoViewState.cpp
    source/luth/renderer/features/RenderPipelineCompiler.cpp

    source/luth/renderer/backend/vulkan/AftermathCrashTracker.cpp
    source/luth/renderer/backend/vulkan/CommandAllocator.cpp
    source/luth/renderer/backend/vulkan/CommandAllocatorPool.cpp
    source/luth/renderer/backend/vulkan/GPUTimerPool.cpp
    source/luth/renderer/backend/vulkan/GpuCheckpoint.cpp
    source/luth/renderer/backend/vulkan/PipelineCache.cpp
    source/luth/renderer/backend/vulkan/TimelineSemaphore.cpp
    source/luth/renderer/backend/vulkan/TlasBuilder.cpp
    source/luth/renderer/backend/vulkan/UploadContext.cpp
    source/luth/renderer/backend/vulkan/VulkanAccelerationStructure.cpp
    source/luth/renderer/backend/vulkan/VulkanAllocator.cpp
    source/luth/renderer/backend/vulkan/VulkanBackend.cpp
    source/luth/renderer/backend/vulkan/VulkanBuffer.cpp
    source/luth/renderer/backend/vulkan/VulkanComputePipeline.cpp
    source/luth/renderer/backend/vulkan/VulkanContext.cpp
    source/luth/renderer/backend/vulkan/VulkanDescriptors.cpp
    source/luth/renderer/backend/vulkan/VulkanPipeline.cpp
    source/luth/renderer/backend/vulkan/VulkanShader.cpp
    source/luth/renderer/backend/vulkan/VulkanSwapchain.cpp
    source/luth/renderer/backend/vulkan/VulkanTexture.cpp

    source/luth/renderer/shader/ShaderReloadCoordinator.cpp
    source/luth/renderer/debug/ViewGpuProfiler.cpp
    source/luth/renderer/debug/ProfilingProvenance.cpp
    source/luth/renderer/debug/GraphInstrumentation.cpp
    source/luth/renderer/debug/CaptureFinalization.cpp
    source/luth/renderer/debug/CaptureRecordingSession.cpp
    source/luth/renderer/debug/FrameDebuggerContext.cpp

    source/luth/renderer/lighting/CascadeBuilder.cpp
    source/luth/renderer/lighting/EmissiveLightGatherer.cpp
    source/luth/renderer/lighting/FogVolumeGatherer.cpp
    source/luth/renderer/lighting/IBLPrecompute.cpp
    source/luth/renderer/lighting/LightGatherer.cpp

    source/luth/renderer/material/Material.cpp
    source/luth/renderer/material/MaterialGraphCodegen.cpp
    source/luth/renderer/material/MaterialLayoutGuard.cpp
    source/luth/renderer/material/MaterialSystem.cpp

    source/luth/renderer/passes/ImGuiPass.cpp

    source/luth/renderer/pipeline/PipelineManager.cpp

    source/luth/renderer/rendergraph/ArchivedImage.cpp
    source/luth/renderer/rendergraph/FrameEventTree.cpp
    source/luth/renderer/rendergraph/RenderGraph.cpp
    source/luth/renderer/rendergraph/RenderResourceCache.cpp

    source/luth/renderer/resources/BoneMatrixBuffer.cpp
    source/luth/renderer/resources/Buffer.cpp
    source/luth/renderer/resources/Mesh.cpp
    source/luth/renderer/resources/Model.cpp
    source/luth/renderer/resources/Texture.cpp

    source/luth/renderer/shader/Shader.cpp
    source/luth/renderer/shader/ShaderCompiler.cpp
    source/luth/renderer/shader/ShaderLibrary.cpp
    source/luth/renderer/shader/ShaderWatcher.cpp
    source/luth/renderer/shader/SlangCompiler.cpp

    source/luth/renderer/subsystems/DebugDrawSubsystem.cpp
    source/luth/renderer/subsystems/DebugDrawRecording.cpp
    source/luth/renderer/subsystems/EditorOverlaysSubsystem.cpp
    source/luth/renderer/subsystems/GTAOSubsystem.cpp
    source/luth/renderer/subsystems/GeometrySubsystem.cpp
    source/luth/renderer/subsystems/GlobalSubsystem.cpp
    source/luth/renderer/subsystems/LightingSubsystem.cpp
    source/luth/renderer/subsystems/LightingClusterViz.cpp
    source/luth/renderer/subsystems/LightingClusterVizState.cpp
    source/luth/renderer/subsystems/PathTraceSubsystem.cpp
    source/luth/renderer/subsystems/PostProcessSubsystem.cpp
    source/luth/renderer/subsystems/ReflectionsSubsystem.cpp
    source/luth/renderer/subsystems/RtRestirGiSubsystem.cpp
    source/luth/renderer/subsystems/RtGiReservoirViz.cpp
    source/luth/renderer/subsystems/RtRestirSubsystem.cpp
    source/luth/renderer/subsystems/RtSubsystem.cpp
    source/luth/renderer/subsystems/SkinningSubsystem.cpp
    source/luth/renderer/subsystems/SlangParityGuard.cpp
    source/luth/renderer/subsystems/SvgfDenoiser.cpp
    source/luth/renderer/subsystems/TransparencySubsystem.cpp
    source/luth/renderer/subsystems/VolumetricSubsystem.cpp

    source/luth/assets/Asset.cpp
    source/luth/assets/AssetDatabase.cpp
    source/luth/assets/AssetManager.cpp
    source/luth/assets/AssetSerializer.cpp
    source/luth/assets/FileSystem.cpp
    source/luth/assets/FileWatcher.cpp
    source/luth/assets/Image.cpp
    source/luth/assets/MetaFile.cpp

    source/luth/assets/importers/AnimationClipImporter.cpp
    source/luth/assets/importers/MaterialImporter.cpp
    source/luth/assets/importers/ModelImporter.cpp
    source/luth/assets/importers/PhysicsMaterialImporter.cpp
    source/luth/assets/importers/ProjectTextureIndex.cpp
    source/luth/assets/importers/ShaderImporter.cpp
    source/luth/assets/importers/TextureBaker.cpp
    source/luth/assets/importers/TextureCompressor.cpp
    source/luth/assets/importers/TextureImporter.cpp
    source/luth/assets/importers/TextureResolver.cpp

    source/luth/scene/Entity.cpp
    source/luth/scene/Scene.cpp
    source/luth/scene/SceneSerializer.cpp

    source/luth/scene/systems/AnimationSystem.cpp
    source/luth/scene/systems/DebugGizmos.cpp
    source/luth/scene/systems/LightingSystem.cpp
    source/luth/scene/systems/PhysicsSystem.cpp
    source/luth/scene/systems/PickingSystem.cpp
    source/luth/scene/systems/PlayerControllerSystem.cpp
    source/luth/scene/systems/RenderingSystem.cpp
    source/luth/scene/systems/SystemRegistry.cpp
)
add_library(Luth::Luth ALIAS Luth)

add_library(luth_fiber_asm OBJECT source/luth/jobs/FiberPrimitive.asm)
target_sources(Luth PRIVATE
    source/luth/platform/WinWindow.cpp
    $<TARGET_OBJECTS:luth_fiber_asm>
)
set_property(TARGET luth_fiber_asm PROPERTY FOLDER Luth)

target_compile_features(Luth PUBLIC cxx_std_20)
target_precompile_headers(Luth PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/source/luthpch.h)
target_include_directories(Luth
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/source
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/source
)

target_link_libraries(Luth
    PUBLIC
        Luth::BuildConfig
        assimp::assimp
        glfw
        glm::glm
        ImGui::ImGui
        ImGuizmo::ImGuizmo
        spdlog::spdlog_header_only
        Tracy::TracyClient
        spirv-cross-core
        spirv-cross-glsl
        Jolt::Jolt
        bc7enc::bc7enc
        slang_compiler
        entt
        nlohmann_json
        stb
        GPUOpen::VulkanMemoryAllocator
        Vulkan::Vulkan
        Threads::Threads
)

target_link_libraries(Luth PUBLIC ws2_32 dbghelp)

if(DEFINED ENV{AFTERMATH_SDK} AND NOT "$ENV{AFTERMATH_SDK}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{AFTERMATH_SDK}" LUTH_AFTERMATH_SDK)
    set(LUTH_AFTERMATH_DLL "${LUTH_AFTERMATH_SDK}/lib/x64/GFSDK_Aftermath_Lib.x64.dll"
        CACHE INTERNAL "Nsight Aftermath runtime")
    target_include_directories(Luth PRIVATE "${LUTH_AFTERMATH_SDK}/include")
    target_compile_definitions(Luth PRIVATE LUTH_ENABLE_AFTERMATH=1)
endif()

luth_enable_asan(Luth)
luth_set_output_directories(Luth)
set_property(TARGET Luth PROPERTY FOLDER "Luth")