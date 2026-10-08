#include "luthpch.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/features/GTAOFeature.h"
#include "luth/renderer/features/DeformationFeature.h"
#include "luth/renderer/features/VisibilityFeature.h"
#include "luth/renderer/features/DepthPrepassFeature.h"
#include "luth/renderer/features/SlimGBufferFeature.h"
#include "luth/renderer/features/CsmFeature.h"
#include "luth/renderer/features/ClusteredLightingFeature.h"
#include "luth/renderer/features/FogComputeFeature.h"
#include "luth/renderer/features/FogCompositeFeature.h"
#include "luth/renderer/features/RefractionBackdropFeature.h"
#include "luth/renderer/features/TransparencyFeature.h"
#include "luth/renderer/features/TaaFeature.h"
#include "luth/renderer/features/BloomFeature.h"
#include "luth/renderer/features/CompositeFeature.h"
#include "luth/renderer/features/GridFeature.h"
#include "luth/renderer/features/SelectionMaskFeature.h"
#include "luth/renderer/features/OutlineFeature.h"
#include "luth/renderer/features/DebugDrawFeature.h"
#include "luth/renderer/features/SlimVizFeature.h"
#include "luth/renderer/features/ClusterVizFeature.h"
#include "luth/renderer/features/FogVizFeature.h"
#include "luth/renderer/features/rt/GiReservoirVizFeature.h"
#include "luth/renderer/features/SkyFeature.h"
#include "luth/renderer/features/ForwardOpaqueCompatibility.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"

#include "luth/renderer/debug/CaptureRecordingSession.h"
#include "luth/renderer/debug/CaptureFinalization.h"
#include "luth/scene/systems/RenderingSystem.h"
#include "luth/scene/systems/SystemRegistry.h"
#include "luth/scene/systems/LightingSystem.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/material/MaterialSystem.h"
#include "luth/renderer/resources/BoneMatrixBuffer.h"
#include "luth/renderer/backend/vulkan/VulkanBackend.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/renderer/backend/vulkan/VulkanAllocator.h"
#include "luth/renderer/backend/vulkan/VulkanShader.h"
#include "luth/renderer/backend/vulkan/DynamicRendering.h"
#include "luth/renderer/material/Material.h"
#include "luth/renderer/resources/Model.h"
#include "luth/renderer/resources/Buffer.h"
#include "luth/renderer/lighting/IBLPrecompute.h"
#include "luth/renderer/draw/DrawCommand.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/assets/AssetManager.h"
#include "luth/assets/AssetDatabase.h"
#include "luth/assets/FileSystem.h"
#include "luth/core/FrameData.h"
#include "luth/core/types/LuthMath.h"
#include "luth/core/time/Time.h"
#include "luth/core/diagnostics/Profiler.h"
#include "luth/scene/Components.h"
#include "luth/scene/Scene.h"

#include <vk_mem_alloc.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>
#include <string>
#include <vector>

namespace Luth
{
    using namespace Component;

    RenderPipeline::RenderPipeline(RenderingSystem& system)
        : m_System(system)
        , m_Denoise(std::make_unique<SvgfDenoiser>(DenoiserChannel::Di))
        , m_DenoiseGi(std::make_unique<SvgfDenoiser>(DenoiserChannel::Gi))
        , m_DenoiseRefl(std::make_unique<SvgfDenoiser>(DenoiserChannel::Reflections))
        , m_DenoiseDiSpec(std::make_unique<SvgfDenoiser>(DenoiserChannel::DiSpecular))
    {
    }

    RenderPipeline::~RenderPipeline() = default;

    // ---- Lifecycle: Initialize / Shutdown ----

    void RenderPipeline::Initialize(u32 viewportWidth, u32 viewportHeight)
    {
        LH_PROFILE_FUNCTION();
        if (Renderer::GetBackend()->GetAPI() != RenderBackend::API::Vulkan) return;

        m_System.GetSceneTargets().Allocate(viewportWidth, viewportHeight);

        m_Global.Init(*this);

        BoneMatrixBuffer::Init();
        m_EditorOverlays.Init();
        m_DebugDraw.Init();
        m_PostProcess.Init();

        // Lighting owns Set 3 + shadow map + IBL + skybox VB/SPVs. Engine ships no HDR; an empty path triggers
        // IBL::Precompute's silent dummy-cubemap fallback. Editor::OnProjectChanged invokes ReloadSkybox once
        // the project paths are live.
        m_Lighting.Init(*this, FileSystem::HasProject()
            ? FileSystem::ResolveAsset("textures/environment.hdr")
            : fs::path{});

        // Geometry owns Set 5 + cull + PBR + DepthPrepass.
        // Init creates layouts + descriptors + cull pipeline; pipelines that need geoLayouts build below.
        m_Geometry.Init(*this);
        // Transparency's Set 6 layout must exist before its BuildPipelines below appends it to geoLayouts
        // (same Init-before-BuildPipelines invariant as Geometry's Set 5).
        m_Transparency.Init();

        // Shadow / skybox / PBR / DepthPrepass pipelines all need the shared 6-layout vector.
        std::vector<VkDescriptorSetLayout> geoLayouts = {
            m_Global.GetSetLayout(),
            VulkanContext::Get().GetBindlessSet().GetLayout(),
            MaterialSystem::GetDescriptorSetLayout(),
            m_Lighting.GetSetLayout(),
            BoneMatrixBuffer::GetDescriptorSetLayout(),
            m_Geometry.GetSet5Layout()
        };
        m_Lighting.BuildPipelines(geoLayouts);
        m_Geometry.BuildPipelines(geoLayouts);
        m_Transparency.BuildPipelines(geoLayouts);
        m_EditorOverlays.BuildPipelines(geoLayouts);
        m_DebugDraw.BuildPipelines();

        m_GTAO.Init();
        RenderPipelineDefinition gtaoDefinition;
        gtaoDefinition.AddFeature<GTAOFeature>(m_GTAO, m_GtaoStates, &m_System.GetFrameDebugger());
        PipelineInputContract gtaoInputs;
        gtaoInputs.resources = {{RenderResources::SurfaceDepth}, {GtaoResources::Parameters}};
        auto gtaoCompiled = RenderPipelineCompiler{}.Compile(std::move(gtaoDefinition), {}, gtaoInputs);
        if (!gtaoCompiled.ReplaceIfValid(m_GtaoPipeline))
            throw std::runtime_error("GTAO feature definition failed semantic validation");
        m_Volumetric.Init(*this);
        m_Rt.Init(*this);
        m_Restir.Init(*this);
        m_RestirGi.Init(*this);
        m_SlangParity.Init(*this);
        m_PathTrace.Init(*this);
        m_Reflections.Init(*this);
        m_Denoise->Init(*this);
        m_DenoiseGi->Init(*this);
        m_DenoiseRefl->Init(*this);
        m_DenoiseDiSpec->Init(*this);
        m_Skinning.Init();
        RenderPipelineDefinition deformationDefinition;
        deformationDefinition.AddFeature<VisibilityFeature>(m_Geometry, &m_System.GetFrameDebugger());
        deformationDefinition.AddFeature<DeformationFeature>(m_Skinning);
        PipelineInputContract deformationInputs;
        deformationInputs.resources = {{DeformationResources::Parameters}, {VisibilityResources::Parameters},
            {RenderResources::ObjectData}, {RenderResources::InitializedIndirectData}};
        auto deformationCompiled = RenderPipelineCompiler{}.Compile(std::move(deformationDefinition), {}, deformationInputs);
        if (!deformationCompiled.ReplaceIfValid(m_GeometryPreparationPipeline))
            throw std::runtime_error("Geometry preparation feature definition failed semantic validation");
        RenderPipelineDefinition depthDefinition;
        depthDefinition.AddFeature<DepthPrepassFeature>(m_Geometry, &m_System.GetFrameDebugger());
        depthDefinition.AddFeature<SlimGBufferFeature>(m_Geometry, &m_System.GetFrameDebugger());
        PipelineInputContract depthInputs;
        depthInputs.resources = {{RenderResources::CameraVisibleDraws}, {DepthPrepassResources::Target},
            {DepthPrepassResources::Bindings}, {SlimGBufferResources::NormalTarget}, {SlimGBufferResources::RoughnessTarget},
            {SlimGBufferResources::MotionTarget}, {SlimGBufferResources::MaterialTarget}, {SlimGBufferResources::Bindings}};
        depthInputs.capabilities = {&DeformationResources::DeformedGeometry};
        auto depthCompiled = RenderPipelineCompiler{}.Compile(std::move(depthDefinition), {}, depthInputs);
        if (!depthCompiled.ReplaceIfValid(m_SurfacePreparationComposition))
            throw std::runtime_error("Surface preparation feature definition failed semantic validation");

        RenderPipelineDefinition csmDefinition;
        csmDefinition.AddFeature<CsmFeature>(m_Lighting, &m_System.GetFrameDebugger());
        PipelineInputContract csmInputs;
        csmInputs.resources = {{CsmResources::Parameters}, {CsmResources::Bindings},
            {RenderResources::CascadeVisibleDraws, ResourceOutputPresence::Optional}};
        csmInputs.capabilities = {&DeformationResources::DeformedGeometry};
        auto csmCompiled = RenderPipelineCompiler{}.Compile(std::move(csmDefinition), {}, csmInputs);
        if (!csmCompiled.ReplaceIfValid(m_CsmComposition))
            throw std::runtime_error("CSM feature definition failed semantic validation");
        RenderPipelineDefinition clusterDefinition;
        clusterDefinition.AddFeature<ClusteredLightingFeature>(m_Lighting, &m_System.GetFrameDebugger());
        PipelineInputContract clusterInputs;
        clusterInputs.resources = {{ClusterResources::Bindings},
            {ClusterResources::UploadedLights, ResourceOutputPresence::Optional}};
        auto clusterCompiled = RenderPipelineCompiler{}.Compile(std::move(clusterDefinition), {}, clusterInputs);
        if (!clusterCompiled.ReplaceIfValid(m_ClusterComposition))
            throw std::runtime_error("Clustered lighting feature definition failed semantic validation");
        RenderPipelineDefinition fogDefinition;
        fogDefinition.AddFeature<FogComputeFeature>(m_Volumetric, &m_System.GetFrameDebugger());
        PipelineInputContract fogInputs;
        fogInputs.resources = {{FogResources::Bindings},
            {FogResources::Volumes, ResourceOutputPresence::Optional},
            {RenderResources::LightData, ResourceOutputPresence::Optional},
            {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
            {RenderResources::LightIndices, ResourceOutputPresence::Optional},
            {RenderResources::ShadowCascades, ResourceOutputPresence::Optional}};
        auto fogCompiled = RenderPipelineCompiler{}.Compile(std::move(fogDefinition), {}, fogInputs);
        if (!fogCompiled.ReplaceIfValid(m_FogComputeComposition))
            throw std::runtime_error("Fog compute feature definition failed semantic validation");
        RenderPipelineDefinition transparencyDefinition;
        transparencyDefinition.AddFeature<TransparencyFeature>(m_Transparency, &m_System.GetFrameDebugger());
        PipelineInputContract transparencyInputs;
        transparencyInputs.resources = {{TransparencyResources::Bindings}, {RenderResources::FoggedHDR}, {RenderResources::LitDepth},
            {RenderResources::OpaquePickingIDs}, {RenderResources::CameraVisibleDraws},
            {RenderResources::ResolvedFog, ResourceOutputPresence::Optional}, {RenderResources::RefractionBackdrop, ResourceOutputPresence::Optional},
            {RenderResources::LightData, ResourceOutputPresence::Optional}, {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
            {RenderResources::LightIndices, ResourceOutputPresence::Optional}};
        transparencyInputs.capabilities = {&DeformationResources::DeformedGeometry};
        auto transparencyCompiled = RenderPipelineCompiler{}.Compile(std::move(transparencyDefinition), {}, transparencyInputs);
        if (!transparencyCompiled.ReplaceIfValid(m_TransparencyComposition))
            throw std::runtime_error("Transparency definition failed semantic validation");
        RenderPipelineDefinition visualizationDefinition;
        visualizationDefinition.AddFeature<SlimVizFeature>(m_PostProcess, &m_System.GetFrameDebugger());
        visualizationDefinition.AddFeature<ClusterVizFeature>(m_Lighting, &m_System.GetFrameDebugger());
        visualizationDefinition.AddFeature<FogVizFeature>(m_Volumetric, &m_System.GetFrameDebugger());
        visualizationDefinition.AddFeature<GiReservoirVizFeature>(m_RestirGi);
        PipelineInputContract visualizationInputs;
        visualizationInputs.resources = {{RenderResources::TonemappedLDR}, {SlimVizResources::Bindings}, {ClusterVizResources::Bindings},
            {FogVizResources::Bindings}, {GiReservoirVizResources::Bindings},
            {GiReservoirVizResources::SpatialReservoir, ResourceOutputPresence::Optional}, {RenderResources::FogDensity, ResourceOutputPresence::Optional},
            {RenderResources::ResolvedFog, ResourceOutputPresence::Optional},
            {RenderResources::SurfaceDepth, ResourceOutputPresence::Optional}, {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
            {RenderResources::Normal, ResourceOutputPresence::Optional}, {RenderResources::Roughness, ResourceOutputPresence::Optional},
            {RenderResources::MotionVectors, ResourceOutputPresence::Optional}, {RenderResources::MaterialID, ResourceOutputPresence::Optional}};
        auto visualizationCompiled = RenderPipelineCompiler{}.Compile(std::move(visualizationDefinition), {}, visualizationInputs);
        if (!visualizationCompiled.ReplaceIfValid(m_VisualizationComposition))
            throw std::runtime_error("Visualization definition failed semantic validation");
        RenderPipelineDefinition outlineDefinition;
        outlineDefinition.AddFeature<OutlineFeature>(m_EditorOverlays, &m_System.GetFrameDebugger());
        PipelineInputContract outlineInputs;
        outlineInputs.resources = {{RenderResources::VisualizedLDR}, {OutlineResources::Bindings},
            {RenderResources::SelectionMask, ResourceOutputPresence::Optional},
            {RenderResources::SelectionDepth, ResourceOutputPresence::Optional},
            {RenderResources::LitDepth, ResourceOutputPresence::Optional}};
        auto outlineCompiled = RenderPipelineCompiler{}.Compile(std::move(outlineDefinition), {}, outlineInputs);
        if (!outlineCompiled.ReplaceIfValid(m_OutlineComposition))
            throw std::runtime_error("Outline definition failed semantic validation");
        RenderPipelineDefinition debugDrawDefinition;
        debugDrawDefinition.AddFeature<DebugDrawFeature>(m_DebugDraw, &m_System.GetFrameDebugger());
        PipelineInputContract debugDrawInputs; debugDrawInputs.resources = {{RenderResources::OutlinedLDR}, {DebugDrawResources::Bindings}};
        auto debugDrawCompiled = RenderPipelineCompiler{}.Compile(std::move(debugDrawDefinition), {}, debugDrawInputs);
        if (!debugDrawCompiled.ReplaceIfValid(m_DebugDrawComposition))
            throw std::runtime_error("DebugDraw definition failed semantic validation");
        RenderPipelineDefinition selectionDefinition;
        selectionDefinition.AddFeature<SelectionMaskFeature>(m_EditorOverlays, &m_System.GetFrameDebugger());
        PipelineInputContract selectionInputs; selectionInputs.resources = {{SelectionMaskResources::Bindings}};
        auto selectionCompiled = RenderPipelineCompiler{}.Compile(std::move(selectionDefinition), {}, selectionInputs);
        if (!selectionCompiled.ReplaceIfValid(m_SelectionMaskComposition))
            throw std::runtime_error("SelectionMask definition failed semantic validation");
        RenderPipelineDefinition gridDefinition;
        gridDefinition.AddFeature<GridFeature>(m_EditorOverlays, &m_System.GetFrameDebugger());
        PipelineInputContract gridInputs;
        gridInputs.resources = {{RenderResources::ResolvedHDR}, {GridResources::Bindings},
            {RenderResources::LitDepth, ResourceOutputPresence::Optional},
            {RenderResources::BloomOutput, ResourceOutputPresence::Optional}};
        auto gridCompiled = RenderPipelineCompiler{}.Compile(std::move(gridDefinition), {}, gridInputs);
        if (!gridCompiled.ReplaceIfValid(m_GridComposition))
            throw std::runtime_error("Grid definition failed semantic validation");
        RenderPipelineDefinition compositeDefinition;
        compositeDefinition.AddFeature<CompositeFeature>(m_PostProcess, &m_System.GetFrameDebugger());
        PipelineInputContract compositeInputs;
        compositeInputs.resources = {{RenderResources::GridHDR}, {RenderResources::BloomOutput, ResourceOutputPresence::Optional}, {CompositeResources::Bindings}};
        auto compositeCompiled = RenderPipelineCompiler{}.Compile(std::move(compositeDefinition), {}, compositeInputs);
        if (!compositeCompiled.ReplaceIfValid(m_CompositeComposition))
            throw std::runtime_error("Composite definition failed semantic validation");
        RenderPipelineDefinition bloomDefinition;
        bloomDefinition.AddFeature<BloomFeature>(m_PostProcess, &m_System.GetFrameDebugger());
        PipelineInputContract bloomInputs;
        bloomInputs.resources = {{RenderResources::ResolvedHDR}, {BloomResources::Bindings}};
        auto bloomCompiled = RenderPipelineCompiler{}.Compile(std::move(bloomDefinition), {}, bloomInputs);
        if (!bloomCompiled.ReplaceIfValid(m_BloomComposition))
            throw std::runtime_error("Bloom definition failed semantic validation");
        RenderPipelineDefinition taaDefinition;
        taaDefinition.AddFeature<TaaFeature>(m_PostProcess, &m_System.GetFrameDebugger());
        PipelineInputContract taaInputs;
        taaInputs.resources = {{RenderResources::TransparentHDR}, {RenderResources::MotionVectors},
            {RenderResources::LitDepth}, {TaaResources::Bindings}};
        auto taaCompiled = RenderPipelineCompiler{}.Compile(std::move(taaDefinition), {}, taaInputs);
        if (!taaCompiled.ReplaceIfValid(m_TaaComposition))
            throw std::runtime_error("TAA definition failed semantic validation");
        RenderPipelineDefinition backdropDefinition;
        backdropDefinition.AddFeature<RefractionBackdropFeature>(m_Transparency);
        PipelineInputContract backdropInputs;
        backdropInputs.resources = {{RenderResources::FoggedHDR}, {RefractionResources::Bindings}};
        auto backdropCompiled = RenderPipelineCompiler{}.Compile(std::move(backdropDefinition), {}, backdropInputs);
        if (!backdropCompiled.ReplaceIfValid(m_RefractionComposition))
            throw std::runtime_error("Refraction backdrop feature definition failed semantic validation");
        RenderPipelineDefinition fogCompositeDefinition;
        fogCompositeDefinition.AddFeature<FogCompositeFeature>(m_Volumetric, &m_System.GetFrameDebugger());
        PipelineInputContract fogCompositeInputs;
        fogCompositeInputs.resources = {{RenderResources::SkyHDR}, {RenderResources::SurfaceDepth},
            {FogCompositeResources::Bindings}, {RenderResources::ResolvedFog, ResourceOutputPresence::Optional}};
        auto fogCompositeCompiled = RenderPipelineCompiler{}.Compile(std::move(fogCompositeDefinition), {}, fogCompositeInputs);
        if (!fogCompositeCompiled.ReplaceIfValid(m_FogCompositeComposition))
            throw std::runtime_error("Fog composite feature definition failed semantic validation");
        RenderPipelineDefinition skyDefinition;
        skyDefinition.AddFeature<SkyFeature>(m_Lighting, &m_System.GetFrameDebugger());
        PipelineInputContract skyInputs;
        skyInputs.resources = {{RenderResources::OpaqueHDR}, {RenderResources::LitDepth}, {SkyResources::Bindings}};
        auto skyCompiled = RenderPipelineCompiler{}.Compile(std::move(skyDefinition), {}, skyInputs);
        if (!skyCompiled.ReplaceIfValid(m_SkyComposition))
            throw std::runtime_error("Sky feature definition failed semantic validation");
        RenderPipelineDefinition forwardDefinition;
        forwardDefinition.AddFeature<HybridForwardOpaqueFeature>(m_Geometry, &m_System.GetFrameDebugger());
        PipelineInputContract forwardInputs;
        forwardInputs.resources = {{ForwardOpaqueResources::Bindings}, {ForwardOpaqueResources::ColorTarget},
            {ForwardOpaqueResources::PickingTarget}, {RenderResources::CameraVisibleDraws}, {RenderResources::SurfaceDepth},
            {RenderResources::ShadowCascades, ResourceOutputPresence::Optional},
            {RenderResources::AmbientOcclusion, ResourceOutputPresence::Optional},
            {RenderResources::LightData, ResourceOutputPresence::Optional}, {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
            {RenderResources::LightIndices, ResourceOutputPresence::Optional},
            {ForwardCompatibilityResources::SunShadowMask, ResourceOutputPresence::Optional},
            {ForwardCompatibilityResources::DenoisedDiffuseDI, ResourceOutputPresence::Optional},
            {ForwardCompatibilityResources::DenoisedDiffuseGI, ResourceOutputPresence::Optional},
            {ForwardCompatibilityResources::DenoisedReflectionRadiance, ResourceOutputPresence::Optional},
            {ForwardCompatibilityResources::DenoisedSpecularDI, ResourceOutputPresence::Optional}};
        forwardInputs.capabilities = {&DeformationResources::DeformedGeometry};
        auto forwardCompiled = RenderPipelineCompiler{}.Compile(std::move(forwardDefinition), {}, forwardInputs);
        if (!forwardCompiled.ReplaceIfValid(m_ForwardComposition))
            throw std::runtime_error("Forward opaque definition failed semantic validation");
        // Shader hot-reload callback: pulls fresh SPIR-V into the cached blob and rebuilds pipelines that use it.
        // Fires after ShaderLibrary::Reload has already recompiled and re-reflected the single-stage shader.
        // Library keys are the shader filename (e.g. "pbr_vert.slang", "gtao_main.slang").
        ShaderLibrary::SetReloadCallback([this](const std::string& name) {
            // No vkDeviceWaitIdle: old pipelines are deferred-destroyed via VulkanContext::PushDeletion, which
            // drains MAX_FRAMES_IN_FLIGHT frames later in AcquireImage; by then the GPU has retired any command
            // buffer that bound them. Keeps shader save under steady frame pacing.
            auto vk = std::static_pointer_cast<VulkanShader>(ShaderLibrary::Get(name));
            if (!vk || !vk->IsValid())
            {
                LH_LOG(Renderer, error, "Shader reload: '{}' invalid - keeping existing pipelines", name);
                return;
            }
            const auto& spv = vk->GetSpirV();

            std::vector<VkDescriptorSetLayout> geoLayouts = {
                m_Global.GetSetLayout(),
                VulkanContext::Get().GetBindlessSet().GetLayout(),
                MaterialSystem::GetDescriptorSetLayout(),
                m_Lighting.GetSetLayout(),
                BoneMatrixBuffer::GetDescriptorSetLayout(),
                m_Geometry.GetSet5Layout()
            };
            // Subsystems handle their own shaders + pipeline rebuilds. Order matters: fullscreen.slang must
            // reach both PostProcess and EditorOverlays (PostProcess returns false for it; EditorOverlays
            // returns true). Debug shaders + IBL precompute remain RP residual.
            // Transparency runs OUTSIDE the || chain (overlays precedent): it must also see
            // pbr_vert.slang / pbr_skinned.slang (handled = true by Geometry) to invalidate its variants.
            const bool transparencyHandled = m_Transparency.OnShaderReloaded(name, spv);
            // SlangParity gate runs OUTSIDE the || chain: it re-scans restir_gi_initial.slang, which RestirGi
            // consumes first (short-circuiting the chain), and it rebuilds no pipeline of its own.
            m_SlangParity.OnShaderReloaded(name, spv);
            const bool handled = m_Lighting.OnShaderReloaded(name, spv, geoLayouts)
                              || m_Geometry.OnShaderReloaded(name, spv, geoLayouts)
                              || m_GTAO.OnShaderReloaded(name, spv)
                              || m_Volumetric.OnShaderReloaded(name, spv)
                              || m_Skinning.OnShaderReloaded(name, spv)
                              || m_Rt.OnShaderReloaded(name, spv)
                              || m_Restir.OnShaderReloaded(name, spv)
                              || m_RestirGi.OnShaderReloaded(name, spv)
                              || m_PathTrace.OnShaderReloaded(name, spv)
                              || m_Reflections.OnShaderReloaded(name, spv)
                              || m_Denoise->OnShaderReloaded(name, spv)
                              || m_DenoiseGi->OnShaderReloaded(name, spv)
                              || m_DenoiseRefl->OnShaderReloaded(name, spv)
                              || m_DenoiseDiSpec->OnShaderReloaded(name, spv);
            // PostProcess returns false for fullscreen.slang so EditorOverlays still gets to rebuild its outline/grid pipelines below.
            const bool ppHandled       = m_PostProcess.OnShaderReloaded(name, spv);
            const bool overlaysHandled = m_EditorOverlays.OnShaderReloaded(name, spv, geoLayouts);
            const bool debugHandled    = m_DebugDraw.OnShaderReloaded(name, spv);
            if (handled || ppHandled || overlaysHandled || debugHandled || transparencyHandled)
            {
                if      (name == "debugBlit.slang")  m_System.GetFrameDebugger().blitFragSpv  = spv;
                else if (name == "debugDepth.slang") m_System.GetFrameDebugger().depthFragSpv = spv;
                LH_LOG(Renderer, info, "Pipelines rebuilt after shader reload: {}", name);
                return;
            }

            // Debug-shader-only path (no pipeline rebuild on RP side; FrameDebuggerContext rebuilds lazily).
            if      (name == "debugBlit.slang")  m_System.GetFrameDebugger().blitFragSpv  = spv;
            else if (name == "debugDepth.slang") m_System.GetFrameDebugger().depthFragSpv = spv;
            // IBL precompute shaders refresh in the library; ReloadSkybox() must run to re-bake.
        });

        // Shader hot-reload watcher (engine-shaders dir; project dirs added via RenderingSystem::OnProjectLoaded).
        // Queues background-thread detections for main-thread Poll at the top of Execute.
        m_ShaderWatcher.Start(FileSystem::EngineAssetsPath("shaders"));

        m_GPUTimers.Init(256);   // headroom over the current ~70-pass RT graph; see ReadResults overflow warn
        RegisterNamedTextures();
    }

    void RenderPipeline::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        auto& s = m_System;

        m_ShaderWatcher.Stop();
        ShaderLibrary::SetReloadCallback(nullptr);
        m_GPUTimers.Shutdown();

        BoneMatrixBuffer::Shutdown();

        // Release per-view state before the shared layouts it references.
        for (auto& [targets, vr] : m_ViewResources)
            DestroyViewResources(vr);
        m_ViewResources.clear();
        m_GtaoPipeline.reset();
        m_GtaoStates.ReleaseAll([] { Renderer::WaitForGPU(); });

        // Subsystems own their layouts/pools/samplers/pipelines.
        m_Transparency.Shutdown();
        m_GeometryPreparationPipeline.reset();
        m_SurfacePreparationComposition.reset();
        m_CsmComposition.reset();
        m_ClusterComposition.reset();
        m_FogComputeComposition.reset();
        m_FogCompositeComposition.reset();
        m_RefractionComposition.reset();
        m_TransparencyComposition.reset();
        m_TaaComposition.reset();
        m_BloomComposition.reset();
        m_VisualizationComposition.reset();
        m_OutlineComposition.reset();
        m_DebugDrawComposition.reset();
        m_SelectionMaskComposition.reset();
        m_GridComposition.reset();
        m_CompositeComposition.reset();
        m_SkyComposition.reset();
        m_ForwardComposition.reset();
        m_Skinning.Shutdown();
        m_DenoiseDiSpec->Shutdown();
        m_DenoiseRefl->Shutdown();
        m_DenoiseGi->Shutdown();
        m_Denoise->Shutdown();
        m_Reflections.Shutdown();
        m_PathTrace.Shutdown();
        m_SlangParity.Shutdown();
        m_RestirGi.Shutdown();
        m_Restir.Shutdown();
        m_Rt.Shutdown();
        m_DebugDraw.Shutdown();
        m_EditorOverlays.Shutdown();
        m_PostProcess.Shutdown();
        m_Volumetric.Shutdown();
        m_GTAO.Shutdown();
        m_Geometry.Shutdown();
        m_Lighting.Shutdown();
        m_Global.Shutdown();
    }

    void RenderPipeline::OnResize(u32 width, u32 height)
    {
        // Scene-panel resize. FrameTargets is already resized by RenderingSystem::Resize;
        // EnsureViewResources picks up the size change and rebuilds textures + descriptors.
        EnsureViewResources(m_System.GetSceneTargets());
        RegisterNamedTextures();
    }

    void RenderPipeline::PrepareForTargets(FrameTargets& targets)
    {
        m_CurrentViewResources = &EnsureViewResources(targets);
        if (m_CurrentViewResources->taa) m_CurrentViewResources->taa->recorded = false;
    }

    void RenderPipeline::ExecuteMinimal()
    {
        LH_PROFILE_FUNCTION();
        auto& s = m_System;
        RG::RenderGraph rg(m_System.GetFrameAllocator());
        AddImGuiPass(rg, RG::ResourceHandle{}); // invalid -> ImGuiPass skips the optional Read
        rg.Compile();
        Renderer::ExecuteGraph(rg, Renderer::GetFrameData()->GetFrameIndex(), nullptr);
    }

    bool RenderPipeline::Execute(const RenderView& view, QueueRecorders recorders)
    {
        LH_PROFILE_FUNCTION();
        VkCommandBuffer primaryCmd = recorders.gA;
        auto& s = m_System;
        m_CurrentView          = &view;
        m_CurrentViewResources = view.targets ? &EnsureViewResources(*view.targets) : nullptr;

        RG::RenderGraph rg(m_System.GetFrameAllocator());

        // Import this frame's tagged-heap regions for RG barrier tracking.
        // Buffers can change identity each frame (heap allocator may reuse pages or grow backings).
        const auto& objectRegion   = m_Geometry.GetObjectRegion();
        const auto& indirectRegion = m_Geometry.GetIndirectRegion();
        RG::BufferDesc objDesc { "ObjectSSBO",     objectRegion.size,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
        RG::BufferDesc indDesc { "IndirectBuffer", indirectRegion.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT };
        RG::BufferHandle hObjectBuf   = rg.ImportBuffer(objDesc, (void*)objectRegion.buffer,   RG::ResourceState::Undefined);
        RG::BufferHandle hIndirectBuf = rg.ImportBuffer(indDesc, (void*)indirectRegion.buffer, RG::ResourceState::Undefined);

        // Deform: per-frame compute skinning into each mesh's deformed buffer, as the FIRST graphics
        // pass so raster geometry (gA) reads the current-frame deformation. Decoupled from needTlas:
        // raster always needs it, even when no RT consumer builds a TLAS this frame.
        const bool ptEnabled = m_PathTrace.IsEnabled() && m_CurrentViewResources
                            && m_Rt.GetTlas() != VK_NULL_HANDLE;
        VisibilityParameters visibilityParams;
        visibilityParams.cameraPlanes = CreateFrustumFromCamera(m_Global.GetCachedViewProj()).planes;
        visibilityParams.viewIndex = view.viewIndex;
        visibilityParams.regionStride = k_IndirectRegionStride;
        visibilityParams.maxViews = k_MaxViews;
        visibilityParams.objectCount = m_Geometry.GetGPUObjectCount();
        visibilityParams.realtime = !ptEnabled;
        visibilityParams.cullCascades = m_Global.GetShadowParams().castShadows
            && (m_Global.GetShadowParams().mode == ShadowingMode::RasterCSM || view.camera.enableVolumetricFog);
        if (visibilityParams.cullCascades)
            for (u32 cascade = 0; cascade < k_ShadowCascadeCount; ++cascade)
                visibilityParams.cascadePlanes[cascade] = CreateFrustumFromCamera(m_Global.GetCascades().lightSpaceMatrix[cascade]).planes;
        const GraphBufferRef objectInput{hObjectBuf, {&objectRegion, objectRegion.offset, objectRegion.size}};
        const GraphBufferRef indirectInput{hIndirectBuf, {&indirectRegion, indirectRegion.offset, indirectRegion.size}};
        const DeformationParameters deformationParams{s.GetWindSettings(), Time::GetTime()};
        const std::array deformationBindings{
            RenderInputBinding::Present(DeformationResources::Parameters, deformationParams),
            RenderInputBinding::Present(VisibilityResources::Parameters, visibilityParams),
            RenderInputBinding::Present(RenderResources::ObjectData, objectInput),
            RenderInputBinding::Present(RenderResources::InitializedIndirectData, indirectInput)};
        FrameRenderInputs deformationFrame;
        deformationFrame.renderFrameIndex = Renderer::GetFrameData()->GetRenderFrameIndex();
        deformationFrame.snapshot = &s.GetActiveSnapshot();
        deformationFrame.resources = deformationBindings;
        ViewRenderInputs deformationView;
        deformationView.id = view.id;
        VisibleDrawRange cameraVisible;
        CascadeDrawRanges cascadeVisible;
        const std::array visibilityOutputs{
            RenderOutputBinding::Capture(RenderResources::CameraVisibleDraws, cameraVisible),
            RenderOutputBinding::Capture(RenderResources::CascadeVisibleDraws, cascadeVisible)};
        const auto deformationBuild = m_GeometryPreparationPipeline->Build(rg, deformationFrame, deformationView,
            s.GetFrameAllocator(), visibilityOutputs);
        if (!deformationBuild.success)
        {
            for (const auto& diagnostic : deformationBuild.diagnostics)
                LH_LOG(Renderer, error, "Geometry preparation composition: {}", diagnostic.message);
            return false;
        }

        // PathTrace replaces the entire real-time pipeline: its megakernel output feeds the post chain via
        // hdrForPost below. These passes can't be dead-pass-culled in PT (GeometryPass is alive via its
        // attachments; every RT/denoise/volumetric pass is alive via its external-resource writes; see
        // CullDeadPasses), so PT must skip REGISTERING them. Only Deform + the lighting-set populate + TLAS +
        // PathTrace + the post chain run in PT. ClusterBuild/LightAssign stay (cheap, keep lightDescSet valid).
        // The TLAS-ready term makes ptEnabled imply ptActive below: a cold boot with PT pre-enabled renders
        // one real-time frame (which builds the TLAS) before PT takes over; never a black frame / invalid
        // geoOutput for the !ptActive overlays.

        // Real-time geometry inputs, hoisted so the post chain + overlays can reference them; produced only
        // on the real-time path (PT traces its own primary rays, so it needs none of these).
        RG::ResourceHandle shadowHandles[k_ShadowCascadeCount]{};
        ShadowCascadeRefs shadowOutputs;
        GraphTextureRef surfaceDepth{}, motionVectors{};
        GraphTextureRef normalOutput, roughnessOutput, materialOutput;
        SlimGBufferOutput  slimGB{};
        if (!ptEnabled)
        {
            hIndirectBuf = cameraVisible.indirect.handle;

            // Z-prepass produces SceneDepth before forward shading. The render graph can schedule it in parallel with the shadow cascades.
            const u32 depthSlot = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex()) % MAX_FRAMES_IN_FLIGHT;
            const std::array<VkDescriptorSet, 6> depthSets{
                m_CurrentViewResources->globalDescriptorSet[depthSlot],
                VulkanContext::Get().GetBindlessSet().GetSet(), MaterialSystem::GetDescriptorSet(depthSlot),
                m_Lighting.GetLightDescSet(depthSlot), BoneMatrixBuffer::GetDescriptorSet(depthSlot),
                m_Geometry.GetObjectSSBODescSet(depthSlot)};
            // CSM is also required by volumetric scatter in either shadow mode.
            const CsmParameters csmParams{visibilityParams.cullCascades};
            const auto csmNative = m_Lighting.PrepareCsmBindings(depthSets,
                view.captureRequested && s.GetFrameDebugger().state == DebuggerState::CaptureRequested);
            const CsmBindingRef csmNativeRef{&csmNative};
            const std::array csmResources{RenderInputBinding::Present(CsmResources::Parameters, csmParams),
                RenderInputBinding::Present(CsmResources::Bindings, csmNativeRef),
                csmParams.enabled ? RenderInputBinding::Present(RenderResources::CascadeVisibleDraws, cascadeVisible)
                    : RenderInputBinding::Absent(RenderResources::CascadeVisibleDraws)};
            const std::array csmCapabilities{&DeformationResources::DeformedGeometry};
            FrameRenderInputs csmFrame;
            csmFrame.renderFrameIndex = Renderer::GetFrameData()->GetRenderFrameIndex();
            csmFrame.draws = &s.GetDrawList(); csmFrame.snapshot = &s.GetActiveSnapshot();
            csmFrame.resources = csmResources; csmFrame.capabilities = csmCapabilities;
            ViewRenderInputs csmView;
            csmView.id = view.id;
            ShadowCascadeRefs csmOutput;
            const std::array csmExports{RenderOutputBinding::Capture(RenderResources::ShadowCascades, csmOutput)};
            const auto csmBuild = m_CsmComposition->Build(rg, csmFrame, csmView, s.GetFrameAllocator(), csmExports);
            if (!csmBuild.success)
            {
                for (const auto& diagnostic : csmBuild.diagnostics)
                    LH_LOG(Renderer, error, "CSM composition: {}", diagnostic.message);
                return false;
            }
            shadowOutputs = csmOutput;
            for (u32 i = 0; i < k_ShadowCascadeCount; ++i)
                shadowHandles[i] = csmOutput.cascades[i].handle;
            const auto depthNative = m_Geometry.PrepareDepthPrepassBindings(depthSets,
                view.captureRequested && s.GetFrameDebugger().state == DebuggerState::CaptureRequested);
            const auto slimNative = m_Geometry.PrepareSlimGBufferBindings(depthSets,
                view.captureRequested && s.GetFrameDebugger().state == DebuggerState::CaptureRequested);
            const SlimGBufferBindingRef slimNativeRef{&slimNative};
            const DepthPrepassBindingRef depthNativeRef{&depthNative};
            const auto depthTarget = m_Geometry.ImportDepthTarget(rg, *view.targets->GetSceneDepth());
            const auto normalTarget = m_Geometry.ImportSlimTarget(rg, *view.targets->GetSlimNormal(), "SlimNormal", RG::TextureFormat::RG16_Float);
            const auto roughnessTarget = m_Geometry.ImportSlimTarget(rg, *view.targets->GetSlimRoughness(), "SlimRoughness", RG::TextureFormat::R8_Unorm);
            const auto motionTarget = m_Geometry.ImportSlimTarget(rg, *view.targets->GetSlimMotion(), "SlimMotion", RG::TextureFormat::RG16_Float);
            const auto materialTarget = m_Geometry.ImportSlimTarget(rg, *view.targets->GetSlimMaterialID(), "SlimMaterialID", RG::TextureFormat::R16_Uint);

            const std::array depthBindings{
                RenderInputBinding::Present(RenderResources::CameraVisibleDraws, cameraVisible),
                RenderInputBinding::Present(DepthPrepassResources::Target, depthTarget),
                RenderInputBinding::Present(DepthPrepassResources::Bindings, depthNativeRef),
                RenderInputBinding::Present(SlimGBufferResources::NormalTarget, normalTarget),
                RenderInputBinding::Present(SlimGBufferResources::RoughnessTarget, roughnessTarget),
                RenderInputBinding::Present(SlimGBufferResources::MotionTarget, motionTarget),
                RenderInputBinding::Present(SlimGBufferResources::MaterialTarget, materialTarget),
                RenderInputBinding::Present(SlimGBufferResources::Bindings, slimNativeRef)};
            const std::array depthCapabilities{&DeformationResources::DeformedGeometry};
            FrameRenderInputs depthFrame;
            depthFrame.renderFrameIndex = Renderer::GetFrameData()->GetRenderFrameIndex();
            depthFrame.draws = &s.GetDrawList();
            depthFrame.snapshot = &s.GetActiveSnapshot();
            depthFrame.capabilities = depthCapabilities; // Published after geometry-preparation succeeds.
            depthFrame.resources = depthBindings;
            ViewRenderInputs depthView;
            depthView.id = view.id;
            depthView.width = m_CurrentViewResources->width;
            depthView.height = m_CurrentViewResources->height;
            GraphTextureRef depthOutput;

            const std::array depthOutputs{
                RenderOutputBinding::Capture(RenderResources::PrepassDepth, depthOutput),
                RenderOutputBinding::Capture(RenderResources::SurfaceDepth, surfaceDepth),
                RenderOutputBinding::Capture(RenderResources::Normal, normalOutput),
                RenderOutputBinding::Capture(RenderResources::Roughness, roughnessOutput),
                RenderOutputBinding::Capture(RenderResources::MotionVectors, motionVectors),
                RenderOutputBinding::Capture(RenderResources::MaterialID, materialOutput)};
            const auto depthBuild = m_SurfacePreparationComposition->Build(rg, depthFrame, depthView,
                s.GetFrameAllocator(), depthOutputs);
            if (!depthBuild.success)
            {
                for (const auto& diagnostic : depthBuild.diagnostics)
                    LH_LOG(Renderer, error, "Surface preparation composition: {}", diagnostic.message);
                return false;
            }
            slimGB = {normalOutput.handle, roughnessOutput.handle, motionVectors.handle, materialOutput.handle};
        }

        // Freeze native cluster resources before recording; retain shared Set 3 bindings.
        Memory::GPUSubRegion lightSSBORegion{};
        u32 pointCount = 0, spotCount = 0;
        if (auto* lighting = SystemRegistry::GetSystem<LightingSystem>())
        {
            const auto& lights = lighting->GetLights();
            lightSSBORegion = m_Lighting.UploadLightSSBO(lights);
            pointCount = static_cast<u32>(lights.points.size());
            spotCount = static_cast<u32>(lights.spots.size());
        }
        const u64 clusterFrameIndex = Renderer::GetFrameData()->GetRenderFrameIndex();
        const u32 clusterSlot = static_cast<u32>(clusterFrameIndex) % MAX_FRAMES_IN_FLIGHT;
        const auto clusterNative = m_Lighting.PrepareClusterBindings(clusterFrameIndex,
            m_CurrentViewResources->clusterBuildDescSet[clusterSlot],
            m_CurrentViewResources->lightAssignDescSet[clusterSlot], view.camera,
            m_CurrentViewResources->width, m_CurrentViewResources->height, lightSSBORegion, pointCount, spotCount);
        const ClusterBindingRef clusterBinding{&clusterNative};
        GraphBufferRef uploadedLights;
        if (lightSSBORegion.buffer)
            uploadedLights = m_Lighting.ImportLightingBuffer(rg, "LightSSBO", lightSSBORegion);
        const std::array clusterResources{RenderInputBinding::Present(ClusterResources::Bindings, clusterBinding),
            uploadedLights.handle.IsValid() ? RenderInputBinding::Present(ClusterResources::UploadedLights, uploadedLights)
                : RenderInputBinding::Absent(ClusterResources::UploadedLights)};
        FrameRenderInputs clusterFrame;
        clusterFrame.renderFrameIndex = clusterFrameIndex; clusterFrame.resources = clusterResources;
        ViewRenderInputs clusterView;
        clusterView.id = view.id; clusterView.width = m_CurrentViewResources->width;
        clusterView.height = m_CurrentViewResources->height;
        GraphBufferRef lightData, clusterGrid, lightIndices;
        const std::array clusterOutputs{RenderOutputBinding::Capture(RenderResources::LightData, lightData),
            RenderOutputBinding::Capture(RenderResources::ClusterGrid, clusterGrid),
            RenderOutputBinding::Capture(RenderResources::LightIndices, lightIndices)};
        const auto clusterBuild = m_ClusterComposition->Build(rg, clusterFrame, clusterView, s.GetFrameAllocator(), clusterOutputs);
        if (!clusterBuild.success)
        {
            for (const auto& diagnostic : clusterBuild.diagnostics)
                LH_LOG(Renderer, error, "Clustered lighting composition: {}", diagnostic.message);
            return false;
        }
        const Memory::GPUSubRegion clusterGridRegion = clusterGrid.binding.slice ? *clusterGrid.binding.slice : Memory::GPUSubRegion{};
        const Memory::GPUSubRegion lightIndexRegion = lightIndices.binding.slice ? *lightIndices.binding.slice : Memory::GPUSubRegion{};
        m_Lighting.WriteSet3PerView(lightSSBORegion, clusterGridRegion, lightIndexRegion);
        // RT acceleration structures: per-frame skinning + skinned BLAS refit + TLAS build, on AsyncCompute.
        // Built BEFORE the volumetric chain so the inject-scatter pass's RT fog-shadow rayQuery reads a BUILT TLAS;
        // passes execute in registration order on the shared compute primary; the inline AS barrier gives memory visibility,
        // not execution ordering. Multi-view guard inside RtSubsystem short-circuits the second view (TLAS is scene-global).
        const bool runRtShadows = (m_Global.GetShadowParams().mode == ShadowingMode::RtShadows)
                               && m_Global.GetShadowParams().castShadows;
        // Per-view fog toggle: also gates the volumetric term in needTlas, so a fog-off view doesn't
        // build a TLAS the (then-unregistered) scatter pass would never read.
        const bool volumetricEnabled = view.camera.enableVolumetricFog;
        // Build the TLAS whenever ANY RT consumer needs it: RT shadows / ReSTIR DI/GI / PathTrace /
        // reflections / volumetric RT fog shadows. The RT sun-shadow trace below stays runRtShadows-only.
        const bool needTlas = runRtShadows || m_Restir.IsEnabled() || m_RestirGi.IsEnabled()
                            || m_PathTrace.IsEnabled() || m_Reflections.IsEnabled()
                            || (volumetricEnabled && m_Volumetric.IsRtShadowsEnabled());
        if (needTlas)
            m_Rt.AddTlasBuildPass(rg);

        // Native CPU preparation freezes the graph's fog bindings. RT table pairing remains
        // an explicit native compatibility dependency until the scene provider migrates.
        const u32 fogFrameAbs = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex());
        Memory::GPUSubRegion fogVolumeRegion{};
        if (volumetricEnabled && !ptEnabled)
            if (auto* lighting = SystemRegistry::GetSystem<LightingSystem>())
                fogVolumeRegion = m_Volumetric.UploadFogVolumeSSBO(lighting->GetFogVolumes());
        const auto fogNative = m_Volumetric.PrepareComputeBindings(*m_CurrentViewResources->fog, fogFrameAbs,
            view.camera, m_CurrentViewResources->globalDescriptorSet[fogFrameAbs % MAX_FRAMES_IN_FLIGHT],
            volumetricEnabled && !ptEnabled, m_Volumetric.IsRtShadowsEnabled(), &m_Rt,
            fogVolumeRegion, lightSSBORegion, clusterGridRegion, lightIndexRegion);
        GraphBufferRef fogVolumes;
        if (fogNative.enabled && fogVolumeRegion.buffer)
            fogVolumes = LightingSubsystem::ImportLightingBuffer(rg, "FogVolumes", fogVolumeRegion);
        const ShadowCascadeRefs fogShadows = shadowOutputs;
        bool haveFogShadows = true;
        for (u32 i = 0; i < k_ShadowCascadeCount; ++i)
        {
            haveFogShadows &= shadowHandles[i].IsValid();
        }
        const FogComputeBindingRef fogBinding{&fogNative};
        const std::array fogResources{RenderInputBinding::Present(FogResources::Bindings, fogBinding),
            fogVolumes.handle.IsValid() ? RenderInputBinding::Present(FogResources::Volumes, fogVolumes) : RenderInputBinding::Absent(FogResources::Volumes),
            lightData.handle.IsValid() ? RenderInputBinding::Present(RenderResources::LightData, lightData) : RenderInputBinding::Absent(RenderResources::LightData),
            clusterGrid.handle.IsValid() ? RenderInputBinding::Present(RenderResources::ClusterGrid, clusterGrid) : RenderInputBinding::Absent(RenderResources::ClusterGrid),
            lightIndices.handle.IsValid() ? RenderInputBinding::Present(RenderResources::LightIndices, lightIndices) : RenderInputBinding::Absent(RenderResources::LightIndices),
            haveFogShadows ? RenderInputBinding::Present(RenderResources::ShadowCascades, fogShadows) : RenderInputBinding::Absent(RenderResources::ShadowCascades)};
        FrameRenderInputs fogFrame; fogFrame.renderFrameIndex = fogFrameAbs; fogFrame.resources = fogResources;
        ViewRenderInputs fogView; fogView.id = view.id; fogView.camera = &view.camera;
        fogView.width = m_CurrentViewResources->width; fogView.height = m_CurrentViewResources->height;
        GraphTextureRef fogDensity, fogIntegrated, fogResolved;
        const std::array fogOutputs{RenderOutputBinding::Capture(RenderResources::FogDensity, fogDensity),
            RenderOutputBinding::Capture(FogResources::IntegratedScatter, fogIntegrated),
            RenderOutputBinding::Capture(RenderResources::ResolvedFog, fogResolved)};
        const auto fogBuild = m_FogComputeComposition->Build(rg, fogFrame, fogView, s.GetFrameAllocator(), fogOutputs);
        if (!fogBuild.success)
        {
            for (const auto& diagnostic : fogBuild.diagnostics)
                LH_LOG(Renderer, error, "Fog compute composition: {}", diagnostic.message);
            return false;
        }
        const RG::ResourceHandle volResolvedHandle = fogResolved.handle;
        // Path-traced reference mode: a megakernel that bypasses the entire raster + ReSTIR chain. When active,
        // its HDR output (ptColor) feeds the post chain in place of the raster sceneColor; every raster/RT-GI
        // pass below produces handles nothing consumes, so the RG dead-pass culls them. AsyncCompute, after the
        // TLAS build (which the needTlas gate above keeps alive).
        const bool usePathTrace = ptEnabled;
        RG::ResourceHandle ptColorHandle{};
        if (usePathTrace)
            ptColorHandle = m_PathTrace.AddPasses(rg);
        const bool ptActive = usePathTrace && ptColorHandle.IsValid();

        // RT sun-shadow trace: per-view (each view's depth/camera/mask differ), so this runs on every view's RG.
        // Writes per-view R8 mask, consumed by GeometryPass via Read(handle). AsyncCompute pass overlaps with the
        // GTAO chain below. Gated on RT mode + CastShadows; CSM mode (or CastShadows=false) returns invalid handle
        // and GeometryPass skips the Read. Threads surfaceDepth.handle + slimGB.normal so RG transitions them from
        // DSA/COLOR_ATTACHMENT to SHADER_READ_ONLY_OPTIMAL ahead of the raygen sample (descriptor declared that layout).
        RG::ResourceHandle rtShadowMaskHandle{};
        if (runRtShadows && !ptEnabled)
            rtShadowMaskHandle = m_Rt.AddRtSunShadowsPass(rg, surfaceDepth.handle, slimGB.normal);

        // ReSTIR DI: shadowed direct lighting for point lights via per-pixel reservoir RIS + one
        // visibility ray, then a demodulated-irradiance shade. AsyncCompute; reads prepass depth +
        // slim normal, traces the same TLAS the sun-shadow pass uses. Returns an invalid handle when
        // disabled or before the TLAS exists; GeometryPass then skips the Read and pbr.frag's point
        // loop runs instead (the restirParams.x flag gates the consumption).
        RtRestirSubsystem::Outputs restirOut = ptEnabled
            ? RtRestirSubsystem::Outputs{}
            : m_Restir.AddPasses(rg, surfaceDepth.handle, slimGB.normal, slimGB.motion, slimGB.roughness);
        RG::ResourceHandle restirDIHandle = restirOut.di;

        // Denoise the demodulated DI (SVGF; swappable to NRD/RELAX). Transparent filter: consumes the
        // ReSTIR DI handle, returns the denoised handle GeometryPass reads + Set 3 b5 binds. Invalid in
        // (ReSTIR off / pre-TLAS) -> invalid out, and pbr.frag falls back to its own cluster light loop.
        RG::ResourceHandle denoisedDIHandle = m_Denoise->AddPasses(rg, DenoiseInputs{
            restirDIHandle, surfaceDepth.handle, slimGB.normal, slimGB.motion,
            slimGB.roughness, slimGB.materialID, {}, {} });

        // Denoise the demodulated ReSTIR-DI specular (4th SVGF instance, DenoiserChannel::DiSpecular). Surface-
        // motion reproject (direct point-light specular is surface-attached, not a reflection's virtual
        // image). svgfDiSpecDenoised feeds pbr.frag Set 3 b8; restirParams.z gates + scales the composite.
        // Gated on the specular toggle: with it off pbr.frag zeroes restirParams.z and never samples b8, so
        // the whole denoise chain would otherwise run dead (~0.5 ms). Invalid handle -> GeometryPass skips b8.
        RG::ResourceHandle denoisedDiSpecHandle{};
        if (m_System.GetRestirSettings().specular)
            denoisedDiSpecHandle = m_DenoiseDiSpec->AddPasses(rg, DenoiseInputs{
                restirOut.spec, surfaceDepth.handle, slimGB.normal, slimGB.motion,
                slimGB.roughness, slimGB.materialID, {}, {} });
        // Half-res DI: AddPasses returns the half svgfDiHalf / svgfDiSpecHalf handles; bilaterally upscale
        // each into the full svgfDenoised / svgfDiSpecDenoised that GeometryPass / pbr Set 3 b5/b8 consume.
        if (m_System.GetRestirSettings().halfResolution)
        {
            if (denoisedDIHandle.IsValid())
                denoisedDIHandle = m_Restir.AddUpscalePass(rg, denoisedDIHandle, surfaceDepth.handle, slimGB.normal, false);
            if (denoisedDiSpecHandle.IsValid())
                denoisedDiSpecHandle = m_Restir.AddUpscalePass(rg, denoisedDiSpecHandle, surfaceDepth.handle, slimGB.normal, true);
        }

        // ReSTIR GI: 1-bounce indirect diffuse via per-pixel reservoir resampling. Returns the demodulated
        // GI image; restirParams.y gates the remodulation in pbr.frag. Invalid when disabled / no TLAS.
        GraphBufferRef giSpatialReservoir;
        RG::ResourceHandle giDIHandle = ptEnabled
            ? RG::ResourceHandle{}
            : m_RestirGi.AddPasses(rg, surfaceDepth.handle, slimGB.normal, slimGB.motion, &giSpatialReservoir);

        // Denoise the demodulated GI (second SVGF instance, DenoiserChannel::Gi). Same transparent-filter
        // contract as DI: consumes the GI handle, returns the denoised handle GeometryPass reads + Set 3 b6
        // binds. Invalid in -> invalid out (pbr.frag then adds nothing under the .y gate).
        RG::ResourceHandle denoisedGiHandle = m_DenoiseGi->AddPasses(rg, DenoiseInputs{
            giDIHandle, surfaceDepth.handle, slimGB.normal, slimGB.motion,
            slimGB.roughness, slimGB.materialID, {}, {} });
        // Half-res GI: AddPasses returns the half-res svgfGiHalf handle; bilaterally upscale it into the
        // full-res svgfGiDenoised that GeometryPass / pbr Set 3 b6 consume. Full-res mode is a no-op.
        if (denoisedGiHandle.IsValid() && m_System.GetRestirGiSettings().halfResolution)
            denoisedGiHandle = m_RestirGi.AddUpscalePass(rg, denoisedGiHandle, surfaceDepth.handle, slimGB.normal);

        // RT specular reflections: one GGX-VNDF ray/pixel from the slim G-buffer, then
        // a dedicated specular SVGF (3rd instance, DenoiserChannel::Reflections). The DenoiseInputs.motion
        // slot carries slim ROUGHNESS (the spec reproject's b3: it computes the reflection's motion
        // internally via hit-distance virtual reprojection; hitDist rides reflRadiance's alpha).
        // denoisedReflHandle feeds GeometryPass (pbr.frag composites it via Set 3 b7). AsyncCompute,
        // after the TLAS build (needTlas gate includes Reflections).
        RG::ResourceHandle reflHandle = ptEnabled
            ? RG::ResourceHandle{}
            : m_Reflections.AddPasses(rg, surfaceDepth.handle, slimGB.normal, slimGB.roughness);
        RG::ResourceHandle denoisedReflHandle = m_DenoiseRefl->AddPasses(rg, DenoiseInputs{
            reflHandle, surfaceDepth.handle, slimGB.normal, slimGB.roughness,
            slimGB.roughness, slimGB.materialID, {}, {} });
        // Half-res reflections: AddPasses returns the half svgfSpecHalf handle; bilaterally upscale it into
        // the full-res svgfSpecDenoised that pbr.frag Set 3 b7 consumes. Full-res mode is a no-op.
        if (denoisedReflHandle.IsValid() && m_System.GetReflectionsSettings().halfResolution)
            denoisedReflHandle = m_Reflections.AddUpscalePass(rg, denoisedReflHandle, surfaceDepth.handle, slimGB.normal);

        // Legacy depth/geometry bridge shares graph-local references with the compiled
        // feature. Disabled/PT frames publish absent AO and register no GTAO passes.

        const auto* preparedGtao = GetGtaoViewState(view.id);
        const GtaoFrameParameters gtaoParams{
            preparedGtao && preparedGtao->uniformEnabled, !ptEnabled,
            static_cast<u32>(Renderer::GetFrameData()->GetFrameIndex())};
        const std::array gtaoBindings{
            RenderInputBinding::Present(RenderResources::SurfaceDepth, surfaceDepth),
            RenderInputBinding::Present(GtaoResources::Parameters, gtaoParams)};
        FrameRenderInputs featureFrame;
        featureFrame.renderFrameIndex = Renderer::GetFrameData()->GetRenderFrameIndex();
        ViewRenderInputs featureView;
        featureView.id = view.id;
        featureView.resourceGeneration = m_CurrentViewResources->generation;
        featureView.width = m_CurrentViewResources->width;
        featureView.height = m_CurrentViewResources->height;
        featureView.camera = &view.camera;
        featureView.resources = gtaoBindings;
        GraphTextureRef aoOutput;
        const std::array gtaoOutputs{RenderOutputBinding::Capture(RenderResources::AmbientOcclusion, aoOutput)};
        const auto gtaoBuild = m_GtaoPipeline->Build(rg, featureFrame, featureView,
            m_System.GetFrameAllocator(), gtaoOutputs);
        if (!gtaoBuild.success)
        {
            for (const auto& diagnostic : gtaoBuild.diagnostics)
                LH_LOG(Renderer, error, "GTAO composition: {}", diagnostic.message);
            return false; // Never compile/record a graph with invalid contracts.
        }

        // Real-time lit chain (geometry -> skybox -> fog composite -> transparent -> TAA). Skipped in PT; the
        // megakernel output drives the post chain via hdrForPost below. geoOutput/selection outputs/resolvedHdr hoisted
        // for the overlays + post chain; default-invalid in PT (the overlays that read them are !ptActive too).
        GeometryOutput      geoOutput{};
        GraphTextureRef selectionMask, selectionDepth;
        GraphTextureRef resolvedHdr;
        if (!ptEnabled)
        {
            const u32 forwardSlot = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex()) % MAX_FRAMES_IN_FLIGHT;
            const std::array<VkDescriptorSet, 6> forwardSets{m_CurrentViewResources->globalDescriptorSet[forwardSlot],
                VulkanContext::Get().GetBindlessSet().GetSet(), MaterialSystem::GetDescriptorSet(forwardSlot),
                m_CurrentViewResources->lightDescSet[forwardSlot], BoneMatrixBuffer::GetDescriptorSet(forwardSlot),
                m_Geometry.GetObjectSSBODescSet(forwardSlot)};
            const auto forwardNative = m_Geometry.PrepareForwardOpaqueBindings(forwardSets,
                s.GetShadeMode() == ShadeMode::Wireframe, s.GetShadeMode() == ShadeMode::WireframeShaded,
                view.captureRequested && s.GetFrameDebugger().state == DebuggerState::CaptureRequested,
                cameraVisible, s.GetDrawList(), s.GetActiveSnapshot());
            const ForwardOpaqueBindingRef forwardBinding{&forwardNative};
            const auto colorTarget = m_Geometry.ImportForwardTarget(rg, *view.targets->GetSceneColor(),
                "SceneColor", RG::TextureFormat::RGBA16_Float, RG::ResourceState::ShaderResource);
            const auto pickingTarget = m_Geometry.ImportForwardTarget(rg, *view.targets->GetEntityIDBuffer(),
                "EntityID", RG::TextureFormat::R32_Uint, RG::ResourceState::Undefined);
            // Transitional RT references are barrier reads; existing native Set 3 owns their descriptors.
            const GraphTextureRef sunSignal{rtShadowMaskHandle, {}}, diSignal{denoisedDIHandle, {}},
                giSignal{denoisedGiHandle, {}}, reflectionSignal{denoisedReflHandle, {}}, specularSignal{denoisedDiSpecHandle, {}};
            const auto optionalImage = [](auto key, const GraphTextureRef& value) {
                return value.handle.IsValid() ? RenderInputBinding::Present(key, value) : RenderInputBinding::Absent(key);
            };
            const auto optionalBuffer = [](auto key, const GraphBufferRef& value) {
                return value.handle.IsValid() ? RenderInputBinding::Present(key, value) : RenderInputBinding::Absent(key);
            };
            const std::array forwardResources{RenderInputBinding::Present(ForwardOpaqueResources::Bindings, forwardBinding),
                RenderInputBinding::Present(ForwardOpaqueResources::ColorTarget, colorTarget),
                RenderInputBinding::Present(ForwardOpaqueResources::PickingTarget, pickingTarget),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, surfaceDepth),
                RenderInputBinding::Present(RenderResources::CameraVisibleDraws, cameraVisible),
                shadowOutputs.cascades[0].handle.IsValid() ? RenderInputBinding::Present(RenderResources::ShadowCascades, shadowOutputs)
                    : RenderInputBinding::Absent(RenderResources::ShadowCascades),
                optionalImage(RenderResources::AmbientOcclusion, aoOutput),
                optionalBuffer(RenderResources::LightData, lightData), optionalBuffer(RenderResources::ClusterGrid, clusterGrid),
                optionalBuffer(RenderResources::LightIndices, lightIndices),
                optionalImage(ForwardCompatibilityResources::SunShadowMask, sunSignal),
                optionalImage(ForwardCompatibilityResources::DenoisedDiffuseDI, diSignal),
                optionalImage(ForwardCompatibilityResources::DenoisedDiffuseGI, giSignal),
                optionalImage(ForwardCompatibilityResources::DenoisedReflectionRadiance, reflectionSignal),
                optionalImage(ForwardCompatibilityResources::DenoisedSpecularDI, specularSignal)};
            const std::array forwardCapabilities{&DeformationResources::DeformedGeometry};
            FrameRenderInputs forwardFrame;
            forwardFrame.resources = forwardResources; forwardFrame.capabilities = forwardCapabilities;
            ViewRenderInputs forwardView;
            forwardView.id = view.id; forwardView.width = m_CurrentViewResources->width; forwardView.height = m_CurrentViewResources->height;
            GraphTextureRef opaqueOutput, litOutput, pickingOutput;
            const std::array forwardExports{RenderOutputBinding::Capture(RenderResources::OpaqueHDR, opaqueOutput),
                RenderOutputBinding::Capture(RenderResources::LitDepth, litOutput),
                RenderOutputBinding::Capture(RenderResources::OpaquePickingIDs, pickingOutput)};
            const auto forwardBuild = m_ForwardComposition->Build(rg, forwardFrame, forwardView, s.GetFrameAllocator(), forwardExports);
            if (!forwardBuild.success)
            {
                for (const auto& diagnostic : forwardBuild.diagnostics)
                    LH_LOG(Renderer, error, "Forward opaque composition: {}", diagnostic.message);
                return false;
            }
            geoOutput = {opaqueOutput.handle, litOutput.handle, pickingOutput.handle};
            const std::array<VkDescriptorSet, 5> selectionSets{forwardSets[0], forwardSets[1], forwardSets[2],
                forwardSets[3], forwardSets[4]};
            const auto selectionNative = m_EditorOverlays.PrepareSelectionMaskBindings(m_CurrentViewResources->overlays,
                selectionSets, view.camera, m_CurrentViewResources->currentJitter,
                s.GetDrawList(), s.GetActiveSnapshot(), view.drawSelectionOutline);
            const SelectionMaskBindingRef selectionBinding{&selectionNative};
            const std::array selectionResources{RenderInputBinding::Present(SelectionMaskResources::Bindings, selectionBinding)};
            FrameRenderInputs selectionFrame; selectionFrame.renderFrameIndex = Renderer::GetFrameData()->GetRenderFrameIndex();
            selectionFrame.resources = selectionResources;
            ViewRenderInputs selectionView; selectionView.id = view.id;
            selectionView.width = view.targets->GetSceneColor()->GetWidth(); selectionView.height = view.targets->GetSceneColor()->GetHeight();

            const std::array selectionExports{RenderOutputBinding::Capture(RenderResources::SelectionMask, selectionMask),
                RenderOutputBinding::Capture(RenderResources::SelectionDepth, selectionDepth)};
            const auto selectionBuild = m_SelectionMaskComposition->Build(rg, selectionFrame, selectionView,
                s.GetFrameAllocator(), selectionExports);
            if (!selectionBuild.success)
            {
                for (const auto& diagnostic : selectionBuild.diagnostics)
                    LH_LOG(Renderer, error, "SelectionMask composition: {}", diagnostic.message);
                return false;
            }

            const u32 skySlot = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex()) % MAX_FRAMES_IN_FLIGHT;
            const std::array<VkDescriptorSet, 5> skySets{m_CurrentViewResources->globalDescriptorSet[skySlot],
                VulkanContext::Get().GetBindlessSet().GetSet(), MaterialSystem::GetDescriptorSet(skySlot),
                m_CurrentViewResources->lightDescSet[skySlot], BoneMatrixBuffer::GetDescriptorSet(skySlot)};
            const auto skyNative = m_Lighting.PrepareSkyBindings(skySets);
            const SkyBindingRef skyBinding{&skyNative};
            const GraphTextureRef opaqueHdr = opaqueOutput;
            const GraphTextureRef litDepth = litOutput;
            const std::array skyResources{RenderInputBinding::Present(RenderResources::OpaqueHDR, opaqueHdr),
                RenderInputBinding::Present(RenderResources::LitDepth, litDepth),
                RenderInputBinding::Present(SkyResources::Bindings, skyBinding)};
            FrameRenderInputs skyFrame;
            skyFrame.renderFrameIndex = Renderer::GetFrameData()->GetRenderFrameIndex(); skyFrame.resources = skyResources;
            ViewRenderInputs skyView;
            skyView.id = view.id; skyView.width = m_CurrentViewResources->width; skyView.height = m_CurrentViewResources->height;
            GraphTextureRef skyOutput;
            const std::array skyExports{RenderOutputBinding::Capture(RenderResources::SkyHDR, skyOutput)};
            const auto skyBuild = m_SkyComposition->Build(rg, skyFrame, skyView, s.GetFrameAllocator(), skyExports);
            if (!skyBuild.success)
            {
                for (const auto& diagnostic : skyBuild.diagnostics)
                    LH_LOG(Renderer, error, "Sky composition: {}", diagnostic.message);
                return false;
            }
            // Volumetric composite: blends fog into sceneColor (alpha-blend) BEFORE bloom so bright
            // in-scattered fog can bloom + the grid overlays unfogged lines. Off -> uses skyboxColor unchanged.
            const auto fogCompositeNative = m_Volumetric.PrepareCompositeBindings(*m_CurrentViewResources->fog,
                fogFrameAbs, view.camera, m_CurrentViewResources->globalDescriptorSet[skySlot], volResolvedHandle.IsValid());
            const FogCompositeBindingRef fogCompositeBinding{&fogCompositeNative};
            const std::array fogCompositeResources{RenderInputBinding::Present(RenderResources::SkyHDR, skyOutput),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, surfaceDepth),
                RenderInputBinding::Present(FogCompositeResources::Bindings, fogCompositeBinding),
                fogResolved.handle.IsValid() ? RenderInputBinding::Present(RenderResources::ResolvedFog, fogResolved)
                    : RenderInputBinding::Absent(RenderResources::ResolvedFog)};
            FrameRenderInputs fogCompositeFrame;
            fogCompositeFrame.renderFrameIndex = skyFrame.renderFrameIndex; fogCompositeFrame.resources = fogCompositeResources;
            GraphTextureRef foggedOutput;
            const std::array fogCompositeExports{RenderOutputBinding::Capture(RenderResources::FoggedHDR, foggedOutput)};
            const auto fogCompositeBuild = m_FogCompositeComposition->Build(rg, fogCompositeFrame, skyView,
                s.GetFrameAllocator(), fogCompositeExports);
            if (!fogCompositeBuild.success)
            {
                for (const auto& diagnostic : fogCompositeBuild.diagnostics)
                    LH_LOG(Renderer, error, "Fog composite composition: {}", diagnostic.message);
                return false;
            }
            const RG::ResourceHandle fogColor = foggedOutput.handle;
            // Snapshot the pre-transparent scene (opaque + fog) into the per-view refraction backdrop so glass
            // can sample the refracted background. RG orders the copy after the composite (reads fogColor as
            // TransferSrc) and before the transparent pass's Set 6 b3 sample (declared Read there). Skipped
            // when the transparent bucket is empty (matches transparency preparation).
            const auto backdropNative = m_Transparency.PrepareBackdropBindings(m_CurrentViewResources->transparency->refractionBackdrop,
                !m_System.GetDrawList().transparent.empty());
            const RefractionBackdropBindingRef backdropBinding{&backdropNative};
            const std::array backdropResources{RenderInputBinding::Present(RenderResources::FoggedHDR, foggedOutput),
                RenderInputBinding::Present(RefractionResources::Bindings, backdropBinding)};
            FrameRenderInputs backdropFrame;
            backdropFrame.renderFrameIndex = skyFrame.renderFrameIndex; backdropFrame.resources = backdropResources;
            GraphTextureRef backdropOutput;
            const std::array backdropExports{RenderOutputBinding::Capture(RenderResources::RefractionBackdrop, backdropOutput)};
            const auto backdropBuild = m_RefractionComposition->Build(rg, backdropFrame, skyView,
                s.GetFrameAllocator(), backdropExports);
            if (!backdropBuild.success)
            {
                for (const auto& diagnostic : backdropBuild.diagnostics)
                    LH_LOG(Renderer, error, "Refraction backdrop composition: {}", diagnostic.message);
                return false;
            }
            const RG::ResourceHandle backdropHandle = backdropOutput.handle;
            // Transparent tier: after the fog composite so glass blends over the fogged background (its own
            // fog is per-fragment at the glass depth, sampled from the resolved atlas inside pbr_transparent.frag).
            GraphTextureRef transparentStage = foggedOutput;
            if (m_CurrentViewResources)
            {
                const u32 frameAbsT = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex());
                m_Transparency.WritePerFrame(*m_CurrentViewResources->transparency, m_CurrentViewResources->fog,
                    m_Volumetric.GetSampler(), frameAbsT);
                {
                    std::array<VkDescriptorSet, 7> transparencySets;
                    std::copy(forwardSets.begin(), forwardSets.end(), transparencySets.begin());
                    transparencySets[6] = m_CurrentViewResources->transparency->transparentDescSet[frameAbsT % MAX_FRAMES_IN_FLIGHT];
                    const auto transparencyNative = m_Transparency.PrepareTransparencyBindings(m_Geometry, transparencySets,
                        s.GetShadeMode() == ShadeMode::Wireframe,
                        view.captureRequested && s.GetFrameDebugger().state == DebuggerState::CaptureRequested,
                        view.camera.view, cameraVisible, s.GetDrawList(), s.GetActiveSnapshot(),
                        fogResolved.binding, backdropOutput.binding, &m_Rt,
                        s.GetTransparencySettings().mode == TransparencyMode::OIT,
                        *m_CurrentViewResources->transparency, s.GetTransparencySettings().maxResolveK);
                    const TransparencyBindingRef transparencyBinding{&transparencyNative};
                    const auto optionalImage = [](auto key, const GraphTextureRef& value) {
                        return value.handle.IsValid() ? RenderInputBinding::Present(key, value) : RenderInputBinding::Absent(key);
                    };
                    const std::array transparencyResources{RenderInputBinding::Present(TransparencyResources::Bindings, transparencyBinding),
                        RenderInputBinding::Present(RenderResources::FoggedHDR, foggedOutput),
                        RenderInputBinding::Present(RenderResources::LitDepth, litOutput),
                        RenderInputBinding::Present(RenderResources::OpaquePickingIDs, pickingOutput),
                        RenderInputBinding::Present(RenderResources::CameraVisibleDraws, cameraVisible),
                        optionalImage(RenderResources::ResolvedFog, fogResolved), optionalImage(RenderResources::RefractionBackdrop, backdropOutput),
                        optionalBuffer(RenderResources::LightData, lightData), optionalBuffer(RenderResources::ClusterGrid, clusterGrid),
                        optionalBuffer(RenderResources::LightIndices, lightIndices)};
                    FrameRenderInputs transparencyFrame; transparencyFrame.renderFrameIndex = skyFrame.renderFrameIndex;
                    transparencyFrame.resources = transparencyResources; transparencyFrame.capabilities = forwardCapabilities;
                    GraphTextureRef transparentOutput, finalPicking, transparentDepth;
                    const std::array transparencyExports{RenderOutputBinding::Capture(RenderResources::TransparentHDR, transparentOutput),
                        RenderOutputBinding::Capture(RenderResources::FinalPickingIDs, finalPicking),
                        RenderOutputBinding::Capture(TransparencyResources::Depth, transparentDepth)};
                    const auto transparencyBuild = m_TransparencyComposition->Build(rg, transparencyFrame, skyView,
                        s.GetFrameAllocator(), transparencyExports);
                    if (!transparencyBuild.success)
                    {
                        for (const auto& diagnostic : transparencyBuild.diagnostics)
                            LH_LOG(Renderer, error, "Transparency composition: {}", diagnostic.message);
                        return false;
                    }
                    transparentStage = transparentOutput;
                    geoOutput.entityID = finalPicking.handle; geoOutput.depth = transparentDepth.handle;
                }
            }
            const auto& pps = s.GetPostProcessSettings();
            const auto taaNative = m_PostProcess.PrepareTaaBindings(m_CurrentViewResources->taa,
                skyFrame.renderFrameIndex, m_CurrentViewResources->generation,
                m_Global.GetCachedSkyReproj(), pps.taaTemporalAlpha, pps.taaEnabled);
            const TaaBindingRef taaBinding{&taaNative};
            const GraphTextureRef taaDepth{geoOutput.depth, surfaceDepth.binding};
            const std::array taaResources{RenderInputBinding::Present(RenderResources::TransparentHDR, transparentStage),
                RenderInputBinding::Present(RenderResources::MotionVectors, motionVectors),
                RenderInputBinding::Present(RenderResources::LitDepth, taaDepth),
                RenderInputBinding::Present(TaaResources::Bindings, taaBinding)};
            FrameRenderInputs taaFrame; taaFrame.renderFrameIndex = skyFrame.renderFrameIndex;
            taaFrame.resources = taaResources;
            const std::array taaExports{RenderOutputBinding::Capture(RenderResources::ResolvedHDR, resolvedHdr)};
            const auto taaBuild = m_TaaComposition->Build(rg, taaFrame, skyView, s.GetFrameAllocator(), taaExports);
            if (!taaBuild.success)
            {
                for (const auto& diagnostic : taaBuild.diagnostics)
                    LH_LOG(Renderer, error, "TAA composition: {}", diagnostic.message);
                return false;
            }
        }
        // Native downstream descriptors follow the actual selected stage, including cold TAA pass-through.
        const TextureBindingRef postSource = ptActive ? TextureBindingRef{m_CurrentViewResources->ptColor.get()}
            : resolvedHdr.binding;
        // HDR source for the post chain: the PT megakernel output replaces the raster sceneColor when PT
        // is active (the realtime chain is not registered). Grid is editor-overlay-only -> off in PT.
        RG::ResourceHandle hdrForPost  = ptActive ? ptColorHandle : resolvedHdr.handle;
        // Resolve the active shade mode once (PT forces Lit). Hoisted here so the bloom gate and the slim-viz dispatch below share it.
        const ShadeMode shadeMode = ptActive ? ShadeMode::Lit : m_System.GetShadeMode();
        // Bloom consumes the selected pre-grid HDR stage; disabled contributions publish absence.
        const auto& bloomSettings = m_System.GetPostProcessSettings();
        const auto bloomNative = m_PostProcess.PrepareBloomBindings(m_CurrentViewResources->bloom, postSource,
            Renderer::GetFrameData()->GetRenderFrameIndex(), bloomSettings.bloomThreshold, bloomSettings.bloomRadius,
            bloomSettings.bloomStrength > 0.0f && shadeMode == ShadeMode::Lit);
        const BloomBindingRef bloomBinding{&bloomNative};
        const GraphTextureRef bloomSource{hdrForPost, postSource};
        const std::array bloomResources{RenderInputBinding::Present(RenderResources::ResolvedHDR, bloomSource),
            RenderInputBinding::Present(BloomResources::Bindings, bloomBinding)};
        FrameRenderInputs bloomFrame; bloomFrame.renderFrameIndex = Renderer::GetFrameData()->GetRenderFrameIndex();
        bloomFrame.resources = bloomResources;
        ViewRenderInputs bloomView; bloomView.id = view.id;
        bloomView.width = view.targets->GetSceneColor()->GetWidth(); bloomView.height = view.targets->GetSceneColor()->GetHeight();
        GraphTextureRef bloomOutput;
        const std::array bloomExports{RenderOutputBinding::Capture(RenderResources::BloomOutput, bloomOutput)};
        const auto bloomBuild = m_BloomComposition->Build(rg, bloomFrame, bloomView, s.GetFrameAllocator(), bloomExports);
        if (!bloomBuild.success)
        {
            for (const auto& diagnostic : bloomBuild.diagnostics)
                LH_LOG(Renderer, error, "Bloom composition: {}", diagnostic.message);
            return false;
        }

        const auto gridNative = m_EditorOverlays.PrepareGridBindings(m_CurrentViewResources->overlays,
            s.GetCameraParams(), m_CurrentViewResources->currentJitter, bloomFrame.renderFrameIndex, view.drawGrid && !ptActive);
        const GridBindingRef gridBinding{&gridNative};
        const GraphTextureRef gridDepth{geoOutput.depth, surfaceDepth.binding};
        const std::array gridResources{RenderInputBinding::Present(RenderResources::ResolvedHDR, bloomSource),
            RenderInputBinding::Present(GridResources::Bindings, gridBinding),
            !ptActive ? RenderInputBinding::Present(RenderResources::LitDepth, gridDepth)
                : RenderInputBinding::Absent(RenderResources::LitDepth),
            bloomOutput.handle.IsValid() ? RenderInputBinding::Present(RenderResources::BloomOutput, bloomOutput)
                : RenderInputBinding::Absent(RenderResources::BloomOutput)};
        FrameRenderInputs gridFrame; gridFrame.renderFrameIndex = bloomFrame.renderFrameIndex; gridFrame.resources = gridResources;
        GraphTextureRef gridStage;
        const std::array gridExports{RenderOutputBinding::Capture(RenderResources::GridHDR, gridStage)};
        const auto gridBuild = m_GridComposition->Build(rg, gridFrame, bloomView, s.GetFrameAllocator(), gridExports);
        if (!gridBuild.success)
        {
            for (const auto& diagnostic : gridBuild.diagnostics)
                LH_LOG(Renderer, error, "Grid composition: {}", diagnostic.message);
            return false;
        }
        const auto compositeParameters = MakeCompositeUniforms(bloomSettings,
            IsDataDebugMode(s.GetRenderMode() == RenderMode::PathTrace ? ShadeMode::Lit : shadeMode),
            bloomOutput.handle.IsValid(), Time::GetTime());
        const auto compositeNative = m_PostProcess.PrepareCompositeBindings(m_CurrentViewResources->composite,
            postSource, bloomOutput.binding, view.targets->GetLDROutput(), bloomFrame.renderFrameIndex, compositeParameters);
        const CompositeBindingRef compositeBinding{&compositeNative};

        const std::array compositeResources{RenderInputBinding::Present(RenderResources::GridHDR, gridStage),
            bloomOutput.handle.IsValid() ? RenderInputBinding::Present(RenderResources::BloomOutput, bloomOutput)
                : RenderInputBinding::Absent(RenderResources::BloomOutput),
            RenderInputBinding::Present(CompositeResources::Bindings, compositeBinding)};
        FrameRenderInputs compositeFrame; compositeFrame.renderFrameIndex = bloomFrame.renderFrameIndex;
        compositeFrame.resources = compositeResources;
        GraphTextureRef tonemappedLdr;
        const std::array compositeExports{RenderOutputBinding::Capture(RenderResources::TonemappedLDR, tonemappedLdr)};
        const auto compositeBuild = m_CompositeComposition->Build(rg, compositeFrame, bloomView,
            s.GetFrameAllocator(), compositeExports);
        if (!compositeBuild.success)
        {
            for (const auto& diagnostic : compositeBuild.diagnostics)
                LH_LOG(Renderer, error, "Composite composition: {}", diagnostic.message);
            return false;
        }
        RG::ResourceHandle ldrOutput = tonemappedLdr.handle;

        // Slim G-buffer ShadeMode toggles overwrite LDROutput with a decoded attachment. Mode index = enum offset
        // from ShadeMode::SlimNormal (0..3). Motion scale hardcoded: the frame-debugger panel exposes a slider for
        // per-capture tuning; live viz uses a sensible default matching the existing thumbnail UX. PT mode forces
        // Lit (the debug-viz blits read the culled G-buffer / cluster / reservoir state, meaningless over the PT
        // image). shadeMode was resolved above (hoisted for the bloom gate).
        const bool slimVizEnabled = !ptEnabled && shadeMode >= ShadeMode::SlimNormal && shadeMode <= ShadeMode::SlimMaterialID;
        const u32 slimMode = slimVizEnabled ? static_cast<u32>(shadeMode) - static_cast<u32>(ShadeMode::SlimNormal) : 0;
        const auto slimVizNative = m_PostProcess.PrepareSlimVizBindings(m_CurrentViewResources->slimViz, slimMode, 20.0f, slimVizEnabled);
        const SlimVizBindingRef slimVizBinding{&slimVizNative};
        const auto clusterVizNative = m_Lighting.PrepareClusterVizBindings(m_CurrentViewResources->clusterViz,
            m_CurrentViewResources->lightDescSet[bloomFrame.renderFrameIndex % MAX_FRAMES_IN_FLIGHT],
            clusterGridRegion, bloomView.width, bloomView.height, view.camera.nearZ, view.camera.farZ,
            !ptEnabled && shadeMode == ShadeMode::ClustersDensity);
        const ClusterVizBindingRef clusterVizBinding{&clusterVizNative};
        const bool fogVizEnabled = !ptEnabled && (shadeMode == ShadeMode::VolumetricDensity || shadeMode == ShadeMode::VolumetricInScatter);
        const u32 fogVizMode = shadeMode == ShadeMode::VolumetricInScatter ? 1u : 0u;
        const auto& fogVizSettings = s.GetVolumetricSettings();
        const auto fogVizNative = m_Volumetric.PrepareVizBindings(m_CurrentViewResources->fog, bloomFrame.renderFrameIndex,
            m_CurrentViewResources->globalDescriptorSet[bloomFrame.renderFrameIndex % MAX_FRAMES_IN_FLIGHT], fogVizMode,
            fogVizMode ? fogVizSettings.vizScaleInScatter : fogVizSettings.vizScaleDensity, fogVizSettings.vizOpacity,
            fogVizEnabled && fogResolved.handle.IsValid());
        const FogVizBindingRef fogVizBinding{&fogVizNative};
        const auto& giVizSettings = s.GetRestirGiSettings();
        const auto& giVizTexture = m_CurrentViewResources->restirGiDI;
        const auto giVizNative = m_RestirGi.PrepareReservoirVizBindings(m_CurrentViewResources->giReservoirVizDescSet,
            view.targets->GetSceneDepth(), m_CurrentViewResources->restirGiSpatial, bloomView.width, bloomView.height,
            giVizTexture ? giVizTexture->GetWidth() : bloomView.width, giVizTexture ? giVizTexture->GetHeight() : bloomView.height,
            giVizSettings.temporalMCap, giVizSettings.spatialNeighbours, giVizSettings.maxReservoirAge,
            !ptEnabled && shadeMode == ShadeMode::RestirGiReservoir && giVizSettings.enabled && giSpatialReservoir.handle.IsValid());
        const GiReservoirVizBindingRef giVizBinding{&giVizNative};
        const auto optionalSlimInput = [](auto key, const GraphTextureRef& ref) {
            return ref.handle.IsValid() ? RenderInputBinding::Present(key, ref) : RenderInputBinding::Absent(key);
        };
        const std::array visualizationResources{RenderInputBinding::Present(RenderResources::TonemappedLDR, tonemappedLdr),
            RenderInputBinding::Present(SlimVizResources::Bindings, slimVizBinding),
            RenderInputBinding::Present(ClusterVizResources::Bindings, clusterVizBinding),
            RenderInputBinding::Present(FogVizResources::Bindings, fogVizBinding),
            RenderInputBinding::Present(GiReservoirVizResources::Bindings, giVizBinding),
            giSpatialReservoir.handle.IsValid() ? RenderInputBinding::Present(GiReservoirVizResources::SpatialReservoir, giSpatialReservoir)
                : RenderInputBinding::Absent(GiReservoirVizResources::SpatialReservoir),
            optionalSlimInput(RenderResources::FogDensity, fogDensity), optionalSlimInput(RenderResources::ResolvedFog, fogResolved),
            optionalSlimInput(RenderResources::SurfaceDepth, surfaceDepth),
            clusterGrid.handle.IsValid() ? RenderInputBinding::Present(RenderResources::ClusterGrid, clusterGrid)
                : RenderInputBinding::Absent(RenderResources::ClusterGrid),
            optionalSlimInput(RenderResources::Normal, normalOutput), optionalSlimInput(RenderResources::Roughness, roughnessOutput),
            optionalSlimInput(RenderResources::MotionVectors, motionVectors), optionalSlimInput(RenderResources::MaterialID, materialOutput)};
        FrameRenderInputs visualizationFrame; visualizationFrame.renderFrameIndex = bloomFrame.renderFrameIndex; visualizationFrame.resources = visualizationResources;
        GraphTextureRef visualizedStageLdr;
        const std::array visualizationExports{RenderOutputBinding::Capture(RenderResources::VisualizedLDR, visualizedStageLdr)};
        const auto visualizationBuild = m_VisualizationComposition->Build(rg, visualizationFrame, bloomView, s.GetFrameAllocator(), visualizationExports);
        if (!visualizationBuild.success)
        {
            for (const auto& diagnostic : visualizationBuild.diagnostics)
                LH_LOG(Renderer, error, "Visualization composition: {}", diagnostic.message);
            return false;
        }
        ldrOutput = visualizedStageLdr.handle;
        // Selection outline + debug shapes need the raster G-buffer (entityID mask + scene depth), which
        // PT culls, so both are off in PT mode (the reference is an offline-accumulation view, not interactive).
        const auto outlineNative = m_EditorOverlays.PrepareOutlineBindings(m_CurrentViewResources->overlays,
            view.camera, bloomView.width, bloomView.height, view.drawSelectionOutline && !ptActive);
        const OutlineBindingRef outlineBinding{&outlineNative};
        const GraphTextureRef visualizedLdr{ldrOutput, tonemappedLdr.binding};
        const GraphTextureRef outlineDepth{geoOutput.depth, surfaceDepth.binding};
        const auto optionalOutlineInput = [](auto key, const GraphTextureRef& ref) {
            return ref.handle.IsValid() ? RenderInputBinding::Present(key, ref) : RenderInputBinding::Absent(key);
        };
        const std::array outlineResources{RenderInputBinding::Present(RenderResources::VisualizedLDR, visualizedLdr),
            RenderInputBinding::Present(OutlineResources::Bindings, outlineBinding),
            optionalOutlineInput(RenderResources::SelectionMask, selectionMask),
            optionalOutlineInput(RenderResources::SelectionDepth, selectionDepth),
            !ptActive ? RenderInputBinding::Present(RenderResources::LitDepth, outlineDepth)
                : RenderInputBinding::Absent(RenderResources::LitDepth)};
        FrameRenderInputs outlineFrame; outlineFrame.renderFrameIndex = bloomFrame.renderFrameIndex;
        outlineFrame.resources = outlineResources;
        GraphTextureRef outlinedLdr;
        const std::array outlineExports{RenderOutputBinding::Capture(RenderResources::OutlinedLDR, outlinedLdr)};
        const auto outlineBuild = m_OutlineComposition->Build(rg, outlineFrame, bloomView, s.GetFrameAllocator(), outlineExports);
        if (!outlineBuild.success)
        {
            for (const auto& diagnostic : outlineBuild.diagnostics)
                LH_LOG(Renderer, error, "Outline composition: {}", diagnostic.message);
            return false;
        }
        const auto debugDrawNative = m_DebugDraw.PrepareBindings(DebugDraw::GetForRender(bloomFrame.renderFrameIndex),
            m_Global.GetCachedViewProj(), bloomFrame.renderFrameIndex, view.drawDebugShapes && !ptActive);
        const DebugDrawBindingRef debugDrawBinding{&debugDrawNative};
        const std::array debugDrawInputs{RenderInputBinding::Present(RenderResources::OutlinedLDR, outlinedLdr),
            RenderInputBinding::Present(DebugDrawResources::Bindings, debugDrawBinding)};
        FrameRenderInputs debugDrawFrame; debugDrawFrame.renderFrameIndex = bloomFrame.renderFrameIndex; debugDrawFrame.resources = debugDrawInputs;
        GraphTextureRef finalViewLdr;
        const std::array debugDrawExports{RenderOutputBinding::Capture(RenderResources::FinalViewLDR, finalViewLdr)};
        const auto debugDrawBuild = m_DebugDrawComposition->Build(rg, debugDrawFrame, bloomView, s.GetFrameAllocator(), debugDrawExports);
        if (!debugDrawBuild.success)
        {
            for (const auto& diagnostic : debugDrawBuild.diagnostics)
                LH_LOG(Renderer, error, "DebugDraw composition: {}", diagnostic.message);
            return false;
        }
        RG::ResourceHandle finalOutput = finalViewLdr.handle;
        if (view.emitImGuiPass)
            AddImGuiPass(rg, finalOutput);

        rg.Compile();

        // Capture render graph snapshot for Frame Debugger panel
        auto& graphSnapshot = m_System.CaptureGraphSnapshot(rg);

        // Read GPU timing + pipeline stats from completed frames and fill snapshot. ReadStats must run
        // BEFORE ReadResults; they share the frame counter that ReadResults advances.
        std::vector<float> gpuTimes;
        std::vector<RG::GpuPipelineStats> gpuStats;
        u32 nonCulledCount = 0;
        for (auto& p : graphSnapshot.passes)
            if (!p.culled) nonCulledCount++;

        m_GPUTimers.ReadStats(nonCulledCount, gpuStats);
        m_GPUTimers.ReadResults(nonCulledCount, gpuTimes);
        float totalMs = 0.0f;
        RG::GpuPipelineStats total{};
        u32 timerIdx = 0;
        for (auto& p : graphSnapshot.passes)
        {
            if (p.culled) continue;
            if (timerIdx < (u32)gpuTimes.size())
            {
                p.gpuTimeMs = gpuTimes[timerIdx];
                if (gpuTimes[timerIdx] > 0.0f) totalMs += gpuTimes[timerIdx];
            }
            if (timerIdx < (u32)gpuStats.size() && gpuStats[timerIdx].valid)
            {
                p.stats = gpuStats[timerIdx];
                total.inputVertices   += p.stats.inputVertices;
                total.inputPrimitives += p.stats.inputPrimitives;
                total.vsInvocations   += p.stats.vsInvocations;
                total.clipInvocations += p.stats.clipInvocations;
                total.clipPrimitives  += p.stats.clipPrimitives;
                total.fsInvocations   += p.stats.fsInvocations;
                total.valid = true;
            }
            timerIdx++;
        }
        graphSnapshot.totalGpuTimeMs = totalMs;
        graphSnapshot.totalStats = total;

        m_System.BeginViewCapture(view);

        bool hasComputeWork = false;
        CaptureSource recordedSource = m_System.GetFrameDebugger().requestedSource;
        {
            CaptureRecordingSession recording(m_System.GetFrameDebugger(), rg, view.id,
                m_CurrentViewResources ? m_CurrentViewResources->generation : 0, view.captureRequested);
            recordedSource = recording.Source();
            hasComputeWork = Renderer::RecordGraph(recorders, rg, &m_GPUTimers);
        }
        // Non-primary views: transition LDR -> SHADER_READ so the scene view's ImGui pass can sample it (scene
        // view's RG already does this via ImGuiPass's builder.Read(sceneColor)).
        // Recorded into recorders.gB because the LDR is written by PBR / post-process in the gB segment; gA runs
        // first on the GPU timeline, so recording the transition there would precede the write and the next-frame
        // PBR would see the image in SHADER_READ_ONLY_OPTIMAL instead of the expected COLOR_ATTACHMENT_OPTIMAL.
        if (!view.emitImGuiPass && view.targets && view.targets->GetLDROutput())
        {
            auto vkLdr = std::static_pointer_cast<VKTexture>(view.targets->GetLDROutput());
            VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = vkLdr->GetImage();
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            vkCmdPipelineBarrier(recorders.gB,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        // Finalize capture (only the source view; matches the sink gate above).
        if (view.captureRequested && m_System.GetFrameDebugger().state == DebuggerState::CaptureRequested)
        {
            CaptureFinalizationInputs capture;
            capture.source = recordedSource;
            capture.viewProj = m_Global.GetCachedViewProj();
            capture.cascades = m_Global.GetCascades();
            capture.shadowParams = m_Global.GetShadowParams();
            // Snapshot captured-view metadata + Set 0 binding sources for replay.
            // invariant: replay reads these instead of m_CurrentViewResources / live IBL
            // textures, since the live state reflects whichever view ran last and IBL
            // can change mid-Freeze.
            capture.view.targets         = view.targets;
            capture.view.id = view.id;
            capture.view.resourceGeneration = m_CurrentViewResources ? m_CurrentViewResources->generation : 0;
            capture.view.viewIndex       = view.viewIndex;
            if (view.targets && view.targets->GetSceneColor())
            {
                capture.view.width  = view.targets->GetSceneColor()->GetWidth();
                capture.view.height = view.targets->GetSceneColor()->GetHeight();
            }
            const u32 replaySlot = m_System.GetFrameDebugger().capturedFrame.capturedRenderFrameIndex % MAX_FRAMES_IN_FLIGHT;
            capture.replayBindings.sets = {m_CurrentViewResources->globalDescriptorSet[replaySlot],
                VulkanContext::Get().GetBindlessSet().GetSet(), MaterialSystem::GetDescriptorSet(replaySlot),
                m_Lighting.GetLightDescSet(replaySlot), BoneMatrixBuffer::GetDescriptorSet(replaySlot),
                m_Geometry.GetObjectSSBODescSet(replaySlot)};
            const auto indirect = m_Geometry.GetIndirectRegion();
            capture.replayBindings.indirectBuffer = indirect.buffer;
            capture.replayBindings.indirectOffset = indirect.offset;
            capture.replayBindings.indirectSize = indirect.size;
            capture.replayBindings.regionsPerView = k_IndirectRegionsPerView;
            capture.replayBindings.regionStride = k_IndirectRegionStride;
            m_Global.GetLastUboBytes(capture.globalUboBytes);
            capture.irradiance     = m_Lighting.GetIrradianceMap();
            capture.prefiltered    = m_Lighting.GetPrefilteredMap();
            capture.brdf           = m_Lighting.GetBRDFLut();
            const auto* gtaoState = GetGtaoViewState(view.id);
            capture.gtaoFinal      = gtaoState ? gtaoState->finalAO : nullptr;
            capture.iblIntensity   = view.camera.iblIntensity;
            capture.skyboxIntensity = view.camera.skyboxIntensity;
            // Resolve descendants once at capture; replay reads this without touching m_CurrentView (stack-allocated, dangles in Frozen).
            {
                std::unordered_set<entt::entity> resolved;
                m_EditorOverlays.CollectSelectedHandles(view.camera.selectedEntities, resolved);
                capture.selectionHandles.assign(resolved.begin(), resolved.end());
            }

            m_System.FinalizeViewCapture(capture, graphSnapshot);
        }

        return hasComputeWork;
    }

    void RenderPipeline::RegisterNamedTextures()
    {
        m_NamedTextures.clear();
        if (m_Lighting.GetShadowMap())                     m_NamedTextures["ShadowMap"]    = m_Lighting.GetShadowMap();
        if (m_System.GetSceneTargets().GetSceneColor())    m_NamedTextures["SceneColor"]   = m_System.GetSceneTargets().GetSceneColor();
        if (m_System.GetSceneTargets().GetSceneDepth())    m_NamedTextures["SceneDepth"]   = m_System.GetSceneTargets().GetSceneDepth();
        if (m_System.GetSceneTargets().GetLDROutput())     m_NamedTextures["LDROutput"]    = m_System.GetSceneTargets().GetLDROutput();
        if (m_System.GetSceneTargets().GetEntityIDBuffer())m_NamedTextures["EntityID"]     = m_System.GetSceneTargets().GetEntityIDBuffer();
        // Scene-view bloom textures; Frame Debugger is scene-view-only.
        if (auto it = m_ViewResources.find(m_System.GetViews().Find(&m_System.GetSceneTargets()).value); it != m_ViewResources.end()) {
            for (u32 i = 0; i < BloomViewState::kMipCount; ++i)
                if (it->second.bloom && it->second.bloom->mips[i]) m_NamedTextures["BloomMip" + std::to_string(i)] = it->second.bloom->mips[i];
            if (it->second.fog && it->second.fog->volDensity)          m_NamedTextures["VolDensity"]           = it->second.fog->volDensity;
            if (it->second.fog && it->second.fog->volInScatter)        m_NamedTextures["VolInScatter"]         = it->second.fog->volInScatter;
            if (it->second.fog && it->second.fog->volInScatterHistA)   m_NamedTextures["VolInScatterHistA"]   = it->second.fog->volInScatterHistA;
            if (it->second.fog && it->second.fog->volInScatterHistB)   m_NamedTextures["VolInScatterHistB"]   = it->second.fog->volInScatterHistB;
            if (it->second.reflRadiance)        m_NamedTextures["Reflections"]         = it->second.reflRadiance;
        }
        if (m_Lighting.GetIrradianceMap())  m_NamedTextures["IrradianceMap"]  = m_Lighting.GetIrradianceMap();
        if (m_Lighting.GetPrefilteredMap()) m_NamedTextures["PrefilteredMap"] = m_Lighting.GetPrefilteredMap();
        if (m_Lighting.GetBRDFLut())        m_NamedTextures["BRDF_LUT"]       = m_Lighting.GetBRDFLut();
        // Slim G-buffer attachments. Empty until SlimGBufferPass writes them.
        if (m_System.GetSceneTargets().GetSlimNormal())     m_NamedTextures["SlimNormal"]     = m_System.GetSceneTargets().GetSlimNormal();
        if (m_System.GetSceneTargets().GetSlimRoughness())  m_NamedTextures["SlimRoughness"]  = m_System.GetSceneTargets().GetSlimRoughness();
        if (m_System.GetSceneTargets().GetSlimMotion())     m_NamedTextures["SlimMotion"]     = m_System.GetSceneTargets().GetSlimMotion();
        if (m_System.GetSceneTargets().GetSlimMaterialID()) m_NamedTextures["SlimMaterialID"] = m_System.GetSceneTargets().GetSlimMaterialID();
    }

    std::shared_ptr<Texture> RenderPipeline::GetNamedTexture(const std::string& name) const
    {
        auto it = m_NamedTextures.find(name);
        return (it != m_NamedTextures.end()) ? it->second : nullptr;
    }

    // ---- Compatibility frame-debugger forwarders into RenderingSystem ----

    const RG::RenderGraphSnapshot& RenderPipeline::GetGraphSnapshot() const
    {
        return m_System.GetGraphSnapshot();
    }

    VkImageView RenderPipeline::GetPerDrawPreviewView()  const { return m_System.GetPerDrawPreviewView(); }
    u64         RenderPipeline::GetPerDrawPreviewKey()   const { return m_System.GetPerDrawPreviewKey(); }
    u32         RenderPipeline::GetPerDrawPreviewWidth() const { return m_System.GetPerDrawPreviewWidth(); }
    u32         RenderPipeline::GetPerDrawPreviewHeight()const { return m_System.GetPerDrawPreviewHeight(); }
    VkImageView RenderPipeline::GetDepthPreviewView()    const { return m_System.GetDepthPreviewView(); }
    u32         RenderPipeline::GetDepthPreviewWidth()   const { return m_System.GetDepthPreviewWidth(); }
    u32         RenderPipeline::GetDepthPreviewHeight()  const { return m_System.GetDepthPreviewHeight(); }
    void        RenderPipeline::ResetPreviewCacheKeys() { m_System.ResetPreviewCacheKeys(); }

    void RenderPipeline::ReplayPassUpToDraw(u32 passIdx, u32 localDrawIdx)
    {
        m_System.ReplayPassUpToDraw(passIdx, localDrawIdx);
    }

    void RenderPipeline::BlitArchivedDepthToPreview(u32 archiveIdx, int layer, float nearZ, float farZ)
    {
        m_System.BlitArchivedDepthToPreview(archiveIdx, layer, nearZ, farZ);
    }

    void RenderPipeline::BlitArchivedSlimToPreview(u32 archiveIdx, u32 mode, float scale)
    {
        m_System.BlitArchivedSlimToPreview(archiveIdx, mode, scale);
    }

    VkImageView RenderPipeline::GetSlimPreviewView()   const { return m_System.GetSlimPreviewView(); }
    u32         RenderPipeline::GetSlimPreviewWidth()  const { return m_System.GetSlimPreviewWidth(); }
    u32         RenderPipeline::GetSlimPreviewHeight() const { return m_System.GetSlimPreviewHeight(); }

    // ---- Public-API forwarders into subsystems (preserve caller compat) ----

    void RenderPipeline::UpdateGlobalUniforms(const CameraParams& camera,
                                              const CascadeData& cascades,
                                              const DirectionalLightShadowParams& shadowParams)
    {
        m_Global.UpdateUBO(camera, cascades, shadowParams);
    }

    void RenderPipeline::BuildGPUObjectBuffer(const RenderSnapshot& snapshot)
    {
        m_Geometry.BuildGPUObjectBuffer(snapshot);
    }

    u32 RenderPipeline::EnsureMaterialRegistered(std::shared_ptr<Material> material)
    {
        return m_Geometry.EnsureMaterialRegistered(material);
    }

    void RenderPipeline::UpdateGTAOUBO()
    {
        if (!m_CurrentViewResources) return;
        auto* state = m_GtaoStates.Find({m_CurrentViewResources->id});
        if (!state) return;
        auto settings = m_System.GetPostProcessSettings().gtao;
        settings.enabled = settings.enabled && m_GTAO.IsReady();
        m_GTAO.UpdateUBO(**state, m_CurrentViewResources->globalDescriptorSet, settings,
            Renderer::GetFrameData()->GetRenderFrameIndex());
    }

    void RenderPipeline::ReloadSkybox(const fs::path& hdrPath)
    {
        std::vector<VkDescriptorSetLayout> geoLayouts = {
            m_Global.GetSetLayout(),
            VulkanContext::Get().GetBindlessSet().GetLayout(),
            MaterialSystem::GetDescriptorSetLayout(),
            m_Lighting.GetSetLayout(),
            BoneMatrixBuffer::GetDescriptorSetLayout(),
            m_Geometry.GetSet5Layout()
        };
        m_Lighting.ReloadSkybox(hdrPath, geoLayouts);

        // Rewrite Set 0 IBL bindings on every cached view (new textures behind the old views).
        GlobalViewWriteContext ctx{};
        ctx.haveIBL          = m_Lighting.IsIBLReady();
        ctx.iblSampler       = m_Lighting.GetIBLSampler();
        ctx.gtaoSampler      = m_GTAO.GetSampler();
        if (ctx.haveIBL)
        {
            ctx.irradianceView  = std::static_pointer_cast<VKTexture>(m_Lighting.GetIrradianceMap())->GetImageView();
            ctx.prefilteredView = std::static_pointer_cast<VKTexture>(m_Lighting.GetPrefilteredMap())->GetImageView();
            ctx.brdfView        = std::static_pointer_cast<VKTexture>(m_Lighting.GetBRDFLut())->GetImageView();
        }
        for (auto& [targets, vr] : m_ViewResources)
        {
            const auto* gtao = GetGtaoViewState({vr.id});
            ctx.gtaoFinalView = gtao ? gtao->finalBinding.view : VK_NULL_HANDLE;
            m_Global.WriteView(vr, ctx);
        }
    }
}
