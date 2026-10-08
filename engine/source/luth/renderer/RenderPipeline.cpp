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
#include "luth/renderer/features/SortedTransparencyFeature.h"
#include "luth/renderer/features/SkyFeature.h"
#include "luth/renderer/features/ForwardOpaqueCompatibility.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include "luth/renderer/debug/FrameDebuggerContext.h"
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
        , m_Debugger(std::make_unique<FrameDebuggerContext>(*this))
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
        m_EditorOverlays.Init(*this);
        m_DebugDraw.Init(*this);
        m_PostProcess.Init(*this);

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
        m_Transparency.Init(*this);

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
        RenderPipelineDefinition sortedDefinition;
        sortedDefinition.AddFeature<SortedTransparencyFeature>(m_Transparency, &m_System.GetFrameDebugger());
        PipelineInputContract sortedInputs;
        sortedInputs.resources = {{TransparencyResources::Bindings}, {RenderResources::FoggedHDR}, {RenderResources::LitDepth},
            {RenderResources::OpaquePickingIDs}, {RenderResources::CameraVisibleDraws},
            {RenderResources::ResolvedFog, ResourceOutputPresence::Optional}, {RenderResources::RefractionBackdrop, ResourceOutputPresence::Optional},
            {RenderResources::LightData, ResourceOutputPresence::Optional}, {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
            {RenderResources::LightIndices, ResourceOutputPresence::Optional}};
        sortedInputs.capabilities = {&DeformationResources::DeformedGeometry};
        auto sortedCompiled = RenderPipelineCompiler{}.Compile(std::move(sortedDefinition), {}, sortedInputs);
        if (!sortedCompiled.ReplaceIfValid(m_SortedTransparencyComposition))
            throw std::runtime_error("Sorted transparency definition failed semantic validation");
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

        VkDevice device = VulkanContext::Get().GetDevice();

        // Release per-view state before the shared layouts it references.
        for (auto& [targets, vr] : m_ViewResources)
            DestroyViewResources(vr);
        m_ViewResources.clear();
        m_GtaoPipeline.reset();
        m_GtaoStates.ReleaseAll([] { Renderer::WaitForGPU(); });

        m_Debugger->Shutdown();
        m_System.GetFrameDebugger().Shutdown(device);

        // Subsystems own their layouts/pools/samplers/pipelines.
        m_Transparency.Shutdown();
        m_GeometryPreparationPipeline.reset();
        m_SurfacePreparationComposition.reset();
        m_CsmComposition.reset();
        m_ClusterComposition.reset();
        m_FogComputeComposition.reset();
        m_FogCompositeComposition.reset();
        m_RefractionComposition.reset();
        m_SortedTransparencyComposition.reset();
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
        m_CurrentViewResources->taaRecorded = false;
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
        GraphTextureRef surfaceDepth{};
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
            GraphTextureRef normalOutput, roughnessOutput, motionOutput, materialOutput;
            const std::array depthOutputs{
                RenderOutputBinding::Capture(RenderResources::PrepassDepth, depthOutput),
                RenderOutputBinding::Capture(RenderResources::SurfaceDepth, surfaceDepth),
                RenderOutputBinding::Capture(RenderResources::Normal, normalOutput),
                RenderOutputBinding::Capture(RenderResources::Roughness, roughnessOutput),
                RenderOutputBinding::Capture(RenderResources::MotionVectors, motionOutput),
                RenderOutputBinding::Capture(RenderResources::MaterialID, materialOutput)};
            const auto depthBuild = m_SurfacePreparationComposition->Build(rg, depthFrame, depthView,
                s.GetFrameAllocator(), depthOutputs);
            if (!depthBuild.success)
            {
                for (const auto& diagnostic : depthBuild.diagnostics)
                    LH_LOG(Renderer, error, "Surface preparation composition: {}", diagnostic.message);
                return false;
            }
            slimGB = {normalOutput.handle, roughnessOutput.handle, motionOutput.handle, materialOutput.handle};
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
        if (volResolvedHandle.IsValid())
        {
            m_Volumetric.WriteVizPerFrame(*m_CurrentViewResources, fogFrameAbs);
        }

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
        RG::ResourceHandle giDIHandle = ptEnabled
            ? RG::ResourceHandle{}
            : m_RestirGi.AddPasses(rg, surfaceDepth.handle, slimGB.normal, slimGB.motion);

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
        // megakernel output drives the post chain via hdrForPost below. geoOutput/maskOutput/taaColor hoisted
        // for the overlays + post chain; default-invalid in PT (the overlays that read them are !ptActive too).
        GeometryOutput      geoOutput{};
        SelectionMaskOutput maskOutput{};
        RG::ResourceHandle  taaColor{};
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
            maskOutput = view.drawSelectionOutline
                         ? m_EditorOverlays.AddSelectionMaskPass(rg)
                         : SelectionMaskOutput{};
            // The legacy opaque producer exports typed stages; sky reuses their imports.
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
            // when the transparent bucket is empty (matches AddPasses' own early-out).
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
            RG::ResourceHandle transparentColor = fogColor;
            if (m_CurrentViewResources)
            {
                const u32 frameAbsT = static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex());
                m_Transparency.WritePerFrame(*m_CurrentViewResources->transparency, m_CurrentViewResources->fog,
                    m_Volumetric.GetSampler(), frameAbsT);
                if (s.GetTransparencySettings().mode == TransparencyMode::OIT)
                    transparentColor = m_Transparency.AddPasses(rg, fogColor, geoOutput.entityID, geoOutput.depth,
                        volumetricEnabled ? volResolvedHandle : RG::ResourceHandle{}, backdropHandle, hIndirectBuf);
                else
                {
                    std::array<VkDescriptorSet, 7> sortedSets;
                    std::copy(forwardSets.begin(), forwardSets.end(), sortedSets.begin());
                    sortedSets[6] = m_CurrentViewResources->transparency->transparentDescSet[frameAbsT % MAX_FRAMES_IN_FLIGHT];
                    const auto sortedNative = m_Transparency.PrepareSortedBindings(m_Geometry, sortedSets,
                        s.GetShadeMode() == ShadeMode::Wireframe,
                        view.captureRequested && s.GetFrameDebugger().state == DebuggerState::CaptureRequested,
                        view.camera.view, cameraVisible, s.GetDrawList(), s.GetActiveSnapshot(),
                        fogResolved.binding, backdropOutput.binding, &m_Rt);
                    const SortedTransparencyBindingRef sortedBinding{&sortedNative};
                    const auto optionalImage = [](auto key, const GraphTextureRef& value) {
                        return value.handle.IsValid() ? RenderInputBinding::Present(key, value) : RenderInputBinding::Absent(key);
                    };
                    const std::array sortedResources{RenderInputBinding::Present(TransparencyResources::Bindings, sortedBinding),
                        RenderInputBinding::Present(RenderResources::FoggedHDR, foggedOutput),
                        RenderInputBinding::Present(RenderResources::LitDepth, litOutput),
                        RenderInputBinding::Present(RenderResources::OpaquePickingIDs, pickingOutput),
                        RenderInputBinding::Present(RenderResources::CameraVisibleDraws, cameraVisible),
                        optionalImage(RenderResources::ResolvedFog, fogResolved), optionalImage(RenderResources::RefractionBackdrop, backdropOutput),
                        optionalBuffer(RenderResources::LightData, lightData), optionalBuffer(RenderResources::ClusterGrid, clusterGrid),
                        optionalBuffer(RenderResources::LightIndices, lightIndices)};
                    FrameRenderInputs sortedFrame; sortedFrame.renderFrameIndex = skyFrame.renderFrameIndex;
                    sortedFrame.resources = sortedResources; sortedFrame.capabilities = forwardCapabilities;
                    GraphTextureRef transparentOutput, finalPicking, transparentDepth;
                    const std::array sortedExports{RenderOutputBinding::Capture(RenderResources::TransparentHDR, transparentOutput),
                        RenderOutputBinding::Capture(RenderResources::FinalPickingIDs, finalPicking),
                        RenderOutputBinding::Capture(TransparencyResources::Depth, transparentDepth)};
                    const auto sortedBuild = m_SortedTransparencyComposition->Build(rg, sortedFrame, skyView,
                        s.GetFrameAllocator(), sortedExports);
                    if (!sortedBuild.success)
                    {
                        for (const auto& diagnostic : sortedBuild.diagnostics)
                            LH_LOG(Renderer, error, "Sorted transparency composition: {}", diagnostic.message);
                        return false;
                    }
                    transparentColor = transparentOutput.handle;
                    geoOutput.entityID = finalPicking.handle; geoOutput.depth = transparentDepth.handle;
                }
            }
            // TAA Resolve: Karis14 YCoCg-clip, HDR-domain, after the fog composite + before bloom/grid.
            // WriteTaaResolvePerFrame rebinds the parity-picked history-prev; the resolve writes history-curr.
            const PostProcessSettings& pps = m_System.GetPostProcessSettings();
            const bool taaEnabled = pps.taaEnabled && m_CurrentViewResources;
            if (taaEnabled)
                m_PostProcess.WriteTaaResolvePerFrame(*m_CurrentViewResources,
                    static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex()));
            taaColor = taaEnabled
                       ? m_PostProcess.AddTaaResolvePass(rg, transparentColor, slimGB.motion, surfaceDepth.handle)
                       : transparentColor;
        }
        // Bloom/composite source rebind runs in BOTH paths: PT -> the ptColor display image; else the TAA
        // chain output (taaHistoryCurr) when TAA is on, else SceneColor. Without it the bindings statically reference SceneColor.
        if (m_CurrentViewResources)
            m_PostProcess.UpdateBloomCompositeInput(*m_CurrentViewResources, *view.targets,
                static_cast<u32>(Renderer::GetFrameData()->GetRenderFrameIndex()));
        // HDR source for the post chain: the PT megakernel output replaces the raster sceneColor when PT
        // is active (the raster chain above is then dead-pass-culled). Grid is editor-overlay-only -> off in PT.
        RG::ResourceHandle hdrForPost  = ptActive ? ptColorHandle : taaColor;
        // Resolve the active shade mode once (PT forces Lit). Hoisted here so the bloom gate and the slim-viz dispatch below share it.
        const ShadeMode shadeMode = ptActive ? ShadeMode::Lit : m_System.GetShadeMode();
        // Bloom is skipped at strength 0 (composite adds bloom x strength; AddCompositePass guards an
        // invalid handle) and for every non-Lit mode: bloom is a radiance effect that smears over data
        // views and clutters radiance debug. Reads PRE-grid color so grid lines don't bloom.
        RG::ResourceHandle bloomResult = (m_System.GetPostProcessSettings().bloomStrength > 0.0f
                                          && shadeMode == ShadeMode::Lit)
                                         ? m_PostProcess.AddBloomPasses(rg, hdrForPost)
                                         : RG::ResourceHandle{};
        RG::ResourceHandle gridColor   = (view.drawGrid && !ptActive)
                                         ? m_EditorOverlays.AddGridPass(rg, hdrForPost, geoOutput.depth)
                                         : hdrForPost;
        RG::ResourceHandle ldrOutput = m_PostProcess.AddCompositePass(rg, gridColor, bloomResult);

        // Slim G-buffer ShadeMode toggles overwrite LDROutput with a decoded attachment. Mode index = enum offset
        // from ShadeMode::SlimNormal (0..3). Motion scale hardcoded: the frame-debugger panel exposes a slider for
        // per-capture tuning; live viz uses a sensible default matching the existing thumbnail UX. PT mode forces
        // Lit (the debug-viz blits read the culled G-buffer / cluster / reservoir state, meaningless over the PT
        // image). shadeMode was resolved above (hoisted for the bloom gate).
        if (shadeMode >= ShadeMode::SlimNormal && shadeMode <= ShadeMode::SlimMaterialID)
        {
            const u32 slimMode = static_cast<u32>(shadeMode) - static_cast<u32>(ShadeMode::SlimNormal);
            ldrOutput = m_PostProcess.AddSlimVizPass(rg, ldrOutput, slimGB, slimMode, /*motionScale*/20.0f);
        }
        else if (shadeMode == ShadeMode::ClustersDensity)
        {
            ldrOutput = m_Lighting.AddClusterVizPass(rg, ldrOutput, surfaceDepth.handle);
        }
        else if ((shadeMode == ShadeMode::VolumetricDensity ||
                  shadeMode == ShadeMode::VolumetricInScatter) &&
                 volResolvedHandle.IsValid() && m_CurrentViewResources)
        {
            const u32 vizMode = (shadeMode == ShadeMode::VolumetricDensity) ? 0u : 1u;
            ldrOutput = m_Volumetric.AddVizPass(rg, ldrOutput, fogDensity.handle, volResolvedHandle, surfaceDepth.handle, vizMode);
        }
        else if (shadeMode == ShadeMode::RestirGiReservoir && m_RestirGi.IsEnabled() && m_CurrentViewResources)
        {
            ldrOutput = m_RestirGi.AddReservoirVizPass(rg, ldrOutput, surfaceDepth.handle);
        }

        // Selection outline + debug shapes need the raster G-buffer (entityID mask + scene depth), which
        // PT culls, so both are off in PT mode (the reference is an offline-accumulation view, not interactive).
        RG::ResourceHandle finalOutput = (view.drawSelectionOutline && !ptActive)
                                         ? m_EditorOverlays.AddOutlinePass(rg, ldrOutput, maskOutput, geoOutput.depth)
                                         : ldrOutput;
        if (view.drawDebugShapes && !ptActive)
            finalOutput = m_DebugDraw.AddDebugDrawPass(rg, finalOutput);
        if (view.emitImGuiPass)
            AddImGuiPass(rg, finalOutput);

        rg.Compile();

        // Capture render graph snapshot for Frame Debugger panel
        m_GraphSnapshot = CaptureSnapshot(rg);

        // Read GPU timing + pipeline stats from completed frames and fill snapshot. ReadStats must run
        // BEFORE ReadResults; they share the frame counter that ReadResults advances.
        std::vector<float> gpuTimes;
        std::vector<RG::GpuPipelineStats> gpuStats;
        u32 nonCulledCount = 0;
        for (auto& p : m_GraphSnapshot.passes)
            if (!p.culled) nonCulledCount++;

        m_GPUTimers.ReadStats(nonCulledCount, gpuStats);
        m_GPUTimers.ReadResults(nonCulledCount, gpuTimes);
        float totalMs = 0.0f;
        RG::GpuPipelineStats total{};
        u32 timerIdx = 0;
        for (auto& p : m_GraphSnapshot.passes)
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
        m_GraphSnapshot.totalGpuTimeMs = totalMs;
        m_GraphSnapshot.totalStats = total;

        // Wire the archive sink for this capture. The sink copies each tracked RT after the pass that writes it.
        // Gate on the per-view captureRequested flag (set by the view's owner: RenderingSystem for the scene view,
        // GamePanel for the game view) so the chosen capture source's RG installs the sink, not the editor's by default.
        if (view.captureRequested && m_System.GetFrameDebugger().state == DebuggerState::CaptureRequested)
        {
            // Create the debug sampler for ImGui archive previews. Idempotent; returns immediately once blitPipeline is already set.
            m_Debugger->InitDebugBlitResources();

            // Invalidate per-draw and depth preview caches. Cache keys are (passIdx, drawIdx) / (archiveIdx,
            // layer+1), which can collide across captures even though the underlying scene state has changed
            // (camera moved -> recapture -> same indices, new content). Without this reset, re-clicking the
            // same draw or cascade slice after recapture would hit stale cached previews.
            m_Debugger->ResetPreviewCacheKeys();

            m_System.GetFrameDebugger().BeginCapture(VulkanContext::Get().GetDevice(),
                                            VulkanContext::Get().GetAllocator());
            m_System.GetFrameDebugger().RegisterTrackedRT("SceneColor");
            m_System.GetFrameDebugger().RegisterTrackedRT("SceneDepth");
            // ShadowPass imports per-cascade resources named "ShadowMap.C<i>" (one per cascade, each a
            // single-layer view onto the shared 4-layer array). Track each variant so the sink archives them;
            // without this, cascade nodes have no primary output and the panel shows "no output preview".
            for (u32 ci = 0; ci < k_ShadowCascadeCount; ++ci)
                m_System.GetFrameDebugger().RegisterTrackedRT("ShadowMap.C" + std::to_string(ci));
            m_System.GetFrameDebugger().RegisterTrackedRT("LDROutput");
            m_System.GetFrameDebugger().RegisterTrackedRT("EntityID");
            m_System.GetFrameDebugger().RegisterTrackedRT("BloomAFinal");
            m_System.GetFrameDebugger().RegisterTrackedRT("GTAOLinearDepth");
            m_System.GetFrameDebugger().RegisterTrackedRT("GTAORawAO");
            m_System.GetFrameDebugger().RegisterTrackedRT("GTAOFinal");
            // Slim G-buffer attachments. Archive sink copies all 4 after the pass.
            m_System.GetFrameDebugger().RegisterTrackedRT("SlimNormal");
            m_System.GetFrameDebugger().RegisterTrackedRT("SlimRoughness");
            m_System.GetFrameDebugger().RegisterTrackedRT("SlimMotion");
            m_System.GetFrameDebugger().RegisterTrackedRT("SlimMaterialID");
            rg.SetArchiveSink(&m_System.GetFrameDebugger());
        }

        // Only the capturing view needs serial Phase-1 dispatch: its lambdas push into shared FrameDebugger
        // metadata vectors. Non-capturing views' pushes are suppressed below, so they record in parallel
        // exactly as in non-capture frames.
        if (view.captureRequested && m_System.GetFrameDebugger().state == DebuggerState::CaptureRequested)
            rg.SetSerialize(true);

        // Mask state to Inactive around non-capturing views' RG execute so their lambdas' BeginCapturePass /
        // CaptureXX early-return: no pushes, no race, no need for SetSerialize.
        const DebuggerState savedDbgState = m_System.GetFrameDebugger().state;
        const bool suppressDebuggerMetadata = !view.captureRequested
                                              && savedDbgState == DebuggerState::CaptureRequested;
        if (suppressDebuggerMetadata)
            m_System.GetFrameDebugger().state = DebuggerState::Inactive;

        const bool hasComputeWork = Renderer::RecordGraph(recorders, rg, &m_GPUTimers);

        if (suppressDebuggerMetadata)
            m_System.GetFrameDebugger().state = savedDbgState;

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
            // Per-draw replay re-derives inputs from CapturedDrawCall + frozen indirect/object SSBOs.

            m_System.GetFrameDebugger().capturedFrame.resources      = m_GraphSnapshot.resources;
            m_System.GetFrameDebugger().capturedFrame.totalGpuTimeMs = m_GraphSnapshot.totalGpuTimeMs;

            // Copy per-pass GPU times into captured passes
            {
                u32 capturedIdx = 0;
                for (auto& ps : m_GraphSnapshot.passes)
                {
                    if (ps.culled) continue;
                    if (capturedIdx < m_System.GetFrameDebugger().capturedFrame.passes.size())
                        m_System.GetFrameDebugger().capturedFrame.passes[capturedIdx].gpuTimeMs = ps.gpuTimeMs;
                    capturedIdx++;
                }
            }

            // Snapshot capture-time camera viewProj for the Frozen-state auto-recapture comparison (see top of Update).
            m_System.GetFrameDebugger().FinalizeCapture(m_Global.GetCachedViewProj());

            // Stamp CSM state into the captured frame so the cascade detail panel always shows GPU-true values
            // from the moment of capture, even if the user later twiddles light settings on the live editor side.
            auto& cf = m_System.GetFrameDebugger().capturedFrame;
            cf.cascadeSplitsViewZ = m_Global.GetCascades().splitsViewZ;
            cf.shadowBias         = m_Global.GetShadowParams().shadowBias;
            cf.shadowNormalBias   = m_Global.GetShadowParams().shadowNormalBias;
            cf.cascadeTexelSize   = m_Global.GetCascades().texelSize;
            for (u32 i = 0; i < k_ShadowCascadeCount; ++i)
                cf.lightSpaceMatrix[i] = m_Global.GetCascades().lightSpaceMatrix[i];

            // Snapshot captured-view metadata + Set 0 binding sources for replay.
            // invariant: replay reads these instead of m_CurrentViewResources / live IBL
            // textures, since the live state reflects whichever view ran last and IBL
            // can change mid-Freeze.
            cf.capturedView.targets         = view.targets;
            cf.capturedView.id = view.id;
            cf.capturedView.resourceGeneration = m_CurrentViewResources ? m_CurrentViewResources->generation : 0;
            cf.capturedView.viewIndex       = view.viewIndex;
            if (view.targets && view.targets->GetSceneColor())
            {
                cf.capturedView.width  = view.targets->GetSceneColor()->GetWidth();
                cf.capturedView.height = view.targets->GetSceneColor()->GetHeight();
            }
            m_Global.GetLastUboBytes(cf.capturedGlobalUboBytes);
            cf.capturedIrradiance     = m_Lighting.GetIrradianceMap();
            cf.capturedPrefiltered    = m_Lighting.GetPrefilteredMap();
            cf.capturedBRDF           = m_Lighting.GetBRDFLut();
            const auto* gtaoState = GetGtaoViewState(view.id);
            cf.capturedGTAOFinal      = gtaoState ? gtaoState->finalAO : nullptr;
            cf.capturedIblIntensity   = view.camera.iblIntensity;
            cf.capturedSkyboxIntensity = view.camera.skyboxIntensity;
            // Resolve descendants once at capture; replay reads this without touching m_CurrentView (stack-allocated, dangles in Frozen).
            {
                std::unordered_set<entt::entity> resolved;
                m_EditorOverlays.CollectSelectedHandles(view.camera.selectedEntities, resolved);
                cf.capturedSelectionHandles.assign(resolved.begin(), resolved.end());
            }

            cf.valid = true;
            // Snapshot which source produced this capture so viewport overlays survive the user toggling requestedSource between captures.
            m_System.GetFrameDebugger().capturedSource = m_System.GetFrameDebugger().requestedSource;
            m_System.GetFrameDebugger().state          = DebuggerState::Frozen;
        }

        return hasComputeWork;
    }

    RG::RenderGraphSnapshot RenderPipeline::CaptureSnapshot(const RG::RenderGraph& rg)
    {
        LH_PROFILE_FUNCTION();
        auto& s = m_System;

        RG::RenderGraphSnapshot snapshot;

        // Snapshot resources
        auto& resources = const_cast<RG::RenderGraph&>(rg).GetResources();
        snapshot.resources.reserve(resources.size());
        for (auto& res : resources)
        {
            RG::ResourceSnapshot rs;
            rs.name        = res.desc.name;
            rs.width       = res.desc.width;
            rs.height      = res.desc.height;
            rs.format      = res.desc.format;
            rs.isExternal  = res.external;
            rs.isTransient = res.isTransient;
            snapshot.resources.push_back(std::move(rs));
        }

        // Snapshot passes
        auto& passes = rg.GetPasses();
        snapshot.passes.reserve(passes.size());
        for (auto& pass : passes)
        {
            RG::PassSnapshot ps;
            ps.name                = pass.name;
            ps.culled              = pass.culled;
            ps.numColorAttachments = (u32)pass.colorAttachments.size();
            ps.hasDepth            = pass.hasDepth;

            for (auto& r : pass.reads)
            {
                RG::PassSnapshotResource sr;
                sr.index = r.index;
                sr.name  = (r.index > 0 && r.index <= resources.size()) ? resources[r.index - 1].desc.name : "?";
                ps.reads.push_back(std::move(sr));
            }

            for (auto& w : pass.writes)
            {
                RG::PassSnapshotResource sw;
                sw.index = w.index;
                sw.name  = (w.index > 0 && w.index <= resources.size()) ? resources[w.index - 1].desc.name : "?";
                ps.writes.push_back(std::move(sw));
            }

            // Compute primaryOutputIndex from first color write, or depth if depth-only pass
            if (!pass.colorAttachments.empty())
            {
                u32 idx = pass.colorAttachments[0].handle.index;
                if (idx > 0 && idx <= resources.size())
                    ps.primaryOutputIndex = (int)(idx - 1);
            }
            else if (pass.hasDepth)
            {
                u32 idx = pass.depthAttachment.handle.index;
                if (idx > 0 && idx <= resources.size())
                    ps.primaryOutputIndex = (int)(idx - 1);
            }

            snapshot.passes.push_back(std::move(ps));
        }

        // Barrier inspector: fill from the solved graph when capture is on (off by default).
        if (RG::RenderGraph::BarrierCapture())
            rg.CaptureBarrierRecords(snapshot);

        // Compute geometry stats from the current DrawList (built before pass dispatch)
        u32 totalDraws = (u32)(m_System.GetDrawList().opaque.size() + m_System.GetDrawList().cutout.size() + m_System.GetDrawList().transparent.size());
        u32 totalIndices = 0;
        auto sumIndices = [&](const std::vector<DrawCommand>& draws) {
            for (auto& dc : draws)
            {
                if (!dc.model) continue;
                auto mesh = dc.model->GetMesh(dc.meshIndex);
                if (mesh && mesh->GetIndexBuffer())
                    totalIndices += mesh->GetIndexBuffer()->GetCount();
            }
        };
        sumIndices(m_System.GetDrawList().opaque);
        sumIndices(m_System.GetDrawList().cutout);
        sumIndices(m_System.GetDrawList().transparent);

        // Enrich per-pass pipeline state (known at RenderingSystem level, not RenderGraph)
        for (auto& ps : snapshot.passes)
        {
            if (ps.culled) continue;

            if (ps.name == "ShadowPass")
            {
                ps.depthTest = true; ps.depthWrite = true;
                ps.blendEnabled = false;
                ps.cullMode = VK_CULL_MODE_FRONT_BIT;
                ps.shaderName = "shadowDepth";
                ps.drawCalls = totalDraws;
                ps.indices = totalIndices;
            }
            else if (ps.name == "GeometryPass")
            {
                ps.depthTest = true; ps.depthWrite = true;
                ps.blendEnabled = false;
                ps.cullMode = VK_CULL_MODE_BACK_BIT;
                ps.shaderName = "pbr";
                ps.drawCalls = totalDraws;
                ps.indices = totalIndices;
            }
            else if (ps.name == "SkyboxPass")
            {
                ps.depthTest = true; ps.depthWrite = false;
                ps.blendEnabled = false;
                ps.cullMode = VK_CULL_MODE_BACK_BIT;
                ps.shaderName = "skybox";
                ps.drawCalls = 1; ps.indices = 0;
            }
            else if (ps.name == "PostProcess")
            {
                ps.depthTest = false; ps.depthWrite = false;
                ps.blendEnabled = false;
                ps.cullMode = VK_CULL_MODE_NONE;
                ps.shaderName = "postprocess";
                ps.drawCalls = 1; ps.indices = 0;
            }
            else if (ps.name == "ImGuiPass")
            {
                ps.depthTest = false; ps.depthWrite = false;
                ps.blendEnabled = true;
                ps.cullMode = VK_CULL_MODE_NONE;
                ps.shaderName = "imgui";
                ps.drawCalls = 0; ps.indices = 0; // ImGui manages its own draws
            }
        }

        return snapshot;
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
            for (u32 i = 0; i < ViewResources::kBloomMipCount; ++i)
                if (it->second.bloomMip[i]) m_NamedTextures["BloomMip" + std::to_string(i)] = it->second.bloomMip[i];
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

    // ---- Frame debugger: forwarders into FrameDebuggerContext ----

    VkImageView RenderPipeline::GetPerDrawPreviewView()  const { return m_Debugger->GetPerDrawPreviewView(); }
    u64         RenderPipeline::GetPerDrawPreviewKey()   const { return m_Debugger->GetPerDrawPreviewKey(); }
    u32         RenderPipeline::GetPerDrawPreviewWidth() const { return m_Debugger->GetPerDrawPreviewWidth(); }
    u32         RenderPipeline::GetPerDrawPreviewHeight()const { return m_Debugger->GetPerDrawPreviewHeight(); }
    VkImageView RenderPipeline::GetDepthPreviewView()    const { return m_Debugger->GetDepthPreviewView(); }
    u32         RenderPipeline::GetDepthPreviewWidth()   const { return m_Debugger->GetDepthPreviewWidth(); }
    u32         RenderPipeline::GetDepthPreviewHeight()  const { return m_Debugger->GetDepthPreviewHeight(); }
    void        RenderPipeline::ResetPreviewCacheKeys() { m_Debugger->ResetPreviewCacheKeys(); }

    void RenderPipeline::ReplayPassUpToDraw(u32 passIdx, u32 localDrawIdx)
    {
        m_Debugger->ReplayPassUpToDraw(passIdx, localDrawIdx);
    }

    void RenderPipeline::BlitArchivedDepthToPreview(u32 archiveIdx, int layer, float nearZ, float farZ)
    {
        m_Debugger->BlitArchivedDepthToPreview(archiveIdx, layer, nearZ, farZ);
    }

    void RenderPipeline::BlitArchivedSlimToPreview(u32 archiveIdx, u32 mode, float scale)
    {
        m_Debugger->BlitArchivedSlimToPreview(archiveIdx, mode, scale);
    }

    VkImageView RenderPipeline::GetSlimPreviewView()   const { return m_Debugger->GetSlimPreviewView(); }
    u32         RenderPipeline::GetSlimPreviewWidth()  const { return m_Debugger->GetSlimPreviewWidth(); }
    u32         RenderPipeline::GetSlimPreviewHeight() const { return m_Debugger->GetSlimPreviewHeight(); }

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

    void RenderPipeline::UpdatePostProcessUBO() { m_PostProcess.UpdateUBO(); }
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
