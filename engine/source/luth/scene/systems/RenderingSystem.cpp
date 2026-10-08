#include "luthpch.h"
#include "luth/renderer/debug/NativeDebugOutputs.h"
#include "luth/renderer/presentation/ViewPresentation.h"
#include <imgui.h>
#include "luth/renderer/shader/ShaderReloadCoordinator.h"
#include "luth/scene/systems/RenderingSystem.h"
#include "luth/scene/systems/LightingSystem.h"
#include "luth/scene/systems/SystemRegistry.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/debug/FrameDebuggerContext.h"
#include "luth/renderer/debug/CaptureFinalization.h"
#include "luth/renderer/debug/GraphInstrumentation.h"
#include "luth/renderer/debug/ViewGpuProfiler.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanBackend.h"
#include "luth/assets/FileSystem.h"
#include "luth/scene/Scene.h"
#include "luth/scene/components/Transform.h"
#include "luth/core/diagnostics/Profiler.h"

namespace Luth
{
    // ---- Construction / Destruction ----

    RenderingSystem::RenderingSystem(u32 viewportWidth, u32 viewportHeight)
    {
        m_SceneViewId = RegisterView(m_SceneTargets);
        m_FrameAllocator = std::make_unique<Memory::LinearAllocator>(1 * Memory::MB);
        m_Pipeline       = std::make_unique<RenderPipeline>(*this);
        m_Pipeline->Initialize(viewportWidth, viewportHeight);
        m_GpuProfiler = std::make_unique<ViewGpuProfiler>();
        m_CaptureContext = std::make_unique<FrameDebuggerContext>(*this, m_Pipeline->GetGeometry(),
            m_Pipeline->GetLighting(), m_Pipeline->GetPostProcess(), m_Pipeline->GetEditorOverlays());
        m_ShaderReload = std::make_unique<ShaderReloadCoordinator>();
        if (Renderer::GetBackend()->GetAPI() == RenderBackend::API::Vulkan)
        {
            m_Pipeline->RegisterShaderReloadConsumers(*m_ShaderReload);
            m_ShaderReload->AddConsumer("CapturePreview", [this](const auto& name, const auto& spv) {
                if (name == "debugBlit.slang") m_FrameDebugger.blitFragSpv = spv;
                else if (name == "debugDepth.slang") m_FrameDebugger.depthFragSpv = spv;
                else return false;
                return true;
            });
            m_ShaderReload->Start(FileSystem::EngineAssetsPath("shaders"));
        }
    }

    RenderingSystem::~RenderingSystem()
    {
        // Replay borrows native domains and descriptor resources. Retire capture first,
        // after completion, while those dependencies and the Vulkan device are alive.
        m_ShaderReload->Stop();
        Renderer::WaitForGPU();
        m_GpuProfiler->Shutdown();
        m_CaptureContext->Shutdown();
        m_FrameDebugger.Shutdown(VulkanContext::Get().GetDevice());
        m_CaptureContext.reset();
        m_DebugOutputs.Clear();
        m_Pipeline->Shutdown();
    }

    void RenderingSystem::ReloadSkybox(const fs::path& hdrPath)
    {
        m_Pipeline->ReloadSkybox(hdrPath);
        m_DebugOutputs.ReplaceShared(CollectSharedDebugOutputs(m_Pipeline->GetLighting()));
    }

    std::shared_ptr<Texture> RenderingSystem::GetNamedTexture(const std::string& name) const
    {
        const auto* view = m_Views.Get(m_SceneViewId);
        return view ? GetNamedTexture(view->id, view->generation, name) : nullptr;
    }

    std::shared_ptr<Texture> RenderingSystem::GetNamedTexture(RenderViewId id, u64 generation, const std::string& name) const
    {
        const auto* view = m_Views.Get(id);
        if (!view || view->generation != generation) return {};
        return m_DebugOutputs.Find(id, generation, name);
    }

    void RenderingSystem::RefreshViewDebugOutputs(RenderViewId id, FrameTargets& targets)
    {
        const auto* view = m_Views.Get(id);
        if (!view || view->targets != &targets) throw std::invalid_argument("Debug outputs require the registered view owner");
        m_DebugOutputs.ReplaceShared(CollectSharedDebugOutputs(m_Pipeline->GetLighting()));
        m_DebugOutputs.ReplaceView(id, view->generation, CollectViewDebugOutputs(targets,
            m_Pipeline->GetViewResources(&targets), m_Pipeline->GetGtaoViewState(id)));
    }

    void RenderingSystem::ReplayPassUpToDraw(u32 passIdx, u32 localDrawIdx)
    {
        m_CaptureContext->ReplayPassUpToDraw(passIdx, localDrawIdx);
    }

    void RenderingSystem::BlitArchivedDepthToPreview(u32 archiveIdx, int layer, float nearZ, float farZ)
    {
        m_CaptureContext->BlitArchivedDepthToPreview(archiveIdx, layer, nearZ, farZ);
    }

    const RG::RenderGraphSnapshot& RenderingSystem::GetGraphSnapshot() const
    {
        return m_GraphSnapshot;
    }

    RG::RenderGraphSnapshot& RenderingSystem::CaptureGraphSnapshot(const RG::RenderGraph& graph, RenderViewId id, u64 generation)
    {
        m_GraphSnapshot = Luth::CaptureGraphSnapshot(graph, m_DrawList);
        m_GraphSnapshot.viewId = id.value;
        m_GraphSnapshot.resourceGeneration = generation;
        return m_GraphSnapshot;
    }

    GPUTimerPool* RenderingSystem::PrepareViewProfiling(RenderViewId id, u64 generation, u64 frame,
        const RG::RenderGraph& graph, RG::RenderGraphSnapshot& snapshot, bool applyPrevious)
    {
        m_Profiling.Prepare(id, generation, frame, graph);
        return m_GpuProfiler->Prepare(*m_Profiling.Find(id), snapshot, applyPrevious);
    }

    void RenderingSystem::SubmitViewProfiling(RenderViewId id, u64 frame, SubmissionCompletionToken token)
    {
        if (m_Profiling.Submit(id, frame, token)) m_GpuProfiler->Submit(id, frame, token);
    }
    void RenderingSystem::AppendViewPresentation(RG::RenderGraph& graph, RG::ResourceHandle finalLdr, bool emitImGui)
    {
        if (finalLdr.IsValid()) AddViewOutputExport(graph, finalLdr);
        else if (!emitImGui) throw std::invalid_argument("Secondary view requires final LDR output");
        if (!emitImGui) return;
        auto& swapchain = static_cast<VulkanBackend*>(Renderer::GetBackend())->GetSwapchain();
        const auto imageIndex = swapchain.GetCurrentFrameIndex();
        ViewPresentationInputs inputs;
        inputs.backbuffer.name = "Backbuffer";
        inputs.backbuffer.width = swapchain.GetExtent().width;
        inputs.backbuffer.height = swapchain.GetExtent().height;
        inputs.backbuffer.format = RG::TextureFormat::BGRA8_Unorm;
        inputs.image = (void*)swapchain.GetImage(imageIndex);
        inputs.imageView = (void*)swapchain.GetImageView(imageIndex);
        inputs.drawData = ImGui::GetDrawData();
        AddViewImGuiPass(graph, inputs, m_FrameDebugger, finalLdr);
    }

    void RenderingSystem::ExecuteMinimal()
    {
        RG::RenderGraph graph(GetFrameAllocator());
        AppendViewPresentation(graph, {}, true); // Frozen outputs already sampled; no duplicate imports.
        graph.Compile();
        Renderer::ExecuteGraph(graph, Renderer::GetFrameData()->GetFrameIndex(), nullptr);
    }
    void RenderingSystem::BeginViewCapture(const RenderView& view)
    {
        if (!view.captureRequested || m_FrameDebugger.state != DebuggerState::CaptureRequested) return;
        m_CaptureContext->InitDebugBlitResources();
        m_CaptureContext->ResetPreviewCacheKeys();
        m_FrameDebugger.BeginCapture(VulkanContext::Get().GetDevice(), VulkanContext::Get().GetAllocator());
    }

    bool RenderingSystem::FinalizeViewCapture(const CaptureFinalizationInputs& inputs,
        const RG::RenderGraphSnapshot& snapshot)
    {
        return Luth::FinalizeViewCapture(m_FrameDebugger, inputs, snapshot, m_Views);
    }

    void RenderingSystem::ResetPreviewCacheKeys()
    {
        m_CaptureContext->ResetPreviewCacheKeys();
    }

    void RenderingSystem::ExitCapture()
    {
        // Free GPU-owned archives BEFORE clearing the metadata vectors.
        m_FrameDebugger.DestroyArchives();
        m_FrameDebugger.state = DebuggerState::Inactive;
        m_FrameDebugger.capturedFrame.Clear();
        // Drop the per-draw replay cache key so the next capture starts clean; the preview texture
        // itself is reused across captures.
        m_CaptureContext->ResetPreviewCacheKeys();
    }

    VkImageView RenderingSystem::GetPerDrawPreviewView()   const { return m_CaptureContext->GetPerDrawPreviewView(); }
    u64         RenderingSystem::GetPerDrawPreviewKey()    const { return m_CaptureContext->GetPerDrawPreviewKey(); }
    u32         RenderingSystem::GetPerDrawPreviewWidth()  const { return m_CaptureContext->GetPerDrawPreviewWidth(); }
    u32         RenderingSystem::GetPerDrawPreviewHeight() const { return m_CaptureContext->GetPerDrawPreviewHeight(); }
    VkImageView RenderingSystem::GetDepthPreviewView()     const { return m_CaptureContext->GetDepthPreviewView(); }
    u32         RenderingSystem::GetDepthPreviewWidth()    const { return m_CaptureContext->GetDepthPreviewWidth(); }
    u32         RenderingSystem::GetDepthPreviewHeight()   const { return m_CaptureContext->GetDepthPreviewHeight(); }

    void RenderingSystem::BlitArchivedSlimToPreview(u32 archiveIdx, u32 mode, float scale)
    {
        m_CaptureContext->BlitArchivedSlimToPreview(archiveIdx, mode, scale);
    }
    VkImageView RenderingSystem::GetSlimPreviewView()      const { return m_CaptureContext->GetSlimPreviewView(); }
    u32         RenderingSystem::GetSlimPreviewWidth()     const { return m_CaptureContext->GetSlimPreviewWidth(); }
    u32         RenderingSystem::GetSlimPreviewHeight()    const { return m_CaptureContext->GetSlimPreviewHeight(); }

    // ---- Project lifecycle ----

    void RenderingSystem::OnProjectLoaded()
    {
        if (!FileSystem::HasProject()) return;
        m_ShaderReload->AddProjectDir(FileSystem::AssetsPath("shaders"));
    }

    void RenderingSystem::OnProjectUnloaded()
    {
        m_ShaderReload->RemoveProjectDir();
    }

    // ---- Per-frame dispatcher ----

    void RenderingSystem::Update(Scene* scene)
    {
        LH_PROFILE_FUNCTION();
        (void)scene; // Render path reads the snapshot, not the registry.

        m_FrameAllocator->Reset();

        // Drain pending shader reloads once per frame (FileWatcher detections from its bg thread).
        // Formerly lived inside RenderPipeline::Execute and ran twice per frame when both Scene + Game
        // viewports were open.
        m_ShaderReload->Poll();

        // ---- Frame Debugger: Frozen state ----
        // Strict snapshot model with auto-recapture on camera move.
        //
        // While Frozen, the live render graph is NOT rebuilt or re-executed. The LDR output target
        // retains the LAST CAPTURED image (no other code writes it in this state), so the editor's
        // ScenePanel (which samples it through ImGui) keeps showing the GPU-true captured frame.
        //
        // Each Frozen tick recomputes the camera viewProj cheaply (no GPU upload) and bit-compares
        // against captureViewProj. A mismatch means the camera moved: flip the state machine back to
        // CaptureRequested and fall through to the normal capture flow below; FrameDebugger::BeginCapture
        // tears down the prior archives.
        if (m_FrameDebugger.state == DebuggerState::Frozen)
        {
            if (Renderer::GetBackend()->GetAPI() != RenderBackend::API::Vulkan) return;

            // Auto-recapture-on-camera-move is only meaningful for Scene captures: the comparison camera
            // (m_CameraParams = editor) matches the source. For Game captures the captureViewProj came
            // from the game camera, so this comparison would always report "moved" and loop the state
            // machine every frame. Game captures stay Frozen until the user explicitly disables.
            bool cameraMoved = false;
            if (m_FrameDebugger.capturedSource == CaptureSource::Scene)
            {
                // Mirror the Vulkan Y-flip from UpdateGlobalUniforms so the comparison matches the GPU's
                // view at capture time.
                Mat4 currentProj = m_CameraParams.projection;
                currentProj[1][1] *= -1.0f;
                Mat4 currentViewProj = currentProj * m_CameraParams.view;

                // Pack viewProj + IBL intensities into one struct for a single memcmp. Catches user
                // inspector tweaks to Sun/Sky settings mid-Freeze. Cascade splits / shadow bias would
                // need the lighting system to recompute during Frozen; out of scope.
                struct CompareKey
                {
                    Mat4  viewProj;
                    float iblIntensity;
                    float skyboxIntensity;
                    float _pad[2] = { 0.0f, 0.0f };
                };
                CompareKey live{};
                live.viewProj        = currentViewProj;
                live.iblIntensity    = m_CameraParams.iblIntensity;
                live.skyboxIntensity = m_CameraParams.skyboxIntensity;
                CompareKey captured{};
                captured.viewProj        = m_FrameDebugger.capturedFrame.captureViewProj;
                captured.iblIntensity    = m_FrameDebugger.capturedFrame.capturedIblIntensity;
                captured.skyboxIntensity = m_FrameDebugger.capturedFrame.capturedSkyboxIntensity;
                cameraMoved = std::memcmp(&live, &captured, sizeof(CompareKey)) != 0;

                // Throttle to ~10 Hz at 60 fps. Per-recapture GPU work (~10 vkCmdCopyImage + barriers,
                // mostly cascade depth) saturates mid-tier GPUs at frame rate; 6x less keeps the editor
                // smooth without visibly stale overlays.
                static constexpr u64 k_AutoRecaptureMinIntervalFrames = 6;
                if (cameraMoved)
                {
                    const u64 currentFrame = Renderer::GetFrameData()->GetFrameIndex();
                    if (currentFrame - m_FrameDebugger.lastRecaptureFrameIndex
                        < k_AutoRecaptureMinIntervalFrames)
                        cameraMoved = false;
                }
            }

            if (!cameraMoved)
            {
                // Static or throttled: minimal graph, just blit ImGui to the swapchain. Drop queued
                // views; letting the queue grow unbounded spikes the frame when the debugger exits.
                m_QueuedViews.clear();
                ExecuteMinimal();
                return;
            }

            // Camera moved: re-trigger capture and fall through.
            m_FrameDebugger.state = DebuggerState::CaptureRequested;
            m_FrameDebugger.lastRecaptureFrameIndex = Renderer::GetFrameData()->GetFrameIndex();
        }

        if (Renderer::GetBackend()->GetAPI() != RenderBackend::API::Vulkan)
            return;

        // Snapshot is captured at end of game stage; render stage only reads.
        const RenderSnapshot& snapshot = Renderer::GetFrameData()->RenderFrame().Snapshot;
        m_ActiveSnapshot = &snapshot;

        // Build GPU object buffer (after materials are registered)
        m_Pipeline->BuildGPUObjectBuffer(snapshot);

        // Partition snapshot mesh rows into opaque/cutout/transparent buckets. Must follow
        // BuildGPUObjectBuffer so gpuObjectIndex/entityIndex reference the freshly populated indirect buffer.
        m_DrawListBuilder.Build(snapshot, m_Pipeline->GetMaterialSlotMap(), m_Pipeline->GetEntityToSSBOIndex(), m_DrawList);

        // LightGatherer + CSM fit. Set 3 is per-view now (cluster grid + light index differ per
        // view), so the LightSSBO upload + Set 3 binding writes happen inside BuildGraph per view.
        if (auto* lighting = SystemRegistry::GetSystem<LightingSystem>())
            lighting->UpdateFor(snapshot, m_CameraParams, m_EmissiveLightSettings, m_RestirSettings.enabled);

        // Primary view: always rendered, emits the per-frame ImGui pass.
        RenderView sceneView;
        sceneView.id                   = m_SceneViewId;
        sceneView.targets              = &m_SceneTargets;
        sceneView.camera               = m_CameraParams;
        sceneView.viewIndex            = 0;
        sceneView.drawGrid             = m_GridVisible;
        sceneView.drawSelectionOutline = true;
        sceneView.drawDebugShapes      = true;
        sceneView.emitImGuiPass        = true;
        // Capture-source gate: only the scene view installs the archive sink when the user has chosen
        // Scene as the source. Game capture lives on GamePanel's queued view.
        sceneView.captureRequested     = (m_FrameDebugger.state == DebuggerState::CaptureRequested
                                          && m_FrameDebugger.requestedSource == CaptureSource::Scene);

        // Per-view 3-submit topology: each view gets its own gA / compute / gB primary cmd buffers, submitted with
        // timeline-semaphore waits at boundaries. Queued views record first (their LDRs are sampled by the scene
        // view's ImGui pass), then the scene view closes with the ImGui pass + present barrier. Cross-view ordering
        // for shared resources (m_ShadowMap, IBL maps) is enforced by view K+1's gA submit waiting on view K's gB
        // signal at EARLY_FRAGMENT_TESTS_BIT.
        std::erase_if(m_QueuedViews, [this](const RenderView& view) {
            const auto* registered = m_Views.Get(view.id);
            return !registered || registered->targets != view.targets;
        });
        const u64 frameIndex  = Renderer::GetFrameData()->GetFrameIndex();
        const u32 totalViews  = (u32)m_QueuedViews.size() + 1;  // queued + scene view
        LH_CORE_ASSERT(totalViews <= MAX_VIEWS_PER_FRAME, "view count exceeds MAX_VIEWS_PER_FRAME");
        u32 viewSlot = 0;

        for (const RenderView& v : m_QueuedViews)
        {
            QueueRecorders r = Renderer::BeginPrimaryCmd(frameIndex, viewSlot);
            const bool hasCompute = RecordView(v, r);
            SubmitViewProfiling(v.id, Renderer::GetFrameData()->GetRenderFrameIndex(), Renderer::EndPrimaryCmdAndSubmit(r, frameIndex, viewSlot, hasCompute, /*isLastView=*/false));
            if (auto* state = m_Pipeline->GetViewResources(v.targets))
            {
                state->cameraHistory.Commit(Renderer::GetFrameData()->GetRenderFrameIndex(), state->generation);
                if (state->taa && state->taa->recorded)
                    state->taa->history.Commit(Renderer::GetFrameData()->GetRenderFrameIndex(), state->generation);
                else if (state->taa) state->taa->history.Invalidate();
            }
            ++viewSlot;
        }
        m_QueuedViews.clear();

        QueueRecorders r = Renderer::BeginPrimaryCmd(frameIndex, viewSlot);
        const bool hasCompute = RecordView(sceneView, r);
        SubmitViewProfiling(sceneView.id, Renderer::GetFrameData()->GetRenderFrameIndex(), Renderer::EndPrimaryCmdAndSubmit(r, frameIndex, viewSlot, hasCompute, /*isLastView=*/true));
        if (auto* state = m_Pipeline->GetViewResources(sceneView.targets))
        {
            state->cameraHistory.Commit(Renderer::GetFrameData()->GetRenderFrameIndex(), state->generation);
            if (state->taa && state->taa->recorded)
                state->taa->history.Commit(Renderer::GetFrameData()->GetRenderFrameIndex(), state->generation);
            else if (state->taa) state->taa->history.Invalidate();
        }
    }

    // ---- Per-view record ----

    bool RenderingSystem::RecordView(const RenderView& view, QueueRecorders recorders)
    {
        LH_PROFILE_FUNCTION();

        if (!view.targets || Renderer::GetBackend()->GetAPI() != RenderBackend::API::Vulkan)
            return false;

        // Cascade fit is camera-dependent so this refits per view (~1 ms GPU with game panel open;
        // frustum-union fit is backlog). m_Lights was already gathered once in Update before this loop;
        // UpdateFor here only needs the cascade rebuild for view.camera. (Re-gathering m_Lights from
        // the same snapshot is idempotent; left as a no-cost guard against future signature drift.)
        auto* lighting = SystemRegistry::GetSystem<LightingSystem>();
        lighting->UpdateFor(Renderer::GetFrameData()->RenderFrame().Snapshot, view.camera,
                            m_EmissiveLightSettings, m_RestirSettings.enabled);

        // Must precede the per-view UBO writes below; they read m_CurrentViewResources, which PrepareForTargets sets.
        m_Pipeline->PrepareForTargets(*view.targets);

        // Light UBO (Set 3) is hoisted to Update: view-independent, and a single global Set 3 would
        // race across views otherwise.
        m_Pipeline->UpdateGlobalUniforms(view.camera, lighting->GetCascades(), lighting->GetShadowParams());

        m_Pipeline->UpdateGTAOUBO();

        return m_Pipeline->Execute(view, recorders);
    }

    // ---- Resize ----

    void RenderingSystem::Resize(u32 width, u32 height)
    {
        LH_PROFILE_FUNCTION();

        // Guard against unsigned underflow from negative float->u32 casts at startup
        if (m_SceneTargets.IsAllocated() && width > 0 && height > 0 && width <= 16384 && height <= 16384)
        {
            // Drain GPU + drop ViewResources before swapping textures; see GamePanel::SetOnResize for
            // the same hazard description.
            ResizeView(m_SceneViewId, m_SceneTargets, width, height);
            m_Pipeline->OnResize(width, height);
        }
    }

    u64 RenderingSystem::InvalidateView(RenderViewId id)
    {
        const u64 generation = m_Views.Invalidate(id);
        m_DebugOutputs.Release(id);
        if (m_FrameDebugger.capturedFrame.capturedView.id == id) ExitCapture();
        return generation;
    }

    void RenderingSystem::ResizeView(RenderViewId id, FrameTargets& targets, u32 width, u32 height)
    {
        const auto* registered = m_Views.Get(id);
        if (!registered || registered->targets != &targets)
            throw std::invalid_argument("Resize requires the registered view owner");
        if (!width || !height || width > 16384 || height > 16384) return;
        if (targets.GetSceneColor() && targets.GetSceneColor()->GetWidth() == width &&
            targets.GetSceneColor()->GetHeight() == height) return;
        Renderer::WaitForGPU();
        InvalidateView(id);
        m_Pipeline->ReleaseViewResources(targets);
        if (targets.IsAllocated()) targets.Resize(width, height);
        else targets.Allocate(width, height);
    }

    void RenderingSystem::ReleaseView(RenderViewId id)
    {
        const auto* registered = m_Views.Get(id);
        if (!registered) return;
        Renderer::WaitForGPU();
        if (m_FrameDebugger.capturedFrame.capturedView.id == id) ExitCapture();
        auto* targets = static_cast<FrameTargets*>(const_cast<void*>(registered->targets));
        m_Pipeline->ReleaseViewResources(*targets);
        std::erase_if(m_QueuedViews, [id](const RenderView& view) { return view.id == id; });
        m_GpuProfiler->Release(id);
        m_Profiling.Release(id);
        m_DebugOutputs.Release(id);
        m_Views.Release(id);
    }
}
