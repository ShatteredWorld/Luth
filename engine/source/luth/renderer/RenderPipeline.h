#pragma once

#include "luth/core/types/LuthMath.h"
#include "luth/core/UUID.h"
#include "luth/renderer/CameraParams.h"
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/features/rt/RtSceneFeature.h"
#include "luth/renderer/QueueRecorders.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/renderer/rendergraph/RenderGraphSnapshot.h"
#include "luth/renderer/lighting/LightTypes.h"
#include "luth/renderer/backend/vulkan/VulkanPipeline.h"
#include "luth/renderer/backend/vulkan/VulkanComputePipeline.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/renderer/pipeline/PipelineManager.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/renderer/shader/ShaderWatcher.h"
#include "luth/renderer/subsystems/GlobalSubsystem.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"
#include "luth/renderer/subsystems/GTAOSubsystem.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
#include "luth/renderer/subsystems/DebugDrawSubsystem.h"
#include "luth/renderer/subsystems/RtSubsystem.h"
#include "luth/renderer/subsystems/RtRestirSubsystem.h"
#include "luth/renderer/subsystems/RtRestirGiSubsystem.h"
#include "luth/renderer/subsystems/SlangParityGuard.h"
#include "luth/renderer/subsystems/PathTraceSubsystem.h"
#include "luth/renderer/subsystems/ReflectionsSubsystem.h"
#include "luth/renderer/subsystems/SkinningSubsystem.h"
#include "luth/renderer/subsystems/IDenoiser.h"
#include "luth/memory/GPUTaggedPageAllocator.h"

#include <entt/entt.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Luth
{
    class Entity;
    class FrameTargets;
    class Material;
    class RenderingSystem;

    struct GeometryOutput;
    struct RenderSnapshot;
    namespace fs = std::filesystem;

    // Per-view input to RenderPipeline::Execute. One RenderView per visible viewport. targets is non-owning.
    //
    // viewIndex selects the view's slice of the shared indirect buffer:
    // regions [viewIndex * k_IndirectRegionsPerView, +k_IndirectRegionsPerView).
    // emitImGuiPass: only the primary view records the ImGui pass (backbuffer write + ImGui::GetDrawData,
    // once per frame, after ImGui::Render).
    struct RenderView
    {
        RenderViewId  id;
        FrameTargets* targets              = nullptr;
        CameraParams  camera;
        u32           viewIndex            = 0;
        bool          drawGrid             = true;
        bool          drawSelectionOutline = true;
        bool          drawDebugShapes      = false;  // off by default; scene view enables explicitly
        bool          emitImGuiPass        = true;
        // Set by the view's owner when the user has requested a Frame Debugger capture and selected this
        // view (Scene or Game) as the source. Drives the BeginCapture / archive-sink wiring in
        // RenderPipeline::Execute. Decoupled from emitImGuiPass so capture can target the game view.
        bool          captureRequested     = false;
    };

    // Legacy GPU resource bridge bound to a registered view. Keyed by stable view ID in
    // RenderPipeline::m_ViewResources, allocated on first use by EnsureViewResources, recreated on size
    // change, destroyed on ReleaseViewResources or pipeline shutdown. Having a distinct set per view lets
    // multiple subgraphs share one primary command buffer without mid-frame vkUpdateDescriptorSets aliasing.
    struct ViewResources
    {
        // Registration identity survives resize; generation invalidates incompatible replay.
        u64 id     = 0;
        u64 generation = 0;
        u32 width  = 0;
        u32 height = 0;

        // Owns directly stored descriptor sets; domain-state bridges carry their own pools.
        VkDescriptorPool descPool = VK_NULL_HANDLE;

        // Set 0 descriptor: bindings 0 (Global UBO) + 5 (GTAO UBO) are rebound per render-stage to fresh
        // GPUTaggedPageAllocator regions in UpdateGlobalUniforms / UpdateGTAOUBO against per-frame slot.
        // IBL samplers (1-3) and GTAO final sampler (4) are stable, replicated across all slots at WriteView time.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> globalDescriptorSet{};

        std::shared_ptr<BloomViewState> bloom; // Borrowed postprocessing domain state.

        // Compatibility bridge; the volumetric domain owns allocation and lifecycle.
        std::shared_ptr<FogViewState> fog;

        // Composite remains cycled; native preparation rebinds its per-frame uniform slot.
        std::shared_ptr<CompositeViewState> composite;

        // Editor overlays: allocated for every view, bound only by the scene view
        // (game view's subgraph skips both passes via flags).
        std::shared_ptr<EditorOverlayViewState> overlays; // Borrowed native editor domain state.

        // Slim G-buffer live viz (ShadeMode toggle). Single set, written once at AllocateViewResources
        // time pointing at the 4 slim FrameTargets. Bindings: 0=normal, 1=roughness, 2=motion, 3=matID.
        std::shared_ptr<SlimVizViewState> slimViz; // Borrowed postprocess domain state.

        // Forward+ cluster compute descriptor sets. Cycled: each frame the cluster AABB + grid
        // tagged-heap regions get rewritten into the slot's bindings before dispatch.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> clusterBuildDescSet{};
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> lightAssignDescSet{};

        // Set 3 (Lighting). Per-view because cluster grid + light index are per-view; LightSSBO
        // also lives in a per-view tagged-heap region. b3 (shadow sampler) written once at view
        // alloc time, propagates to all slots. b0/b1/b2 rebound each frame by UploadLightingResources.
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> lightDescSet{};

        std::shared_ptr<TransparencyViewState> transparency; // Borrowed domain state during migration.

        // Cluster debug viz: domain-owned depth descriptor and retained SceneDepth.
        // by WriteClusterVizView at AllocateViewResources time.
        std::shared_ptr<ClusterVizViewState> clusterViz; // Borrowed lighting domain state.

        // Per-view previous-frame view-projection; feeds ubo.prevViewProjection for motion vectors.
        // GlobalSubsystem::m_CachedViewProj is shared across views, so multi-view rendering (Scene +
        // Game panel) cross-contaminates the prev-VP. Per-view storage keeps each view's prev-VP
        // independent. Identity-initialized, so frame 0 has nonsense motion; settles by frame 1.
        Mat4 prevViewProj{ 1.0f };
        ViewHistoryState cameraHistory;
        // Per-view previous-frame camera position; feeds ubo.prevCameraPos for DI temporal BASIC's
        // view-dependent spec target. Per-view for the same multi-view reason as prevViewProj.
        Vec3 prevCameraPos{ 0.0f };
        // Un-jittered twin of prevViewProj; pairs into GlobalSubsystem's cached sky reprojection
        // (prevVP * inv(currVP)) for the TAA resolve's depth==1 fallback (sky rasterizes no motion).
        Mat4 prevViewProjNoJitter{ 1.0f };

        // Prev-frame near/far for the resolve pass's reprojection slice math; needed because the
        // current frame's nearZ/farZ may differ if camera FOV/clip planes animate.
        f32 prevNearZ = 0.0f;
        f32 prevFarZ  = 0.0f;

        // TAA owns histories and descriptors. Camera jitter remains in the legacy view bridge.
        std::shared_ptr<TaaViewState> taa; // Borrowed postprocessing domain state.
        Vec2 currentJitter{ 0.0f, 0.0f };
        Vec2 prevJitter{ 0.0f, 0.0f };

        // RT sun-shadow mask: viewport-sized R8 storage image, written by raygen on
        // AsyncCompute and sampled by pbr.frag (Set 3 binding 4) when ShadowingMode::RtShadows is
        // active. Like TAA histories, lifetime is: persistent, recreated on resize. The cycled
        // descriptor set carries the pass-local bindings (SceneDepth + slimNormal + mask storage).
        std::shared_ptr<Texture> sunShadowMask;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> rtShadowPassDescSet{};

        // ReSTIR DI (Bitterli 2020). restirReservoir is a SINGLE Garlic device-local large-tagged
        // scratch buffer (w*h*32 B): initial writes it, temporal merges history into it in-place
        // (Set 2 b2); same-frame lifetime only. Destroyed on resize via FreeTagAndDestroy; tags stay
        // in NextReservoirTag's reserved high range, disjoint from the per-frame FreeTag(N-2) sweep.
        // restirDI is viewport-sized rgba16f STORAGE+SAMPLED (demodulated diffuse irradiance,
        // consumed by pbr.frag Set 3 b5). The cycled set carries Set 2's depth/normal + motion
        // samplers + reservoir SSBOs + DI storage image.
        Memory::GPUSubRegion restirReservoir{};
        u32 restirReservoirTag = 0;
        // Spatial-reuse output AND temporal history (post-spatial topology): temporal reads it as
        // prev (Set 2 b4), spatial overwrites it (b6, RG WAR barrier on the shared import), shade
        // consumes it; it then persists as next frame's history with final visibility already folded
        // into W. Reserved high tag, freed only on resize/destroy.
        Memory::GPUSubRegion restirSpatial{};
        u32 restirSpatialTag = 0;
        std::shared_ptr<Texture> restirDI;
        // Demodulated specular DI: rgb = Li*[D*G/(4*NoL*NoV)]*NdotL*W (F0-free, remodulated by
        // pbr.frag's split-sum envBRDF at Set 3 b8). Shade writes it at Set 2 b8; the DiSpecular SVGF
        // channel denoises it. Same shape as restirDI; written every frame so no bootstrap clear.
        std::shared_ptr<Texture> restirDISpec;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> restirDescSet{};

        // ReSTIR GI (Ouyang 2021): sibling of the DI buffers above, w*h*64 B each (GIReservoir is
        // a world-space path vertex, not a light index). Same scratch + spatial/history shape;
        // tags mint from RtRestirGiSubsystem's disjoint 0xFFFF8000 reserved range. restirGiDI is the
        // viewport-sized rgba16f STORAGE+SAMPLED demodulated indirect-diffuse image (pbr.frag Set 3 b6).
        Memory::GPUSubRegion restirGiReservoir{};
        u32 restirGiReservoirTag = 0;
        Memory::GPUSubRegion restirGiSpatial{};
        u32 restirGiSpatialTag = 0;
        std::shared_ptr<Texture> restirGiDI;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> restirGiDescSet{};
        VkDescriptorSet giReservoirVizDescSet = VK_NULL_HANDLE;  // ShadeMode::RestirGiReservoir debug viz (b0 depth, b1 spatial reservoir)

        // SVGF denoiser output: viewport-sized RGBA16F STORAGE+SAMPLED, same shape as restirDI. The
        // denoiser reads restirDI (noisy demodulated DI) and writes the denoised result here; pbr.frag
        // Set 3 b5 samples THIS (not restirDI), so the denoiser owns the slot whenever ReSTIR is on and
        // the A/B is denoise-vs-raw with no binding swap. History + per-pass sets grow as the SVGF
        // passes land. see arch/rendering-pipeline.md
        std::shared_ptr<Texture> svgfDenoised;
        VkDescriptorSet svgfPassthroughDescSet = VK_NULL_HANDLE;

        // SVGF temporal history: RGBA16F storage images kept in GENERAL, ping-pong by frame parity
        // (curr = [p], prev = [p^1]). colorHist = integrated color (rgb) + variance (a); moments =
        // (mu1, mu2, histLen); geom = (linearZ, octN.x, octN.y) for the disocclusion test.
        // Bootstrap-cleared so frame 0's prev read is well-defined. The two reproject sets are pre-built
        // per parity (set[p] reads [p^1], writes [p]); AddPasses binds svgfReprojectDescSet[frameAbs & 1].
        std::shared_ptr<Texture> svgfColorHist[2];
        std::shared_ptr<Texture> svgfMoments[2];
        std::shared_ptr<Texture> svgfGeom[2];
        VkDescriptorSet svgfReprojectDescSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

        // A-trous ping-pong (RGBA16F storage, GENERAL). Moments writes svgfAtrous[0] (a-trous level-0
        // input); the wavelet levels ping-pong [0]/[1] by iteration parity, the final level also writes
        // svgfDenoised. Moments/a-trous sets are pre-built per parity, bound by index; no UAB rewrite.
        std::shared_ptr<Texture> svgfAtrous[2];
        VkDescriptorSet svgfMomentsDescSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkDescriptorSet svgfAtrousDescSet[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };

        // ReSTIR GI SVGF: flat parallel set to the DI fields above (mirroring restirDI/restirGiDI).
        // A second SvgfDenoiser instance (DenoiserChannel::Gi) drives these; svgfGiDenoised feeds Set 3
        // b6. Same shapes/clears as DI. see arch/rendering-pipeline.md
        std::shared_ptr<Texture> svgfGiDenoised;
        VkDescriptorSet svgfGiPassthroughDescSet = VK_NULL_HANDLE;
        std::shared_ptr<Texture> svgfGiColorHist[2];
        std::shared_ptr<Texture> svgfGiMoments[2];
        std::shared_ptr<Texture> svgfGiGeom[2];
        VkDescriptorSet svgfGiReprojectDescSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        std::shared_ptr<Texture> svgfGiAtrous[2];
        VkDescriptorSet svgfGiMomentsDescSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkDescriptorSet svgfGiAtrousDescSet[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        // Half-res GI: svgfGi* history + reservoirs allocate at half extent; the a-trous final writes
        // svgfGiHalf, a bilateral upscale resolves it into the full-res svgfGiDenoised. giHalfCached
        // drives EnsureViewResources realloc on a runtime toggle.
        std::shared_ptr<Texture> svgfGiHalf;
        u32 giHalfCached = ~0u;
        VkDescriptorSet giUpscaleDescSet = VK_NULL_HANDLE;   // half-res GI bilateral-upscale set (Set 1)
        // Half-res DI (both channels): a-trous finals write svgfDiHalf / svgfDiSpecHalf; bilateral upscales
        // resolve them into the full-res svgfDenoised / svgfDiSpecDenoised. diHalfCached drives realloc.
        std::shared_ptr<Texture> svgfDiHalf;
        std::shared_ptr<Texture> svgfDiSpecHalf;
        u32 diHalfCached = ~0u;
        VkDescriptorSet diUpscaleDescSet     = VK_NULL_HANDLE;   // half-res DI diffuse upscale set
        VkDescriptorSet diSpecUpscaleDescSet = VK_NULL_HANDLE;   // half-res DI specular upscale set

        // RT-reflection specular SVGF: flat parallel to the GI SVGF fields. A third SvgfDenoiser
        // instance (DenoiserChannel::Reflections) denoises reflRadiance via the hit-distance
        // virtual-reprojection spec reproject; svgfSpecDenoised feeds pbr.frag Set 3 b7 (the reflection composite).
        // The geom-history's spare channel carries hitDist for reflected-depth disocclusion (vs the diffuse
        // geom's unused .a). Same shapes/clears as the GI SVGF.
        std::shared_ptr<Texture> svgfSpecDenoised;
        VkDescriptorSet svgfSpecPassthroughDescSet = VK_NULL_HANDLE;
        std::shared_ptr<Texture> svgfSpecColorHist[2];
        std::shared_ptr<Texture> svgfSpecMoments[2];
        std::shared_ptr<Texture> svgfSpecGeom[2];
        VkDescriptorSet svgfSpecReprojectDescSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        std::shared_ptr<Texture> svgfSpecAtrous[2];
        VkDescriptorSet svgfSpecMomentsDescSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkDescriptorSet svgfSpecAtrousDescSet[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        // Half-res reflections: the a-trous final writes svgfSpecHalf; a bilateral upscale resolves it into
        // the full-res svgfSpecDenoised. reflHalfCached drives realloc. Mirrors the svgfDiSpecHalf trio.
        std::shared_ptr<Texture> svgfSpecHalf;

        // ReSTIR-DI specular SVGF: flat parallel to the spec fields above. A 4th SvgfDenoiser
        // instance (DenoiserChannel::DiSpecular) denoises restirDISpec via the SURFACE-MOTION reproject
        // (direct point-light specular is surface-attached, not a reflection's virtual image, so it reuses
        // svgf_reproject.slang, not the hit-distance spec variant); svgfDiSpecDenoised feeds pbr.frag Set 3
        // b8. Same shapes/clears as the GI SVGF. see arch/rendering-pipeline.md
        std::shared_ptr<Texture> svgfDiSpecDenoised;
        VkDescriptorSet svgfDiSpecPassthroughDescSet = VK_NULL_HANDLE;
        std::shared_ptr<Texture> svgfDiSpecColorHist[2];
        std::shared_ptr<Texture> svgfDiSpecMoments[2];
        std::shared_ptr<Texture> svgfDiSpecGeom[2];
        VkDescriptorSet svgfDiSpecReprojectDescSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        std::shared_ptr<Texture> svgfDiSpecAtrous[2];
        VkDescriptorSet svgfDiSpecMomentsDescSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkDescriptorSet svgfDiSpecAtrousDescSet[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };

        // Path-traced reference mode. ptAccum = viewport-sized RGBA32F STORAGE, the
        // in-place fp32 progressive running mean, kept GENERAL, only ever touched by the PT megakernel
        // (never sampled, so fp32 precision survives thousands of samples). ptColor = RGBA16F
        // STORAGE+SAMPLED display copy the post chain (bloom/tonemap) samples; written fresh each frame.
        // ptSampleCount is the CPU-side accumulated path count (running-mean weight); reset on
        // camera/scene change. ptDescSet binds b0=ptAccum, b1=ptColor; stable, single (not cycled).
        std::shared_ptr<Texture> ptAccum;
        std::shared_ptr<Texture> ptColor;
        VkDescriptorSet          ptDescSet = VK_NULL_HANDLE;
        u32                      ptSampleCount = 0;
        // FNV hash of the reset inputs (camera VP + scene instances + lights + settings + manual salt)
        // at the last accumulating frame. A mismatch this frame zeroes the accumulation. 0 means frame 0 resets.
        u64                      ptResetHash = 0;

        // RT specular reflections. reflRadiance = viewport-sized RGBA16F STORAGE+SAMPLED;
        // rgb = demodulated specular radiance (Li*F*G1 / Fenv), a = hitDist. The trace writes every
        // pixel each frame (reflection or env fallback), so no cross-frame read and no bootstrap clear.
        // reflDescSet binds Set 2 b0 = reflRadiance (GENERAL) + b1-b3 = depth/slimNormal/slimRoughness
        // samplers; stable per-view (single, not cycled). The specular denoiser's svgfSpec* history
        // lands beside the GI SVGF fields above.
        std::shared_ptr<Texture> reflRadiance;
        VkDescriptorSet          reflDescSet = VK_NULL_HANDLE;
        VkDescriptorSet          reflUpscaleDescSet = VK_NULL_HANDLE;   // half-res reflection bilateral-upscale set
        u32                      reflHalfCached = ~0u;                  // last-applied halfResolution; drives realloc
    };

    // Orchestrates per-frame render-graph assembly and execution. Created by RenderingSystem and
    // invoked once per frame after the DrawList and GPUObjectBuffer have been populated. Owns the
    // per-domain subsystems; each subsystem contributes its render-graph passes and manages the
    // descriptor lifecycle for the
    // Sets it owns. See arch/rendering-pipeline.md.
    class RenderPipeline
    {
    public:
        explicit RenderPipeline(RenderingSystem& system);
        ~RenderPipeline();

        // One-time init: allocates all Vulkan pipeline resources (UBOs, descriptor sets, samplers, SPIR-V,
        // IBL maps, pipelines). Runs when the Vulkan backend is active; no-op otherwise. Called from
        // RenderingSystem::ctor after FrameTargets has been allocated.
        void Initialize(u32 viewportWidth, u32 viewportHeight);

        // Tear down all Vulkan resources. Called from RenderingSystem::dtor.
        void Shutdown();

        // Build and record the render graph for one view into the per-view QueueRecorders triplet (gA / compute /
        // gB primary command buffers). Begin/end/submit is owned by RenderingSystem::Update; all visible views
        // share the per-frame ring of recorders. Caches the active RenderView + ViewResources on the pipeline so
        // passes can read them without a per-pass parameter. Returns true iff any pass routed to async-compute,
        // for the backend's per-view submit topology to decide whether to issue the compute submit at all.
        bool Execute(const RenderView& view, QueueRecorders recorders);

        // Minimal graph (ImGui only). Used by the Frame Debugger Frozen state when the camera hasn't moved:
        // the LDR output still holds the last captured image, so redrawing just the UI is enough to keep
        // Dear ImGui responsive without rebuilding the full graph.
        void ExecuteMinimal();

        // Post-resize hook: rebuilds the scene view's ViewResources entry at the new size. Game panel
        // resizes route through its own FrameTargets and hit EnsureViewResources directly.
        void OnResize(u32 width, u32 height);

        // Cache m_CurrentViewResources before the per-view UBO writes in RenderingSystem::RecordView run
        // (they read it). Safe to call every frame; only re-allocates on target-size change.
        void PrepareForTargets(FrameTargets& targets);
        void PrepareRtScene(const RenderView&, const DirectionalLightShadowParams&);

        // Reload the environment HDR -> irradiance + prefiltered cubemaps + BRDF LUT.
        void ReloadSkybox(const fs::path& hdrPath);

        // Per-frame CPU-side GPU state prep. Called from RenderingSystem::RenderToView before the graph
        // executes. The CascadeData + DirectionalLightShadowParams are produced by LightingSystem and cached
        // on this Pipeline for the remainder of the view (Execute reads them through
        // m_FrameCascades / m_FrameShadowParams).
        void UpdateGlobalUniforms(const CameraParams& camera, const CascadeData& cascades, const DirectionalLightShadowParams& shadowParams);

        void UpdateGTAOUBO();
        // Allocates this frame's Object + Indirect regions from GPUTaggedPageAllocator, populates them from
        // snapshot, and rewrites Set 5 + cull descriptors. Tagged with the absolute render-frame index so
        // FreeTag(N-2) reclaims them once the GPU has retired the consuming frame.
        void BuildGPUObjectBuffer(const RenderSnapshot& snapshot);
        u32  EnsureMaterialRegistered(std::shared_ptr<Material> material);

        // Editor + frame-debugger lookups (RenderingSystem forwards to these).
        std::shared_ptr<Texture> GetNamedTexture(const std::string& name) const;
        void ReplayPassUpToDraw(u32 passIdx, u32 localDrawIdx);
        void BlitArchivedDepthToPreview(u32 archiveIdx, int layer, float nearZ, float farZ);
        void BlitArchivedSlimToPreview(u32 archiveIdx, u32 mode, float scale);

        // Maps consumed by DrawListBuilder (populated by BuildGPUObjectBuffer).
        const std::unordered_map<UUID, u32, UUIDHash>& GetMaterialSlotMap() const { return m_Geometry.GetMaterialSlotMap(); }
        const std::unordered_map<entt::entity, u32>& GetEntityToSSBOIndex() const { return m_Geometry.GetEntityToSSBOIndex(); }

        // Entity lookup table for mouse picking (index 0 = null sentinel; valid entities start at 1).
        // Populated by BuildGPUObjectBuffer.
        const std::vector<entt::entity>& GetEntityLookup() const { return m_Geometry.GetEntityLookup(); }

        // Compatibility access to the system-owned watcher. Reload lifecycle and Poll belong to RenderingSystem.
        ShaderWatcher& GetShaderWatcher();
        void RegisterShaderReloadConsumers(class ShaderReloadCoordinator&);

        // Owning RenderingSystem (set by ctor). Compatibility subsystems read scene state
        // through this accessor.
        RenderingSystem&       GetSystem()       { return m_System; }
        const RenderingSystem& GetSystem() const { return m_System; }

        // Active per-view scratch (set during Execute; consumed by compatibility subsystems).
        const RenderView*    GetCurrentView()                { return m_CurrentView; }
        ViewResources*       GetCurrentViewResources()       { return m_CurrentViewResources; }
        const ViewResources* GetCurrentViewResources() const { return m_CurrentViewResources; }

        // Subsystem accessors: preferred path for cross-subsystem reads.
        GlobalSubsystem&         GetGlobal()         { return m_Global; }
        const GlobalSubsystem&   GetGlobal()   const { return m_Global; }
        LightingSubsystem&       GetLighting()       { return m_Lighting; }
        const LightingSubsystem& GetLighting() const { return m_Lighting; }
        GeometrySubsystem&       GetGeometry()       { return m_Geometry; }
        const GeometrySubsystem& GetGeometry() const { return m_Geometry; }
        GTAOSubsystem&           GetGTAO()         { return m_GTAO; }
        const GTAOSubsystem&     GetGTAO()   const { return m_GTAO; }
        const GtaoViewState* GetGtaoViewState(RenderViewId id) const
        {
            const auto* state = m_GtaoStates.Find(id);
            return state ? state->get() : nullptr;
        }
        VolumetricSubsystem&         GetVolumetric()        { return m_Volumetric; }
        const VolumetricSubsystem&   GetVolumetric()  const { return m_Volumetric; }
        TransparencySubsystem&       GetTransparency()       { return m_Transparency; }
        const TransparencySubsystem& GetTransparency() const { return m_Transparency; }
        PostProcessSubsystem&       GetPostProcess()       { return m_PostProcess; }
        const PostProcessSubsystem& GetPostProcess() const { return m_PostProcess; }

    private:




        RenderingSystem& m_System;

        // Active RenderView + ViewResources for the current Execute call.
        // Passes read these instead of taking the view as a parameter.
        const RenderView*  m_CurrentView          = nullptr;
        ViewResources*     m_CurrentViewResources = nullptr;

        // Per-view resource cache. Entries are owned here; panels call ReleaseViewResources on destruction.
        std::unordered_map<u64, ViewResources> m_ViewResources;

        // ---- Constants (shared with RS-side callers when needed) ----
    public:
        static constexpr u32 k_MaxGPUObjects          = 4096;
        static constexpr u32 k_MaxViews               = 2; // scene + game; bump for PIP / reflections later
        static constexpr u32 k_IndirectRegionsPerView = 1 + k_ShadowCascadeCount;
        static constexpr u32 k_IndirectRegionCount    = k_MaxViews * k_IndirectRegionsPerView;
        static constexpr u32 k_IndirectRegionStride   = k_MaxGPUObjects;

        // Return the cached ViewResources for targets, allocating on first use and rebuilding
        // textures/descriptors on size change. Release* is called from the owning panel's dtor.
        // Host-only; safe to call outside a frame.
        ViewResources& EnsureViewResources(FrameTargets& targets);
        void           ReleaseViewResources(FrameTargets& targets);

        // Replay validates the registration and generation before dereferencing targets.
        bool HasViewResources(FrameTargets* targets, RenderViewId id, u64 generation) const;

        // Lookup without minting: returns nullptr if `targets` isn't in the map.
        // Used by replay to fetch the captured view's resources.
        ViewResources*       GetViewResources(FrameTargets* targets);
        const ViewResources* GetViewResources(FrameTargets* targets) const;

    private:
        // Split allocation + per-group descriptor writes for readability.
        void AllocateViewResources(ViewResources& vr, FrameTargets& targets);
        void RecreateViewTextures(ViewResources& vr, u32 fullW, u32 fullH, u32 halfW, u32 halfH);
        void DestroyViewResources(ViewResources& vr);

        // ---- Subsystems (own their domain state + lifecycle + passes) ----
        GlobalSubsystem         m_Global;
        LightingSubsystem       m_Lighting;
        GeometrySubsystem       m_Geometry;
        GTAOSubsystem           m_GTAO;
        GtaoViewStateStore       m_GtaoStates;
        std::unique_ptr<CompiledRenderPipeline> m_GtaoPipeline;
        VolumetricSubsystem     m_Volumetric;
        TransparencySubsystem   m_Transparency;
        PostProcessSubsystem    m_PostProcess;
        EditorOverlaysSubsystem m_EditorOverlays;
        DebugDrawSubsystem      m_DebugDraw;
        bool                    m_RtNativeInitialized = false;
        std::unique_ptr<CompiledRenderPipeline> m_RtSceneComposition;
        std::unique_ptr<CompiledRenderPipeline> m_RtSunShadowComposition;
        std::unique_ptr<PreparedPipelineFrame> m_RtScenePlan;
        RtSceneParameters m_RtSceneParameters;
        RtSubsystem             m_Rt;
        RtRestirSubsystem       m_Restir;
        RtRestirGiSubsystem     m_RestirGi;
        SlangParityGuard        m_SlangParity;   // GLSL vs Slang bindless-SPIR-V parity guard (default-OFF)
        PathTraceSubsystem      m_PathTrace;
        ReflectionsSubsystem    m_Reflections;
        SkinningSubsystem       m_Skinning;
        std::unique_ptr<CompiledRenderPipeline> m_GeometryPreparationPipeline;
        std::unique_ptr<CompiledRenderPipeline> m_SurfacePreparationComposition;
        std::unique_ptr<CompiledRenderPipeline> m_CsmComposition;
        std::unique_ptr<CompiledRenderPipeline> m_ClusterComposition;
        std::unique_ptr<CompiledRenderPipeline> m_FogComputeComposition;
        std::unique_ptr<CompiledRenderPipeline> m_FogCompositeComposition;
        std::unique_ptr<CompiledRenderPipeline> m_RefractionComposition;
        std::unique_ptr<CompiledRenderPipeline> m_TransparencyComposition;
        std::unique_ptr<CompiledRenderPipeline> m_TaaComposition;
        std::unique_ptr<CompiledRenderPipeline> m_BloomComposition;
        std::unique_ptr<CompiledRenderPipeline> m_VisualizationComposition;
        std::unique_ptr<CompiledRenderPipeline> m_OutlineComposition;
        std::unique_ptr<CompiledRenderPipeline> m_DebugDrawComposition;
        std::unique_ptr<CompiledRenderPipeline> m_SelectionMaskComposition;
        std::unique_ptr<CompiledRenderPipeline> m_GridComposition;
        std::unique_ptr<CompiledRenderPipeline> m_CompositeComposition;
        std::unique_ptr<CompiledRenderPipeline> m_SkyComposition;
        std::unique_ptr<CompiledRenderPipeline> m_ForwardComposition;
        std::unique_ptr<IDenoiser> m_Denoise;     // DI SVGF; swappable to NRD/RELAX via the settings toggle
        std::unique_ptr<IDenoiser> m_DenoiseGi;   // GI SVGF: second instance (DenoiserChannel::Gi)
        std::unique_ptr<IDenoiser> m_DenoiseRefl; // specular SVGF: third instance (DenoiserChannel::Reflections)
        std::unique_ptr<IDenoiser> m_DenoiseDiSpec; // ReSTIR-DI specular SVGF: 4th instance (DenoiserChannel::DiSpecular)

    public:
        EditorOverlaysSubsystem&       GetEditorOverlays()       { return m_EditorOverlays; }
        const EditorOverlaysSubsystem& GetEditorOverlays() const { return m_EditorOverlays; }
        bool HasRtNativeResources() const { return m_RtNativeInitialized; }
        RtSubsystem&                   GetRt()                   { return m_Rt; }
        const RtSubsystem&             GetRt()             const { return m_Rt; }
        RtRestirSubsystem&             GetRestir()               { return m_Restir; }
        const RtRestirSubsystem&       GetRestir()         const { return m_Restir; }
        RtRestirGiSubsystem&           GetRestirGi()             { return m_RestirGi; }
        const RtRestirGiSubsystem&     GetRestirGi()       const { return m_RestirGi; }
        PathTraceSubsystem&            GetPathTrace()            { return m_PathTrace; }
        const PathTraceSubsystem&      GetPathTrace()      const { return m_PathTrace; }
        ReflectionsSubsystem&          GetReflections()          { return m_Reflections; }
        const ReflectionsSubsystem&    GetReflections()    const { return m_Reflections; }
        SkinningSubsystem&             GetSkinning()             { return m_Skinning; }
        const SkinningSubsystem&       GetSkinning()       const { return m_Skinning; }
        IDenoiser&                     GetDenoise()              { return *m_Denoise; }
        const IDenoiser&               GetDenoise()        const { return *m_Denoise; }
        IDenoiser&                     GetDenoiseGi()            { return *m_DenoiseGi; }
        const IDenoiser&               GetDenoiseGi()      const { return *m_DenoiseGi; }
        IDenoiser&                     GetDenoiseRefl()          { return *m_DenoiseRefl; }
        const IDenoiser&               GetDenoiseRefl()    const { return *m_DenoiseRefl; }
        IDenoiser&                     GetDenoiseDiSpec()        { return *m_DenoiseDiSpec; }
        const IDenoiser&               GetDenoiseDiSpec()  const { return *m_DenoiseDiSpec; }

    private:
        // Profiling and named output ownership live in RenderingSystem.

        // ---- Shader hot-reload (engine + project dirs) ----

    public:
        // Accessors forwarded to the frame-debugger context so editor panels can sample preview textures
        // and invalidate caches without needing access to the context class directly.
        VkImageView GetPerDrawPreviewView()  const;
        u64         GetPerDrawPreviewKey()   const;
        u32         GetPerDrawPreviewWidth() const;
        u32         GetPerDrawPreviewHeight()const;
        VkImageView GetDepthPreviewView()    const;
        u32         GetDepthPreviewWidth()   const;
        u32         GetDepthPreviewHeight()  const;
        VkImageView GetSlimPreviewView()     const;
        u32         GetSlimPreviewWidth()    const;
        u32         GetSlimPreviewHeight()   const;
        const RG::RenderGraphSnapshot& GetGraphSnapshot() const;

        // Resets the per-draw preview cache key; called from RS::ExitCapture.
        void ResetPreviewCacheKeys();
    };
}
