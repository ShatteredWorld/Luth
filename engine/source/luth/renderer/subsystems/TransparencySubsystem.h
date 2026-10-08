#pragma once

#include "luth/core/types/LuthTypes.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/renderer/features/RefractionBackdropBindings.h"
#include "luth/renderer/features/TransparencyViewState.h"
#include "luth/renderer/features/TransparencyBindings.h"
#include "luth/renderer/pipeline/PipelineManager.h"

#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace Luth
{
    struct ViewResources;

    // Owns the transparent tier: the pass slot after skybox + volumetric composite where
    // Transparent/Fade draws land (GeometryPass renders opaque + cutout only). Sorted mode:
    // per-view back-to-front indirect draws with pbr_transparent.frag (corrected inputs: rayQuery
    // sun shadow, cluster lights, fragment-depth froxel fog; never the opaque-coupled screen-space
    // buffers). OIT mode layers a PPLL store + resolve on top.
    // invariant: Init() before RenderPipeline's BuildPipelines block; Set 6 joins geoLayouts there.
    class TransparencySubsystem
    {
    public:
        void Init();
        void BuildPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts);
        void Shutdown();

        // Handles pbr_transparent.frag (returns true); pbr.vert / pbr_skinned.slang reloads
        // invalidate the cached variants but return false so GeometrySubsystem still owns them.
        bool OnShaderReloaded(const std::string& name, const std::vector<u32>& spv);

        // Set 6 b0 <- parity-picked resolved fog atlas (the volumetric composite's b1 rule).
        void WritePerFrame(TransparencyViewState&, const std::shared_ptr<FogViewState>&, VkSampler fogSampler, u32 frameAbs);
        std::shared_ptr<TransparencyViewState> EnsureView(RenderViewId, u32 width, u32 height, u32 layers);
        void ReleaseView(RenderViewId);

        // Set 6 b1/b2 (heads + nodes, all cycled slots) + the resolve set <- the view's OIT
        // resources. Called from AllocateViewResources + on resize/budget reallocation.
        void WriteOitView(TransparencyViewState& vr);

        // Reserved Garlic tag range for per-view OIT node pools: disjoint from ReSTIR DI
        // (0xFFFF0000+) and GI (0xFFFF8000+); outside the per-frame FreeTag(N-2) sweep.
        u32 NextNodePoolTag();

        static std::vector<u32> SortedOrder(const DrawList&, const Mat4& view);
        TransparencyBindings PrepareSortedBindings(GeometrySubsystem&, const std::array<VkDescriptorSet, 7>&,
            bool wireframe, bool captureDraws, const Mat4&, const VisibleDrawRange&, const DrawList&, const RenderSnapshot&,
            TextureBindingRef fog, TextureBindingRef backdrop, const RtSubsystem*);
        TransparencyBindings PrepareTransparencyBindings(GeometrySubsystem&, const std::array<VkDescriptorSet, 7>&,
            bool wireframe, bool captureDraws, const Mat4&, const VisibleDrawRange&, const DrawList&, const RenderSnapshot&,
            TextureBindingRef fog, TextureBindingRef backdrop, const RtSubsystem*, bool oit,
            const TransparencyViewState&, u32 maxResolveK);
        static std::array<RG::ResourceHandle, 3> AddOitPasses(RG::RenderGraph&, RG::ResourceHandle color,
            RG::ResourceHandle picking, RG::ResourceHandle depth, const VisibleDrawRange&, u32 width, u32 height,
            const TransparencyBindings&, std::span<const RG::ResourceHandle>, std::span<const RG::BufferHandle>,
            bool fogValid, FrameDebugger*);
        static std::array<RG::ResourceHandle, 3> AddSortedPass(RG::RenderGraph&, RG::ResourceHandle color,
            RG::ResourceHandle picking, RG::ResourceHandle depth, const VisibleDrawRange&, u32 width, u32 height,
            const TransparencyBindings&, std::span<const RG::ResourceHandle>, std::span<const RG::BufferHandle>,
            bool fogValid, FrameDebugger*);
        static RefractionBackdropBindings PrepareBackdropBindings(const std::shared_ptr<Texture>&, bool enabled);
        static GraphTextureRef AddBackdropCopyPass(RG::RenderGraph&, RG::ResourceHandle source,
            const RefractionBackdropBindings&);

        VkDescriptorSetLayout GetSetLayout()        const { return m_TransparentSetLayout; }
        VkDescriptorSetLayout GetResolveSetLayout() const { return m_ResolveSetLayout; }

    private:
        TransparencyBindings PrepareDrawBindings(GeometrySubsystem&, const std::array<VkDescriptorSet, 7>&,
            bool wireframe, bool captureDraws, const Mat4&, const VisibleDrawRange&, const DrawList&, const RenderSnapshot&,
            TextureBindingRef fog, TextureBindingRef backdrop, const RtSubsystem*, bool oit);
        static void RecordDraws(const TransparencyBindings&, VkBuffer indirect, u32 width, u32 height,
            bool fogValid, u32 capacity, RG::RenderPassContext&, FrameDebugger*);
        void BuildResolvePipeline();

        // Matches pbr_transparent_shading.glsl's push-constant block (16 B, FRAGMENT).
        struct TransparentPC
        {
            u64 geomTable    = 0;  // geometry-table BDA for the shadow ray's cutout alpha test
            u32 flags        = 0;  // bit0 = fog atlas valid this frame
            u32 nodeCapacity = 0;  // OIT store only
        };
        static_assert(sizeof(TransparentPC) == 16, "must match the shader push-constant block");

        TransparencyViewStateStore m_ViewStates;

        // Set 6 (transparent pass-local): b0 fog atlas sampler3D (UAB, parity rewrite), b1 OIT heads
        // storage image + b2 OIT nodes SSBO (UAB + partially-bound: written when the PPLL lands;
        // the sorted pipeline never statically uses them).
        VkDescriptorSetLayout m_TransparentSetLayout = VK_NULL_HANDLE;
        // OIT resolve pass-local (Set 1 of the fullscreen pipeline): b0 heads, b1 nodes.
        VkDescriptorSetLayout m_ResolveSetLayout = VK_NULL_HANDLE;
        // Set 6 b3 refraction-backdrop sampler (linear, clamp-to-edge): the screen-space refraction tap.
        VkSampler m_BackdropSampler = VK_NULL_HANDLE;

        PipelineManager  m_SortedPm;
        PipelineManager  m_SortedSkinnedPm;
        PipelineManager  m_OitPm;
        PipelineManager  m_OitSkinnedPm;
        std::vector<u32> m_TransparentFragSpv;
        std::vector<u32> m_OitStoreFragSpv;
        std::vector<u32> m_FullscreenVertSpv;
        std::vector<u32> m_ResolveFragSpv;

        std::unique_ptr<VKPipeline> m_ResolvePipeline;

        u32 m_NextNodePoolTag = 0xFFFFC000u;
    };
}
