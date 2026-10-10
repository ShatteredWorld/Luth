#pragma once

#include "luth/core/types/LuthTypes.h"
#include "luth/renderer/backend/vulkan/TlasBuilder.h"
#include "luth/renderer/backend/vulkan/VulkanRtMeshResources.h"
#include "luth/renderer/backend/vulkan/VulkanAllocator.h"
#include "luth/renderer/backend/vulkan/VulkanComputePipeline.h"
#include "luth/renderer/features/rt/RtSunShadowBindings.h"
#include "luth/renderer/features/rt/RtSunShadowViewState.h"
#include "luth/renderer/rendergraph/RenderGraphResources.h"

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <vulkan/vulkan.h>

namespace Luth
{
    class RenderPipeline;
    class FrameTargets;
    struct ViewResources;
    namespace RG { class RenderGraph; }

    // Frame-local native scene packet. Recording is serialized on the existing compute primary.
    // Valid empty fallback and populated scene bindings are distinct states.
    struct PreparedRtScene
    {
        PreparedRtScene() = default;
        PreparedRtScene(const PreparedRtScene&) = delete;
        PreparedRtScene& operator=(const PreparedRtScene&) = delete;
        u64 frameIndex = 0;
        std::array<PreparedBlasBuild, 2> blas;
        PreparedTlasBuild tlas;
        VkAccelerationStructureKHR emptyFallback = VK_NULL_HANDLE;
        PFN_vkCmdPipelineBarrier2 barrier = vkCmdPipelineBarrier2;
        mutable bool recorded = false;
        VkAccelerationStructureKHR GetTlas() const
        { return tlas.result.tlas != VK_NULL_HANDLE ? tlas.result.tlas : emptyFallback; }
        VkDeviceAddress GetGeometryTableBDA() const { return tlas.result.geomTableBDA; }
        bool HasSceneData() const
        { return tlas.result.instanceCount != 0 && tlas.result.tlas != VK_NULL_HANDLE && tlas.result.geomTableBDA != 0; }
        TlasBuildResult ReuseCandidate() const
        { return recorded ? tlas.result : TlasBuildResult{}; }
        void Record(VkCommandBuffer) const;
    };

    // Houses RT-domain state: per-frame TLAS rebuild + skinned-BLAS refit on AsyncCompute with
    // hash-based dirty skip and a multi-view guard, plus the sun-shadow surface:
    //   - Persistent empty TLAS at Init (Set 0 binding 6 never null; rt_sun_shadows.comp reads it
    //     statically and PARTIALLY_BOUND is conservatively interpreted by validation).
    //   - Sun-shadow pass: rt_sun_shadows.comp (rayQuery-in-compute) on VKComputePipeline, cutout
    //     alpha-tested via material_bindings_rt.slang (Set 3 Material + Set 4 bindless). No SBT/miss surface.
    //   - AddRtSunShadowsPass on AsyncCompute (writes per-view R8 shadow mask). Consumed by
    //     pbr.frag (Set 3 binding 4) when ShadowingMode::RtShadows is active.
    class RtSubsystem
    {
    public:
        void Init(RenderPipeline& pipeline);
        void Shutdown();

        bool OnShaderReloaded(const std::string& name, const std::vector<u32>& spv);

        // CPU preparation must precede all per-view descriptor writes and recording.
        // Repeated views of the same render frame share one paired scene packet.
        std::shared_ptr<const PreparedRtScene> PrepareScene(std::span<const MeshDrawSnapshot>, u64 frameIndex,
            const std::unordered_map<UUID, u32, UUIDHash>& materialSlots, bool markEmitters);
        bool IsPreparedFor(u64 frameIndex) const
        { return m_PreparedScene && m_PreparedScene->frameIndex == frameIndex; }
        // Captures the prepared packet; recording performs no global/domain lookups or allocation.
        void AddTlasBuildPass(RG::RenderGraph& rg);
        void AddTlasBuildPass(RG::RenderGraph& rg, std::shared_ptr<const PreparedRtScene>);

        // Freeze native view/scene bindings before registration. The async pass captures the
        // packet by value and preserves the inline AS-build -> compute-query-read barrier.
        RtSunShadowBindings PrepareShadowBindings(const ViewResources&, const FrameTargets&,
            u64 frameIndex, RenderViewId, u64 generation, const PreparedRtScene*) const;
        static GraphTextureRef AddRtSunShadowsPass(RG::RenderGraph&, RG::ResourceHandle depth,
            RG::ResourceHandle normal, const RtSunShadowBindings&);

        // Per-view pass-local descriptor set writer. Binds:
        //   set 2 binding 0 = SceneDepth sampler (linear clamp)
        //   set 2 binding 1 = SlimNormal sampler (RG16F oct, linear clamp)
        //   set 2 binding 2 = sunShadowMask storage image (R8, GENERAL layout)
        // Called from ViewResources::AllocateViewResources + on resize via EnsureViewResources.
        std::shared_ptr<RtSunShadowViewState> EnsureShadowView(RenderViewId, const FrameTargets&);
        void ReleaseShadowView(RenderViewId);
        void WriteShadowPassView(const RtSunShadowViewState&);

        VkDescriptorSetLayout GetShadowPassLayout() const { return m_ShadowPassSetLayout; }

        // Returns the paired CPU-prepared TLAS before recording; otherwise the persistent
        // empty TLAS so Set 0 binding 6 is never null. Keeping them as separate fields prevents
        // the hash-skip PushDeletion in AddTlasBuildPass from accidentally destroying the
        // persistent handle when a per-frame TLAS first replaces it.
        VkAccelerationStructureKHR GetTlas() const
        {
            return m_LastResult.tlas != VK_NULL_HANDLE ? m_LastResult.tlas : m_PersistentEmptyTlas;
        }

        // Per-frame geometry-table BDA (restir_gi_initial.comp push constant). Built in lockstep with
        // the TLAS, so it pairs with whatever GetTlas() returns from the same m_LastResult. Zero before
        // the first real build (only the empty TLAS exists -> all rays miss -> table never deref'd).
        VkDeviceAddress GetGeometryTableBDA() const { return m_LastResult.geomTableBDA; }

    private:
        void BuildShadowPipeline();

        // Frame-0 binding-6 safety: built at Init, kept alive until Shutdown. The persistent
        // empty TLAS is separate from m_LastResult; PrepareScene publishes a populated pair
        // or keeps the valid empty fallback before GlobalSubsystem::UpdateUBO writes bindings.
        // m_PersistentEmptyTlas + its
        // storage buffer stay alive (the deletion-queue retires per-frame TLAS handles fenced
        // on N+2; the persistent handle has no fence and outlives them all).
        VkAccelerationStructureKHR m_PersistentEmptyTlas      = VK_NULL_HANDLE;
        VkBuffer                   m_PersistentEmptyTlasBuf   = VK_NULL_HANDLE;
        VmaAllocation              m_PersistentEmptyTlasAlloc = nullptr;

        // Sun-shadow compute pipeline + per-pass descriptor layout. Pipeline layout is [Global, Light,
        // m_ShadowPassSetLayout, Material, bindless]: Sets 0/1/2 plus Set 3/4 for the cutout alpha-test.
        // The Light layout (PBR's Set 3) remaps to Set 1 per-pipeline-layout, not a global change; the same
        // VkDescriptorSet binds at different set indices for each pipeline.
        VkDescriptorSetLayout m_ShadowPassSetLayout = VK_NULL_HANDLE;
        VkSampler             m_ShadowPassSampler   = VK_NULL_HANDLE;  // linear clamp-to-edge for depth + normal

        std::unique_ptr<VKComputePipeline> m_SunShadowsPipeline;
        std::vector<u32> m_ShadowSpv;  // cached for hot-reload pipeline rebuild

        RenderPipeline* m_Pipeline      = nullptr;
        TlasBuildResult m_LastResult{};
        u64             m_BlasReadyGeneration = 0;  // ++ when a deferred BLAS first-builds; forces one TLAS rebuild (H1)
        std::shared_ptr<const PreparedRtScene> m_PreparedScene;
        VulkanRtMeshResources m_MeshResources;
        RtSunShadowViewStates m_ShadowViews;
    };
}
