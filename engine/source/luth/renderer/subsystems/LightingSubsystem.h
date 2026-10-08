#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/features/ClusterVizBindings.h"

#include "luth/core/types/LuthTypes.h"
#include "luth/core/FrameData.h"
#include "luth/renderer/CameraParams.h"
#include "luth/renderer/lighting/LightTypes.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/renderer/backend/vulkan/VulkanPipeline.h"
#include "luth/renderer/backend/vulkan/VulkanComputePipeline.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/memory/GPUTaggedPageAllocator.h"

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Luth
{
    class RenderPipeline;
    struct DrawList;
    struct RenderSnapshot;
    struct FrameDebugger;
    struct CsmBindings
    {
        VkPipeline rigid = VK_NULL_HANDLE, deformed = VK_NULL_HANDLE;
        VkPipelineLayout rigidLayout = VK_NULL_HANDLE, deformedLayout = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 6> sets{};
        const Texture* texture = nullptr;
        VkImage image = VK_NULL_HANDLE;
        std::array<VkImageView, k_ShadowCascadeCount> layers{};
        bool captureDraws = false;
    };
    namespace fs = std::filesystem;

    struct SkyBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkBuffer vertex = VK_NULL_HANDLE;
        std::shared_ptr<VKVertexBuffer> retainedVertex;
        std::array<VkDescriptorSet, 5> sets{};
    };

    struct ClusterBuildConstants
    {
        Mat4 invProjection;
        Vec2 viewportSize;
        Vec2 pad{};
        float nearZ = 0, farZ = 0;
        u32 tilesX = k_ClusterTilesX, tilesY = k_ClusterTilesY;
    };
    struct LightAssignConstants
    {
        Mat4 view;
        u32 pointLightCount = 0, spotLightCount = 0;
        u32 maxLightsPerCluster = k_MaxLightsPerCluster, pad = 0;
    };
    struct ClusterBindings
    {
        VkPipeline build = VK_NULL_HANDLE, assign = VK_NULL_HANDLE;
        VkPipelineLayout buildLayout = VK_NULL_HANDLE, assignLayout = VK_NULL_HANDLE;
        VkDescriptorSet buildSet = VK_NULL_HANDLE, assignSet = VK_NULL_HANDLE;
        Memory::GPUSubRegion lights{}, aabb{}, grid{}, indices{}, counter{};
        ClusterBuildConstants buildConstants{};
        LightAssignConstants assignConstants{};
        bool ready = false;
    };

    // Owns Set 3, shadow map, IBL maps, skybox VB + shadow/skybox pipelines.
    // invariant: Init() must precede BuildPipelines(geoLayouts); the latter needs Set 5.
    class LightingSubsystem
    {
    public:
        void Init(RenderPipeline& pipeline, const fs::path& hdrPath);
        void BuildPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts);
        void Shutdown();

        // Re-bake IBL from a new HDR. Rebuilds the skybox pipeline (prefiltered mip count may change)
        // and rewrites every cached view's Set 0.
        void ReloadSkybox(const fs::path& hdrPath, const std::vector<VkDescriptorSetLayout>& geoLayouts);

        // Hot-reload hook. Returns true if name matched and was handled.
        bool OnShaderReloaded(const std::string& name, const std::vector<u32>& spv,
                              const std::vector<VkDescriptorSetLayout>& geoLayouts);

        // Render-graph contributions.
        CsmBindings PrepareCsmBindings(const std::array<VkDescriptorSet, 6>& sets, bool captureDraws) const;
        static GraphTextureRef ImportShadowTarget(RG::RenderGraph&, const CsmBindings&, u32 cascadeIndex);
        static RG::ResourceHandle AddShadowPass(RG::RenderGraph&, RG::ResourceHandle target,
            const VisibleDrawRange&, const CsmBindings&, u32 cascadeIndex,
            const DrawList&, const RenderSnapshot&, FrameDebugger*);
        SkyBindings PrepareSkyBindings(const std::array<VkDescriptorSet, 5>& sets) const;
        static RG::ResourceHandle AddSkyboxPass(RG::RenderGraph&, RG::ResourceHandle sceneColor,
            RG::ResourceHandle sceneDepth, u32 width, u32 height, const SkyBindings&, FrameDebugger*);

        // Native cluster preparation freezes per-view bindings before graph recording.
        ClusterBindings PrepareClusterBindings(u64 renderFrameIndex, VkDescriptorSet buildSet,
            VkDescriptorSet assignSet, const CameraParams&, u32 width, u32 height,
            const Memory::GPUSubRegion& lights, u32 pointCount, u32 spotCount) const;
        static GraphBufferRef ImportLightingBuffer(RG::RenderGraph&, const char* name, const Memory::GPUSubRegion&);
        static std::array<RG::BufferHandle, 2> AddClusterBuildPass(RG::RenderGraph&,
            RG::BufferHandle aabb, RG::BufferHandle grid, const ClusterBindings&, FrameDebugger*);
        static std::array<RG::BufferHandle, 2> AddLightAssignPass(RG::RenderGraph&,
            RG::BufferHandle lights, RG::BufferHandle aabb, RG::BufferHandle grid,
            RG::BufferHandle indices, RG::BufferHandle counter, const ClusterBindings&, FrameDebugger*);
        // Per-frame LightSSBO upload. Allocates from tagged heap, copies the gathered header +
        // point-light array. The caller passes its physical slice explicitly to bind the
        // same backing for its b0 read. Returns the region for WriteSet3PerView's b0 write.
        Memory::GPUSubRegion UploadLightSSBO(const GatheredLights& lights);

        // Per-view Set 3 b0/b1/b2 write. Called from BuildGraph after the cluster + assign passes
        // have produced their outputs. b3 (shadow sampler) is stable, written at view-alloc time.
        void WriteSet3PerView(const Memory::GPUSubRegion& lightSSBORegion,
                              const Memory::GPUSubRegion& clusterGridRegion,
                              const Memory::GPUSubRegion& lightIndexRegion);

        // Writes Set 3 b3 (shadow sampler) to every slot of the per-view lightDescSet[]. Called
        // from AllocateViewResources after the descriptor pool has allocated the set.
        void WriteShadowView(struct ViewResources& vr);

        // Cluster debug viz, gated by ShadeMode::ClustersDensity in BuildGraph. Samples SceneDepth
        // to derive the per-fragment Olsson slice, then reads the per-view cluster grid and heat-maps
        // the lights-per-cluster count over LDR.
        ClusterVizBindings PrepareClusterVizBindings(std::shared_ptr<ClusterVizViewState> state, VkDescriptorSet lightingSet,
            const Memory::GPUSubRegion& grid,
            u32 width, u32 height, float nearZ, float farZ, bool enabled) const;
        RG::ResourceHandle AddClusterVizPass(RG::RenderGraph& rg, RG::ResourceHandle ldrInput,
            RG::ResourceHandle sceneDepth, RG::BufferHandle grid,
            const ClusterVizBindings& packet, FrameDebugger* debugger);
        // Domain-owned depth-sampler state. Stable across frames; source replacement uses a safe point.
        // The shared cycled lighting set remains an explicit native compatibility dependency.
        std::shared_ptr<ClusterVizViewState> EnsureClusterVizView(RenderViewId, class FrameTargets&);
        void ReleaseClusterVizView(RenderViewId);
        void WriteClusterVizView(ClusterVizViewState&);

        VkDescriptorSetLayout GetClusterBuildLayout() const { return m_ClusterBuildSetLayout; }
        VkDescriptorSetLayout GetLightAssignLayout()  const { return m_LightAssignSetLayout; }

        // ---- Accessors ----
        VkDescriptorSetLayout GetSetLayout() const          { return m_LightSetLayout; }
        // Delegates to the active ViewResources; Set 3 is per-view.
        VkDescriptorSet       GetLightDescSet(u32 slot) const;
        VkSampler             GetShadowSampler() const      { return m_ShadowSampler; }
        const std::shared_ptr<Texture>& GetShadowMap() const { return m_ShadowMap; }
        VkImageView           GetShadowLayerView(u32 i) const { return m_ShadowLayerViews[i]; }
        VKPipeline*           GetShadowPipeline() const     { return m_ShadowPipeline.get(); }
        VKPipeline*           GetShadowSkinnedPipeline() const { return m_ShadowSkinnedPipeline.get(); }

        // Depth-prepass + selection pipelines reuse this null-fragment SPV.
        // Temp accessor; folds away once those pipelines move into Geometry/EditorOverlays.
        const std::vector<u32>& GetShadowFragSpv() const { return m_ShadowFragSpv; }

        const std::shared_ptr<Texture>& GetIrradianceMap()  const { return m_IrradianceMap; }
        const std::shared_ptr<Texture>& GetPrefilteredMap() const { return m_PrefilteredMap; }
        const std::shared_ptr<Texture>& GetBRDFLut()        const { return m_BRDFLut; }
        VkSampler                       GetIBLSampler()     const { return m_IBLSampler; }
        bool IsIBLReady() const { return m_IrradianceMap && m_PrefilteredMap && m_BRDFLut && m_IBLSampler; }

    private:
        void CreateShadowResources(VkDevice device);
        void LoadIBL(const fs::path& hdrPath);
        void BuildShadowPipelines(const std::vector<VkDescriptorSetLayout>& geoLayouts);
        void BuildSkyboxPipeline(const std::vector<VkDescriptorSetLayout>& geoLayouts);

        RenderPipeline* m_Pipeline = nullptr;

        // Set 3 + shadow map. m_LightSetLayout shared across views; per-view lightDescSet[] lives
        // on ViewResources and is allocated from vr.descPool in AllocateViewResources.
        std::shared_ptr<Texture> m_ShadowMap;
        VkImageView              m_ShadowLayerViews[k_ShadowCascadeCount] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkSampler                m_ShadowSampler        = VK_NULL_HANDLE;
        VkSampler                m_SunShadowMaskSampler = VK_NULL_HANDLE;  // Set 3 binding 4: RT sun shadow mask (linear, clamp-to-edge, no compare)
        VkDescriptorSetLayout    m_LightSetLayout = VK_NULL_HANDLE;

        // Shadow pipelines + SPV.
        std::unique_ptr<VKPipeline> m_ShadowPipeline;
        std::unique_ptr<VKPipeline> m_ShadowSkinnedPipeline;
        std::vector<u32>            m_ShadowVertSpv;
        std::vector<u32>            m_ShadowFragSpv;
        std::vector<u32>            m_ShadowSkinnedVertSpv;

        // IBL.
        std::shared_ptr<Texture> m_IrradianceMap;
        std::shared_ptr<Texture> m_PrefilteredMap;
        std::shared_ptr<Texture> m_BRDFLut;
        VkSampler                m_IBLSampler = VK_NULL_HANDLE;

        // Skybox.
        std::unique_ptr<VKPipeline>     m_SkyboxPipeline;
        std::shared_ptr<VKVertexBuffer> m_SkyboxVB;
        std::vector<u32>                m_SkyboxVertSpv;
        std::vector<u32>                m_SkyboxFragSpv;

        // Forward+ compute pipelines. Layouts shared across views; per-view descriptor sets live
        // on ViewResources because the cluster grid + AABB buffers are per-view tagged-heap regions.
        std::unique_ptr<VKComputePipeline> m_ClusterBuildPipeline;
        VkDescriptorSetLayout              m_ClusterBuildSetLayout = VK_NULL_HANDLE;
        std::vector<u32>                   m_ClusterBuildSpv;

        std::unique_ptr<VKComputePipeline> m_LightAssignPipeline;
        VkDescriptorSetLayout              m_LightAssignSetLayout = VK_NULL_HANDLE;
        std::vector<u32>                   m_LightAssignSpv;

        // Cluster debug viz. Two-set pipeline: set 0 owns the depth sampler (stable per-view, written
        // by WriteClusterVizView); set 1 reuses the Set 3 lightDescSet for its cluster grid read.
        ClusterVizViewStateStore m_ClusterVizStates;
        std::unique_ptr<VKPipeline> m_ClusterVizPipeline;
        VkDescriptorSetLayout       m_ClusterVizDescSetLayout = VK_NULL_HANDLE;
        VkSampler                   m_ClusterVizDepthSampler  = VK_NULL_HANDLE;
        std::vector<u32>            m_FullscreenVertSpv;
        std::vector<u32>            m_ClusterVizFragSpv;



    };
}
