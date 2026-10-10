#pragma once

#include "luth/core/types/LuthTypes.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include "luth/renderer/backend/vulkan/VulkanComputePipeline.h"
#include "luth/renderer/features/GtaoViewState.h"

#include <memory>
#include <string>
#include <vector>

namespace Luth
{
    class Texture;
    struct FrameDebugger;
    struct CameraParams;
    struct GTAOSettings;

    // Owns the 3 GTAO compute layouts/pipelines/SPVs + linear-clamp sampler. Per-frame: rebinds
    // Set 0 binding 5 + GTAO main set binding 2 to the same tagged-heap region in one batched write;
    // both must stay atomic.
    class GTAOSubsystem
    {
    public:
        void Init();
        void Shutdown();

        bool OnShaderReloaded(const std::string& name, const std::vector<u32>& spv);

        // Per-render-stage rebind of the GTAO settings UBO.
        void UpdateUBO(GtaoViewState&, const std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT>&,
            const GTAOSettings&, u64 renderFrameIndex);

        // Stable per-view writes (sceneDepth/linDepth/rawAO/finalAO image bindings).
        // The UBO at GTAO-main binding 2 is rewritten by UpdateUBO.
        void EnsureView(GtaoViewStateStore&, RenderViewId, u32 width, u32 height, const Texture& depth);
        void WriteView(GtaoViewState&, const Texture& depth);
        bool IsReady() const { return m_PrefilterPipeline && m_MainPipeline && m_DenoisePipeline; }

        // Render-graph contributions (compute passes).
        RG::ResourceHandle AddPrefilterPass(RG::RenderGraph&, RG::ResourceHandle,
            const GtaoViewState&, const CameraParams&, FrameDebugger*);
        RG::ResourceHandle AddMainPass(RG::RenderGraph&, RG::ResourceHandle,
            const GtaoViewState&, const CameraParams&, u64 renderFrameIndex, u32 shaderFrameIndex, FrameDebugger*);
        RG::ResourceHandle AddDenoisePass(RG::RenderGraph&, RG::ResourceHandle rawAO,
            RG::ResourceHandle linearDepth, const GtaoViewState&, FrameDebugger*);

        VkSampler             GetSampler()        const { return m_Sampler; }
        VkDescriptorSetLayout GetPrefilterLayout()const { return m_PrefilterDescLayout; }
        VkDescriptorSetLayout GetMainLayout()     const { return m_MainDescLayout; }
        VkDescriptorSetLayout GetDenoiseLayout()  const { return m_DenoiseDescLayout; }

    private:
        std::unique_ptr<VKComputePipeline> m_PrefilterPipeline;
        std::unique_ptr<VKComputePipeline> m_MainPipeline;
        std::unique_ptr<VKComputePipeline> m_DenoisePipeline;

        VkSampler             m_Sampler             = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_PrefilterDescLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_MainDescLayout      = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_DenoiseDescLayout   = VK_NULL_HANDLE;

        std::vector<u32> m_PrefilterSpv;
        std::vector<u32> m_MainSpv;
        std::vector<u32> m_DenoiseSpv;
    };
}
