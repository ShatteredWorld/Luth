#include "luthpch.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include <limits>
namespace Luth
{
    std::shared_ptr<ClusterVizViewState> LightingSubsystem::EnsureClusterVizView(RenderViewId id, FrameTargets& targets)
    {
        const auto depth = targets.GetSceneDepth();
        if (!depth) throw std::invalid_argument("ClusterViz: sampled depth unavailable");
        u64 generation = 1;
        if (const auto* prior = m_ClusterVizStates.Find(id))
        {
            generation = (*prior)->sourceGeneration;
            if ((*prior)->depth != depth)
            {
                if (generation == std::numeric_limits<u64>::max()) throw std::runtime_error("ClusterViz: source generation exhausted");
                ++generation;
            }
        }
        return m_ClusterVizStates.Ensure(id, ClusterVizViewState::Config(depth->GetWidth(), depth->GetHeight(), generation),
            [&](const ViewStateConfig& config) {
                auto state = ClusterVizViewState::Create(id, config, m_ClusterVizDescSetLayout);
                state->depth = depth; WriteClusterVizView(*state); return state;
            }, [] { Renderer::WaitForGPU(); });
    }
    void LightingSubsystem::ReleaseClusterVizView(RenderViewId id)
    {
        m_ClusterVizStates.Release(id, [] { Renderer::WaitForGPU(); });
    }
    void LightingSubsystem::WriteClusterVizView(ClusterVizViewState& state)
    {
        if (!state.depth) throw std::invalid_argument("ClusterViz: missing stable depth source");
        if (!state.set || !m_ClusterVizDepthSampler) return;
        const auto depth = std::static_pointer_cast<VKTexture>(state.depth);
        VkDescriptorImageInfo info{m_ClusterVizDepthSampler, depth->GetImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = state.set; write.dstBinding = 0; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo = &info;
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), 1, &write, 0, nullptr);
    }
}
