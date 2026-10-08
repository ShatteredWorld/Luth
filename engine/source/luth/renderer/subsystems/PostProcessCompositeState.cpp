#include "luthpch.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include <limits>

namespace Luth
{
    std::shared_ptr<CompositeViewState> PostProcessSubsystem::EnsureCompositeView(RenderViewId id,
        FrameTargets& targets, const std::shared_ptr<BloomViewState>& bloom)
    {
        if (!bloom) throw std::invalid_argument("Composite: bloom fallback unavailable");
        const std::array sources{targets.GetSceneColor(), bloom->mips[0]};
        for (const auto& source : sources)
            if (!source) throw std::invalid_argument("Composite: sampled source unavailable");
        u64 sourceGeneration = 1;
        if (const auto* prior = m_CompositeStates.Find(id))
        {
            sourceGeneration = (*prior)->sourceGeneration;
            if ((*prior)->sources != sources)
            {
                if (sourceGeneration == std::numeric_limits<u64>::max())
                    throw std::runtime_error("Composite: source generation exhausted");
                ++sourceGeneration;
            }
        }
        const auto config = CompositeViewState::Config(sources[0]->GetWidth(), sources[0]->GetHeight(), sourceGeneration);
        return m_CompositeStates.Ensure(id, config, [&](const ViewStateConfig& requested) {
            auto state = CompositeViewState::Create(id, requested, m_DescSetLayout);
            state->sources = sources;
            WriteCompositeView(*state);
            return state;
        }, [] { Renderer::WaitForGPU(); });
    }
    void PostProcessSubsystem::ReleaseCompositeView(RenderViewId id)
    {
        m_CompositeStates.Release(id, [] { Renderer::WaitForGPU(); });
    }
    void PostProcessSubsystem::WriteCompositeView(CompositeViewState& state)
    {
        std::array<VkDescriptorImageInfo, 2> images{};
        for (u32 i = 0; i < images.size(); ++i)
        {
            if (!state.sources[i]) throw std::invalid_argument("Composite: missing stable source");
            images[i] = {m_Sampler, std::static_pointer_cast<VKTexture>(state.sources[i])->GetImageView(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
        std::array<VkWriteDescriptorSet, 2 * MAX_FRAMES_IN_FLIGHT> writes{};
        for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot)
            for (u32 binding = 0; binding < images.size(); ++binding)
            {
                auto& write = writes[2 * slot + binding];
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = state.sets[slot]; write.dstBinding = binding; write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo = &images[binding];
            }
        vkUpdateDescriptorSets(VulkanContext::Get().GetDevice(), (u32)writes.size(), writes.data(), 0, nullptr);
    }
}
