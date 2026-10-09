#pragma once
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/core/FrameData.h"
#include <vulkan/vulkan.h>
#include <array>
#include <stdexcept>

namespace Luth
{
    struct RtSunShadowPoolBudget
    {
        std::array<VkDescriptorPoolSize, 2> sizes;
        u32 maxSets;
        explicit RtSunShadowPoolBudget(u32 frames) : sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * frames},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, frames}}}, maxSets(frames)
        {
            if (!frames) throw std::invalid_argument("RtSunShadow: invalid descriptor frame count");
        }
    };
    // Physical resources only. Retained sampled sources prevent native view identity reuse
    // while their stable descriptors and the mask are still referenced.
    struct RtSunShadowViewState
    {
        RtSunShadowViewState() = default;
        ~RtSunShadowViewState();
        RtSunShadowViewState(const RtSunShadowViewState&) = delete;
        RtSunShadowViewState& operator=(const RtSunShadowViewState&) = delete;
        static ViewStateConfig Config(u32 width, u32 height, u64 depthIdentity, u64 normalIdentity);
        static std::shared_ptr<RtSunShadowViewState> Create(RenderViewId,
            const ViewStateConfig&, VkDescriptorSetLayout);
        RenderViewId id;
        u32 width = 0, height = 0;
        std::shared_ptr<Texture> mask, depthSource, normalSource;
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> sets{};
    };
    using RtSunShadowViewStates = FeatureViewStates<std::shared_ptr<RtSunShadowViewState>>;
}
