#pragma once
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include <array>
#include <vulkan/vulkan.h>
namespace Luth
{
    struct ClusterVizViewState
    {
        ClusterVizViewState() = default;
        ClusterVizViewState(const ClusterVizViewState&) = delete;
        ClusterVizViewState& operator=(const ClusterVizViewState&) = delete;
        ~ClusterVizViewState();
        static ViewStateConfig Config(u32 width, u32 height, u64 sourceGeneration = 1);
        static std::shared_ptr<ClusterVizViewState> Create(RenderViewId, const ViewStateConfig&, VkDescriptorSetLayout);
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::shared_ptr<Texture> depth;
        u64 sourceGeneration = 1;
    };
    using ClusterVizViewStateStore = FeatureViewStates<std::shared_ptr<ClusterVizViewState>>;
}
