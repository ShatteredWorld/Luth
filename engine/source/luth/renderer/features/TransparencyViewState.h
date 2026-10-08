#pragma once
#include "luth/core/FrameData.h"
#include "luth/renderer/features/RenderViewState.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
#include <array>
#include <vulkan/vulkan.h>

namespace Luth
{
    struct FogViewState;
    // Owns physical resources and native bindings; graph handles remain execution-local.
    struct TransparencyViewState
    {
        TransparencyViewState() = default;
        ~TransparencyViewState();
        TransparencyViewState(const TransparencyViewState&) = delete;
        TransparencyViewState& operator=(const TransparencyViewState&) = delete;
        static ViewStateConfig Config(u32 width, u32 height, u32 layers);
        static u64 NodeBytes(const ViewStateConfig&);
        static std::shared_ptr<TransparencyViewState> Create(RenderViewId, const ViewStateConfig&,
            const std::array<VkDescriptorSetLayout, 2>&, u32 reservedTag);
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> transparentDescSet{};
        VkDescriptorSet oitResolveDescSet = VK_NULL_HANDLE;
        std::shared_ptr<Texture> refractionBackdrop, oitHeads;
        Memory::GPUSubRegion oitNodes{};
        u32 oitNodesTag = 0;
        u32 width = 0, height = 0, oitLayersCached = 0;
        // Retain each cycled descriptor's sampled fog sources until that slot is rewritten.
        std::array<std::shared_ptr<FogViewState>, MAX_FRAMES_IN_FLIGHT> fogBindings{};
    };
    using TransparencyViewStateStore = FeatureViewStates<std::shared_ptr<TransparencyViewState>>;
}
