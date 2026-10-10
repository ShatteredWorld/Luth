#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/features/CompositeViewState.h"
#include "luth/renderer/settings/PostProcessSettings.h"

namespace Luth
{
    PostProcessUBO MakeCompositeUniforms(const PostProcessSettings&, bool dataDebug, bool hasBloom, float time);
    struct CompositeBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        TextureBindingRef source, bloom, output;
        VkImage outputImage = VK_NULL_HANDLE;
        VkImageView outputView = VK_NULL_HANDLE;
        VkDescriptorBufferInfo uniform{}; // Preserve the tagged heap's physical offset and range.
        u32 width = 0, height = 0;
        PostProcessUBO parameters{};
        std::shared_ptr<CompositeViewState> state;
        std::shared_ptr<Texture> outputOwner;
    };
}
