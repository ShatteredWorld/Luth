#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/features/rt/GiDenoiserViewState.h"
#include "luth/renderer/settings/SvgfSettings.h"

namespace Luth
{
    struct GiDenoiserBindings
    {
        std::array<VkPipeline, 4> pipelines{}; // Copy, reproject, moments, a-trous.
        std::array<VkPipelineLayout, 4> layouts{};
        VkDescriptorSet globalSet = VK_NULL_HANDLE, copySet = VK_NULL_HANDLE;
        VkDescriptorSet reprojectSet = VK_NULL_HANDLE, momentsSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> atrousSets{};
        // Noisy diffuse, depth, normal, motion, material ID, roughness.
        std::array<TextureBindingRef, 6> sources{};
        std::array<VkImage, 6> sourceImages{};
        std::array<VkImageView, 6> sourceViews{};
        // Current color/moments, a-trous ping/pong. Previous history and geom retain native cross-frame rules.
        std::array<TextureBindingRef, 4> working{};
        std::array<VkImage, 4> workingImages{};
        std::array<VkImageView, 4> workingViews{};
        TextureBindingRef output;
        VkImage outputImage = VK_NULL_HANDLE;
        VkImageView outputView = VK_NULL_HANDLE;
        std::shared_ptr<const GiDenoiserViewState> retained;
        SvgfSettings settings;
        RenderViewId view;
        u64 generation = 0, frameIndex = 0;
        u32 width = 0, height = 0, fullWidth = 0, fullHeight = 0;
        bool ChainReady() const {
            return settings.enabled && pipelines[1] && pipelines[2] && pipelines[3]
                && reprojectSet && momentsSet && atrousSets[0] && atrousSets[1];
        }
        bool Ready() const { return ChainReady() || (pipelines[0] && copySet); }
    };
}
