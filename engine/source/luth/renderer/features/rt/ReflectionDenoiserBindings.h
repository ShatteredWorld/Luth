#pragma once
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/features/rt/ReflectionDenoiserViewState.h"
#include "luth/renderer/settings/SvgfSettings.h"

namespace Luth
{
    struct ReflectionDenoiserBindings
    {
        std::array<VkPipeline, 4> pipelines{}; // Copy, reproject, moments, a-trous.
        std::array<VkPipelineLayout, 4> layouts{};
        VkDescriptorSet globalSet = VK_NULL_HANDLE, copySet = VK_NULL_HANDLE;
        VkDescriptorSet reprojectSet = VK_NULL_HANDLE, momentsSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> atrousSets{};
        // Noisy radiance/hit distance, depth, normal, roughness, material ID.
        std::array<TextureBindingRef, 5> sources{};
        std::array<VkImage, 5> sourceImages{};
        std::array<VkImageView, 5> sourceViews{};
        // Current color/moments, a-trous ping/pong. Previous history and geom retain native cross-frame rules.
        std::array<TextureBindingRef, 4> working{};
        std::array<VkImage, 4> workingImages{};
        std::array<VkImageView, 4> workingViews{};
        TextureBindingRef output;
        VkImage outputImage = VK_NULL_HANDLE;
        VkImageView outputView = VK_NULL_HANDLE;
        std::shared_ptr<const ReflectionDenoiserViewState> retained;
        SvgfSettings settings;
        bool historyValid = false;
        f32 HistoryCap() const { return historyValid ? static_cast<f32>(settings.historyCap) : 0.0f; }
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
