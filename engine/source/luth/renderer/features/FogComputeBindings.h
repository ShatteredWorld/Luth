#pragma once
#include "luth/core/types/LuthMath.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
#include "luth/renderer/features/RenderResource.h"
#include <vulkan/vulkan.h>

namespace Luth
{
    class RtSubsystem;
    struct FogImageBinding
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        TextureBindingRef binding;
    };
    struct FogInjectConstants
    {
        Mat4 invView{1};
        u32 volDimX = 0, volDimY = 0, volDimZ = 0, pad = 0;
        u64 geomTableBDA = 0;
    };
    static_assert(sizeof(FogInjectConstants) == 88);
    struct FogIntegrateConstants
    {
        Vec4 nearFarPad{};
        u32 volDimX = 0, volDimY = 0, volDimZ = 0, pad = 0;
    };
    struct FogResolveConstants
    {
        Mat4 invView{1};
        u32 volDimX = 0, volDimY = 0, volDimZ = 0, pad = 0;
    };
    struct FogComputeBindings
    {
        std::array<VkPipeline, 4> pipelines{};
        std::array<VkPipelineLayout, 4> layouts{};
        std::array<VkDescriptorSet, 4> sets{};
        VkDescriptorSet global = VK_NULL_HANDLE, material = VK_NULL_HANDLE, bindless = VK_NULL_HANDLE;
        FogImageBinding density, scratch, previous, current;
        Memory::GPUSubRegion volumes{}, lights{}, grid{}, indices{};
        FogInjectConstants inject;
        FogIntegrateConstants integrate;
        FogResolveConstants resolve;
        // Temporary native RT bridge: TLAS recording still prepares the paired table.
        // Replace with prepared bindings when the scene provider migrates in M13.
        const RtSubsystem* rayScene = nullptr;
        bool currentHistoryA = false, rtShadows = false, enabled = false, ready = false;
    };
}
