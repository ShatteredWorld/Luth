#include "luthpch.h"
#include "luth/renderer/features/FogViewState.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    ViewStateConfig FogViewState::Config(u32 width, u32 height, VolumetricSettings::Quality quality)
    {
        if (!width || !height || static_cast<u32>(quality) > static_cast<u32>(VolumetricSettings::Quality::High))
            throw std::invalid_argument("Fog: invalid extent or quality");
        return {width, height, static_cast<u32>(quality)};
    }

    std::shared_ptr<FogViewState> FogViewState::Create(RenderViewId id, const ViewStateConfig& config,
        const std::array<VkDescriptorSetLayout, 6>& layouts)
    {
        if (!id.value) throw std::invalid_argument("Fog: invalid view identity");
        Config(config.width, config.height, static_cast<VolumetricSettings::Quality>(config.signature));
        for (auto layout : layouts)
            if (!layout) throw std::runtime_error("Fog: native layout is unavailable");
        auto state = std::make_shared<FogViewState>();
        const auto quality = static_cast<VolumetricSettings::Quality>(config.signature);
        const auto dims = Volumetric::GetAtlasDims(quality);
        state->volQualityCached = static_cast<u32>(quality);
        state->volDimX = dims.x; state->volDimY = dims.y; state->volDimZ = dims.z;
        auto makeVolume = [&] {
            return std::make_shared<VKTexture>(dims.x, dims.y, dims.z,
                TextureFormat::RGBA16F, VK_IMAGE_USAGE_STORAGE_BIT);
        };
        state->volDensity = makeVolume();
        state->volInScatter = makeVolume();
        state->volInScatterHistA = makeVolume();
        state->volInScatterHistB = makeVolume();

        // Six cycled layouts: density (1 storage, 1 buffer, 1 sampler), scatter
        // (1 storage, 3 buffers, 2 samplers), integrate (1 storage, 1 sampler),
        // resolve (1 storage, 2 samplers), composite/viz (3 samplers each).
        state->device = VulkanContext::Get().GetDevice();
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * MAX_FRAMES_IN_FLIGHT},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * MAX_FRAMES_IN_FLIGHT},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12 * MAX_FRAMES_IN_FLIGHT}
        };
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = 6 * MAX_FRAMES_IN_FLIGHT;
        pool.poolSizeCount = 3; pool.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(state->device, &pool, nullptr, &state->pool) != VK_SUCCESS)
            throw std::runtime_error("Fog: descriptor pool allocation failed");
        std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT>* sets[] = {
            &state->volInjectDensityDescSet, &state->volInjectScatterDescSet,
            &state->volIntegrateDescSet, &state->volResolveDescSet,
            &state->volCompositeDescSet, &state->volVizDescSet
        };
        const char* names[] = {"Density", "Scatter", "Integrate", "Resolve", "Composite", "Viz"};
        for (u32 pass = 0; pass < layouts.size(); ++pass)
        {
            if (!layouts[pass]) throw std::runtime_error("Fog: native layout is unavailable");
            std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> cycled;
            cycled.fill(layouts[pass]);
            VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            alloc.descriptorPool = state->pool;
            alloc.descriptorSetCount = MAX_FRAMES_IN_FLIGHT; alloc.pSetLayouts = cycled.data();
            if (vkAllocateDescriptorSets(state->device, &alloc, sets[pass]->data()) != VK_SUCCESS)
                throw std::runtime_error("Fog: descriptor set allocation failed");
            for (u32 slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot)
            {
                const auto name = std::string("Fog.") + names[pass] + ".View" + std::to_string(id.value)
                    + ".Slot" + std::to_string(slot);
                VulkanContext::SetDebugName((*sets[pass])[slot], name.c_str());
            }
        }

        // Preserve the legacy zero bootstrap and GENERAL layout for scratch/history.
        // Density is fully overwritten by injection and was never bootstrap-cleared.
        VkImage images[] = {
            static_cast<VKTexture*>(state->volInScatter.get())->GetImage(),
            static_cast<VKTexture*>(state->volInScatterHistA.get())->GetImage(),
            static_cast<VKTexture*>(state->volInScatterHistB.get())->GetImage()
        };
        VulkanContext::Get().ImmediateSubmit([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier barriers[3]{};
            for (u32 i = 0; i < 3; ++i)
            {
                auto& barrier = barriers[i];
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = images[i];
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 3, barriers);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (auto image : images)
                vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            for (auto& barrier : barriers)
            {
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            }
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 3, barriers);
        });
        return state;
    }

    FogViewState::~FogViewState()
    {
        if (!pool) return; // Headless state has no native pool.
        VulkanContext::Get().PushDeletion([device = device, pool = pool] {
            vkDestroyDescriptorPool(device, pool, nullptr);
        });
        // Retire resources in the same completion slot as their descriptor pool.
        volDensity.reset(); volInScatter.reset();
        volInScatterHistA.reset(); volInScatterHistB.reset();
        depthSource.reset();
    }
}
