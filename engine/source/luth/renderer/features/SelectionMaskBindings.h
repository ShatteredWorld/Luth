#pragma once
#include "luth/renderer/features/EditorOverlayViewState.h"
#include "luth/renderer/features/RenderResource.h"
#include "luth/renderer/draw/DrawCommand.h"
#include "luth/renderer/resources/Buffer.h"
#include <string>
#include <vector>
namespace Luth
{
    struct SelectionMaskPushConstants { ObjectPushConstants base; Vec2 jitter; };
    static_assert(sizeof(SelectionMaskPushConstants) == 88);
    struct SelectionMaskDraw
    {
        VkBuffer vertex = VK_NULL_HANDLE, index = VK_NULL_HANDLE;
        u32 indexCount = 0, entityIndex = 0;
        bool skinned = false;
        ObjectPushConstants constants{};
        std::string meshName, entityName;
        std::shared_ptr<VertexBuffer> vertexOwner;
        std::shared_ptr<IndexBuffer> indexOwner;
    };
    struct SelectionMaskBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE, skinnedPipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE, skinnedLayout = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 5> sets{};
        std::shared_ptr<EditorOverlayViewState> state;
        std::array<VkImage, 2> images{};
        std::array<VkImageView, 2> views{};
        u32 width = 0, height = 0;
        Vec2 jitter{};
        std::vector<SelectionMaskDraw> draws;
        bool enabled = false;
    };
    struct SelectionMaskGraphOutput { RG::ResourceHandle mask, depth; };
}
