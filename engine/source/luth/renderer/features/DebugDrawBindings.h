#pragma once
#include "luth/core/DebugDraw.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
namespace Luth
{
    struct DebugDrawBindings
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        Memory::GPUSubRegion vertices;
        Mat4 viewProj{1};
        u64 renderFrameIndex = 0;
        u32 vertexCount = 0;
        bool enabled = false;
    };
}
