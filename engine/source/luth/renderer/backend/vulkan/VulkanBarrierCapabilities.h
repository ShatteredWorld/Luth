#pragma once
#include <vulkan/vulkan.h>

namespace Luth
{
    // Enabled native mechanisms, not technique activation. Ray queries execute in
    // ordinary fragment/compute stages and do not require an RT-pipeline stage.
    struct VulkanBarrierCapabilities
    {
        bool accelerationStructures = false;
        bool rayTracingPipelines = false;
        // Temporary bridge from the current all-or-nothing device RT package.
        static constexpr VulkanBarrierCapabilities ForEnabledRtPackage(bool enabled)
        { return {enabled, enabled}; }
        constexpr VkPipelineStageFlags2 RayTracingShaderStage() const
        { return rayTracingPipelines ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR : 0; }
        constexpr VkPipelineStageFlags2 SampledImageReadStages() const
        {
            return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                | RayTracingShaderStage();
        }
        constexpr VkPipelineStageFlags2 ComputeHistoryWaitStages() const
        {
            return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                | (accelerationStructures ? VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR : 0);
        }
        constexpr VkPipelineStageFlags2 DeformationReadStages() const
        {
            return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                | (accelerationStructures ? VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR : 0);
        }
    };
}
