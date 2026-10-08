#include "luthpch.h"
#include "luth/renderer/backend/vulkan/GPUTimerPool.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"

namespace Luth
{
    void GPUTimerPool::Init(u32 maxPasses)
    {
        m_MaxPasses = maxPasses;

        auto& ctx = VulkanContext::Get();
        VkDevice device = ctx.GetDevice();
        m_TimestampPeriod = ctx.GetPhysicalDeviceProperties().limits.timestampPeriod;

        if (m_TimestampPeriod == 0.0f)
        {
            LH_LOG(Renderer, warn, "GPU does not support timestamps (timestampPeriod == 0)");
            return;
        }

        VkQueryPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        poolInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        poolInfo.queryCount = maxPasses * 2;  // begin + end per pass

        for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            VkResult result = vkCreateQueryPool(device, &poolInfo, nullptr, &m_Pools[i]);
            LH_CORE_ASSERT(result == VK_SUCCESS, "Failed to create GPU timer query pool");
        }

        // Pipeline-statistics pool (graphics passes only): one query per pass. Skipped on GPUs lacking
        // pipelineStatisticsQuery + inheritedQueries; stats then stay unavailable but timing still works.
        m_StatsSupported = ctx.SupportsPipelineStats();
        if (m_StatsSupported)
        {
            VkQueryPoolCreateInfo statsInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
            statsInfo.queryType          = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            statsInfo.queryCount         = maxPasses;
            statsInfo.pipelineStatistics = k_StatsFlags;
            for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                VkResult result = vkCreateQueryPool(device, &statsInfo, nullptr, &m_StatsPools[i]);
                LH_CORE_ASSERT(result == VK_SUCCESS, "Failed to create GPU pipeline-stats query pool");
            }
        }

        m_RecordingSlot = 0;
        m_Initialized = true;
        LH_LOG(Renderer, info, "GPUTimerPool initialized: {} max passes, {:.2f} ns/tick", maxPasses, m_TimestampPeriod);
    }

    void GPUTimerPool::Shutdown()
    {
        if (!m_Initialized) return;

        VkDevice device = VulkanContext::Get().GetDevice();
        for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (m_Pools[i] != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(device, m_Pools[i], nullptr);
                m_Pools[i] = VK_NULL_HANDLE;
            }
            if (m_StatsPools[i] != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(device, m_StatsPools[i], nullptr);
                m_StatsPools[i] = VK_NULL_HANDLE;
            }
        }
        m_Initialized = false;
    }

    void GPUTimerPool::ResetForFrame(VkCommandBuffer cmd)
    {
        if (!m_Initialized) return;

        u32 poolIndex = m_RecordingSlot;
        vkCmdResetQueryPool(cmd, m_Pools[poolIndex], 0, m_MaxPasses * 2);
        if (m_StatsSupported)
            vkCmdResetQueryPool(cmd, m_StatsPools[poolIndex], 0, m_MaxPasses);
    }

    void GPUTimerPool::WriteTimestamp(VkCommandBuffer cmd, u32 passIndex, bool isBegin)
    {
        if (!m_Initialized) return;
        if (passIndex >= m_MaxPasses) return;

        u32 poolIndex  = m_RecordingSlot;
        u32 queryIndex = passIndex * 2 + (isBegin ? 0 : 1);

        VkPipelineStageFlagBits2 stage = isBegin
            ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
            : VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

        vkCmdWriteTimestamp2(cmd, stage, m_Pools[poolIndex], queryIndex);
    }

    void GPUTimerPool::ReadResults(u32 slot, u32 passCount, std::vector<float>& outTimesMs)
    {
        outTimesMs.assign(passCount, -1.0f);
        if (!m_Initialized || slot >= MAX_FRAMES_IN_FLIGHT || !passCount || passCount > m_MaxPasses) return;
        const u32 queryCount = passCount * 2;
        // Availability is checked even after completion; query errors remain explicitly unavailable.
        std::vector<u64> raw(queryCount * 2);
        const VkResult result = vkGetQueryPoolResults(VulkanContext::Get().GetDevice(), m_Pools[slot],
            0, queryCount, raw.size() * sizeof(u64), raw.data(), 2 * sizeof(u64),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result != VK_SUCCESS && result != VK_NOT_READY) return;
        for (u32 i = 0; i < passCount; ++i)
        {
            if (!raw[i * 4 + 1] || !raw[i * 4 + 3]) continue;
            outTimesMs[i] = static_cast<float>((raw[i * 4 + 2] - raw[i * 4]) * (double)m_TimestampPeriod / 1e6);
        }
    }
    void GPUTimerPool::BeginStats(VkCommandBuffer cmd, u32 passIndex)
    {
        if (!m_StatsSupported || passIndex >= m_MaxPasses) return;
        u32 slot = m_RecordingSlot;
        vkCmdBeginQuery(cmd, m_StatsPools[slot], passIndex, 0);
    }

    void GPUTimerPool::EndStats(VkCommandBuffer cmd, u32 passIndex)
    {
        if (!m_StatsSupported || passIndex >= m_MaxPasses) return;
        u32 slot = m_RecordingSlot;
        vkCmdEndQuery(cmd, m_StatsPools[slot], passIndex);
    }

    // Reads a completed slot using frozen recording metadata.
    // Per-query availability flags compute passes / disabled frames.
    void GPUTimerPool::ReadStats(u32 readSlot, u32 passCount, bool recorded, std::vector<RG::GpuPipelineStats>& out)
    {
        out.assign(passCount, {});
        if (!m_StatsSupported || !recorded || readSlot >= MAX_FRAMES_IN_FLIGHT) return;
        if (passCount == 0 || passCount > m_MaxPasses) return;

        const u32 stride = k_StatsValues + 1;  // + availability word
        std::vector<u64> raw(passCount * stride);
        VkResult result = vkGetQueryPoolResults(
            VulkanContext::Get().GetDevice(),
            m_StatsPools[readSlot],
            0, passCount,
            raw.size() * sizeof(u64), raw.data(),
            stride * sizeof(u64),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result != VK_SUCCESS && result != VK_NOT_READY) return;

        for (u32 i = 0; i < passCount; i++)
        {
            const u64* q = &raw[i * stride];
            if (q[k_StatsValues] == 0) continue;  // unavailable -> compute pass, or stats not recorded
            RG::GpuPipelineStats& s = out[i];
            s.inputVertices   = q[0];
            s.inputPrimitives = q[1];
            s.vsInvocations   = q[2];
            s.clipInvocations = q[3];
            s.clipPrimitives  = q[4];
            s.fsInvocations   = q[5];
            s.valid = true;
        }
    }
}
