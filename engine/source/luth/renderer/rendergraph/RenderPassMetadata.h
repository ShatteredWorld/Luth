#pragma once
#include "luth/core/types/LuthTypes.h"
#include <string>
#include <utility>

namespace Luth::RG
{
    // Frozen CPU diagnostics, independent of pass names and GPU query results.
    // Counts describe recorded commands/index budgets, not GPU-visible primitives.
    struct RenderPassMetadata
    {
        std::string shaderName;
        bool pipelineStateAvailable = false, pipelineStateMixed = false;
        bool depthTest = false, depthWrite = false, blendEnabled = false;
        u32 cullMode = 0;
        bool geometryStatsAvailable = false, indirectDraws = false;
        u32 drawCalls = 0, indices = 0;

        static RenderPassMetadata Graphics(std::string shader, bool depthTest, bool depthWrite,
            bool blend, u32 cull, u32 draws = 1, u32 indices = 0)
        {
            RenderPassMetadata result;
            result.shaderName = std::move(shader); result.pipelineStateAvailable = true;
            result.depthTest = depthTest; result.depthWrite = depthWrite;
            result.blendEnabled = blend; result.cullMode = cull;
            result.geometryStatsAvailable = true; result.drawCalls = draws; result.indices = indices;
            return result;
        }
        template<class Packets> void AddDraws(const Packets& packets)
        {
            for (const auto& packet : packets) { ++drawCalls; indices += packet.indexCount; }
        }
    };
}
