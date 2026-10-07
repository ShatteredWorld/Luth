#pragma once

#include "luth/renderer/features/RenderFeature.h"
#include <concepts>
#include <limits>
#include <memory>
#include <utility>

namespace Luth
{
    class RenderPipelineDefinition
    {
    public:
        RenderPipelineDefinition() = default;
        RenderPipelineDefinition(RenderPipelineDefinition&&) = default;
        RenderPipelineDefinition& operator=(RenderPipelineDefinition&&) = default;
        RenderPipelineDefinition(const RenderPipelineDefinition&) = delete;
        RenderPipelineDefinition& operator=(const RenderPipelineDefinition&) = delete;

        template<std::derived_from<IRenderFeature> Feature, class... Args>
        FeatureInstanceId AddFeature(Args&&... args)
        {
            if (m_Features.size() >= std::numeric_limits<FeatureInstanceId>::max())
                throw std::length_error("Too many render features");
            const auto id = static_cast<FeatureInstanceId>(m_Features.size() + 1);
            m_Features.push_back({id, &RenderFeatureType<Feature>,
                std::make_unique<Feature>(std::forward<Args>(args)...)});
            return id;
        }
        void Before(FeatureInstanceId first, FeatureInstanceId second)
        {
            m_Ordering.emplace_back(first, second);
        }
    private:
        friend class RenderPipelineCompiler;
        struct Entry
        {
            FeatureInstanceId id;
            FeatureTypeId concreteType;
            std::unique_ptr<IRenderFeature> feature;
        };
        std::vector<Entry> m_Features;
        std::vector<std::pair<FeatureInstanceId, FeatureInstanceId>> m_Ordering;
    };
}
