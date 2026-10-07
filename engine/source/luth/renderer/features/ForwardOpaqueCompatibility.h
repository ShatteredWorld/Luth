#pragma once
#include "luth/renderer/features/ForwardOpaqueFeature.h"

namespace Luth
{
    // Temporary explicit extension until the optional RT package supplies hybrid shading.
    namespace ForwardCompatibilityResources
    {
#define LUTH_FORWARD_COMPAT(Name) \
        inline constexpr ResourceKeyIdentity Name##Identity{"ForwardCompatibility." #Name}; \
        inline constexpr RenderResourceKey<GraphTextureRef> Name{&Name##Identity}
        LUTH_FORWARD_COMPAT(SunShadowMask);
        LUTH_FORWARD_COMPAT(DenoisedDiffuseDI);
        LUTH_FORWARD_COMPAT(DenoisedDiffuseGI);
        LUTH_FORWARD_COMPAT(DenoisedReflectionRadiance);
        LUTH_FORWARD_COMPAT(DenoisedSpecularDI);
#undef LUTH_FORWARD_COMPAT
    }
    class HybridForwardOpaqueFeature final : public ForwardOpaqueFeature
    {
    public:
        using ForwardOpaqueFeature::ForwardOpaqueFeature;
        FeatureInfo Describe() const override;
    protected:
        void AppendCompatibilityReads(RenderFeatureContext&, std::vector<RG::ResourceHandle>&) const override;
    };
}
