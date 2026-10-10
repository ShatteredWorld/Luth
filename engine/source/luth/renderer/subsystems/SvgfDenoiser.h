#pragma once

#include "luth/renderer/subsystems/IDenoiser.h"
#include "luth/renderer/backend/vulkan/VulkanComputePipeline.h"

#include "luth/renderer/features/rt/DiDenoiserViewState.h"
#include "luth/renderer/features/rt/GiDenoiserViewState.h"
#include "luth/renderer/features/rt/ReflectionDenoiserViewState.h"
#include "luth/renderer/features/rt/DiDenoiserBindings.h"
#include "luth/renderer/features/rt/GiDenoiserBindings.h"
#include "luth/renderer/features/rt/ReflectionDenoiserBindings.h"
#include <memory>
#include <string>
#include <vector>

namespace Luth
{
    class FrameTargets;
    class RenderPipeline;
    struct ViewResources;
    struct SvgfSettings;

    // Which signal this instance denoises. Selects domain-owned view state, settings and
    // RG/debug pass names. Di/Gi
    // denoise a demodulated diffuse irradiance; Reflections denoises the RT specular radiance via a
    // SPECULAR reproject variant (svgf_spec_reproject.slang, hit-distance virtual reprojection,
    // b3 = slim roughness). DiSpecular denoises the ReSTIR-DI demodulated specular with the ordinary
    // MOTION reproject (svgf_reproject.slang, b3 = slim motion); direct point-light specular is
    // surface-attached, not a reflection's virtual image. see arch/rendering-pipeline.md
    enum class DenoiserChannel { Di, Gi, Reflections, DiSpecular };

    // Custom SVGF (Schied 2017) diffuse denoiser. Consumes a ReSTIR pass's demodulated irradiance
    // (DI or GI per channel) and returns a denoised image the GeometryPass reads + the lighting set
    // binds. Disabled -> a no-op pass-through (copies input -> output) preserving the output contract;
    // reproject -> moments -> a-trous chain when enabled. see arch/rendering-pipeline.md
    class SvgfDenoiser : public IDenoiser
    {
    public:
        explicit SvgfDenoiser(DenoiserChannel channel = DenoiserChannel::Di) : m_Channel(channel) {}

        void Init(RenderPipeline& pipeline) override;
        void Shutdown() override;
        bool OnShaderReloaded(const std::string& name, const std::vector<u32>& spv) override;
        void AllocateViewSets(ViewResources& vr) override;
        void WriteView(ViewResources& vr, FrameTargets& targets) override;
        RG::ResourceHandle AddPasses(RG::RenderGraph& rg, const DenoiseInputs& in) override;
        bool IsEnabled() const override;
        std::shared_ptr<DiDenoiserViewState> EnsureDiView(RenderViewId, FrameTargets&,
            const std::shared_ptr<RestirDiViewState>&);
        void ReleaseDiView(RenderViewId);
        std::shared_ptr<GiDenoiserViewState> EnsureGiView(RenderViewId, FrameTargets&,
            const std::shared_ptr<RestirGiViewState>&);
        void ReleaseGiView(RenderViewId);
        std::shared_ptr<ReflectionDenoiserViewState> EnsureReflectionView(RenderViewId, FrameTargets&,
            const std::shared_ptr<ReflectionViewState>&);
        void ReleaseReflectionView(RenderViewId);
        DiDenoiserBindings PrepareDiBindings(const ViewResources&, u64 frameIndex, RenderViewId,
            u64 generation, const SvgfSettings&) const;
        static RG::ResourceHandle AddDiPasses(RG::RenderGraph&,
            const std::array<RG::ResourceHandle, 6>&, const DiDenoiserBindings&);
        GiDenoiserBindings PrepareGiBindings(const ViewResources&, u64 frameIndex, RenderViewId,
            u64 generation, const SvgfSettings&) const;
        static RG::ResourceHandle AddGiPasses(RG::RenderGraph&,
            const std::array<RG::ResourceHandle, 6>&, const GiDenoiserBindings&);
        ReflectionDenoiserBindings PrepareReflectionBindings(const ViewResources&, u64 frameIndex, RenderViewId,
            u64 generation, const SvgfSettings&) const;
        static RG::ResourceHandle AddReflectionPasses(RG::RenderGraph&,
            const std::array<RG::ResourceHandle, 5>&, const ReflectionDenoiserBindings&);

    private:
        void WriteNativeView(ViewResources&, FrameTargets&);
        // Channel-selected UI settings; graph recording uses frozen native packets.
        const SvgfSettings& Settings() const;

        DiDenoiserViewStates m_DiViews;
        GiDenoiserViewStates m_GiViews;
        ReflectionDenoiserViewStates m_ReflectionViews;
        u64 m_NextSourceGeneration = 1;
        DenoiserChannel m_Channel = DenoiserChannel::Di;
        RenderPipeline* m_Pipeline = nullptr;

        std::unique_ptr<VKComputePipeline> m_PassthroughPipeline;
        std::unique_ptr<VKComputePipeline> m_ReprojectPipeline;
        std::unique_ptr<VKComputePipeline> m_MomentsPipeline;
        std::unique_ptr<VKComputePipeline> m_AtrousPipeline;

        VkSampler             m_Sampler        = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_PassLayout      = VK_NULL_HANDLE;  // passthrough set: b0 DI in, b1 out
        VkDescriptorSetLayout m_ReprojectLayout = VK_NULL_HANDLE;  // reproject set: b0-b3 in, b4-b9 history, b10 matID
        VkDescriptorSetLayout m_MomentsLayout   = VK_NULL_HANDLE;  // moments set: b0-b1 hist, b2-b3 samplers, b4 out
        VkDescriptorSetLayout m_AtrousLayout    = VK_NULL_HANDLE;  // a-trous set: b0 in, b1-b2 samplers, b3 out, b4 denoised, b5 rough

        std::vector<u32> m_PassthroughSpv;
        std::vector<u32> m_ReprojectSpv;
        std::vector<u32> m_MomentsSpv;
        std::vector<u32> m_AtrousSpv;
    };
}
