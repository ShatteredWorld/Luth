#include "luthpch.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/settings/SvgfSettings.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/scene/systems/RenderingSystem.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/core/FrameData.h"

namespace Luth
{
    namespace {
        // gbufferScale/dispatchW/dispatchH let a channel run at a working resolution below the full
        // G-buffer (half-res GI). scale==1 + dispatch==full is the identity path for full-res channels.
        // Mirrors svgf_reproject.slang's push_constant (7 floats + 3 ints = 40 B); the spec variant
        // shares the layout (confidenceScale unused there - its input alpha is hitDist).
        struct SvgfReprojectPC {
            f32 alphaColor;
            f32 alphaMoments;
            f32 historyCap;
            f32 depthThreshold;
            f32 normalThreshold;
            i32 gbufferScale;
            i32 dispatchW;
            i32 dispatchH;
            f32 antiFireflySigma;   // 3x3 mean + k*sigma clamp on the incoming sample; 0 = off
            f32 confidenceScale;    // reservoir-confidence history-cap shortening; 0 = off / spec variant
        };
        static_assert(sizeof(SvgfReprojectPC) == 40, "SvgfReprojectPC must match svgf_reproject.slang push_constant");

        // Mirrors svgf_moments.slang's push_constant (2 floats + 3 ints = 20 B).
        struct SvgfMomentsPC {
            f32 phiDepth;
            f32 phiNormal;
            i32 gbufferScale;
            i32 dispatchW;
            i32 dispatchH;
        };
        static_assert(sizeof(SvgfMomentsPC) == 20, "SvgfMomentsPC must match svgf_moments.slang push_constant");

        // Mirrors svgf_atrous.slang's push_constant (2 ints + 3 floats + 3 ints + 1 float = 36 B).
        struct SvgfAtrousPC {
            i32 stepSize;
            i32 writeFinal;
            f32 phiColor;
            f32 phiNormal;
            f32 phiDepth;
            i32 gbufferScale;
            i32 dispatchW;
            i32 dispatchH;
            f32 phiRough;   // roughness edge-stop; spec channels only, diffuse channels pass 0
        };
        static_assert(sizeof(SvgfAtrousPC) == 36, "SvgfAtrousPC must match svgf_atrous.slang push_constant");

        // Borrowed domain-state pointers used only while preparing immutable descriptor bindings.
        // All arrays are length two; denoised/noisy are single images.
        struct ChannelRefs {
            std::shared_ptr<Texture>* colorHist;   // [2]
            std::shared_ptr<Texture>* moments;     // [2]
            std::shared_ptr<Texture>* geom;        // [2]
            std::shared_ptr<Texture>* atrous;      // [2]
            std::shared_ptr<Texture>* denoised;    // single
            std::shared_ptr<Texture>* noisy;       // single: restirDI / restirGiDI (denoiser input)
            VkDescriptorSet*          passthroughSet;
            VkDescriptorSet*          reprojectSet; // [2]
            VkDescriptorSet*          momentsSet;   // [2]
            VkDescriptorSet*          atrousSet;    // [2]
        };
        ChannelRefs Resolve(DenoiserChannel ch, ViewResources& vr) {
            if (ch == DenoiserChannel::Reflections)
            {
                auto& state = *vr.reflectionDenoiser;
                return {state.svgfColorHist, state.svgfMoments, state.svgfGeom, state.svgfAtrous,
                    state.WorkingOutput(), state.Noisy(),
                    &state.svgfPassthroughDescSet, state.svgfReprojectDescSet,
                    state.svgfMomentsDescSet, state.svgfAtrousDescSet};
            }
            if (ch == DenoiserChannel::DiSpecular)
            {
                const auto& state = *vr.diSpecDenoiser;
                const bool diSpecHalf = state.width != state.svgfDenoised->GetWidth()
                    || state.height != state.svgfDenoised->GetHeight();
                return { vr.diSpecDenoiser->svgfColorHist, vr.diSpecDenoiser->svgfMoments, vr.diSpecDenoiser->svgfGeom, vr.diSpecDenoiser->svgfAtrous,
                         diSpecHalf ? &vr.diSpecDenoiser->svgfDiHalf : &vr.diSpecDenoiser->svgfDenoised, vr.diSpecDenoiser->Noisy(),
                         &vr.diSpecDenoiser->svgfPassthroughDescSet, vr.diSpecDenoiser->svgfReprojectDescSet,
                         vr.diSpecDenoiser->svgfMomentsDescSet, vr.diSpecDenoiser->svgfAtrousDescSet };
            }
            if (ch == DenoiserChannel::Gi)
            {
                if (!vr.giDenoiser) throw std::invalid_argument("SVGF GI: missing denoiser view owner");
                auto& state = *vr.giDenoiser;
                const bool half = state.width != state.svgfDenoised->GetWidth()
                    || state.height != state.svgfDenoised->GetHeight();
                return {state.svgfColorHist, state.svgfMoments, state.svgfGeom, state.svgfAtrous,
                    half ? &state.svgfGiHalf : &state.svgfDenoised, state.Noisy(),
                    &state.svgfPassthroughDescSet, state.svgfReprojectDescSet,
                    state.svgfMomentsDescSet, state.svgfAtrousDescSet};
            }
            const auto& state = *vr.diDenoiser;
            const bool diHalf = state.width != state.svgfDenoised->GetWidth()
                || state.height != state.svgfDenoised->GetHeight();
            return { vr.diDenoiser->svgfColorHist, vr.diDenoiser->svgfMoments, vr.diDenoiser->svgfGeom, vr.diDenoiser->svgfAtrous,
                     diHalf ? &vr.diDenoiser->svgfDiHalf : &vr.diDenoiser->svgfDenoised, vr.restirDi ? &vr.restirDi->restirDI : nullptr,
                     &vr.diDenoiser->svgfPassthroughDescSet, vr.diDenoiser->svgfReprojectDescSet,
                     vr.diDenoiser->svgfMomentsDescSet, vr.diDenoiser->svgfAtrousDescSet };
        }
    }

    std::shared_ptr<DiDenoiserViewState> SvgfDenoiser::EnsureDiView(RenderViewId id,
        FrameTargets& targets, const std::shared_ptr<RestirDiViewState>& input)
    {
        if ((m_Channel != DenoiserChannel::Di && m_Channel != DenoiserChannel::DiSpecular) || !m_PassLayout || !m_ReprojectLayout
            || !m_MomentsLayout || !m_AtrousLayout || !input) return {};
        std::array<std::shared_ptr<Texture>, 5> sources{targets.GetSceneDepth(), targets.GetSlimNormal(),
            targets.GetSlimMotion(), targets.GetSlimMaterialID(), targets.GetSlimRoughness()};
        std::array<VkImageView, 5> views{};
        for (u32 i = 0; i < sources.size(); ++i) {
            if (!sources[i]) throw std::invalid_argument("DI denoiser: missing descriptor source");
            views[i] = std::static_pointer_cast<VKTexture>(sources[i])->GetImageView();
        }
        const auto* prior = m_DiViews.Find(id);
        const u64 generation = prior && (*prior)->input == input && (*prior)->sources == sources
            && (*prior)->sourceViews == views ? (*prior)->sourceGeneration : m_NextSourceGeneration++;
        const bool half = input->width != targets.GetSceneColor()->GetWidth()
            || input->height != targets.GetSceneColor()->GetHeight();
        const auto config = DiDenoiserViewState::Config(targets.GetSceneColor()->GetWidth(),
            targets.GetSceneColor()->GetHeight(), half, generation);
        return m_DiViews.Ensure(id, config, [&](const ViewStateConfig& c) {
            auto state = DiDenoiserViewState::Create(id, c,
                {m_PassLayout, m_ReprojectLayout, m_MomentsLayout, m_AtrousLayout}, m_Channel == DenoiserChannel::Di ? DiDenoiserSignal::Diffuse : DiDenoiserSignal::Specular);
            state->input = input; state->sources = sources; state->sourceViews = views;
            ViewResources bridge; bridge.restirDi = input;
            if (m_Channel == DenoiserChannel::Di) bridge.diDenoiser = state;
            else bridge.diSpecDenoiser = state;
            WriteNativeView(bridge, targets);
            return state;
        }, [] { Renderer::WaitForGPU(); });
    }
    void SvgfDenoiser::ReleaseDiView(RenderViewId id)
    {
        m_DiViews.Release(id, [] { Renderer::WaitForGPU(); });
    }

    std::shared_ptr<GiDenoiserViewState> SvgfDenoiser::EnsureGiView(RenderViewId id,
        FrameTargets& targets, const std::shared_ptr<RestirGiViewState>& input)
    {
        if (m_Channel != DenoiserChannel::Gi || !m_PassLayout || !m_ReprojectLayout
            || !m_MomentsLayout || !m_AtrousLayout || !input) return {};
        const auto& color = targets.GetSceneColor();
        if (!color || !input->restirGiDI) throw std::invalid_argument("GI denoiser: missing color or raw input");
        std::array<std::shared_ptr<Texture>, 5> sources{targets.GetSceneDepth(), targets.GetSlimNormal(),
            targets.GetSlimMotion(), targets.GetSlimMaterialID(), targets.GetSlimRoughness()};
        std::array<VkImageView, 5> views{};
        for (u32 i = 0; i < sources.size(); ++i) {
            if (!sources[i] || sources[i]->GetWidth() != color->GetWidth() || sources[i]->GetHeight() != color->GetHeight())
                throw std::invalid_argument("GI denoiser: incompatible descriptor source");
            views[i] = std::static_pointer_cast<VKTexture>(sources[i])->GetImageView();
            if (!views[i]) throw std::invalid_argument("GI denoiser: missing source image view");
        }
        const auto* prior = m_GiViews.Find(id);
        const u64 generation = prior && (*prior)->input == input && (*prior)->sources == sources
            && (*prior)->sourceViews == views ? (*prior)->sourceGeneration : m_NextSourceGeneration++;
        const bool half = input->width != targets.GetSceneColor()->GetWidth()
            || input->height != targets.GetSceneColor()->GetHeight();
        const auto config = GiDenoiserViewState::Config(targets.GetSceneColor()->GetWidth(),
            targets.GetSceneColor()->GetHeight(), half, generation);
        const auto extent = RestirGiViewState::WorkingExtent(config);
        if (input->width != extent[0] || input->height != extent[1] || input->restirGiDI->GetWidth() != extent[0]
            || input->restirGiDI->GetHeight() != extent[1])
            throw std::invalid_argument("GI denoiser: incompatible raw working extent");
        return m_GiViews.Ensure(id, config, [&](const ViewStateConfig& c) {
            auto state = GiDenoiserViewState::Create(id, c,
                {m_PassLayout, m_ReprojectLayout, m_MomentsLayout, m_AtrousLayout});
            state->input = input; state->sources = sources; state->sourceViews = views;
            ViewResources bridge; bridge.restirGi = input; bridge.giDenoiser = state;
            WriteNativeView(bridge, targets);
            return state;
        }, [] { Renderer::WaitForGPU(); });
    }
    void SvgfDenoiser::ReleaseGiView(RenderViewId id)
    {
        m_GiViews.Release(id, [] { Renderer::WaitForGPU(); });
    }

    std::shared_ptr<ReflectionDenoiserViewState> SvgfDenoiser::EnsureReflectionView(RenderViewId id,
        FrameTargets& targets, const std::shared_ptr<ReflectionViewState>& input)
    {
        if (m_Channel != DenoiserChannel::Reflections || !m_PassLayout || !m_ReprojectLayout
            || !m_MomentsLayout || !m_AtrousLayout || !input) return {};
        if (input->id != id) throw std::invalid_argument("Reflection denoiser: incompatible raw view identity");
        const auto& color = targets.GetSceneColor();
        if (!color || !input->radiance) throw std::invalid_argument("Reflection denoiser: missing color or raw input");
        std::array<std::shared_ptr<Texture>, 4> sources{targets.GetSceneDepth(), targets.GetSlimNormal(),
            targets.GetSlimRoughness(), targets.GetSlimMaterialID()};
        std::array<VkImageView, 4> views{};
        for (u32 i = 0; i < sources.size(); ++i) {
            if (!sources[i] || sources[i]->GetWidth() != color->GetWidth() || sources[i]->GetHeight() != color->GetHeight())
                throw std::invalid_argument("Reflection denoiser: incompatible descriptor source");
            views[i] = std::static_pointer_cast<VKTexture>(sources[i])->GetImageView();
            if (!views[i]) throw std::invalid_argument("Reflection denoiser: missing source image view");
        }
        const auto* prior = m_ReflectionViews.Find(id);
        const u64 generation = prior && (*prior)->input == input && (*prior)->sources == sources
            && (*prior)->sourceViews == views ? (*prior)->sourceGeneration : m_NextSourceGeneration++;
        const bool half = input->width != targets.GetSceneColor()->GetWidth()
            || input->height != targets.GetSceneColor()->GetHeight();
        const auto config = ReflectionDenoiserViewState::Config(targets.GetSceneColor()->GetWidth(),
            targets.GetSceneColor()->GetHeight(), half, generation);
        const auto extent = ReflectionViewState::WorkingExtent(config);
        if (input->width != extent[0] || input->height != extent[1] || input->radiance->GetWidth() != extent[0]
            || input->radiance->GetHeight() != extent[1])
            throw std::invalid_argument("Reflection denoiser: incompatible raw working extent");
        return m_ReflectionViews.Ensure(id, config, [&](const ViewStateConfig& c) {
            auto state = ReflectionDenoiserViewState::Create(id, c,
                {m_PassLayout, m_ReprojectLayout, m_MomentsLayout, m_AtrousLayout});
            state->input = input; state->sources = sources; state->sourceViews = views;
            ViewResources bridge; bridge.reflection = input; bridge.reflectionDenoiser = state;
            WriteNativeView(bridge, targets);
            return state;
        }, [] { Renderer::WaitForGPU(); });
    }
    void SvgfDenoiser::ReleaseReflectionView(RenderViewId id)
    {
        m_ReflectionViews.Release(id, [] { Renderer::WaitForGPU(); });
    }

    const SvgfSettings& SvgfDenoiser::Settings() const
    {
        auto& sys = m_Pipeline->GetSystem();
        if (m_Channel == DenoiserChannel::Reflections) return sys.GetSvgfSpecSettings();
        if (m_Channel == DenoiserChannel::DiSpecular)  return sys.GetSvgfDiSpecSettings();
        return m_Channel == DenoiserChannel::Gi ? sys.GetSvgfGiSettings() : sys.GetSvgfSettings();
    }


    bool SvgfDenoiser::IsEnabled() const
    {
        return m_Pipeline && Settings().enabled;
    }

    void SvgfDenoiser::Init(RenderPipeline& pipeline)
    {
        LH_PROFILE_FUNCTION();
        m_Pipeline = &pipeline;
        VkDevice device = VulkanContext::Get().GetDevice();

        // Linear clamp-to-edge: matches the ReSTIR/RT pass samplers for SceneDepth/SlimNormal.
        VkSamplerCreateInfo sampCI{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sampCI.magFilter    = VK_FILTER_LINEAR;
        sampCI.minFilter    = VK_FILTER_LINEAR;
        sampCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vkCreateSampler(device, &sampCI, nullptr, &m_Sampler);

        // Passthrough set (pass-local): b0 demodulated-DI sampler, b1 output storage image.
        {
            VkDescriptorSetLayoutBinding b[2]{};
            b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            ci.bindingCount = 2; ci.pBindings = b;
            vkCreateDescriptorSetLayout(device, &ci, nullptr, &m_PassLayout);
        }

        // Reproject set (pass-local): b0-b3 current-frame samplers (DI, depth, normal, motion);
        // b4-b6 prev history storage (color+variance, moments+histLen, geom); b7-b9 curr history
        // storage; b10 slim matID sampler (motion variant's material history gate; the spec variant's
        // shader leaves it undeclared). History stays GENERAL (storage), so no UAB / per-frame rewrite;
        // the two sets are pre-built per parity and bound by frameAbs & 1. The denoised output moved to
        // the a-trous final level, so the reproject no longer binds it.
        {
            VkDescriptorSetLayoutBinding b[11]{};
            for (u32 i = 0; i < 11; ++i)
            {
                b[i].binding         = i;
                b[i].descriptorCount = 1;
                b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
                b[i].descriptorType  = (i < 4 || i == 10) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            }
            VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            ci.bindingCount = 11; ci.pBindings = b;
            vkCreateDescriptorSetLayout(device, &ci, nullptr, &m_ReprojectLayout);
        }

        // Moments set (pass-local): b0 colorHist[curr] storage, b1 moments[curr] storage, b2 depth
        // sampler, b3 normal sampler, b4 svgfAtrous[0] storage (a-trous level-0 input).
        {
            VkDescriptorSetLayoutBinding b[5]{};
            const VkDescriptorType types[5] = {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            };
            for (u32 i = 0; i < 5; ++i)
            {
                b[i].binding = i; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                b[i].descriptorType = types[i];
            }
            VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            ci.bindingCount = 5; ci.pBindings = b;
            vkCreateDescriptorSetLayout(device, &ci, nullptr, &m_MomentsLayout);
        }

        // A-trous set (pass-local): b0 svgfAtrous[IN] storage, b1 depth sampler, b2 normal sampler,
        // b3 svgfAtrous[OUT] storage, b4 svgfDenoised storage (final level), b5 slim roughness sampler
        // (spec channels' edge-stop; diffuse channels disable via phiRough 0). Two sets by iter parity.
        {
            VkDescriptorSetLayoutBinding b[6]{};
            const VkDescriptorType types[6] = {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            };
            for (u32 i = 0; i < 6; ++i)
            {
                b[i].binding = i; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                b[i].descriptorType = types[i];
            }
            VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            ci.bindingCount = 6; ci.pBindings = b;
            vkCreateDescriptorSetLayout(device, &ci, nullptr, &m_AtrousLayout);
        }

        // Reflections denoises a specular signal -> a SPECULAR reproject variant (hit-distance virtual
        // reprojection); same layout/pcRange as the diffuse reproject. Moments/a-trous/passthrough shared.
        const char* reprojShader = (m_Channel == DenoiserChannel::Reflections)
            ? "shaders/svgf_spec_reproject.slang" : "shaders/svgf_reproject.slang";
        if (auto sh = ShaderLibrary::LoadEngine("shaders/svgf_passthrough.slang"))
            m_PassthroughSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine(reprojShader))
            m_ReprojectSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/svgf_moments.slang"))
            m_MomentsSpv = sh->GetSpirV();
        if (auto sh = ShaderLibrary::LoadEngine("shaders/svgf_atrous.slang"))
            m_AtrousSpv = sh->GetSpirV();
        if (m_PassthroughSpv.empty() || m_ReprojectSpv.empty() || m_MomentsSpv.empty() || m_AtrousSpv.empty())
        {
            LH_LOG(Renderer, error, "SvgfDenoiser: failed to load svgf_passthrough/reproject/moments/atrous.comp SPIR-V");
            return;
        }

        const std::vector<VkDescriptorSetLayout> passLayouts = { m_PassLayout };
        m_PassthroughPipeline = std::make_unique<VKComputePipeline>(
            m_PassthroughSpv, passLayouts, std::vector<VkPushConstantRange>{});

        VkDescriptorSetLayout globalLayout = m_Pipeline->GetGlobal().GetSetLayout();

        // Reproject binds Set 0 (global UBO: nearZ/farZ/viewportSize) + the pass-local set.
        const std::vector<VkDescriptorSetLayout> reprojLayouts = { globalLayout, m_ReprojectLayout };
        VkPushConstantRange reprojPc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SvgfReprojectPC) };
        m_ReprojectPipeline = std::make_unique<VKComputePipeline>(
            m_ReprojectSpv, reprojLayouts, std::vector<VkPushConstantRange>{ reprojPc });

        const std::vector<VkDescriptorSetLayout> momentsLayouts = { globalLayout, m_MomentsLayout };
        VkPushConstantRange momentsPc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SvgfMomentsPC) };
        m_MomentsPipeline = std::make_unique<VKComputePipeline>(
            m_MomentsSpv, momentsLayouts, std::vector<VkPushConstantRange>{ momentsPc });

        const std::vector<VkDescriptorSetLayout> atrousLayouts = { globalLayout, m_AtrousLayout };
        VkPushConstantRange atrousPc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SvgfAtrousPC) };
        m_AtrousPipeline = std::make_unique<VKComputePipeline>(
            m_AtrousSpv, atrousLayouts, std::vector<VkPushConstantRange>{ atrousPc });
    }

    void SvgfDenoiser::Shutdown()
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();
        m_DiViews.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_GiViews.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_ReflectionViews.ReleaseAll([] { Renderer::WaitForGPU(); });
        m_PassthroughPipeline.reset();
        m_ReprojectPipeline.reset();
        m_MomentsPipeline.reset();
        m_AtrousPipeline.reset();
        if (m_Sampler)         vkDestroySampler(device, m_Sampler, nullptr);
        if (m_PassLayout)      vkDestroyDescriptorSetLayout(device, m_PassLayout, nullptr);
        if (m_ReprojectLayout) vkDestroyDescriptorSetLayout(device, m_ReprojectLayout, nullptr);
        if (m_MomentsLayout)   vkDestroyDescriptorSetLayout(device, m_MomentsLayout, nullptr);
        if (m_AtrousLayout)    vkDestroyDescriptorSetLayout(device, m_AtrousLayout, nullptr);
        m_Sampler        = VK_NULL_HANDLE;
        m_PassLayout      = VK_NULL_HANDLE;
        m_ReprojectLayout = VK_NULL_HANDLE;
        m_MomentsLayout   = VK_NULL_HANDLE;
        m_AtrousLayout    = VK_NULL_HANDLE;
        m_PassthroughSpv.clear();
        m_ReprojectSpv.clear();
        m_MomentsSpv.clear();
        m_AtrousSpv.clear();
        m_Pipeline = nullptr;
    }

    bool SvgfDenoiser::OnShaderReloaded(const std::string& name, const std::vector<u32>& spv)
    {
        LH_PROFILE_FUNCTION();
        if (!m_Pipeline) return false;

        // Defer the old pipeline's destruction; an in-flight frame may still bind it. PushDeletion
        // drains it MAX_FRAMES_IN_FLIGHT frames later (no vkDeviceWaitIdle).
        if (name == "svgf_passthrough.slang" && m_PassLayout != VK_NULL_HANDLE)
        {
            if (m_PassthroughPipeline)
                VulkanContext::Get().PushDeletion([p = m_PassthroughPipeline.release()]() { delete p; });
            m_PassthroughSpv = spv;
            const std::vector<VkDescriptorSetLayout> layouts = { m_PassLayout };
            m_PassthroughPipeline = std::make_unique<VKComputePipeline>(
                m_PassthroughSpv, layouts, std::vector<VkPushConstantRange>{});
            m_DiViews.ForEach([](auto& state) { state->history.Invalidate(); });
            m_GiViews.ForEach([](auto& state) { state->history.Invalidate(); });
            return true;
        }
        const char* myReproj = (m_Channel == DenoiserChannel::Reflections)
            ? "svgf_spec_reproject.slang" : "svgf_reproject.slang";
        if (name == myReproj && m_ReprojectLayout != VK_NULL_HANDLE)
        {
            if (m_ReprojectPipeline)
                VulkanContext::Get().PushDeletion([p = m_ReprojectPipeline.release()]() { delete p; });
            m_ReprojectSpv = spv;
            const std::vector<VkDescriptorSetLayout> layouts = {
                m_Pipeline->GetGlobal().GetSetLayout(), m_ReprojectLayout };
            VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SvgfReprojectPC) };
            m_ReprojectPipeline = std::make_unique<VKComputePipeline>(
                m_ReprojectSpv, layouts, std::vector<VkPushConstantRange>{ pcRange });
            m_DiViews.ForEach([](auto& state) { state->history.Invalidate(); });
            m_GiViews.ForEach([](auto& state) { state->history.Invalidate(); });
            return true;
        }
        if (name == "svgf_moments.slang" && m_MomentsLayout != VK_NULL_HANDLE)
        {
            if (m_MomentsPipeline)
                VulkanContext::Get().PushDeletion([p = m_MomentsPipeline.release()]() { delete p; });
            m_MomentsSpv = spv;
            const std::vector<VkDescriptorSetLayout> layouts = {
                m_Pipeline->GetGlobal().GetSetLayout(), m_MomentsLayout };
            VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SvgfMomentsPC) };
            m_MomentsPipeline = std::make_unique<VKComputePipeline>(
                m_MomentsSpv, layouts, std::vector<VkPushConstantRange>{ pcRange });
            m_DiViews.ForEach([](auto& state) { state->history.Invalidate(); });
            m_GiViews.ForEach([](auto& state) { state->history.Invalidate(); });
            return true;
        }
        if (name == "svgf_atrous.slang" && m_AtrousLayout != VK_NULL_HANDLE)
        {
            if (m_AtrousPipeline)
                VulkanContext::Get().PushDeletion([p = m_AtrousPipeline.release()]() { delete p; });
            m_AtrousSpv = spv;
            const std::vector<VkDescriptorSetLayout> layouts = {
                m_Pipeline->GetGlobal().GetSetLayout(), m_AtrousLayout };
            VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SvgfAtrousPC) };
            m_AtrousPipeline = std::make_unique<VKComputePipeline>(
                m_AtrousSpv, layouts, std::vector<VkPushConstantRange>{ pcRange });
            m_DiViews.ForEach([](auto& state) { state->history.Invalidate(); });
            m_GiViews.ForEach([](auto& state) { state->history.Invalidate(); });
            return true;
        }
        return false;
    }

    void SvgfDenoiser::AllocateViewSets(ViewResources&)
    {
        // All channels allocate descriptors in their domain-owned state stores.
    }
    void SvgfDenoiser::WriteView(ViewResources&, FrameTargets&)
    {
        // Immutable domain bindings are written only during Ensure.
    }
    void SvgfDenoiser::WriteNativeView(ViewResources& vr, FrameTargets& targets)
    {
        LH_PROFILE_FUNCTION();
        VkDevice device = VulkanContext::Get().GetDevice();
        ChannelRefs c = Resolve(m_Channel, vr);
        auto viewOf = [](const std::shared_ptr<Texture>& t) {
            return std::static_pointer_cast<VKTexture>(t)->GetImageView();
        };

        // Passthrough set: b0 noisy input sampler, b1 denoised storage.
        if (*c.passthroughSet != VK_NULL_HANDLE && c.noisy && *c.noisy && *c.denoised)
        {
            VkDescriptorImageInfo diIn{ m_Sampler, viewOf(*c.noisy), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo outImg{ VK_NULL_HANDLE, viewOf(*c.denoised), VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet w[2]{};
            w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w[0].dstSet = *c.passthroughSet; w[0].dstBinding = 0;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].descriptorCount = 1; w[0].pImageInfo = &diIn;
            w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w[1].dstSet = *c.passthroughSet; w[1].dstBinding = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].descriptorCount = 1; w[1].pImageInfo = &outImg;
            vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
        }

        // Reproject sets: pre-build both parities (set[p] reads prev = [p^1], writes curr = [p]).
        // Reproject b3: DI/GI bind slim MOTION; the specular variant binds slim ROUGHNESS (it computes
        // the reflection's own motion internally via hit-distance virtual reprojection). b10 slim matID
        // is written for every channel (layout parity); only the motion variant's shader reads it.
        const std::shared_ptr<Texture> b3Tex = (m_Channel == DenoiserChannel::Reflections)
            ? targets.GetSlimRoughness() : targets.GetSlimMotion();
        if (c.reprojectSet[0] != VK_NULL_HANDLE && c.noisy && *c.noisy
            && c.colorHist[0] && c.moments[0] && c.geom[0]
            && targets.GetSceneDepth() && targets.GetSlimNormal() && b3Tex
            && targets.GetSlimMaterialID())
        {
            const VkImageView depthV  = viewOf(targets.GetSceneDepth());
            const VkImageView normalV = viewOf(targets.GetSlimNormal());
            const VkImageView motionV = viewOf(b3Tex);
            const VkImageView diV     = viewOf(*c.noisy);
            const VkImageView matIdV  = viewOf(targets.GetSlimMaterialID());

            for (u32 p = 0; p < 2; ++p)
            {
                const u32 q = p ^ 1u;  // prev parity
                VkDescriptorImageInfo info[11]{};
                info[0]  = { m_Sampler, diV,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                info[1]  = { m_Sampler, depthV,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                info[2]  = { m_Sampler, normalV, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                info[3]  = { m_Sampler, motionV, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                info[4]  = { VK_NULL_HANDLE, viewOf(c.colorHist[q]), VK_IMAGE_LAYOUT_GENERAL };
                info[5]  = { VK_NULL_HANDLE, viewOf(c.moments[q]),   VK_IMAGE_LAYOUT_GENERAL };
                info[6]  = { VK_NULL_HANDLE, viewOf(c.geom[q]),      VK_IMAGE_LAYOUT_GENERAL };
                info[7]  = { VK_NULL_HANDLE, viewOf(c.colorHist[p]), VK_IMAGE_LAYOUT_GENERAL };
                info[8]  = { VK_NULL_HANDLE, viewOf(c.moments[p]),   VK_IMAGE_LAYOUT_GENERAL };
                info[9]  = { VK_NULL_HANDLE, viewOf(c.geom[p]),      VK_IMAGE_LAYOUT_GENERAL };
                info[10] = { m_Sampler, matIdV,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

                VkWriteDescriptorSet w[11]{};
                for (u32 i = 0; i < 11; ++i)
                {
                    w[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    w[i].dstSet          = c.reprojectSet[p];
                    w[i].dstBinding      = i;
                    w[i].descriptorType  = (i < 4 || i == 10) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    w[i].descriptorCount = 1;
                    w[i].pImageInfo      = &info[i];
                }
                vkUpdateDescriptorSets(device, 11, w, 0, nullptr);
            }
        }

        // Moments sets: per parity p, b0/b1 = colorHist[p]/moments[p] (the reproject's curr output),
        // b2/b3 depth/normal samplers, b4 svgfAtrous[0] (a-trous level-0 input; shared, not ping-ponged).
        if (c.momentsSet[0] != VK_NULL_HANDLE
            && c.colorHist[0] && c.moments[0] && c.atrous[0]
            && targets.GetSceneDepth() && targets.GetSlimNormal())
        {
            const VkImageView depthV  = viewOf(targets.GetSceneDepth());
            const VkImageView normalV = viewOf(targets.GetSlimNormal());
            const VkImageView a0V     = viewOf(c.atrous[0]);

            for (u32 p = 0; p < 2; ++p)
            {
                VkDescriptorImageInfo info[5]{};
                info[0] = { VK_NULL_HANDLE, viewOf(c.colorHist[p]), VK_IMAGE_LAYOUT_GENERAL };
                info[1] = { VK_NULL_HANDLE, viewOf(c.moments[p]),   VK_IMAGE_LAYOUT_GENERAL };
                info[2] = { m_Sampler, depthV,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                info[3] = { m_Sampler, normalV, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                info[4] = { VK_NULL_HANDLE, a0V, VK_IMAGE_LAYOUT_GENERAL };

                const VkDescriptorType types[5] = {
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                };
                VkWriteDescriptorSet w[5]{};
                for (u32 i = 0; i < 5; ++i)
                {
                    w[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    w[i].dstSet = c.momentsSet[p]; w[i].dstBinding = i;
                    w[i].descriptorType = types[i]; w[i].descriptorCount = 1; w[i].pImageInfo = &info[i];
                }
                vkUpdateDescriptorSets(device, 5, w, 0, nullptr);
            }
        }

        // A-trous sets: per iter parity ip, b0 = svgfAtrous[ip] (in), b3 = svgfAtrous[ip^1] (out),
        // b1/b2 depth/normal samplers, b4 svgfDenoised (final-level output), b5 slim roughness (spec
        // channels' edge-stop; bound for every channel, diffuse ones disable via phiRough 0).
        if (c.atrousSet[0] != VK_NULL_HANDLE
            && c.atrous[0] && c.atrous[1] && *c.denoised
            && targets.GetSceneDepth() && targets.GetSlimNormal() && targets.GetSlimRoughness())
        {
            const VkImageView depthV  = viewOf(targets.GetSceneDepth());
            const VkImageView normalV = viewOf(targets.GetSlimNormal());
            const VkImageView denV    = viewOf(*c.denoised);
            const VkImageView roughV  = viewOf(targets.GetSlimRoughness());

            for (u32 ip = 0; ip < 2; ++ip)
            {
                const u32 op = ip ^ 1u;
                VkDescriptorImageInfo info[6]{};
                info[0] = { VK_NULL_HANDLE, viewOf(c.atrous[ip]), VK_IMAGE_LAYOUT_GENERAL };
                info[1] = { m_Sampler, depthV,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                info[2] = { m_Sampler, normalV, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                info[3] = { VK_NULL_HANDLE, viewOf(c.atrous[op]), VK_IMAGE_LAYOUT_GENERAL };
                info[4] = { VK_NULL_HANDLE, denV, VK_IMAGE_LAYOUT_GENERAL };
                info[5] = { m_Sampler, roughV,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

                const VkDescriptorType types[6] = {
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                };
                VkWriteDescriptorSet w[6]{};
                for (u32 i = 0; i < 6; ++i)
                {
                    w[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    w[i].dstSet = c.atrousSet[ip]; w[i].dstBinding = i;
                    w[i].descriptorType = types[i]; w[i].descriptorCount = 1; w[i].pImageInfo = &info[i];
                }
                vkUpdateDescriptorSets(device, 6, w, 0, nullptr);
            }
        }
    }

    RG::ResourceHandle SvgfDenoiser::AddPasses(RG::RenderGraph&, const DenoiseInputs&)
    {
        // All channels record through typed feature adapters and frozen native packets.
        return {};
    }
}
