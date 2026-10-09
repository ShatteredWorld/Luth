#include "luthpch.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/subsystems/GlobalSubsystem.h"
#include "luth/renderer/subsystems/GTAOSubsystem.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/renderer/subsystems/EditorOverlaysSubsystem.h"
#include "luth/renderer/subsystems/VolumetricSubsystem.h"
#include "luth/renderer/subsystems/RtSubsystem.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include "luth/scene/systems/RenderingSystem.h"
#include "luth/renderer/backend/vulkan/VulkanContext.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
#include "luth/renderer/backend/vulkan/VulkanBuffer.h"
#include "luth/renderer/resources/Texture.h"
#include "luth/renderer/settings/GTAOSettings.h"
#include "luth/renderer/settings/PostProcessSettings.h"
#include "luth/renderer/settings/VolumetricSettings.h"
#include "luth/memory/GPUTaggedPageAllocator.h"
#include "luth/core/diagnostics/Log.h"

namespace Luth
{
    // Per-view pool: cycled sets allocate MAX_FRAMES_IN_FLIGHT instances each. Capacity bumped on
    // every subsystem addition; silent vkAllocateDescriptorSets failure on overflow returns
    // VK_NULL_HANDLE handles and skips the draw with no log. Bump generously; pool memory is cheap.

    namespace {
        // Build the per-view Set 0 write context from RP-side state.
        // GTAO state is prepared independently before writing the compatibility global set.
        GlobalViewWriteContext MakeGlobalCtx(const RenderPipeline& rp, const ViewResources& vr)
        {
            const auto& lighting = rp.GetLighting();
            GlobalViewWriteContext ctx{};
            ctx.haveIBL          = lighting.IsIBLReady();
            ctx.iblSampler       = lighting.GetIBLSampler();
            ctx.gtaoSampler      = rp.GetGTAO().GetSampler();
            if (ctx.haveIBL)
            {
                ctx.irradianceView  = std::static_pointer_cast<VKTexture>(lighting.GetIrradianceMap())->GetImageView();
                ctx.prefilteredView = std::static_pointer_cast<VKTexture>(lighting.GetPrefilteredMap())->GetImageView();
                ctx.brdfView        = std::static_pointer_cast<VKTexture>(lighting.GetBRDFLut())->GetImageView();
            }
            const auto* gtao = rp.GetGtaoViewState({vr.id});
            ctx.gtaoFinalView = gtao ? gtao->finalBinding.view : VK_NULL_HANDLE;
            return ctx;
        }
    }

    ViewResources& RenderPipeline::EnsureViewResources(FrameTargets& targets)
    {
        const RenderViewId id = m_System.GetViews().Find(&targets);
        if (!id.value) throw std::invalid_argument("Unregistered render view targets");
        auto [it, inserted] = m_ViewResources.try_emplace(id.value);
        ViewResources& vr = it->second;

        if (!targets.GetSceneColor())
            return vr;

        const u32 newW = targets.GetSceneColor()->GetWidth();
        const u32 newH = targets.GetSceneColor()->GetHeight();
        m_GTAO.EnsureView(m_GtaoStates, id, newW, newH, *targets.GetSceneDepth());
        auto fog = m_Volumetric.EnsureView(id, targets, m_System.GetVolumetricSettings().quality);
        const bool fogReplaced = vr.fog && vr.fog != fog;
        vr.fog = std::move(fog);
        if (fogReplaced) vr.generation = m_System.InvalidateView(id);
        auto transparency = m_Transparency.EnsureView(id, newW, newH, m_System.GetTransparencySettings().avgLayersBudget);
        const bool transparencyReplaced = vr.transparency && vr.transparency != transparency;
        vr.transparency = std::move(transparency);
        if (transparencyReplaced) vr.generation = m_System.InvalidateView(id);

        auto taa = m_PostProcess.EnsureTaaView(id, targets);
        const bool taaReplaced = vr.taa && vr.taa != taa;
        vr.taa = std::move(taa);
        if (taaReplaced) vr.generation = m_System.InvalidateView(id);

        auto bloom = m_PostProcess.EnsureBloomView(id, newW, newH);
        const bool bloomReplaced = vr.bloom && vr.bloom != bloom;
        vr.bloom = std::move(bloom);
        if (bloomReplaced) vr.generation = m_System.InvalidateView(id);
        auto composite = m_PostProcess.EnsureCompositeView(id, targets, vr.bloom);
        const bool compositeReplaced = vr.composite && vr.composite != composite;
        vr.composite = std::move(composite);
        if (compositeReplaced) vr.generation = m_System.InvalidateView(id);
        auto slimViz = m_PostProcess.EnsureSlimVizView(id, targets);
        const bool slimVizReplaced = vr.slimViz && vr.slimViz != slimViz;
        vr.slimViz = std::move(slimViz);
        if (slimVizReplaced) vr.generation = m_System.InvalidateView(id);
        auto clusterViz = m_Lighting.EnsureClusterVizView(id, targets);
        const bool clusterVizReplaced = vr.clusterViz && vr.clusterViz != clusterViz;
        vr.clusterViz = std::move(clusterViz);
        if (clusterVizReplaced) vr.generation = m_System.InvalidateView(id);
        auto overlays = m_EditorOverlays.EnsureView(id, targets);
        const bool overlaysReplaced = vr.overlays && vr.overlays != overlays;
        vr.overlays = std::move(overlays);
        if (overlaysReplaced) vr.generation = m_System.InvalidateView(id);

        auto shadow = m_RtNativeInitialized ? m_Rt.EnsureShadowView(id, targets) : nullptr;
        const bool shadowReplaced = vr.rtShadow && vr.rtShadow != shadow;
        vr.rtShadow = std::move(shadow);
        if (shadowReplaced) vr.generation = m_System.InvalidateView(id);

        auto restirDi = m_RtNativeInitialized ? m_Restir.EnsureView(id, targets,
            m_System.GetRestirSettings().halfResolution) : nullptr;
        const bool restirDiReplaced = vr.restirDi && vr.restirDi != restirDi;
        vr.restirDi = std::move(restirDi);
        if (restirDiReplaced) vr.generation = m_System.InvalidateView(id);

        auto* diffuseDenoiser = static_cast<SvgfDenoiser*>(m_Denoise.get());
        auto diDenoiser = m_RtNativeInitialized ? diffuseDenoiser->EnsureDiView(id, targets, vr.restirDi) : nullptr;
        const bool diDenoiserReplaced = vr.diDenoiser && vr.diDenoiser != diDenoiser;
        vr.diDenoiser = std::move(diDenoiser);
        if (diDenoiserReplaced) vr.generation = m_System.InvalidateView(id);

        if (inserted || vr.descPool == VK_NULL_HANDLE)
        {
            // Borrow the owner's identity; native allocation does not mint a new view.
            vr.id = id.value;
            vr.generation = m_System.GetViews().Get(id)->generation;
            AllocateViewResources(vr, targets);
        }
        else if (vr.width != newW || vr.height != newH ||
                 (m_RtNativeInitialized && (vr.giHalfCached != (m_System.GetRestirGiSettings().halfResolution ? 1u : 0u) ||
                   vr.diHalfCached != (m_System.GetRestirSettings().halfResolution ? 1u : 0u) ||
                   vr.reflHalfCached != (m_System.GetReflectionsSettings().halfResolution ? 1u : 0u))))
        {
            // Stable descriptor slots may still be referenced by earlier submissions.
            Renderer::WaitForGPU();
            vr.generation = m_System.InvalidateView(id);
            const u32 halfW = std::max(newW / 2, 1u);
            const u32 halfH = std::max(newH / 2, 1u);
            RecreateViewTextures(vr, newW, newH, halfW, halfH);


            if (m_RtNativeInitialized)
            {
                m_RestirGi.WriteView(vr, targets);      // re-bind GI Set 2 depth/normal + reservoir + new GI image
                m_RestirGi.WriteReservoirVizView(vr, targets);  // re-bind GI reservoir-viz depth + spatial reservoir
                m_RestirGi.WriteUpscaleView(vr, targets);       // re-bind GI upscale half-input + full output
                m_Restir.WriteUpscaleView(vr, targets);         // re-bind DI diffuse + specular upscale sets
                m_PathTrace.WriteView(vr);              // re-bind PT accumulator + display image (recreated on resize)
                m_Reflections.WriteView(vr, targets);   // re-bind reflection output + slim G-buffer samplers
                m_Reflections.WriteUpscaleView(vr, targets);    // re-bind refl upscale half-input + full output
                m_DenoiseGi->WriteView(vr, targets);    // re-bind GI SVGF inputs + output to the new images
                m_DenoiseRefl->WriteView(vr, targets);  // re-bind specular SVGF inputs + output to the new images
                m_DenoiseDiSpec->WriteView(vr, targets);// re-bind ReSTIR-DI specular SVGF
            }
            m_Lighting.WriteShadowView(vr);         // re-bind Set 3 b4 sun mask + b5 denoised DI + b6 denoised GI
            // Set 0 bindings 1-4 reference the (re)created IBL + GTAO textures.
            m_Global.WriteView(vr, MakeGlobalCtx(*this, vr));
        }

        vr.width  = newW;
        vr.height = newH;
        // Source views can change without an extent change. Refresh the consumer's mask binding.
        if (shadowReplaced || diDenoiserReplaced) m_Lighting.WriteShadowView(vr);
        if (diDenoiserReplaced && m_RtNativeInitialized) m_Restir.WriteUpscaleView(vr, targets);
        if (restirDiReplaced && m_RtNativeInitialized) {
            m_DenoiseDiSpec->WriteView(vr, targets);
        }
        return vr;
    }

    void RenderPipeline::ReleaseViewResources(FrameTargets& targets)
    {
        const auto id = m_System.GetViews().Find(&targets);
        if (m_GtaoPipeline) m_GtaoPipeline->ReleaseView(id);
        m_Volumetric.ReleaseView(id);
        m_Rt.ReleaseShadowView(id);
        static_cast<SvgfDenoiser*>(m_Denoise.get())->ReleaseDiView(id);
        m_Restir.ReleaseView(id);
        m_Transparency.ReleaseView(id);
        m_PostProcess.ReleaseTaaView(id);
        m_EditorOverlays.ReleaseView(id);
        m_PostProcess.ReleaseSlimVizView(id);
        m_Lighting.ReleaseClusterVizView(id);
        m_PostProcess.ReleaseCompositeView(id);
        m_PostProcess.ReleaseBloomView(id);
        auto it = m_ViewResources.find(id.value);
        if (it == m_ViewResources.end()) return;
        DestroyViewResources(it->second);
        if (m_CurrentViewResources == &it->second) m_CurrentViewResources = nullptr;
        m_ViewResources.erase(it);
    }

    bool RenderPipeline::HasViewResources(FrameTargets* targets, RenderViewId id, u64 generation) const
    {
        if (!targets) return false;
        if (!m_System.GetViews().Matches(id, targets, generation)) return false;
        auto it = m_ViewResources.find(id.value);
        return it != m_ViewResources.end() && it->second.generation == generation;
    }

    ViewResources* RenderPipeline::GetViewResources(FrameTargets* targets)
    {
        if (!targets) return nullptr;
        auto it = m_ViewResources.find(m_System.GetViews().Find(targets).value);
        return it == m_ViewResources.end() ? nullptr : &it->second;
    }

    const ViewResources* RenderPipeline::GetViewResources(FrameTargets* targets) const
    {
        if (!targets) return nullptr;
        auto it = m_ViewResources.find(m_System.GetViews().Find(targets).value);
        return it == m_ViewResources.end() ? nullptr : &it->second;
    }

    void RenderPipeline::AllocateViewResources(ViewResources& vr, FrameTargets& targets)
    {
        VkDevice device = VulkanContext::Get().GetDevice();

        const VulkanViewPool budget(m_RtNativeInitialized, MAX_FRAMES_IN_FLIGHT);
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        poolInfo.maxSets       = budget.maxSets;
        poolInfo.poolSizeCount = budget.count;
        poolInfo.pPoolSizes    = budget.sizes.data();
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &vr.descPool);

        // Set 0 UBO bindings (0 + 5) and Grid set binding 0 are written per render-stage
        // by UpdateGlobalUniforms / UpdateGTAOUBO; nothing to allocate up front.

        const u32 fullW = targets.GetSceneColor()->GetWidth();
        const u32 fullH = targets.GetSceneColor()->GetHeight();
        const u32 halfW = std::max(fullW / 2, 1u);
        const u32 halfH = std::max(fullH / 2, 1u);
        RecreateViewTextures(vr, fullW, fullH, halfW, halfH);

        auto allocSingle = [&](VkDescriptorSetLayout layout, VkDescriptorSet& outSet, const char* tag) {
            if (layout == VK_NULL_HANDLE) return;
            VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            ai.descriptorPool     = vr.descPool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &layout;
            vkAllocateDescriptorSets(device, &ai, &outSet);
            VulkanContext::SetDebugName(outSet, tag);
        };
        auto allocCycled = [&](VkDescriptorSetLayout layout,
                               std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT>& outArr,
                               const char* tagPrefix) {
            if (layout == VK_NULL_HANDLE) return;
            VkDescriptorSetLayout layouts[MAX_FRAMES_IN_FLIGHT];
            for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) layouts[i] = layout;
            VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            ai.descriptorPool     = vr.descPool;
            ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
            ai.pSetLayouts        = layouts;
            VkResult result = vkAllocateDescriptorSets(device, &ai, outArr.data());
            if (result != VK_SUCCESS)
            {
                LH_LOG(Renderer, error, "ViewResources: allocCycled '{}' failed (VkResult {}); bump pool sizes", tagPrefix, (int)result);
                for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) outArr[i] = VK_NULL_HANDLE;
                return;
            }
            for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            {
                char name[64]; std::snprintf(name, sizeof(name), "%s.Slot%u", tagPrefix, i);
                VulkanContext::SetDebugName(outArr[i], name);
            }
        };


        allocCycled(m_Global.GetSetLayout(),             vr.globalDescriptorSet,   "View.Global");



        allocCycled(m_Lighting.GetSetLayout(),           vr.lightDescSet,         "View.Light");
        allocCycled(m_Lighting.GetClusterBuildLayout(),  vr.clusterBuildDescSet,  "View.ClusterBuild");
        allocCycled(m_Lighting.GetLightAssignLayout(),   vr.lightAssignDescSet,   "View.LightAssign");
        if (m_RtNativeInitialized)
        {
            allocCycled(m_RestirGi.GetSetLayout(),           vr.restirGiDescSet,      "View.RestirGi");
            allocSingle(m_RestirGi.GetReservoirVizLayout(),  vr.giReservoirVizDescSet,"View.GiReservoirViz");
            allocSingle(m_RestirGi.GetUpscaleLayout(),       vr.giUpscaleDescSet,     "View.GiUpscale");
            allocSingle(m_Restir.GetUpscaleLayout(),         vr.diUpscaleDescSet,     "View.DiUpscale");
            allocSingle(m_Restir.GetUpscaleLayout(),         vr.diSpecUpscaleDescSet, "View.DiSpecUpscale");
            allocSingle(m_PathTrace.GetSetLayout(),          vr.ptDescSet,            "View.PathTrace");
            allocSingle(m_Reflections.GetSetLayout(),        vr.reflDescSet,          "View.Reflections");
            allocSingle(m_Reflections.GetUpscaleLayout(),    vr.reflUpscaleDescSet,   "View.ReflUpscale");
            m_DenoiseGi->AllocateViewSets(vr);
            m_DenoiseRefl->AllocateViewSets(vr);
            m_DenoiseDiSpec->AllocateViewSets(vr);
        }
        m_Lighting.WriteShadowView(vr);
        if (m_RtNativeInitialized)
        {
            m_RestirGi.WriteView(vr, targets);
            m_RestirGi.WriteReservoirVizView(vr, targets);
            m_RestirGi.WriteUpscaleView(vr, targets);
            m_Restir.WriteUpscaleView(vr, targets);
            m_PathTrace.WriteView(vr);
            m_Reflections.WriteView(vr, targets);
            m_Reflections.WriteUpscaleView(vr, targets);    // bind refl upscale half-input + full output
            m_DenoiseGi->WriteView(vr, targets);
            m_DenoiseRefl->WriteView(vr, targets);
            m_DenoiseDiSpec->WriteView(vr, targets);
        }
        // Global writes borrow the final AO binding from the independent GTAO state.
        m_Global.WriteView(vr, MakeGlobalCtx(*this, vr));
    }

    void RenderPipeline::RecreateViewTextures(ViewResources& vr, u32 fullW, u32 fullH, u32 halfW, u32 halfH)
    {
        // This remaining compatibility allocation group is entirely RT-owned.
        if (!m_RtNativeInitialized) return;

        // Half-res GI (RestirGiSettings::halfResolution): GI reservoirs + restirGiDI + svgfGi* history
        // allocate at half extent; svgfGiDenoised stays full (the bilateral-upscale output). giHalfCached
        // lets EnsureViewResources detect a runtime toggle and realloc (like other allocation settings).
        const bool giHalf = m_System.GetRestirGiSettings().halfResolution;
        const u32  giW    = giHalf ? halfW : fullW;
        const u32  giH    = giHalf ? halfH : fullH;
        vr.giHalfCached   = giHalf ? 1u : 0u;

        // Half-res DI (RestirSettings::halfResolution): both DI channels (diffuse + specular) trace +
        // denoise at half; svgfDenoised + svgfDiSpecDenoised stay full (the bilateral-upscale outputs).
        const bool diHalf = m_System.GetRestirSettings().halfResolution;
        const u32  diW    = diHalf ? halfW : fullW;
        const u32  diH    = diHalf ? halfH : fullH;
        vr.diHalfCached   = diHalf ? 1u : 0u;

        // Half-res reflections (ReflectionsSettings::halfResolution): reflRadiance trace output + svgfSpec*
        // history allocate at half; svgfSpecDenoised stays full (the bilateral-upscale output).
        const bool reflHalf = m_System.GetReflectionsSettings().halfResolution;
        const u32  reflW    = reflHalf ? halfW : fullW;
        const u32  reflH    = reflHalf ? halfH : fullH;
        vr.reflHalfCached   = reflHalf ? 1u : 0u;

        // ReSTIR GI demodulated indirect-diffuse image: GI working res (half when halfResolution).
        // STORAGE for the GI shade pass's imageStore + SAMPLED (ctor) for the denoiser input.
        vr.restirGiDI = std::make_shared<VKTexture>(
            giW, giH, TextureFormat::RGBA16F,
            /*arrayLayers*/ 1, /*createFlags*/ 0u, /*mipLevels*/ 1,
            VK_IMAGE_USAGE_STORAGE_BIT);

        // Path-traced reference mode. ptAccum = viewport-sized RGBA32F STORAGE: the in-place fp32
        // progressive running mean, kept GENERAL, only ever touched by the PT megakernel (read-before-write
        // cross-frame, so bootstrap-cleared below). ptColor = RGBA16F STORAGE+SAMPLED display copy the post
        // chain samples; fully written each frame, so no clear (svgfDenoised pattern).
        vr.ptAccum = std::make_shared<VKTexture>(
            fullW, fullH, TextureFormat::RGBA32F,
            /*arrayLayers*/ 1, /*createFlags*/ 0u, /*mipLevels*/ 1,
            VK_IMAGE_USAGE_STORAGE_BIT);
        vr.ptColor = std::make_shared<VKTexture>(
            fullW, fullH, TextureFormat::RGBA16F,
            /*arrayLayers*/ 1, /*createFlags*/ 0u, /*mipLevels*/ 1,
            VK_IMAGE_USAGE_STORAGE_BIT);

        // RT specular reflections: reflection working res (half when halfResolution). STORAGE for the
        // trace's imageStore + SAMPLED (ctor) for pbr.frag's Set 3 b7 read. rgb = demodulated specular
        // radiance, a = hitDist. Fully written each frame (reflection or env fallback), so no bootstrap clear.
        vr.reflRadiance = std::make_shared<VKTexture>(
            reflW, reflH, TextureFormat::RGBA16F,
            /*arrayLayers*/ 1, /*createFlags*/ 0u, /*mipLevels*/ 1,
            VK_IMAGE_USAGE_STORAGE_BIT);

        // GI SVGF: flat parallel set to the DI history above. History + a-trous run at the GI working res
        // (half when halfResolution); svgfGiDenoised stays FULL (the bilateral-upscale output). svgfGiHalf
        // is the half a-trous final the upscale reads; written each frame, so no clear.
        vr.svgfGiDenoised = std::make_shared<VKTexture>(fullW, fullH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        vr.svgfGiHalf     = std::make_shared<VKTexture>(giW, giH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        for (u32 i = 0; i < 2; ++i)
        {
            vr.svgfGiColorHist[i] = std::make_shared<VKTexture>(giW, giH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
            vr.svgfGiMoments[i]   = std::make_shared<VKTexture>(giW, giH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
            vr.svgfGiGeom[i]      = std::make_shared<VKTexture>(giW, giH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        }
        vr.svgfGiAtrous[0] = std::make_shared<VKTexture>(giW, giH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        vr.svgfGiAtrous[1] = std::make_shared<VKTexture>(giW, giH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);

        // Specular (RT-reflection) SVGF history: flat parallel to the GI SVGF. History + a-trous run
        // at the reflection working res (half when halfResolution); svgfSpecDenoised stays FULL (the
        // bilateral-upscale output). svgfSpecGeom carries hitDist in its .a (vs GI's unused .a) for
        // reflected-depth disocclusion. svgfSpecHalf is the half a-trous final the upscale reads; written
        // each frame, so no clear. Cross-frame history below is bootstrap-cleared (at reflW/reflH).
        vr.svgfSpecDenoised = std::make_shared<VKTexture>(fullW, fullH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        vr.svgfSpecHalf     = std::make_shared<VKTexture>(reflW, reflH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        for (u32 i = 0; i < 2; ++i)
        {
            vr.svgfSpecColorHist[i] = std::make_shared<VKTexture>(reflW, reflH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
            vr.svgfSpecMoments[i]   = std::make_shared<VKTexture>(reflW, reflH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
            vr.svgfSpecGeom[i]      = std::make_shared<VKTexture>(reflW, reflH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        }
        vr.svgfSpecAtrous[0] = std::make_shared<VKTexture>(reflW, reflH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        vr.svgfSpecAtrous[1] = std::make_shared<VKTexture>(reflW, reflH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);

        // ReSTIR-DI specular SVGF history: flat parallel to the reflection-spec SVGF above.
        // Bootstrap-cleared below (cross-frame temporal read).
        vr.svgfDiSpecDenoised = std::make_shared<VKTexture>(fullW, fullH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        for (u32 i = 0; i < 2; ++i)
        {
            vr.svgfDiSpecColorHist[i] = std::make_shared<VKTexture>(diW, diH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
            vr.svgfDiSpecMoments[i]   = std::make_shared<VKTexture>(diW, diH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
            vr.svgfDiSpecGeom[i]      = std::make_shared<VKTexture>(diW, diH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        }
        vr.svgfDiSpecAtrous[0] = std::make_shared<VKTexture>(diW, diH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        vr.svgfDiSpecAtrous[1] = std::make_shared<VKTexture>(diW, diH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);
        // Half-res DI specular a-trous final (upscale input); svgfDiSpecDenoised stays full.
        vr.svgfDiSpecHalf = std::make_shared<VKTexture>(diW, diH, TextureFormat::RGBA16F, 1, 0u, 1, VK_IMAGE_USAGE_STORAGE_BIT);

        // ReSTIR GI scratch reservoir: same lifecycle as the DI one but 64 B/pixel (a GIReservoir is
        // a world-space path vertex). Tags from the GI subsystem's disjoint 0xFFFF8000 range.
        if (vr.restirGiReservoirTag != 0)
        {
            Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(vr.restirGiReservoirTag);
            vr.restirGiReservoir = {};
        }
        vr.restirGiReservoirTag = m_RestirGi.NextReservoirTag();
        vr.restirGiReservoir = Memory::GPUTaggedPageAllocator::Get().AllocateLargeTaggedDeviceLocal(
            vr.restirGiReservoirTag, static_cast<u64>(giW) * static_cast<u64>(giH) * 64u, 16);

        // ReSTIR GI spatial-output / temporal-history buffer, 64 B/pixel (same topology as DI's).
        if (vr.restirGiSpatialTag != 0)
        {
            Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(vr.restirGiSpatialTag);
            vr.restirGiSpatial = {};
        }
        vr.restirGiSpatialTag = m_RestirGi.NextReservoirTag();
        vr.restirGiSpatial = Memory::GPUTaggedPageAllocator::Get().AllocateLargeTaggedDeviceLocal(
            vr.restirGiSpatialTag, static_cast<u64>(giW) * static_cast<u64>(giH) * 64u, 16);

        // Bootstrap clear: freshly-allocated VMA storage images have UNDEFINED layout and undefined
        // pixel content. The SVGF reproject
        // imageLoads its prev history on frame 0; without this clear the first read is NaN-prone garbage
        // (and imageLoad needs GENERAL). One-shot submit per view-resize only.
        VkImage clearTargets[25] = {
            // GI SVGF history: bootstrap-cleared so frame 0's prev imageLoad is well-defined.
            std::static_pointer_cast<VKTexture>(vr.svgfGiColorHist[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfGiColorHist[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfGiMoments[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfGiMoments[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfGiGeom[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfGiGeom[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfGiAtrous[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfGiAtrous[1])->GetImage(),
            // PathTrace fp32 accumulator: read-before-write cross-frame, so zero it to GENERAL on resize.
            std::static_pointer_cast<VKTexture>(vr.ptAccum)->GetImage(),
            // Specular SVGF history: frame 0's prev imageLoad must be well-defined.
            std::static_pointer_cast<VKTexture>(vr.svgfSpecColorHist[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfSpecColorHist[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfSpecMoments[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfSpecMoments[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfSpecGeom[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfSpecGeom[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfSpecAtrous[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfSpecAtrous[1])->GetImage(),
            // ReSTIR-DI specular SVGF history: frame 0's prev imageLoad must be well-defined.
            std::static_pointer_cast<VKTexture>(vr.svgfDiSpecColorHist[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfDiSpecColorHist[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfDiSpecMoments[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfDiSpecMoments[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfDiSpecGeom[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfDiSpecGeom[1])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfDiSpecAtrous[0])->GetImage(),
            std::static_pointer_cast<VKTexture>(vr.svgfDiSpecAtrous[1])->GetImage(),
        };
        constexpr u32 kClearCount = 25;
        VulkanContext::Get().ImmediateSubmit([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier toDst[kClearCount]{};
            for (u32 i = 0; i < kClearCount; ++i)
            {
                toDst[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                toDst[i].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
                toDst[i].newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toDst[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toDst[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toDst[i].image               = clearTargets[i];
                toDst[i].subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                toDst[i].srcAccessMask       = 0;
                toDst[i].dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, kClearCount, toDst);

            VkClearColorValue zero{ { 0.0f, 0.0f, 0.0f, 0.0f } };
            VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            for (u32 i = 0; i < kClearCount; ++i)
                vkCmdClearColorImage(cmd, clearTargets[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);

            VkImageMemoryBarrier toGen[kClearCount]{};
            for (u32 i = 0; i < kClearCount; ++i)
            {
                toGen[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                toGen[i].oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toGen[i].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
                toGen[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toGen[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toGen[i].image               = clearTargets[i];
                toGen[i].subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                toGen[i].srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
                toGen[i].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            }
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, kClearCount, toGen);
        });
    }

    void RenderPipeline::DestroyViewResources(ViewResources& vr)
    {
        // Pool destruction frees every descriptor set allocated from it.
        vr.overlays.reset();
        vr.slimViz.reset();
        vr.composite.reset();
        vr.bloom.reset();
        vr.fog.reset();
        vr.transparency.reset();
        vr.taa.reset();
        vr.rtShadow.reset();
        vr.restirDi.reset();
        vr.restirGiDI.reset();
        vr.diDenoiser.reset();
        vr.svgfGiDenoised.reset();
        vr.svgfGiHalf.reset();

        vr.svgfDiSpecHalf.reset();
        for (u32 i = 0; i < 2; ++i)
        {
            vr.svgfGiColorHist[i].reset();
            vr.svgfGiMoments[i].reset();
            vr.svgfGiGeom[i].reset();
        }
        vr.svgfGiAtrous[0].reset();
        vr.svgfGiAtrous[1].reset();
        vr.ptAccum.reset();
        vr.ptColor.reset();
        vr.reflRadiance.reset();
        vr.svgfSpecDenoised.reset();
        vr.svgfSpecHalf.reset();
        for (u32 i = 0; i < 2; ++i)
        {
            vr.svgfSpecColorHist[i].reset();
            vr.svgfSpecMoments[i].reset();
            vr.svgfSpecGeom[i].reset();
        }
        vr.svgfSpecAtrous[0].reset();
        vr.svgfSpecAtrous[1].reset();
        vr.svgfDiSpecDenoised.reset();
        for (u32 i = 0; i < 2; ++i)
        {
            vr.svgfDiSpecColorHist[i].reset();
            vr.svgfDiSpecMoments[i].reset();
            vr.svgfDiSpecGeom[i].reset();
        }
        vr.svgfDiSpecAtrous[0].reset();
        vr.svgfDiSpecAtrous[1].reset();
        // ReSTIR GI reserved-range tags: same deferred-destroy path as the DI tags above.
        if (vr.restirGiReservoirTag != 0)
        {
            Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(vr.restirGiReservoirTag);
            vr.restirGiReservoirTag = 0;
            vr.restirGiReservoir = {};
        }
        if (vr.restirGiSpatialTag != 0)
        {
            Memory::GPUTaggedPageAllocator::Get().FreeTagAndDestroy(vr.restirGiSpatialTag);
            vr.restirGiSpatialTag = 0;
            vr.restirGiSpatial = {};
        }

        if (vr.descPool != VK_NULL_HANDLE)
        {
            VulkanContext::Get().PushDeletion([pool = vr.descPool]() {
                vkDestroyDescriptorPool(VulkanContext::Get().GetDevice(), pool, nullptr);
            });
            vr.descPool = VK_NULL_HANDLE;
        }
        vr = {};
    }
}
