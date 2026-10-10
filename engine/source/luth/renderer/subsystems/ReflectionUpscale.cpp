#include "luthpch.h"
#include "luth/renderer/subsystems/ReflectionsSubsystem.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/Renderer.h"
#include "luth/renderer/settings/SvgfSettings.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"
namespace Luth
{
    namespace {
        struct ReflectionUpscalePC { i32 fullW, fullH, halfW, halfH; f32 phiDepth, phiNormal; };
        static_assert(sizeof(ReflectionUpscalePC) == 24);
    }
    std::shared_ptr<ReflectionUpscaleViewState> ReflectionsSubsystem::EnsureUpscaleView(RenderViewId id,
        const FrameTargets& targets, const std::shared_ptr<ReflectionDenoiserViewState>& denoiser)
    {
        if (!m_UpscaleSetLayout || !m_Sampler || !denoiser) return {};
        if (denoiser->id != id) throw std::invalid_argument("Reflection upscale: incompatible denoiser identity");
        if (!targets.GetSceneColor()) throw std::invalid_argument("Reflection upscale: missing view output");
        const std::array sources{targets.GetSceneDepth(), targets.GetSlimNormal()};
        std::array<VkImageView, 2> views{};
        for (u32 i = 0; i < sources.size(); ++i) {
            if (!sources[i] || sources[i]->GetWidth() != targets.GetSceneColor()->GetWidth() ||
                sources[i]->GetHeight() != targets.GetSceneColor()->GetHeight())
                throw std::invalid_argument("Reflection upscale: incompatible full-resolution source");
            views[i] = std::static_pointer_cast<VKTexture>(sources[i])->GetImageView();
            if (!views[i]) throw std::invalid_argument("Reflection upscale: missing source view");
        }
        const auto* prior = m_UpscaleViews.Find(id);
        const u64 generation = prior && (*prior)->denoiser == denoiser && (*prior)->sources == sources &&
            (*prior)->sourceViews == views ? (*prior)->sourceGeneration : m_NextUpscaleGeneration++;
        const ViewStateConfig config{targets.GetSceneColor()->GetWidth(), targets.GetSceneColor()->GetHeight(), 0, generation};
        return m_UpscaleViews.Ensure(id, config, [&](const ViewStateConfig& requested) {
            return ReflectionUpscaleViewState::Create(id, requested, m_UpscaleSetLayout, m_Sampler, denoiser, sources);
        }, [] { Renderer::WaitForGPU(); });
    }

    ReflectionUpscaleBindings ReflectionsSubsystem::PrepareUpscaleBindings(const ViewResources& vr, u64 frame,
        RenderViewId id, u64 generation, const SvgfSettings& settings) const
    {
        ReflectionUpscaleBindings native;
        if (!vr.reflectionUpscale) return native;
        const auto& state = *vr.reflectionUpscale;
        const auto& owner = state.denoiser;
        if (state.id != id || !owner || owner->id != id)
            throw std::invalid_argument("Reflection upscale: incompatible view owner");
        native.view = id; native.generation = generation; native.frameIndex = frame;
        native.retained = vr.reflectionUpscale;
        if (m_UpscalePipeline) {
            native.pipeline = m_UpscalePipeline->GetHandle(); native.layout = m_UpscalePipeline->GetLayout();
        }
        native.set = state.set;
        native.globalSet = vr.globalDescriptorSet[frame % MAX_FRAMES_IN_FLIGHT];
        native.width = owner->width; native.height = owner->height;
        native.fullWidth = vr.width; native.fullHeight = vr.height;
        native.phiDepth = settings.depthThreshold;
        auto freeze = [](const std::shared_ptr<Texture>& texture, TextureBindingRef& binding, VkImage& image, VkImageView& view) {
            if (!texture) return;
            const auto vk = std::static_pointer_cast<VKTexture>(texture);
            binding = {texture.get()}; image = vk->GetImage(); view = vk->GetImageView();
        };
        freeze(owner->svgfHalf, native.sources[0], native.sourceImages[0], native.sourceViews[0]);
        for (u32 i = 0; i < 2; ++i) {
            freeze(state.sources[i], native.sources[i + 1], native.sourceImages[i + 1], native.sourceViews[i + 1]);
            native.sourceViews[i + 1] = state.sourceViews[i];
        }
        freeze(owner->svgfDenoised, native.output, native.outputImage, native.outputView);
        return native;
    }

    RG::ResourceHandle ReflectionsSubsystem::AddUpscalePass(RG::RenderGraph& graph,
        const std::array<RG::ResourceHandle, 3>& inputs, const ReflectionUpscaleBindings& native)
    {
        if (!inputs[0].IsValid() || !native.Ready()) return inputs[0];
        // At a 1x1 extent, half resolution is already full resolution. The denoiser
        // writes the final image directly; do not sample its unused working image.
        if (inputs[0].index <= graph.GetResources().size() && native.outputImage &&
            graph.GetResources()[inputs[0].index - 1].image == native.outputImage)
            return inputs[0];
        struct Data { RG::ResourceHandle half, depth, normal, out; };
        RG::ResourceHandle output;
        const ReflectionUpscalePC pc{static_cast<i32>(native.fullWidth), static_cast<i32>(native.fullHeight),
            static_cast<i32>(native.width), static_cast<i32>(native.height), native.phiDepth, native.phiNormal};
        graph.AddComputePass<Data>("ReflUpscale", RG::QueueFamily::AsyncCompute,
            [&](Data& data, RG::RenderPassBuilder& builder) {
                data.half = builder.ReadStorageImageGeneral(inputs[0]);
                if (inputs[1].IsValid()) data.depth = builder.ReadStorageImage(inputs[1]);
                if (inputs[2].IsValid()) data.normal = builder.ReadStorageImage(inputs[2]);
                RG::TextureDesc desc; desc.name = "SvgfSpecDenoised";
                desc.width = native.fullWidth; desc.height = native.fullHeight; desc.format = RG::TextureFormat::RGBA16_Float;
                data.out = builder.WriteStorageImage(graph.ImportResource(desc, (void*)native.outputImage,
                    (void*)native.outputView, RG::ResourceState::Undefined));
                output = data.out;
            }, [native, pc](Data&, RG::RenderPassContext& ctx) {
                const VkDescriptorSet sets[]{native.globalSet, native.set};
                vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, native.pipeline);
                vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, native.layout, 0, 2, sets, 0, nullptr);
                vkCmdPushConstants(ctx.commandBuffer, native.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(ctx.commandBuffer, (native.fullWidth + 7) / 8, (native.fullHeight + 7) / 8, 1);
            });
        return output;
    }
}
