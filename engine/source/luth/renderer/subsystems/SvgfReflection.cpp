#include "luthpch.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/backend/vulkan/VulkanTexture.h"

namespace Luth
{
    namespace {
        struct ReprojectPC {
            f32 alphaColor, alphaMoments, historyCap, depthThreshold, normalThreshold;
            i32 gbufferScale, dispatchW, dispatchH;
            f32 antiFireflySigma, confidenceScale;
        };
        struct MomentsPC { f32 phiDepth, phiNormal; i32 gbufferScale, dispatchW, dispatchH; };
        struct AtrousPC {
            i32 stepSize, writeFinal; f32 phiColor, phiNormal, phiDepth;
            i32 gbufferScale, dispatchW, dispatchH; f32 phiRough;
        };
        static_assert(sizeof(ReprojectPC) == 40 && sizeof(MomentsPC) == 20 && sizeof(AtrousPC) == 36);
        void Bind(VkCommandBuffer cmd, const ReflectionDenoiserBindings& native, u32 pipeline, VkDescriptorSet set)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, native.pipelines[pipeline]);
            if (pipeline) {
                const VkDescriptorSet sets[]{native.globalSet, set};
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, native.layouts[pipeline], 0, 2, sets, 0, nullptr);
            } else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, native.layouts[0], 0, 1, &set, 0, nullptr);
        }
        void Dispatch(VkCommandBuffer cmd, const ReflectionDenoiserBindings& native) {
            vkCmdDispatch(cmd, (native.width + 7) / 8, (native.height + 7) / 8, 1);
        }
    }
    ReflectionDenoiserBindings SvgfDenoiser::PrepareReflectionBindings(const ViewResources& vr, u64 frame,
        RenderViewId id, u64 generation, const SvgfSettings& settings) const
    {
        ReflectionDenoiserBindings native;
        if (m_Channel != DenoiserChannel::Reflections) return native;
        const auto& owner = vr.reflectionDenoiser;
        if (!owner || !owner->input) return native;
        const auto& state = *owner;
        if (state.id != id || state.input->id != id)
            throw std::invalid_argument("DenoiseReflection: incompatible native view owner");
        native.retained = owner; native.settings = settings;
        native.view = id; native.generation = generation; native.frameIndex = frame;
        native.historyValid = state.history.CanReuse(frame, generation,
            vr.cameraHistory.CanReuse(frame, generation)) && state.input->history.CanReuse(frame, generation, true);
        native.fullWidth = vr.width; native.fullHeight = vr.height;
        native.width = state.width; native.height = state.height;
        const u32 parity = static_cast<u32>(frame & 1u);
        native.globalSet = vr.globalDescriptorSet[frame % MAX_FRAMES_IN_FLIGHT];
        native.copySet = state.svgfPassthroughDescSet;
        native.reprojectSet = state.svgfReprojectDescSet[parity]; native.momentsSet = state.svgfMomentsDescSet[parity];
        std::copy_n(state.svgfAtrousDescSet, 2, native.atrousSets.begin());
        const VKComputePipeline* pipelines[]{m_PassthroughPipeline.get(), m_ReprojectPipeline.get(), m_MomentsPipeline.get(), m_AtrousPipeline.get()};
        for (u32 i = 0; i < 4; ++i) if (pipelines[i]) {
            native.pipelines[i] = pipelines[i]->GetHandle(); native.layouts[i] = pipelines[i]->GetLayout();
        }
        auto freeze = [](const std::shared_ptr<Texture>& texture, TextureBindingRef& binding, VkImage& image, VkImageView& view) {
            if (!texture) return;
            binding = {texture.get()}; const auto vk = std::static_pointer_cast<VKTexture>(texture);
            image = vk->GetImage(); view = vk->GetImageView();
        };
        freeze(*state.Noisy(), native.sources[0], native.sourceImages[0], native.sourceViews[0]);
        for (u32 i = 0; i < state.sources.size(); ++i) {
            freeze(state.sources[i], native.sources[i + 1], native.sourceImages[i + 1], native.sourceViews[i + 1]);
            native.sourceViews[i + 1] = state.sourceViews[i]; // Descriptor views selected by the owner.
        }
        const std::shared_ptr<Texture> working[]{state.svgfColorHist[parity], state.svgfMoments[parity], state.svgfAtrous[0], state.svgfAtrous[1]};
        for (u32 i = 0; i < 4; ++i) freeze(working[i], native.working[i], native.workingImages[i], native.workingViews[i]);
        // Match the descriptor source selected from both working dimensions.
        const bool half = state.width != state.svgfDenoised->GetWidth() || state.height != state.svgfDenoised->GetHeight();
        freeze(half ? state.svgfHalf : state.svgfDenoised, native.output, native.outputImage, native.outputView);
        return native;
    }
    RG::ResourceHandle SvgfDenoiser::AddReflectionPasses(RG::RenderGraph& graph,
        const std::array<RG::ResourceHandle, 5>& inputs, const ReflectionDenoiserBindings& native)
    {
        if (!inputs[0].IsValid() || !native.Ready()) return {};
        auto import = [&](VkImage image, VkImageView view, const char* name) {
            RG::TextureDesc desc; desc.name = name; desc.width = native.width; desc.height = native.height;
            desc.format = RG::TextureFormat::RGBA16_Float;
            return graph.ImportResource(desc, (void*)image, (void*)view, RG::ResourceState::Undefined);
        };
        if (!native.ChainReady()) {
            struct Data { RG::ResourceHandle in, out; }; RG::ResourceHandle output;
            graph.AddComputePass<Data>("SvgfSpecPassthrough", RG::QueueFamily::AsyncCompute,
                [&](Data& data, RG::RenderPassBuilder& builder) {
                    data.in = builder.ReadStorageImage(inputs[0]);
                    data.out = builder.WriteStorageImage(import(native.outputImage, native.outputView, "SvgfDenoised"));
                    output = data.out;
                }, [native](Data&, RG::RenderPassContext& ctx) {
                    Bind(ctx.commandBuffer, native, 0, native.copySet); Dispatch(ctx.commandBuffer, native);
                });
            return output;
        }
        const auto& s = native.settings;
        const i32 scale = native.width == native.fullWidth && native.height == native.fullHeight ? 1 : 2;
        const i32 width = static_cast<i32>(native.width), height = static_cast<i32>(native.height);
        const ReprojectPC rpc{s.alphaColor, s.alphaMoments, native.HistoryCap(), s.depthThreshold,
            s.normalThreshold, scale, width, height, s.antiFireflySigma, 0.0f};
        const MomentsPC mpc{s.phiDepth, s.phiNormal, scale, width, height};
        struct ReprojectData { RG::ResourceHandle color, moments; };
        RG::ResourceHandle color, moments;
        graph.AddComputePass<ReprojectData>("SvgfSpecReproject", RG::QueueFamily::AsyncCompute,
            [&](ReprojectData& data, RG::RenderPassBuilder& builder) {
                for (u32 i = 0; i < 4; ++i) builder.ReadStorageImage(inputs[i]);
                data.color = builder.WriteStorageImage(import(native.workingImages[0], native.workingViews[0], "SvgfColorHistCurr"));
                data.moments = builder.WriteStorageImage(import(native.workingImages[1], native.workingViews[1], "SvgfMomentsCurr"));
                color = data.color; moments = data.moments;
            }, [native, rpc](ReprojectData&, RG::RenderPassContext& ctx) {
                Bind(ctx.commandBuffer, native, 1, native.reprojectSet);
                vkCmdPushConstants(ctx.commandBuffer, native.layouts[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpc), &rpc);
                Dispatch(ctx.commandBuffer, native);
            });
        struct MomentsData { RG::ResourceHandle out; }; RG::ResourceHandle a0;
        graph.AddComputePass<MomentsData>("SvgfSpecMoments", RG::QueueFamily::AsyncCompute,
            [&](MomentsData& data, RG::RenderPassBuilder& builder) {
                builder.ReadStorageImageGeneral(color); builder.ReadStorageImageGeneral(moments);
                builder.ReadStorageImage(inputs[1]); builder.ReadStorageImage(inputs[2]);
                data.out = builder.WriteStorageImage(import(native.workingImages[2], native.workingViews[2], "SvgfAtrous0")); a0 = data.out;
            }, [native, mpc](MomentsData&, RG::RenderPassContext& ctx) {
                Bind(ctx.commandBuffer, native, 2, native.momentsSet);
                vkCmdPushConstants(ctx.commandBuffer, native.layouts[2], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mpc), &mpc);
                Dispatch(ctx.commandBuffer, native);
            });
        RG::ResourceHandle atrous[2]{a0, {}}, output;
        const u32 iterations = std::max(1u, s.atrousIterations);
        for (u32 i = 0; i < iterations; ++i) {
            const u32 in = i & 1u, out = in ^ 1u; const bool final = i == iterations - 1;
            const AtrousPC pc{1 << i, final ? 1 : 0, s.phiColor, s.phiNormal, s.phiDepth, scale, width, height,
                s.phiRough};
            struct Data { RG::ResourceHandle in, out, denoised; };
            graph.AddComputePass<Data>("SvgfSpecAtrous", RG::QueueFamily::AsyncCompute,
                [&](Data& data, RG::RenderPassBuilder& builder) {
                    data.in = builder.ReadStorageImageGeneral(atrous[in]);
                    builder.ReadStorageImage(inputs[1]); builder.ReadStorageImage(inputs[2]); builder.ReadStorageImage(inputs[3]);
                    if (!atrous[out].IsValid()) atrous[out] = import(native.workingImages[2 + out], native.workingViews[2 + out], "SvgfAtrousAlt");
                    data.out = builder.WriteStorageImage(atrous[out]); atrous[out] = data.out;
                    if (final) { data.denoised = builder.WriteStorageImage(import(native.outputImage, native.outputView, "SvgfDenoised")); output = data.denoised; }
                }, [native, pc, in](Data&, RG::RenderPassContext& ctx) {
                    Bind(ctx.commandBuffer, native, 3, native.atrousSets[in]);
                    vkCmdPushConstants(ctx.commandBuffer, native.layouts[3], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    Dispatch(ctx.commandBuffer, native);
                });
        }
        return output;
    }
}
