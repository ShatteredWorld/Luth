#include <doctest/doctest.h>
#include "luth/renderer/features/rt/ReflectionDenoiserFeature.h"
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/rt/ReflectionDenoiserBindings.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include "luth/renderer/RenderPipeline.h"
#include <limits>

using namespace Luth;
namespace {
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    struct Fixture {
        ReflectionDenoiserBindings native;
        u32 fullWidth = 1279, fullHeight = 719, width = 1279, height = 719;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture(bool half = false, u32 fullW = 1279, u32 fullH = 719) {
            fullWidth = fullW; fullHeight = fullH;
            width = half ? std::max(fullWidth / 2, 1u) : fullWidth;
            height = half ? std::max(fullHeight / 2, 1u) : fullHeight;
            native.fullWidth = fullWidth; native.fullHeight = fullHeight; native.width = width; native.height = height;
            native.view = {1}; native.generation = 3; native.frameIndex = (1ull << 32) + 43;
            native.pipelines.fill(Native<VkPipeline>(1)); native.layouts.fill(Native<VkPipelineLayout>(2));
            native.globalSet = native.copySet = native.reprojectSet = native.momentsSet = Native<VkDescriptorSet>(3);
            native.atrousSets.fill(Native<VkDescriptorSet>(4));
            for (u32 i = 0; i < 5; ++i) { native.sources[i] = {Native<const Texture*>(10 + i)};
                native.sourceImages[i] = Native<VkImage>(20 + i); native.sourceViews[i] = Native<VkImageView>(30 + i); }
            for (u32 i = 0; i < 4; ++i) { native.working[i] = {Native<const Texture*>(40 + i)};
                native.workingImages[i] = Native<VkImage>(50 + i); native.workingViews[i] = Native<VkImageView>(60 + i); }
            native.output = {Native<const Texture*>(70)}; native.outputImage = Native<VkImage>(80); native.outputView = Native<VkImageView>(90);
            RenderPipelineDefinition definition; definition.AddFeature<ReflectionDenoiserFeature>();
            auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, Inputs());
            REQUIRE(result.pipeline); pipeline = std::move(result.pipeline);
        }
        static PipelineInputContract Inputs() {
            PipelineInputContract inputs;
            inputs.resources = {{ReflectionDenoiserResources::Bindings},
                {ReflectionResources::Radiance, ResourceOutputPresence::Optional},
                {RenderResources::SurfaceDepth}, {RenderResources::Normal}, {RenderResources::MaterialID}, {RenderResources::Roughness}};
            return inputs;
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& output,
            bool noisy = true, bool packet = true) {
            const std::array formats{RG::TextureFormat::RGBA16_Float, RG::TextureFormat::D32_Float, RG::TextureFormat::RG16_Float,
                RG::TextureFormat::R8_Unorm, RG::TextureFormat::R16_Uint};
            std::array<GraphTextureRef, 5> textures;
            for (u32 i = 0; i < 5; ++i) {
                RG::TextureDesc desc; desc.width = i ? fullWidth : width; desc.height = i ? fullHeight : height; desc.format = formats[i];
                textures[i] = {graph.ImportResource(desc, (void*)Native<VkImage>(20 + i), (void*)Native<VkImageView>(30 + i), RG::ResourceState::ColorAttachment), {Native<const Texture*>(10 + i)}};
            }
            const ReflectionDenoiserBindingRef ref{packet ? &native : nullptr};
            const auto rawKey = ReflectionResources::Radiance;
            const std::array resources{RenderInputBinding::Present(ReflectionDenoiserResources::Bindings, ref),
                noisy ? RenderInputBinding::Present(rawKey, textures[0]) : RenderInputBinding::Absent(rawKey),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, textures[1]), RenderInputBinding::Present(RenderResources::Normal, textures[2]),
                RenderInputBinding::Present(RenderResources::Roughness, textures[3]),
                RenderInputBinding::Present(RenderResources::MaterialID, textures[4])};
            FrameRenderInputs frame; frame.renderFrameIndex = (1ull << 32) + 43; frame.resources = resources;
            ViewRenderInputs view; view.id = {1}; view.resourceGeneration = 3; view.width = fullWidth; view.height = fullHeight;
            const std::array exports{RenderOutputBinding::Capture(ReflectionDenoiserResources::Radiance, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("ReflectionDenoiserFeature: full half and narrow extents preserve native chain topology") {
    bool half = false; u32 w = 1279, h = 719;
    SUBCASE("Full") {} SUBCASE("Half") { half = true; } SUBCASE("Narrow half") { half = true; w = 1; h = 9; }
    Fixture f(half, w, h); Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success); REQUIRE(graph.GetPasses().size() == 7);
    CHECK(graph.GetPasses()[0].name == "SvgfSpecReproject"); CHECK(graph.GetPasses()[1].name == "SvgfSpecMoments");
    REQUIRE(graph.GetResources().size() == 10); CHECK(output.binding.texture == f.native.output.texture);
    CHECK(graph.GetResources()[output.handle.index - 1].desc.width == f.width); CHECK(graph.GetResources()[output.handle.index - 1].desc.height == f.height);
    CHECK(graph.GetPasses()[0].reads.size() == 4); // Roughness virtual reprojection, no surface motion.
    CHECK(graph.GetPasses()[0].writes[0].index == graph.GetPasses()[1].reads[0].index);
    CHECK(graph.GetPasses()[1].writes[0].index == graph.GetPasses()[2].reads[0].index);
    for (u32 i = 2; i < 7; ++i) {
        CHECK(graph.GetPasses()[i].name == "SvgfSpecAtrous"); CHECK(graph.GetPasses()[i].reads.size() == 4); // Ping-pong, depth, normal, roughness.
        if (i > 2) CHECK(graph.GetPasses()[i - 1].writes[0].index == graph.GetPasses()[i].reads[0].index);
    }
    graph.Compile(); for (const auto& pass : graph.GetPasses()) { CHECK_FALSE(pass.culled); CHECK(pass.queueFamily == RG::QueueFamily::AsyncCompute); }
}
TEST_CASE("ReflectionDenoiserFeature: disabled denoising and incomplete chains retain raw copy output") {
    Fixture f(true);
    SUBCASE("Disabled") { f.native.settings.enabled = false; }
    SUBCASE("Cold reproject") { f.native.pipelines[1] = VK_NULL_HANDLE; }
    SUBCASE("Cold moments") { f.native.momentsSet = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success); REQUIRE(graph.GetPasses().size() == 1); CHECK(graph.GetPasses()[0].name == "SvgfSpecPassthrough");
    CHECK(graph.GetPasses()[0].reads.size() == 1); CHECK(graph.GetResources().size() == 6); CHECK(output.handle.IsValid());
    CHECK(output.binding.texture == f.native.output.texture);
}
TEST_CASE("ReflectionDenoiserFeature: absent reflections and dormant pipelines clear graph-local outputs") {
    Fixture f; bool haveNoisy = true, packet = true;
    SUBCASE("Reflections absent") { haveNoisy = false; } SUBCASE("Packet absent") { packet = false; }
    SUBCASE("All cold") { f.native.pipelines.fill(VK_NULL_HANDLE); }
    SUBCASE("Raw cold") { f.native.settings.enabled = false; f.native.pipelines[0] = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    GraphTextureRef output{{99, 0}, {Native<const Texture*>(100)}};
    REQUIRE(f.Build(graph, scratch, output, haveNoisy, packet).success); CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid()); CHECK(output.binding.texture == nullptr);
    SvgfDenoiser dormant(DenoiserChannel::Reflections); ViewResources vr;
    CHECK_FALSE(dormant.PrepareReflectionBindings(vr, 43, {1}, 3, {}).Ready());
}
TEST_CASE("ReflectionDenoiserFeature: stale or incompatible frozen inputs fail before registration") {
    Fixture f;
    SUBCASE("View") { f.native.view = {2}; } SUBCASE("Generation") { ++f.native.generation; }
    SUBCASE("Frame truncation") { f.native.frameIndex = 43; } SUBCASE("Full extent") { ++f.native.fullWidth; }
    SUBCASE("Working extent") { --f.native.width; } SUBCASE("Source texture") { f.native.sources[3].texture = Native<const Texture*>(99); }
    SUBCASE("Source image") { f.native.sourceImages[0] = Native<VkImage>(99); }
    SUBCASE("Source view") { f.native.sourceViews[4] = Native<VkImageView>(99); }
    SUBCASE("Source subresource") { f.native.sources[1].baseLayer = 1; }
    SUBCASE("Output null") { f.native.outputImage = VK_NULL_HANDLE; }
    SUBCASE("Output subresource") { f.native.output.baseMip = 1; }
    SUBCASE("Output aliases source") { f.native.outputImage = f.native.sourceImages[0]; }
    SUBCASE("Global descriptor") { f.native.globalSet = VK_NULL_HANDLE; }
    SUBCASE("Chain layout") { f.native.layouts[2] = VK_NULL_HANDLE; }
    SUBCASE("Unsafe shift") { f.native.settings.atrousIterations = 32; }
    SUBCASE("Nonfinite settings") { f.native.settings.alphaColor = std::numeric_limits<f32>::quiet_NaN(); }
    SUBCASE("Nonfinite roughness threshold") { f.native.settings.phiRough = std::numeric_limits<f32>::infinity(); }
    SUBCASE("Working image null") { f.native.workingViews[0] = VK_NULL_HANDLE; }
    SUBCASE("Working alias") { f.native.workingImages[0] = f.native.workingImages[2]; }
    SUBCASE("Working texture alias") { f.native.working[0] = f.native.working[1]; }
    SUBCASE("Working sampled alias") { f.native.workingImages[0] = f.native.sourceImages[2]; }
    SUBCASE("Working output alias") { f.native.workingImages[3] = f.native.outputImage; }
    SUBCASE("Copy layout") { f.native.settings.enabled = false; f.native.layouts[0] = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    CHECK_FALSE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty()); CHECK_FALSE(output.handle.IsValid());
}
TEST_CASE("ReflectionDenoiserFeature: recorded jobs retain domain resources independently of the preparation packet") {
    Fixture f; auto retained = std::make_shared<ReflectionDenoiserViewState>(); std::weak_ptr<ReflectionDenoiserViewState> weak = retained;
    f.native.retained = retained; retained.reset();
    Memory::LinearAllocator scratch(128 * 1024);
    { RG::RenderGraph graph(scratch); GraphTextureRef output; REQUIRE(f.Build(graph, scratch, output).success);
      f.native.retained.reset(); f.native.pipelines.fill(VK_NULL_HANDLE); f.native.settings.enabled = false;
      CHECK_FALSE(weak.expired()); CHECK(graph.GetPasses().size() == 7); }
    CHECK(weak.expired());
}
TEST_CASE("ReflectionDenoiserFeature: compiler orders reflections before its optional denoiser and requires surface contracts") {
    auto inputs = Fixture::Inputs(); inputs.resources.erase(inputs.resources.begin() + 1); // Reflections have an explicit feature producer.
    for (auto key : {ResourceKeyRef{RtSceneResources::Parameters}, ResourceKeyRef{ReflectionResources::Bindings}, ResourceKeyRef{RenderResources::LightData}})
        inputs.resources.push_back({key});
    inputs.resources.push_back({RtSceneResources::Scene, ResourceOutputPresence::Optional}); inputs.capabilities = {&RtSceneResources::RayScene};
    RenderPipelineDefinition definition; const auto denoise = definition.AddFeature<ReflectionDenoiserFeature>(); const auto di = definition.AddFeature<ReflectionFeature>();
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), RendererCapabilities{{&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries}}, inputs);
    REQUIRE(result.pipeline); CHECK(result.pipeline->FeatureOrder()[0] == di); CHECK(result.pipeline->FeatureOrder()[1] == denoise);
    inputs = Fixture::Inputs(); inputs.resources.erase(inputs.resources.begin() + 4); // Material ID.
    RenderPipelineDefinition invalid; invalid.AddFeature<ReflectionDenoiserFeature>();
    auto failed = RenderPipelineCompiler{}.Compile(std::move(invalid), {}, inputs); CHECK_FALSE(failed.pipeline); CHECK_FALSE(failed.diagnostics.empty());
}

TEST_CASE("ReflectionDenoiserFeature: runtime iteration changes preserve final output and unique imports") {
    Fixture f;
    SUBCASE("Zero clamps to one") { f.native.settings.atrousIterations = 0; }
    SUBCASE("One") { f.native.settings.atrousIterations = 1; }
    SUBCASE("Even") { f.native.settings.atrousIterations = 2; }
    SUBCASE("Maximum safe shift") { f.native.settings.atrousIterations = 31; }
    Memory::LinearAllocator scratch(256 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success);
    CHECK(graph.GetPasses().size() == 2 + std::max(1u, f.native.settings.atrousIterations));
    CHECK(graph.GetPasses().back().writes.back().index == output.handle.index);
    CHECK(graph.GetResources().size() == 10);
    for (const auto image : f.native.workingImages) {
        u32 count = 0; for (const auto& node : graph.GetResources()) count += node.image == image;
        CHECK(count == 1);
    }
}

TEST_CASE("ReflectionDenoiserFeature: preexisting physical imports fail before pass registration") {
    Fixture f; VkImage image = f.native.outputImage; VkImageView view = f.native.outputView;
    SUBCASE("Output") {}
    SUBCASE("Working history") { image = f.native.workingImages[0]; view = f.native.workingViews[0]; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    RG::TextureDesc desc; desc.width = f.width; desc.height = f.height; desc.format = RG::TextureFormat::RGBA16_Float;
    graph.ImportResource(desc, (void*)image, (void*)view, RG::ResourceState::Undefined);
    CHECK_FALSE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid());
}


