#include <doctest/doctest.h>
#include "luth/renderer/features/rt/GiUpscaleFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include <algorithm>
#include <limits>

using namespace Luth;
namespace {
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    struct Fixture {
        GiUpscaleBindings native;
        u32 width, height, fullWidth, fullHeight;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture(bool half = true, u32 w = 1279, u32 h = 719)
            : width(half ? std::max(w / 2, 1u) : w), height(half ? std::max(h / 2, 1u) : h), fullWidth(w), fullHeight(h)
        {
            native.view = {1}; native.generation = 3; native.frameIndex = (1ull << 32) + 43;
            native.width = width; native.height = height; native.fullWidth = w; native.fullHeight = h;
            native.pipeline = Native<VkPipeline>(1); native.layout = Native<VkPipelineLayout>(2);
            native.globalSet = native.set = Native<VkDescriptorSet>(3);
            for (u32 i = 0; i < 3; ++i) {
                native.sources[i] = {Native<const Texture*>(10 + i)};
                native.sourceImages[i] = Native<VkImage>(20 + i); native.sourceViews[i] = Native<VkImageView>(30 + i);
            }
            native.output = {Native<const Texture*>(40)}; native.outputImage = Native<VkImage>(50); native.outputView = Native<VkImageView>(60);
            RenderPipelineDefinition definition; definition.AddFeature<GiUpscaleFeature>();
            auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, Inputs());
            REQUIRE(result.pipeline); pipeline = std::move(result.pipeline);
        }
        static PipelineInputContract Inputs() {
            PipelineInputContract inputs;
            inputs.resources = {{GiUpscaleResources::Bindings},
                {GiDenoiserResources::Diffuse, ResourceOutputPresence::Optional},
                {RenderResources::SurfaceDepth}, {RenderResources::Normal}};
            return inputs;
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& output,
            bool present = true, bool packet = true) {
            std::array<GraphTextureRef, 3> textures;
            const std::array formats{RG::TextureFormat::RGBA16_Float, RG::TextureFormat::D32_Float, RG::TextureFormat::RG16_Float};
            const bool full = width == fullWidth && height == fullHeight;
            for (u32 i = 0; i < 3; ++i) {
                RG::TextureDesc desc; desc.width = i ? fullWidth : width; desc.height = i ? fullHeight : height; desc.format = formats[i];
                const bool final = i == 0 && full;
                textures[i] = {graph.ImportResource(desc, (void*)Native<VkImage>(final ? 50 : 20 + i),
                    (void*)Native<VkImageView>(final ? 60 : 30 + i), RG::ResourceState::ComputeWrite),
                    {Native<const Texture*>(final ? 40 : 10 + i)}};
            }
            const GiUpscaleBindingRef ref{packet ? &native : nullptr};
            const auto inputKey = GiDenoiserResources::Diffuse;
            const std::array resources{RenderInputBinding::Present(GiUpscaleResources::Bindings, ref),
                present ? RenderInputBinding::Present(inputKey, textures[0]) : RenderInputBinding::Absent(inputKey),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, textures[1]), RenderInputBinding::Present(RenderResources::Normal, textures[2])};
            FrameRenderInputs frame; frame.renderFrameIndex = (1ull << 32) + 43; frame.resources = resources;
            ViewRenderInputs view; view.id = {1}; view.resourceGeneration = 3; view.width = fullWidth; view.height = fullHeight;
            const std::array exports{RenderOutputBinding::Capture(GiUpscaleResources::Diffuse, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("GiUpscaleFeature: preserves half-resolution native topology") {
    u32 w = 1279, h = 719;
    SUBCASE("Ordinary") {} SUBCASE("Narrow") { w = 1; h = 9; }
    Fixture f(true, w, h); Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success); REQUIRE(graph.GetPasses().size() == 1);
    CHECK(graph.GetPasses()[0].name == "GiUpscale");
    REQUIRE(graph.GetPasses()[0].reads.size() == 3);
    for (u32 i = 0; i < 3; ++i) CHECK(graph.GetPasses()[0].reads[i].index == i + 1);
    REQUIRE(output.handle.IsValid()); CHECK(output.binding.texture == f.native.output.texture);
    CHECK(graph.GetResources().size() == 4);
    CHECK(graph.GetResources()[output.handle.index - 1].desc.width == w); CHECK(graph.GetResources()[output.handle.index - 1].desc.height == h);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK(graph.GetPasses()[0].queueFamily == RG::QueueFamily::AsyncCompute);
}
TEST_CASE("GiUpscaleFeature: full and collapsed extents forward final output without native work") {
    u32 w = 1279, h = 719; bool half = false;
    SUBCASE("Full") {}
    SUBCASE("Collapsed half") { half = true; w = h = 1; }
    Fixture f(half, w, h); f.native.pipeline = VK_NULL_HANDLE;
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 3);
    CHECK(output.handle.index == 1); CHECK(output.binding.texture == f.native.output.texture);
}
TEST_CASE("GiUpscaleFeature: absent signals packets and cold half-resolution pipelines clear outputs") {
    Fixture f; bool input = true, packet = true;
    SUBCASE("Signal absent") { input = false; } SUBCASE("Packet absent") { packet = false; }
    SUBCASE("Pipeline absent") { f.native.pipeline = VK_NULL_HANDLE; }
    SUBCASE("Native set absent") { f.native.set = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{99, 0}, {Native<const Texture*>(100)}};
    REQUIRE(f.Build(graph, scratch, output, input, packet).success); CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid()); CHECK(output.binding.texture == nullptr);
}
TEST_CASE("GiUpscaleFeature: stale incompatible and aliased bindings fail before registration") {
    Fixture f;
    SUBCASE("View") { f.native.view = {2}; } SUBCASE("Generation") { ++f.native.generation; }
    SUBCASE("Frame truncation") { f.native.frameIndex = 43; }
    SUBCASE("Full extent") { ++f.native.fullWidth; } SUBCASE("Working extent") { --f.native.width; }
    SUBCASE("Source texture") { f.native.sources[0].texture = Native<const Texture*>(99); }
    SUBCASE("Source image") { f.native.sourceImages[1] = Native<VkImage>(99); }
    SUBCASE("Source view") { f.native.sourceViews[2] = Native<VkImageView>(99); }
    SUBCASE("Source subresource") { f.native.sources[0].baseLayer = 1; }
    SUBCASE("Output null") { f.native.outputImage = VK_NULL_HANDLE; }
    SUBCASE("Output subresource") { f.native.output.mipCount = 2; }
    SUBCASE("Output image alias") { f.native.outputImage = f.native.sourceImages[0]; }
    SUBCASE("Output texture alias") { f.native.output.texture = f.native.sources[2].texture; }
    SUBCASE("Source alias") { f.native.sourceImages[2] = f.native.sourceImages[1]; }
    SUBCASE("Depth parameter") { f.native.phiDepth = std::numeric_limits<f32>::quiet_NaN(); }
    SUBCASE("Normal parameter") { f.native.phiNormal = std::numeric_limits<f32>::infinity(); }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    CHECK_FALSE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty()); CHECK_FALSE(output.handle.IsValid());
}
TEST_CASE("GiUpscaleFeature: full-resolution forwarding validates lighting output identity") {
    Fixture f(false);
    SUBCASE("Image") { f.native.outputImage = Native<VkImage>(99); }
    SUBCASE("View") { f.native.outputView = Native<VkImageView>(99); }
    SUBCASE("Texture") { f.native.output.texture = Native<const Texture*>(99); }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    CHECK_FALSE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty()); CHECK_FALSE(output.handle.IsValid());
}
TEST_CASE("GiUpscaleFeature: recording retains prepared state independently of packet lifetime") {
    Fixture f; auto state = std::make_shared<GiUpscaleViewState>(); std::weak_ptr<GiUpscaleViewState> retained = state;
    f.native.retained = state; state.reset(); Memory::LinearAllocator scratch(128 * 1024);
    { RG::RenderGraph graph(scratch); GraphTextureRef output; REQUIRE(f.Build(graph, scratch, output).success);
      f.native.retained.reset(); f.native.pipeline = VK_NULL_HANDLE; f.native.phiDepth = 9; CHECK_FALSE(retained.expired()); }
    CHECK(retained.expired());
}
TEST_CASE("GiUpscaleFeature: compiler orders denoising and rejects missing or duplicate contracts") {
    auto inputs = Fixture::Inputs();
    inputs.resources.erase(inputs.resources.begin() + 1);
    inputs.resources.push_back({GiDenoiserResources::Bindings});
    inputs.resources.push_back({RestirGiResources::Diffuse, ResourceOutputPresence::Optional});
    inputs.resources.push_back({RenderResources::MotionVectors});
    inputs.resources.push_back({RenderResources::MaterialID});
    inputs.resources.push_back({RenderResources::Roughness});
    RenderPipelineDefinition definition;
    const auto upscale = definition.AddFeature<GiUpscaleFeature>();
    const auto denoise = definition.AddFeature<GiDenoiserFeature>();
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
    REQUIRE(result.pipeline); CHECK(result.pipeline->FeatureOrder()[0] == denoise);
    CHECK(result.pipeline->FeatureOrder()[1] == upscale);
    auto missing = Fixture::Inputs(); missing.resources.pop_back();
    RenderPipelineDefinition invalid; invalid.AddFeature<GiUpscaleFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(invalid), {}, missing).pipeline);
    RenderPipelineDefinition duplicate; duplicate.AddFeature<GiUpscaleFeature>(); duplicate.AddFeature<GiUpscaleFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(duplicate), {}, Fixture::Inputs()).pipeline);
}

TEST_CASE("GiUpscaleFeature: preexisting full output import fails before native registration") {
    Fixture f; Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    RG::TextureDesc desc; desc.width = f.fullWidth; desc.height = f.fullHeight; desc.format = RG::TextureFormat::RGBA16_Float;
    graph.ImportResource(desc, (void*)f.native.outputImage, (void*)f.native.outputView, RG::ResourceState::ComputeWrite);
    CHECK_FALSE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty()); CHECK_FALSE(output.handle.IsValid());
}

TEST_CASE("GiUpscaleFeature: compiler orders the complete GI signal chain independently of declaration order") {
    auto inputs = Fixture::Inputs(); inputs.resources.erase(inputs.resources.begin() + 1);
    inputs.resources.push_back({GiDenoiserResources::Bindings});
    inputs.resources.push_back({RestirGiResources::Bindings});
    inputs.resources.push_back({RtSceneResources::Parameters});
    inputs.resources.push_back({RtSceneResources::Scene, ResourceOutputPresence::Optional});
    inputs.resources.push_back({RenderResources::LightData});
    inputs.resources.push_back({RenderResources::MotionVectors});
    inputs.resources.push_back({RenderResources::MaterialID});
    inputs.resources.push_back({RenderResources::Roughness});
    inputs.capabilities = {&RtSceneResources::RayScene};
    RenderPipelineDefinition definition;
    const auto upscale = definition.AddFeature<GiUpscaleFeature>();
    const auto denoise = definition.AddFeature<GiDenoiserFeature>();
    const auto raw = definition.AddFeature<RestirGiFeature>();
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition),
        RendererCapabilities{{&RtSceneResources::AccelerationStructures, &RtSceneResources::RayQueries}}, inputs);
    REQUIRE(result.pipeline);
    REQUIRE(result.pipeline->FeatureOrder().size() == 3);
    CHECK(result.pipeline->FeatureOrder()[0] == raw);
    CHECK(result.pipeline->FeatureOrder()[1] == denoise);
    CHECK(result.pipeline->FeatureOrder()[2] == upscale);
}

