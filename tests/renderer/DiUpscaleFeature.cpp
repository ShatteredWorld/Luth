#include <doctest/doctest.h>
#include "luth/renderer/features/rt/DiUpscaleFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/rendergraph/RenderGraph.h"
#include <algorithm>
#include <limits>

using namespace Luth;
namespace {
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    struct Fixture {
        DiUpscaleBindings native;
        bool specular;
        u32 width, height, fullWidth, fullHeight;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture(bool half = true, bool spec = false, u32 w = 1279, u32 h = 719)
            : specular(spec), width(half ? std::max(w / 2, 1u) : w), height(half ? std::max(h / 2, 1u) : h), fullWidth(w), fullHeight(h)
        {
            native.signal = spec ? DiDenoiserSignal::Specular : DiDenoiserSignal::Diffuse;
            native.view = {1}; native.generation = 3; native.frameIndex = (1ull << 32) + 43;
            native.width = width; native.height = height; native.fullWidth = w; native.fullHeight = h;
            native.pipeline = Native<VkPipeline>(1); native.layout = Native<VkPipelineLayout>(2);
            native.globalSet = native.set = Native<VkDescriptorSet>(3);
            for (u32 i = 0; i < 3; ++i) {
                native.sources[i] = {Native<const Texture*>(10 + i)};
                native.sourceImages[i] = Native<VkImage>(20 + i); native.sourceViews[i] = Native<VkImageView>(30 + i);
            }
            native.output = {Native<const Texture*>(40)}; native.outputImage = Native<VkImage>(50); native.outputView = Native<VkImageView>(60);
            RenderPipelineDefinition definition; definition.AddFeature<DiUpscaleFeature>(native.signal);
            auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, Inputs(spec));
            REQUIRE(result.pipeline); pipeline = std::move(result.pipeline);
        }
        static PipelineInputContract Inputs(bool specular = false) {
            PipelineInputContract inputs;
            inputs.resources = {{specular ? DiUpscaleResources::SpecularBindings : DiUpscaleResources::Bindings},
                {specular ? DiDenoiserResources::Specular : DiDenoiserResources::Diffuse, ResourceOutputPresence::Optional},
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
            const DiUpscaleBindingRef ref{packet ? &native : nullptr};
            const auto inputKey = specular ? DiDenoiserResources::Specular : DiDenoiserResources::Diffuse;
            const std::array resources{RenderInputBinding::Present(specular ? DiUpscaleResources::SpecularBindings : DiUpscaleResources::Bindings, ref),
                present ? RenderInputBinding::Present(inputKey, textures[0]) : RenderInputBinding::Absent(inputKey),
                RenderInputBinding::Present(RenderResources::SurfaceDepth, textures[1]), RenderInputBinding::Present(RenderResources::Normal, textures[2])};
            FrameRenderInputs frame; frame.renderFrameIndex = (1ull << 32) + 43; frame.resources = resources;
            ViewRenderInputs view; view.id = {1}; view.resourceGeneration = 3; view.width = fullWidth; view.height = fullHeight;
            const std::array exports{RenderOutputBinding::Capture(specular ? DiUpscaleResources::Specular : DiUpscaleResources::Diffuse, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("DiUpscaleFeature: both signals preserve half-resolution native topology") {
    bool specular = false; u32 w = 1279, h = 719;
    SUBCASE("Diffuse") {} SUBCASE("Specular") { specular = true; } SUBCASE("Narrow") { w = 1; h = 9; }
    Fixture f(true, specular, w, h); Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success); REQUIRE(graph.GetPasses().size() == 1);
    CHECK(graph.GetPasses()[0].name == (specular ? "DiSpecUpscale" : "DiUpscale"));
    REQUIRE(graph.GetPasses()[0].reads.size() == 3);
    for (u32 i = 0; i < 3; ++i) CHECK(graph.GetPasses()[0].reads[i].index == i + 1);
    REQUIRE(output.handle.IsValid()); CHECK(output.binding.texture == f.native.output.texture);
    CHECK(graph.GetResources().size() == 4);
    CHECK(graph.GetResources()[output.handle.index - 1].desc.width == w); CHECK(graph.GetResources()[output.handle.index - 1].desc.height == h);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK(graph.GetPasses()[0].queueFamily == RG::QueueFamily::AsyncCompute);
}
TEST_CASE("DiUpscaleFeature: full and collapsed extents forward final output without native work") {
    bool specular = false; u32 w = 1279, h = 719; bool half = false;
    SUBCASE("Diffuse full") {} SUBCASE("Specular full") { specular = true; }
    SUBCASE("Collapsed half") { half = true; w = h = 1; }
    Fixture f(half, specular, w, h); f.native.pipeline = VK_NULL_HANDLE;
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 3);
    CHECK(output.handle.index == 1); CHECK(output.binding.texture == f.native.output.texture);
}
TEST_CASE("DiUpscaleFeature: absent signals packets and cold half-resolution pipelines clear outputs") {
    Fixture f; bool input = true, packet = true;
    SUBCASE("Signal absent") { input = false; } SUBCASE("Packet absent") { packet = false; }
    SUBCASE("Pipeline absent") { f.native.pipeline = VK_NULL_HANDLE; }
    SUBCASE("Native set absent") { f.native.set = VK_NULL_HANDLE; }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{99, 0}, {Native<const Texture*>(100)}};
    REQUIRE(f.Build(graph, scratch, output, input, packet).success); CHECK(graph.GetPasses().empty());
    CHECK_FALSE(output.handle.IsValid()); CHECK(output.binding.texture == nullptr);
}
TEST_CASE("DiUpscaleFeature: stale incompatible and aliased bindings fail before registration") {
    Fixture f(true, true);
    SUBCASE("Signal") { f.native.signal = DiDenoiserSignal::Diffuse; }
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
TEST_CASE("DiUpscaleFeature: full-resolution forwarding validates lighting output identity") {
    Fixture f(false);
    SUBCASE("Image") { f.native.outputImage = Native<VkImage>(99); }
    SUBCASE("View") { f.native.outputView = Native<VkImageView>(99); }
    SUBCASE("Texture") { f.native.output.texture = Native<const Texture*>(99); }
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    CHECK_FALSE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty()); CHECK_FALSE(output.handle.IsValid());
}
TEST_CASE("DiUpscaleFeature: recording retains prepared state independently of packet lifetime") {
    Fixture f; auto state = std::make_shared<DiUpscaleViewState>(); std::weak_ptr<DiUpscaleViewState> retained = state;
    f.native.retained = state; state.reset(); Memory::LinearAllocator scratch(128 * 1024);
    { RG::RenderGraph graph(scratch); GraphTextureRef output; REQUIRE(f.Build(graph, scratch, output).success);
      f.native.retained.reset(); f.native.pipeline = VK_NULL_HANDLE; f.native.phiDepth = 9; CHECK_FALSE(retained.expired()); }
    CHECK(retained.expired());
}
TEST_CASE("DiUpscaleFeature: compiler orders both denoisers and rejects missing or duplicate contracts") {
    PipelineInputContract inputs;
    inputs.resources = {{DiUpscaleResources::Bindings}, {DiUpscaleResources::SpecularBindings},
        {DiDenoiserResources::Bindings}, {DiDenoiserResources::SpecularBindings},
        {RestirDiResources::Diffuse, ResourceOutputPresence::Optional}, {RestirDiResources::Specular, ResourceOutputPresence::Optional},
        {RenderResources::SurfaceDepth}, {RenderResources::Normal}, {RenderResources::MotionVectors}, {RenderResources::MaterialID}, {RenderResources::Roughness}};
    RenderPipelineDefinition definition;
    const auto upSpec = definition.AddFeature<DiUpscaleFeature>(DiDenoiserSignal::Specular);
    const auto upDi = definition.AddFeature<DiUpscaleFeature>();
    const auto denoiseSpec = definition.AddFeature<DiDenoiserFeature>(DiDenoiserSignal::Specular);
    const auto denoiseDi = definition.AddFeature<DiDenoiserFeature>();
    auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
    REQUIRE(result.pipeline);
    const auto& order = result.pipeline->FeatureOrder();
    CHECK(std::find(order.begin(), order.end(), denoiseSpec) < std::find(order.begin(), order.end(), upSpec));
    CHECK(std::find(order.begin(), order.end(), denoiseDi) < std::find(order.begin(), order.end(), upDi));
    auto missing = Fixture::Inputs(); missing.resources.pop_back();
    RenderPipelineDefinition invalid; invalid.AddFeature<DiUpscaleFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(invalid), {}, missing).pipeline);
    RenderPipelineDefinition duplicate; duplicate.AddFeature<DiUpscaleFeature>(); duplicate.AddFeature<DiUpscaleFeature>();
    CHECK_FALSE(RenderPipelineCompiler{}.Compile(std::move(duplicate), {}, Fixture::Inputs()).pipeline);
    CHECK_THROWS_AS(DiUpscaleFeature(static_cast<DiDenoiserSignal>(2)), std::invalid_argument);
}

TEST_CASE("DiUpscaleFeature: both channels share surface handles and isolate optional signals") {
    Fixture f; auto spec = f.native; spec.signal = DiDenoiserSignal::Specular;
    spec.sources[0] = {Native<const Texture*>(17)}; spec.sourceImages[0] = Native<VkImage>(27); spec.sourceViews[0] = Native<VkImageView>(37);
    spec.output = {Native<const Texture*>(47)}; spec.outputImage = Native<VkImage>(57); spec.outputView = Native<VkImageView>(67);
    bool enabled = true; SUBCASE("Both") {} SUBCASE("Specular absent") { enabled = false; }
    RenderPipelineDefinition definition; definition.AddFeature<DiUpscaleFeature>(); definition.AddFeature<DiUpscaleFeature>(DiDenoiserSignal::Specular);
    auto inputs = Fixture::Inputs(); inputs.resources.push_back({DiUpscaleResources::SpecularBindings});
    inputs.resources.push_back({DiDenoiserResources::Specular, ResourceOutputPresence::Optional});
    auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs); REQUIRE(compiled.pipeline);
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    std::array<GraphTextureRef, 4> textures;
    const std::array formats{RG::TextureFormat::RGBA16_Float, RG::TextureFormat::D32_Float, RG::TextureFormat::RG16_Float, RG::TextureFormat::RGBA16_Float};
    for (u32 i = 0; i < 4; ++i) {
        RG::TextureDesc desc; desc.width = i == 1 || i == 2 ? 1279 : 639; desc.height = i == 1 || i == 2 ? 719 : 359; desc.format = formats[i];
        const u32 source = i == 3 ? 7 : i;
        textures[i] = {graph.ImportResource(desc, (void*)Native<VkImage>(20 + source), (void*)Native<VkImageView>(30 + source),
            RG::ResourceState::ComputeWrite), {Native<const Texture*>(10 + source)}};
    }
    const DiUpscaleBindingRef diffuseRef{&f.native}, specularRef{&spec};
    const std::array resources{RenderInputBinding::Present(DiUpscaleResources::Bindings, diffuseRef),
        RenderInputBinding::Present(DiUpscaleResources::SpecularBindings, specularRef),
        RenderInputBinding::Present(DiDenoiserResources::Diffuse, textures[0]),
        enabled ? RenderInputBinding::Present(DiDenoiserResources::Specular, textures[3]) : RenderInputBinding::Absent(DiDenoiserResources::Specular),
        RenderInputBinding::Present(RenderResources::SurfaceDepth, textures[1]), RenderInputBinding::Present(RenderResources::Normal, textures[2])};
    FrameRenderInputs frame; frame.resources = resources; frame.renderFrameIndex = f.native.frameIndex;
    ViewRenderInputs view; view.id = {1}; view.resourceGeneration = 3; view.width = 1279; view.height = 719;
    GraphTextureRef diffuse, specular{{99, 0}, {Native<const Texture*>(100)}};
    const std::array exports{RenderOutputBinding::Capture(DiUpscaleResources::Diffuse, diffuse),
        RenderOutputBinding::Capture(DiUpscaleResources::Specular, specular)};
    REQUIRE(compiled.pipeline->Build(graph, frame, view, scratch, exports).success);
    CHECK(diffuse.handle.IsValid()); CHECK(graph.GetPasses().size() == (enabled ? 2 : 1));
    if (enabled) {
        CHECK(specular.handle.IsValid()); CHECK(diffuse.handle.index != specular.handle.index);
        CHECK(graph.GetPasses()[0].reads[1].index == graph.GetPasses()[1].reads[1].index);
        CHECK(graph.GetPasses()[0].reads[2].index == graph.GetPasses()[1].reads[2].index);
    } else { CHECK_FALSE(specular.handle.IsValid()); CHECK(specular.binding.texture == nullptr); }
}