#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/SlimVizFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/memory/LinearAllocator.h"
#include <limits>
using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        PostProcessSubsystem native;
        SlimVizBindings bindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.pipeline = Native<VkPipeline>(1);
            bindings.layout = Native<VkPipelineLayout>(2); bindings.set = Native<VkDescriptorSet>(3);
            bindings.parameters = {0, 20.0f};
            for (u32 i = 0; i < 4; ++i)
                bindings.sources[i] = std::shared_ptr<Texture>(Native<Texture*>(10 + i), [](Texture*){});
            RenderPipelineDefinition definition; definition.AddFeature<SlimVizFeature>(native);
            PipelineInputContract inputs; inputs.resources = {{RenderResources::TonemappedLDR}, {SlimVizResources::Bindings},
                {RenderResources::Normal, ResourceOutputPresence::Optional}, {RenderResources::Roughness, ResourceOutputPresence::Optional},
                {RenderResources::MotionVectors, ResourceOutputPresence::Optional}, {RenderResources::MaterialID, ResourceOutputPresence::Optional}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& output,
            bool present = true, bool packetPresent = true, u32 width = 640, bool producer = false, bool wrongFormat = false)
        {
            RG::TextureDesc desc; desc.width = width; desc.height = 480; desc.format = RG::TextureFormat::RGBA8_Unorm;
            const GraphTextureRef color{graph.ImportResource(desc, (void*)20, (void*)21, RG::ResourceState::ShaderResource),
                {Native<const Texture*>(9)}};
            const std::array formats{RG::TextureFormat::RG16_Float, RG::TextureFormat::R8_Unorm,
                RG::TextureFormat::RG16_Float, RG::TextureFormat::R16_Uint};
            std::array<GraphTextureRef, 4> sources;
            for (u32 i = 0; i < 4; ++i)
            {
                desc.format = wrongFormat && i == 3 ? RG::TextureFormat::R8_Unorm : formats[i];
                sources[i] = {graph.ImportResource(desc, (void*)(uintptr_t)(22 + i * 2), (void*)(uintptr_t)(23 + i * 2), RG::ResourceState::Undefined),
                    {Native<const Texture*>(10 + i)}};
            }
            if (producer)
            {
                struct Data { std::array<RG::ResourceHandle, 4> outputs; };
                graph.AddPass<Data>("SurfaceProducer", [&](Data& data, RG::RenderPassBuilder& builder) {
                    for (u32 i = 0; i < 4; ++i) sources[i].handle = data.outputs[i] = builder.Write(sources[i].handle);
                }, [](Data&, RG::RenderPassContext&){});
            }
            const SlimVizBindingRef binding{packetPresent ? &bindings : nullptr};
            const auto optional = [&](auto key, const GraphTextureRef& ref) {
                return present ? RenderInputBinding::Present(key, ref) : RenderInputBinding::Absent(key);
            };
            const std::array inputs{RenderInputBinding::Present(RenderResources::TonemappedLDR, color),
                RenderInputBinding::Present(SlimVizResources::Bindings, binding), optional(RenderResources::Normal, sources[0]),
                optional(RenderResources::Roughness, sources[1]), optional(RenderResources::MotionVectors, sources[2]),
                optional(RenderResources::MaterialID, sources[3])};
            FrameRenderInputs frame; frame.resources = inputs; ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::VisualizedLDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("SlimVizFeature: all decoding modes reuse LDR and G-buffer imports [renderfeatures]")
{
    for (u32 mode = 0; mode < 4; ++mode)
    {
        Fixture f; f.bindings.parameters.mode = mode; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef output; REQUIRE(f.Build(graph, scratch, output).success);
        REQUIRE(graph.GetPasses().size() == 1); CHECK(graph.GetResources().size() == 5);
        const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "SlimVizPass"); CHECK_FALSE(pass.isCompute);
        CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK_FALSE(pass.hasDepth);
        REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
        CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
        CHECK(pass.colorAttachments[0].clearValue.color.float32[0] == 0); CHECK(pass.colorAttachments[0].clearValue.color.float32[3] == 1);
        REQUIRE(pass.reads.size() == 4); for (u32 i = 0; i < 4; ++i) CHECK(pass.reads[i].index == i + 2);
        CHECK(output.handle.index == 1); CHECK(output.handle.version == 1); CHECK(output.binding.texture == Native<const Texture*>(9));
        graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
    }
}
TEST_CASE("SlimVizFeature: disabled PT and cold pipeline publish graph-local LDR aliases [renderfeatures]")
{
    Fixture f; bool present = true;
    SUBCASE("disabled") { f.bindings.enabled = false; f.bindings.sources = {}; }
    SUBCASE("PT without G-buffer") { f.bindings.enabled = false; present = false; }
    SUBCASE("cold") { f.bindings.pipeline = VK_NULL_HANDLE; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
        REQUIRE(f.Build(graph, scratch, output, present).success); CHECK(graph.GetPasses().empty());
        CHECK(output.handle.index == 1); CHECK(output.handle.version == 0); CHECK(output.binding.texture == Native<const Texture*>(9));
    }
}
TEST_CASE("SlimVizFeature: invalid native inputs and numerical contracts reject before registration [renderfeatures]")
{
    Fixture f; bool present = true, packet = true, wrongFormat = false; u32 width = 640;
    SUBCASE("packet") { packet = false; }
    SUBCASE("layout") { f.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("set") { f.bindings.set = VK_NULL_HANDLE; }
    SUBCASE("missing sources") { present = false; }
    SUBCASE("source owner") { f.bindings.sources[2].reset(); }
    SUBCASE("source mismatch") { f.bindings.sources[1] = f.bindings.sources[0]; }
    SUBCASE("extent") { width = 641; }
    SUBCASE("integer material format") { wrongFormat = true; }
    SUBCASE("mode") { f.bindings.parameters.mode = 4; }
    SUBCASE("scale") { f.bindings.parameters.scale = std::numeric_limits<float>::quiet_NaN(); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
    CHECK_FALSE(f.Build(graph, scratch, output, present, packet, width, false, wrongFormat).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 5);
}
TEST_CASE("SlimVizFeature: frozen jobs retain sources and read producer versions with barriers [renderfeatures]")
{
    Fixture f; std::weak_ptr<Texture> retained = f.bindings.sources[3];
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(f.Build(graph, scratch, output, true, true, 640, true).success); f.bindings.sources = {};
        CHECK_FALSE(retained.expired()); REQUIRE(graph.GetPasses().size() == 2);
        for (const auto& read : graph.GetPasses()[1].reads) CHECK(read.version == 1);
        graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[1].culled);
        u32 transitions = 0; for (const auto& barrier : graph.GetPasses()[1].preBarriers)
            if (barrier.resource.index > 1 && barrier.after == RG::ResourceState::ShaderResource) ++transitions;
        CHECK(transitions == 4);
    }
    CHECK(retained.expired());
}
TEST_CASE("SlimVizFeature: disabled and cold preparation need no native device [renderfeatures]")
{
    PostProcessSubsystem native;
    const auto disabled = native.PrepareSlimVizBindings(VK_NULL_HANDLE, {}, 99, 20, false);
    CHECK_FALSE(disabled.enabled); CHECK_FALSE(disabled.pipeline); CHECK_FALSE(disabled.sources[0]);
    const auto cold = native.PrepareSlimVizBindings(VK_NULL_HANDLE, {}, 0, 20, true);
    CHECK(cold.enabled); CHECK_FALSE(cold.pipeline); CHECK_FALSE(cold.sources[0]);
}
