#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/DebugDrawFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/DebugDrawSubsystem.h"
#include "luth/memory/LinearAllocator.h"
#include <limits>
using namespace Luth;
namespace
{
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    struct Fixture
    {
        DebugDrawSubsystem native; DebugDrawBindings bindings; std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.enabled = true; bindings.pipeline = Native<VkPipeline>(1); bindings.layout = Native<VkPipelineLayout>(2);
            bindings.vertices = {Native<VkBuffer>(3), 4096, 4 * sizeof(DebugVertex)}; bindings.vertexCount = 4;
            bindings.renderFrameIndex = 17;
            RenderPipelineDefinition definition; definition.AddFeature<DebugDrawFeature>(native);
            PipelineInputContract inputs; inputs.resources = {{RenderResources::OutlinedLDR}, {DebugDrawResources::Bindings}};
            auto result = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs); REQUIRE(result.pipeline);
            pipeline = std::move(result.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, GraphTextureRef& output,
            bool packet = true, u32 width = 640, RG::TextureFormat format = RG::TextureFormat::RGBA8_Unorm, bool producer = false)
        {
            RG::TextureDesc desc; desc.width = width; desc.height = 480; desc.format = format;
            GraphTextureRef color{graph.ImportResource(desc, (void*)20, (void*)21, RG::ResourceState::ShaderResource), {Native<const Texture*>(9)}};
            if (producer)
            {
                struct Data { RG::ResourceHandle output; };
                graph.AddPass<Data>("OutlinePass", [&](Data& data, RG::RenderPassBuilder& builder) { color.handle = data.output = builder.Write(color.handle); },
                    [](Data&, RG::RenderPassContext&){});
            }
            const DebugDrawBindingRef ref{packet ? &bindings : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::OutlinedLDR, color),
                RenderInputBinding::Present(DebugDrawResources::Bindings, ref)};
            FrameRenderInputs frame; frame.renderFrameIndex = 17; frame.resources = resources;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::FinalViewLDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("DebugDrawFeature: final LDR lines preserve native load store and registration order [renderfeatures]")
{
    Fixture f; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(f.Build(graph, scratch, output, true, 640, RG::TextureFormat::RGBA8_Unorm, true).success);
    REQUIRE(graph.GetPasses().size() == 2); CHECK(graph.GetPasses()[0].name == "OutlinePass");
    const auto& pass = graph.GetPasses()[1]; CHECK(pass.name == "DebugDrawPass"); CHECK_FALSE(pass.isCompute);
    CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK_FALSE(pass.hasDepth); CHECK(pass.reads.empty());
    REQUIRE(pass.colorAttachments.size() == 1); CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE); CHECK(output.handle.index == 1); CHECK(output.handle.version == 2);
    CHECK(output.binding.texture == Native<const Texture*>(9)); CHECK(graph.GetResources().size() == 1); CHECK(graph.GetBuffers().empty());
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled); CHECK_FALSE(graph.GetPasses()[1].culled);
    bool waw = false; for (const auto& barrier : graph.GetPasses()[1].preBarriers) if (barrier.reason == RG::BarrierReason::Waw) waw = true;
    CHECK(waw);
}
TEST_CASE("DebugDrawFeature: disabled PT cold and empty lines publish fresh aliases [renderfeatures]")
{
    Fixture f;
    SUBCASE("disabled view") { f.bindings.enabled = false; }
    SUBCASE("PT") { f.bindings.enabled = false; f.bindings.vertices = {}; }
    SUBCASE("cold") { f.bindings.pipeline = VK_NULL_HANDLE; }
    SUBCASE("empty") { f.bindings.vertexCount = 0; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{55, 1}, {}};
        REQUIRE(f.Build(graph, scratch, output).success); CHECK(graph.GetPasses().empty()); CHECK(output.handle.index == 1); CHECK(output.handle.version == 0);
    }
}
TEST_CASE("DebugDrawFeature: invalid frozen upload and matrix reject before registration [renderfeatures]")
{
    Fixture f; bool packet = true; u32 width = 640; auto format = RG::TextureFormat::RGBA8_Unorm;
    SUBCASE("packet") { packet = false; }
    SUBCASE("extent") { ++width; }
    SUBCASE("format") { format = RG::TextureFormat::RGBA16_Float; }
    SUBCASE("layout") { f.bindings.layout = VK_NULL_HANDLE; }
    SUBCASE("buffer") { f.bindings.vertices.buffer = VK_NULL_HANDLE; }
    SUBCASE("short upload") { --f.bindings.vertices.size; }
    SUBCASE("unaligned slice") { ++f.bindings.vertices.offset; }
    SUBCASE("odd endpoint count") { f.bindings.vertexCount = 3; }
    SUBCASE("wrong render frame") { ++f.bindings.renderFrameIndex; }
    SUBCASE("matrix") { f.bindings.viewProj[2][1] = std::numeric_limits<float>::quiet_NaN(); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    CHECK_FALSE(f.Build(graph, scratch, output, packet, width, format).success); CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty());
}
TEST_CASE("DebugDrawFeature: disabled and cold preparation do not require upload context [renderfeatures]")
{
    DebugDrawSubsystem native; const std::array<DebugVertex, 2> lines{};
    CHECK_FALSE(native.PrepareBindings(lines, Mat4(1), 17, false).enabled);
    const auto cold = native.PrepareBindings(lines, Mat4(1), 17, true);
    CHECK(cold.enabled); CHECK_FALSE(cold.pipeline); CHECK_FALSE(cold.vertices.buffer); CHECK(cold.vertexCount == 0);
    const auto empty = native.PrepareBindings({}, Mat4(1), 17, true); CHECK_FALSE(empty.vertices.buffer);
}
