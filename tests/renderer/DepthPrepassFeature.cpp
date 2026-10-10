#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/DepthPrepassFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/GeometrySubsystem.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
using namespace Luth::RenderResources;

namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        GeometrySubsystem native;
        DepthPrepassBindings nativeBindings;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        DrawList draws;
        RenderSnapshot snapshot;
        Memory::GPUSubRegion slice{Native<VkBuffer>(1), 8192, 819200};
        Fixture()
        {
            nativeBindings.rigid = Native<VkPipeline>(2);
            nativeBindings.rigidLayout = Native<VkPipelineLayout>(3);
            nativeBindings.sets.fill(Native<VkDescriptorSet>(4));
            RenderPipelineDefinition definition;
            definition.AddFeature<DepthPrepassFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{CameraVisibleDraws}, {DepthPrepassResources::Target}, {DepthPrepassResources::Bindings}};
            inputs.capabilities = {&DeformationResources::DeformedGeometry};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, u32 viewIndex = 0, u32 targetWidth = 640,
            RG::TextureFormat format = RG::TextureFormat::D32_Float, bool framePresent = true,
            bool capabilityPresent = true, bool bindingPresent = true)
        {
            RG::TextureDesc desc;
            desc.name = "SceneDepth"; desc.width = targetWidth; desc.height = 480; desc.format = format;
            const GraphTextureRef target{graph.ImportResource(desc, (void*)5, (void*)6, RG::ResourceState::Undefined),
                {Native<const Texture*>(7)}};
            const auto indirect = graph.ImportBuffer({"Indirect", slice.size, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT},
                (void*)slice.buffer, RG::ResourceState::Undefined);
            const VisibleDrawRange visible{{indirect, {&slice, slice.offset, slice.size}}, viewIndex * 5 * 4096, 23};
            const DepthPrepassBindingRef binding{bindingPresent ? &nativeBindings : nullptr};
            const std::array resources{RenderInputBinding::Present(CameraVisibleDraws, visible),
                RenderInputBinding::Present(DepthPrepassResources::Target, target),
                RenderInputBinding::Present(DepthPrepassResources::Bindings, binding)};
            const std::array capabilities{&DeformationResources::DeformedGeometry};
            FrameRenderInputs frame;
            if (framePresent) { frame.draws = &draws; frame.snapshot = &snapshot; }
            frame.resources = resources;
            if (capabilityPresent) frame.capabilities = capabilities;
            ViewRenderInputs view;
            view.id = {u64(viewIndex) + 1}; view.width = 640; view.height = 480;
            const std::array exports{RenderOutputBinding::Capture(PrepassDepth, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}

TEST_CASE("DepthPrepassFeature: native attachment and indirect read preserve graph contracts [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output;
    REQUIRE(fixture.Build(graph, scratch, output).success);
    REQUIRE(graph.GetPasses().size() == 1);
    const auto& pass = graph.GetPasses()[0];
    CHECK(pass.name == "DepthPrepass");
    CHECK_FALSE(pass.isCompute);
    CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    CHECK(pass.hasDepth);
    CHECK(pass.colorAttachments.empty());
    CHECK(pass.depthAttachment.handle == output.handle);
    CHECK(pass.depthAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
    CHECK(pass.depthAttachment.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK(pass.depthAttachment.clearValue.depthStencil.depth == 1.0f);
    REQUIRE(pass.bufferReads.size() == 1);
    CHECK(pass.bufferReadStates[0] == RG::ResourceState::IndirectRead);
    CHECK(output.handle.index == 1);
    CHECK(output.handle.version == 1);
    CHECK(output.binding.texture == Native<const Texture*>(7));
    CHECK(graph.GetResources().size() == 1); // Borrow imported target; never re-import.
    graph.Compile();
    CHECK_FALSE(graph.GetPasses()[0].culled);
}

TEST_CASE("DepthPrepassFeature: two-view graphs remain independent and failures reset exports [renderfeatures]")
{
    Fixture fixture;
    GraphTextureRef output;
    for (u32 view : {0u, 1u})
    {
        Memory::LinearAllocator scratch(64 * 1024);
        RG::RenderGraph graph(scratch);
        REQUIRE(fixture.Build(graph, scratch, output, view).success);
        CHECK(graph.GetPasses().size() == 1);
    }
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    CHECK_FALSE(fixture.Build(graph, scratch, output, 0, 639).success);
    CHECK_FALSE(output.handle.IsValid());
    CHECK(graph.GetPasses().empty());
}

TEST_CASE("DepthPrepassFeature: invalid inputs reject before native registration [renderfeatures]")
{
    Fixture fixture;
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output;
    bool frame = true, capability = true, binding = true;
    auto format = RG::TextureFormat::D32_Float;
    SUBCASE("frame missing") { frame = false; }
    SUBCASE("capability missing") { capability = false; }
    SUBCASE("native packet missing") { binding = false; }
    SUBCASE("wrong target format") { format = RG::TextureFormat::R8_Unorm; }
    SUBCASE("range outside slice") { fixture.slice.size = 16; }
    SUBCASE("descriptor missing") { fixture.nativeBindings.sets[0] = VK_NULL_HANDLE; }
    const auto result = fixture.Build(graph, scratch, output, 0, 640, format, frame, capability, binding);
    CHECK_FALSE(result.success);
    CHECK_FALSE(result.diagnostics.empty());
    CHECK_FALSE(output.handle.IsValid());
    CHECK(graph.GetPasses().empty());
}

TEST_CASE("DepthPrepassFeature: cold native pipeline still declares a depth clear [renderfeatures]")
{
    Fixture fixture;
    fixture.nativeBindings = {};
    Memory::LinearAllocator scratch(64 * 1024);
    RG::RenderGraph graph(scratch);
    GraphTextureRef output;
    REQUIRE(fixture.Build(graph, scratch, output).success);
    REQUIRE(graph.GetPasses().size() == 1);
    CHECK(graph.GetPasses()[0].depthAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
    CHECK(output.handle.IsValid()); // Actual attachment clear remains a producer.
}

TEST_CASE("DepthPrepassFeature: native preparation freezes supplied descriptor slots [renderfeatures]")
{
    GeometrySubsystem native;
    std::array<VkDescriptorSet, 6> sets;
    sets.fill(Native<VkDescriptorSet>(11));
    const auto prepared = native.PrepareDepthPrepassBindings(sets, true);
    sets.fill(Native<VkDescriptorSet>(22));
    for (const auto set : prepared.sets) CHECK(set == Native<VkDescriptorSet>(11));
    CHECK(prepared.captureDraws);
}
