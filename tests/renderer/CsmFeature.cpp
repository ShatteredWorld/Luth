#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/CsmFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/LightingSubsystem.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        LightingSubsystem native;
        CsmBindings bindings;
        DrawList draws;
        RenderSnapshot snapshot;
        Memory::GPUSubRegion slice{Native<VkBuffer>(1), 8192, 819200};
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            bindings.rigid = Native<VkPipeline>(2);
            bindings.rigidLayout = Native<VkPipelineLayout>(3);
            bindings.sets.fill(Native<VkDescriptorSet>(4));
            bindings.texture = Native<const Texture*>(5);
            bindings.image = Native<VkImage>(6);
            for (u32 i = 0; i < 4; ++i) bindings.layers[i] = Native<VkImageView>(7 + i);
            RenderPipelineDefinition definition;
            definition.AddFeature<CsmFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{CsmResources::Parameters}, {CsmResources::Bindings},
                {RenderResources::CascadeVisibleDraws, ResourceOutputPresence::Optional}};
            inputs.capabilities = {&DeformationResources::DeformedGeometry};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline);
            pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            ShadowCascadeRefs& output, bool enabled = true, bool rangesPresent = true,
            bool framePresent = true, u32 viewIndex = 0)
        {
            const auto indirect = graph.ImportBuffer({"Indirect", slice.size, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT},
                (void*)slice.buffer, RG::ResourceState::Undefined);
            CascadeDrawRanges ranges;
            for (u32 i = 0; i < 4; ++i)
                ranges.cascades[i] = {{indirect, {&slice, slice.offset, slice.size}}, (viewIndex * 5 + i + 1) * 4096, 23};
            const CsmParameters params{enabled};
            const CsmBindingRef binding{&bindings};
            const std::array resources{RenderInputBinding::Present(CsmResources::Parameters, params),
                RenderInputBinding::Present(CsmResources::Bindings, binding),
                rangesPresent ? RenderInputBinding::Present(RenderResources::CascadeVisibleDraws, ranges)
                    : RenderInputBinding::Absent(RenderResources::CascadeVisibleDraws)};
            const std::array capabilities{&DeformationResources::DeformedGeometry};
            FrameRenderInputs frame;
            frame.resources = resources; frame.capabilities = capabilities;
            if (framePresent) { frame.draws = &draws; frame.snapshot = &snapshot; }
            ViewRenderInputs view; view.id = {u64(viewIndex) + 1};
            const std::array exports{RenderOutputBinding::Capture(RenderResources::ShadowCascades, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("CsmFeature: four layer imports preserve attachment and indirect contracts [renderfeatures]")
{
    Fixture fixture;
    for (u32 view : {0u, 1u})
    {
        Memory::LinearAllocator scratch(64 * 1024);
        RG::RenderGraph graph(scratch);
        ShadowCascadeRefs output;
        REQUIRE(fixture.Build(graph, scratch, output, true, true, true, view).success);
        REQUIRE(graph.GetPasses().size() == 4);
        REQUIRE(graph.GetResources().size() == 4);
        for (u32 i = 0; i < 4; ++i)
        {
            const auto& pass = graph.GetPasses()[i];
            CHECK(pass.name == "ShadowPass.C" + std::to_string(i));
            CHECK(pass.debugMetadata.shaderName == "shadowDepth");
            CHECK(pass.debugMetadata.pipelineStateAvailable); CHECK(pass.debugMetadata.indirectDraws);
            CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
            CHECK_FALSE(pass.isCompute);
            CHECK(pass.hasDepth);
            CHECK(pass.depthAttachment.handle == output.cascades[i].handle);
            CHECK(pass.depthAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
            CHECK(pass.depthAttachment.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
            CHECK(pass.depthAttachment.clearValue.depthStencil.depth == 1.0f);
            REQUIRE(pass.bufferReads.size() == 1);
            CHECK(pass.bufferReadStates[0] == RG::ResourceState::IndirectRead);
            CHECK(output.cascades[i].binding.texture == fixture.bindings.texture);
            CHECK(output.cascades[i].binding.baseLayer == i);
            CHECK(output.cascades[i].binding.layerCount == 1);
            CHECK(output.cascades[i].handle.version == 1);
        }
        graph.Compile();
        for (const auto& pass : graph.GetPasses()) CHECK_FALSE(pass.culled);
    }
}
TEST_CASE("CsmFeature: disabled output resets without shadow imports or frame preparation [renderfeatures]")
{
    Fixture fixture;
    ShadowCascadeRefs output;
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        REQUIRE(fixture.Build(graph, scratch, output).success);
    }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch, output, false, false, false).success);
    CHECK(graph.GetPasses().empty());
    CHECK(graph.GetResources().empty());
    for (const auto& cascade : output.cascades) CHECK_FALSE(cascade.handle.IsValid());
}
TEST_CASE("CsmFeature: invalid active inputs fail before registering cascades [renderfeatures]")
{
    Fixture fixture;
    bool ranges = true, frame = true;
    SUBCASE("ranges absent") { ranges = false; }
    SUBCASE("frame absent") { frame = false; }
    SUBCASE("last layer absent") { fixture.bindings.layers[3] = VK_NULL_HANDLE; }
    SUBCASE("descriptor absent") { fixture.bindings.sets[0] = VK_NULL_HANDLE; }
    SUBCASE("last range outside slice") { fixture.slice.size = 4 * 4096 * sizeof(VkDrawIndexedIndirectCommand); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    ShadowCascadeRefs output;
    CHECK_FALSE(fixture.Build(graph, scratch, output, true, ranges, frame).success);
    CHECK(graph.GetPasses().empty());
    CHECK(graph.GetResources().empty());
    for (const auto& cascade : output.cascades) CHECK_FALSE(cascade.handle.IsValid());
}
TEST_CASE("CsmFeature: cold pipelines keep cascade clears and descriptor preparation is frozen [renderfeatures]")
{
    Fixture fixture;
    fixture.bindings.rigid = VK_NULL_HANDLE;
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    ShadowCascadeRefs output;
    REQUIRE(fixture.Build(graph, scratch, output).success);
    CHECK(graph.GetPasses().size() == 4);
    std::array<VkDescriptorSet, 6> sets;
    sets.fill(Native<VkDescriptorSet>(11));
    const auto prepared = fixture.native.PrepareCsmBindings(sets, true);
    sets.fill(Native<VkDescriptorSet>(12));
    for (auto set : prepared.sets) CHECK(set == Native<VkDescriptorSet>(11));
    CHECK(prepared.captureDraws);
}
