#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/SortedTransparencyFeature.h"
#include "luth/renderer/features/SortedTransparencyBindings.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/TransparencySubsystem.h"
#include "luth/renderer/draw/DrawList.h"
#include "luth/core/RenderSnapshot.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        TransparencySubsystem native;
        SortedTransparencyBindings bindings;
        Memory::GPUSubRegion region{};
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        GraphTextureRef color, depth, picking;
        Fixture()
        {
            region.buffer = Native<VkBuffer>(1); region.offset = 256; region.size = 20 * sizeof(VkDrawIndexedIndirectCommand);
            bindings.sets.fill(Native<VkDescriptorSet>(2));
            ForwardDrawPacket draw; draw.pipeline = Native<VkPipeline>(3); draw.layout = Native<VkPipelineLayout>(4);
            draw.vertex = Native<VkBuffer>(5); draw.index = Native<VkBuffer>(6); draw.objectIndex = 2;
            draw.indirectOffset = region.offset + 6 * sizeof(VkDrawIndexedIndirectCommand);
            bindings.draws.push_back(draw);
            RenderPipelineDefinition definition; definition.AddFeature<SortedTransparencyFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{TransparencyResources::Bindings}, {RenderResources::FoggedHDR}, {RenderResources::LitDepth},
                {RenderResources::OpaquePickingIDs}, {RenderResources::CameraVisibleDraws},
                {RenderResources::ResolvedFog, ResourceOutputPresence::Optional}, {RenderResources::RefractionBackdrop, ResourceOutputPresence::Optional},
                {RenderResources::LightData, ResourceOutputPresence::Optional}, {RenderResources::ClusterGrid, ResourceOutputPresence::Optional},
                {RenderResources::LightIndices, ResourceOutputPresence::Optional}};
            inputs.capabilities = {&DeformationResources::DeformedGeometry};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch, bool sampled = false,
            bool packet = true, bool invalidExtent = false, bool lighting = false, bool mismatchedSample = false)
        {
            auto image = [&](u32 index, RG::TextureFormat format) {
                RG::TextureDesc desc; desc.name = "Stage"; desc.width = invalidExtent ? 641 : 640; desc.height = 480; desc.format = format;
                return GraphTextureRef{graph.ImportResource(desc, (void*)(uintptr_t)(index * 10), (void*)(uintptr_t)(index * 10 + 1),
                    RG::ResourceState::ShaderResource), {Native<const Texture*>(index * 10 + 2)}};
            };
            auto source = image(1, RG::TextureFormat::RGBA16_Float);
            auto depthSource = image(2, RG::TextureFormat::D32_Float);
            auto ids = image(3, RG::TextureFormat::R32_Uint);
            auto fog = image(4, RG::TextureFormat::RGBA16_Float); auto backdrop = image(5, RG::TextureFormat::RGBA16_Float);
            if (sampled) { bindings.fog = fog.binding; bindings.backdrop = backdrop.binding; }
            if (mismatchedSample) bindings.backdrop.texture = Native<const Texture*>(99);
            RG::BufferDesc desc; desc.name = "Indirect"; desc.size = region.size;
            const VisibleDrawRange visible{{graph.ImportBuffer(desc, (void*)region.buffer, RG::ResourceState::StorageBufferWrite),
                {&region, region.offset, region.size}}, 4, 8};
            GraphBufferRef lights;
            if (lighting) lights = {graph.ImportBuffer(desc, (void*)region.buffer, RG::ResourceState::StorageBufferWrite),
                {&region, region.offset, region.size}};
            const SortedTransparencyBindingRef binding{packet ? &bindings : nullptr};
            const std::array resources{RenderInputBinding::Present(TransparencyResources::Bindings, binding),
                RenderInputBinding::Present(RenderResources::FoggedHDR, source), RenderInputBinding::Present(RenderResources::LitDepth, depthSource),
                RenderInputBinding::Present(RenderResources::OpaquePickingIDs, ids), RenderInputBinding::Present(RenderResources::CameraVisibleDraws, visible),
                sampled ? RenderInputBinding::Present(RenderResources::ResolvedFog, fog) : RenderInputBinding::Absent(RenderResources::ResolvedFog),
                sampled ? RenderInputBinding::Present(RenderResources::RefractionBackdrop, backdrop) : RenderInputBinding::Absent(RenderResources::RefractionBackdrop),
                lighting ? RenderInputBinding::Present(RenderResources::LightData, lights) : RenderInputBinding::Absent(RenderResources::LightData), RenderInputBinding::Absent(RenderResources::ClusterGrid),
                RenderInputBinding::Absent(RenderResources::LightIndices)};
            const std::array caps{&DeformationResources::DeformedGeometry};
            FrameRenderInputs frame; frame.resources = resources; frame.capabilities = caps;
            ViewRenderInputs view; view.id = {1}; view.width = 640; view.height = 480;
            const std::array outputs{RenderOutputBinding::Capture(RenderResources::TransparentHDR, color),
                RenderOutputBinding::Capture(RenderResources::FinalPickingIDs, picking),
                RenderOutputBinding::Capture(TransparencyResources::Depth, depth)};
            return pipeline->Build(graph, frame, view, scratch, outputs);
        }
    };
}
TEST_CASE("SortedTransparencyFeature: native attachments expose actual color picking depth outputs [renderfeatures]")
{
    Fixture fixture; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch, true).success); REQUIRE(graph.GetPasses().size() == 1);
    CHECK(graph.GetResources().size() == 5); CHECK(graph.GetBuffers().size() == 1);
    const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "TransparentPass");
    CHECK_FALSE(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    REQUIRE(pass.colorAttachments.size() == 2);
    CHECK(pass.colorAttachments[0].handle == fixture.color.handle); CHECK(pass.colorAttachments[1].handle == fixture.picking.handle);
    for (const auto& attachment : pass.colorAttachments) { CHECK(attachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD); CHECK(attachment.storeOp == VK_ATTACHMENT_STORE_OP_STORE); }
    CHECK(pass.hasDepth); CHECK(pass.depthAttachment.handle == fixture.depth.handle);
    CHECK(pass.depthAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD); CHECK(pass.depthAttachment.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK(fixture.color.handle.index == 1); CHECK(fixture.depth.handle.index == 2); CHECK(fixture.picking.handle.index == 3);
    CHECK(fixture.color.handle.version == 1); CHECK(fixture.depth.handle.version == 1); CHECK(fixture.picking.handle.version == 1);
    REQUIRE(pass.reads.size() == 2); CHECK(pass.reads[0].index == 4); CHECK(pass.reads[1].index == 5);
    REQUIRE(pass.bufferReads.size() == 1); CHECK(pass.bufferReads[0].index == 1);
    CHECK(pass.bufferReadStates[0] == RG::ResourceState::IndirectRead);
    graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
}
TEST_CASE("SortedTransparencyFeature: empty packets pass all stages through per graph [renderfeatures]")
{
    Fixture fixture; fixture.bindings.draws.clear(); fixture.bindings.sets = {};
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        REQUIRE(fixture.Build(graph, scratch).success); CHECK(graph.GetPasses().empty());
        CHECK(fixture.color.handle.index == 1); CHECK(fixture.picking.handle.index == 3); CHECK(fixture.depth.handle.index == 2);
        CHECK(fixture.color.handle.version == 0); CHECK(fixture.picking.handle.version == 0); CHECK(fixture.depth.handle.version == 0);
    }
}
TEST_CASE("SortedTransparencyFeature: lighting uses fragment storage reads and deformed draws need no vertex binding [renderfeatures]")
{
    Fixture fixture; fixture.bindings.draws[0].deformed = true; fixture.bindings.draws[0].vertex = VK_NULL_HANDLE;
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch, false, true, false, true).success);
    REQUIRE(graph.GetPasses().size() == 1);
    const auto& pass = graph.GetPasses()[0]; REQUIRE(pass.bufferReads.size() == 2);
    CHECK(pass.bufferReads[0].index == 2); CHECK(pass.bufferReadStates[0] == RG::ResourceState::FragmentStorageRead);
    CHECK(pass.bufferReads[1].index == 1); CHECK(pass.bufferReadStates[1] == RG::ResourceState::IndirectRead);
}
TEST_CASE("SortedTransparencyFeature: rejects sampled descriptors that differ from declared resources [renderfeatures]")
{
    Fixture fixture; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    CHECK_FALSE(fixture.Build(graph, scratch, true, true, false, false, true).success);
    CHECK(graph.GetPasses().empty()); CHECK_FALSE(fixture.color.handle.IsValid());
}
TEST_CASE("SortedTransparencyFeature: rejects invalid stage and prepared slices before registration [renderfeatures]")
{
    Fixture fixture; bool packet = true, extent = false;
    SUBCASE("packet") { packet = false; }
    SUBCASE("extent") { extent = true; }
    SUBCASE("descriptor") { fixture.bindings.sets[6] = VK_NULL_HANDLE; }
    SUBCASE("draw layout") { fixture.bindings.draws[0].layout = VK_NULL_HANDLE; }
    SUBCASE("draw index") { fixture.bindings.draws[0].index = VK_NULL_HANDLE; }
    SUBCASE("draw vertex") { fixture.bindings.draws[0].vertex = VK_NULL_HANDLE; }
    SUBCASE("slice too short") { fixture.region.size = sizeof(VkDrawIndexedIndirectCommand); }
    SUBCASE("object outside range") { fixture.bindings.draws[0].objectIndex = 8; }
    SUBCASE("offset mismatch") { fixture.bindings.draws[0].indirectOffset++; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    CHECK_FALSE(fixture.Build(graph, scratch, false, packet, extent).success); CHECK(graph.GetPasses().empty());
    CHECK_FALSE(fixture.color.handle.IsValid()); CHECK_FALSE(fixture.picking.handle.IsValid()); CHECK_FALSE(fixture.depth.handle.IsValid());
}
TEST_CASE("SortedTransparencyFeature: sorting is view specific and never mutates shared bucket [renderfeatures]")
{
    DrawList draws; draws.transparent.resize(3);
    for (auto& draw : draws.transparent) draw.modelMatrix = Mat4(1);
    draws.transparent[0].modelMatrix[3].z = -2; draws.transparent[1].modelMatrix[3].z = -9; draws.transparent[2].modelMatrix[3].z = -5;
    const auto scene = TransparencySubsystem::SortedOrder(draws, Mat4(1));
    REQUIRE(scene.size() == 3); CHECK(scene[0] == 1); CHECK(scene[1] == 2); CHECK(scene[2] == 0);
    Mat4 reverse(1); reverse[2][2] = -1;
    const auto game = TransparencySubsystem::SortedOrder(draws, reverse);
    CHECK(game[0] == 0); CHECK(game[1] == 2); CHECK(game[2] == 1);
    CHECK(draws.transparent[0].modelMatrix[3].z == -2); CHECK(draws.transparent[1].modelMatrix[3].z == -9);
    CHECK(draws.transparent[2].modelMatrix[3].z == -5);
    CHECK(TransparencySubsystem::SortedOrder(DrawList{}, Mat4(1)).empty());
}
TEST_CASE("SortedTransparencyFeature: empty native preparation needs no Vulkan device [renderfeatures]")
{
    TransparencySubsystem native; GeometrySubsystem geometry; DrawList draws; RenderSnapshot snapshot;
    const auto packet = native.PrepareSortedBindings(geometry, {}, false, true, Mat4(1), {}, draws, snapshot, {}, {}, nullptr);
    CHECK(packet.draws.empty()); CHECK(packet.captureDraws); CHECK_FALSE(packet.rayScene);
}
