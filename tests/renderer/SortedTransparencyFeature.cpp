#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/features/TransparencyFeature.h"
#include "luth/renderer/features/TransparencyBindings.h"
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
        TransparencyBindings bindings;
        Memory::GPUSubRegion region{};
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        GraphTextureRef color, depth, picking;
        void EnableOit()
        {
            bindings.oit = true; bindings.headsImage = Native<VkImage>(101); bindings.headsView = Native<VkImageView>(102);
            bindings.heads = {Native<const Texture*>(103)}; bindings.width = 640; bindings.height = 480;
            bindings.nodes.buffer = Native<VkBuffer>(104); bindings.nodes.offset = 512; bindings.nodes.size = 16 + 16 * 640 * 480;
            bindings.capacity = 640 * 480; bindings.resolvePipeline = Native<VkPipeline>(105);
            bindings.resolveLayout = Native<VkPipelineLayout>(106); bindings.resolveSet = Native<VkDescriptorSet>(107);
        }
        Fixture()
        {
            region.buffer = Native<VkBuffer>(1); region.offset = 256; region.size = 20 * sizeof(VkDrawIndexedIndirectCommand);
            bindings.sets.fill(Native<VkDescriptorSet>(2));
            ForwardDrawPacket draw; draw.pipeline = Native<VkPipeline>(3); draw.layout = Native<VkPipelineLayout>(4);
            draw.vertex = Native<VkBuffer>(5); draw.index = Native<VkBuffer>(6); draw.objectIndex = 2;
            draw.indirectOffset = region.offset + 6 * sizeof(VkDrawIndexedIndirectCommand);
            bindings.draws.push_back(draw);
            RenderPipelineDefinition definition; definition.AddFeature<TransparencyFeature>(native);
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
            const TransparencyBindingRef binding{packet ? &bindings : nullptr};
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
TEST_CASE("OitTransparencyFeature: clear store resolve preserve states and export actual stage versions [renderfeatures]")
{
    Fixture fixture; fixture.EnableOit(); Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch, true, true, false, true).success);
    REQUIRE(graph.GetPasses().size() == 3); CHECK(graph.GetResources().size() == 6); CHECK(graph.GetBuffers().size() == 3);
    const auto& clear = graph.GetPasses()[0]; const auto& store = graph.GetPasses()[1]; const auto& resolve = graph.GetPasses()[2];
    CHECK(clear.name == "OITClear"); CHECK(store.name == "OITStore"); CHECK(resolve.name == "OITResolve");
    for (const auto& pass : graph.GetPasses()) CHECK(pass.queueFamily == RG::QueueFamily::Graphics);
    CHECK(clear.isCompute); CHECK_FALSE(store.isCompute); CHECK_FALSE(resolve.isCompute);
    REQUIRE(clear.writes.size() == 1); CHECK(clear.writeStates[0] == RG::ResourceState::TransferDst);
    REQUIRE(clear.bufferWrites.size() == 1); CHECK(clear.bufferWriteStates[0] == RG::ResourceState::TransferDst);
    CHECK(store.colorAttachments.empty()); CHECK(store.hasDepth); CHECK(store.depthAttachment.handle == fixture.depth.handle);
    REQUIRE(store.writes.size() >= 1); REQUIRE(store.bufferWrites.size() == 1);
    CHECK(store.bufferWriteStates[0] == RG::ResourceState::FragmentStorageWrite);
    REQUIRE(store.bufferReads.size() == 2); CHECK(store.bufferReadStates[0] == RG::ResourceState::FragmentStorageRead);
    CHECK(store.bufferReadStates[1] == RG::ResourceState::IndirectRead);
    REQUIRE(resolve.colorAttachments.size() == 2); CHECK_FALSE(resolve.hasDepth);
    CHECK(resolve.colorAttachments[0].handle == fixture.color.handle); CHECK(resolve.colorAttachments[1].handle == fixture.picking.handle);
    REQUIRE(resolve.reads.size() == 1); CHECK(resolve.readStates[0] == RG::ResourceState::FragmentStorageRead);
    REQUIRE(resolve.bufferReads.size() == 1); CHECK(resolve.bufferReadStates[0] == RG::ResourceState::FragmentStorageRead);
    CHECK(resolve.reads[0].index == 6); CHECK(resolve.reads[0].version == 2);
    CHECK(resolve.bufferReads[0].index == 3); CHECK(resolve.bufferReads[0].version == 2);
    CHECK(fixture.color.handle.index == 1); CHECK(fixture.picking.handle.index == 3); CHECK(fixture.depth.handle.index == 2);
    CHECK(fixture.color.handle.version == 1); CHECK(fixture.picking.handle.version == 1); CHECK(fixture.depth.handle.version == 1);
    CHECK(graph.GetResources()[5].initialState == RG::ResourceState::FragmentStorageRead);
    CHECK(graph.GetBuffers()[2].initialState == RG::ResourceState::FragmentStorageRead);
    graph.Compile(); for (const auto& pass : graph.GetPasses()) CHECK_FALSE(pass.culled);
    const auto hasImageBarrier = [](const auto& pass, auto before, auto after) {
        return std::any_of(pass.preBarriers.begin(), pass.preBarriers.end(), [&](const auto& barrier) {
            return barrier.resource.index == 6 && barrier.before == before && barrier.after == after;
        });
    };
    const auto hasBufferBarrier = [](const auto& pass, auto before, auto after) {
        return std::any_of(pass.bufferPreBarriers.begin(), pass.bufferPreBarriers.end(), [&](const auto& barrier) {
            return barrier.resource.index == 3 && barrier.before == before && barrier.after == after;
        });
    };
    CHECK(hasImageBarrier(clear, RG::ResourceState::FragmentStorageRead, RG::ResourceState::TransferDst));
    CHECK(hasImageBarrier(store, RG::ResourceState::TransferDst, RG::ResourceState::FragmentStorageWrite));
    CHECK(hasImageBarrier(resolve, RG::ResourceState::FragmentStorageWrite, RG::ResourceState::FragmentStorageRead));
    CHECK(hasBufferBarrier(clear, RG::ResourceState::FragmentStorageRead, RG::ResourceState::TransferDst));
    CHECK(hasBufferBarrier(store, RG::ResourceState::TransferDst, RG::ResourceState::FragmentStorageWrite));
    CHECK(hasBufferBarrier(resolve, RG::ResourceState::FragmentStorageWrite, RG::ResourceState::FragmentStorageRead));
}
TEST_CASE("OitTransparencyFeature: invalid persistent packet fails before imports [renderfeatures]")
{
    Fixture fixture; fixture.EnableOit();
    SUBCASE("heads image") { fixture.bindings.headsImage = VK_NULL_HANDLE; }
    SUBCASE("heads view") { fixture.bindings.headsView = VK_NULL_HANDLE; }
    SUBCASE("extent") { fixture.bindings.width++; }
    SUBCASE("node header") { fixture.bindings.nodes.size = 16; }
    SUBCASE("node alignment") { fixture.bindings.nodes.offset++; }
    SUBCASE("capacity") { fixture.bindings.capacity++; }
    SUBCASE("resolve pipeline") { fixture.bindings.resolvePipeline = VK_NULL_HANDLE; }
    SUBCASE("resolve set") { fixture.bindings.resolveSet = VK_NULL_HANDLE; }
    SUBCASE("resolve count") { fixture.bindings.maxResolveK = 17; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    CHECK_FALSE(fixture.Build(graph, scratch).success); CHECK(graph.GetPasses().empty());
    CHECK(graph.GetResources().size() == 5); CHECK(graph.GetBuffers().size() == 1);
    CHECK_FALSE(fixture.picking.handle.IsValid());
}
TEST_CASE("OitTransparencyFeature: empty OIT packets and cold preparation require no Vulkan device [renderfeatures]")
{
    Fixture fixture; fixture.EnableOit(); fixture.bindings.draws.clear();
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    REQUIRE(fixture.Build(graph, scratch).success); CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 5);
    CHECK(fixture.picking.handle.version == 0);
    GeometrySubsystem geo; DrawList draws; RenderSnapshot snapshot; TransparencyViewState state;
    const auto cold = fixture.native.PrepareTransparencyBindings(geo, {}, false, false, Mat4(1), {}, draws, snapshot,
        {}, {}, nullptr, true, state, 30);
    CHECK_FALSE(cold.oit); CHECK(cold.draws.empty());
}
TEST_CASE("OitTransparencyFeature: one compiled topology switches modes without stale graph handles [renderfeatures]")
{
    Fixture fixture; fixture.EnableOit();
    for (const bool oit : {true, false, true})
    {
        fixture.bindings.oit = oit; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        REQUIRE(fixture.Build(graph, scratch).success);
        CHECK(graph.GetPasses().size() == (oit ? 3 : 1));
        CHECK(graph.GetPasses()[0].name == (oit ? "OITClear" : "TransparentPass"));
        CHECK(fixture.picking.handle.index == 3); CHECK(fixture.picking.handle.version == 1);
    }
}
TEST_CASE("OitTransparencyFeature: rejects a duplicate physical heads import [renderfeatures]")
{
    Fixture fixture; fixture.EnableOit(); Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
    RG::TextureDesc desc; desc.name = "AlreadyImportedHeads"; desc.width = 640; desc.height = 480; desc.format = RG::TextureFormat::R32_Uint;
    graph.ImportResource(desc, (void*)fixture.bindings.headsImage, (void*)fixture.bindings.headsView, RG::ResourceState::FragmentStorageRead);
    CHECK_FALSE(fixture.Build(graph, scratch).success); CHECK(graph.GetPasses().empty());
    CHECK(graph.GetResources().size() == 6); CHECK_FALSE(fixture.picking.handle.IsValid());
}
