#include <doctest/doctest.h>
#include "luth/renderer/features/BloomFeature.h"
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
        BloomBindings packet;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            packet.enabled = true; packet.downPipeline = Native<VkPipeline>(1); packet.upPipeline = Native<VkPipeline>(2);
            packet.downLayout = Native<VkPipelineLayout>(3); packet.upLayout = Native<VkPipelineLayout>(4);
            packet.prefilterSet = Native<VkDescriptorSet>(5); packet.width = 801; packet.height = 601;
            packet.state = std::make_shared<BloomViewState>(); packet.source = {Native<const Texture*>(10)};
            for (u32 i = 0; i < BloomViewState::kMipCount; ++i)
            {
                packet.mips[i] = {Native<const Texture*>(20 + i)};
                packet.images[i] = Native<VkImage>(30 + i); packet.views[i] = Native<VkImageView>(40 + i);
                if (i < BloomViewState::kMipCount - 1)
                { packet.downSets[i] = Native<VkDescriptorSet>(50 + i); packet.upSets[i] = Native<VkDescriptorSet>(60 + i); }
            }
            RenderPipelineDefinition definition; definition.AddFeature<BloomFeature>(native);
            PipelineInputContract inputs; inputs.resources = {{RenderResources::ResolvedHDR}, {BloomResources::Bindings}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool present = true, u32 width = 801)
        {
            RG::TextureDesc desc; desc.name = "Resolved"; desc.width = width; desc.height = 601;
            desc.format = RG::TextureFormat::RGBA16_Float;
            GraphTextureRef source{graph.ImportResource(desc, (void*)Native<VkImage>(70),
                (void*)Native<VkImageView>(71), RG::ResourceState::ShaderResource), {Native<const Texture*>(10)}};
            const BloomBindingRef binding{present ? &packet : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::ResolvedHDR, source),
                RenderInputBinding::Present(BloomResources::Bindings, binding)};
            FrameRenderInputs frame; frame.resources = resources; frame.renderFrameIndex = 7;
            ViewRenderInputs view; view.id = {1}; view.width = 801; view.height = 601;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::BloomOutput, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("BloomFeature: eleven graphics compute passes reuse six physical mip nodes [renderfeatures]")
{
    Fixture fixture; Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(fixture.Build(graph, scratch, output).success);
    REQUIRE(graph.GetPasses().size() == 11); REQUIRE(graph.GetResources().size() == 7);
    CHECK(graph.GetPasses()[0].name == "BloomPrefilter");
    for (u32 i = 0; i < 5; ++i)
    {
        CHECK(graph.GetPasses()[1 + i].name == "BloomDown" + std::to_string(i));
        CHECK(graph.GetPasses()[6 + i].name == "BloomUp" + std::to_string(4 - i));
    }
    CHECK(output.handle.index == 2); CHECK(output.binding.texture == fixture.packet.mips[0].texture);
    CHECK(graph.GetPasses()[0].reads[0].index == 1);
    for (u32 i = 0; i < 6; ++i)
    {
        const auto [w, h] = BloomViewState::MipExtent(BloomViewState::Config(801, 601), i);
        CHECK(graph.GetResources()[i + 1].desc.width == w); CHECK(graph.GetResources()[i + 1].desc.height == h);
        CHECK(graph.GetResources()[i + 1].initialState == RG::ResourceState::Undefined);
    }
    graph.Compile();
    for (const auto& pass : graph.GetPasses())
    { CHECK(pass.isCompute); CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK_FALSE(pass.culled); }
    for (u32 i = 6; i < 11; ++i)
    {
        const auto& pass = graph.GetPasses()[i]; REQUIRE(pass.reads.size() == 2); REQUIRE(pass.writes.size() == 1);
        CHECK(pass.reads[1].index == pass.writes[0].index);
        CHECK(pass.readStates[1] == RG::ResourceState::ComputeReadStorage);
    }
}
TEST_CASE("BloomFeature: disabled and cold contributions publish absence each graph [renderfeatures]")
{
    Fixture fixture;
    SUBCASE("disabled") { fixture.packet.enabled = false; }
    SUBCASE("cold down") { fixture.packet.downPipeline = VK_NULL_HANDLE; }
    SUBCASE("cold up") { fixture.packet.upPipeline = VK_NULL_HANDLE; }
    for (u32 i = 0; i < 2; ++i)
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch);
        GraphTextureRef output{{99, 1}, {Native<const Texture*>(99)}};
        REQUIRE(fixture.Build(graph, scratch, output).success);
        CHECK_FALSE(output.handle.IsValid()); CHECK_FALSE(output.binding.texture);
        CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 1);
    }
}
TEST_CASE("BloomFeature: malformed packets fail before any pyramid import [renderfeatures]")
{
    Fixture fixture; bool present = true; u32 width = 801;
    SUBCASE("packet") { present = false; }
    SUBCASE("state") { fixture.packet.state.reset(); }
    SUBCASE("input extent") { width = 800; }
    SUBCASE("packet extent") { fixture.packet.height = 600; }
    SUBCASE("layout") { fixture.packet.upLayout = VK_NULL_HANDLE; }
    SUBCASE("prefilter") { fixture.packet.prefilterSet = VK_NULL_HANDLE; }
    SUBCASE("down set") { fixture.packet.downSets[4] = VK_NULL_HANDLE; }
    SUBCASE("up set") { fixture.packet.upSets[3] = VK_NULL_HANDLE; }
    SUBCASE("sampled source") { fixture.packet.source.texture = Native<const Texture*>(99); }
    SUBCASE("mip image") { fixture.packet.images[5] = VK_NULL_HANDLE; }
    SUBCASE("mip view") { fixture.packet.views[5] = VK_NULL_HANDLE; }
    SUBCASE("mip subresource") { fixture.packet.mips[5].baseMip = 1; }
    SUBCASE("duplicate image") { fixture.packet.images[5] = fixture.packet.images[0]; }
    SUBCASE("duplicate texture") { fixture.packet.mips[5] = fixture.packet.mips[0]; }
    SUBCASE("already imported") { fixture.packet.images[5] = Native<VkImage>(70); }
    SUBCASE("source feedback") { fixture.packet.mips[5] = fixture.packet.source; }
    SUBCASE("threshold NaN") { fixture.packet.threshold = std::numeric_limits<float>::quiet_NaN(); }
    SUBCASE("radius NaN") { fixture.packet.radius = std::numeric_limits<float>::quiet_NaN(); }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{99, 1}, {Native<const Texture*>(99)}};
    CHECK_FALSE(fixture.Build(graph, scratch, output, present, width).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 1);
}
TEST_CASE("BloomFeature: cached contribution switches on off on without stale output [renderfeatures]")
{
    Fixture fixture;
    for (u32 i = 0; i < 3; ++i)
    {
        fixture.packet.enabled = i != 1;
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(fixture.Build(graph, scratch, output).success);
        CHECK(graph.GetPasses().size() == (i == 1 ? 0 : 11)); CHECK(output.handle.IsValid() == (i != 1));
    }
}
TEST_CASE("BloomFeature: graph retains view state until recorded jobs retire [renderfeatures]")
{
    Fixture fixture; std::weak_ptr<BloomViewState> retained = fixture.packet.state;
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(fixture.Build(graph, scratch, output).success);
        fixture.packet.state.reset(); CHECK_FALSE(retained.expired());
    }
    CHECK(retained.expired());
}
TEST_CASE("BloomFeature: cold preparation needs no Vulkan device [renderfeatures]")
{
    PostProcessSubsystem native; auto state = std::make_shared<BloomViewState>();
    const auto packet = native.PrepareBloomBindings(state, {}, 7, 1.0f, 0.7f, true);
    CHECK(packet.enabled); CHECK_FALSE(packet.downPipeline); CHECK(packet.state == state);
    CHECK_FALSE(native.PrepareBloomBindings({}, {}, 7, 1.0f, 0.7f, false).enabled);
}
