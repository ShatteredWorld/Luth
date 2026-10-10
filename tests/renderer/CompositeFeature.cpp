#include <doctest/doctest.h>
#include "luth/renderer/features/CompositeFeature.h"
#include "luth/renderer/features/RenderPipelineCompiler.h"
#include "luth/renderer/subsystems/PostProcessSubsystem.h"
#include "luth/memory/LinearAllocator.h"

using namespace Luth;
namespace
{
    template<class T> T Native(u64 value) { return reinterpret_cast<T>(static_cast<uintptr_t>(value)); }
    struct Fixture
    {
        PostProcessSubsystem native;
        CompositeBindings packet;
        std::unique_ptr<CompiledRenderPipeline> pipeline;
        Fixture()
        {
            packet.pipeline = Native<VkPipeline>(1); packet.layout = Native<VkPipelineLayout>(2);
            packet.set = Native<VkDescriptorSet>(3); packet.width = 801; packet.height = 601;
            packet.state = std::make_shared<CompositeViewState>(); packet.source = {Native<const Texture*>(10)};
            packet.outputOwner = std::shared_ptr<Texture>(Native<Texture*>(20), [](Texture*) {});
            packet.output = {packet.outputOwner.get()}; packet.outputImage = Native<VkImage>(30); packet.outputView = Native<VkImageView>(31);
            packet.uniform = {Native<VkBuffer>(40), 256, sizeof(PostProcessUBO)};
            RenderPipelineDefinition definition; definition.AddFeature<CompositeFeature>(native);
            PipelineInputContract inputs;
            inputs.resources = {{RenderResources::GridHDR}, {RenderResources::BloomOutput, ResourceOutputPresence::Optional},
                {CompositeResources::Bindings}};
            auto compiled = RenderPipelineCompiler{}.Compile(std::move(definition), {}, inputs);
            REQUIRE(compiled.pipeline); pipeline = std::move(compiled.pipeline);
        }
        PipelineBuildResult Build(RG::RenderGraph& graph, Memory::LinearAllocator& scratch,
            GraphTextureRef& output, bool hasBloom = false, bool present = true, u32 width = 801)
        {
            RG::TextureDesc desc; desc.name = "GridHDR"; desc.width = width; desc.height = 601;
            desc.format = RG::TextureFormat::RGBA16_Float;
            GraphTextureRef hdr{graph.ImportResource(desc, (void*)Native<VkImage>(50), (void*)Native<VkImageView>(51),
                RG::ResourceState::ShaderResource), {Native<const Texture*>(10)}};
            GraphTextureRef bloom;
            if (hasBloom)
            {
                desc.name = "Bloom"; desc.width = 400; desc.height = 300;
                bloom = {graph.ImportResource(desc, (void*)Native<VkImage>(52), (void*)Native<VkImageView>(53),
                    RG::ResourceState::ComputeWrite), {Native<const Texture*>(11)}};
            }
            const CompositeBindingRef binding{present ? &packet : nullptr};
            const std::array resources{RenderInputBinding::Present(RenderResources::GridHDR, hdr),
                hasBloom ? RenderInputBinding::Present(RenderResources::BloomOutput, bloom) : RenderInputBinding::Absent(RenderResources::BloomOutput),
                RenderInputBinding::Present(CompositeResources::Bindings, binding)};
            FrameRenderInputs frame; frame.resources = resources; frame.renderFrameIndex = 7;
            ViewRenderInputs view; view.id = {1}; view.width = 801; view.height = 601;
            const std::array exports{RenderOutputBinding::Capture(RenderResources::TonemappedLDR, output)};
            return pipeline->Build(graph, frame, view, scratch, exports);
        }
    };
}
TEST_CASE("CompositeFeature: optional bloom and HDR inputs produce one RGBA8 LDR pass [renderfeatures]")
{
    Fixture fixture; bool bloom = false;
    SUBCASE("absent bloom") {}
    SUBCASE("present bloom") { bloom = true; fixture.packet.bloom = {Native<const Texture*>(11)}; fixture.packet.parameters.bloomStrength = 0.5f; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
    REQUIRE(fixture.Build(graph, scratch, output, bloom).success);
    REQUIRE(graph.GetPasses().size() == 1); CHECK(graph.GetResources().size() == (bloom ? 3 : 2));
    const auto& pass = graph.GetPasses()[0]; CHECK(pass.name == "PostProcess"); CHECK_FALSE(pass.isCompute);
    CHECK(pass.queueFamily == RG::QueueFamily::Graphics); CHECK(pass.reads.size() == (bloom ? 2 : 1));
    CHECK(pass.reads[0].index == 1); REQUIRE(pass.colorAttachments.size() == 1);
    CHECK(pass.colorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
    CHECK(pass.colorAttachments[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    CHECK(output.handle.index == (bloom ? 3 : 2)); CHECK(output.binding.texture == fixture.packet.output.texture);
    const auto& target = graph.GetResources()[output.handle.index - 1];
    CHECK(target.desc.format == RG::TextureFormat::RGBA8_Unorm); CHECK(target.desc.width == 801); CHECK(target.desc.height == 601);
    CHECK(target.initialState == RG::ResourceState::ShaderResource);
    graph.Compile(); CHECK_FALSE(pass.culled);
    if (bloom)
    {
        bool transitioned = false;
        for (const auto& barrier : pass.preBarriers)
            if (barrier.resource.index == 2) { CHECK(barrier.after == RG::ResourceState::ShaderResource); transitioned = true; }
        CHECK(transitioned);
    }
    CHECK(fixture.packet.uniform.offset == 256); CHECK(fixture.packet.uniform.range == sizeof(PostProcessUBO));
}
TEST_CASE("CompositeFeature: invalid bindings fail before LDR import [renderfeatures]")
{
    Fixture fixture; bool present = true, bloom = false; u32 width = 801;
    SUBCASE("packet") { present = false; }
    SUBCASE("pipeline") { fixture.packet.pipeline = VK_NULL_HANDLE; }
    SUBCASE("layout") { fixture.packet.layout = VK_NULL_HANDLE; }
    SUBCASE("set") { fixture.packet.set = VK_NULL_HANDLE; }
    SUBCASE("state") { fixture.packet.state.reset(); }
    SUBCASE("uniform") { fixture.packet.uniform.buffer = VK_NULL_HANDLE; }
    SUBCASE("uniform range") { fixture.packet.uniform.range = 1; }
    SUBCASE("source mismatch") { fixture.packet.source.texture = Native<const Texture*>(99); }
    SUBCASE("input extent") { width = 800; }
    SUBCASE("packet extent") { fixture.packet.height = 600; }
    SUBCASE("output image") { fixture.packet.outputImage = VK_NULL_HANDLE; }
    SUBCASE("output view") { fixture.packet.outputView = VK_NULL_HANDLE; }
    SUBCASE("output owner") { fixture.packet.outputOwner.reset(); }
    SUBCASE("output subresource") { fixture.packet.output.mipCount = 2; }
    SUBCASE("feedback") { fixture.packet.output = fixture.packet.source; }
    SUBCASE("already imported") { fixture.packet.outputImage = Native<VkImage>(50); }
    SUBCASE("absent bloom strength") { fixture.packet.parameters.bloomStrength = 0.5f; }
    SUBCASE("absent bloom descriptor") { fixture.packet.bloom = {Native<const Texture*>(11)}; }
    SUBCASE("present bloom descriptor mismatch") { bloom = true; fixture.packet.bloom = {Native<const Texture*>(99)}; }
    Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output{{99, 1}, {Native<const Texture*>(99)}};
    CHECK_FALSE(fixture.Build(graph, scratch, output, bloom, present, width).success);
    CHECK_FALSE(output.handle.IsValid()); CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == (bloom ? 2 : 1));
}
TEST_CASE("CompositeFeature: recorded graph retains output and descriptor state [renderfeatures]")
{
    Fixture fixture; std::weak_ptr<CompositeViewState> state = fixture.packet.state;
    std::weak_ptr<Texture> outputOwner = fixture.packet.outputOwner;
    {
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(fixture.Build(graph, scratch, output).success);
        fixture.packet.state.reset(); fixture.packet.outputOwner.reset();
        CHECK_FALSE(state.expired()); CHECK_FALSE(outputOwner.expired());
    }
    CHECK(state.expired()); CHECK(outputOwner.expired());
}
TEST_CASE("CompositeFeature: cached topology accepts bloom presence changes without stale reads [renderfeatures]")
{
    Fixture fixture;
    for (u32 i = 0; i < 3; ++i)
    {
        const bool bloom = i != 1;
        fixture.packet.bloom = bloom ? TextureBindingRef{Native<const Texture*>(11)} : TextureBindingRef{};
        fixture.packet.parameters.bloomStrength = bloom ? 0.5f : 0.0f;
        Memory::LinearAllocator scratch(64 * 1024); RG::RenderGraph graph(scratch); GraphTextureRef output;
        REQUIRE(fixture.Build(graph, scratch, output, bloom).success);
        CHECK(graph.GetPasses()[0].reads.size() == (bloom ? 2 : 1)); CHECK(output.handle.IsValid());
    }
}
TEST_CASE("CompositeFeature: uniform preparation preserves grading and explicit fallback policy [renderfeatures]")
{
    PostProcessSettings settings; settings.bloomStrength = 0.7f; settings.tonemapOp = TonemapOperator::AgX;
    settings.exposure = 2.0f; settings.contrast = 1.3f; settings.saturation = 0.6f;
    settings.vignetteAmount = 0.1f; settings.vignetteHardness = 0.2f; settings.grainAmount = 0.3f;
    settings.sharpness = 0.4f; settings.chromaticAberration = 0.5f;
    settings.shadowBalance = Vec3{0.7f}; settings.midtoneBalance = Vec3{0.8f}; settings.highlightBalance = Vec3{0.9f};
    const auto lit = MakeCompositeUniforms(settings, false, true, 42.0f);
    CHECK(lit.bloomThreshold == settings.bloomThreshold); CHECK(lit.bloomStrength == settings.bloomStrength);
    CHECK(lit.tonemapOp == (int)TonemapOperator::AgX); CHECK(lit.exposure == settings.exposure);
    CHECK(lit.contrast == settings.contrast); CHECK(lit.saturation == settings.saturation);
    CHECK(lit.vignetteAmount == settings.vignetteAmount); CHECK(lit.vignetteHardness == settings.vignetteHardness);
    CHECK(lit.grainAmount == settings.grainAmount); CHECK(lit.sharpness == settings.sharpness);
    CHECK(lit.chromaticAberration == settings.chromaticAberration); CHECK(lit.time == 42.0f);
    CHECK(lit.shadowBalance == settings.shadowBalance); CHECK(lit.midtoneBalance == settings.midtoneBalance);
    CHECK(lit.highlightBalance == settings.highlightBalance);
    CHECK(MakeCompositeUniforms(settings, true, false, 42.0f).tonemapOp == -1);
    CHECK(MakeCompositeUniforms(settings, false, false, 42.0f).bloomStrength == 0.0f);
    CHECK(settings.bloomStrength == 0.7f);
}
TEST_CASE("CompositeFeature: cold preparation needs no native device [renderfeatures]")
{
    PostProcessSubsystem native; auto state = std::make_shared<CompositeViewState>();
    const auto packet = native.PrepareCompositeBindings(state, {}, {}, {}, 7, {});
    CHECK_FALSE(packet.pipeline); CHECK(packet.state == state); CHECK_FALSE(packet.uniform.buffer);
}
