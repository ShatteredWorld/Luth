#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/features/rt/ReflectionUpscaleViewState.h"
#include "luth/renderer/subsystems/ReflectionsSubsystem.h"
#include "luth/renderer/settings/SvgfSettings.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"

using namespace Luth;
namespace {
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    std::shared_ptr<Texture> Borrowed(u64 n) { return {Native<Texture*>(n), [](Texture*) {}}; }
}
TEST_CASE("ReflectionUpscaleViewState: invalid and dormant configurations do not access a device") {
    const ViewStateConfig config{1279, 719, 0, 1};
    auto owner = std::make_shared<ReflectionDenoiserViewState>();
    std::array<std::shared_ptr<Texture>, 2> sources{Borrowed(1), Borrowed(2)};
    CHECK_THROWS_AS(ReflectionUpscaleViewState::Create({}, config, {}, {}, owner, sources), std::invalid_argument);
    CHECK_THROWS_AS(ReflectionUpscaleViewState::Create({1}, config, {}, {}, owner, sources), std::invalid_argument);
    owner->id = {1}; owner->svgfHalf = Borrowed(3); owner->svgfDenoised = Borrowed(5);
    CHECK_THROWS_AS(ReflectionUpscaleViewState::Create({1}, config, {}, {}, owner, sources), std::runtime_error);
    owner->id = {2};
    CHECK_THROWS_AS(ReflectionUpscaleViewState::Create({1}, config, {}, {}, owner, sources), std::invalid_argument);
    ReflectionsSubsystem dormant; FrameTargets targets; ViewResources vr;
    CHECK_FALSE(dormant.EnsureUpscaleView({1}, targets, owner));
    CHECK_FALSE(dormant.PrepareUpscaleBindings(vr, 42, {1}, 3, {}).Ready());
    dormant.ReleaseView({1});
}
TEST_CASE("ReflectionUpscaleViewState: independent stores retain sources and preserve failed replacements") {
    ReflectionUpscaleViewStates store; int waits = 0; auto safe = [&] { ++waits; };
    auto denoiser = std::make_shared<ReflectionDenoiserViewState>();
    std::weak_ptr<ReflectionDenoiserViewState> retained = denoiser;
    auto factory = [&](const ViewStateConfig& c) {
        auto s = std::make_shared<ReflectionUpscaleViewState>(); s->sourceGeneration = c.resourceGeneration;
        s->denoiser = denoiser; s->sources = {Borrowed(1), Borrowed(2)}; return s;
    };
    ViewStateConfig config{1279, 719, 0, 1};
    auto scene = store.Ensure({1}, config, factory, safe), game = store.Ensure({2}, config, factory, safe);
    CHECK(scene != game); CHECK(store.Ensure({1}, config, factory, safe) == scene); CHECK(waits == 0);
    ++config.resourceGeneration; auto replaced = store.Ensure({1}, config, factory, safe);
    CHECK(replaced != scene); CHECK(*store.Find({2}) == game); CHECK(waits == 1);
    ++config.resourceGeneration;
    CHECK_THROWS_AS(store.Ensure({1}, config, [](const auto&) -> std::shared_ptr<ReflectionUpscaleViewState> {
        throw std::runtime_error("allocation failed");
    }, safe), std::runtime_error);
    CHECK(*store.Find({1}) == replaced); CHECK(*store.Find({2}) == game);
    denoiser.reset(); CHECK_FALSE(retained.expired());
    store.Release({1}, safe); CHECK_FALSE(store.Find({1})); CHECK(*store.Find({2}) == game);
    store.ReleaseAll(safe); CHECK(waits == 4); CHECK_FALSE(store.Find({2}));
    scene.reset(); replaced.reset(); CHECK_FALSE(retained.expired()); game.reset(); CHECK(retained.expired());
}
TEST_CASE("ReflectionUpscaleViewState: local descriptor budget preserves the compatibility total") {
    const ReflectionUpscalePoolBudget local;
    CHECK(local.maxSets == 1);
    CHECK(local.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); CHECK(local.sizes[0].descriptorCount == 3);
    CHECK(local.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); CHECK(local.sizes[1].descriptorCount == 1);
    for (u32 frames : {2u, 3u}) {
        const VulkanViewPool shared(true, frames);
        CHECK(shared.maxSets + local.maxSets == 205 - 15 * frames - 49);
        CHECK(shared.sizes[1].descriptorCount + local.sizes[1].descriptorCount == 248 - 11 * frames - 117);
        CHECK(shared.sizes[2].descriptorCount + local.sizes[0].descriptorCount == 317 - 32 * frames - 118);
    }
}
TEST_CASE("ReflectionUpscaleViewState: native registration preserves reflection upscale name and borrowed handles") {
    ReflectionUpscaleBindings native;
    native.pipeline = Native<VkPipeline>(1); native.layout = Native<VkPipelineLayout>(2);
    native.set = native.globalSet = Native<VkDescriptorSet>(3);
    native.width = 639; native.height = 359; native.fullWidth = 1279; native.fullHeight = 719;
    native.outputImage = Native<VkImage>(90); native.outputView = Native<VkImageView>(91);
    auto state = std::make_shared<ReflectionUpscaleViewState>(); std::weak_ptr<ReflectionUpscaleViewState> retained = state;
    native.retained = state; state.reset(); Memory::LinearAllocator scratch(128 * 1024);
    {
        RG::RenderGraph graph(scratch); std::array<RG::ResourceHandle, 3> handles;
        for (u32 i = 0; i < 3; ++i) {
            RG::TextureDesc desc; desc.width = i ? 1279 : 639; desc.height = i ? 719 : 359;
            desc.format = i == 1 ? RG::TextureFormat::D32_Float : i == 2 ? RG::TextureFormat::RG16_Float : RG::TextureFormat::RGBA16_Float;
            handles[i] = graph.ImportResource(desc, (void*)Native<VkImage>(10 + i), (void*)Native<VkImageView>(20 + i), RG::ResourceState::ComputeWrite);
        }
        const auto output = ReflectionsSubsystem::AddUpscalePass(graph, handles, native);
        REQUIRE(output.IsValid()); REQUIRE(graph.GetPasses().size() == 1);
        CHECK(graph.GetPasses()[0].name == "ReflUpscale");
        REQUIRE(graph.GetPasses()[0].reads.size() == 3);
        for (u32 i = 0; i < 3; ++i) CHECK(graph.GetPasses()[0].reads[i].index == handles[i].index);
        CHECK(graph.GetResources().size() == 4);
        CHECK(graph.GetResources()[output.index - 1].desc.width == 1279); CHECK(graph.GetResources()[output.index - 1].desc.height == 719);
        graph.Compile(); CHECK_FALSE(graph.GetPasses()[0].culled);
        CHECK(graph.GetPasses()[0].queueFamily == RG::QueueFamily::AsyncCompute);
        native.retained.reset(); native.pipeline = VK_NULL_HANDLE; CHECK_FALSE(retained.expired());
    }
    CHECK(retained.expired());
}
TEST_CASE("ReflectionUpscaleViewState: unavailable native upscale preserves input without registering work") {
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    const RG::ResourceHandle input{1, 0}; ReflectionUpscaleBindings native;
    CHECK(ReflectionsSubsystem::AddUpscalePass(graph, {input, {}, {}}, native).index == input.index);
    CHECK_FALSE(ReflectionsSubsystem::AddUpscalePass(graph, {}, native).IsValid());
    CHECK(graph.GetPasses().empty());
}
TEST_CASE("ReflectionUpscaleViewState: full and collapsed half extents forward the already full output") {
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    ReflectionUpscaleBindings native; native.pipeline = Native<VkPipeline>(1); native.layout = Native<VkPipelineLayout>(2);
    native.set = native.globalSet = Native<VkDescriptorSet>(3);
    native.fullWidth = native.fullHeight = native.width = native.height = 1;
    SUBCASE("Collapsed half") {}
    SUBCASE("Full resolution") { native.fullWidth = native.width = 1279; native.fullHeight = native.height = 719; }
    native.outputImage = Native<VkImage>(90); native.outputView = Native<VkImageView>(91);
    RG::TextureDesc desc; desc.width = native.width; desc.height = native.height; desc.format = RG::TextureFormat::RGBA16_Float;
    const auto input = graph.ImportResource(desc, (void*)native.outputImage, (void*)native.outputView, RG::ResourceState::ComputeWrite);
    CHECK(ReflectionsSubsystem::AddUpscalePass(graph, {input, {}, {}}, native).index == input.index);
    CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 1);
}
