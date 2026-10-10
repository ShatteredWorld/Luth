#include <doctest/doctest.h>
#include "luth/renderer/features/rt/GiDenoiserViewState.h"
#include "luth/renderer/features/rt/RestirGiViewState.h"
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/features/rt/DiUpscaleViewState.h"
#include "luth/renderer/subsystems/RtRestirSubsystem.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"

using namespace Luth;
namespace {
    template<class T> T Native(u64 n) { return reinterpret_cast<T>(static_cast<uintptr_t>(n)); }
    std::shared_ptr<Texture> Borrowed(u64 n) { return {Native<Texture*>(n), [](Texture*) {}}; }
}
TEST_CASE("DiUpscaleViewState: invalid and dormant configurations do not access a device") {
    const ViewStateConfig config{1279, 719, 0, 1};
    std::array<std::shared_ptr<DiDenoiserViewState>, 2> owners;
    std::array<std::shared_ptr<Texture>, 2> sources{Borrowed(1), Borrowed(2)};
    CHECK_THROWS_AS(DiUpscaleViewState::Create({}, config, {}, {}, owners, sources), std::invalid_argument);
    CHECK_THROWS_AS(DiUpscaleViewState::Create({1}, config, {}, {}, owners, sources), std::invalid_argument);
    for (u32 i = 0; i < 2; ++i) {
        owners[i] = std::make_shared<DiDenoiserViewState>(); owners[i]->id = {1};
        owners[i]->signal = i ? DiDenoiserSignal::Specular : DiDenoiserSignal::Diffuse;
        owners[i]->svgfDiHalf = Borrowed(3 + i); owners[i]->svgfDenoised = Borrowed(5 + i);
    }
    CHECK_THROWS_AS(DiUpscaleViewState::Create({1}, config, {}, {}, owners, sources), std::runtime_error);
    owners[1]->signal = DiDenoiserSignal::Diffuse;
    CHECK_THROWS_AS(DiUpscaleViewState::Create({1}, config, {}, {}, owners, sources), std::invalid_argument);
    RtRestirSubsystem dormant; FrameTargets targets; ViewResources vr;
    CHECK_FALSE(dormant.EnsureUpscaleView({1}, targets, owners[0], owners[1]));
    CHECK_FALSE(dormant.PrepareUpscaleBindings(vr, 42, {1}, 3, DiDenoiserSignal::Specular, {}).Ready());
    CHECK_THROWS_AS(dormant.PrepareUpscaleBindings(vr, 42, {1}, 3, static_cast<DiDenoiserSignal>(2), {}), std::invalid_argument);
    dormant.ReleaseView({1});
}
TEST_CASE("DiUpscaleViewState: independent stores retain sources and preserve failed replacements") {
    DiUpscaleViewStates store; int waits = 0; auto safe = [&] { ++waits; };
    auto diffuse = std::make_shared<DiDenoiserViewState>(), specular = std::make_shared<DiDenoiserViewState>();
    specular->signal = DiDenoiserSignal::Specular;
    std::weak_ptr<DiDenoiserViewState> retained = specular;
    auto factory = [&](const ViewStateConfig& c) {
        auto s = std::make_shared<DiUpscaleViewState>(); s->sourceGeneration = c.resourceGeneration;
        s->denoisers = {diffuse, specular}; s->sources = {Borrowed(1), Borrowed(2)}; return s;
    };
    ViewStateConfig config{1279, 719, 0, 1};
    auto scene = store.Ensure({1}, config, factory, safe), game = store.Ensure({2}, config, factory, safe);
    CHECK(scene != game); CHECK(store.Ensure({1}, config, factory, safe) == scene); CHECK(waits == 0);
    ++config.resourceGeneration; auto replaced = store.Ensure({1}, config, factory, safe);
    CHECK(replaced != scene); CHECK(*store.Find({2}) == game); CHECK(waits == 1);
    ++config.resourceGeneration;
    CHECK_THROWS_AS(store.Ensure({1}, config, [](const auto&) -> std::shared_ptr<DiUpscaleViewState> {
        throw std::runtime_error("allocation failed");
    }, safe), std::runtime_error);
    CHECK(*store.Find({1}) == replaced); CHECK(*store.Find({2}) == game);
    specular.reset(); CHECK_FALSE(retained.expired());
    store.Release({1}, safe); CHECK_FALSE(store.Find({1})); CHECK(*store.Find({2}) == game);
    store.ReleaseAll(safe); CHECK(waits == 4); CHECK_FALSE(store.Find({2}));
    scene.reset(); replaced.reset(); CHECK_FALSE(retained.expired()); game.reset(); CHECK(retained.expired());
}
TEST_CASE("DiUpscaleViewState: descriptor budget accounts for both channels") {
    const DiUpscalePoolBudget local;
    CHECK(local.maxSets == 2);
    CHECK(local.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); CHECK(local.sizes[0].descriptorCount == 2 * 3);
    CHECK(local.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); CHECK(local.sizes[1].descriptorCount == 2);
    for (u32 frames : {2u, 3u}) {
        const VulkanViewPool shared(true, frames);
        CHECK(shared.maxSets + GiDenoiserPoolBudget{}.maxSets + RestirGiPoolBudget(frames).maxSets + local.maxSets == 205 - 14 * frames - 30);
        CHECK(shared.sizes[1].descriptorCount + GiDenoiserPoolBudget{}.sizes[1].descriptorCount + RestirGiPoolBudget(frames).sizes[1].descriptorCount + local.sizes[1].descriptorCount == 248 - 10 * frames - 63);
        CHECK(shared.sizes[2].descriptorCount + GiDenoiserPoolBudget{}.sizes[0].descriptorCount + RestirGiPoolBudget(frames).sizes[0].descriptorCount + local.sizes[0].descriptorCount == 317 - 29 * frames - 63);
    }
}
TEST_CASE("DiUpscaleViewState: native registration preserves both upscale names and borrowed handles") {
    bool specular = false; SUBCASE("Diffuse") {} SUBCASE("Specular") { specular = true; }
    DiUpscaleBindings native; native.signal = specular ? DiDenoiserSignal::Specular : DiDenoiserSignal::Diffuse;
    native.pipeline = Native<VkPipeline>(1); native.layout = Native<VkPipelineLayout>(2);
    native.set = native.globalSet = Native<VkDescriptorSet>(3);
    native.width = 639; native.height = 359; native.fullWidth = 1279; native.fullHeight = 719;
    native.outputImage = Native<VkImage>(90); native.outputView = Native<VkImageView>(91);
    auto state = std::make_shared<DiUpscaleViewState>(); std::weak_ptr<DiUpscaleViewState> retained = state;
    native.retained = state; state.reset(); Memory::LinearAllocator scratch(128 * 1024);
    {
        RG::RenderGraph graph(scratch); std::array<RG::ResourceHandle, 3> handles;
        for (u32 i = 0; i < 3; ++i) {
            RG::TextureDesc desc; desc.width = i ? 1279 : 639; desc.height = i ? 719 : 359;
            desc.format = i == 1 ? RG::TextureFormat::D32_Float : i == 2 ? RG::TextureFormat::RG16_Float : RG::TextureFormat::RGBA16_Float;
            handles[i] = graph.ImportResource(desc, (void*)Native<VkImage>(10 + i), (void*)Native<VkImageView>(20 + i), RG::ResourceState::ComputeWrite);
        }
        const auto output = RtRestirSubsystem::AddUpscalePass(graph, handles, native);
        REQUIRE(output.IsValid()); REQUIRE(graph.GetPasses().size() == 1);
        CHECK(graph.GetPasses()[0].name == (specular ? "DiSpecUpscale" : "DiUpscale"));
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
TEST_CASE("DiUpscaleViewState: unavailable native upscale preserves input without registering work") {
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    const RG::ResourceHandle input{1, 0}; DiUpscaleBindings native;
    CHECK(RtRestirSubsystem::AddUpscalePass(graph, {input, {}, {}}, native).index == input.index);
    CHECK_FALSE(RtRestirSubsystem::AddUpscalePass(graph, {}, native).IsValid());
    CHECK(graph.GetPasses().empty());
}
TEST_CASE("DiUpscaleViewState: collapsed half extent forwards the already full output") {
    Memory::LinearAllocator scratch(128 * 1024); RG::RenderGraph graph(scratch);
    DiUpscaleBindings native; native.pipeline = Native<VkPipeline>(1); native.layout = Native<VkPipelineLayout>(2);
    native.set = native.globalSet = Native<VkDescriptorSet>(3);
    native.fullWidth = native.fullHeight = native.width = native.height = 1;
    native.outputImage = Native<VkImage>(90); native.outputView = Native<VkImageView>(91);
    RG::TextureDesc desc; desc.width = desc.height = 1; desc.format = RG::TextureFormat::RGBA16_Float;
    const auto input = graph.ImportResource(desc, (void*)native.outputImage, (void*)native.outputView, RG::ResourceState::ComputeWrite);
    CHECK(RtRestirSubsystem::AddUpscalePass(graph, {input, {}, {}}, native).index == input.index);
    CHECK(graph.GetPasses().empty()); CHECK(graph.GetResources().size() == 1);
}
