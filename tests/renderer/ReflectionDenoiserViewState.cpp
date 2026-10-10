#include <doctest/doctest.h>
#include "luth/renderer/features/rt/ReflectionDenoiserViewState.h"
#include "luth/renderer/subsystems/SvgfDenoiser.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/backend/vulkan/VulkanViewPool.h"

using namespace Luth;
namespace
{
    class SizedTexture final : public Texture
    {
    public:
        SizedTexture(u32 w, u32 h) : width(w), height(h) {}
        void Bind(u32) const override {}
        u32 GetWidth() const override { return width; }
        u32 GetHeight() const override { return height; }
        u32 GetRendererID() const override { return 0; }
        const fs::path& GetPath() const override { return path; }
        TextureFormat GetFormat() const override { return TextureFormat::RGBA8; }
        std::string GetFormatString() const override { return "RGBA8"; }
        TextureWrapMode GetWrapMode() const override { return TextureWrapMode::Repeat; }
        void SetWrapMode(TextureWrapMode) override {}
        std::pair<TextureFilterMode, TextureFilterMode> GetFilterMode() const override
        { return {TextureFilterMode::Linear, TextureFilterMode::Linear}; }
        void SetFilterMode(TextureFilterMode, TextureFilterMode) override {}
        int GetMipLevels() const override { return 1; }
        void GenerateMipmaps() override {}
    private:
        u32 width, height;
        fs::path path;
    };
}



TEST_CASE("ReflectionDenoiserViewState: invalid configuration fails before native allocation")
{
    CHECK_THROWS_AS(ReflectionDenoiserViewState::Config(0, 1, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(ReflectionDenoiserViewState::Config(1, 0, false, 1), std::invalid_argument);
    CHECK_THROWS_AS(ReflectionDenoiserViewState::Config(1, 1, false, 0), std::invalid_argument);
    const auto config = ReflectionDenoiserViewState::Config(321, 181, true, 4);
    CHECK(ReflectionViewState::WorkingExtent(config) == std::array<u32, 2>{160, 90});
    CHECK_THROWS_AS(ReflectionDenoiserViewState::Create({}, config, {}), std::invalid_argument);
    CHECK_THROWS_AS(ReflectionDenoiserViewState::Create({1}, config, {}), std::runtime_error);
    SvgfDenoiser dormant(DenoiserChannel::Reflections);
    FrameTargets targets;
    CHECK_FALSE(dormant.EnsureReflectionView({1}, targets, {}));
    dormant.ReleaseReflectionView({1});
}

TEST_CASE("ReflectionDenoiserViewState: view replacement retains old bindings until borrowers release")
{
    ReflectionDenoiserViewStates states;
    int waits = 0, creates = 0;
    auto safe = [&] { ++waits; };
    auto factory = [&](const ViewStateConfig& c) {
        ++creates;
        auto state = std::make_shared<ReflectionDenoiserViewState>();
        state->sourceGeneration = c.resourceGeneration;
        return state;
    };
    auto config = ReflectionDenoiserViewState::Config(321, 181, false, 1);
    auto first = states.Ensure({1}, config, factory, safe);
    std::weak_ptr<ReflectionDenoiserViewState> old = first;
    CHECK(states.Ensure({1}, config, factory, safe) == first);
    CHECK(creates == 1); CHECK(waits == 0);
    auto secondView = states.Ensure({2}, config, factory, safe);
    CHECK(secondView != first);
    config.resourceGeneration = 2; // Different raw reflection or G-buffer binding, same extent.
    auto replacement = states.Ensure({1}, config, factory, safe);
    CHECK(replacement != first); CHECK(waits == 1);
    CHECK_FALSE(old.expired()); first.reset(); CHECK(old.expired());
    CHECK(*states.Find({2}) == secondView);
    config.signature = 1;
    CHECK_THROWS_AS(states.Ensure({1}, config, [](const ViewStateConfig&) -> std::shared_ptr<ReflectionDenoiserViewState> {
        throw std::runtime_error("allocation failed");
    }, safe), std::runtime_error);
    CHECK(*states.Find({1}) == replacement);
    states.Release({1}, safe); CHECK_FALSE(states.Find({1}));
    CHECK(*states.Find({2}) == secondView);
    states.ReleaseAll(safe); CHECK_FALSE(states.Find({2}));
    CHECK(waits == 4);
}

TEST_CASE("ReflectionDenoiserViewState: local descriptor budget covers both temporal and iteration parities")
{
    const ReflectionDenoiserPoolBudget budget;
    CHECK(budget.maxSets == 1 + 2 + 2 + 2);
    CHECK(budget.sizes[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    CHECK(budget.sizes[0].descriptorCount == 1 + 2 * 5 + 2 * 2 + 2 * 3);
    CHECK(budget.sizes[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    CHECK(budget.sizes[1].descriptorCount == 1 + 2 * 6 + 2 * 3 + 2 * 3);
    for (const u32 frames : {2u, 3u}) {
        const VulkanViewPool shared(true, frames);
        CHECK(shared.maxSets + budget.maxSets == 205 - 15 * frames - 42);
        CHECK(shared.sizes[1].descriptorCount + budget.sizes[1].descriptorCount == 248 - 11 * frames - 92);
        CHECK(shared.sizes[2].descriptorCount + budget.sizes[0].descriptorCount == 317 - 32 * frames - 97);
    }
}

TEST_CASE("ReflectionDenoiserViewState: retained raw input is independent of DI and released with borrowers")
{
    auto raw = std::make_shared<ReflectionViewState>();
    raw->radiance = std::shared_ptr<Texture>(reinterpret_cast<Texture*>(1), [](Texture*) {});
    ReflectionDenoiserViewState state;
    CHECK(state.Noisy() == nullptr);
    state.input = raw;
    CHECK(state.Noisy() == &raw->radiance);
    std::weak_ptr<ReflectionViewState> retained = raw;
    raw.reset(); CHECK_FALSE(retained.expired());
    state.input.reset(); CHECK(retained.expired()); CHECK(state.Noisy() == nullptr);
    SvgfDenoiser dormant(DenoiserChannel::Reflections); FrameTargets targets;
    CHECK_FALSE(dormant.EnsureReflectionView({1}, targets, {})); dormant.ReleaseReflectionView({1});
}

TEST_CASE("ReflectionDenoiserViewState: output routing compares both dimensions and forwards collapsed extents")
{
    for (const auto [w, h] : {std::pair{801u, 601u}, std::pair{1u, 600u}, std::pair{800u, 1u}, std::pair{1u, 1u}}) {
        ReflectionDenoiserViewState state;
        CHECK(state.WorkingOutput() == nullptr);
        state.svgfDenoised = std::make_shared<SizedTexture>(w, h);
        state.width = w; state.height = h;
        CHECK(state.WorkingOutput() == &state.svgfDenoised);
        const auto half = ReflectionViewState::WorkingExtent(ReflectionDenoiserViewState::Config(w, h, true, 1));
        state.width = half[0]; state.height = half[1];
        state.svgfHalf = std::make_shared<SizedTexture>(half[0], half[1]);
        CHECK(state.WorkingOutput() == (w == 1 && h == 1 ? &state.svgfDenoised : &state.svgfHalf));
    }
}
