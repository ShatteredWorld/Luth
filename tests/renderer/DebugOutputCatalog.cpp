#include <doctest/doctest.h>
#include "luth/core/diagnostics/Log.h"
#include "luth/renderer/debug/DebugOutputCatalog.h"
#include "luth/renderer/debug/NativeDebugOutputs.h"
#include "luth/renderer/FrameTargets.h"
#include "luth/renderer/RenderPipeline.h"
#include "luth/renderer/resources/Texture.h"

using namespace Luth;
namespace
{
    class TestTexture final : public Texture
    {
    public:
        void Bind(u32) const override {}
        u32 GetWidth() const override { return 1; }
        u32 GetHeight() const override { return 1; }
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
        fs::path path;
    };
}

TEST_CASE("DebugOutputCatalog: views and generations qualify identical names and shared fallbacks")
{
    DebugOutputCatalog catalog;
    const auto scene = std::make_shared<TestTexture>();
    const auto game = std::make_shared<TestTexture>();
    const auto shared = std::make_shared<TestTexture>();
    catalog.ReplaceShared({{"ShadowMap", shared}});
    catalog.ReplaceView({1}, 3, {{"SceneColor", scene}});
    catalog.ReplaceView({2}, 7, {{"SceneColor", game}});
    CHECK(catalog.Find({1}, 3, "SceneColor") == scene);
    CHECK(catalog.Find({2}, 7, "SceneColor") == game);
    CHECK(catalog.Find({1}, 3, "ShadowMap") == shared);
    CHECK(catalog.Find({2}, 7, "ShadowMap") == shared);
    CHECK_FALSE(catalog.Find({1}, 2, "SceneColor"));
    CHECK_FALSE(catalog.Find({1}, 2, "ShadowMap"));
    CHECK_FALSE(catalog.Find({3}, 7, "ShadowMap"));
    CHECK_FALSE(catalog.Find({1}, 3, "Missing"));
    catalog.ReplaceView({1}, 4, {});
    CHECK_FALSE(catalog.Find({1}, 3, "SceneColor"));
    CHECK_FALSE(catalog.Find({1}, 4, "SceneColor"));
    CHECK(catalog.Find({2}, 7, "SceneColor") == game);
}

TEST_CASE("DebugOutputCatalog: replacement and release never retain physical resources")
{
    DebugOutputCatalog catalog;
    auto texture = std::make_shared<TestTexture>();
    std::weak_ptr<Texture> retired = texture;
    catalog.ReplaceView({1}, 1, {{"Output", texture}});
    catalog.ReplaceShared({{"Shared", texture}});
    CHECK(texture.use_count() == 1);
    texture.reset();
    CHECK(retired.expired());
    CHECK_FALSE(catalog.Find({1}, 1, "Output"));
    CHECK_FALSE(catalog.Find({1}, 1, "Shared"));
    auto replacement = std::make_shared<TestTexture>();
    catalog.ReplaceView({1}, 2, {{"Output", replacement}});
    catalog.ReplaceShared({{"Shared", replacement}});
    CHECK(catalog.Find({1}, 2, "Output") == replacement);
    CHECK(catalog.Find({1}, 2, "Shared") == replacement);
    catalog.Release({1});
    CHECK_FALSE(catalog.Find({1}, 2, "Output"));
    CHECK_FALSE(catalog.Find({1}, 2, "Shared"));
    catalog.ReplaceView({2}, 1, {});
    catalog.Clear();
    CHECK_FALSE(catalog.Find({2}, 1, "Shared"));
}

TEST_CASE("DebugOutputCatalog: malformed replacement preserves the previous scope")
{
    DebugOutputCatalog catalog;
    auto texture = std::make_shared<TestTexture>();
    catalog.ReplaceView({1}, 1, {{"Output", texture}});
    catalog.ReplaceShared({{"Shared", texture}});
    CHECK_THROWS_AS(catalog.ReplaceView({0}, 1, {}), std::invalid_argument);
    CHECK_THROWS_AS(catalog.ReplaceView({1}, 0, {}), std::invalid_argument);
    CHECK_THROWS_AS(catalog.ReplaceView({1}, 2, {{"Output", texture}, {"Output", {}}}), std::invalid_argument);
    CHECK_THROWS_AS(catalog.ReplaceView({1}, 2, {{"", texture}}), std::invalid_argument);
    CHECK_THROWS_AS(catalog.ReplaceShared({{"Shared", texture}, {"Shared", {}}}), std::invalid_argument);
    CHECK(catalog.Find({1}, 1, "Output") == texture);
    CHECK(catalog.Find({1}, 1, "Shared") == texture);
    // An explicit unavailable local output shadows a shared name.
    catalog.ReplaceView({1}, 2, {{"Shared", {}}});
    CHECK_FALSE(catalog.Find({1}, 2, "Shared"));
}

TEST_CASE("DebugOutputCatalog: released identity cannot resolve a reopened target")
{
    RenderViewRegistry views;
    DebugOutputCatalog catalog;
    int targets = 0;
    auto texture = std::make_shared<TestTexture>();
    const auto oldId = views.Register(&targets);
    catalog.ReplaceView(oldId, 1, {{"Output", texture}});
    catalog.Release(oldId); views.Release(oldId);
    const auto newId = views.Register(&targets);
    catalog.ReplaceView(newId, 1, {{"Output", texture}});
    CHECK_FALSE(oldId == newId);
    CHECK_FALSE(catalog.Find(oldId, 1, "Output"));
    CHECK(catalog.Find(newId, 1, "Output") == texture);
}

TEST_CASE("DebugOutputCatalog: native cold view preserves target names without allocations")
{
    FrameTargets targets;
    DebugOutputCatalog catalog;
    const auto outputs = CollectViewDebugOutputs(targets, nullptr, nullptr);
    REQUIRE(outputs.size() == 8);
    catalog.ReplaceView({1}, 1, outputs);
    for (const auto& output : outputs)
    {
        CHECK_FALSE(output.name.empty());
        CHECK(output.texture.expired());
        CHECK_FALSE(catalog.Find({1}, 1, output.name));
    }
}

TEST_CASE("DebugOutputCatalog: native domain contributions borrow the selected view states")
{
    FrameTargets targets;
    ViewResources state;
    GtaoViewState ao;
    state.bloom = std::make_shared<BloomViewState>();
    state.fog = std::make_shared<FogViewState>();
    for (auto& mip : state.bloom->mips) mip = std::make_shared<TestTexture>();
    state.fog->volDensity = std::make_shared<TestTexture>();
    state.fog->volInScatter = std::make_shared<TestTexture>();
    state.fog->volInScatterHistA = std::make_shared<TestTexture>();
    state.fog->volInScatterHistB = std::make_shared<TestTexture>();
    state.reflRadiance = std::make_shared<TestTexture>();
    ao.linearDepth = std::make_shared<TestTexture>();
    ao.rawAO = std::make_shared<TestTexture>();
    ao.finalAO = std::make_shared<TestTexture>();
    DebugOutputCatalog catalog;
    catalog.ReplaceView({1}, 1, CollectViewDebugOutputs(targets, &state, &ao));
    catalog.ReplaceView({2}, 1, CollectViewDebugOutputs(targets, nullptr, nullptr));
    for (u32 i = 0; i < BloomViewState::kMipCount; ++i)
    {
        CHECK(catalog.Find({1}, 1, "BloomMip" + std::to_string(i)) == state.bloom->mips[i]);
        CHECK_FALSE(catalog.Find({2}, 1, "BloomMip" + std::to_string(i)));
        CHECK(state.bloom->mips[i].use_count() == 1);
    }
    CHECK(catalog.Find({1}, 1, "VolDensity") == state.fog->volDensity);
    CHECK(catalog.Find({1}, 1, "VolInScatter") == state.fog->volInScatter);
    CHECK(catalog.Find({1}, 1, "VolInScatterHistA") == state.fog->volInScatterHistA);
    CHECK(catalog.Find({1}, 1, "VolInScatterHistB") == state.fog->volInScatterHistB);
    CHECK(catalog.Find({1}, 1, "Reflections") == state.reflRadiance);
    CHECK(catalog.Find({1}, 1, "GTAOLinearDepth") == ao.linearDepth);
    CHECK(catalog.Find({1}, 1, "GTAORawAO") == ao.rawAO);
    CHECK(catalog.Find({1}, 1, "GTAOFinal") == ao.finalAO);
    catalog.ReplaceView({1}, 2, CollectViewDebugOutputs(targets, nullptr, nullptr));
    CHECK_FALSE(catalog.Find({1}, 2, "VolDensity"));
    CHECK_FALSE(catalog.Find({1}, 2, "GTAOFinal"));
}
