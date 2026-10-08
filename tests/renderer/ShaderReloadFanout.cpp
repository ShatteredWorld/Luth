#include <doctest/doctest.h>
#include "luth/renderer/shader/ShaderReloadFanout.h"
using namespace Luth;

TEST_CASE("ShaderReloadFanout: shared SVGF shaders reach all four channels")
{
    ShaderReloadFanout fanout;
    std::vector<std::string> visited;
    const std::vector<u32> spv{0x07230203, 17};
    for (const std::string channel : {"DI", "GI", "Reflections", "DI_Specular"})
        fanout.Add(channel, [&, channel](const auto& name, const auto& words) {
            visited.push_back(channel);
            CHECK(&words == &spv);
            return name == "svgf_moments.slang" || name == "svgf_passthrough.slang"
                || name == (channel == "Reflections" ? "svgf_spec_reproject.slang" : "svgf_reproject.slang");
        });
    CHECK(fanout.Notify("svgf_moments.slang", spv) == std::vector<std::string>{"DI", "GI", "Reflections", "DI_Specular"});
    CHECK(visited == std::vector<std::string>{"DI", "GI", "Reflections", "DI_Specular"});
    visited.clear();
    CHECK(fanout.Notify("svgf_reproject.slang", spv) == std::vector<std::string>{"DI", "GI", "DI_Specular"});
    CHECK(visited.size() == 4);
    CHECK(fanout.Notify("svgf_spec_reproject.slang", spv) == std::vector<std::string>{"Reflections"});
    CHECK(fanout.Notify("svgf_passthrough.slang", spv).size() == 4);
}

TEST_CASE("ShaderReloadFanout: fullscreen material and parity clients do not mask siblings")
{
    ShaderReloadFanout fanout;
    u32 calls = 0;
    fanout.Add("Geometry", [&](const auto& name, const auto&) { ++calls; return name == "pbr_vert.slang"; });
    fanout.Add("Transparency", [&](const auto& name, const auto&) { ++calls; return name == "pbr_vert.slang"; });
    fanout.Add("PostProcess", [&](const auto& name, const auto&) { ++calls; return name == "fullscreen.slang"; });
    fanout.Add("EditorOverlays", [&](const auto& name, const auto&) { ++calls; return name == "fullscreen.slang"; });
    fanout.Add("ReSTIR_GI", [&](const auto& name, const auto&) { ++calls; return name == "restir_gi_initial.slang"; });
    fanout.Add("SlangParity", [&](const auto& name, const auto&) { ++calls; return name == "restir_gi_initial.slang"; });
    CHECK(fanout.Notify("fullscreen.slang", {1}) == std::vector<std::string>{"PostProcess", "EditorOverlays"});
    CHECK(calls == 6);
    CHECK(fanout.Notify("pbr_vert.slang", {2}) == std::vector<std::string>{"Geometry", "Transparency"});
    CHECK(fanout.Notify("restir_gi_initial.slang", {3}) == std::vector<std::string>{"ReSTIR_GI", "SlangParity"});
    CHECK(fanout.Notify("unhandled.slang", {4}).empty());
    CHECK(calls == 24);
}

TEST_CASE("ShaderReloadFanout: invalid inputs never invoke clients and registrations are explicit")
{
    ShaderReloadFanout fanout;
    u32 calls = 0;
    auto callback = [&](const auto&, const auto&) { ++calls; return true; };
    CHECK_THROWS_AS(fanout.Add("", callback), std::invalid_argument);
    CHECK_THROWS_AS(fanout.Add("EmptyCallback", {}), std::invalid_argument);
    fanout.Add("CapturePreview", callback);
    CHECK_THROWS_AS(fanout.Add("CapturePreview", callback), std::invalid_argument);
    CHECK(fanout.Notify("debugBlit.slang", {}).empty());
    CHECK(fanout.Notify("", {1}).empty());
    CHECK(calls == 0);
    CHECK(fanout.Notify("debugDepth.slang", {1}) == std::vector<std::string>{"CapturePreview"});
    CHECK(calls == 1);
}
