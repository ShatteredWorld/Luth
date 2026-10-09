#include <doctest/doctest.h>
#include "luth/renderer/shader/ShaderVariantCache.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include <spirv.hpp>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include "luth/core/types/LuthMath.h"
#include "luth/assets/FileSystem.h"
using namespace Luth;

namespace
{
    bool HasInstruction(const std::vector<u32>& words, spv::Op op, u32 operand = ~u32(0))
    {
        if (words.size() < 5) throw std::runtime_error("Invalid SPIR-V header");
        for (size_t i = 5; i < words.size();)
        {
            const auto count = words[i] >> 16;
            if (!count || i + count > words.size()) throw std::runtime_error("Invalid SPIR-V instruction bounds");
            if ((words[i] & 0xffff) == static_cast<u32>(op)
                && (operand == ~u32(0) || (count > 1 && words[i + 1] == operand))) return true;
            i += count;
        }
        return false;
    }
    fs::path ProbePath() { return fs::absolute("tests/renderer/shaders/VariantProbe.slang"); }
    struct LibraryScope { LibraryScope() { ShaderLibrary::Init(); } ~LibraryScope() { ShaderLibrary::Shutdown(); } };
    struct LegacyShader final : Shader
    {
        explicit LegacyShader(fs::path source) : path(std::move(source)) {}
        fs::path path;
        std::vector<u32> words{1};
        u32 reloads = 0;
        ShaderStage GetStage() const override { return ShaderStage::Compute; }
        const std::vector<u32>& GetSpirV() const override { return words; }
        const fs::path& GetPath() const override { return path; }
        bool IsValid() const override { return true; }
        void Reload() override { ++reloads; }
    };
}

TEST_CASE("ShaderVariants: canonical source and variant identities isolate cached programs")
{
    u32 calls = 0;
    ShaderVariantCache cache([&](const fs::path&, ShaderCompileVariant variant) {
        ++calls;
        return SlangCompiler::CompileOutput{{variant == ShaderCompileVariant::Raster ? 1u : 2u}, ShaderStage::Compute};
    });
    const auto source = ProbePath();
    const auto raster = cache.Load(source, ShaderCompileVariant::Raster);
    const auto hybrid = cache.Load(source, ShaderCompileVariant::Hybrid);
    REQUIRE(raster); REQUIRE(hybrid);
    CHECK(raster != hybrid); CHECK(raster->spirv != hybrid->spirv);
    CHECK(cache.Load(source.parent_path() / "." / source.filename(), ShaderCompileVariant::Raster) == raster);
    CHECK(calls == 2);
    CHECK(raster->variant == ShaderCompileVariant::Raster);
    CHECK(hybrid->variant == ShaderCompileVariant::Hybrid);
    CHECK(raster->generation == 1);
    CHECK(ShaderVariantCache::Name(source, ShaderCompileVariant::Raster)
        != ShaderVariantCache::Name(source.parent_path() / "other" / source.filename(), ShaderCompileVariant::Raster));
    CHECK_THROWS_AS(cache.Load(source, ShaderCompileVariant::Legacy), std::invalid_argument);
    CHECK(cache.Names().size() == 2);
    cache.Clear(); CHECK(cache.Names().empty()); CHECK(raster->spirv[0] == 1);
}

TEST_CASE("ShaderVariants: reload preserves options immutable snapshots and last valid code")
{
    u32 version = 1;
    ShaderStage stage = ShaderStage::Compute;
    bool fail = false;
    ShaderCompileVariant seen = ShaderCompileVariant::Legacy;
    ShaderVariantCache cache([&](const fs::path&, ShaderCompileVariant variant) {
        seen = variant;
        return SlangCompiler::CompileOutput{fail ? std::vector<u32>{} : std::vector<u32>{version}, stage};
    });
    const auto path = ProbePath();
    const auto name = ShaderVariantCache::Name(path, ShaderCompileVariant::Raster);
    const auto original = cache.Load(path, ShaderCompileVariant::Raster);
    const auto hybrid = cache.Load(path, ShaderCompileVariant::Hybrid);
    version = 2;
    const auto refreshed = cache.Reload(name);
    REQUIRE(refreshed);
    CHECK(seen == ShaderCompileVariant::Raster); CHECK(refreshed->generation == 2);
    CHECK(refreshed->spirv[0] == 2); CHECK(original->generation == 1); CHECK(original->spirv[0] == 1);
    CHECK(cache.Load(path, ShaderCompileVariant::Hybrid) == hybrid);
    fail = true;
    CHECK_FALSE(cache.Reload(name)); CHECK(cache.Find(name) == refreshed);
    fail = false; stage = ShaderStage::Fragment;
    CHECK_FALSE(cache.Reload(name)); CHECK(cache.Find(name) == refreshed);
    CHECK_FALSE(cache.Reload("missing"));
    cache.Clear(); stage = ShaderStage::Unknown;
    CHECK_FALSE(cache.Load(path, ShaderCompileVariant::Raster)); CHECK(cache.Names().empty());
    stage = ShaderStage::Compute;
    CHECK(cache.Load(path, ShaderCompileVariant::Raster)->generation == 1);
}

TEST_CASE("ShaderVariants: Slang session defines propagate into imported ray-query modules")
{
    REQUIRE(SlangCompiler::Available());
    const auto raster = SlangCompiler::CompileReflectStage(ProbePath(), "main", ShaderCompileVariant::Raster);
    const auto hybrid = SlangCompiler::CompileReflectStage(ProbePath(), "main", ShaderCompileVariant::Hybrid);
    REQUIRE_FALSE(raster.spirv.empty()); REQUIRE_FALSE(hybrid.spirv.empty());
    CHECK(raster.stage == ShaderStage::Compute); CHECK(hybrid.stage == ShaderStage::Compute);
    CHECK_FALSE(HasInstruction(raster.spirv, spv::OpCapability, spv::CapabilityRayQueryKHR));
    CHECK_FALSE(HasInstruction(raster.spirv, spv::OpCapability, spv::CapabilityRayTracingKHR));
    CHECK_FALSE(HasInstruction(raster.spirv, spv::OpTypeAccelerationStructureKHR));
    CHECK(HasInstruction(hybrid.spirv, spv::OpCapability, spv::CapabilityRayQueryKHR));
    CHECK(HasInstruction(hybrid.spirv, spv::OpTypeAccelerationStructureKHR));
    const auto legacy = SlangCompiler::CompileReflectStage(ProbePath());
    REQUIRE_FALSE(legacy.spirv.empty()); CHECK(legacy.stage == ShaderStage::Compute);
    CHECK(HasInstruction(legacy.spirv, spv::OpCapability, spv::CapabilityRayQueryKHR));
}

TEST_CASE("ShaderVariants: source reload notifies every legacy alias and both isolated variants")
{
    LibraryScope library;
    const auto path = ProbePath();
    auto legacy = std::make_shared<LegacyShader>(path);
    ShaderLibrary::Register("alias-a", legacy); ShaderLibrary::Register("alias-b", legacy);
    const auto raster = ShaderLibrary::LoadEngineVariant(path.string(), ShaderCompileVariant::Raster);
    const auto hybrid = ShaderLibrary::LoadEngineVariant(path.string(), ShaderCompileVariant::Hybrid);
    REQUIRE(raster); REQUIRE(hybrid);
    std::vector<std::string> legacyNotifications, variantNotifications;
    ShaderLibrary::SetReloadCallback([&](const std::string& name) { legacyNotifications.push_back(name); });
    ShaderLibrary::SetVariantReloadCallback([&](const std::string& name, const std::vector<u32>& words) {
        CHECK_FALSE(words.empty()); variantNotifications.push_back(name);
    });
    ShaderLibrary::ReloadSource(path);
    CHECK(legacy->reloads == 2); CHECK(legacyNotifications.size() == 2);
    REQUIRE(variantNotifications.size() == 2); CHECK(variantNotifications[0] != variantNotifications[1]);
    CHECK(raster->generation == 1); CHECK(hybrid->generation == 1);
    CHECK(ShaderLibrary::LoadEngineVariant(path.string(), ShaderCompileVariant::Raster)->generation == 2);
    ShaderLibrary::ReloadSource(path.parent_path() / "VariantCommon.slang");
    CHECK(legacy->reloads == 2); // Legacy API remains filename/source driven.
    CHECK(variantNotifications.size() == 4); // Conservative imported-module invalidation.
    CHECK(ShaderLibrary::LoadEngineVariant(path.string(), ShaderCompileVariant::Raster)->generation == 3);
    CHECK(ShaderLibrary::LoadEngineVariant(path.string(), ShaderCompileVariant::Hybrid)->generation == 3);
    ShaderLibrary::ReloadSources({path, path.parent_path() / "VariantCommon.slang", path});
    CHECK(legacy->reloads == 4); CHECK(variantNotifications.size() == 6);
    CHECK(ShaderLibrary::LoadEngineVariant(path.string(), ShaderCompileVariant::Raster)->generation == 4);
    CHECK(ShaderLibrary::LoadEngineVariant(path.string(), ShaderCompileVariant::Hybrid)->generation == 4);
}

TEST_CASE("ShaderVariants: production transparency and fog have independent raster programs")
{
    for (const auto* source : {"pbr_transparent.slang", "pbr_oit_store.slang", "volumetric_inject_scatter.slang"})
    {
        CAPTURE(source);
        const auto path = fs::absolute(fs::path("engine/assets/shaders") / source);
        const auto raster = SlangCompiler::CompileReflectStage(path, "main", ShaderCompileVariant::Raster);
        const auto hybrid = SlangCompiler::CompileReflectStage(path, "main", ShaderCompileVariant::Hybrid);
        const auto legacy = SlangCompiler::CompileReflectStage(path);
        REQUIRE_FALSE(raster.spirv.empty()); REQUIRE_FALSE(hybrid.spirv.empty()); REQUIRE_FALSE(legacy.spirv.empty());
        CHECK(raster.stage == hybrid.stage); CHECK(hybrid.stage == legacy.stage);
        CHECK_FALSE(HasInstruction(raster.spirv, spv::OpCapability, spv::CapabilityRayQueryKHR));
        CHECK_FALSE(HasInstruction(raster.spirv, spv::OpCapability, spv::CapabilityRayTracingKHR));
        CHECK_FALSE(HasInstruction(raster.spirv, spv::OpTypeAccelerationStructureKHR));
        CHECK(HasInstruction(hybrid.spirv, spv::OpCapability, spv::CapabilityRayQueryKHR));
        CHECK(HasInstruction(hybrid.spirv, spv::OpTypeAccelerationStructureKHR));
        CHECK(HasInstruction(legacy.spirv, spv::OpCapability, spv::CapabilityRayQueryKHR));
        // Optional standalone artifacts for Vulkan SDK spirv-val verification.
        if (const auto* output = std::getenv("LUTH_SHADER_VARIANT_DUMP"))
            for (const auto& [suffix, code] : {std::pair{"raster", &raster}, std::pair{"hybrid", &hybrid}, std::pair{"legacy", &legacy}})
            {
                std::ofstream stream(fs::path(output) / (std::string(source) + "." + suffix + ".spv"), std::ios::binary);
                stream.write(reinterpret_cast<const char*>(code->spirv.data()), code->spirv.size() * sizeof(u32));
                REQUIRE(stream.good());
            }
    }
}
TEST_CASE("ShaderVariants: import-root changes retain native registrations and notify refreshed programs")
{
    LibraryScope library;
    const auto engine = FileSystem::EnginePath();
    struct RestoreRoot { fs::path root; ~RestoreRoot() { FileSystem::InitEngine(root); } } restore{engine};
    const auto source = ProbePath();
    const auto raster = ShaderLibrary::LoadEngineVariant(source.string(), ShaderCompileVariant::Raster);
    const auto hybrid = ShaderLibrary::LoadEngineVariant(source.string(), ShaderCompileVariant::Hybrid);
    REQUIRE(raster); REQUIRE(hybrid);
    std::vector<std::string> notified;
    ShaderLibrary::SetVariantReloadCallback([&](const auto& name, const auto&) { notified.push_back(name); });
    FileSystem::InitEngine(source.parent_path()); // Different import roots; probe's own import remains local.
    ShaderLibrary::ReloadSource(source.parent_path() / "VariantCommon.slang");
    REQUIRE(notified.size() == 2);
    CHECK(ShaderLibrary::LoadEngineVariant(source.string(), ShaderCompileVariant::Raster)->generation == 2);
    CHECK(ShaderLibrary::LoadEngineVariant(source.string(), ShaderCompileVariant::Hybrid)->generation == 2);
    CHECK(raster->generation == 1); CHECK(hybrid->generation == 1);
    FileSystem::InitEngine(engine);
    CHECK(ShaderLibrary::LoadEngineVariant(source.string(), ShaderCompileVariant::Raster)->generation == 3);
    CHECK(notified.size() == 4);
    ShaderLibrary::ReloadVariants();
    CHECK(ShaderLibrary::LoadEngineVariant(source.string(), ShaderCompileVariant::Hybrid)->generation == 4);
    CHECK(notified.size() == 6);
}