#include "luthpch.h"
#include "ShaderVariantCache.h"
#include <algorithm>
#include <stdexcept>

namespace Luth
{
    ShaderVariantCache::ShaderVariantCache(Compiler compiler)
        : m_Compiler(compiler ? std::move(compiler) : Compiler([](const fs::path& path, ShaderCompileVariant variant) {
            return SlangCompiler::CompileReflectStage(path, "main", variant);
        })) {}

    fs::path ShaderVariantCache::NormalizeSource(const fs::path& path)
    { return fs::weakly_canonical(fs::absolute(path)); }

    std::string ShaderVariantCache::Name(const fs::path& path, ShaderCompileVariant variant)
    {
        if (variant != ShaderCompileVariant::Raster && variant != ShaderCompileVariant::Hybrid)
            throw std::invalid_argument("Shader variants require Raster or Hybrid; legacy assets use ShaderLibrary");
        return NormalizeSource(path).generic_string() + (variant == ShaderCompileVariant::Raster ? "#raster" : "#hybrid");
    }

    std::shared_ptr<const CompiledShaderVariant> ShaderVariantCache::Load(const fs::path& path, ShaderCompileVariant variant)
    {
        const auto name = Name(path, variant);
        if (const auto existing = Find(name)) return existing;
        const auto source = NormalizeSource(path);
        auto result = m_Compiler(source, variant);
        if (result.spirv.empty() || result.stage == ShaderStage::Unknown) return {};
        auto compiled = std::make_shared<CompiledShaderVariant>(CompiledShaderVariant{
            source, variant, result.stage, std::move(result.spirv), 1});
        m_Entries.emplace(name, compiled);
        return compiled;
    }

    std::shared_ptr<const CompiledShaderVariant> ShaderVariantCache::Find(const std::string& name) const
    {
        const auto it = m_Entries.find(name);
        return it == m_Entries.end() ? nullptr : it->second;
    }

    std::shared_ptr<const CompiledShaderVariant> ShaderVariantCache::Reload(const std::string& name)
    {
        const auto old = Find(name);
        if (!old) return {};
        auto result = m_Compiler(old->source, old->variant);
        if (result.spirv.empty() || result.stage != old->stage) return {};
        auto next = std::make_shared<CompiledShaderVariant>(CompiledShaderVariant{
            old->source, old->variant, result.stage, std::move(result.spirv), old->generation + 1});
        m_Entries.at(name) = next;
        return next;
    }

    std::vector<std::string> ShaderVariantCache::Names() const
    {
        std::vector<std::string> names;
        for (const auto& [name, entry] : m_Entries) names.push_back(name);
        std::sort(names.begin(), names.end());
        return names;
    }
}
