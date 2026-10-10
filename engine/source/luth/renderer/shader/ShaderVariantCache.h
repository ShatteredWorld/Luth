#pragma once
#include "SlangCompiler.h"
#include <functional>
#include <memory>
#include <unordered_map>

namespace Luth
{
    struct CompiledShaderVariant
    {
        fs::path source;
        ShaderCompileVariant variant;
        ShaderStage stage;
        std::vector<u32> spirv;
        u64 generation;
    };
    // CPU shader programs, immutable once published. Native domains own pipelines/modules.
    class ShaderVariantCache
    {
    public:
        using Compiler = std::function<SlangCompiler::CompileOutput(const fs::path&, ShaderCompileVariant)>;
        explicit ShaderVariantCache(Compiler compiler = {});
        static fs::path NormalizeSource(const fs::path&);
        static std::string Name(const fs::path&, ShaderCompileVariant);
        std::shared_ptr<const CompiledShaderVariant> Load(const fs::path&, ShaderCompileVariant);
        std::shared_ptr<const CompiledShaderVariant> Find(const std::string&) const;
        std::shared_ptr<const CompiledShaderVariant> Reload(const std::string&);
        std::vector<std::string> Names() const;
        void Clear() { m_Entries.clear(); }
    private:
        Compiler m_Compiler;
        std::unordered_map<std::string, std::shared_ptr<const CompiledShaderVariant>> m_Entries;
    };
}
