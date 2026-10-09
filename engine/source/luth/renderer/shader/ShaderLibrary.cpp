#include "luthpch.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/assets/AssetDatabase.h"
#include "luth/assets/AssetManager.h"
#include "luth/assets/FileSystem.h"
#include "luth/core/diagnostics/Log.h"

namespace Luth
{
    std::unordered_map<std::string, std::shared_ptr<Shader>> ShaderLibrary::s_Shaders;
    std::function<void(const std::string&)> ShaderLibrary::s_ReloadCallback;
    ShaderVariantCache ShaderLibrary::s_Variants;
    std::function<void(const std::string&, const std::vector<u32>&)> ShaderLibrary::s_VariantReloadCallback;

    namespace
    {
        // Session import roots change with the project. Never reuse compiled variants
        // from the prior environment; refresh registrations while retaining snapshots for retiring owners.
        bool VariantEnvironmentChanged()
        {
            static fs::path engineRoot, projectRoot;
            static bool initialized = false;
            const auto assets = FileSystem::EngineAssetsPath();
            const auto engine = assets.empty() ? fs::path{} : ShaderVariantCache::NormalizeSource(assets);
            const auto project = FileSystem::HasProject()
                ? ShaderVariantCache::NormalizeSource(FileSystem::ProjectPath("Library/Generated/shaders")) : fs::path{};
            if (initialized && engineRoot == engine && projectRoot == project) return false;
            engineRoot = engine;
            projectRoot = project;
            initialized = true;
            return true;
        }
    }
    void ShaderLibrary::Init()
    {
        s_Shaders.clear();
        s_Variants.Clear();
        s_VariantReloadCallback = nullptr;
        s_ReloadCallback = nullptr;
    }

    void ShaderLibrary::Shutdown()
    {
        s_VariantReloadCallback = nullptr;
        s_Variants.Clear();
        s_ReloadCallback = nullptr;
        s_Shaders.clear();
    }

    void ShaderLibrary::Register(const std::string& name, std::shared_ptr<Shader> shader)
    {
        if (s_Shaders.count(name))
            LH_LOG(Shaders, warn, "ShaderLibrary: overwriting existing shader '{}'", name);

        s_Shaders[name] = shader;
        LH_LOG(Shaders, debug, "ShaderLibrary: registered '{}'", name);
    }

    std::shared_ptr<Shader> ShaderLibrary::Get(const std::string& name)
    {
        auto it = s_Shaders.find(name);
        if (it == s_Shaders.end())
        {
            LH_LOG(Shaders, error, "ShaderLibrary: shader '{}' not found", name);
            return nullptr;
        }
        return it->second;
    }

    const std::unordered_map<std::string, std::shared_ptr<Shader>>& ShaderLibrary::GetAll()
    {
        return s_Shaders;
    }

    std::shared_ptr<Shader> ShaderLibrary::LoadEngine(const std::string& engineRelPath)
    {
        LH_PROFILE_FUNCTION();
        fs::path abs = FileSystem::EngineAssetsPath(engineRelPath);
        std::string key = abs.filename().string();

        // Idempotent: return cached entry
        if (auto it = s_Shaders.find(key); it != s_Shaders.end())
            return it->second;

        UUID uuid = AssetDatabase::GetUUID(abs);
        auto sh = std::static_pointer_cast<Shader>(AssetManager::LoadImmediate(uuid));
        if (!sh)
        {
            LH_LOG(Shaders, error, "ShaderLibrary::LoadEngine: failed to load '{0}'", engineRelPath);
            return nullptr;
        }

        Register(key, sh);
        return sh;
    }

    bool ShaderLibrary::Reload(const std::string& name)
    {
        LH_PROFILE_FUNCTION();
        if (s_Variants.Find(name))
        {
            const auto next = s_Variants.Reload(name);
            if (!next)
            {
                LH_LOG(Shaders, error, "Shader variant reload '{}' failed - keeping prior compiled program", name);
                return false;
            }
            if (s_VariantReloadCallback) s_VariantReloadCallback(name, next->spirv);
            return true;
        }
        auto it = s_Shaders.find(name);
        if (it == s_Shaders.end())
        {
            LH_LOG(Shaders, error, "ShaderLibrary::Reload: shader '{}' not found", name);
            return false;
        }

        LH_LOG(Shaders, info, "ShaderLibrary: reloading '{}'...", name);
        it->second->Reload();

        if (!it->second->IsValid())
        {
            LH_LOG(Shaders, error, "ShaderLibrary: reload of '{}' failed -- keeping old pipeline", name);
            return false;
        }

        if (s_ReloadCallback)
            s_ReloadCallback(name);

        return true;
    }

    void ShaderLibrary::SetReloadCallback(std::function<void(const std::string&)> cb)
    {
        s_ReloadCallback = std::move(cb);
    }

    std::shared_ptr<const CompiledShaderVariant> ShaderLibrary::LoadEngineVariant(
        const std::string& path, ShaderCompileVariant variant)
    {
        if (VariantEnvironmentChanged()) ReloadVariants();
        return s_Variants.Load(FileSystem::EngineAssetsPath(path), variant);
    }

    void ShaderLibrary::SetVariantReloadCallback(std::function<void(const std::string&, const std::vector<u32>&)> callback)
    { s_VariantReloadCallback = std::move(callback); }

    void ShaderLibrary::ReloadVariants()
    {
        (void)VariantEnvironmentChanged();
        for (const auto& name : s_Variants.Names()) Reload(name);
    }

    void ShaderLibrary::ReloadSource(const fs::path& source)
    { ReloadSources({source}); }

    void ShaderLibrary::ReloadSources(const std::vector<fs::path>& sources)
    {
        if (sources.empty()) return;
        (void)VariantEnvironmentChanged();
        std::vector<std::string> names;
        for (const auto& [name, shader] : s_Shaders)
            for (const auto& source : sources)
                if (ShaderVariantCache::NormalizeSource(shader->GetPath()) == ShaderVariantCache::NormalizeSource(source))
                { names.push_back(name); break; }
        // Imports share session defines. Conservatively refresh all cached variants on
        // source edits until dependency tracking exists; each retains its own options.
        const auto variants = s_Variants.Names();
        names.insert(names.end(), variants.begin(), variants.end());
        for (const auto& name : names) Reload(name);
    }
}
