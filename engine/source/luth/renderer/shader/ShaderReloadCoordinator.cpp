#include "luthpch.h"
#include "luth/renderer/shader/ShaderReloadCoordinator.h"
#include "luth/renderer/shader/ShaderLibrary.h"
#include "luth/renderer/backend/vulkan/VulkanShader.h"

namespace Luth
{
    void ShaderReloadCoordinator::Start(const std::filesystem::path& directory)
    {
        if (m_Started) throw std::logic_error("Shader reload coordinator already started");
        ShaderLibrary::SetReloadCallback([this](const std::string& name) {
            auto shader = std::static_pointer_cast<VulkanShader>(ShaderLibrary::Get(name));
            if (!shader || !shader->IsValid())
            {
                LH_LOG(Renderer, error, "Shader reload: '{}' invalid - keeping existing pipelines", name);
                return;
            }
            // Native clients preserve their existing deferred pipeline destruction policy.
            const auto refreshed = m_Consumers.Notify(name, shader->GetSpirV());
            if (!refreshed.empty())
                LH_LOG(Renderer, info, "Shader reload '{}' refreshed {} native clients", name, refreshed.size());
            // IBL precompute blobs refresh in ShaderLibrary; skybox rebaking remains explicit.
        });
        m_Started = true;
        ShaderLibrary::SetVariantReloadCallback([this](const std::string& name, const std::vector<u32>& spirv) {
            m_Consumers.Notify(name, spirv);
        });
        m_Watcher.Start(directory);
    }

    void ShaderReloadCoordinator::Stop()
    {
        if (!m_Started) return;
        m_Watcher.Stop();
        ShaderLibrary::SetReloadCallback(nullptr);
        ShaderLibrary::SetVariantReloadCallback(nullptr);
        m_Started = false;
    }
}
