#pragma once
#include "luth/renderer/shader/ShaderReloadFanout.h"
#include "luth/renderer/shader/ShaderWatcher.h"

namespace Luth
{
    // One main-thread reload coordinator per renderer. Stop before borrowed native clients retire.
    class ShaderReloadCoordinator
    {
    public:
        ~ShaderReloadCoordinator() { Stop(); }
        void AddConsumer(std::string name, ShaderReloadFanout::Consumer consumer)
        {
            if (m_Started) throw std::logic_error("Register reload consumers before starting the watcher");
            m_Consumers.Add(std::move(name), std::move(consumer));
        }
        void Start(const std::filesystem::path& engineDirectory);
        void Stop();
        void Poll() { m_Watcher.Poll(); }
        void AddProjectDir(const std::filesystem::path& path) { m_Watcher.AddProjectDir(path); }
        void RemoveProjectDir() { m_Watcher.RemoveProjectDir(); }
        ShaderWatcher& Watcher() { return m_Watcher; } // Compatibility only.
    private:
        ShaderWatcher m_Watcher;
        ShaderReloadFanout m_Consumers;
        bool m_Started = false;
    };
}
