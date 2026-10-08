#pragma once

#include "luth/assets/FileWatcher.h"

#include <filesystem>
#include <mutex>
#include <set>
#include <string>

namespace Luth
{
    // Background detections queue shader names for main-thread ShaderLibrary reload.
    // Owned by RenderingSystem through ShaderReloadCoordinator. Poll runs once in
    // RenderingSystem::Update before frozen/normal view recording; stop precedes native teardown.
    class ShaderWatcher
    {
    public:
        void Start(const std::filesystem::path& engineShadersDir);
        void Stop();

        void AddProjectDir(const std::filesystem::path& projectShadersDir);
        void RemoveProjectDir();

        void Poll();

    private:
        void Enqueue(const std::string& shaderName);

        FileWatcher           m_Watcher;
        std::filesystem::path m_ProjectDir;
        std::mutex            m_Mutex;
        std::set<std::string> m_Pending;
    };
}
