#include "luthpch.h"
#include "luth/renderer/shader/ShaderWatcher.h"
#include "luth/renderer/shader/ShaderLibrary.h"

namespace Luth
{
    void ShaderWatcher::Start(const fs::path& engineShadersDir)
    {
        m_Watcher.AddWatch(engineShadersDir);
        m_Watcher.SetCallback([this](const fs::path& changedFile, FileWatcher::FileStatus status) {
            if (status != FileWatcher::FileStatus::Modified) return;

            std::string ext = changedFile.extension().string();
            if (ext != ".slang") return;

            std::lock_guard lock(m_Mutex);
            m_PendingSources.insert(changedFile);
        });
        m_Watcher.Start(true);
    }

    void ShaderWatcher::Stop()
    {
        m_Watcher.Stop();
        std::lock_guard lock(m_Mutex);
        m_PendingSources.clear();
    }

    void ShaderWatcher::AddProjectDir(const fs::path& projectShadersDir)
    {
        if (!fs::exists(projectShadersDir) || !fs::is_directory(projectShadersDir))
            return;
        m_Watcher.AddWatch(projectShadersDir);
        m_ProjectDir = projectShadersDir;
        LH_LOG(Shaders, info, "Shader hot-reload watching project dir: {}", projectShadersDir.string());
    }

    void ShaderWatcher::RemoveProjectDir()
    {
        if (m_ProjectDir.empty()) return;
        m_Watcher.RemoveWatch(m_ProjectDir);
        m_ProjectDir.clear();
    }

    void ShaderWatcher::Poll()
    {
        std::set<fs::path> sources;
        {
            std::lock_guard lock(m_Mutex);
            sources.swap(m_PendingSources);
        }
        ShaderLibrary::ReloadSources(std::vector<fs::path>(sources.begin(), sources.end()));
    }
}
