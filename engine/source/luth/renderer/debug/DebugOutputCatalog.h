#pragma once

#include "luth/renderer/features/RenderViewState.h"
#include <string>
#include <vector>

namespace Luth
{
    class Texture;
    struct DebugOutput { std::string name; std::weak_ptr<Texture> texture; };
    using DebugOutputs = std::vector<DebugOutput>;

    // Tooling names only: no RG handles, descriptor ownership or image lifetime extension.
    // Accessed by the rendering/editor thread, outside parallel graph recording.
    class DebugOutputCatalog
    {
    public:
        void ReplaceShared(const DebugOutputs& outputs) { m_Shared = Validate(outputs); }
        void ReplaceView(RenderViewId id, u64 generation, const DebugOutputs& outputs)
        {
            if (!id.value || !generation) throw std::invalid_argument("Debug outputs require view identity and generation");
            auto names = Validate(outputs);
            m_Views.insert_or_assign(id.value, View{generation, std::move(names)});
        }
        std::shared_ptr<Texture> Find(RenderViewId id, u64 generation, const std::string& name) const
        {
            const auto view = m_Views.find(id.value);
            if (view == m_Views.end() || view->second.generation != generation) return {};
            const auto local = view->second.names.find(name);
            if (local != view->second.names.end()) return local->second.lock();
            const auto shared = m_Shared.find(name);
            return shared == m_Shared.end() ? nullptr : shared->second.lock();
        }
        void Release(RenderViewId id) { m_Views.erase(id.value); }
        void Clear() { m_Views.clear(); m_Shared.clear(); }
    private:
        using Names = std::unordered_map<std::string, std::weak_ptr<Texture>>;
        struct View { u64 generation; Names names; };
        static Names Validate(const DebugOutputs& outputs)
        {
            Names result;
            for (const auto& output : outputs)
                if (output.name.empty() || !result.emplace(output.name, output.texture).second)
                    throw std::invalid_argument("Debug output names must be nonempty and unique within their scope");
            return result;
        }
        Names m_Shared;
        std::unordered_map<u64, View> m_Views;
    };
}
