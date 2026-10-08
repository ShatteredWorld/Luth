#pragma once
#include "luth/core/types/LuthTypes.h"
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Luth
{
    // Explicit native reload clients, in registration order. A match never stops fan-out.
    class ShaderReloadFanout
    {
    public:
        using Consumer = std::function<bool(const std::string&, const std::vector<u32>&)>;
        void Add(std::string name, Consumer consumer)
        {
            if (name.empty() || !consumer) throw std::invalid_argument("Reload consumer requires a name and callback");
            for (const auto& entry : m_Consumers)
                if (entry.name == name) throw std::invalid_argument("Duplicate reload consumer name");
            m_Consumers.push_back({std::move(name), std::move(consumer)});
        }
        std::vector<std::string> Notify(const std::string& shader, const std::vector<u32>& spv) const
        {
            std::vector<std::string> refreshed;
            if (shader.empty() || spv.empty()) return refreshed;
            for (const auto& entry : m_Consumers)
                if (entry.consumer(shader, spv)) refreshed.push_back(entry.name);
            return refreshed;
        }
    private:
        struct Entry { std::string name; Consumer consumer; };
        std::vector<Entry> m_Consumers;
    };
}
