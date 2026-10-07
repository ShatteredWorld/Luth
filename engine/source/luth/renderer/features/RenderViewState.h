#pragma once

#include "luth/core/types/LuthTypes.h"
#include <atomic>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace Luth
{
    struct RenderViewId
    {
        u64 value = 0;
        bool operator==(const RenderViewId&) const = default;
    };

    // RenderingSystem owns registrations; target addresses are only borrowed associations.
    // Validation never dereferences a possibly expired target.
    class RenderViewRegistry
    {
    public:
        RenderViewRegistry() = default;
        RenderViewRegistry(const RenderViewRegistry&) = delete;
        RenderViewRegistry& operator=(const RenderViewRegistry&) = delete;
        struct Registration { RenderViewId id; const void* targets; u64 generation = 1; };

        RenderViewId Register(const void* targets)
        {
            if (!targets) throw std::invalid_argument("A view requires targets");
            if (auto id = Find(targets); id.value) return id;
            static std::atomic<u64> next{1};
            RenderViewId id{next.fetch_add(1, std::memory_order_relaxed)};
            m_Views.emplace(id.value, Registration{id, targets});
            return id;
        }
        RenderViewId Find(const void* targets) const
        {
            for (const auto& [id, view] : m_Views)
                if (view.targets == targets) return view.id;
            return {};
        }
        const Registration* Get(RenderViewId id) const
        {
            auto it = m_Views.find(id.value);
            return it == m_Views.end() ? nullptr : &it->second;
        }
        bool Matches(RenderViewId id, const void* targets, u64 generation) const
        {
            const auto* view = Get(id);
            return view && view->targets == targets && view->generation == generation;
        }
        u64 Invalidate(RenderViewId id)
        {
            auto it = m_Views.find(id.value);
            if (it == m_Views.end()) throw std::invalid_argument("Unknown render view");
            return ++it->second.generation;
        }
        void Release(RenderViewId id) { m_Views.erase(id.value); }

    private:
        std::unordered_map<u64, Registration> m_Views;
    };

    struct ViewStateConfig
    {
        u32 width = 0, height = 0;
        u64 signature = 0;
        u64 resourceGeneration = 0;
        bool operator==(const ViewStateConfig&) const = default;
    };

    // Domain-owned store. The supplied safe-point callback must finish before replacement
    // or retirement; the factory receives no graph handles. Ordinary parameter edits do
    // not change the configuration and do not wait. State owns its native retirement.
    template<class State>
    class FeatureViewStates
    {
    public:
        template<class Factory, class SafePoint>
        State& Ensure(RenderViewId id, const ViewStateConfig& config,
                      Factory&& factory, SafePoint&& safePoint)
        {
            if (!id.value) throw std::invalid_argument("Invalid render view identity");
            auto it = m_States.find(id.value);
            if (it != m_States.end() && it->second->config == config) return it->second->state;
            if (it != m_States.end()) safePoint();
            // Construct first so a failed allocation leaves the previous state available.
            auto replacement = std::make_unique<Entry>(Entry{config, factory(config)});
            if (it != m_States.end())
            {
                it->second.swap(replacement);
                return it->second->state;
            }
            return m_States.emplace(id.value, std::move(replacement)).first->second->state;
        }
        State* Find(RenderViewId id)
        {
            auto it = m_States.find(id.value);
            return it == m_States.end() ? nullptr : &it->second->state;
        }
        const State* Find(RenderViewId id) const
        {
            auto it = m_States.find(id.value);
            return it == m_States.end() ? nullptr : &it->second->state;
        }
        template<class SafePoint>
        void Release(RenderViewId id, SafePoint&& safePoint)
        {
            if (!m_States.contains(id.value)) return;
            safePoint();
            m_States.erase(id.value);
        }
        template<class SafePoint>
        void ReleaseAll(SafePoint&& safePoint)
        {
            if (m_States.empty()) return;
            safePoint();
            m_States.clear();
        }
    private:
        struct Entry { ViewStateConfig config; State state; };
        std::unordered_map<u64, std::unique_ptr<Entry>> m_States;
    };

    // Prepare can inspect validity without advancing history. Commit only after successful
    // recording/submission; absolute render frames detect visibility gaps in each view.
    struct ViewHistoryState
    {
        bool valid = false;
        u64 lastRenderFrame = 0;
        u64 generation = 0;
        bool CanReuse(u64 frame, u64 currentGeneration) const
        {
            return valid && generation == currentGeneration && frame == lastRenderFrame + 1;
        }
        void Invalidate() { valid = false; }
        void Commit(u64 frame, u64 currentGeneration)
        {
            valid = true;
            lastRenderFrame = frame;
            generation = currentGeneration;
        }
    };
}
