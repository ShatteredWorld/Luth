#pragma once

#include "luth/renderer/features/ResourceContract.h"
#include <new>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace Luth
{
    namespace Memory { class LinearAllocator; }

    enum class BlackboardErrorCode
    {
        InvalidKey, TypeMismatch, UnknownKey, UnpublishedResource, AbsentResource,
        DuplicatePublication, UndeclaredRead, UndeclaredWrite, RequiredOutputAbsent,
        MissingOutput, NestedFeatureScope
    };

    class BlackboardError final : public std::logic_error
    {
    public:
        BlackboardError(BlackboardErrorCode, ResourceKeyRef, std::string_view feature);
        BlackboardErrorCode Code() const noexcept { return m_Code; }
        ResourceKeyRef Key() const noexcept { return m_Key; }
    private:
        BlackboardErrorCode m_Code;
        ResourceKeyRef m_Key;
    };

    // Cold-path layout construction; the semantic compiler will retain this immutable layout.
    // Repeated declarations of the same typed key share one dense slot. No frame allocation
    // is needed for key lookup, type checking, or access validation.
    class ResourceSlotLayout
    {
    public:
        struct Slot { ResourceKeyRef key; size_t offset; };

        explicit ResourceSlotLayout(std::span<const ResourceKeyRef> keys);
        ResourceSlotLayout(const ResourceSlotLayout&) = default;
        ResourceSlotLayout(ResourceSlotLayout&&) = default;
        ResourceSlotLayout& operator=(const ResourceSlotLayout&) = delete;
        ResourceSlotLayout& operator=(ResourceSlotLayout&&) = delete;
        size_t ValueBytes() const noexcept { return m_ValueBytes; }
        size_t SlotCount() const noexcept { return m_Slots.size(); }
        std::span<const Slot> Slots() const noexcept { return m_Slots; }
        size_t Find(ResourceKeyRef, std::string_view feature = {}) const;
    private:
        std::vector<Slot> m_Slots;
        std::unordered_map<const ResourceKeyIdentity*, size_t> m_Indices;
        size_t m_ValueBytes = 0;
    };

    // Exactly one blackboard per graph build. Resetting the scratch allocator invalidates
    // the board and every reference returned by it. Values contain no owning GPU objects.
    class RenderBlackboard
    {
    public:
        RenderBlackboard(const ResourceSlotLayout&, Memory::LinearAllocator&);
        RenderBlackboard(ResourceSlotLayout&&, Memory::LinearAllocator&) = delete;
        RenderBlackboard(const ResourceSlotLayout&&, Memory::LinearAllocator&) = delete;
        RenderBlackboard(const RenderBlackboard&) = delete;
        RenderBlackboard& operator=(const RenderBlackboard&) = delete;

        template<class T>
        const T& Get(RenderResourceKey<T> key) const
        {
            const T* value = TryGet(key);
            if (!value) Fail(BlackboardErrorCode::AbsentResource, key);
            return *value;
        }

        template<class T>
        const T* TryGet(RenderResourceKey<T> key) const
        {
            const size_t slot = ReadSlot(key);
            if (m_States[slot] == State::Absent) return nullptr;
            return static_cast<const T*>(ValueAddress(slot));
        }

        template<class T>
        void Publish(RenderResourceKey<T> key, const T& value)
        {
            const size_t slot = WriteSlot(key, false);
            new (ValueAddress(slot)) T(value);
            m_States[slot] = State::Present;
        }

        template<class T>
        void PublishAbsent(RenderResourceKey<T> key)
        {
            const size_t slot = WriteSlot(key, true);
            m_States[slot] = State::Absent;
        }

        class FeatureScope
        {
        public:
            FeatureScope(RenderBlackboard&, const ResourceContract&, std::string_view name);
            FeatureScope(RenderBlackboard&, ResourceContract&&, std::string_view) = delete;
            FeatureScope(RenderBlackboard&, const ResourceContract&&, std::string_view) = delete;
            ~FeatureScope();
            FeatureScope(const FeatureScope&) = delete;
            FeatureScope& operator=(const FeatureScope&) = delete;
            void ValidateOutputs() const;
        private:
            RenderBlackboard& m_Board;
        };

        size_t StorageBytes() const noexcept { return m_Layout.ValueBytes() + m_Layout.SlotCount(); }

    private:
        enum class State : u8 { Unpublished, Absent, Present };
        static_assert(sizeof(State) == 1);
        size_t ReadSlot(ResourceKeyRef) const;
        size_t WriteSlot(ResourceKeyRef, bool absent) const;
        void* ValueAddress(size_t slot) const { return m_Values + m_Layout.Slots()[slot].offset; }
        [[noreturn]] void Fail(BlackboardErrorCode, ResourceKeyRef) const;

        const ResourceSlotLayout& m_Layout;
        byte* m_Values = nullptr;
        State* m_States = nullptr;
        const ResourceContract* m_Contract = nullptr;
        std::string_view m_FeatureName;
    };
}
