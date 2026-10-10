#include "luth/renderer/features/RenderBlackboard.h"
#include "luth/memory/LinearAllocator.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

namespace Luth
{
    namespace
    {
        const char* ErrorName(BlackboardErrorCode code)
        {
            switch (code)
            {
            case BlackboardErrorCode::InvalidKey: return "invalid resource key";
            case BlackboardErrorCode::TypeMismatch: return "resource type mismatch";
            case BlackboardErrorCode::UnknownKey: return "resource has no compiled slot";
            case BlackboardErrorCode::UnpublishedResource: return "resource has not been published";
            case BlackboardErrorCode::AbsentResource: return "required resource is absent";
            case BlackboardErrorCode::DuplicatePublication: return "resource was already published";
            case BlackboardErrorCode::UndeclaredRead: return "undeclared resource read";
            case BlackboardErrorCode::UndeclaredWrite: return "undeclared resource write";
            case BlackboardErrorCode::RequiredOutputAbsent: return "required output cannot be absent";
            case BlackboardErrorCode::MissingOutput: return "advertised output was not published";
            case BlackboardErrorCode::NestedFeatureScope: return "feature scopes cannot nest";
            }
            return "blackboard error";
        }

        std::string ErrorMessage(BlackboardErrorCode code, ResourceKeyRef key, std::string_view feature)
        {
            std::string message(feature.empty() ? "RenderBlackboard" : feature);
            message += ": ";
            message += ErrorName(code);
            if (key.identity) { message += " ["; message += key.identity->name; message += "]"; }
            if (key.type) { message += "; expected "; message += key.type->name; }
            return message;
        }

        void ValidateKey(ResourceKeyRef key, std::string_view feature = {})
        {
            if (!key.identity || !key.type || key.identity->name.empty() || !key.type->size ||
                !key.type->alignment || key.type->alignment > alignof(std::max_align_t) ||
                (key.type->alignment & (key.type->alignment - 1)) != 0)
                throw BlackboardError(BlackboardErrorCode::InvalidKey, key, feature);
        }
    }

    BlackboardError::BlackboardError(BlackboardErrorCode code, ResourceKeyRef key, std::string_view feature)
        : std::logic_error(ErrorMessage(code, key, feature)), m_Code(code), m_Key(key) {}

    ResourceSlotLayout::ResourceSlotLayout(std::span<const ResourceKeyRef> keys)
    {
        m_Slots.reserve(keys.size());
        m_Indices.reserve(keys.size());
        for (ResourceKeyRef key : keys)
        {
            ValidateKey(key);
            const auto existing = m_Indices.find(key.identity);
            if (existing != m_Indices.end())
            {
                if (m_Slots[existing->second].key.type != key.type)
                    throw BlackboardError(BlackboardErrorCode::TypeMismatch, key, {});
                continue;
            }
            const size_t padding = key.type->alignment - 1;
            if (m_ValueBytes > std::numeric_limits<size_t>::max() - padding)
                throw std::length_error("Blackboard slot alignment overflow");
            const size_t offset = (m_ValueBytes + padding) & ~padding;
            if (key.type->size > std::numeric_limits<size_t>::max() - offset)
                throw std::length_error("Blackboard slot size overflow");
            m_Indices.emplace(key.identity, m_Slots.size());
            m_Slots.push_back({key, offset});
            m_ValueBytes = offset + key.type->size;
        }
        if (m_ValueBytes > std::numeric_limits<size_t>::max() - m_Slots.size())
            throw std::length_error("Blackboard storage size overflow");
    }

    size_t ResourceSlotLayout::Find(ResourceKeyRef key, std::string_view feature) const
    {
        ValidateKey(key, feature);
        const auto found = m_Indices.find(key.identity);
        if (found == m_Indices.end())
            throw BlackboardError(BlackboardErrorCode::UnknownKey, key, feature);
        if (m_Slots[found->second].key.type != key.type)
            throw BlackboardError(BlackboardErrorCode::TypeMismatch, key, feature);
        return found->second;
    }

    RenderBlackboard::RenderBlackboard(const ResourceSlotLayout& layout, Memory::LinearAllocator& scratch)
        : m_Layout(layout)
    {
        if (!layout.SlotCount()) return;
        // States follow values in one bounded arena allocation. Only the cold layout owns
        // vectors/maps. The board never grows or allocates per resource access.
        m_Values = static_cast<byte*>(scratch.Allocate(StorageBytes(), alignof(std::max_align_t)));
        m_States = reinterpret_cast<State*>(m_Values + layout.ValueBytes());
        for (size_t i = 0; i < layout.SlotCount(); ++i)
            new (m_States + i) State(State::Unpublished);
    }

    [[noreturn]] void RenderBlackboard::Fail(BlackboardErrorCode code, ResourceKeyRef key) const
    {
        throw BlackboardError(code, key, m_FeatureName);
    }

    const void* RenderBlackboard::TryGetBorrowed(ResourceKeyRef key) const
    {
        const size_t slot = ReadSlot(key);
        return m_States[slot] == State::Absent ? nullptr : ValueAddress(slot);
    }

    void RenderBlackboard::PublishBorrowed(ResourceKeyRef key, const void* value)
    {
        const size_t slot = WriteSlot(key, false);
        // Values are constrained to trivially copyable types by typed key construction.
        // memcpy starts their implicit lifetime in the destination storage (C++20).
        std::memcpy(ValueAddress(slot), value, key.type->size);
        m_States[slot] = State::Present;
    }

    void RenderBlackboard::PublishAbsentBorrowed(ResourceKeyRef key)
    {
        const size_t slot = WriteSlot(key, true);
        m_States[slot] = State::Absent;
    }

    size_t RenderBlackboard::ReadSlot(ResourceKeyRef key) const
    {
        const size_t slot = m_Layout.Find(key, m_FeatureName);
        if (m_Contract)
        {
            const auto& reads = m_Contract->reads;
            const bool declared = std::any_of(reads.begin(), reads.end(), [&](const ResourceRead& read) {
                return read.key.identity == key.identity && read.key.type == key.type;
            });
            if (!declared) Fail(BlackboardErrorCode::UndeclaredRead, key);
        }
        if (m_States[slot] == State::Unpublished) Fail(BlackboardErrorCode::UnpublishedResource, key);
        return slot;
    }

    size_t RenderBlackboard::WriteSlot(ResourceKeyRef key, bool absent) const
    {
        const size_t slot = m_Layout.Find(key, m_FeatureName);
        if (m_Contract)
        {
            const auto& writes = m_Contract->writes;
            const auto declared = std::find_if(writes.begin(), writes.end(), [&](const ResourceWrite& write) {
                return write.key.identity == key.identity && write.key.type == key.type;
            });
            if (declared == writes.end()) Fail(BlackboardErrorCode::UndeclaredWrite, key);
            if (absent && declared->presence == ResourceOutputPresence::Required)
                Fail(BlackboardErrorCode::RequiredOutputAbsent, key);
        }
        if (m_States[slot] != State::Unpublished) Fail(BlackboardErrorCode::DuplicatePublication, key);
        return slot;
    }

    RenderBlackboard::FeatureScope::FeatureScope(RenderBlackboard& board,
        const ResourceContract& contract, std::string_view name) : m_Board(board)
    {
        if (board.m_Contract) board.Fail(BlackboardErrorCode::NestedFeatureScope, {});
        board.m_Contract = &contract;
        board.m_FeatureName = name;
    }

    RenderBlackboard::FeatureScope::~FeatureScope()
    {
        m_Board.m_Contract = nullptr;
        m_Board.m_FeatureName = {};
    }

    void RenderBlackboard::FeatureScope::ValidateOutputs() const
    {
        for (const auto& write : m_Board.m_Contract->writes)
        {
            const size_t slot = m_Board.m_Layout.Find(write.key, m_Board.m_FeatureName);
            if (m_Board.m_States[slot] == State::Unpublished)
                m_Board.Fail(BlackboardErrorCode::MissingOutput, write.key);
            if (m_Board.m_States[slot] == State::Absent && write.presence == ResourceOutputPresence::Required)
                m_Board.Fail(BlackboardErrorCode::RequiredOutputAbsent, write.key);
        }
    }
}
