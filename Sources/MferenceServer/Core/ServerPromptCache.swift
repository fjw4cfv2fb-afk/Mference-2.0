import Foundation
import Mference

public enum ServerPromptCacheMode: String, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
}

struct ServerPromptCacheDomain: Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Sendable, Equatable {
    let message: MFTokenizer.Message
    let rawStopReason: StopReason
}

struct ServerPromptCacheEntry: Sendable, Equatable {
    let domain: ServerPromptCacheDomain
    let inputMessages: [MFTokenizer.Message]
    let tools: [MFTokenizer.FunctionDefinition]
    let assistantTurn: CachedAssistantTurn
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int
    var reasoningEffort: QwenReasoningEffort? = nil
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss
    case hit(effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

enum ServerPromptCacheSlotMatch: Sendable, Equatable {
    case miss
    case hit(slotID: UUID, effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

struct ServerPromptCacheSlot: Sendable, Equatable {
    let id: UUID
    let entry: ServerPromptCacheEntry
}

struct ServerPromptCacheState: Sendable {
    fileprivate let slots: [ServerPromptCacheSlot]
    fileprivate let activeSlotID: UUID?
    fileprivate let maxSlots: Int

    func entry(for slotID: UUID) -> ServerPromptCacheEntry? {
        slots.first { $0.id == slotID }?.entry
    }
}

struct ServerPromptCachePublishOutcome: Sendable {
    let publishedSlotID: UUID?
    let evictedSlotIDs: [UUID]
}

struct ServerPromptCache: Sendable {
    static let multiSlotMemoryThresholdBytes: UInt64 = 2_560 * 1024 * 1024 // 2.5 GiB

    static func recommendedMaxSlots(availableMemoryBytes: UInt64) -> Int {
        availableMemoryBytes > multiSlotMemoryThresholdBytes ? 2 : 1
    }

    private var slots: [ServerPromptCacheSlot] = []
    private var activeSlotID: UUID?
    private(set) var maxSlots: Int

    init(maxSlots: Int = 2) {
        self.maxSlots = max(0, maxSlots)
    }

    var entry: ServerPromptCacheEntry? {
        activeSlotID.flatMap { id in slots.first { $0.id == id }?.entry }
    }

    var slotCount: Int {
        slots.count
    }

    func contains(slotID: UUID) -> Bool {
        slots.contains { $0.id == slotID }
    }

    func entry(for slotID: UUID) -> ServerPromptCacheEntry? {
        slots.first { $0.id == slotID }?.entry
    }

    func snapshotState() -> ServerPromptCacheState {
        ServerPromptCacheState(slots: slots, activeSlotID: activeSlotID, maxSlots: maxSlots)
    }

    mutating func restoreState(_ state: ServerPromptCacheState) {
        slots = state.slots
        activeSlotID = state.activeSlotID
        maxSlots = state.maxSlots
    }

    mutating func invalidate() {
        slots.removeAll()
        activeSlotID = nil
    }

    mutating func removeSlot(_ slotID: UUID) {
        slots.removeAll { $0.id == slotID }
        if activeSlotID == slotID {
            activeSlotID = nil
        }
    }

    mutating func restore(_ saved: ServerPromptCacheEntry) {
        let id = UUID()
        slots.removeAll()
        if maxSlots > 0 {
            slots.append(ServerPromptCacheSlot(id: id, entry: saved))
            activeSlotID = id
        } else {
            activeSlotID = nil
        }
    }

    @discardableResult
    mutating func setMaxSlots(_ newMaxSlots: Int, preferredLiveSlotID: UUID? = nil) -> [UUID] {
        maxSlots = max(0, newMaxSlots)
        var evicted: [UUID] = []
        while slots.count > maxSlots {
            let evictIdx: Int
            if let preferredLiveSlotID,
               let firstNonLive = slots.firstIndex(where: { $0.id != preferredLiveSlotID }) {
                evictIdx = firstNonLive
            } else {
                evictIdx = 0
            }
            let removed = slots.remove(at: evictIdx)
            evicted.append(removed.id)
            if activeSlotID == removed.id {
                activeSlotID = nil
            }
        }
        return evicted
    }

    @discardableResult
    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        reasoningContent: String? = nil,
        stopStringFiltered: Bool = false,
        replacesSlotID: UUID? = nil
    ) -> ServerPromptCachePublishOutcome {
        guard maxSlots > 0,
              result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            var evicted: [UUID] = []
            if let replacesSlotID, slots.contains(where: { $0.id == replacesSlotID }) {
                slots.removeAll { $0.id == replacesSlotID }
                evicted.append(replacesSlotID)
            }
            if maxSlots <= 1 {
                for slot in slots {
                    evicted.append(slot.id)
                }
                slots.removeAll()
            }
            activeSlotID = nil
            return ServerPromptCachePublishOutcome(publishedSlotID: nil, evictedSlotIDs: evicted)
        }
        var evicted: [UUID] = []
        if let replacesSlotID, slots.contains(where: { $0.id == replacesSlotID }) {
            slots.removeAll { $0.id == replacesSlotID }
            evicted.append(replacesSlotID)
        }
        let historicalCalls = calls.map {
            MFTokenizer.HistoricalToolCall(
                id: $0.id,
                name: $0.name,
                arguments: $0.arguments)
        }
        let assistant = MFTokenizer.Message(
            role: .assistant,
            content: content.isEmpty ? nil : content,
            toolCalls: historicalCalls,
            reasoningContent: reasoningContent)
        let newEntry = ServerPromptCacheEntry(
            domain: domain,
            inputMessages: request.messages,
            tools: request.tools,
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition,
            reasoningEffort: request.reasoningEffort)
        let newSlot = ServerPromptCacheSlot(id: UUID(), entry: newEntry)
        slots.append(newSlot)
        activeSlotID = newSlot.id
        while slots.count > maxSlots {
            let removed = slots.removeFirst()
            evicted.append(removed.id)
        }
        return ServerPromptCachePublishOutcome(publishedSlotID: newSlot.id, evictedSlotIDs: evicted)
    }

    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: MFTokenizer,
        rewindablePosition: (Int) -> Int? = { $0 }
    ) -> ServerPromptCacheMatch {
        switch findBestSlotMatch(
            domain: domain,
            request: request,
            renderedPromptIDs: renderedPromptIDs,
            tokenizer: tokenizer,
            canUseSlot: { _ in true },
            rewindablePosition: { _, pos in rewindablePosition(pos) }) {
        case .miss:
            return .miss
        case .hit(_, let effectivePromptIDs, let cachedPromptTokens):
            return .hit(effectivePromptIDs: effectivePromptIDs, cachedPromptTokens: cachedPromptTokens)
        }
    }

    mutating func matchSlot(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: MFTokenizer,
        canUseSlot: (UUID) -> Bool = { _ in true },
        rewindablePosition: (UUID, Int) -> Int? = { _, pos in pos }
    ) -> ServerPromptCacheSlotMatch {
        let best = findBestSlotMatch(
            domain: domain,
            request: request,
            renderedPromptIDs: renderedPromptIDs,
            tokenizer: tokenizer,
            canUseSlot: canUseSlot,
            rewindablePosition: rewindablePosition)
        if case .hit(let slotID, _, _) = best,
           let idx = slots.firstIndex(where: { $0.id == slotID }) {
            let touched = slots.remove(at: idx)
            slots.append(touched)
            activeSlotID = slotID
        }
        return best
    }

    private func findBestSlotMatch(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: MFTokenizer,
        canUseSlot: (UUID) -> Bool,
        rewindablePosition: (UUID, Int) -> Int?
    ) -> ServerPromptCacheSlotMatch {
        var bestSlotID: UUID?
        var bestEffective: [Int32] = []
        var bestCachedTokens = -1

        for slot in slots.reversed() where canUseSlot(slot.id) {
            let candidate = matchSingleEntry(
                slot.entry,
                domain: domain,
                request: request,
                renderedPromptIDs: renderedPromptIDs,
                tokenizer: tokenizer,
                rewindablePosition: { rewindablePosition(slot.id, $0) })
            if case .hit(let effective, let cached) = candidate, cached > bestCachedTokens {
                bestSlotID = slot.id
                bestEffective = effective
                bestCachedTokens = cached
            }
        }

        guard let bestSlotID, bestCachedTokens > 0 else {
            return .miss
        }
        return .hit(
            slotID: bestSlotID,
            effectivePromptIDs: bestEffective,
            cachedPromptTokens: bestCachedTokens)
    }

    private func matchSingleEntry(
        _ entry: ServerPromptCacheEntry,
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: MFTokenizer,
        rewindablePosition: (Int) -> Int?
    ) -> ServerPromptCacheMatch {
        guard entry.domain == domain,
              entry.tools == request.tools,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return .miss
        }

        if renderedPromptIDs.count > entry.kvPosition,
           renderedPromptIDs.prefix(entry.kvPosition)
            .elementsEqual(entry.kvBackedTokenIDs) {
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: entry.kvPosition)
        }

        let reasoningStrippedPossible = !(entry.assistantTurn.message.reasoningContent ?? "").isEmpty
            && tokenizer.dialect == .chatml
        if !tokenizer.isSwiftQwen,
           request.reasoningEffort == entry.reasoningEffort,
           (renderedPromptIDs.count < 32 || renderedPromptIDs.count >= entry.kvPosition || reasoningStrippedPossible) {
            let inputCount = entry.inputMessages.count
            if request.messages.count > inputCount + 1,
               request.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
               assistantMatches(
                request.messages[inputCount],
                entry.assistantTurn.message) {
                let continuation = Array(request.messages.dropFirst(inputCount + 1))
                let contMatch: ServerPromptCacheMatch
                if entry.assistantTurn.message.toolCalls.isEmpty {
                    contMatch = matchTextContinuation(
                        entry: entry,
                        continuation: continuation,
                        tokenizer: tokenizer)
                } else {
                    contMatch = matchToolContinuation(
                        entry: entry,
                        request: request,
                        continuation: continuation,
                        tokenizer: tokenizer)
                }
                if case .hit = contMatch {
                    return contMatch
                }
            }
        }

        // Longest common prefix slice: if renderedPromptIDs shares a common prefix of at least
        // 32 tokens with entry.kvBackedTokenIDs, we can rewind/reuse KV up to a valid checkpoint.
        let maxCommon = min(renderedPromptIDs.count - 1, entry.kvPosition)
        var commonPrefix = 0
        while commonPrefix < maxCommon && renderedPromptIDs[commonPrefix] == entry.kvBackedTokenIDs[commonPrefix] {
            commonPrefix += 1
        }
        if commonPrefix >= 32,
           let targetPos = rewindablePosition(commonPrefix),
           targetPos >= 32,
           targetPos < renderedPromptIDs.count {
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: targetPos)
        }

        return .miss
    }

    private func assistantMatches(
        _ incoming: MFTokenizer.Message,
        _ cached: MFTokenizer.Message
    ) -> Bool {
        guard incoming.role == .assistant,
              cached.role == .assistant,
              incoming.toolCalls == cached.toolCalls,
              incoming.toolCallID == cached.toolCallID,
              incoming.name == cached.name else {
            return false
        }
        let incomingReasoning = (incoming.reasoningContent ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let cachedReasoning = (cached.reasoningContent ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard incomingReasoning == cachedReasoning || incomingReasoning.isEmpty else {
            return false
        }
        let incomingContent = (incoming.content ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let cachedContent = (cached.content ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !cached.toolCalls.isEmpty {
            return incomingContent == cachedContent || incomingContent.isEmpty
        }
        return incomingContent == cachedContent
    }

    private func matchTextContinuation(
        entry: ServerPromptCacheEntry,
        continuation: [MFTokenizer.Message],
        tokenizer: MFTokenizer
    ) -> ServerPromptCacheMatch {
        guard continuation.count == 1,
              continuation[0].role == .user,
              let content = continuation[0].content,
              continuation[0].toolCalls.isEmpty,
              continuation[0].toolCallID == nil,
              entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens else {
            return .miss
        }
        var bridge = tokenizer.encodeTextContinuation(userContent: content)
        if entry.assistantTurn.rawStopReason == .maxTokens {
            bridge = entry.uncommittedBoundaryTokenIDs + bridge
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }

    private func matchToolContinuation(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        continuation: [MFTokenizer.Message],
        tokenizer: MFTokenizer
    ) -> ServerPromptCacheMatch {
        let calls = entry.assistantTurn.message.toolCalls
        guard (entry.assistantTurn.rawStopReason == .toolCalls
               || entry.assistantTurn.rawStopReason == .endOfTurn),
              continuation.count == calls.count,
              zip(continuation, calls).allSatisfy({ message, call in
                  message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
              }) else {
            return .miss
        }
        guard let bridge = try? tokenizer.encodeToolResultContinuation(
            cachedMessages: entry.inputMessages,
            assistant: entry.assistantTurn.message,
            incomingMessages: request.messages,
            tools: request.tools,
            reasoningEffort: request.reasoningEffort),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }
}
