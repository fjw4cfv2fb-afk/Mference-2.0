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

struct ServerPromptCache: Sendable {
    private(set) var entry: ServerPromptCacheEntry?

    mutating func invalidate() {
        entry = nil
    }

    mutating func restore(_ saved: ServerPromptCacheEntry) {
        entry = saved
    }

    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        reasoningContent: String? = nil,
        stopStringFiltered: Bool = false
    ) {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            entry = nil
            return
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
        entry = ServerPromptCacheEntry(
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
    }

    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: MFTokenizer,
        rewindablePosition: (Int) -> Int? = { $0 }
    ) -> ServerPromptCacheMatch {
        guard let entry,
              entry.domain == domain,
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

        if !tokenizer.isSwiftQwen,
           request.reasoningEffort == entry.reasoningEffort {
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
