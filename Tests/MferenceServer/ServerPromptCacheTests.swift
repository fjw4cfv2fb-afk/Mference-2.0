import Foundation
import Testing

@testable import Mference
@testable import MferenceServerCore

@Suite("Server prompt cache")
struct ServerPromptCacheTests {
    @Test(arguments: [false, true], [false, true])
    func explicitBaseEffortNeverUsesLegacyBridge(before: Bool, after: Bool) async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen38)
            .forCheckpoint(CheckpointIdentity.baseQwen38)
        var initial = request(messages: [.init(role: .user, content: "A")])
        initial.reasoningEffort = before ? .low : nil
        var cache = ServerPromptCache()
        cache.publish(domain: domain, request: initial, content: "B", calls: [],
            result: rawResult(prompt: [1], kvBacked: [1, 2], boundary: tok.endOfTurnID, reason: .endOfTurn))
        var continuation = request(messages: initial.messages + [.init(role: .assistant, content: "B"),
            .init(role: .user, content: "C")])
        continuation.reasoningEffort = after ? .medium : nil
        let match = cache.match(domain: domain, request: continuation,
            renderedPromptIDs: [1, 9, 3, 4], tokenizer: tok)
        if before || after { #expect(match == .miss) }
        else if case .hit = match {} else { Issue.record("Legacy bridge must remain available") }
        if case .hit = cache.match(domain: domain, request: continuation,
            renderedPromptIDs: [1, 2, 3, 4], tokenizer: tok) {} else {
            Issue.record("Exact rendered prefixes remain reusable")
        }
    }
    @Test func swiftQwenHistoryIsPreservedAndBaseDomainCannotReuseIt() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen38)
            .forCheckpoint(CheckpointIdentity.swiftQwen38)
        let initial = request(messages: [.init(role: .user, content: "A")])
        func checkpointDomain(_ id: String) -> ServerPromptCacheDomain {
            .init(modelID: id, sourceSnapshotHash: "source", runtimeProfileHash: "profile",
                  maximumContext: 16384, kvStorage: "fp16", fp16RingEnabled: true, templateSHA256: "template")
        }
        let swift = checkpointDomain(CheckpointIdentity.swiftQwen38)
        var cache = ServerPromptCache()
        cache.publish(domain: swift, request: initial, content: "B", calls: [],
                      result: rawResult(prompt: [1], kvBacked: [1, 2], boundary: 3, reason: .endOfTurn),
                      reasoningContent: "Check A")
        #expect(cache.entry?.assistantTurn.message.reasoningContent == "Check A")
        #expect(cache.match(domain: checkpointDomain("qwen3.8-27b-4bit"), request: initial,
                            renderedPromptIDs: [1, 2, 3, 4], tokenizer: tok) == .miss)
        let continuation = request(messages: initial.messages + [
            .init(role: .assistant, content: "B", reasoningContent: "Check A"),
            .init(role: .user, content: "C")])
        #expect(cache.match(domain: swift, request: continuation,
                            renderedPromptIDs: [1, 9, 3, 4], tokenizer: tok) == .miss)
        if case .hit = cache.match(domain: swift, request: continuation,
                                    renderedPromptIDs: [1, 2, 3, 4], tokenizer: tok) {} else {
            Issue.record("an exact Swift prefix should remain reusable")
        }
        cache.invalidate()
        #expect(cache.entry == nil)
    }
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func textContinuationUsesActualGeneratedHistoryAndOnlyPrefillsSuffix() async throws {
        let tokenizer = try await MFTokenizer.load()
        let initial = request(messages: [
            MFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            MFTokenizer.Message(role: .assistant, content: "answer"),
            MFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func capturedOpenCodeToolResultUsesFrozenToolBoundary() async throws {
        let tokenizer = try await MFTokenizer.load()
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let continuation = try validatedFixture("opencode-1.15.11-tool-result.json")
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: initial.messages,
            tools: initial.tools)
        let assistant = continuation.messages[initial.messages.count]
        let prefix = try tokenizer.encodeToolChat(
            messages: initial.messages + [assistant],
            tools: initial.tools)
        let callStart = try #require(prefix.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(prefix.lastIndex(of: tokenizer.toolCallEndID))
        let generatedCall = Array(prefix[callStart...callEnd])
        let kvBacked = initialPrompt + generatedCall
        let historicalCall = try #require(assistant.toolCalls.first)
        let parsedCall = ParsedToolCall(
            id: historicalCall.id,
            name: historicalCall.name,
            arguments: historicalCall.arguments,
            argumentsJSON: try historicalCall.arguments.encoded())
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [parsedCall],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))
        let rendered = try tokenizer.encodeToolChat(
            messages: continuation.messages,
            tools: continuation.tools)

        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected captured OpenCode tool-result hit")
            return
        }
        let bridge = try tokenizer.encodeToolResultContinuation(
            cachedMessages: initial.messages,
            assistant: assistant,
            incomingMessages: continuation.messages,
            tools: continuation.tools)
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(bridge.first == tokenizer.toolResponseID)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await MFTokenizer.load()
        let initial = request(messages: [
            MFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .eos] {
            cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            #expect(cache.entry == nil)
        }

        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt + tokenizer.encode("answer", addBOS: false),
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let changed = request(messages: [
            MFTokenizer.Message(role: .user, content: "changed"),
            MFTokenizer.Message(role: .assistant, content: "answer"),
            MFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(changed.messages),
            addBOS: false)
        #expect(cache.match(
            domain: domain,
            request: changed,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) == .miss)
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await MFTokenizer.load()
        let initial = request(messages: [
            MFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer ",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn),
            stopStringFiltered: matcher.isStopped)
        #expect(cache.entry == nil)
    }

    @Test func partialPrefixMatchReusesCommonTokens() async throws {
        let tokenizer = try await MFTokenizer.load()
        let initialTokens = (1...100).map { Int32($0) }
        let initial = request(messages: [MFTokenizer.Message(role: .user, content: "test")])
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialTokens,
                kvBacked: initialTokens,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        // Continuation shares first 80 tokens, then diverges
        let continuationTokens = (1...80).map { Int32($0) } + [999, 1000]
        let continuation = request(messages: [
            MFTokenizer.Message(role: .user, content: "test"),
            MFTokenizer.Message(role: .assistant, content: "answer"),
            MFTokenizer.Message(role: .user, content: "next")
        ])
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: continuationTokens,
            tokenizer: tokenizer)
        switch match {
        case .hit(let effective, let cached):
            #expect(cached == 80)
            #expect(effective == continuationTokens)
        case .miss:
            Issue.record("Expected prefix hit for 80 common tokens")
        }
    }

    @Test func chatMLStrippedReasoningStillUsesContinuationBridge() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Mference/Core/Tokenization/Fixtures/ChatMLTokenizer")
        let tok = try await MFTokenizer.load(from: fixture, family: .qwen38)
        let initial = request(messages: [.init(role: .user, content: "Explain primes")])
        let prompt = [Int32](1...35)
        let kvBacked = [Int32](1...120)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "77",
            calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked, boundary: tok.endOfTurnID, reason: .endOfTurn),
            reasoningContent: "2+3+5+7+11+13+17+19 = 77")

        let continuation = request(messages: initial.messages + [
            .init(role: .assistant, content: "77"),
            .init(role: .user, content: "Double it"),
        ])
        let renderedStripped = [Int32](1...45)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: renderedStripped,
            tokenizer: tok)
        guard case .hit(let effective, let cached) = match else {
            Issue.record("Expected continuation hit when ChatML template strips historical thinking tokens")
            return
        }
        let bridge = tok.encodeTextContinuation(userContent: "Double it")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
    }

    @Test func interleavedConversationsStayCachedWithoutEviction() async throws {
        let tokenizer = try await MFTokenizer.load()
        #expect(ServerPromptCache.recommendedMaxSlots(availableMemoryBytes: 3 * 1024 * 1024 * 1024) == 2)

        let reqA1 = request(messages: [.init(role: .user, content: "Explain radix sort in Swift")])
        let promptA1 = tokenizer.encode(try tokenizer.applyChatTemplate(reqA1.messages), addBOS: false)
        let kvA1 = promptA1 + tokenizer.encode("Radix sort processes digits", addBOS: false)

        let reqB1 = request(messages: [.init(role: .user, content: "Explain bloom filters briefly")])
        let promptB1 = tokenizer.encode(try tokenizer.applyChatTemplate(reqB1.messages), addBOS: false)
        let kvB1 = promptB1 + tokenizer.encode("Bloom filters test membership", addBOS: false)

        let reqC1 = request(messages: [.init(role: .user, content: "Explain skip lists briefly")])
        let promptC1 = tokenizer.encode(try tokenizer.applyChatTemplate(reqC1.messages), addBOS: false)
        let kvC1 = promptC1 + tokenizer.encode("Skip lists layer pointers", addBOS: false)

        var cache = ServerPromptCache(maxSlots: 2)
        let outA1 = cache.publish(
            domain: domain,
            request: reqA1,
            content: "Radix sort processes digits",
            calls: [],
            result: rawResult(prompt: promptA1, kvBacked: kvA1, boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        let idA1 = try #require(outA1.publishedSlotID)
        #expect(outA1.evictedSlotIDs.isEmpty)
        #expect(cache.slotCount == 1)

        let outB1 = cache.publish(
            domain: domain,
            request: reqB1,
            content: "Bloom filters test membership",
            calls: [],
            result: rawResult(prompt: promptB1, kvBacked: kvB1, boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        let idB1 = try #require(outB1.publishedSlotID)
        #expect(outB1.evictedSlotIDs.isEmpty)
        #expect(cache.slotCount == 2)

        let reqA2 = request(messages: reqA1.messages + [
            .init(role: .assistant, content: "Radix sort processes digits"),
            .init(role: .user, content: "Now show time complexity"),
        ])
        let renderedA2 = tokenizer.encode(try tokenizer.applyChatTemplate(reqA2.messages), addBOS: false)
        let hitA = cache.matchSlot(
            domain: domain,
            request: reqA2,
            renderedPromptIDs: renderedA2,
            tokenizer: tokenizer)
        guard case .hit(let matchedA, _, let cachedA) = hitA else {
            Issue.record("Expected interleaved hit for conversation A")
            return
        }
        #expect(matchedA == idA1)
        #expect(cachedA == kvA1.count)

        let kvA2 = kvA1 + tokenizer.encode("O(nk) time complexity", addBOS: false)
        let outA2 = cache.publish(
            domain: domain,
            request: reqA2,
            content: "O(nk) time complexity",
            calls: [],
            result: rawResult(prompt: renderedA2, kvBacked: kvA2, boundary: tokenizer.endOfTurnID, reason: .endOfTurn),
            replacesSlotID: matchedA)
        let idA2 = try #require(outA2.publishedSlotID)
        #expect(cache.slotCount == 2)
        #expect(cache.contains(slotID: idB1))

        let reqB2 = request(messages: reqB1.messages + [
            .init(role: .assistant, content: "Bloom filters test membership"),
            .init(role: .user, content: "Mention false positives"),
        ])
        let renderedB2 = tokenizer.encode(try tokenizer.applyChatTemplate(reqB2.messages), addBOS: false)
        let hitB = cache.matchSlot(
            domain: domain,
            request: reqB2,
            renderedPromptIDs: renderedB2,
            tokenizer: tokenizer)
        guard case .hit(let matchedB, _, let cachedB) = hitB else {
            Issue.record("Expected interleaved hit for conversation B after A turn 2")
            return
        }
        #expect(matchedB == idB1)
        #expect(cachedB == kvB1.count)

        let outC1 = cache.publish(
            domain: domain,
            request: reqC1,
            content: "Skip lists layer pointers",
            calls: [],
            result: rawResult(prompt: promptC1, kvBacked: kvC1, boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        #expect(outC1.evictedSlotIDs == [idA2])
        #expect(cache.slotCount == 2)
        #expect(cache.contains(slotID: idB1))
    }

    @Test func memoryPressureShrinksCacheToSingleSlotAndPicksLongestMatch() async throws {
        let tokenizer = try await MFTokenizer.load()
        let threshold: UInt64 = 2_560 * 1024 * 1024
        #expect(ServerPromptCache.recommendedMaxSlots(availableMemoryBytes: threshold) == 1)
        #expect(ServerPromptCache.recommendedMaxSlots(availableMemoryBytes: threshold + 1) == 2)

        let common = (1...40).map { Int32($0) }
        let branchA = common + (101...130).map { Int32($0) }
        let branchB = common + (201...220).map { Int32($0) }

        var cache = ServerPromptCache(maxSlots: 2)
        let outA = cache.publish(
            domain: domain,
            request: request(messages: [.init(role: .user, content: "branch A")]),
            content: "A",
            calls: [],
            result: rawResult(prompt: branchA, kvBacked: branchA, boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        let idA = try #require(outA.publishedSlotID)

        let outB = cache.publish(
            domain: domain,
            request: request(messages: [.init(role: .user, content: "branch B")]),
            content: "B",
            calls: [],
            result: rawResult(prompt: branchB, kvBacked: branchB, boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        let idB = try #require(outB.publishedSlotID)

        let extendA = branchA + [301, 302, 303]
        let bestMatch = cache.matchSlot(
            domain: domain,
            request: request(messages: [.init(role: .user, content: "branch A follow-up")]),
            renderedPromptIDs: extendA,
            tokenizer: tokenizer)
        guard case .hit(let bestID, _, let cached) = bestMatch else {
            Issue.record("Expected longest prefix hit across multi-slot cache")
            return
        }
        #expect(bestID == idA)
        #expect(cached == branchA.count)

        let evicted = cache.setMaxSlots(1, preferredLiveSlotID: idB)
        #expect(evicted == [idA])
        #expect(cache.slotCount == 1)
        #expect(cache.contains(slotID: idB))
    }

    private func request(
        messages: [MFTokenizer.Message],
        tools: [MFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: prompt.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary],
            prefillExecution: nil)
    }

    private func validatedFixture(_ name: String) throws -> ValidatedChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name,
            withExtension: nil,
            subdirectory: "Fixtures"))
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(contentsOf: url))
        return try OpenAIRequestValidator.validate(
            request,
            modelID: "gemma-4-26b-a4b-it")
    }
}
