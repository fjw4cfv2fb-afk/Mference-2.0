import CryptoKit
import Foundation
import Mference
#if canImport(Darwin)
import Darwin
@_silgen_name("os_proc_available_memory")
private func mference_os_proc_available_memory() -> Int
#endif

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    case reasoning(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public var reasoningContent: String? = nil
    public let content: String
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage
    /// Operator-only diagnostics, not part of the OpenAI response body.
    public let diagnostics: RuntimeDiagnostics?

    public init(content: String,
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage,
                diagnostics: RuntimeDiagnostics? = nil) {
        self.content = content
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
        self.diagnostics = diagnostics
    }
}

/// A request that has been rendered and measured against the context window.
public struct PreparedGeneration: Sendable {
    public let request: ValidatedChatRequest
    public let promptIDs: [Int32]
    public let needsToolTemplate: Bool

    public init(request: ValidatedChatRequest,
                promptIDs: [Int32] = [],
                needsToolTemplate: Bool = false) {
        self.request = request
        self.promptIDs = promptIDs
        self.needsToolTemplate = needsToolTemplate
    }
}

public protocol ServerInferenceBackend: Sendable {
    var usesSwiftQwenTemplate: Bool { get }
    var supportsQwenReasoningEffort: Bool { get }
    /// Everything that can reject a request must happen here, because the
    /// caller commits the response status once `generate` starts: a streaming
    /// request has `200` and the SSE head on the wire by then, and no status
    /// left to send.
    func prepare(_ request: ValidatedChatRequest) async throws -> PreparedGeneration

    func generate(_ prepared: PreparedGeneration,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

extension ServerInferenceBackend {
    public var usesSwiftQwenTemplate: Bool { false }
    public var supportsQwenReasoningEffort: Bool { usesSwiftQwenTemplate }
    /// Backends that do not tokenize inherit a pass-through. A backend that
    /// renders a prompt must override this, or `generate` receives no tokens.
    public func prepare(_ request: ValidatedChatRequest) async throws -> PreparedGeneration {
        PreparedGeneration(request: request)
    }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var active = false
    private var waiters: [Waiter] = []
    private var claims = 0
    private var shuttingDown = false

    public init(queueLimit: Int) {
        self.queueLimit = queueLimit
    }

    /// Claims a place, renders, then waits its turn. Rendering stays outside
    /// the gate so it never stalls the request that is generating, but the
    /// place is claimed before it starts: a request the queue has no room for
    /// is turned away without having paid for tokenization first.
    public func run<Rendered: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        render: @escaping @Sendable () async throws -> Rendered,
        _ operation: @escaping @Sendable (Rendered) async throws -> T
    ) async throws -> T {
        try claim()
        let rendered: Rendered
        do {
            rendered = try await render()
        } catch {
            claims -= 1
            throw error
        }
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation(rendered)
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await run(onQueued: onQueued, render: {}) { _ in try await operation() }
    }

    /// Occupancy is the one generating request plus the queue behind it.
    /// Outstanding claims count towards it from the moment they are taken, so
    /// two requests rendering at once cannot both take the same free place.
    private func claim() throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        let occupancy = (active ? 1 : 0) + waiters.count + claims
        guard occupancy < queueLimit + 1 else { throw ServerRequestError.queueFull }
        claims += 1
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        // The claim becomes the active slot or a place in the queue.
        claims -= 1
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            return
        }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            active = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }
}

public actor ServerModelSession: ServerLoadedModel {
    public nonisolated let usesSwiftQwenTemplate: Bool
    public nonisolated let supportsQwenReasoningEffort: Bool
    /// Chat dialect of the loaded tokenizer; drives request-validation rules.
    public nonisolated let chatDialect: ChatDialect
    /// Family-derived API model identifier used when --model-id is absent.
    public nonisolated var defaultModelID: String {
        ServerFamilyModelID.modelID(for: modelFamily, checkpointID: checkpointID)
    }
    private nonisolated let modelFamily: ModelFamily
    private nonisolated let checkpointID: String

    private let context: MetalContext
    private let model: Model
    private let tokenizer: MFTokenizer
    private let runner: any ContinuableLogitProducer
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    private let maxContext: Int
    private let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private var promptCache = ServerPromptCache()
    private var liveRunnerCacheEntryID: UUID?
    private var cachedSnapshots: [UUID: RealForwardRunner.PrefixSnapshot] = [:]

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            promptCacheMode: ServerPromptCacheMode = .singlePrefix,
                            expertCacheSlots: Int = 32,
                            prefillChunkTokens: Int = 128,
                            verification: ModelIntegrityPolicy = .fullSha256) async throws -> ServerModelSession {
        _ = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        let tokenizerFolder = MFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw MFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        let tokenizer = try await MFTokenizer.load(forModelDirectory: modelDirectory)
        // DeepSeek ships no chat_template.jinja — its chat framing is native
        // Swift — so the prompt-cache identity hashes a pinned constant that
        // changes only when that native render does. Every other dialect
        // still requires the bundled template.
        let templateData: Data
        if tokenizer.dialect == .glm5 {
            // GLM-5.3 ships a chat_template.jinja but the dialect renders
            // natively (`Glm5ChatTemplate.swift`); the identity follows the
            // native render, not the sidecar.
            templateData = Data("native:glm5:v1".utf8)
        } else if FileManager.default.fileExists(atPath: templateURL.path) {
            templateData = try Data(contentsOf: templateURL)
        } else if tokenizer.dialect == .deepseek {
            templateData = Data("native:deepseek:v1".utf8)
        } else {
            throw MFTokenizerError.missingToolTemplate
        }
        let context = try MetalContext()
        let streamingMode: ExpertStreamingMode = .pread(slotCount: expertCacheSlots)
        let configSlots = expertCacheSlots
        let env = ProcessInfo.processInfo.environment
        let specEnv = env["MFERENCE_SPEC"]?.lowercased()
        let mtpEnv = env["MFERENCE_MTP"]?.lowercased()
        let pldEnv = env["MFERENCE_PLD"]?.lowercased()
        let hasMTPSidecar = ManifestReader.peekSidecarCarried("mtp", directoryURL: modelDirectory)
        let peekedFamily = try? ManifestReader.peekFamily(directoryURL: modelDirectory)
        let qwen36AutoMTP = peekedFamily == .qwen36 && hasMTPSidecar
            && specEnv != "off" && mtpEnv != "0"
        let specActive = specEnv != "off" && (
            specEnv == "on" || specEnv == "mtp" || specEnv == "pld"
                || mtpEnv == "1" || pldEnv == "1" || qwen36AutoMTP
        )
            || ProcessInfo.processInfo.environment["MFERENCE_FORCE_LOGITS_HEAD"] == "0"
        let runtime = RuntimeConfiguration(
            expertCacheSlots: configSlots,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: !specActive)
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: streamingMode,
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: verification)
        let forwardRuntime = try ForwardRunnerFactory.make(model: model,
                                                            context: context,
                                                            maxContext: maxContext,
                                                            runtimeConfiguration: runtime)
        if qwen36AutoMTP,
           env["MFERENCE_SPEC_K"] == nil,
           env["MFERENCE_MTP_K"] == nil,
           env["MFERENCE_SPEC_ADAPTIVE"] != "0" {
            (forwardRuntime.producer as? RealForwardRunner)?.specAdaptiveDraftOverride = true
        }
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let templateDigest = SHA256.hash(data: templateData)
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            forwardRuntime.prefillConfig.mode.rawValue,
            String(forwardRuntime.prefillConfig.chunkTokens),
            runtime.headPath.rawValue,
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: forwardRuntime.kvStorageMode.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  runner: forwardRuntime.producer,
                                  scratch: scratch,
                                  prefillConfig: forwardRuntime.prefillConfig,
                                  maxContext: maxContext,
                                  promptCacheMode: promptCacheMode,
                                  promptCacheDomain: promptCacheDomain)
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: MFTokenizer,
                 runner: any ContinuableLogitProducer,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 promptCacheDomain: ServerPromptCacheDomain) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.usesSwiftQwenTemplate = tokenizer.isSwiftQwen
        self.supportsQwenReasoningEffort = tokenizer.supportsQwenReasoningEffort
        self.chatDialect = tokenizer.dialect
        self.modelFamily = model.config.family
        self.checkpointID = model.modelID
        self.runner = runner
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
    }

    /// Renders the prompt and checks it against the context window. Runs
    /// before the response status is committed, and touches no generation
    /// state, so it is safe to run while another request is generating.
    public func prepare(_ request: ValidatedChatRequest) throws -> PreparedGeneration {
        let needsToolTemplate = !request.tools.isEmpty
            || request.messages.contains {
                $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
            }
        let alignedMessages = ServerIntentAligner.alignMessages(
            request.messages,
            tools: request.tools,
            dialect: chatDialect)
        let alignedRequest = ValidatedChatRequest(
            reasoningEffort: request.reasoningEffort,
            messages: alignedMessages,
            tools: request.tools,
            stream: request.stream,
            includeUsage: request.includeUsage,
            generationConfig: request.generationConfig,
            maximumCompletionTokens: request.maximumCompletionTokens)
        let promptProfile = ServerIntentAligner.classifyPromptThinkingProfile(
            messages: alignedRequest.messages,
            tools: alignedRequest.tools,
            reasoningEffort: alignedRequest.reasoningEffort)
        let encodedPromptIDs: [Int32]
        do {
            encodedPromptIDs = try tokenizer.encodeChat(
                messages: alignedRequest.messages,
                tools: alignedRequest.tools,
                reasoningEffort: alignedRequest.reasoningEffort)
        } catch {
            throw ServerRequestError.invalid(message: String(describing: error),
                                              param: "messages", code: "invalid_chat_template")
        }
        let promptIDs = ServerIntentAligner.applyZeroThinkFastPathIfNeeded(
            promptIDs: encodedPromptIDs,
            tokenizer: tokenizer,
            profile: promptProfile)
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return PreparedGeneration(request: alignedRequest,
                                  promptIDs: promptIDs,
                                  needsToolTemplate: needsToolTemplate)
    }

    private func parkLiveRunnerSnapshotIfNeeded(maxSlots: Int) {
        guard maxSlots > 1,
              let liveID = liveRunnerCacheEntryID,
              promptCache.contains(slotID: liveID),
              let liveEntry = promptCache.entry(for: liveID),
              let realRunner = runner as? RealForwardRunner else {
            return
        }
        if runner.continuationPosition != liveEntry.kvPosition {
            guard (try? runner.prepareForContinuation(expectedPosition: liveEntry.kvPosition)) != nil else {
                promptCache.removeSlot(liveID)
                cachedSnapshots.removeValue(forKey: liveID)
                liveRunnerCacheEntryID = nil
                return
            }
        }
        if let snapshot = realRunner.capturePrefixSnapshot() {
            cachedSnapshots[liveID] = snapshot
        }
    }

    public func generate(
        _ prepared: PreparedGeneration,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let request = prepared.request
        let needsToolTemplate = prepared.needsToolTemplate
        let promptIDs = prepared.promptIDs
        let savedCacheState = promptCache.snapshotState()
        let savedLiveID = liveRunnerCacheEntryID
        let savedSnapshots = cachedSnapshots
        var matchedSlotID: UUID? = nil
        var completed = false
        defer {
            if !completed {
                let rollbackSnap = savedLiveID.flatMap { cachedSnapshots[$0] ?? savedSnapshots[$0] }
                promptCache.restoreState(savedCacheState)
                cachedSnapshots = savedSnapshots
                if let savedLiveID,
                   let savedPos = savedCacheState.entry(for: savedLiveID)?.kvPosition {
                    if let rollbackSnap,
                       (runner as? RealForwardRunner)?.restorePrefixSnapshot(rollbackSnap) == true,
                       (try? runner.prepareForContinuation(expectedPosition: savedPos)) != nil {
                        liveRunnerCacheEntryID = savedLiveID
                    } else if (try? runner.prepareForContinuation(expectedPosition: savedPos)) != nil {
                        liveRunnerCacheEntryID = savedLiveID
                    } else {
                        promptCache.removeSlot(savedLiveID)
                        cachedSnapshots.removeValue(forKey: savedLiveID)
                        liveRunnerCacheEntryID = nil
                        runner.reset()
                    }
                } else {
                    liveRunnerCacheEntryID = nil
                    runner.reset()
                }
            }
        }

        let promptProfile = ServerIntentAligner.classifyPromptThinkingProfile(
            messages: request.messages,
            tools: request.tools,
            reasoningEffort: request.reasoningEffort)
        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            let targetMaxSlots = ServerPromptCache.recommendedMaxSlots(
                availableMemoryBytes: ServerIntentAligner.currentAvailableMemoryBytes())
            let evictedForMemory = promptCache.setMaxSlots(
                targetMaxSlots,
                preferredLiveSlotID: liveRunnerCacheEntryID)
            for id in evictedForMemory {
                cachedSnapshots.removeValue(forKey: id)
            }
            let realRunner = runner as? RealForwardRunner
            let liveID = liveRunnerCacheEntryID
            let cacheMatch = promptCache.matchSlot(
                domain: promptCacheDomain,
                request: request,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer,
                canUseSlot: { slotID in
                    slotID == liveID || (targetMaxSlots > 1 && realRunner != nil && cachedSnapshots[slotID] != nil)
                },
                rewindablePosition: { slotID, maxPosition in
                    if slotID == liveID {
                        return runner.rewindablePosition(atOrBefore: maxPosition)
                    }
                    guard targetMaxSlots > 1 else { return nil }
                    return cachedSnapshots[slotID]?.rewindablePosition(atOrBefore: maxPosition)
                })
            switch cacheMatch {
            case .miss:
                parkLiveRunnerSnapshotIfNeeded(maxSlots: targetMaxSlots)
                if targetMaxSlots <= 1 {
                    promptCache.invalidate()
                    cachedSnapshots.removeAll()
                }
                liveRunnerCacheEntryID = nil
                effectivePromptIDs = promptIDs
                completionStart = .reset
                let eot = tokenizer.endOfTurnID
                if let firstEotIdx = effectivePromptIDs.firstIndex(of: eot),
                   firstEotIdx + 1 >= 32,
                   firstEotIdx + 1 < effectivePromptIDs.count {
                    runner.setBasePrefixCheckpointPosition(firstEotIdx + 1)
                } else {
                    runner.setBasePrefixCheckpointPosition(nil)
                }
            case .hit(let hitSlotID, let effective, let cached):
                let entryKVPos = promptCache.entry(for: hitSlotID)?.kvPosition ?? cached
                let isPartialBranch = cached < entryKVPos
                var slotActivated = true
                if hitSlotID != liveRunnerCacheEntryID {
                    parkLiveRunnerSnapshotIfNeeded(maxSlots: targetMaxSlots)
                    if let snap = cachedSnapshots[hitSlotID],
                       let realRunner,
                       realRunner.restorePrefixSnapshot(snap) {
                        liveRunnerCacheEntryID = hitSlotID
                    } else {
                        slotActivated = false
                    }
                } else if isPartialBranch && targetMaxSlots > 1 {
                    parkLiveRunnerSnapshotIfNeeded(maxSlots: targetMaxSlots)
                }
                if slotActivated {
                    matchedSlotID = isPartialBranch ? nil : hitSlotID
                    effectivePromptIDs = ServerIntentAligner.applyZeroThinkFastPathIfNeeded(
                        promptIDs: effective,
                        tokenizer: tokenizer,
                        profile: promptProfile)
                    completionStart = .resume(cachedPromptTokens: cached)
                    runner.setBasePrefixCheckpointPosition(nil)
                } else {
                    promptCache.removeSlot(hitSlotID)
                    cachedSnapshots.removeValue(forKey: hitSlotID)
                    liveRunnerCacheEntryID = nil
                    effectivePromptIDs = promptIDs
                    completionStart = .reset
                    runner.setBasePrefixCheckpointPosition(nil)
                }
            }
        } else {
            promptCache.invalidate()
            cachedSnapshots.removeAll()
            liveRunnerCacheEntryID = nil
            effectivePromptIDs = promptIDs
            completionStart = .reset
            runner.setBasePrefixCheckpointPosition(nil)
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        let startsInThinking = tokenizer.startsInThinking(
            reasoningEffort: request.reasoningEffort, promptIDs: effectivePromptIDs)
        let config = ServerIntentAligner.tunedGenerationConfig(
            request.generationConfig,
            maxCompletionTokens: request.maximumCompletionTokens,
            remainingContext: maxContext - effectivePromptIDs.count,
            startsInThinking: startsInThinking,
            reasoningEffort: request.reasoningEffort,
            promptProfile: promptProfile)
        let activeContextEstimate = effectivePromptIDs.count + config.maxNewTokens
        let alignEnv = ProcessInfo.processInfo.environment["MFERENCE_INTENT_ALIGN"]?.lowercased()
        let intentAlignEnabled = alignEnv != "0" && alignEnv != "off"
        (runner as? RealForwardRunner)?.specMaxDraftSlotCapOverride = intentAlignEnabled
            ? ServerIntentAligner.recommendedMaxSlotSupportedK(activeContextTokens: activeContextEstimate)
            : nil

        let countsPayloadTokens = tokenizer.dialect == .chatml
            || tokenizer.dialect == .glm5 || tokenizer.dialect == .minicpm
        let decoder = countsPayloadTokens || needsToolTemplate || startsInThinking
            ? StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: Set(request.tools.map(\.name)),
                startsInThought: startsInThinking,
                toolDefinitions: request.tools)
            : nil
        if config.maxThinkingTokens > 0 {
            decoder?.maxThinkingTokens = config.maxThinkingTokens
        }
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var reasoning = ""
        // Always capture reasoning internally for prompt-cache continuation matching,
        // while keeping external SSE reasoning events gated on usesSourceQwenTemplate.
        let emitReasoningEvents = tokenizer.usesSourceQwenTemplate(reasoningEffort: request.reasoningEffort)
        decoder?.onReasoning = { text in
            reasoning += text
            if emitReasoningEvents {
                onEvent(.reasoning(text))
            }
        }
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false

        let result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: effectivePromptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: completionStart,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        let events = if let decoder {
                            try decoder.consume(tokenID: tokenID, delta: delta)
                        } else {
                            delta.isEmpty ? [] : [StructuredAssistantEvent.content(delta)]
                        }
                        for event in events {
                            switch event {
                            case .content(let text):
                                let visible = stopMatcher.push(text)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    case .tail(let text):
                        // Flush text must pass through the decoder like any
                        // delta: committing it directly would reorder it
                        // ahead of a withheld DSML-prefix tail and skip
                        // marker scanning.
                        let events = if let decoder {
                            try decoder.consumeFlushedText(text)
                        } else {
                            text.isEmpty ? [] : [StructuredAssistantEvent.content(text)]
                        }
                        for event in events {
                            switch event {
                            case .content(let flushed):
                                let visible = stopMatcher.push(flushed)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
        }
        if let decodingError { throw decodingError }
        if let decoder {
            for event in try decoder.finish() {
                if case .content(let text) = event {
                    let visible = stopMatcher.push(text)
                    if !visible.isEmpty {
                        content += visible
                        onEvent(.content(visible))
                    }
                }
            }
        }
        if needsToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw GemmaToolCallParserError.malformed
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        if promptCacheMode == .singlePrefix {
            let outcome = promptCache.publish(
                domain: promptCacheDomain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                reasoningContent: reasoning.isEmpty ? nil : reasoning,
                stopStringFiltered: stopMatcher.isStopped,
                replacesSlotID: matchedSlotID)
            for evictedID in outcome.evictedSlotIDs {
                cachedSnapshots.removeValue(forKey: evictedID)
            }
            if let publishedID = outcome.publishedSlotID {
                cachedSnapshots.removeValue(forKey: publishedID)
                liveRunnerCacheEntryID = publishedID
            } else {
                liveRunnerCacheEntryID = nil
            }
        }
        if ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1" {
            let spec = (runner as? RealForwardRunner)?.mtpSpecStats
            let specPart = spec.map {
                let dMs = String(format: "%.1f", Double($0.draftNanos) / 1e6)
                let vMs = String(format: "%.1f", Double($0.verifyNanos) / 1e6)
                let aMs = String(format: "%.1f", Double($0.acceptNanos) / 1e6)
                return " mtp_rounds=\($0.rounds) mtp_drafted=\($0.draftedTokens) mtp_accepted=\($0.acceptedTokens) mtp_emitted=\($0.emittedTokens) mtp_rollbacks=\($0.rollbacks) draft_ms=\(dMs) verify_ms=\(vMs) accept_ms=\(aMs)"
            } ?? ""
            let prefillStr = String(format: "%.4f", result.prefillSeconds)
            let decodeStr = String(format: "%.4f", result.decodeSeconds)
            let metricsLine = "[server-metrics] prefill=\(result.prefillTokens)tok/\(prefillStr)s decode=\(result.newTokens)tok/\(decodeStr)s stop=\(String(describing: result.reason))\(specPart)\n"
            FileHandle.standardError.write(Data(metricsLine.utf8))
        }
        completed = true
        var completion = ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens,
                               completionTokensDetails: decoder?.payloadTokenCounts.map {
                                   .init(reasoningTokens: $0.reasoning,
                                         visibleTokens: stopMatcher.isStopped ? nil : $0.visible)
                               }),
            diagnostics: RuntimeDiagnostics.enabled ? RuntimeDiagnostics(
                result: result,
                memory: .capture(model: model, producer: runner, scratch: scratch)) : nil)
        completion.reasoningContent = (emitReasoningEvents && !reasoning.isEmpty) ? reasoning : nil
        return completion
    }
}

/// Aligns terse or imperfectly phrased user prompts toward direct action/code
/// and configures server generation defaults (loop guard, sliding repetition window,
/// bounded `<think>` token budget, zero-think fast path, and memory-pressure draft caps).
public enum ServerIntentAligner {
    public enum PromptThinkingProfile: Equatable, Sendable {
        case zeroThinkFastPath
        case standard
        case multiPartAlgorithmic
        case strictConstraint
    }

    public static let defaultActionSystemDirective =
        "Complete all derivations and explicitly verify every word-count, length, and formatting constraint inside <think>. Once </think> closes, output the final answer, code, or requested format concisely without repeating intermediate scratchpad steps."

    public static let conciseCodeAndActionDirective =
        "Complete all reasoning concisely inside <think>. Once </think> closes, output the final answer or code directly with zero inline comments."

    public static let conciseVisibleDerivationDirective =
        "Give a direct 4-step solution (under 150 words total): state the simplified closed form immediately, evaluate each step on a single compact line without explaining elementary arithmetic or inverses, and end with FINAL_ANSWER."

    public static func classifyPromptThinkingProfile(
        messages: [MFTokenizer.Message],
        tools: [MFTokenizer.FunctionDefinition] = [],
        reasoningEffort: QwenReasoningEffort? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> PromptThinkingProfile {
        let alignEnv = environment["MFERENCE_INTENT_ALIGN"]?.lowercased()
        if alignEnv == "0" || alignEnv == "off" {
            return .standard
        }
        if reasoningEffort == .off {
            return .zeroThinkFastPath
        }
        if environment["MFERENCE_ZERO_THINK_FASTPATH"] == "1" {
            return .zeroThinkFastPath
        }
        let rawUserText = messages.last(where: { $0.role == .user })?.content?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let lower = rawUserText.lowercased()
        guard !lower.isEmpty else { return .standard }

        // When the prompt explicitly asks to write the derivation in the visible output and end with FINAL_ANSWER,
        // perform a single-pass visible derivation rather than duplicating the derivation inside <think> and visible text.
        if tools.isEmpty && reasoningEffort == nil && lower.contains("final_answer") {
            return .zeroThinkFastPath
        }

        let strictConstraintMarkers = [
            "whitespace-separated words", "matching every rule", "follow every constraint",
            "word count"
        ]
        if strictConstraintMarkers.contains(where: { lower.contains($0) }) {
            return .strictConstraint
        }

        // Multi-part algorithmic/math derivations get a larger thinking budget.
        let multiPartMarkers = [
            "step-by-step", "step by step", "then write", "then compute", "then implement",
            "part 1", "part 2", "prove that", "derive", "multi-step"
        ]
        if multiPartMarkers.contains(where: { lower.contains($0) }) {
            return .multiPartAlgorithmic
        }

        // Simple single-step utility/transform requests bypass <think> directly onto step 1.
        if tools.isEmpty && reasoningEffort == nil && rawUserText.count <= 120 {
            let zeroThinkPrefixes = [
                "convert ", "format as json:", "translate to ", "regex for ",
                "one-liner:", "no think:", "skip thinking:"
            ]
            if zeroThinkPrefixes.contains(where: { lower.hasPrefix($0) }) {
                return .zeroThinkFastPath
            }
        }
        return .standard
    }

    /// Appends `</think>\n\n` directly onto `promptIDs` when `profile == .zeroThinkFastPath`
    /// and the rendered prompt ends inside an open `<think>` block so visible tokens emit on step 1.
    public static func applyZeroThinkFastPathIfNeeded(
        promptIDs: [Int32],
        tokenizer: MFTokenizer,
        profile: PromptThinkingProfile
    ) -> [Int32] {
        guard profile == .zeroThinkFastPath,
              let thinkStartID = tokenizer.thinkStartID,
              let thinkEndID = tokenizer.thinkEndID,
              thinkStartID >= 0, thinkEndID >= 0 else {
            return promptIDs
        }
        let tailWindow = promptIDs.suffix(6)
        guard tailWindow.contains(thinkStartID), !tailWindow.contains(thinkEndID) else {
            return promptIDs
        }
        return promptIDs + [thinkEndID] + tokenizer.encode("\n\n", addBOS: false)
    }

    public static func alignMessages(_ messages: [MFTokenizer.Message],
                                     tools: [MFTokenizer.FunctionDefinition],
                                     dialect: ChatDialect,
                                     environment: [String: String] = ProcessInfo.processInfo.environment) -> [MFTokenizer.Message] {
        guard !messages.isEmpty else { return messages }
        let envMode = environment["MFERENCE_INTENT_ALIGN"]?.lowercased()
        guard envMode != "0" && envMode != "off" else { return messages }
        if messages.contains(where: { $0.role == .system || $0.role == .developer }) {
            return messages
        }
        if envMode == "1" || envMode == "on" || envMode == "always" {
            return [MFTokenizer.Message(role: .system, content: defaultActionSystemDirective)] + messages
        }
        guard let firstUser = messages.first(where: { $0.role == .user }) else {
            return messages
        }
        let firstUserText = firstUser.content?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let firstLower = firstUserText.lowercased()
        guard dialect == .chatml,
              tools.isEmpty,
              !firstUserText.isEmpty else {
            return messages
        }
        if firstLower.contains("final_answer") {
            return [MFTokenizer.Message(role: .system, content: conciseVisibleDerivationDirective)] + messages
        }
        let firstTurnProfile = classifyPromptThinkingProfile(
            messages: [firstUser],
            tools: tools,
            reasoningEffort: nil,
            environment: environment
        )
        if firstTurnProfile == .strictConstraint || firstUserText.count <= 96 {
            return [MFTokenizer.Message(role: .system, content: defaultActionSystemDirective)] + messages
        }
        if firstLower.contains("swift") || firstLower.contains("rust") || firstLower.contains("c++") {
            return [MFTokenizer.Message(role: .system, content: conciseCodeAndActionDirective)] + messages
        }
        return messages
    }

    public static func tunedGenerationConfig(_ base: GenerationConfig,
                                             maxCompletionTokens: Int,
                                             remainingContext: Int,
                                             startsInThinking: Bool,
                                             reasoningEffort: QwenReasoningEffort?,
                                             promptProfile: PromptThinkingProfile = .standard,
                                             environment: [String: String] = ProcessInfo.processInfo.environment) -> GenerationConfig {
        var config = base
        config.maxNewTokens = max(1, min(maxCompletionTokens, remainingContext))
        config.stopStrings = []
        let alignEnv = environment["MFERENCE_INTENT_ALIGN"]?.lowercased()
        if alignEnv == "0" || alignEnv == "off" {
            return config
        }
        if config.loopGuardRepeats == 0 {
            config.loopGuardRepeats = 3
            config.loopGuardPenalty = max(config.loopGuardPenalty, 1.85)
        }
        if config.repetitionWindow == 0 && config.repetitionPenalty != 1.0 {
            config.repetitionWindow = 128
        }
        if promptProfile == .zeroThinkFastPath {
            config.maxThinkingTokens = 0
            return config
        }
        if startsInThinking && config.maxThinkingTokens == 0 && config.maxNewTokens >= 64 {
            if promptProfile == .multiPartAlgorithmic || promptProfile == .strictConstraint {
                if reasoningEffort == nil && config.maxNewTokens >= 512 {
                    // ~62.5% of maxNewTokens (320 at 512, soft-landing [284, 320]) so derivations finish inside <think>.
                    config.maxThinkingTokens = min(768, config.maxNewTokens * 5 / 8)
                } else {
                    config.maxThinkingTokens = min(768, config.maxNewTokens * 4 / 5)
                }
            } else {
                switch reasoningEffort {
                case .low:
                    config.maxThinkingTokens = min(256, config.maxNewTokens * 2 / 3)
                case .medium:
                    config.maxThinkingTokens = min(512, config.maxNewTokens * 5 / 8)
                case .xhigh:
                    config.maxThinkingTokens = min(1024, config.maxNewTokens * 3 / 4)
                default:
                    // Reserve at least ~56% visible headroom on tight max_tokens budgets so coding outputs complete cleanly.
                    if config.maxNewTokens <= 768 {
                        config.maxThinkingTokens = min(512, config.maxNewTokens * 7 / 16)
                    } else {
                        config.maxThinkingTokens = min(768, config.maxNewTokens * 5 / 8)
                    }
                }
            }
        }
        return config
    }

    public static func currentAvailableMemoryBytes(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> UInt64 {
        if let overrideMB = environment["MFERENCE_AVAILABLE_MEMORY_MB"].flatMap(UInt64.init) {
            return overrideMB * 1024 * 1024
        }
        #if canImport(Darwin)
        let avail = mference_os_proc_available_memory()
        if avail > 0 {
            return UInt64(avail)
        }
        #endif
        return ProcessInfo.processInfo.physicalMemory / 2
    }

    /// Dynamically caps `maxSlotSupportedK` under 16 GiB unified-memory pressure or
    /// long active context so speculative verification does not thrash Metal wired buffers.
    public static func recommendedMaxSlotSupportedK(
        availableMemoryBytes: UInt64 = Self.currentAvailableMemoryBytes(),
        activeContextTokens: Int,
        configuredSlotCap: Int = 2
    ) -> Int {
        let cap = max(1, min(4, configuredSlotCap))
        let tightThreshold: UInt64 = 1_536 * 1024 * 1024    // 1.5 GiB
        let moderateThreshold: UInt64 = 3_072 * 1024 * 1024 // 3.0 GiB
        if availableMemoryBytes < tightThreshold || activeContextTokens >= 3072 {
            return 1
        }
        if availableMemoryBytes < moderateThreshold || activeContextTokens >= 1536 {
            return min(cap, 2)
        }
        return cap
    }
}
