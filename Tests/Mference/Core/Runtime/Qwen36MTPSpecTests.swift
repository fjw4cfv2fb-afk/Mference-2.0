import Foundation
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

@Suite(.serialized)
struct Qwen36MTPSpecTests {

    // MARK: - Helpers

    private func withEnv(_ pairs: [(String, String?)], _ body: () async throws -> Void) async throws {
        var saved: [(String, String?)] = []
        saved.reserveCapacity(pairs.count)
        for (key, value) in pairs {
            saved.append((key, ProcessInfo.processInfo.environment[key]))
            if let value {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
        defer {
            for (key, oldValue) in saved.reversed() {
                if let oldValue {
                    setenv(key, oldValue, 1)
                } else {
                    unsetenv(key)
                }
            }
        }
        try await body()
    }

    private func makeAttachedToy() throws -> (dir: URL, shard: URL) {
        let dir = try QwenToySynthetic.write()
        let shard = try QwenToySynthetic.writeMTPShard()
        _ = try MTPAttachTool.run(gturboDirectory: dir.path, shardPath: shard.path)
        return (dir, shard)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// Runs greedy prefill + `steps` decode steps and returns `[postPrefillTok, gen_1, ..., gen_steps]`.
    private func generateGreedy(
        runner: RealForwardRunner,
        logits: MTLBuffer,
        prompt: [Int32],
        steps: Int
    ) async throws -> [UInt32] {
        _ = try await runner.prefillChunked(
            tokens: prompt[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .defaultChunked,
            into: logits,
            onProgress: { _ in }
        )
        var out: [UInt32] = [runner.lastGreedyToken]
        var cur = Int32(bitPattern: runner.lastGreedyToken)
        for step in 0..<steps {
            try await runner.produce(token: cur, position: prompt.count + step, into: logits)
            let next = runner.lastGreedyToken
            out.append(next)
            cur = Int32(bitPattern: next)
        }
        return out
    }

    /// Creates a speculator-armed runner directly so concurrent suites that
    /// call `setenv("MFERENCE_MTP", "0")` cannot interfere during `init`.
    private func makeSpecRunner(
        model: Model,
        context: MetalContext,
        maxContext: Int = 64,
        draftK: Int? = nil,
        runtimeConfiguration: RuntimeConfiguration? = nil
    ) throws -> RealForwardRunner {
        let runner: RealForwardRunner
        if let runtimeConfiguration {
            runner = try RealForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration)
        } else {
            runner = try RealForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext)
        }
        runner.speculator = try Qwen36MTPSpeculator.probe(
            model: model,
            context: context,
            config: model.config,
            maxContext: maxContext)
        runner.specDraftCountOverride = draftK
        return runner
    }

    // MARK: - Tests

    @Test
    func attachProducesLoadableMoEInstall() throws {
        let dir = try QwenToySynthetic.write()
        let shard = try QwenToySynthetic.writeMTPShard()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let result = try MTPAttachTool.run(gturboDirectory: dir.path, shardPath: shard.path)
        #expect(result.tensorCount == 17)
        #expect(result.rewroteWeightsFile == false)

        // Double attach must be rejected before mutating anything.
        #expect(throws: Error.self) {
            _ = try MTPAttachTool.run(gturboDirectory: dir.path, shardPath: shard.path)
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen36Toy(),
                                   integrityPolicy: .fullSha256)

        #expect(model.manifest.sidecars?["mtp"]?.carried == true)
        #expect(model.manifest.sidecars?["mtp"]?.tensorCount == 19)
        let pool = model.manifest.auxiliaryExpertPools?.first { $0.name == "mtp" }
        #expect(pool?.directory == "packed_experts_mtp")
        #expect(pool?.expertsPerLayer == ArchConfig.qwen36Toy().numExperts)
        #expect(pool?.layers.first?.file == "layer_00.bin")
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("packed_experts_mtp/layer_00.bin").path))

        // Verify INT8 quantization on router + scalar gate, INT4 on projections, and +1 norm fold.
        let router = try model.resident(name: "mtp.layers.0.mlp.gate.weight")
        let d = ArchConfig.qwen36Toy().hiddenSize
        let numExperts = ArchConfig.qwen36Toy().numExperts
        #expect(router.dtype == 0)
        #expect(router.length == UInt64(numExperts * d))

        let scalarGate = try model.resident(name: "mtp.layers.0.mlp.shared_expert_gate.weight")
        #expect(scalarGate.dtype == 0)
        #expect(scalarGate.length == UInt64(d))

        let fc = try model.resident(name: "mtp.fc.weight")
        #expect(fc.dtype == 0)
        #expect(fc.length == UInt64(d * (2 * d) / 2))

        // Zero-centered BF16 norm weights (0.0 in shard) must fold +1.0 -> 0x3F80 (1.0f).
        let normView = try model.resident(name: "mtp.norm.weight")
        #expect(normView.dtype == 1)
        let firstBF16 = normView.buffer.contents()
            .advanced(by: Int(normView.offset))
            .load(as: UInt16.self)
        #expect(firstBF16 == 0x3F80)

        let mtp = try Qwen36MTPSpeculator.probe(model: model,
                                               context: ctx,
                                               config: model.config,
                                               maxContext: 64)
        #expect(mtp != nil)
    }

    @Test(arguments: [1, 2, 3, 4])
    func specMatchesPlain_organicDrafts(k: Int) async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let logits = try makeLogits(ctx, vocab: model.config.vocabSize)

        let prompts: [[Int32]] = [
            [1, 2, 3, 4],
            [5, 9, 13, 17, 21, 25, 29, 33],
        ]

        for prompt in prompts {
            let plainRunner = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
            plainRunner.speculator = nil
            #expect(plainRunner.mtp == nil)
            let baseline = try await generateGreedy(runner: plainRunner, logits: logits, prompt: prompt, steps: 20)

            let runner = try makeSpecRunner(model: model, context: ctx, maxContext: 64, draftK: k)
            #expect(runner.mtp is Qwen36MTPSpeculator)
            let specOut = try await generateGreedy(runner: runner, logits: logits, prompt: prompt, steps: 20)
            #expect(specOut == baseline)
            let stats = try #require(runner.mtpSpecStats)
            #expect(stats.rounds > 0)
            #expect(stats.draftedTokens > 0)
            #expect(stats.emittedTokens >= 20 && stats.emittedTokens <= 20 + k)
        }
    }

    @Test
    func specMatchesPlain_injectedDrafts() async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let logits = try makeLogits(ctx, vocab: model.config.vocabSize)
        let prompt: [Int32] = [3, 7, 11, 15]
        let genSteps = 18

        let plainRunner = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
        plainRunner.speculator = nil
        let gold = try await generateGreedy(runner: plainRunner, logits: logits, prompt: prompt, steps: genSteps)

        for multiX in ["1", "0"] {
            // 1) Full acceptance (k = 3)
            try await withEnv([("MFERENCE_USE_MULTIX", multiX)]) {
                let runner = try makeSpecRunner(model: model, context: ctx, maxContext: 64, draftK: 3)
                let mtp = try #require(runner.mtp as? Qwen36MTPSpeculator)
                mtp.draftOverride = { round, k in
                    let base = 1 + round * (k + 1)
                    return (0..<k).map { j in
                        let idx = min(base + j, gold.count - 1)
                        return Int32(bitPattern: gold[idx])
                    }
                }
                let out = try await generateGreedy(runner: runner, logits: logits, prompt: prompt, steps: genSteps)
                #expect(out == gold)
                let stats = try #require(runner.mtpSpecStats)
                #expect(stats.rounds > 0)
                #expect(stats.rollbacks == 0)
                #expect(stats.acceptedTokens == stats.draftedTokens)
                #expect(stats.positionAccepts[0] == stats.positionTrials[0])
            }

            // 2) Zero acceptance (k = 2) -> every round rolls back KV + GDN arena
            try await withEnv([("MFERENCE_USE_MULTIX", multiX)]) {
                let runner = try makeSpecRunner(model: model, context: ctx, maxContext: 64, draftK: 2)
                let mtp = try #require(runner.mtp as? Qwen36MTPSpeculator)
                let wrongTok: Int32 = gold.contains(999) ? 997 : 999
                mtp.draftOverride = { _, k in [Int32](repeating: wrongTok, count: k) }
                let out = try await generateGreedy(runner: runner, logits: logits, prompt: prompt, steps: genSteps)
                #expect(out == gold)
                let stats = try #require(runner.mtpSpecStats)
                #expect(stats.rounds == genSteps)
                #expect(stats.acceptedTokens == 0)
                #expect(stats.rollbacks == stats.rounds)
                #expect(stats.positionTrials[0] == stats.rounds)
                #expect(stats.positionAccepts[0] == 0)
            }

            // 3) Mixed partial acceptance (k = 2): draft[0] matches gold, draft[1] mismatches -> accept 1, rollback 1
            try await withEnv([("MFERENCE_USE_MULTIX", multiX)]) {
                let runner = try makeSpecRunner(model: model, context: ctx, maxContext: 64, draftK: 2)
                let mtp = try #require(runner.mtp as? Qwen36MTPSpeculator)
                mtp.draftOverride = { round, _ in
                    let base = 1 + round * 2
                    let match0 = Int32(bitPattern: gold[min(base, gold.count - 1)])
                    let target1 = base + 1 < gold.count ? gold[base + 1] : 0
                    let bad1: Int32 = target1 == 998 ? 996 : 998
                    return [match0, bad1]
                }
                let out = try await generateGreedy(runner: runner, logits: logits, prompt: prompt, steps: genSteps)
                #expect(out == gold)
                let stats = try #require(runner.mtpSpecStats)
                #expect(stats.rounds > 0)
                #expect(stats.acceptedTokens > 0)
                #expect(stats.acceptedTokens < stats.draftedTokens)
                #expect(stats.rollbacks == stats.rounds)
                #expect(stats.positionAccepts[0] == stats.rounds)
                #expect(stats.positionAccepts[1] == 0)
            }
        }
    }

    @Test
    func killSwitch_disablesMTP() throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let offByMTP = try RealForwardRunner.probeQwen36Speculator(
            model: model, context: ctx, config: model.config, maxContext: 64,
            environment: ["MFERENCE_SPEC": "on", "MFERENCE_MTP": "0"])
        #expect(offByMTP == nil)

        let offBySpec = try RealForwardRunner.probeQwen36Speculator(
            model: model, context: ctx, config: model.config, maxContext: 64,
            environment: ["MFERENCE_SPEC": "off"])
        #expect(offBySpec == nil)

        let armedDefault = try RealForwardRunner.probeQwen36Speculator(
            model: model, context: ctx, config: model.config, maxContext: 64,
            environment: [:])
        #expect(armedDefault is Qwen36MTPSpeculator)
    }

    @Test
    func noMTPSidecar_leavesSpeculatorNilUnlessPLDExplicit() throws {
        let dir = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())

        let defaultSpec = try RealForwardRunner.probeQwen36Speculator(
            model: model, context: ctx, config: model.config, maxContext: 64,
            environment: [:])
        #expect(defaultSpec == nil)

        let explicitPLD = try RealForwardRunner.probeQwen36Speculator(
            model: model, context: ctx, config: model.config, maxContext: 64,
            environment: ["MFERENCE_SPEC": "pld", "MFERENCE_MTP": "0"])
        #expect(explicitPLD is PromptLookupSpeculator)
    }

    @Test
    func multiTurnContinuation_insideAndAfterVerifyRound() async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let logits = try makeLogits(ctx, vocab: model.config.vocabSize)
        let prompt1: [Int32] = [2, 4, 6, 8]
        let turn2Suffix: [Int32] = [31, 37, 41]

        // Baseline single-turn full prompt for post-round continuation.
        let turn1Plain = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
        turn1Plain.speculator = nil
        let turn1Gold = try await generateGreedy(runner: turn1Plain, logits: logits, prompt: prompt1, steps: 6)

        // Case A: Rewind INSIDE a verified round (accepted = 3 -> rewind back to prompt1.count + 2)
        // and continue directly with produce() without re-prefilling.
        let insideRewindPos = prompt1.count + 2
        let insidePlain = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
        insidePlain.speculator = nil
        let insideSeq = try await generateGreedy(runner: insidePlain, logits: logits, prompt: prompt1, steps: 10)
        let insideGoldTail = Array(insideSeq[2..<10])

        do {
            let runner = try makeSpecRunner(model: model, context: ctx, maxContext: 64, draftK: 3)
            let mtp = try #require(runner.mtp as? Qwen36MTPSpeculator)
            // Force all 3 drafts to match on round 0 so runner advances KV past insideRewindPos.
            mtp.draftOverride = { round, k in
                if round == 0 {
                    return (1...k).map { j in Int32(bitPattern: turn1Gold[j]) }
                }
                return []
            }
            _ = try await runner.prefillChunked(tokens: prompt1[...], startPosition: 0,
                                                outputMode: .greedyIfAvailable, config: .defaultChunked,
                                                into: logits, onProgress: { _ in })
            let tok0 = Int32(bitPattern: runner.lastGreedyToken)
            try await runner.produce(token: tok0, position: prompt1.count, into: logits) // emits turn1Gold[1], queues 2 & 3
            try await runner.produce(token: Int32(bitPattern: turn1Gold[1]), position: prompt1.count + 1, into: logits) // emits turn1Gold[2]

            // Now rewind inside the verified span (to prompt1.count + 2), clear override, and continue.
            try runner.prepareForContinuation(expectedPosition: insideRewindPos)
            mtp.draftOverride = nil
            var cur = Int32(bitPattern: turn1Gold[2])
            var continued: [UInt32] = [turn1Gold[2]]
            for s in 0..<7 {
                try await runner.produce(token: cur, position: insideRewindPos + s, into: logits)
                let next = runner.lastGreedyToken
                continued.append(next)
                cur = Int32(bitPattern: next)
            }
            #expect(continued == insideGoldTail)
        }

        // Case B: Normal Turn-2 continuation appending user tokens after Turn 1 verify rounds.
        let fullTurn2Prompt = prompt1 + [Int32(bitPattern: turn1Gold[0]), Int32(bitPattern: turn1Gold[1]),
                                         Int32(bitPattern: turn1Gold[2])] + turn2Suffix
        let turn2Plain = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
        turn2Plain.speculator = nil
        let turn2Gold = try await generateGreedy(runner: turn2Plain, logits: logits, prompt: fullTurn2Prompt, steps: 6)

        do {
            let runner = try makeSpecRunner(model: model, context: ctx, maxContext: 64, draftK: 2)
            _ = try await generateGreedy(runner: runner, logits: logits, prompt: prompt1, steps: 2)
            let contPos = prompt1.count + 3
            try runner.prepareForContinuation(expectedPosition: contPos)
            _ = try await runner.prefillChunked(tokens: turn2Suffix[...],
                                                startPosition: contPos,
                                                outputMode: .greedyIfAvailable,
                                                config: .defaultChunked,
                                                into: logits,
                                                onProgress: { _ in })
            var out: [UInt32] = [runner.lastGreedyToken]
            var cur = Int32(bitPattern: runner.lastGreedyToken)
            for s in 0..<6 {
                try await runner.produce(token: cur, position: fullTurn2Prompt.count + s, into: logits)
                let next = runner.lastGreedyToken
                out.append(next)
                cur = Int32(bitPattern: next)
            }
            #expect(out == turn2Gold)
        }
    }

    @Test
    func sampledMode_usesLogitsWithoutError() async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let logits = try makeLogits(ctx, vocab: model.config.vocabSize)

        let runner = try makeSpecRunner(
            model: model,
            context: ctx,
            maxContext: 64,
            draftK: 2,
            runtimeConfiguration: RuntimeConfiguration(
                expertCacheSlots: 8, prefillChunkTokens: 32, forceLogitsHead: false
            )
        )
        #expect(runner.mtp is Qwen36MTPSpeculator)

        // Switch to full logits mode dynamically (simulating temperature > 0 in MferenceServer).
        runner.setOutputMode(.logits)
        _ = try await runner.prefillChunked(
            tokens: [Int32(1), Int32(2), Int32(3)][...],
            startPosition: 0,
            outputMode: .logits,
            config: .defaultChunked,
            into: logits,
            onProgress: { _ in }
        )
        try await runner.produce(token: 4, position: 3, into: logits)
        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: model.config.vocabSize)
        #expect(Float(ptr[0]).isFinite)
        #expect(Float(ptr[1]).isFinite)

        // Switching back to greedyIfAvailable continues cleanly.
        runner.setOutputMode(.greedyIfAvailable)
        try await runner.produce(token: 5, position: 4, into: logits)
        #expect(runner.lastGreedyToken < UInt32(model.config.vocabSize))
    }

    @Test
    func smallExpertCache_fallsBackToTiledMoEWithoutCrash() async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let ctx = try MetalContext()
        // 8 slots >= topKExperts (8), but < (K + 1) * topKExperts = 5 * 8 = 40 for K=4 on qwen36Toy().
        // Exercises the tiled MoE verify fallback instead of overflowing the 8-slot cache.
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen36Toy(),
                                   streamingMode: .pread(slotCount: 8))
        let logits = try makeLogits(ctx, vocab: model.config.vocabSize)
        let prompt: [Int32] = [1, 3, 5, 7, 9]

        let plainRunner = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
        plainRunner.speculator = nil
        let baseline = try await generateGreedy(runner: plainRunner, logits: logits, prompt: prompt, steps: 14)

        let runner = try makeSpecRunner(model: model, context: ctx, maxContext: 64, draftK: 4)
        let mtp = try #require(runner.mtp as? Qwen36MTPSpeculator)
        // Mix hits and misses while K=4 verifies 5 tokens across an 8-slot cache.
        mtp.draftOverride = { round, k in
            round == 0 ? (1...k).map { Int32(bitPattern: baseline[$0]) } : []
        }
        let specOut = try await generateGreedy(runner: runner, logits: logits, prompt: prompt, steps: 14)
        #expect(specOut == baseline)
    }

    @Test
    func adaptiveDraftDepth_scalesWithAcceptanceRateAndSlotBudget() async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let spec = try #require(try Qwen36MTPSpeculator.probe(
            model: model, context: ctx, config: model.config, maxContext: 64))
        spec.adaptiveDraftEnabled = true

        // Initial default EMA [0.85, 0.75, 0.60, 0.45] selects K = 2.
        #expect(spec.recommendedDraftCount(maxSlotSupportedK: 4) == 2)

        // Consecutive misses drop EMA below threshold and scale down to K = 1.
        for pos in 0..<4 {
            spec.recordAccepted(drafts: [10, 11], emitted: [20], position: 5 + pos, verifyNanos: 1_000)
        }
        #expect(spec.recommendedDraftCount(maxSlotSupportedK: 4) == 1)

        // Phase transition out of <think> into structured code / <tool_call> immediately
        // boosts positional EMA so K steps back up without waiting for slow warmup.
        spec.notifyPhaseTransition(.structuredCodeOrTool)
        #expect(spec.recommendedDraftCount(maxSlotSupportedK: 4) >= 2)

        // Sustained 3-token acceptance scales up to K = 3, bounded by maxSlotSupportedK.
        for pos in 0..<10 {
            spec.recordAccepted(drafts: [10, 11, 12], emitted: [10, 11, 12, 13], position: 10 + pos * 4, verifyNanos: 1_000)
        }
        #expect(spec.recommendedDraftCount(maxSlotSupportedK: 4) == 3)
        #expect(spec.recommendedDraftCount(maxSlotSupportedK: 2) == 2)
    }

    @Test
    func repetitionLoopDetector_identifiesRepeatingSuffixCycles() {
        // Non-repeating sequence -> nil
        #expect(RepetitionLoopDetector.detectLoop(in: [1, 2, 3, 4, 5, 6], minRepeats: 3) == nil)
        // Short 3-run of single token -> ignored (single-token requires >= 4)
        #expect(RepetitionLoopDetector.detectLoop(in: [1, 2, 9, 9, 9], minRepeats: 3) == nil)
        // 4-run of single token -> detected
        #expect(RepetitionLoopDetector.detectLoop(in: [1, 2, 9, 9, 9, 9], minRepeats: 3) == [9])
        // 3 consecutive repeats of a 3-gram [10, 20, 30] -> detected
        let history: [Int32] = [1, 2, 10, 20, 30, 10, 20, 30, 10, 20, 30]
        #expect(RepetitionLoopDetector.detectLoop(in: history, minRepeats: 3) == [10, 20, 30])
        // 3 consecutive repeats of a 32-token code/reasoning cycle (>16 tokens) -> detected with cycle[0] first
        let longCycle = Array(Int32(100)..<Int32(132))
        let longHistory = [Int32(1), Int32(2)] + longCycle + longCycle + longCycle
        let detected = RepetitionLoopDetector.detectLoop(in: longHistory, minRepeats: 3)
        #expect(detected?.first == 100)
        #expect(detected?.count == 32)
    }

    @Test
    func residentWeightSurgery_coAdaptsMTPAndPreservesManifestIntegrity() async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let surgeryResult = try MTPAttachTool.applyResidentWeightSurgery(
            gturboDirectory: dir.path,
            config: MTPAttachTool.SurgeryConfig(
                finalNormScale: 0.96,
                qkNormScale: 1.04,
                qkNormStartLayerRatio: 0.5,
                routerTemperature: 0.92,
                lateLayerResidualScale: 0.94,
                lateLayerStartFraction: 0.5,
                sharedExpertScale: 1.025,
                sharedExpertStartLayerRatio: 0.5,
                mtpNormScale: 1.0,
                mtpHiddenNormScale: 1.02,
                coAdaptMTP: true
            )
        )
        #expect(surgeryResult.coAdaptedMTP == true)
        #expect(surgeryResult.modifiedNormTensors >= 3)
        #expect(surgeryResult.modifiedAffineTensors >= 1)

        // Idempotence guard: manifest records surgery provenance and blocks accidental double-application.
        let manifestData = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
        let manifestJSON = try #require(
            try JSONSerialization.jsonObject(with: manifestData) as? [String: Any])
        #expect(manifestJSON["surgery"] != nil)
        #expect(throws: RepackError.self) {
            _ = try MTPAttachTool.applyResidentWeightSurgery(
                gturboDirectory: dir.path,
                config: MTPAttachTool.SurgeryConfig(finalNormScale: 1.02))
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen36Toy(),
                                   integrityPolicy: .fullSha256)
        let logits = try makeLogits(ctx, vocab: model.config.vocabSize)
        let prompt: [Int32] = [1, 3, 5, 7]

        let plain = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
        plain.speculator = nil
        let baseline = try await generateGreedy(runner: plain, logits: logits, prompt: prompt, steps: 8)

        let specRunner = try makeSpecRunner(model: model, context: ctx, maxContext: 64)
        specRunner.specAdaptiveDraftOverride = true
        let specOut = try await generateGreedy(runner: specRunner, logits: logits, prompt: prompt, steps: 8)
        #expect(specOut == baseline)
    }

    @Test
    func pldHybrid_tracksQwen36ToolCallAndCodeFencePhases() throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let spec = try #require(try Qwen36MTPSpeculator.probe(
            model: model, context: ctx, config: model.config, maxContext: 64))

        // Inside <think> (248068), code fence token (71093) does not open visible code fence.
        spec.notifyTokens(tokens: [248068, 71093][...], startPosition: 0)
        #expect(spec.currentPhase == .thinking)
        spec.notifyTokens(tokens: [248069][...], startPosition: 2)
        #expect(spec.currentPhase == .visibleProse)

        // Actual Qwen 3.6 <tool_call> (248058) and </tool_call> (248059) IDs enter/exit .structuredCodeOrTool,
        // ignoring nested code fences inside tool arguments.
        spec.notifyTokens(tokens: [248058, 71093][...], startPosition: 3)
        #expect(spec.currentPhase == .structuredCodeOrTool)
        spec.notifyTokens(tokens: [248059][...], startPosition: 5)
        #expect(spec.currentPhase == .visibleProse)

        // Standalone markdown code fences (71093 / 52451) toggle .structuredCodeOrTool cleanly.
        spec.notifyTokens(tokens: [71093][...], startPosition: 6)
        #expect(spec.currentPhase == .structuredCodeOrTool)
        spec.notifyTokens(tokens: [52451][...], startPosition: 7)
        #expect(spec.currentPhase == .visibleProse)

        // Idempotent replay of the exact same span must not double-toggle code fences.
        spec.notifyTokens(tokens: [71093][...], startPosition: 8)
        #expect(spec.currentPhase == .structuredCodeOrTool)
        spec.notifyTokens(tokens: [71093][...], startPosition: 8)
        #expect(spec.currentPhase == .structuredCodeOrTool)
    }

    @Test
    func prefixSnapshot_roundTripsBitExactContinuation() async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let logits = try makeLogits(ctx, vocab: model.config.vocabSize)
        let promptA: [Int32] = [1, 3, 5, 7, 9, 11]
        let promptB: [Int32] = [2, 4, 6, 8]

        // Reference continuation directly from promptA
        let refRunner = try makeSpecRunner(model: model, context: ctx, maxContext: 64)
        let fullA = try await generateGreedy(runner: refRunner, logits: logits, prompt: promptA, steps: 6)

        // Interleave promptA -> snapshot -> overwrite runner with promptB -> restore -> continue promptA
        let runner = try makeSpecRunner(model: model, context: ctx, maxContext: 64)
        _ = try await runner.prefillChunked(
            tokens: promptA[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .defaultChunked,
            into: logits,
            onProgress: { _ in })
        var resumedA: [UInt32] = [runner.lastGreedyToken]
        let snapA = try #require(runner.capturePrefixSnapshot())

        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: promptB[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .defaultChunked,
            into: logits,
            onProgress: { _ in })

        #expect(runner.restorePrefixSnapshot(snapA) == true)
        try runner.prepareForContinuation(expectedPosition: promptA.count)
        var cur = Int32(bitPattern: fullA[0])
        for step in 0..<6 {
            try await runner.produce(token: cur, position: promptA.count + step, into: logits)
            let next = runner.lastGreedyToken
            resumedA.append(next)
            cur = Int32(bitPattern: next)
        }
        #expect(resumedA == fullA)
    }

    @Test
    func mtpCacheWraparound_matchesPlainBitExact() async throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let logits = try makeLogits(ctx, vocab: model.config.vocabSize)
        let prompt: [Int32] = [1, 2, 3, 4]
        let steps = 10

        let plainRunner = try RealForwardRunner(model: model, context: ctx, maxContext: 16)
        plainRunner.speculator = nil
        let baseline = try await generateGreedy(runner: plainRunner, logits: logits, prompt: prompt, steps: steps)

        let specRunner = try makeSpecRunner(model: model, context: ctx, maxContext: 16, draftK: 3)
        let specOut = try await generateGreedy(runner: specRunner, logits: logits, prompt: prompt, steps: steps)
        #expect(specOut == baseline)

        // Directly force commitAcceptPass's maxContext wraparound branch (firstMissing reset)
        // to ensure no out-of-bounds index when mtpLen + missingCount exceeds maxContext.
        let mtp = try #require(specRunner.mtp as? Qwen36MTPSpeculator)
        mtp.reset()
        let dBytes = model.config.hiddenSize * MemoryLayout<Float16>.stride
        let dummyHidden = try #require(ctx.device.makeBuffer(length: 4 * dBytes, options: .storageModeShared))
        var pos = 1
        while pos <= 13 {
            try mtp.commitAcceptPass(bonus: Int32(pos),
                                     emitted: [UInt32(pos + 1), UInt32(pos + 2)],
                                     position: pos,
                                     targetNormedHidden: dummyHidden,
                                     acceptedCount: 1)
            pos += 2
        }
        try mtp.commitAcceptPass(bonus: 20,
                                 emitted: [21, 22],
                                 position: pos,
                                 targetNormedHidden: dummyHidden,
                                 acceptedCount: 1)
        try mtp.flushPendingAcceptIfNeeded()
        #expect(mtp.canRunRound(position: pos + 1) == false || mtp.specStats.rounds >= 0)
    }

    @Test
    func pldHybrid_disambiguatesRepeated4GramWith5Gram() throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwen36Toy())
        let spec = try #require(try Qwen36MTPSpeculator.probe(
            model: model, context: ctx, config: model.config, maxContext: 64))

        // Transition to structured code/tool phase and seed history where the 4-gram [20, 30, 40, 50]
        // appears with two conflicting continuations (99 vs 77), while the 5-gram [11, 20, 30, 40, 50]
        // uniquely predicts 77.
        spec.notifyTokens(tokens: [248058, 10, 20, 30, 40, 50, 99, 11, 20, 30, 40, 50, 77, 1, 11, 20, 30, 40][...],
                          startPosition: 0)
        #expect(spec.currentPhase == .structuredCodeOrTool)
        #expect(spec.pldUniqueContinuation(after: 40, extra: 50) == 77)

        // When even the 5-gram conflicts ([11, 20, 30, 40, 50] -> 77 vs 88), a unique 6-gram wins;
        // if 6-gram also conflicts, PLD cleanly declines (returns nil).
        spec.notifyTokens(tokens: [2, 11, 20, 30, 40, 50, 88, 2, 11, 20, 30, 40][...], startPosition: 18)
        #expect(spec.pldUniqueContinuation(after: 40, extra: 50) == 88)
        spec.notifyTokens(tokens: [2, 11, 20, 30, 40, 50, 91, 2, 11, 20, 30, 40][...], startPosition: 30)
        #expect(spec.pldUniqueContinuation(after: 40, extra: 50) == nil)
    }

    @Test
    func residentWeightSurgery_rejectsInvalidScales() throws {
        let (dir, shard) = try makeAttachedToy()
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: shard)
        }

        #expect(throws: RepackError.self) {
            _ = try MTPAttachTool.applyResidentWeightSurgery(
                gturboDirectory: dir.path,
                config: MTPAttachTool.SurgeryConfig(qkNormScale: 0))
        }
        #expect(throws: RepackError.self) {
            _ = try MTPAttachTool.applyResidentWeightSurgery(
                gturboDirectory: dir.path,
                config: MTPAttachTool.SurgeryConfig(sharedExpertScale: -1.0))
        }
        #expect(throws: RepackError.self) {
            _ = try MTPAttachTool.applyResidentWeightSurgery(
                gturboDirectory: dir.path,
                config: MTPAttachTool.SurgeryConfig(lateLinearResidualScale: .nan))
        }
        #expect(throws: RepackError.self) {
            _ = try MTPAttachTool.applyResidentWeightSurgery(
                gturboDirectory: dir.path,
                config: MTPAttachTool.SurgeryConfig(qkNormStartLayerRatio: 1.5))
        }
    }
}
