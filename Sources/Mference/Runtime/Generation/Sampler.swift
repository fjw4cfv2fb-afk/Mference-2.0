import Foundation
import Metal

/// Generation knobs threaded from the caller through the `Generator` into the
/// sampler. Pure value type; one per `generate(...)` call.
///
/// Canonical home is here (the sampler is the primary consumer); `Generator`
/// reuses the same type rather than redeclaring it.
public struct GenerationConfig: Sendable {
    public var maxNewTokens: Int = 256
    public var temperature: Float = 1.0
    public var topK: Int? = nil            // nil = no truncation
    public var topP: Float? = nil          // nil = no nucleus truncation
    public var repetitionPenalty: Float = 1.0
    public var seed: UInt64? = nil         // nil = nondeterministic
    public var stopStrings: [String] = []
    public var extraStopTokens: Set<Int32> = []
    /// Bounded sliding window for repetition penalty (`0` = unbounded full history).
    public var repetitionWindow: Int = 0
    /// Consecutive suffix repeats (`>= 2`) that trigger a single-step `.logits` penalty break
    /// on the fused-greedy / MTP path before resuming fast speculation (`0` = disabled).
    public var loopGuardRepeats: Int = 0
    /// Repetition penalty applied to the repeating cycle tokens when `loopGuardRepeats` fires.
    public var loopGuardPenalty: Float = 1.3
    /// Optional ceiling on `<think>` reasoning tokens before forcing `</think>\n\n` transition (`0` = unlimited).
    public var maxThinkingTokens: Int = 0

    public init(maxNewTokens: Int = 256,
                temperature: Float = 1.0,
                topK: Int? = nil,
                topP: Float? = nil,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stopStrings: [String] = [],
                extraStopTokens: Set<Int32> = [],
                repetitionWindow: Int = 0,
                loopGuardRepeats: Int = 0,
                loopGuardPenalty: Float = 1.3,
                maxThinkingTokens: Int = 0) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stopStrings = stopStrings
        self.extraStopTokens = extraStopTokens
        self.repetitionWindow = repetitionWindow
        self.loopGuardRepeats = loopGuardRepeats
        self.loopGuardPenalty = loopGuardPenalty
        self.maxThinkingTokens = maxThinkingTokens
    }

    public func validate() throws {
        guard maxNewTokens > 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "maxNewTokens must be greater than zero")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "temperature must be finite and nonnegative")
        }
        if let topK, !(1...256).contains(topK) {
            throw GeneratorError.invalidGenerationConfig(
                "topK must be between 1 and 256")
        }
        if let topP, (!topP.isFinite || topP <= 0 || topP > 1) {
            throw GeneratorError.invalidGenerationConfig(
                "topP must be greater than zero and at most one")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw GeneratorError.invalidGenerationConfig(
                "topP below one requires topK; full-vocabulary nucleus sampling is not implemented")
        }
    }

}

/// Which path a `sample(...)` call took.
enum SamplePath: Sendable, Equatable {
    case greedyGPU
    case gpuSampled
    case hostPenalty
}

/// Turns `GenerationConfig` + a logits buffer into one token id, staying
/// GPU-resident wherever the kernels allow.
///
/// The built `sample` kernel already does temperature / top-k / top-p / seeded
/// draw / greedy argmax on GPU reading softmaxed probs, so this type's job is:
/// (1) run the softcap+softmax front-end (`logit_softcap_softmax`), (2) apply
/// repetition penalty — the one policy that needs `history` random access — as
/// a single in-place CPU pass over the (shared) logits before the front-end,
/// and (3) derive a per-position seed so a fixed `seed` is reproducible across
/// token positions.
///
/// The chosen id lands in a 1-element UInt32 buffer. The generation loop reads
/// that value after the command buffer completes.
///
/// Truncation follows mlx-lm's sampler order: Top-P is computed from the
/// model's full probability distribution, Top-K caps that surviving set, and
/// temperature is applied only to the final categorical draw.
final class Sampler {
    private let softcap: LogitSoftcapSoftmax
    private let sampleKernel: Sample
    private let topK64Kernel: SampleTopK64
    let vocab: Int
    private let logitSoftcap: Float

    init(context: MetalContext, vocab: Int = 262_144,
                logitSoftcap: Float = 30.0) throws {
        self.softcap = try LogitSoftcapSoftmax(context: context)
        self.sampleKernel = try Sample(context: context)
        self.topK64Kernel = try SampleTopK64(context: context, vocab: vocab)
        self.vocab = vocab
        self.logitSoftcap = logitSoftcap
    }

    /// Encode the sampler onto `commandBuffer`. `logits` is FP16 [vocab],
    /// post-lm_head and pre-softcap, in a `.storageModeShared` buffer (the
    /// repetition-penalty path edits it in place). `probs` is a preallocated
    /// FP16 [vocab] scratch. `outToken` holds one UInt32. `position` indexes the
    /// per-position seed advance. Returns the path taken.
    @discardableResult
    func sample(commandBuffer: MTLCommandBuffer,
                       logits: MTLBuffer,
                       probs: MTLBuffer,
                       history: [Int32],
                       config: GenerationConfig,
                       position: Int,
                       outToken: MTLBuffer) -> SamplePath {
        let v = UInt32(vocab)

        let appliedPenalty = config.repetitionPenalty != 1.0 && !history.isEmpty
        if appliedPenalty {
            let effectiveHistory = config.repetitionWindow > 0
                ? Array(history.suffix(config.repetitionWindow))
                : history
            applyRepetitionPenaltyInPlace(logits: logits,
                                          history: effectiveHistory,
                                          penalty: config.repetitionPenalty)
        }

        softcap.encode(commandBuffer: commandBuffer,
                       logits: logits, probs: probs, v: v, softcap: logitSoftcap)

        let isGreedy = config.temperature == 0
        let seed = Self.seedFor(config: config, position: position)
        if config.temperature > 0,
           config.topK == 64 {
            topK64Kernel.encode(commandBuffer: commandBuffer,
                                probs: probs,
                                outToken: outToken,
                                temperature: config.temperature,
                                topP: config.topP ?? 1.0,
                                seed: seed)
        } else {
            sampleKernel.encode(commandBuffer: commandBuffer,
                                probs: probs, outToken: outToken, v: v,
                                temperature: isGreedy ? 0.0 : config.temperature,
                                topK: UInt32(config.topK ?? 0),
                                topP: config.topP ?? 1.0,
                                seed: seed,
                                position: UInt32(position))
        }

        if appliedPenalty { return .hostPenalty }
        return isGreedy ? .greedyGPU : .gpuSampled
    }

    // MARK: - Repetition penalty (host, in place)

    /// HF convention: for each token id seen in `history`, a positive logit is
    /// divided by `penalty`, a negative logit multiplied. Edits the shared
    /// `logits` buffer in place — no full-buffer copy, only the unique history
    /// entries are touched (counted for the audit).
    ///
    /// The penalty must act on the POST-softcap logit (HF applies it to the
    /// model's output logits, and Gemma's output includes the 30*tanh(z/30)
    /// cap). Real Gemma 4 raw logits reach the hundreds, deep in tanh
    /// saturation, where dividing the raw value by 1.1 moves the capped logit
    /// by ~nothing — the penalty silently no-ops on exactly the
    /// high-confidence tokens that form repetition loops. So: softcap the raw
    /// value, penalize, and invert through atanh so the downstream
    /// softcap+softmax kernel reproduces the penalized capped logit.
    private func applyRepetitionPenaltyInPlace(logits: MTLBuffer,
                                               history: [Int32],
                                               penalty: Float) {
        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        var seen = Set<Int32>()
        seen.reserveCapacity(history.count)
        for id in history {
            guard id >= 0 && Int(id) < vocab, seen.insert(id).inserted else { continue }
            let i = Int(id)
            let z = Float(ptr[i])
            let penalized: Float
            if logitSoftcap > 0 {
                let capped = logitSoftcap * tanhf(z / logitSoftcap)
                let cappedPenalized = capped > 0 ? capped / penalty : capped * penalty
                // A saturated negative logit times the penalty can leave the
                // softcap's open interval; clamp inside it so atanh stays
                // finite.
                let limit = logitSoftcap * 0.9999
                let clamped = max(min(cappedPenalized, limit), -limit)
                penalized = logitSoftcap * atanhf(clamped / logitSoftcap)
            } else {
                penalized = z > 0 ? z / penalty : z * penalty
            }
            ptr[i] = Float16(penalized)
        }
    }

    /// Two-tier loop-escape penalty: strongly suppresses `cycleTokens[0]` (the exact
    /// token that restarts the repeating suffix cycle at the current position) with
    /// `primaryPenalty`, while applying only a gentle `secondaryPenalty` to the
    /// remaining tokens in the cycle so valid code/JSON syntax tokens (`{`, `"`, `:`, `\n`)
    /// are not starved.
    func penalizeCycleInPlace(logits: MTLBuffer,
                              cycleTokens: [Int32],
                              primaryPenalty: Float,
                              secondaryPenalty: Float = 1.06) {
        guard let head = cycleTokens.first else { return }
        if primaryPenalty > 1.0 {
            applyRepetitionPenaltyInPlace(logits: logits,
                                          history: [head],
                                          penalty: primaryPenalty)
        }
        if cycleTokens.count > 1, secondaryPenalty > 1.0 {
            applyRepetitionPenaltyInPlace(logits: logits,
                                          history: Array(cycleTokens.dropFirst()),
                                          penalty: secondaryPenalty)
        }
    }

    // MARK: - Seed

    /// Deterministic per-position seed when `config.seed != nil` so a fixed seed
    /// reproduces across token positions; clock-derived (non-zero) otherwise.
    /// xorshift64 in the kernel has a fixed point at 0, so we never emit 0.
    static func seedFor(config: GenerationConfig, position: Int) -> UInt64 {
        if let s = config.seed {
            let mixed = Self.splitmix64(s &+ UInt64(bitPattern: Int64(position)))
            return mixed == 0 ? 0x9E3779B97F4A7C15 : mixed
        }
        var t = timespec()
        clock_gettime(CLOCK_MONOTONIC, &t)
        let raw = UInt64(bitPattern: Int64(t.tv_nsec)) &* 0x9E3779B97F4A7C15
            &+ UInt64(bitPattern: Int64(t.tv_sec))
        return raw == 0 ? 0x9E3779B97F4A7C15 : raw
    }

    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

/// Detects deterministic suffix repetition loops (`L` in `1...maxPatternLength` repeated
/// `>= minRepeats` times consecutively at the tail of `history`) so greedy / MTP generation
/// can execute a single-step `.logits` penalty break and immediately resume fast speculation.
public enum RepetitionLoopDetector {
    public static func detectLoop(in history: [Int32],
                                  maxPatternLength: Int = 64,
                                  minRepeats: Int = 3) -> [Int32]? {
        guard minRepeats >= 2, maxPatternLength >= 1 else { return nil }
        let n = history.count
        let maxL = min(maxPatternLength, n / minRepeats)
        guard maxL >= 1 else { return nil }

        for patternLen in 1...maxL {
            // Single-token runs require at least max(minRepeats, 4) identical tokens
            // to avoid false-positive breaks on natural formatting runs (e.g. "   " or "...").
            let requiredRepeats = (patternLen == 1) ? max(minRepeats, 4) : minRepeats
            let span = patternLen * requiredRepeats
            guard n >= span else { continue }

            let baseStart = n - patternLen
            var matches = true
            for r in 1..<requiredRepeats {
                let repStart = n - (r + 1) * patternLen
                for offset in 0..<patternLen {
                    if history[repStart + offset] != history[baseStart + offset] {
                        matches = false
                        break
                    }
                }
                if !matches { break }
            }
            if matches {
                var unique: [Int32] = []
                unique.reserveCapacity(patternLen)
                for offset in 0..<patternLen {
                    let tok = history[baseStart + offset]
                    if !unique.contains(tok) {
                        unique.append(tok)
                    }
                }
                return unique
            }
        }
        return nil
    }
}
