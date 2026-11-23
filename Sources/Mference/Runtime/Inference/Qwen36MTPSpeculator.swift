import Foundation
import Metal

/// Genuine MoE Multi-Token Prediction (MTP) speculative drafter for Qwen 3.6.
///
/// Reconstructs the official Qwen3-Next / Qwen 3.6 `mtp.*` architecture on Metal:
///   concat(pre_fc_norm_embedding(emb), pre_fc_norm_hidden(hPrev)) -> fc ->
///   single full-attention + top-k routed MoE + scalar-gated shared-expert block ->
///   mtp.norm -> target lm_head.
///
/// The 17 resident `mtp.*` tensors are read from `model.resident(name:)`, while the
/// 256 routed experts (`packed_experts_mtp/layer_00.bin`) are mapped directly via
/// `ResidentExpertStreamer(strategy: .mapped)` so only active experts are paged in.
final class Qwen36MTPSpeculator: SpeculativeDrafter {

    // MARK: - Constants & Error Types

    /// GDN speculative verify arena in `RealForwardRunner` holds 4 slots, capping K at 4.
    static let maxDraftSlots = 4
    private static let epsilon: Float = 1e-6

    enum Error: Swift.Error, CustomStringConvertible {
        case invalidWeights(String)
        case commandFailed(String)

        var description: String {
            switch self {
            case .invalidWeights(let detail):
                return "Qwen 3.6 MTP weight error: \(detail)"
            case .commandFailed(let detail):
                return "Qwen 3.6 MTP execution error: \(detail)"
            }
        }
    }

    // MARK: - Loaded Model Weights & Layout

    private struct Weights {
        let fc: TensorView
        let preFcNormEmbedding: TensorView
        let preFcNormHidden: TensorView
        let mtpNorm: TensorView
        let inputNorm: TensorView
        let postAttnNorm: TensorView
        let qProj: TensorView
        let kProj: TensorView
        let vProj: TensorView
        let oProj: TensorView
        let qNorm: TensorView
        let kNorm: TensorView
        let router: TensorView
        let sharedScalarGate: TensorView
        let sharedGate: TensorView
        let sharedUp: TensorView
        let sharedDown: TensorView
        let lmHead: TensorView

        let sharedGateProj: SharedExpertProjection
        let sharedUpProj: SharedExpertProjection
        let sharedDownProj: SharedExpertProjection
        let sharedScalarGateProj: SharedExpertProjection
    }

    private struct QuantMatrixSpec {
        let suffix: String
        let rows: Int
        let cols: Int
        let bits: Int
    }

    private struct NormSpec {
        let suffix: String
        let rows: Int
    }

    // MARK: - Speculative State & Stats

    private let model: Model
    private let cfg: ArchConfig
    private let queue: MTLCommandQueue
    private let weights: Weights
    private let expertStreamer: ResidentExpertStreamer
    private let slab: MTLBuffer
    private let slotTable: MTLBuffer
    private let expertStride: Int
    private let expertOffsets: MoEExpertOffsets
    private let sharedF: Int
    private let moeF: Int
    private let maxContext: Int

    private let d: Int
    private let headDim: Int
    private let numHeads: Int
    private let numKV: Int
    private let qDim: Int
    private let kvDim: Int
    private let numExperts: Int
    private let topK: Int
    private let rotaryDim: UInt32
    private let fullRopeTheta: Float
    private let attnScale: Float
    private let embedOutScale: Float
    private let batchCapacity: Int

    // Decode (single-row) and batched accept (R <= 6) Metal kernels
    private let embed: EmbedLookupInt4
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let rms: RMSNorm
    private let prefillRMS: PrefillRMSNorm
    private let int4: DequantInt4GEMV
    private let int4MultiX: DequantInt4GEMVMultiX
    private let int8: DequantInt8GEMV
    private let rope: RoPE
    private let attention: Attention
    private let elementwise: Elementwise
    private let moe: MoE
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let sharedRuntime: SharedExpertRuntime
    private let prefillSharedExpert: PrefillSharedExpert
    private let draftHead: LMHeadChainInt4

    // Constant unity scales for Gemma-4 router kernels (BF16 1.0)
    private let onesD: MTLBuffer
    private let onesExperts: MTLBuffer

    // Exposed to RealForwardRunner so prefill/produce can record target post-finalNorm hidden states
    let lastHiddenBuf: MTLBuffer

    // MTP KV cache across accepted trajectory [maxContext, kvDim] FP16
    private let mtpK: MTLBuffer
    private let mtpV: MTLBuffer

    // Single-step draft scratch buffers
    private let dEmb: MTLBuffer
    private let dFcIn: MTLBuffer
    private let dH: MTLBuffer
    private let dXNorm: MTLBuffer
    private let dQPacked: MTLBuffer
    private let dQ: MTLBuffer
    private let dGate: MTLBuffer
    private let dAttnOut: MTLBuffer
    private let dOOut: MTLBuffer
    private let dMlpX: MTLBuffer
    private let dRouteIDs: MTLBuffer
    private let dRouteWeights: MTLBuffer
    private let dSlotOffsets: MTLBuffer
    private let dAllHit: MTLBuffer
    private let dSharedScratchGate: MTLBuffer
    private let dSharedScratchUp: MTLBuffer
    private let dSharedScratchAct: MTLBuffer
    private let dScalarGateBuf: MTLBuffer
    private let dMoeActs: MTLBuffer
    private let dH1: MTLBuffer
    private let dH2: MTLBuffer
    private let dStepHidden: MTLBuffer
    private let dTokenBuf: MTLBuffer
    private let seedHiddenBuf: MTLBuffer

    // Batched accept pass (up to batchCapacity rows) + rewind ring for inside-verify continuations
    private let aTokensBuf: MTLBuffer
    private let recentHiddenBuf: MTLBuffer
    private let aEmb: MTLBuffer
    private let aPrevHid: MTLBuffer
    private let aEmbNorm: MTLBuffer
    private let aHidNorm: MTLBuffer
    private let aFcIn: MTLBuffer
    private let aH: MTLBuffer
    private let aXNorm: MTLBuffer
    private let aQPacked: MTLBuffer
    private let aQ: MTLBuffer
    private let aGate: MTLBuffer
    private let aAttnOut: MTLBuffer
    private let aOOut: MTLBuffer
    private let aMlpX: MTLBuffer
    private let aRouteIDs: MTLBuffer
    private let aRouteWeights: MTLBuffer
    private let aSlotOffsets: MTLBuffer
    private let aAllHit: MTLBuffer
    private let aSharedScratchGate: MTLBuffer
    private let aSharedScratchUp: MTLBuffer
    private let aSharedScratchAct: MTLBuffer
    private let aMoeActs: MTLBuffer
    private let aH1: MTLBuffer
    private let aH2: MTLBuffer

    private struct MutableStats {
        var rounds: Int = 0
        var draftedTokens: Int = 0
        var acceptedTokens: Int = 0
        var emittedTokens: Int = 0
        var rollbacks: Int = 0
        var draftNanos: UInt64 = 0
        var verifyNanos: UInt64 = 0
        var acceptNanos: UInt64 = 0
        var positionTrials = [Int](repeating: 0, count: 8)
        var positionAccepts = [Int](repeating: 0, count: 8)
    }

    private var stats = MutableStats()
    private var pending: [UInt32] = []
    private var basePair: Int = -1
    private var mtpLen: Int = 0
    private var haveSeed: Bool = false
    private var seedToken: Int32 = 0
    private var seedMargin: Float = 0
    private var seedExpectedPosition: Int = -1
    private var seedExpectedBonus: Int32 = -1
    private var recentHiddenBase: Int = -1
    private var recentHiddenCount: Int = 0
    private var pendingAcceptCB: MTLCommandBuffer?
    private var pendingAcceptStartNs: UInt64 = 0
    private var pendingAcceptExpectedPos: Int = -1
    private var pendingAcceptExpectedBonus: Int32 = -1
    private var pendingAcceptRCount: Int = 0
    private var emaPositionAccept: [Float] = [0.85, 0.75, 0.60, 0.45]
    private var slowP0EMA: Float = 0.80
    private var throttleCooldownRemaining: Int = 0
    private var roundsSinceThrottleProbe: Int = 0
    private var consecutiveK1Hits: Int = 0
    private(set) var throttledTokensSkipped: Int = 0

    enum GenerationPhase: Equatable, Sendable {
        case thinking
        case visibleProse
        case structuredCodeOrTool
    }

    private(set) var currentPhase: GenerationPhase = .visibleProse
    private(set) var draftTopK: Int
    private let stepTruncationMinMargin: Float
    private let forceStepTruncation: Bool
    private let pldHybridEnabled: Bool = ProcessInfo.processInfo.environment["MFERENCE_MTP_PLD"] != "0"
    private let useMultiXEnabled: Bool = ProcessInfo.processInfo.environment["MFERENCE_USE_MULTIX"] != "0"
    private var tokenHistory: [Int32] = []
    private var historyBasePosition: Int = 0
    private var insideToolCall: Bool = false
    private var insideCodeFence: Bool = false

    // Qwen 3.6 ChatML special token IDs and fence tokens for automatic phase detection
    private static let qwen36EndOfTextID: Int32 = 248044
    private static let qwen36ImStartID: Int32 = 248045
    private static let qwen36ImEndID: Int32 = 248046
    private static let qwen36ToolCallStartID: Int32 = 248058
    private static let qwen36ToolCallEndID: Int32 = 248059
    private static let qwen36ToolResponseStartID: Int32 = 248066
    private static let qwen36ThinkStartID: Int32 = 248068
    private static let qwen36ThinkEndID: Int32 = 248069
    private static let qwen36CodeFenceID: Int32 = 71093
    private static let qwen36SpacedCodeFenceID: Int32 = 52451

    var draftCount: Int
    var adaptiveDraftEnabled: Bool
    var draftOverride: ((_ round: Int, _ k: Int) -> [Int32])?

    var specStats: Qwen38ForwardRunner.MTPSpecStats {
        try? flushPendingAcceptIfNeeded()
        return Qwen38ForwardRunner.MTPSpecStats(
            rounds: stats.rounds,
            draftedTokens: stats.draftedTokens,
            acceptedTokens: stats.acceptedTokens,
            emittedTokens: stats.emittedTokens,
            rollbacks: stats.rollbacks,
            draftNanos: stats.draftNanos,
            verifyNanos: stats.verifyNanos,
            acceptNanos: stats.acceptNanos,
            positionTrials: stats.positionTrials,
            positionAccepts: stats.positionAccepts
        )
    }

    // MARK: - Probe & Initialization

    static func probe(model: Model,
                      context: MetalContext,
                      config: ArchConfig,
                      maxContext: Int) throws -> Qwen36MTPSpeculator? {
        guard config.family == .qwen36,
              model.manifest.sidecars?["mtp"]?.carried == true else {
            return nil
        }
        return try Qwen36MTPSpeculator(model: model, context: context, config: config, maxContext: maxContext)
    }

    private init(model: Model,
                 context: MetalContext,
                 config: ArchConfig,
                 maxContext: Int) throws {
        self.model = model
        self.cfg = config
        self.queue = context.queue
        self.maxContext = max(8, maxContext)

        let d = config.hiddenSize
        let moeF = config.moeIntermediateSize
        let headDim = config.fullHeadDim > 0 ? config.fullHeadDim : config.headDim
        let numHeads = config.numHeads
        let numKV = config.numFullKVHeads > 0 ? config.numFullKVHeads : config.numKVHeads
        let qDim = numHeads * headDim
        let kvDim = numKV * headDim
        let numExperts = config.numExperts
        let topK = config.topKExperts

        guard d > 0, d % Quantization.groupSize == 0,
              moeF > 0, moeF % Quantization.groupSize == 0,
              headDim > 0, numHeads > 0, numKV > 0,
              numExperts > 0, MoE.routedComputeWidths.contains(UInt32(topK)),
              MoE.routerTopKWidths.contains(UInt32(topK)) else {
            throw Error.invalidWeights("unsupported Qwen 3.6 MTP dimensions (d=\(d), moeF=\(moeF), topK=\(topK))")
        }

        self.d = d
        self.moeF = moeF
        self.headDim = headDim
        self.numHeads = numHeads
        self.numKV = numKV
        self.qDim = qDim
        self.kvDim = kvDim
        self.numExperts = numExperts
        self.topK = topK
        self.batchCapacity = Self.maxDraftSlots + 2

        let loaded = try Self.loadWeights(model: model, config: config, d: d, qDim: qDim,
                                          kvDim: kvDim, headDim: headDim, numExperts: numExperts)
        self.weights = loaded.weights
        self.sharedF = loaded.sharedF

        let aux = try Self.loadAuxiliaryExperts(model: model, device: context.device,
                                                config: config, d: d, moeF: moeF)
        self.expertStreamer = aux.streamer
        self.slab = aux.slab
        self.slotTable = aux.slotTable
        self.expertStride = aux.expertStride
        self.expertOffsets = aux.offsets

        let rotaryFactor = config.partialRotaryFactor > 0 ? config.partialRotaryFactor : 0.25
        self.rotaryDim = UInt32(Double(headDim) * rotaryFactor)
        self.fullRopeTheta = Float(config.fullRopeTheta > 0 ? config.fullRopeTheta : config.ropeTheta)
        self.attnScale = config.attentionScale > 0 ? Float(config.attentionScale) : (1.0 / sqrtf(Float(headDim)))
        self.embedOutScale = config.embeddingScaledBySqrtHidden ? sqrtf(Float(d)) : 1.0

        let env = ProcessInfo.processInfo.environment
        let specKEnv = env["MFERENCE_SPEC_K"] ?? env["MFERENCE_MTP_K"]
        let adaptiveEnv = env["MFERENCE_SPEC_ADAPTIVE"]
        let explicitNumericK = specKEnv.flatMap(Int.init)
        let cacheSlots = model.routedExpertCacheSlotCount(layer: 0)
        let autoLowMemoryAdaptive = (explicitNumericK == nil)
            && (adaptiveEnv != "0")
            && (cacheSlots != nil && cacheSlots! <= 32 && config.numExperts >= 64)
        self.draftCount = min(Self.maxDraftSlots, max(1, explicitNumericK ?? 2))
        self.adaptiveDraftEnabled = (specKEnv?.lowercased() == "auto")
            || (adaptiveEnv == "1")
            || autoLowMemoryAdaptive

        let topKEnv = ProcessInfo.processInfo.environment["MFERENCE_MTP_TOP_K"].flatMap(Int.init)
        let resolvedDraftTopK: Int
        if let reqK = topKEnv, [4, 6, 8].contains(reqK), reqK <= topK {
            resolvedDraftTopK = reqK
        } else if topK >= 6 {
            resolvedDraftTopK = 6
        } else {
            resolvedDraftTopK = topK
        }
        self.draftTopK = resolvedDraftTopK

        let minMarginEnv = ProcessInfo.processInfo.environment["MFERENCE_MTP_MIN_MARGIN"].flatMap(Float.init)
        self.stepTruncationMinMargin = minMarginEnv ?? 1.10
        self.forceStepTruncation = (minMarginEnv != nil)

        // Compile / acquire kernels
        self.embed = try EmbedLookupInt4(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context)
        self.int4MultiX = try DequantInt4GEMVMultiX(context: context)
        self.int8 = try DequantInt8GEMV(context: context)
        self.rope = try RoPE(context: context)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.moe = try MoE(
            context: context,
            siluActivation: true,
            specializedD: UInt32(d),
            specializedF: UInt32(moeF),
            specializedNumExperts: UInt32(numExperts),
            specializedTopK: UInt32(resolvedDraftTopK)
        )
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.sharedRuntime = try SharedExpertRuntime(context: context, weightBits: 4, siluActivation: true)
        self.prefillSharedExpert = try PrefillSharedExpert(context: context, weightBits: 4, siluActivation: true)
        self.draftHead = try LMHeadChainInt4(context: context, maxD: d, maxVocab: config.vocabSize)

        let dev = context.device
        let halfBytes = MemoryLayout<Float16>.stride
        let u32Bytes = MemoryLayout<UInt32>.stride
        let aR = self.batchCapacity
        let sharedF = self.sharedF

        self.onesD = try Self.makeUnityBF16Buffer(device: dev, count: d)
        self.onesExperts = try Self.makeUnityBF16Buffer(device: dev, count: numExperts)

        self.lastHiddenBuf = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.mtpK = try Self.makeBuffer(dev, bytes: self.maxContext * kvDim * halfBytes)
        self.mtpV = try Self.makeBuffer(dev, bytes: self.maxContext * kvDim * halfBytes)

        self.dEmb = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.dFcIn = try Self.makeBuffer(dev, bytes: 2 * d * halfBytes)
        self.dH = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.dXNorm = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.dQPacked = try Self.makeBuffer(dev, bytes: 2 * qDim * halfBytes)
        self.dQ = try Self.makeBuffer(dev, bytes: qDim * halfBytes)
        self.dGate = try Self.makeBuffer(dev, bytes: qDim * halfBytes)
        self.dAttnOut = try Self.makeBuffer(dev, bytes: qDim * halfBytes)
        self.dOOut = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.dMlpX = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.dRouteIDs = try Self.makeBuffer(dev, bytes: topK * u32Bytes)
        self.dRouteWeights = try Self.makeBuffer(dev, bytes: topK * halfBytes)
        self.dSlotOffsets = try Self.makeBuffer(dev, bytes: topK * u32Bytes)
        self.dAllHit = try Self.makeBuffer(dev, bytes: u32Bytes)
        self.dSharedScratchGate = try Self.makeBuffer(dev, bytes: sharedF * halfBytes)
        self.dSharedScratchUp = try Self.makeBuffer(dev, bytes: sharedF * halfBytes)
        self.dSharedScratchAct = try Self.makeBuffer(dev, bytes: sharedF * halfBytes)
        self.dScalarGateBuf = try Self.makeBuffer(dev, bytes: u32Bytes)
        self.dMoeActs = try Self.makeBuffer(dev, bytes: topK * moeF * halfBytes)
        self.dH1 = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.dH2 = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.dStepHidden = try Self.makeBuffer(dev, bytes: d * halfBytes)
        self.dTokenBuf = try Self.makeBuffer(dev, bytes: u32Bytes, shared: true)
        self.seedHiddenBuf = try Self.makeBuffer(dev, bytes: d * halfBytes)

        self.aTokensBuf = try Self.makeBuffer(dev, bytes: aR * u32Bytes, shared: true)
        self.recentHiddenBuf = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aEmb = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aPrevHid = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aEmbNorm = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aHidNorm = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aFcIn = try Self.makeBuffer(dev, bytes: aR * 2 * d * halfBytes)
        self.aH = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aXNorm = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aQPacked = try Self.makeBuffer(dev, bytes: aR * 2 * qDim * halfBytes)
        self.aQ = try Self.makeBuffer(dev, bytes: aR * qDim * halfBytes)
        self.aGate = try Self.makeBuffer(dev, bytes: aR * qDim * halfBytes)
        self.aAttnOut = try Self.makeBuffer(dev, bytes: aR * qDim * halfBytes)
        self.aOOut = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aMlpX = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aRouteIDs = try Self.makeBuffer(dev, bytes: aR * topK * u32Bytes)
        self.aRouteWeights = try Self.makeBuffer(dev, bytes: aR * topK * halfBytes)
        self.aSlotOffsets = try Self.makeBuffer(dev, bytes: aR * topK * u32Bytes)
        self.aAllHit = try Self.makeBuffer(dev, bytes: u32Bytes)
        self.aSharedScratchGate = try Self.makeBuffer(dev, bytes: aR * sharedF * halfBytes)
        self.aSharedScratchUp = try Self.makeBuffer(dev, bytes: aR * sharedF * halfBytes)
        self.aSharedScratchAct = try Self.makeBuffer(dev, bytes: aR * sharedF * halfBytes)
        self.aMoeActs = try Self.makeBuffer(dev, bytes: aR * topK * moeF * halfBytes)
        self.aH1 = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
        self.aH2 = try Self.makeBuffer(dev, bytes: aR * d * halfBytes)
    }

    // MARK: - SpeculativeDrafter Lifecycle

    func reset() {
        try? flushPendingAcceptIfNeeded()
        stats = MutableStats()
        emaPositionAccept = [0.85, 0.75, 0.60, 0.45]
        slowP0EMA = 0.80
        throttleCooldownRemaining = 0
        roundsSinceThrottleProbe = 0
        consecutiveK1Hits = 0
        throttledTokensSkipped = 0
        currentPhase = .visibleProse
        insideToolCall = false
        insideCodeFence = false
        tokenHistory.removeAll(keepingCapacity: true)
        historyBasePosition = 0
        pending.removeAll(keepingCapacity: true)
        resetKVState()
    }

    var capturedTokenHistory: (basePosition: Int, tokens: [Int32]) {
        (historyBasePosition, tokenHistory)
    }

    func restoreTokenHistory(basePosition: Int, tokens: [Int32]) {
        try? flushPendingAcceptIfNeeded()
        resetKVState()
        pending.removeAll(keepingCapacity: true)
        emaPositionAccept = [0.85, 0.75, 0.60, 0.45]
        slowP0EMA = 0.80
        throttleCooldownRemaining = 0
        roundsSinceThrottleProbe = 0
        consecutiveK1Hits = 0
        historyBasePosition = max(0, basePosition)
        tokenHistory = tokens
        recomputePhaseAndFenceState(boostEMA: true)
    }

    /// Boosts or resets the per-position acceptance EMA when transitioning out of
    /// `<think>` prose into visible output or structured code / `<tool_call>` blocks
    /// so draft depth `K` immediately steps up on high-predictability spans.
    func notifyPhaseTransition(_ phase: GenerationPhase) {
        currentPhase = phase
        switch phase {
        case .thinking:
            break
        case .visibleProse:
            throttleCooldownRemaining = 0
            roundsSinceThrottleProbe = 0
            slowP0EMA = max(slowP0EMA, 0.80)
            emaPositionAccept[0] = max(emaPositionAccept[0], 0.90)
            emaPositionAccept[1] = max(emaPositionAccept[1], 0.78)
            emaPositionAccept[2] = max(emaPositionAccept[2], 0.64)
            emaPositionAccept[3] = max(emaPositionAccept[3], 0.48)
        case .structuredCodeOrTool:
            throttleCooldownRemaining = 0
            roundsSinceThrottleProbe = 0
            slowP0EMA = max(slowP0EMA, 0.80)
            emaPositionAccept[0] = max(emaPositionAccept[0], 0.94)
            emaPositionAccept[1] = max(emaPositionAccept[1], 0.86)
            emaPositionAccept[2] = max(emaPositionAccept[2], 0.74)
            emaPositionAccept[3] = max(emaPositionAccept[3], 0.55)
        }
    }

    private func advancePhaseAndFenceState(for tok: Int32, boostEMA: Bool) {
        if tok == Self.qwen36ThinkStartID {
            insideToolCall = false
            insideCodeFence = false
            currentPhase = .thinking
        } else if tok == Self.qwen36ThinkEndID {
            insideToolCall = false
            insideCodeFence = false
            if boostEMA {
                notifyPhaseTransition(.visibleProse)
            } else {
                currentPhase = .visibleProse
            }
        } else if tok == Self.qwen36ToolCallStartID {
            insideToolCall = true
            insideCodeFence = false
            if boostEMA {
                notifyPhaseTransition(.structuredCodeOrTool)
            } else {
                currentPhase = .structuredCodeOrTool
            }
        } else if tok == Self.qwen36ToolCallEndID {
            insideToolCall = false
            insideCodeFence = false
            if boostEMA {
                notifyPhaseTransition(.visibleProse)
            } else {
                currentPhase = .visibleProse
            }
        } else if tok == Self.qwen36ToolResponseStartID
                    || tok == Self.qwen36ImStartID
                    || tok == Self.qwen36ImEndID
                    || tok == Self.qwen36EndOfTextID {
            insideToolCall = false
            insideCodeFence = false
            currentPhase = .visibleProse
        } else if (tok == Self.qwen36CodeFenceID || tok == Self.qwen36SpacedCodeFenceID),
                  !insideToolCall,
                  currentPhase != .thinking {
            insideCodeFence.toggle()
            let nextPhase: GenerationPhase = insideCodeFence ? .structuredCodeOrTool : .visibleProse
            if boostEMA {
                notifyPhaseTransition(nextPhase)
            } else {
                currentPhase = nextPhase
            }
        }
    }

    private func recomputePhaseAndFenceState(boostEMA: Bool) {
        currentPhase = .visibleProse
        insideToolCall = false
        insideCodeFence = false
        for tok in tokenHistory {
            advancePhaseAndFenceState(for: tok, boostEMA: boostEMA)
        }
    }

    func notifyTokens(tokens: ArraySlice<Int32>, startPosition: Int) {
        guard !tokens.isEmpty else { return }
        if tokenHistory.isEmpty && historyBasePosition == 0 && startPosition > 0 {
            historyBasePosition = startPosition
        }
        let relStart = startPosition - historyBasePosition
        if relStart < 0 {
            tokenHistory.removeAll(keepingCapacity: true)
            historyBasePosition = max(0, startPosition)
            tokenHistory.append(contentsOf: tokens)
            recomputePhaseAndFenceState(boostEMA: true)
            return
        }
        if relStart < tokenHistory.count {
            let existingEnd = relStart + tokens.count
            if existingEnd <= tokenHistory.count,
               tokenHistory[relStart..<existingEnd].elementsEqual(tokens) {
                return
            }
            tokenHistory.removeSubrange(relStart..<tokenHistory.count)
            tokenHistory.append(contentsOf: tokens)
            recomputePhaseAndFenceState(boostEMA: true)
            return
        }
        if relStart > tokenHistory.count {
            tokenHistory.removeAll(keepingCapacity: true)
            historyBasePosition = startPosition
            insideToolCall = false
            insideCodeFence = false
            currentPhase = .visibleProse
        }
        tokenHistory.append(contentsOf: tokens)
        for tok in tokens {
            advancePhaseAndFenceState(for: tok, boostEMA: true)
        }
    }

    /// Selects draft depth `K` dynamically based on rolling per-position acceptance EMA,
    /// active generation phase (`.thinking`, `.visibleProse`, `.structuredCodeOrTool`),
    /// and the hardware/memory slot ceiling (`maxSlotSupportedK`).
    func recommendedDraftCount(maxSlotSupportedK: Int = Qwen36MTPSpeculator.maxDraftSlots,
                               lowMemorySlotProfile: Bool = false,
                               allowThrottleToZero: Bool = false) -> Int {
        let cap = max(1, min(Self.maxDraftSlots, maxSlotSupportedK))
        guard adaptiveDraftEnabled else {
            return min(draftCount, cap)
        }
        let p0 = emaPositionAccept[0]
        let p1 = emaPositionAccept[1]
        let p2 = emaPositionAccept[2]

        if lowMemorySlotProfile {
            if allowThrottleToZero {
                if throttleCooldownRemaining > 0 {
                    return 0
                }
                if stats.rounds >= 12 && roundsSinceThrottleProbe >= 8 && slowP0EMA < 0.50 {
                    return 0
                }
            }
            let k: Int
            switch currentPhase {
            case .structuredCodeOrTool:
                k = (cap >= 2 && p0 >= 0.78 && p1 >= 0.56) ? 2 : 1
            case .thinking:
                k = (cap >= 2 && p0 >= 0.84 && p1 >= 0.62) ? 2 : 1
            case .visibleProse:
                k = (cap >= 2 && p0 >= 0.82 && p1 >= 0.62) ? 2 : 1
            }
            return min(k, cap)
        }

        let k: Int
        switch currentPhase {
        case .structuredCodeOrTool:
            if p0 < 0.58 || p1 < 0.34 {
                k = 1
            } else if cap >= 3 && p0 >= 0.92 && p1 >= 0.88 && p2 >= 0.72 {
                k = 3
            } else {
                k = 2
            }
        case .visibleProse, .thinking:
            if p0 < 0.62 || p1 < 0.38 {
                k = 1
            } else if cap >= 3 && p0 >= 0.93 && p1 >= 0.88 && p2 >= 0.74 {
                k = 3
            } else {
                k = 2
            }
        }
        return min(k, cap)
    }

    /// Resolves draft depth `K` and advances any active `K = 0` cooldown window.
    func stepThrottleAndResolveDraftK(maxSlotSupportedK: Int,
                                      lowMemorySlotProfile: Bool,
                                      allowThrottleToZero: Bool) -> Int {
        let k = recommendedDraftCount(maxSlotSupportedK: maxSlotSupportedK,
                                      lowMemorySlotProfile: lowMemorySlotProfile,
                                      allowThrottleToZero: allowThrottleToZero)
        if k == 0 {
            if throttleCooldownRemaining > 0 {
                throttleCooldownRemaining -= 1
                if throttleCooldownRemaining == 0 {
                    slowP0EMA = max(slowP0EMA, 0.58)
                    emaPositionAccept[0] = max(emaPositionAccept[0], 0.64)
                    roundsSinceThrottleProbe = 0
                }
            } else {
                throttleCooldownRemaining = 23
            }
            throttledTokensSkipped += 1
            return 0
        }
        return k
    }

    func prepareForContinuation(expectedPosition: Int) throws {
        try flushPendingAcceptIfNeeded()
        pending.removeAll(keepingCapacity: true)
        throttleCooldownRemaining = 0
        roundsSinceThrottleProbe = 0
        consecutiveK1Hits = 0
        let relExpected = expectedPosition - historyBasePosition
        if relExpected <= 0 {
            tokenHistory.removeAll(keepingCapacity: true)
            historyBasePosition = 0
            insideToolCall = false
            insideCodeFence = false
            currentPhase = .visibleProse
        } else if relExpected < tokenHistory.count {
            tokenHistory.removeSubrange(relExpected..<tokenHistory.count)
            recomputePhaseAndFenceState(boostEMA: true)
        }
        guard expectedPosition > 0 else {
            resetKVState()
            return
        }

        // Preserve contiguous MTP KV prefix when rewinding or continuing inside verified span.
        if basePair != -1 {
            let syncedEnd = basePair + mtpLen
            if expectedPosition < syncedEnd {
                haveSeed = false
                seedExpectedPosition = -1
                seedExpectedBonus = -1

                // If rewound inside recent verify window, restore exact target h_{expectedPosition - 1}
                // so a subsequent produce/draft at expectedPosition has the true hidden state.
                let neededHiddenPos = expectedPosition - 1
                if neededHiddenPos >= recentHiddenBase,
                   neededHiddenPos < recentHiddenBase + recentHiddenCount {
                    try copyRecentHiddenToLastHidden(rowIndex: neededHiddenPos - recentHiddenBase)
                }

                if expectedPosition - 1 >= basePair {
                    mtpLen = (expectedPosition - 1) - basePair
                } else {
                    basePair = -1
                    mtpLen = 0
                }
            } else if expectedPosition > syncedEnd {
                // Gap in token/hidden history (e.g. a new user turn): clear stale KV & seed.
                resetKVState()
            }
            return
        }

        haveSeed = false
        seedExpectedPosition = -1
        seedExpectedBonus = -1
    }

    func consumePending(token: Int32, position: Int) throws -> UInt32? {
        guard !pending.isEmpty else { return nil }
        return pending.removeFirst()
    }

    func canRunRound(position: Int) -> Bool {
        try? flushPendingAcceptIfNeeded()
        guard pending.isEmpty, position > 0 else { return false }
        return position + 2 <= maxContext && mtpLen + 1 < maxContext
    }

    func draft(bonus: Int32, position: Int, count: Int) throws -> [Int32] {
        try flushPendingAcceptIfNeeded()
        let availableTarget = maxContext - position - 1
        let targetK = min(count, availableTarget, Self.maxDraftSlots)
        guard targetK > 0 else { return [] }

        if let override = draftOverride {
            return Array(override(stats.rounds, targetK).prefix(targetK))
        }

        // Invalidate MTP KV if prefill / continuation advanced position leaving a gap.
        if basePair != -1 && basePair + mtpLen != position {
            resetKVState()
        }

        // Reset KV if appending targetK + 1 speculative slots would exceed MTP cache capacity.
        let kvBase: Int
        if basePair == -1 {
            kvBase = 0
        } else if mtpLen + targetK + 2 > maxContext {
            resetKVState()
            kvBase = 0
        } else {
            kvBase = mtpLen
        }

        let startNs = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        defer { stats.draftNanos += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - startNs }

        var drafts: [Int32] = []
        drafts.reserveCapacity(targetK)

        let seedValid = haveSeed && seedExpectedPosition == position && seedExpectedBonus == bonus
        var curTok: Int32
        var curHidden: MTLBuffer
        var nextSlot: Int
        let firstMargin: Float

        if seedValid {
            drafts.append(seedToken)
            curTok = seedToken
            curHidden = seedHiddenBuf
            firstMargin = seedMargin
            // Pair (position - 1) already lives in slot (mtpLen - 1); pair (position) goes to kvBase.
            nextSlot = kvBase
        } else {
            // Pair (position - 1) has not run yet; compute draft[0] directly from (lastHiddenBuf, bonus).
            curTok = try runSingleDraftStep(token: bonus, prevHidden: lastHiddenBuf, slot: kvBase)
            drafts.append(curTok)
            curHidden = dStepHidden
            firstMargin = draftHead.lastTop1Margin
            nextSlot = kvBase + 1
        }

        let historyAligned = pldHybridEnabled
            && tokenHistory.count >= 4
            && position == historyBasePosition + tokenHistory.count - 1
            && tokenHistory.last == bonus

        if targetK >= 2,
           shouldTruncateAfterFirstDraft(
               token: curTok,
               margin: firstMargin,
               pldConfirmsFirst: {
                   historyAligned && pldUniqueContinuation(after: bonus) == curTok
               }) {
            return drafts
        }

        if drafts.count < targetK,
           historyAligned,
           (currentPhase == .structuredCodeOrTool || insideToolCall || insideCodeFence),
           let pldSecond = pldUniqueContinuation(after: bonus, extra: curTok) {
            drafts.append(pldSecond)
            if drafts.count < targetK,
               let pldThird = pldUniqueContinuation(after: bonus, extra: curTok, secondExtra: pldSecond) {
                drafts.append(pldThird)
            }
            return drafts
        }

        while drafts.count < targetK {
            curTok = try runSingleDraftStep(token: curTok, prevHidden: curHidden, slot: nextSlot)
            drafts.append(curTok)
            curHidden = dStepHidden
            nextSlot += 1
        }

        return drafts
    }

    private enum PLDMatchResult {
        case none
        case ambiguous
        case unique(Int32)

        mutating func record(_ candidate: Int32) {
            switch self {
            case .none:
                self = .unique(candidate)
            case .unique(let existing):
                if existing != candidate {
                    self = .ambiguous
                }
            case .ambiguous:
                break
            }
        }
    }

    /// Zero-allocation single-pass prompt/history n-gram matcher (checking 6/5/4/3-grams
    /// simultaneously) that disambiguates repeated 4-grams using 5/6-gram context and
    /// returns the highest-order unique non-delimiter continuation token.
    func pldUniqueContinuation(
        after bonus: Int32,
        extra: Int32? = nil,
        secondExtra: Int32? = nil
    ) -> Int32? {
        guard pldHybridEnabled else { return nil }
        let count = tokenHistory.count
        guard count >= 4 else { return nil }

        let t0: Int32
        let t1: Int32
        let t2: Int32
        let t3: Int32
        let t4: Int32
        let t5: Int32
        let avail: Int

        if let secondExtra, let extra {
            t0 = secondExtra
            t1 = extra
            t2 = bonus
            t3 = tokenHistory[count - 2]
            t4 = tokenHistory[count - 3]
            t5 = tokenHistory[count - 4]
            avail = 6
        } else if let extra {
            t0 = extra
            t1 = bonus
            t2 = tokenHistory[count - 2]
            t3 = tokenHistory[count - 3]
            t4 = tokenHistory[count - 4]
            t5 = count >= 5 ? tokenHistory[count - 5] : 0
            avail = min(6, count + 1)
        } else {
            t0 = bonus
            t1 = tokenHistory[count - 2]
            t2 = tokenHistory[count - 3]
            t3 = tokenHistory[count - 4]
            t4 = count >= 5 ? tokenHistory[count - 5] : 0
            t5 = count >= 6 ? tokenHistory[count - 6] : 0
            avail = min(6, count)
        }

        var match3: PLDMatchResult = .none
        var match4: PLDMatchResult = .none
        var match5: PLDMatchResult = .none
        var match6: PLDMatchResult = .none

        for nextPos in stride(from: count - 1, through: 3, by: -1) {
            guard tokenHistory[nextPos - 1] == t0,
                  tokenHistory[nextPos - 2] == t1,
                  tokenHistory[nextPos - 3] == t2 else {
                continue
            }
            let cand = tokenHistory[nextPos]
            match3.record(cand)
            guard nextPos >= 4, tokenHistory[nextPos - 4] == t3 else {
                continue
            }
            match4.record(cand)
            guard avail >= 5, nextPos >= 5, tokenHistory[nextPos - 5] == t4 else {
                continue
            }
            match5.record(cand)
            if avail >= 6, nextPos >= 6, tokenHistory[nextPos - 6] == t5 {
                match6.record(cand)
            }
        }

        if case .unique(let tok) = match6 { return tok < Self.qwen36EndOfTextID ? tok : nil }
        if case .ambiguous = match6 { return nil }
        if case .unique(let tok) = match5 { return tok < Self.qwen36EndOfTextID ? tok : nil }
        if case .ambiguous = match5 { return nil }
        if case .unique(let tok) = match4 { return tok < Self.qwen36EndOfTextID ? tok : nil }
        if case .ambiguous = match4 { return nil }
        if case .unique(let tok) = match3 { return tok < Self.qwen36EndOfTextID ? tok : nil }
        return nil
    }

    /// EVICT-style step-wise draft truncation: skips the 2nd sequential MTP draft
    /// pass (returning `K=1`) when `draft[0]` lands on a phase/turn boundary token
    /// or has a low top-1 vs top-2 logit margin, unless PLD uniquely confirms
    /// `draft[0]` from active prompt/history context.
    private func shouldTruncateAfterFirstDraft(
        token: Int32,
        margin: Float,
        pldConfirmsFirst: () -> Bool = { false }
    ) -> Bool {
        guard adaptiveDraftEnabled || forceStepTruncation else { return false }
        guard stepTruncationMinMargin > 0 else { return false }
        if token == Self.qwen36ThinkStartID
            || token == Self.qwen36ThinkEndID
            || token == Self.qwen36ToolCallStartID
            || token == Self.qwen36ToolCallEndID
            || token >= Self.qwen36EndOfTextID {
            return true
        }
        let threshold: Float
        switch currentPhase {
        case .structuredCodeOrTool:
            threshold = min(0.25, stepTruncationMinMargin * 0.45)
        case .thinking, .visibleProse:
            threshold = stepTruncationMinMargin
        }
        if margin >= threshold {
            return false
        }
        return !pldConfirmsFirst()
    }

    func recordAccepted(drafts: [Int32], emitted: [UInt32], position: Int, verifyNanos: UInt64) {
        guard !emitted.isEmpty else { return }

        var accepted = 0
        while accepted < drafts.count,
              accepted < emitted.count - 1,
              emitted[accepted] == UInt32(bitPattern: drafts[accepted]) {
            accepted += 1
        }

        if emitted.count > 1 {
            pending = Array(emitted.dropFirst())
        } else {
            pending.removeAll(keepingCapacity: true)
        }

        stats.rounds += 1
        roundsSinceThrottleProbe += 1
        stats.draftedTokens += drafts.count
        stats.acceptedTokens += accepted
        stats.emittedTokens += emitted.count
        if accepted < drafts.count {
            stats.rollbacks += 1
        }
        stats.verifyNanos += verifyNanos

        let hit0: Float = (accepted > 0) ? 1.0 : 0.0
        slowP0EMA = 0.94 * slowP0EMA + 0.06 * hit0

        let alpha: Float = 0.20
        for i in 0..<drafts.count {
            if i < stats.positionTrials.count {
                stats.positionTrials[i] += 1
            }
            if i < accepted, i < stats.positionAccepts.count {
                stats.positionAccepts[i] += 1
            }
            if i < emaPositionAccept.count {
                let hit: Float = (i < accepted) ? 1.0 : 0.0
                emaPositionAccept[i] = (1.0 - alpha) * emaPositionAccept[i] + alpha * hit
            }
        }
        if drafts.count == 1 {
            if accepted == 1 {
                consecutiveK1Hits += 1
                if emaPositionAccept[0] >= 0.84 && consecutiveK1Hits >= 3 {
                    emaPositionAccept[1] = min(0.70, emaPositionAccept[1] + 0.03)
                }
            } else {
                consecutiveK1Hits = 0
            }
        } else {
            consecutiveK1Hits = 0
            if accepted == drafts.count, drafts.count < emaPositionAccept.count {
                let nextIdx = drafts.count
                emaPositionAccept[nextIdx] = min(0.92, emaPositionAccept[nextIdx] + 0.10)
            }
        }
    }

    // MARK: - Accept Sync Pass

    func flushPendingAcceptIfNeeded() throws {
        guard let cb = pendingAcceptCB else { return }
        pendingAcceptCB = nil
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            pendingAcceptRCount = 0
            pendingAcceptExpectedPos = -1
            pendingAcceptExpectedBonus = -1
            throw Error.commandFailed(cb.error?.localizedDescription ?? "Qwen 3.6 MTP accept command buffer did not complete")
        }
        stats.acceptNanos += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - pendingAcceptStartNs
        let rawSeedTok = dTokenBuf.contents().load(as: UInt32.self)
        seedToken = Int32(bitPattern: rawSeedTok)
        seedMargin = draftHead.lastTop1Margin
        haveSeed = true
        seedExpectedPosition = pendingAcceptExpectedPos
        seedExpectedBonus = pendingAcceptExpectedBonus
        mtpLen += pendingAcceptRCount
        pendingAcceptRCount = 0
    }

    /// Synchronizes all committed target `(hidden, nextToken)` pairs into the MTP KV cache
    /// and precomputes the next round's first draft token (`seedToken`) and post-norm state
    /// (`seedHiddenBuf`) asynchronously in a single GPU command buffer.
    func commitAcceptPass(bonus: Int32,
                          emitted: [UInt32],
                          position: Int,
                          targetNormedHidden: MTLBuffer,
                          acceptedCount accepted: Int) throws {
        try flushPendingAcceptIfNeeded()
        let startNs = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

        let targetEndPair = position + accepted
        var firstMissing: Int
        if basePair == -1 {
            basePair = position - 1
            mtpLen = 0
            firstMissing = position - 1
        } else {
            let syncedEnd = basePair + mtpLen
            if syncedEnd == position || syncedEnd == position - 1 {
                firstMissing = syncedEnd
            } else {
                // Gap or rewind mismatch: rebuild cleanly from (position - 1).
                basePair = position - 1
                mtpLen = 0
                firstMissing = position - 1
            }
        }

        var missingCount = (targetEndPair - firstMissing) + 1
        guard missingCount > 0 else { return }

        if mtpLen + missingCount > maxContext {
            basePair = position - 1
            mtpLen = 0
            firstMissing = position - 1
            missingCount = (targetEndPair - firstMissing) + 1
        }

        let startSlot = mtpLen
        let rCount = missingCount
        precondition(rCount <= batchCapacity,
                     "Qwen36MTPSpeculator accept batch \(rCount) exceeds capacity \(batchCapacity)")

        let tokensPtr = aTokensBuf.contents().bindMemory(to: UInt32.self, capacity: batchCapacity)
        let dBytes = d * MemoryLayout<Float16>.stride

        // Populate aTokensBuf and prepare blits into aPrevHid, recentHiddenBuf, and lastHiddenBuf.
        for r in 0..<rCount {
            let pair = firstMissing + r
            if pair == position - 1 {
                tokensPtr[r] = UInt32(bitPattern: bonus)
            } else {
                let emittedIdx = pair - position
                tokensPtr[r] = emitted[emittedIdx]
            }
        }

        guard let cb = queue.makeCommandBuffer() else {
            throw Error.commandFailed("unable to create Qwen 3.6 MTP accept command buffer")
        }

        // Snapshot [h_{position-1}, h_position, ..., h_{position+accepted}] for rewind recovery
        // before overwriting lastHiddenBuf with h_{position+accepted}.
        guard let prepBlit = cb.makeBlitCommandEncoder() else {
            throw Error.commandFailed("unable to create Qwen 3.6 MTP accept blit encoder")
        }
        prepBlit.copy(from: lastHiddenBuf, sourceOffset: 0,
                      to: recentHiddenBuf, destinationOffset: 0, size: dBytes)
        prepBlit.copy(from: targetNormedHidden, sourceOffset: 0,
                      to: recentHiddenBuf, destinationOffset: dBytes,
                      size: (accepted + 1) * dBytes)

        for r in 0..<rCount {
            let pair = firstMissing + r
            if pair == position - 1 {
                prepBlit.copy(from: lastHiddenBuf, sourceOffset: 0,
                              to: aPrevHid, destinationOffset: r * dBytes, size: dBytes)
            } else {
                let srcRow = pair - position
                prepBlit.copy(from: targetNormedHidden, sourceOffset: srcRow * dBytes,
                              to: aPrevHid, destinationOffset: r * dBytes, size: dBytes)
            }
        }
        prepBlit.copy(from: targetNormedHidden, sourceOffset: accepted * dBytes,
                      to: lastHiddenBuf, destinationOffset: 0, size: dBytes)
        prepBlit.endEncoding()

        recentHiddenBase = position - 1
        recentHiddenCount = accepted + 2

        try encodeAcceptPass(cb, rowCount: rCount, startSlot: startSlot)
        cb.commit()
        pendingAcceptCB = cb
        pendingAcceptStartNs = startNs
        pendingAcceptExpectedPos = position + accepted + 1
        pendingAcceptExpectedBonus = Int32(bitPattern: emitted[accepted])
        pendingAcceptRCount = rCount
    }

    // MARK: - Single-Step Draft & Batched Accept Encoders

    private func runSingleDraftStep(token: Int32, prevHidden: MTLBuffer, slot: Int) throws -> Int32 {
        guard let cb = queue.makeCommandBuffer() else {
            throw Error.commandFailed("unable to create Qwen 3.6 MTP draft command buffer")
        }

        let halfBytes = MemoryLayout<Float16>.stride
        let dBytes = d * halfBytes
        let kvBytes = kvDim * halfBytes
        let emb = model.embedding

        embed.encode(commandBuffer: cb,
                     table: emb.buffer, tableOffset: Int(emb.offset),
                     scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                     biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                     out: dEmb,
                     tokenId: UInt32(bitPattern: token),
                     d: UInt32(d),
                     outScale: embedOutScale)

        rms.encodeBF16W(commandBuffer: cb,
                        x: dEmb,
                        weight: weights.preFcNormEmbedding.buffer,
                        weightOffset: Int(weights.preFcNormEmbedding.offset),
                        out: dFcIn, outOffset: 0,
                        d: UInt32(d), eps: Self.epsilon)
        rms.encodeBF16W(commandBuffer: cb,
                        x: prevHidden,
                        weight: weights.preFcNormHidden.buffer,
                        weightOffset: Int(weights.preFcNormHidden.offset),
                        out: dFcIn, outOffset: dBytes,
                        d: UInt32(d), eps: Self.epsilon)

        encodeInt4(cb, weights: weights.fc, x: dFcIn, y: dH, rows: d, cols: 2 * d)

        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.inputNorm.buffer,
                        weightOffset: Int(weights.inputNorm.offset),
                        out: dXNorm, d: UInt32(d), eps: Self.epsilon)

        let slotOffset = slot * kvBytes
        encodeInt4(cb, weights: weights.qProj, x: dXNorm, y: dQPacked, rows: 2 * qDim, cols: d)
        encodeInt4(cb, weights: weights.kProj, x: dXNorm, y: mtpK, yOffset: slotOffset, rows: kvDim, cols: d)
        encodeInt4(cb, weights: weights.vProj, x: dXNorm, y: mtpV, yOffset: slotOffset, rows: kvDim, cols: d)

        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: dQPacked, q: dQ, gate: dGate,
                                     heads: numHeads, dim: headDim, rows: 1)

        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: dQ,
                               weight: weights.qNorm.buffer, weightOffset: Int(weights.qNorm.offset),
                               out: dQ,
                               headDim: UInt32(headDim), numHeads: numHeads, eps: Self.epsilon)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: mtpK, xOffset: slotOffset,
                               weight: weights.kNorm.buffer, weightOffset: Int(weights.kNorm.offset),
                               out: mtpK, outOffset: slotOffset,
                               headDim: UInt32(headDim), numHeads: numKV, eps: Self.epsilon)

        rope.encodeNeoxSubdim(commandBuffer: cb, data: dQ,
                              position: UInt32(slot),
                              headDim: UInt32(headDim), numHeads: UInt32(numHeads),
                              rotaryDim: rotaryDim, theta: fullRopeTheta)
        rope.encodeNeoxSubdim(commandBuffer: cb, data: mtpK, dataOffset: slotOffset,
                              position: UInt32(slot),
                              headDim: UInt32(headDim), numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim, theta: fullRopeTheta)

        attention.encodeFull(commandBuffer: cb,
                             q: dQ, k: mtpK, v: mtpV, out: dAttnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(numHeads), numKVHeads: UInt32(numKV),
                             seqLen: UInt32(slot + 1), scale: attnScale)
        elementwise.encodeSigmoidGateMul(commandBuffer: cb, out: dAttnOut, gate: dGate, count: qDim)
        encodeInt4(cb, weights: weights.oProj, x: dAttnOut, y: dOOut, rows: d, cols: qDim)
        elementwise.encodeResidualAdd(commandBuffer: cb, hidden: dH, delta: dOOut, count: d)

        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.postAttnNorm.buffer,
                        weightOffset: Int(weights.postAttnNorm.offset),
                        out: dMlpX, d: UInt32(d), eps: Self.epsilon)

        moe.encodeRouterGemma4(commandBuffer: cb,
                               weights: weights.router.buffer, weightsOffset: Int(weights.router.offset),
                               scales: weights.router.buffer, scalesOffset: Int(weights.router.scaleOffset),
                               biases: weights.router.buffer, biasesOffset: Int(weights.router.biasOffset),
                               hidden: dMlpX,
                               effectiveScale: onesD,
                               perExpertScale: onesExperts,
                               outIndices: dRouteIDs,
                               outWeights: dRouteWeights,
                               numExperts: UInt32(numExperts),
                               d: UInt32(d),
                               topK: UInt32(draftTopK))
        moe.encodeSlotLookup(commandBuffer: cb,
                             indices: dRouteIDs,
                             table: slotTable,
                             slotStride: expertStride,
                             slotOffsets: dSlotOffsets,
                             allHit: dAllHit,
                             numExperts: UInt32(numExperts),
                             topK: UInt32(draftTopK))

        try sharedRuntime.encode(commandBuffer: cb,
                                 x: dMlpX,
                                 gate: weights.sharedGateProj,
                                 up: weights.sharedUpProj,
                                 down: weights.sharedDownProj,
                                 y: dH1,
                                 scratchGate: dSharedScratchGate,
                                 scratchUp: dSharedScratchUp,
                                 scratchAct: dSharedScratchAct)
        int8.encode(commandBuffer: cb,
                    weights: weights.sharedScalarGate.buffer,
                    weightsOffset: Int(weights.sharedScalarGate.offset),
                    scales: weights.sharedScalarGate.buffer,
                    scalesOffset: Int(weights.sharedScalarGate.scaleOffset),
                    biases: weights.sharedScalarGate.buffer,
                    biasesOffset: Int(weights.sharedScalarGate.biasOffset),
                    x: dMlpX, y: dScalarGateBuf,
                    m: 1, n: UInt32(d))
        elementwise.encodeSigmoidScalarMul(commandBuffer: cb, y: dH1, gate: dScalarGateBuf, count: d)

        moe.encodeSlotMapGuardedFFN(commandBuffer: cb,
                                    slab: slab,
                                    slotOffsets: dSlotOffsets,
                                    allHit: dAllHit,
                                    routedOffsets: expertOffsets,
                                    x: dMlpX,
                                    acts: dMoeActs,
                                    routingWeights: dRouteWeights,
                                    residual: dH1,
                                    y: dH2,
                                    hidden: dH,
                                    d: UInt32(d),
                                    f: UInt32(moeF),
                                    topK: UInt32(draftTopK))

        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.mtpNorm.buffer, weightOffset: Int(weights.mtpNorm.offset),
                        out: dStepHidden, d: UInt32(d), eps: Self.epsilon)
        draftHead.encodeGreedyFromNormed(commandBuffer: cb,
                                         normedHidden: dStepHidden,
                                         normedOffset: 0,
                                         weights: weights.lmHead.buffer,
                                         weightsOffset: Int(weights.lmHead.offset),
                                         scales: weights.lmHead.buffer,
                                         scalesOffset: Int(weights.lmHead.scaleOffset),
                                         biases: weights.lmHead.buffer,
                                         biasesOffset: Int(weights.lmHead.biasOffset),
                                         outToken: dTokenBuf,
                                         d: UInt32(d),
                                         vocab: UInt32(cfg.vocabSize))

        try finish(cb)
        let rawToken = dTokenBuf.contents().load(as: UInt32.self)
        return Int32(bitPattern: rawToken)
    }

    private func encodeAcceptPass(_ cb: MTLCommandBuffer, rowCount rCount: Int, startSlot: Int) throws {
        let halfBytes = MemoryLayout<Float16>.stride
        let dBytes = d * halfBytes
        let kvBytes = kvDim * halfBytes
        let emb = model.embedding

        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer, tableOffset: Int(emb.offset),
                            scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                            tokens: aTokensBuf, out: aEmb,
                            t: UInt32(rCount), d: UInt32(d), outScale: embedOutScale)

        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: aEmb,
                               weight: weights.preFcNormEmbedding.buffer,
                               weightOffset: Int(weights.preFcNormEmbedding.offset),
                               out: aEmbNorm, t: UInt32(rCount), d: UInt32(d), eps: Self.epsilon)
        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: aPrevHid,
                               weight: weights.preFcNormHidden.buffer,
                               weightOffset: Int(weights.preFcNormHidden.offset),
                               out: aHidNorm, t: UInt32(rCount), d: UInt32(d), eps: Self.epsilon)

        guard let fcBlit = cb.makeBlitCommandEncoder() else {
            throw Error.commandFailed("unable to create Qwen 3.6 MTP fc-pack blit")
        }
        for r in 0..<rCount {
            fcBlit.copy(from: aEmbNorm, sourceOffset: r * dBytes,
                        to: aFcIn, destinationOffset: r * 2 * dBytes, size: dBytes)
            fcBlit.copy(from: aHidNorm, sourceOffset: r * dBytes,
                        to: aFcIn, destinationOffset: (r * 2 + 1) * dBytes, size: dBytes)
        }
        fcBlit.endEncoding()

        encodeInt4Batch(cb, weights: weights.fc, x: aFcIn, y: aH, rows: d, cols: 2 * d, tokens: rCount)

        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: aH,
                               weight: weights.inputNorm.buffer,
                               weightOffset: Int(weights.inputNorm.offset),
                               out: aXNorm, t: UInt32(rCount), d: UInt32(d), eps: Self.epsilon)

        let kvStartOffset = startSlot * kvBytes
        encodeInt4Batch(cb, weights: weights.qProj, x: aXNorm, y: aQPacked, rows: 2 * qDim, cols: d, tokens: rCount)
        encodeInt4Batch(cb, weights: weights.kProj, x: aXNorm, y: mtpK, yOffset: kvStartOffset, rows: kvDim, cols: d, tokens: rCount)
        encodeInt4Batch(cb, weights: weights.vProj, x: aXNorm, y: mtpV, yOffset: kvStartOffset, rows: kvDim, cols: d, tokens: rCount)

        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: aQPacked, q: aQ, gate: aGate,
                                     heads: numHeads, dim: headDim, rows: rCount)

        prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
            commandBuffer: cb,
            q: aQ, qOffset: 0,
            k: mtpK, kOffset: kvStartOffset,
            qWeight: weights.qNorm.buffer, qWeightOffset: Int(weights.qNorm.offset),
            kWeight: weights.kNorm.buffer, kWeightOffset: Int(weights.kNorm.offset),
            startPosition: UInt32(startSlot),
            queryCount: UInt32(rCount),
            headDim: UInt32(headDim),
            numQHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKV),
            qTokenStrideElements: UInt32(qDim),
            kvTokenStrideElements: UInt32(kvDim),
            theta: fullRopeTheta,
            rotaryDim: rotaryDim,
            eps: Self.epsilon)

        for r in 0..<rCount {
            let rowOff = r * qDim * halfBytes
            attention.encodeFull(commandBuffer: cb,
                                 q: aQ, qOffset: rowOff,
                                 k: mtpK, v: mtpV,
                                 out: aAttnOut, outOffset: rowOff,
                                 headDim: UInt32(headDim),
                                 numQHeads: UInt32(numHeads), numKVHeads: UInt32(numKV),
                                 seqLen: UInt32(startSlot + r + 1), scale: attnScale)
        }

        elementwise.encodeSigmoidGateMul(commandBuffer: cb, out: aAttnOut, gate: aGate, count: rCount * qDim)
        encodeInt4Batch(cb, weights: weights.oProj, x: aAttnOut, y: aOOut, rows: d, cols: qDim, tokens: rCount)
        elementwise.encodeResidualAdd(commandBuffer: cb, hidden: aH, delta: aOOut, count: rCount * d)

        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: aH,
                               weight: weights.postAttnNorm.buffer,
                               weightOffset: Int(weights.postAttnNorm.offset),
                               out: aMlpX, t: UInt32(rCount), d: UInt32(d), eps: Self.epsilon)

        let u32Bytes = MemoryLayout<UInt32>.stride
        for r in 0..<rCount {
            moe.encodeRouterGemma4(commandBuffer: cb,
                                   weights: weights.router.buffer,
                                   weightsOffset: Int(weights.router.offset),
                                   scales: weights.router.buffer,
                                   scalesOffset: Int(weights.router.scaleOffset),
                                   biases: weights.router.buffer,
                                   biasesOffset: Int(weights.router.biasOffset),
                                   hidden: aMlpX,
                                   hiddenOffset: r * dBytes,
                                   effectiveScale: onesD,
                                   perExpertScale: onesExperts,
                                   outIndices: aRouteIDs,
                                   outIndicesOffset: r * draftTopK * u32Bytes,
                                   outWeights: aRouteWeights,
                                   outWeightsOffset: r * draftTopK * halfBytes,
                                   numExperts: UInt32(numExperts),
                                   d: UInt32(d),
                                   topK: UInt32(draftTopK))
        }

        // Note: `router_slot_lookup_k8_batch` writes one summary status to `aAllHit[0]`.
        moe.encodeSlotLookupBatch(commandBuffer: cb,
                                  indices: aRouteIDs,
                                  table: slotTable,
                                  slotStride: expertStride,
                                  slotOffsets: aSlotOffsets,
                                  allHit: aAllHit,
                                  numExperts: UInt32(numExperts),
                                  topK: UInt32(draftTopK),
                                  queryCount: UInt32(rCount))

        _ = try prefillSharedExpert.encodeBlock(commandBuffer: cb,
                                                x: aMlpX, y: aH1,
                                                gate: weights.sharedGateProj,
                                                up: weights.sharedUpProj,
                                                down: weights.sharedDownProj,
                                                scratchGate: aSharedScratchGate,
                                                scratchUp: aSharedScratchUp,
                                                scratchAct: aSharedScratchAct,
                                                queryCount: rCount,
                                                d: d,
                                                intermediate: sharedF,
                                                xStrideElements: d,
                                                yStrideElements: d)
        try prefillSharedExpert.encodeQwenScalarGate(commandBuffer: cb,
                                                     x: aMlpX,
                                                     gate: weights.sharedScalarGateProj,
                                                     y: aH1,
                                                     queryCount: rCount,
                                                     d: d,
                                                     xStrideElements: d,
                                                     yStrideElements: d)

        moe.encodeSlotMapGuardedFFNBatch(commandBuffer: cb,
                                         slab: slab,
                                         slotOffsets: aSlotOffsets,
                                         allHit: aAllHit,
                                         allHitOffset: 0,
                                         routedOffsets: expertOffsets,
                                         x: aMlpX,
                                         acts: aMoeActs,
                                         routingWeights: aRouteWeights,
                                         residual: aH1,
                                         y: aH2,
                                         hidden: aH,
                                         d: UInt32(d),
                                         f: UInt32(moeF),
                                         topK: UInt32(draftTopK),
                                         queryCount: rCount)

        let lastRowOffset = (rCount - 1) * dBytes
        rms.encodeBF16W(commandBuffer: cb,
                        x: aH, xOffset: lastRowOffset,
                        weight: weights.mtpNorm.buffer, weightOffset: Int(weights.mtpNorm.offset),
                        out: seedHiddenBuf, d: UInt32(d), eps: Self.epsilon)
        draftHead.encodeGreedyFromNormed(commandBuffer: cb,
                                         normedHidden: seedHiddenBuf,
                                         normedOffset: 0,
                                         weights: weights.lmHead.buffer,
                                         weightsOffset: Int(weights.lmHead.offset),
                                         scales: weights.lmHead.buffer,
                                         scalesOffset: Int(weights.lmHead.scaleOffset),
                                         biases: weights.lmHead.buffer,
                                         biasesOffset: Int(weights.lmHead.biasOffset),
                                         outToken: dTokenBuf,
                                         d: UInt32(d),
                                         vocab: UInt32(cfg.vocabSize))
    }

    // MARK: - Kernel Encoding Helpers

    private func encodeInt4(_ cb: MTLCommandBuffer,
                            weights: TensorView,
                            x: MTLBuffer, xOffset: Int = 0,
                            y: MTLBuffer, yOffset: Int = 0,
                            rows: Int, cols: Int) {
        int4.encode(commandBuffer: cb,
                    weights: weights.buffer, weightsOffset: Int(weights.offset),
                    scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                    x: x, xOffset: xOffset,
                    y: y, yOffset: yOffset,
                    m: UInt32(rows), n: UInt32(cols))
    }

    private func encodeInt4Batch(_ cb: MTLCommandBuffer,
                                 weights: TensorView,
                                 x: MTLBuffer,
                                 y: MTLBuffer, yOffset: Int = 0,
                                 rows: Int, cols: Int,
                                 tokens: Int) {
        let useMultiX = useMultiXEnabled
            && tokens <= DequantInt4GEMVMultiX.maxTokens
            && Int(weights.offset).isMultiple(of: 2)
        if useMultiX {
            int4MultiX.encode(commandBuffer: cb,
                              weights: weights.buffer, weightsOffset: Int(weights.offset),
                              scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                              biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                              x: x, y: y, yOffset: yOffset,
                              m: rows, n: cols, tokens: tokens)
        } else {
            let xStride = cols * MemoryLayout<Float16>.stride
            let yStride = rows * MemoryLayout<Float16>.stride
            for r in 0..<tokens {
                encodeInt4(cb, weights: weights,
                           x: x, xOffset: r * xStride,
                           y: y, yOffset: yOffset + r * yStride,
                           rows: rows, cols: cols)
            }
        }
    }

    private func copyRecentHiddenToLastHidden(rowIndex: Int) throws {
        try flushPendingAcceptIfNeeded()
        guard let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            throw Error.commandFailed("unable to create Qwen 3.6 MTP continuation blit")
        }
        let dBytes = d * MemoryLayout<Float16>.stride
        blit.copy(from: recentHiddenBuf, sourceOffset: rowIndex * dBytes,
                  to: lastHiddenBuf, destinationOffset: 0, size: dBytes)
        blit.endEncoding()
        try finish(cb)
    }

    private func resetKVState() {
        basePair = -1
        mtpLen = 0
        haveSeed = false
        seedExpectedPosition = -1
        seedExpectedBonus = -1
        recentHiddenBase = -1
        recentHiddenCount = 0
        pendingAcceptRCount = 0
        pendingAcceptExpectedPos = -1
        pendingAcceptExpectedBonus = -1
    }

    private func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw Error.commandFailed(cb.error?.localizedDescription ?? "command buffer did not complete")
        }
    }

    // MARK: - Weight & Auxiliary Pool Loading

    private static func loadWeights(model: Model,
                                    config: ArchConfig,
                                    d: Int, qDim: Int, kvDim: Int, headDim: Int,
                                    numExperts: Int) throws -> (weights: Weights, sharedF: Int) {
        func requireQuant(_ spec: QuantMatrixSpec) throws -> TensorView {
            let fullName = "mtp.\(spec.suffix)"
            let v = try model.resident(name: fullName)
            let expectedBytes = UInt64(spec.rows * spec.cols * spec.bits / 8)
            let expectedAux = UInt64(spec.rows * (spec.cols / Quantization.groupSize) * 2)
            guard v.dtype == 0,
                  v.shape.0 == UInt32(spec.rows), v.shape.1 == UInt32(spec.cols),
                  v.length == expectedBytes,
                  v.scaleLength == expectedAux, v.biasLength == expectedAux else {
                throw Error.invalidWeights(
                    "\(fullName) mismatch: expected INT\(spec.bits) [\(spec.rows), \(spec.cols)], got dtype=\(v.dtype) len=\(v.length) shape=(\(v.shape.0), \(v.shape.1))")
            }
            return v
        }

        func requireNorm(_ spec: NormSpec) throws -> TensorView {
            let fullName = "mtp.\(spec.suffix)"
            let v = try model.resident(name: fullName)
            guard v.dtype == 1,
                  v.shape.0 == UInt32(spec.rows),
                  v.length == UInt64(spec.rows * 2),
                  v.scaleLength == 0, v.biasLength == 0 else {
                throw Error.invalidWeights(
                    "\(fullName) norm mismatch: expected BF16 [\(spec.rows)], got dtype=\(v.dtype) len=\(v.length)")
            }
            return v
        }

        let sharedGateRaw = try model.resident(name: "mtp.layers.0.mlp.shared_expert.gate_proj.weight")
        let sharedF = Int(sharedGateRaw.shape.0)
        guard sharedF > 0, sharedF % Quantization.groupSize == 0 else {
            throw Error.invalidWeights("invalid shared_expert intermediate size \(sharedF)")
        }

        let fc = try requireQuant(QuantMatrixSpec(suffix: "fc.weight", rows: d, cols: 2 * d, bits: 4))
        let preFcNormEmbedding = try requireNorm(NormSpec(suffix: "pre_fc_norm_embedding.weight", rows: d))
        let preFcNormHidden = try requireNorm(NormSpec(suffix: "pre_fc_norm_hidden.weight", rows: d))
        let mtpNorm = try requireNorm(NormSpec(suffix: "norm.weight", rows: d))
        let inputNorm = try requireNorm(NormSpec(suffix: "layers.0.input_layernorm.weight", rows: d))
        let postAttnNorm = try requireNorm(NormSpec(suffix: "layers.0.post_attention_layernorm.weight", rows: d))
        let qProj = try requireQuant(QuantMatrixSpec(suffix: "layers.0.self_attn.q_proj.weight", rows: 2 * qDim, cols: d, bits: 4))
        let kProj = try requireQuant(QuantMatrixSpec(suffix: "layers.0.self_attn.k_proj.weight", rows: kvDim, cols: d, bits: 4))
        let vProj = try requireQuant(QuantMatrixSpec(suffix: "layers.0.self_attn.v_proj.weight", rows: kvDim, cols: d, bits: 4))
        let oProj = try requireQuant(QuantMatrixSpec(suffix: "layers.0.self_attn.o_proj.weight", rows: d, cols: qDim, bits: 4))
        let qNorm = try requireNorm(NormSpec(suffix: "layers.0.self_attn.q_norm.weight", rows: headDim))
        let kNorm = try requireNorm(NormSpec(suffix: "layers.0.self_attn.k_norm.weight", rows: headDim))
        let router = try requireQuant(QuantMatrixSpec(suffix: "layers.0.mlp.gate.weight", rows: numExperts, cols: d, bits: 8))
        let sharedScalarGate = try requireQuant(QuantMatrixSpec(suffix: "layers.0.mlp.shared_expert_gate.weight", rows: 1, cols: d, bits: 8))
        let sharedGate = try requireQuant(QuantMatrixSpec(suffix: "layers.0.mlp.shared_expert.gate_proj.weight", rows: sharedF, cols: d, bits: 4))
        let sharedUp = try requireQuant(QuantMatrixSpec(suffix: "layers.0.mlp.shared_expert.up_proj.weight", rows: sharedF, cols: d, bits: 4))
        let sharedDown = try requireQuant(QuantMatrixSpec(suffix: "layers.0.mlp.shared_expert.down_proj.weight", rows: d, cols: sharedF, bits: 4))

        func makeProj(_ v: TensorView) -> SharedExpertProjection {
            SharedExpertProjection(
                weights: v.buffer, scales: v.buffer, biases: v.buffer,
                weightsOffset: Int(v.offset), scalesOffset: Int(v.scaleOffset), biasesOffset: Int(v.biasOffset),
                rows: v.shape.0, cols: v.shape.1)
        }

        let lm = model.lmHead
        guard lm.scaleLength > 0, lm.biasLength > 0 else {
            throw Error.invalidWeights("target lmHead lacks INT4 scales/biases")
        }

        let weights = Weights(
            fc: fc,
            preFcNormEmbedding: preFcNormEmbedding,
            preFcNormHidden: preFcNormHidden,
            mtpNorm: mtpNorm,
            inputNorm: inputNorm,
            postAttnNorm: postAttnNorm,
            qProj: qProj,
            kProj: kProj,
            vProj: vProj,
            oProj: oProj,
            qNorm: qNorm,
            kNorm: kNorm,
            router: router,
            sharedScalarGate: sharedScalarGate,
            sharedGate: sharedGate,
            sharedUp: sharedUp,
            sharedDown: sharedDown,
            lmHead: lm,
            sharedGateProj: makeProj(sharedGate),
            sharedUpProj: makeProj(sharedUp),
            sharedDownProj: makeProj(sharedDown),
            sharedScalarGateProj: makeProj(sharedScalarGate)
        )
        return (weights, sharedF)
    }

    private static func loadAuxiliaryExperts(model: Model,
                                             device: MTLDevice,
                                             config: ArchConfig,
                                             d: Int,
                                             moeF: Int)
        throws -> (streamer: ResidentExpertStreamer,
                   slab: MTLBuffer,
                   slotTable: MTLBuffer,
                   expertStride: Int,
                   offsets: MoEExpertOffsets) {
        let pools = model.manifest.auxiliaryExpertPools?.filter { $0.name == "mtp" } ?? []
        guard pools.count == 1, let pool = pools.first else {
            throw Error.invalidWeights("missing auxiliaryExpertPool named 'mtp'")
        }
        guard pool.expertsPerLayer == config.numExperts,
              pool.layers.count == 1,
              pool.layers[0].layer == 0 else {
            throw Error.invalidWeights("invalid 'mtp' auxiliary expert pool geometry")
        }

        let groupsD = d / Quantization.groupSize
        let groupsF = moeF / Quantization.groupSize
        let gateWOff = 0
        let gateSOff = moeF * d / 2
        let gateBOff = gateSOff + moeF * groupsD * 2
        let gateBytes = gateBOff + moeF * groupsD * 2
        let upWOff = gateBytes
        let upSOff = upWOff + moeF * d / 2
        let upBOff = upSOff + moeF * groupsD * 2
        let downWOff = 2 * gateBytes
        let downSOff = downWOff + d * moeF / 2
        let downBOff = downSOff + d * groupsF * 2
        let rawBlobBytes = UInt64(downBOff + d * groupsF * 2)

        guard pool.expertStride >= rawBlobBytes else {
            throw Error.invalidWeights("mtp expertStride \(pool.expertStride) < required \(rawBlobBytes)")
        }

        let rawFile = pool.layers[0].file
        let relative = rawFile.contains("/") ? rawFile : "\(pool.directory)/\(rawFile)"
        guard let fileEntry = model.manifest.files[relative] else {
            throw ModelError.missingFile(name: relative)
        }
        let expectedBytes = UInt64(pool.expertsPerLayer) * pool.expertStride
        guard fileEntry.size == expectedBytes else {
            throw ModelError.tensorSizeMismatch(name: relative, expected: expectedBytes, actual: fileEntry.size)
        }

        let url = model.directoryURL.appendingPathComponent(relative)
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        let actualSize = (attrs[.size] as? NSNumber)?.uint64Value ?? 0
        guard actualSize == expectedBytes else {
            throw ModelError.tensorSizeMismatch(name: relative, expected: expectedBytes, actual: actualSize)
        }
        if model.integrityPolicy == .fullSha256 {
            try Sha256Verifier.verifyFile(at: url, named: relative, expectedHex: fileEntry.sha256)
        }

        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: expectedBytes,
            expertsPerLayer: pool.expertsPerLayer,
            expertStride: pool.expertStride
        )
        let streamer = try ResidentExpertStreamer(layout: layout, device: device, strategy: .mapped, slabOnly: true)
        guard let binding = streamer.contiguousSlabBinding else {
            throw Error.invalidWeights("mtp auxiliary streamer did not expose contiguousSlabBinding")
        }

        let offsets = MoEExpertOffsets(
            gateWOff: UInt32(gateWOff), gateSOff: UInt32(gateSOff), gateBOff: UInt32(gateBOff),
            upWOff: UInt32(upWOff), upSOff: UInt32(upSOff), upBOff: UInt32(upBOff),
            downWOff: UInt32(downWOff), downSOff: UInt32(downSOff), downBOff: UInt32(downBOff)
        )
        return (streamer, binding.slab, binding.table, binding.expertStride, offsets)
    }

    private static func makeBuffer(_ device: MTLDevice, bytes: Int, shared: Bool = false) throws -> MTLBuffer {
        let opts: MTLResourceOptions = shared ? .storageModeShared : .storageModePrivate
        guard let buf = device.makeBuffer(length: max(bytes, 4), options: opts) else {
            throw MetalError.noDevice
        }
        return buf
    }

    private static func makeUnityBF16Buffer(device: MTLDevice, count: Int) throws -> MTLBuffer {
        let buf = try makeBuffer(device, bytes: count * MemoryLayout<UInt16>.stride, shared: true)
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count {
            ptr[i] = 0x3F80 // Float(1.0) in BF16
        }
        return buf
    }
}
