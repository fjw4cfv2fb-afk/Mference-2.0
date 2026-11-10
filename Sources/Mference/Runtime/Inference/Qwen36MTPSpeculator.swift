import Foundation
import Metal

public enum Qwen36MTPError: Error {
    case invalidConfiguration(String)
    case commandFailed(String)
    case invalidInput(String)
}

/// Native MTP (multi-token-prediction) speculator for Qwen 3.6 35B A3B MoE.
///
/// Uses the checkpoint's native MTP draft layer:
///   fc(concat(norm_e(embed(tok)), norm_h(hidden)))
/// followed by 1 transformer block (full attention + SwiGLU MLP)
/// sharing the target's embedding table and lm_head.
final class Qwen36MTPSpeculator {

    struct Stats {
        var rounds = 0
        var draftedTokens = 0
        var acceptedTokens = 0
        var emittedTokens = 0
        var rollbacks = 0
        var draftNanos: UInt64 = 0
        var verifyNanos: UInt64 = 0
        var acceptNanos: UInt64 = 0
    }

    /// Resolved MTP tensor views (`mtp.*` names in resident index).
    private struct Weights {
        let fc: TensorView
        let preFcNormEmbedding: TensorView
        let preFcNormHidden: TensorView
        let finalNorm: TensorView
        let inputNorm: TensorView
        let postAttnNorm: TensorView
        let q: TensorView
        let k: TensorView
        let v: TensorView
        let o: TensorView
        let qNorm: TensorView
        let kNorm: TensorView
        let mlpGate: SharedExpertProjection
        let mlpUp: SharedExpertProjection
        let mlpDown: SharedExpertProjection
    }

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig
    private let maxContext: Int
    private let weights: Weights

    /// Number of draft tokens per round (default 1).
    let draftCapacity: Int
    var draftCount: Int {
        didSet {
            precondition(draftCount >= 1 && draftCount <= draftCapacity,
                         "draftCount outside 1...\(draftCapacity)")
        }
    }

    // Kernels (cached via MetalContext)
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let elementwise: Elementwise
    private let rope: RoPE
    private let mtpMLP: SharedExpertRuntime
    private let draftHead: LMHeadChainInt4

    // Drafter KV cache
    private let mtpKVDim: Int
    private let mtpK: MTLBuffer
    private let mtpV: MTLBuffer
    private var mtpLen = 0
    private var basePair = 0
    private var started = false
    private var haveSeed = false
    private var seedToken: Int32 = 0
    private let seedHiddenBuf: MTLBuffer
    let lastHiddenBuf: MTLBuffer

    // Draft scratch (single token)
    private let dEmb: MTLBuffer
    private let dFcIn: MTLBuffer
    private let dH: MTLBuffer
    private let dNormed: MTLBuffer
    private let dQPacked: MTLBuffer
    private let dQ: MTLBuffer
    private let dGate: MTLBuffer
    private let dAttnOut: MTLBuffer
    private let dOut1: MTLBuffer
    private let dMlpX: MTLBuffer
    private let dMlpGate: MTLBuffer
    private let dMlpUp: MTLBuffer
    private let dMlpAct: MTLBuffer
    private let dMlpOut: MTLBuffer
    private let dHOutA: MTLBuffer
    private let dHOutB: MTLBuffer
    private let dTokenBuf: MTLBuffer

    // Pending verified tokens: (expectInput, emit)
    private var pending: [(expectInput: Int32, emit: UInt32)] = []
    private var pendingPosition = 0

    private(set) var stats = Stats()
    private static let epsilon: Float = 1e-6

    /// Probe the resident index for MTP tensors; returns nil when absent.
    static func probe(model: Model,
                      context: MetalContext,
                      config: ArchConfig,
                      maxContext: Int) throws -> Qwen36MTPSpeculator? {
        guard (try? model.resident(name: "mtp.fc.weight")) != nil else { return nil }
        return try Qwen36MTPSpeculator(model: model, context: context, config: config, maxContext: maxContext)
    }

    private init(model: Model,
                 context: MetalContext,
                 config: ArchConfig,
                 maxContext: Int) throws {
        self.model = model
        self.ctx = context
        self.cfg = config
        self.maxContext = maxContext

        let D = config.hiddenSize
        let F = config.moeIntermediateSize > 0 ? config.moeIntermediateSize : config.intermediateSize
        let headDim = config.fullHeadDim > 0 ? config.fullHeadDim : config.headDim
        let numKV = config.numFullKVHeads > 0 ? config.numFullKVHeads : config.numKVHeads
        let qDim = config.numHeads * headDim
        let kvDim = numKV * headDim
        self.mtpKVDim = kvDim

        func view(_ name: String) throws -> TensorView {
            try model.resident(name: "mtp.\(name)")
        }
        func projection(_ view: TensorView, rows: Int, cols: Int) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                   scales: view.buffer,
                                   biases: view.buffer,
                                   weightsOffset: Int(view.offset),
                                   scalesOffset: Int(view.scaleOffset),
                                   biasesOffset: Int(view.biasOffset),
                                   rows: UInt32(rows),
                                   cols: UInt32(cols))
        }
        self.weights = Weights(
            fc: try view("fc.weight"),
            preFcNormEmbedding: try view("pre_fc_norm_embedding.weight"),
            preFcNormHidden: try view("pre_fc_norm_hidden.weight"),
            finalNorm: try view("norm.weight"),
            inputNorm: try view("layers.0.input_layernorm.weight"),
            postAttnNorm: try view("layers.0.post_attention_layernorm.weight"),
            q: try view("layers.0.self_attn.q_proj.weight"),
            k: try view("layers.0.self_attn.k_proj.weight"),
            v: try view("layers.0.self_attn.v_proj.weight"),
            o: try view("layers.0.self_attn.o_proj.weight"),
            qNorm: try view("layers.0.self_attn.q_norm.weight"),
            kNorm: try view("layers.0.self_attn.k_norm.weight"),
            mlpGate: projection(try view("layers.0.mlp.gate_proj.weight"), rows: F, cols: D),
            mlpUp: projection(try view("layers.0.mlp.up_proj.weight"), rows: F, cols: D),
            mlpDown: projection(try view("layers.0.mlp.down_proj.weight"), rows: D, cols: F))

        let requestedK = ProcessInfo.processInfo.environment["MFERENCE_MTP_K"].flatMap(Int.init)
        let capacity = min(max(requestedK ?? 1, 1), 4)
        self.draftCapacity = capacity
        self.draftCount = capacity

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.rope = try RoPE(context: context)
        self.mtpMLP = try SharedExpertRuntime(context: context, weightBits: 4,
                                              siluActivation: config.hiddenActivation == "silu",
                                              specializedD: D, specializedF: F)
        self.draftHead = try LMHeadChainInt4(context: context, maxD: D,
                                             maxVocab: config.vocabSize)

        let device = context.device
        func buf(_ elements: Int, _ label: String,
                 stride: Int = MemoryLayout<Float16>.stride,
                 options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(elements, 1) * stride,
                                               options: options) else {
                throw Qwen36MTPError.invalidConfiguration(
                    "unable to allocate Qwen 3.6 MTP scratch (\(label))")
            }
            made.label = "qwen36.mtp.\(label)"
            return made
        }

        self.mtpK = try buf(maxContext * kvDim, "mtp.K")
        self.mtpV = try buf(maxContext * kvDim, "mtp.V")
        self.seedHiddenBuf = try buf(D, "mtp.seedHidden")
        self.lastHiddenBuf = try buf(D, "mtp.lastHidden")

        self.dEmb = try buf(D, "d.emb")
        self.dFcIn = try buf(2 * D, "d.fcIn")
        self.dH = try buf(D, "d.h")
        self.dNormed = try buf(D, "d.normed")
        self.dQPacked = try buf(2 * qDim, "d.qPacked")
        self.dQ = try buf(qDim, "d.q")
        self.dGate = try buf(qDim, "d.gate")
        self.dAttnOut = try buf(qDim, "d.attnOut")
        self.dOut1 = try buf(D, "d.out1")
        self.dMlpX = try buf(D, "d.mlpX")
        self.dMlpGate = try buf(F, "d.mlpGate")
        self.dMlpUp = try buf(F, "d.mlpUp")
        self.dMlpAct = try buf(F, "d.mlpAct")
        self.dMlpOut = try buf(D, "d.mlpOut")
        self.dHOutA = try buf(D, "d.hOutA")
        self.dHOutB = try buf(D, "d.hOutB")
        self.dTokenBuf = try buf(1, "d.token", stride: MemoryLayout<UInt32>.stride)
    }

    // MARK: - State Management

    func reset() {
        pending.removeAll(keepingCapacity: true)
        pendingPosition = 0
        started = false
        haveSeed = false
        mtpLen = 0
        basePair = 0
    }

    var hasPending: Bool { !pending.isEmpty }

    func prepareForContinuation(expectedPosition: Int) throws {
        pending.removeAll(keepingCapacity: true)
        started = false
        haveSeed = false
        pendingPosition = 0
    }

    func consumePending(token: Int32, position: Int) throws -> UInt32? {
        guard !pending.isEmpty else { return nil }
        let next = pending.removeFirst()
        guard position == pendingPosition, token == next.expectInput else {
            pending.removeAll(keepingCapacity: true)
            return nil
        }
        pendingPosition += 1
        stats.emittedTokens += 1
        return next.emit
    }

    func canRunRound(position: Int) -> Bool {
        return position >= 1 && position + 2 <= maxContext
    }

    // MARK: - Drafting

    func draft(bonus: Int32, position: Int, count: Int = 1) throws -> [Int32] {
        let P = position
        let kRound = min(count, maxContext - P - 1)
        guard kRound >= 1 else { return [] }

        if !started || basePair + mtpLen < P - 1 {
            started = true
            basePair = P - 1
            mtpLen = 0
            haveSeed = false
        }

        let draftStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        stats.rounds += 1
        var drafts: [Int32] = []
        var slot = mtpLen
        var tok: Int32
        var hPrev: MTLBuffer
        if haveSeed {
            drafts.append(seedToken)
            tok = seedToken
            hPrev = seedHiddenBuf
        } else {
            tok = bonus
            hPrev = lastHiddenBuf
        }
        var useA = true
        while drafts.count < kRound {
            let hOut = useA ? dHOutA : dHOutB
            let next = try runDraftForward(token: tok, hPrev: hPrev, hOut: hOut, slot: slot)
            slot += 1
            tok = next
            hPrev = hOut
            useA.toggle()
            drafts.append(next)
        }
        stats.draftNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - draftStart
        stats.draftedTokens += drafts.count
        return drafts
    }

    private func runDraftForward(token: Int32, hPrev: MTLBuffer,
                                 hOut: MTLBuffer, slot: Int) throws -> Int32 {
        precondition(slot >= 0 && slot < maxContext, "MTP slot out of range")
        let D = UInt32(cfg.hiddenSize)
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Qwen36MTPError.commandFailed("unable to create MTP command buffer")
        }

        let emb = model.embedding
        embedInt4.encode(commandBuffer: cb,
                         table: emb.buffer, tableOffset: Int(emb.offset),
                         scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                         biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                         out: dEmb,
                         tokenId: UInt32(bitPattern: token),
                         d: D, outScale: 1.0)
        rms.encodeBF16W(commandBuffer: cb,
                        x: dEmb,
                        weight: weights.preFcNormEmbedding.buffer,
                        weightOffset: Int(weights.preFcNormEmbedding.offset),
                        out: dFcIn,
                        d: D, eps: Self.epsilon)
        rms.encodeBF16W(commandBuffer: cb,
                        x: hPrev,
                        weight: weights.preFcNormHidden.buffer,
                        weightOffset: Int(weights.preFcNormHidden.offset),
                        out: dFcIn, outOffset: cfg.hiddenSize * MemoryLayout<Float16>.stride,
                        d: D, eps: Self.epsilon)
        int4.encode(commandBuffer: cb,
                    weights: weights.fc.buffer, weightsOffset: Int(weights.fc.offset),
                    scales: weights.fc.buffer, scalesOffset: Int(weights.fc.scaleOffset),
                    biases: weights.fc.buffer, biasesOffset: Int(weights.fc.biasOffset),
                    x: dFcIn, y: dH,
                    m: D, n: 2 * D)
        try encodeMTPLayerSingle(cb, slot: slot)
        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.finalNorm.buffer,
                        weightOffset: Int(weights.finalNorm.offset),
                        out: hOut,
                        d: D, eps: Self.epsilon)
        let lm = model.lmHead
        draftHead.encodeGreedyDecode(commandBuffer: cb,
                                     hidden: dH,
                                     normWeight: weights.finalNorm.buffer,
                                     normOffset: Int(weights.finalNorm.offset),
                                     weights: lm.buffer, weightsOffset: Int(lm.offset),
                                     scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                     biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                     outToken: dTokenBuf,
                                     d: D, vocab: UInt32(cfg.vocabSize),
                                     rmsEps: Self.epsilon)
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw Qwen36MTPError.commandFailed("MTP draft pass failed: \(String(describing: cb.error))")
        }
        return Int32(bitPattern: dTokenBuf.contents().load(as: UInt32.self))
    }

    private func encodeMTPLayerSingle(_ cb: MTLCommandBuffer, slot: Int) throws {
        let D = UInt32(cfg.hiddenSize)
        let headDim = cfg.fullHeadDim > 0 ? cfg.fullHeadDim : cfg.headDim
        let numKV = cfg.numFullKVHeads > 0 ? cfg.numFullKVHeads : cfg.numKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
        let h = MemoryLayout<Float16>.stride
        let slotOffset = slot * mtpKVDim * h

        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.inputNorm.buffer,
                        weightOffset: Int(weights.inputNorm.offset),
                        out: dNormed,
                        d: D, eps: Self.epsilon)
        func gemv(_ w: TensorView, x: MTLBuffer, xOffset: Int = 0,
                  y: MTLBuffer, yOffset: Int = 0, m: UInt32, n: UInt32) {
            int4.encode(commandBuffer: cb,
                        weights: w.buffer, weightsOffset: Int(w.offset),
                        scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                        biases: w.buffer, biasesOffset: Int(w.biasOffset),
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
        }
        gemv(weights.q, x: dNormed, y: dQPacked, m: 2 * qDim, n: D)
        gemv(weights.k, x: dNormed, y: mtpK, yOffset: slotOffset,
             m: UInt32(mtpKVDim), n: D)
        gemv(weights.v, x: dNormed, y: mtpV, yOffset: slotOffset,
             m: UInt32(mtpKVDim), n: D)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: dQPacked, q: dQ, gate: dGate,
                                     heads: cfg.numHeads, dim: headDim)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: dQ,
                               weight: weights.qNorm.buffer,
                               weightOffset: Int(weights.qNorm.offset),
                               out: dQ,
                               headDim: UInt32(headDim), numHeads: cfg.numHeads,
                               eps: Self.epsilon)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: mtpK, xOffset: slotOffset,
                               weight: weights.kNorm.buffer,
                               weightOffset: Int(weights.kNorm.offset),
                               out: mtpK, outOffset: slotOffset,
                               headDim: UInt32(headDim), numHeads: numKV,
                               eps: Self.epsilon)
        let ropePosition = UInt32(basePair + slot)
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: dQ,
                              position: ropePosition,
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: mtpK, dataOffset: slotOffset,
                              position: ropePosition,
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        attention.encodeFull(commandBuffer: cb,
                             q: dQ,
                             k: mtpK, v: mtpV,
                             out: dAttnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(numKV),
                             seqLen: UInt32(slot + 1),
                             scale: Float(cfg.attentionScale))
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: dAttnOut, gate: dGate,
                                         count: Int(qDim))
        gemv(weights.o, x: dAttnOut, y: dOut1, m: D, n: qDim)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: dH, delta: dOut1,
                                      count: cfg.hiddenSize)
        rms.encodeBF16W(commandBuffer: cb,
                        x: dH,
                        weight: weights.postAttnNorm.buffer,
                        weightOffset: Int(weights.postAttnNorm.offset),
                        out: dMlpX,
                        d: D, eps: Self.epsilon)
        try mtpMLP.encode(commandBuffer: cb,
                          x: dMlpX,
                          gate: weights.mlpGate, up: weights.mlpUp,
                          down: weights.mlpDown,
                          y: dMlpOut,
                          scratchGate: dMlpGate,
                          scratchUp: dMlpUp,
                          scratchAct: dMlpAct)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: dH, delta: dMlpOut,
                                      count: cfg.hiddenSize)
    }

    // MARK: - Acceptance Recording

    func recordAccepted(drafts: [Int32], emitted: [UInt32], position: Int) {
        let accepted = emitted.count - 1
        stats.acceptedTokens += accepted
        stats.rounds += 1
        if accepted < drafts.count {
            stats.rollbacks += 1
        }
        pending = (1..<emitted.count).map {
            (expectInput: Int32(bitPattern: emitted[$0 - 1]), emit: emitted[$0])
        }
        pendingPosition = position + 1
        stats.emittedTokens += 1
        mtpLen += (accepted + 1)
    }
}
