import Metal

/// Final BF16 RMSNorm, INT4 affine lm-head projection, and greedy argmax.
/// The hot path writes one token ID without materializing vocab-sized logits.
final class LMHeadChainInt4 {
    static let rowsPerThreadgroup = 8

    private static let rowSummaryStride = 3

    /// Shape this instance compiles a specialized head pipeline for. Taken
    /// from the loaded model so both Gemma 4 (2816/262144) and Qwen 3.6
    /// (2048/248320) get constant-folded loop bounds.
    private let specializedD: UInt32
    private let specializedVocab: UInt32

    private let rms: RMSNorm
    private let rowGreedy: MTLComputePipelineState
    private let rowGreedySpecialized: MTLComputePipelineState
    private let rowReducer: MTLComputePipelineState
    private let xNormedBuffer: MTLBuffer
    private let rowSummariesBuffer: MTLBuffer
    private let maxD: Int
    private let maxVocab: Int

    init(context: MetalContext,
         maxD: Int = 2816,
         maxVocab: Int = 262144) throws {
        self.rms = try RMSNorm(context: context)
        self.rowGreedy = try context.pipeline("lm_head_greedy_int4_rows_chunk_raw")
        self.specializedD = UInt32(maxD)
        self.specializedVocab = UInt32(maxVocab)
        self.rowGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_rows_chunk_raw",
            constants: [
                MetalFunctionConstant(index: 10, value: .uint32(UInt32(maxD))),
                MetalFunctionConstant(index: 11, value: .uint32(UInt32(maxVocab))),
                MetalFunctionConstant(index: 13, value: .bool(true)),
            ])
        self.rowReducer = try context.pipeline("lm_head_greedy_int4_rows_reduce")
        self.maxD = maxD
        self.maxVocab = maxVocab

        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let xLength = max(maxD, 1) * MemoryLayout<Float16>.size
        let summaryLength = rowGroups * Self.rowSummaryStride * MemoryLayout<Float>.size
        guard let xNormedBuffer = context.device.makeBuffer(
                  length: xLength,
                  options: .storageModePrivate),
              let rowSummariesBuffer = context.device.makeBuffer(
                  length: summaryLength,
                  options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.xNormedBuffer = xNormedBuffer
        self.rowSummariesBuffer = rowSummariesBuffer
    }

    func encodeGreedyDecode(commandBuffer: MTLCommandBuffer,
                            hidden: MTLBuffer,
                            hiddenOffset: Int = 0,
                            normWeight: MTLBuffer,
                            normOffset: Int = 0,
                            weights: MTLBuffer,
                            weightsOffset: Int = 0,
                            scales: MTLBuffer,
                            scalesOffset: Int = 0,
                            biases: MTLBuffer,
                            biasesOffset: Int = 0,
                            outToken: MTLBuffer,
                            d: UInt32,
                            vocab: UInt32,
                            rmsEps: Float = 1e-6) {
        precondition(Int(d) <= maxD, "d=\(d) exceeds wrapper maxD=\(maxD)")
        precondition(Int(vocab) <= maxVocab,
                     "vocab=\(vocab) exceeds wrapper maxVocab=\(maxVocab)")
        precondition(Int(d) % Quantization.groupSize == 0,
                     "d must be a multiple of \(Quantization.groupSize)")
        precondition(hiddenOffset >= 0, "hiddenOffset must be non-negative")
        precondition(weightsOffset % 2 == 0,
                     "lm_head_greedy_int4_rows_chunk_raw needs a 2-aligned weightsOffset")

        rms.encodeBF16W(commandBuffer: commandBuffer,
                        x: hidden,
                        xOffset: hiddenOffset,
                        weight: normWeight,
                        weightOffset: normOffset,
                        out: xNormedBuffer,
                        d: d,
                        eps: rmsEps)

        encodeGreedyFromNormed(commandBuffer: commandBuffer,
                               normedHidden: xNormedBuffer,
                               normedOffset: 0,
                               weights: weights,
                               weightsOffset: weightsOffset,
                               scales: scales,
                               scalesOffset: scalesOffset,
                               biases: biases,
                               biasesOffset: biasesOffset,
                               outToken: outToken,
                               d: d,
                               vocab: vocab)
    }

    /// Runs the INT4 greedy LM-head projection and argmax reduction directly on
    /// an already-RMSNorm'd hidden state, avoiding a duplicate RMSNorm pass when
    /// the caller has already computed the normalized vector (e.g. MTP draft/seed).
    func encodeGreedyFromNormed(commandBuffer: MTLCommandBuffer,
                                normedHidden: MTLBuffer,
                                normedOffset: Int = 0,
                                weights: MTLBuffer,
                                weightsOffset: Int = 0,
                                scales: MTLBuffer,
                                scalesOffset: Int = 0,
                                biases: MTLBuffer,
                                biasesOffset: Int = 0,
                                outToken: MTLBuffer,
                                d: UInt32,
                                vocab: UInt32) {
        precondition(Int(d) <= maxD, "d=\(d) exceeds wrapper maxD=\(maxD)")
        precondition(Int(vocab) <= maxVocab,
                     "vocab=\(vocab) exceeds wrapper maxVocab=\(maxVocab)")
        precondition(Int(d) % Quantization.groupSize == 0,
                     "d must be a multiple of \(Quantization.groupSize)")
        precondition(normedOffset >= 0, "normedOffset must be non-negative")
        precondition(weightsOffset % 2 == 0,
                     "lm_head_greedy_int4_rows_chunk_raw needs a 2-aligned weightsOffset")

        let rowGroups = (Int(vocab) + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let specialized = d == specializedD && vocab == specializedVocab
            encoder.setComputePipelineState(specialized ? rowGreedySpecialized : rowGreedy)
            encoder.setBuffer(normedHidden, offset: normedOffset, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 4)
            var dValue = d
            var vocabValue = vocab
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)

            let threadgroupSize = MTLSize(
                width: 32 * Self.rowsPerThreadgroup,
                height: 1,
                depth: 1)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroupSize)

            encoder.memoryBarrier(scope: .buffers)

            encoder.setComputePipelineState(rowReducer)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outToken, offset: 0, index: 1)
            var rowGroupCount = UInt32(rowGroups)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)

            let reduceTGSize = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreads(reduceTGSize, threadsPerThreadgroup: reduceTGSize)
            encoder.endEncoding()
        }
    }

    /// Top-1 minus top-2 logit margin from the most recently completed greedy
    /// reduction (`lm_head_greedy_int4_rows_reduce` writes this into `summaries[0]`).
    var lastTop1Margin: Float {
        rowSummariesBuffer.contents().assumingMemoryBound(to: Float.self)[0]
    }
}
