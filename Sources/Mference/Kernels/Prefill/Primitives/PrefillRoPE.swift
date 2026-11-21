import Foundation
import Metal

final class PrefillRoPE {
    private let psoDefaultNeox: MTLComputePipelineState
    private let psoProportionalNeox: MTLComputePipelineState
    private let psoNeoxSubdim: MTLComputePipelineState
    private let psoNeoxSubdimMRoPE: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoDefaultNeox = try context.pipeline("prefill_rope_default_neox_block")
        self.psoProportionalNeox = try context.pipeline("prefill_rope_proportional_neox_block")
        self.psoNeoxSubdim = try context.pipeline("prefill_rope_neox_subdim_block")
        self.psoNeoxSubdimMRoPE = try context.pipeline("prefill_rope_neox_subdim_mrope_block")
    }

    /// Qwen 3.5 / 3.6 interleaved M-RoPE variant of `encodeNeoxSubdim`.
    /// `mropeTriples` stores `(pos_t, pos_h, pos_w)` `UInt32` triples for `queryCount` tokens.
    func encodeNeoxSubdimMRoPE(commandBuffer: MTLCommandBuffer,
                               data: MTLBuffer,
                               dataOffset: Int = 0,
                               mropeTriples: MTLBuffer,
                               mropeTriplesOffset: Int = 0,
                               queryCount: UInt32,
                               headDim: UInt32,
                               numHeads: UInt32,
                               rotaryDim: UInt32,
                               tokenStrideElements: UInt32,
                               theta: Float,
                               mropeLimits: SIMD2<UInt32> = SIMD2<UInt32>(33, 30)) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(rotaryDim % 2 == 0, "rotaryDim must be even")
        precondition(rotaryDim <= headDim, "rotaryDim must not exceed headDim")
        precondition(tokenStrideElements >= numHeads * headDim,
                     "token stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoNeoxSubdimMRoPE)
        enc.setBuffer(data, offset: dataOffset, index: 0)
        enc.setBuffer(mropeTriples, offset: mropeTriplesOffset, index: 1)
        var hd = headDim
        var heads = numHeads
        var stride = tokenStrideElements
        var thetaVar = theta
        var rotary = rotaryDim
        var limits = mropeLimits
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&thetaVar, length: MemoryLayout<Float>.size, index: 5)
        enc.setBytes(&rotary, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&limits, length: MemoryLayout<SIMD2<UInt32>>.size, index: 7)
        let pairs = Int(rotaryDim / 2)
        enc.dispatchThreads(
            MTLSize(width: pairs, height: Int(numHeads), depth: Int(queryCount)),
            threadsPerThreadgroup: MTLSize(width: min(pairs, psoNeoxSubdimMRoPE.maxTotalThreadsPerThreadgroup),
                                           height: 1,
                                           depth: 1))
        enc.endEncoding()
    }

    /// Qwen-style partial RoPE over a chunk: rotation confined to the first
    /// `rotaryDim` elements of each head, pairing (i, rotaryDim/2 + i),
    /// frequency divisor = rotaryDim.
    func encodeNeoxSubdim(commandBuffer: MTLCommandBuffer,
                          data: MTLBuffer,
                          dataOffset: Int = 0,
                          startPosition: UInt32,
                          queryCount: UInt32,
                          headDim: UInt32,
                          numHeads: UInt32,
                          rotaryDim: UInt32,
                          tokenStrideElements: UInt32,
                          theta: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(rotaryDim % 2 == 0, "rotaryDim must be even")
        precondition(rotaryDim <= headDim, "rotaryDim must not exceed headDim")
        precondition(tokenStrideElements >= numHeads * headDim,
                     "token stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoNeoxSubdim)
        enc.setBuffer(data, offset: dataOffset, index: 0)
        var start = startPosition
        var hd = headDim
        var heads = numHeads
        var stride = tokenStrideElements
        var thetaVar = theta
        var rotary = rotaryDim
        enc.setBytes(&start, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&thetaVar, length: MemoryLayout<Float>.size, index: 5)
        enc.setBytes(&rotary, length: MemoryLayout<UInt32>.size, index: 6)

        let pairs = Int(rotaryDim) / 2
        enc.dispatchThreads(
            MTLSize(width: pairs, height: Int(numHeads), depth: Int(queryCount)),
            threadsPerThreadgroup: MTLSize(width: min(pairs, psoNeoxSubdim.maxTotalThreadsPerThreadgroup),
                                           height: 1,
                                           depth: 1))
        enc.endEncoding()
    }

    func encodeDefaultNeox(commandBuffer: MTLCommandBuffer,
                                  data: MTLBuffer,
                                  dataOffset: Int = 0,
                                  startPosition: UInt32,
                                  queryCount: UInt32,
                                  headDim: UInt32,
                                  numHeads: UInt32,
                                  tokenStrideElements: UInt32,
                                  theta: Float = 10_000.0) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(tokenStrideElements >= numHeads * headDim,
                     "token stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoDefaultNeox)
        enc.setBuffer(data, offset: dataOffset, index: 0)
        var start = startPosition
        var hd = headDim
        var heads = numHeads
        var stride = tokenStrideElements
        var thetaVar = theta
        enc.setBytes(&start, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&thetaVar, length: MemoryLayout<Float>.size, index: 5)

        let pairs = Int(headDim) / 2
        enc.dispatchThreads(
            MTLSize(width: pairs, height: Int(numHeads), depth: Int(queryCount)),
            threadsPerThreadgroup: MTLSize(width: min(pairs, psoDefaultNeox.maxTotalThreadsPerThreadgroup),
                                           height: 1,
                                           depth: 1))
        enc.endEncoding()
    }

    func encodeProportionalNeox(commandBuffer: MTLCommandBuffer,
                                       data: MTLBuffer,
                                       dataOffset: Int = 0,
                                       startPosition: UInt32,
                                       queryCount: UInt32,
                                       headDim: UInt32,
                                       numHeads: UInt32,
                                       rotatedPairs: UInt32,
                                       tokenStrideElements: UInt32,
                                       theta: Float = 1_000_000.0) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(rotatedPairs * 2 <= headDim,
                     "rotatedPairs * 2 must not exceed headDim")
        precondition(tokenStrideElements >= numHeads * headDim,
                     "token stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoProportionalNeox)
        enc.setBuffer(data, offset: dataOffset, index: 0)
        var start = startPosition
        var hd = headDim
        var heads = numHeads
        var stride = tokenStrideElements
        var thetaVar = theta
        var rp = rotatedPairs
        enc.setBytes(&start, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&thetaVar, length: MemoryLayout<Float>.size, index: 5)
        enc.setBytes(&rp, length: MemoryLayout<UInt32>.size, index: 6)

        enc.dispatchThreads(
            MTLSize(width: Int(rotatedPairs), height: Int(numHeads), depth: Int(queryCount)),
            threadsPerThreadgroup: MTLSize(width: min(Int(rotatedPairs),
                                                      psoProportionalNeox.maxTotalThreadsPerThreadgroup),
                                           height: 1,
                                           depth: 1))
        enc.endEncoding()
    }
}
