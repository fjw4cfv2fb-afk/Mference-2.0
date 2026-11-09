import Foundation
import Metal

/// Single-kernel Gemma 4 Q/K/V epilogue.
///
/// Equivalent to:
///   Q = rmsnorm_bf16w_perhead(Q, q_norm); rope_neox(Q)
///   K = rmsnorm_bf16w_perhead(K, k_norm); rope_neox(K)
///   V = rmsnorm_no_scale_perhead(V)
final class FusedQKVEpilogue {
    private struct Shape: Hashable {
        var headDim: UInt32
        var numQHeads: UInt32
        var numKVHeads: UInt32
        var rotatedPairs: UInt32
    }

    private let pso: MTLComputePipelineState
    private let specializedPSOs: [Shape: MTLComputePipelineState]
    private let qwenGatedPSO: MTLComputePipelineState
    private let qwenGatedSpecializedPSOs: [Shape: MTLComputePipelineState]
    private static let realDecodeShapes: [Shape] = [
        Shape(headDim: 256, numQHeads: 16, numKVHeads: 8, rotatedPairs: 128),
        Shape(headDim: 512, numQHeads: 16, numKVHeads: 2, rotatedPairs: 64),
    ]
    private static let realQwenGatedShapes: [Shape] = [
        Shape(headDim: 256, numQHeads: 16, numKVHeads: 2, rotatedPairs: 64),
    ]

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("fused_qkv_epilogue")
        var variants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                "fused_qkv_epilogue",
                constants: [
                    MetalFunctionConstant(index: 82, value: .uint32(shape.headDim)),
                    MetalFunctionConstant(index: 83, value: .uint32(shape.numQHeads)),
                    MetalFunctionConstant(index: 84, value: .uint32(shape.numKVHeads)),
                    MetalFunctionConstant(index: 85, value: .uint32(shape.rotatedPairs)),
                    MetalFunctionConstant(index: 86, value: .bool(true)),
                ])
        }
        self.specializedPSOs = variants

        self.qwenGatedPSO = try context.pipeline("fused_qwen_gated_qkv_epilogue")
        var qwenVariants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realQwenGatedShapes {
            qwenVariants[shape] = try context.pipeline(
                "fused_qwen_gated_qkv_epilogue",
                constants: [
                    MetalFunctionConstant(index: 82, value: .uint32(shape.headDim)),
                    MetalFunctionConstant(index: 83, value: .uint32(shape.numQHeads)),
                    MetalFunctionConstant(index: 84, value: .uint32(shape.numKVHeads)),
                    MetalFunctionConstant(index: 85, value: .uint32(shape.rotatedPairs)),
                    MetalFunctionConstant(index: 86, value: .bool(true)),
                ])
        }
        self.qwenGatedSpecializedPSOs = qwenVariants
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                       q: MTLBuffer,
                       qOffset: Int = 0,
                       k: MTLBuffer,
                       kOffset: Int = 0,
                       v: MTLBuffer,
                       vOffset: Int = 0,
                       qWeight: MTLBuffer,
                       qWeightOffset: Int = 0,
                       kWeight: MTLBuffer,
                       kWeightOffset: Int = 0,
                       headDim: UInt32,
                       numQHeads: UInt32,
                       numKVHeads: UInt32,
                       position: UInt32,
                       theta: Float,
                       rotatedPairs: UInt32,
                       eps: Float) {
        precondition(headDim <= 512,
                     "headDim > 512 exceeds the fused QKV epilogue scratch")
        precondition(rotatedPairs * 2 <= headDim,
                     "rotatedPairs must fit inside one NeoX head")
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(
            specializedPSOs[Shape(headDim: headDim,
                                  numQHeads: numQHeads,
                                  numKVHeads: numKVHeads,
                                  rotatedPairs: rotatedPairs)] ?? pso)
        enc.setBuffer(q,       offset: qOffset,       index: 0)
        enc.setBuffer(k,       offset: kOffset,       index: 1)
        enc.setBuffer(v,       offset: vOffset,       index: 2)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 3)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 4)
        var headDimVar = headDim
        var numQVar = numQHeads
        var numKVVar = numKVHeads
        var posVar = position
        var thetaVar = theta
        var rotatedVar = rotatedPairs
        var epsVar = eps
        enc.setBytes(&headDimVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&numQVar,    length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&numKVVar,   length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&posVar,     length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&thetaVar,   length: MemoryLayout<Float>.size,  index: 9)
        enc.setBytes(&rotatedVar, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&epsVar,     length: MemoryLayout<Float>.size,  index: 11)

        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let groups = Int(numQHeads + 2 * numKVHeads)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeQwenGated(commandBuffer cb: MTLCommandBuffer,
                         qPacked: MTLBuffer,
                         qPackedOffset: Int = 0,
                         qOut: MTLBuffer,
                         qOutOffset: Int = 0,
                         gateOut: MTLBuffer,
                         gateOutOffset: Int = 0,
                         kInOut: MTLBuffer,
                         kInOutOffset: Int = 0,
                         qWeight: MTLBuffer,
                         qWeightOffset: Int = 0,
                         kWeight: MTLBuffer,
                         kWeightOffset: Int = 0,
                         headDim: UInt32,
                         numQHeads: UInt32,
                         numKVHeads: UInt32,
                         position: UInt32,
                         theta: Float,
                         rotaryDim: UInt32,
                         eps: Float) {
        precondition(headDim <= 512,
                     "headDim > 512 exceeds the fused QKV epilogue scratch")
        precondition(rotaryDim <= headDim && rotaryDim.isMultiple(of: 2),
                     "rotaryDim must be even and fit inside headDim")
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        let pipeline = qwenGatedSpecializedPSOs[Shape(headDim: headDim,
                                                      numQHeads: numQHeads,
                                                      numKVHeads: numKVHeads,
                                                      rotatedPairs: rotaryDim)] ?? qwenGatedPSO
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(qPacked, offset: qPackedOffset, index: 0)
        enc.setBuffer(qOut,    offset: qOutOffset,    index: 1)
        enc.setBuffer(gateOut, offset: gateOutOffset, index: 2)
        enc.setBuffer(kInOut,  offset: kInOutOffset,  index: 3)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 4)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 5)
        var headDimVar = headDim
        var numQVar = numQHeads
        var numKVVar = numKVHeads
        var posVar = position
        var thetaVar = theta
        var rotaryVar = rotaryDim
        var epsVar = eps
        enc.setBytes(&headDimVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&numQVar,    length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&numKVVar,   length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&posVar,     length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&thetaVar,   length: MemoryLayout<Float>.size,  index: 10)
        enc.setBytes(&rotaryVar,  length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&epsVar,     length: MemoryLayout<Float>.size,  index: 12)

        let threads = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        let groups = Int(numQHeads + numKVHeads)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
