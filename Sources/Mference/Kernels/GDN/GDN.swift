import Foundation
import Metal

/// Gated-DeltaNet linear attention kernels (Qwen 3.6 layers with layer-mask
/// value 2). Wraps the `gdn.metal` module: causal depthwise conv + SiLU,
/// per-head q/k norm with the delta-rule scales folded in, the gated delta
/// recurrence (decode step and chunked prefill), and the gated output norm.
///
/// Buffer layouts follow the projection outputs directly:
///  - `mixed_qkv` / `conv_out` rows: `[q: Hk*Dk][k: Hk*Dk][v: Hv*Dv]`
///  - recurrent state: FP32 `[Hv, Dv, Dk]` (owned by `GDNStateManager`)
///  - conv tail: FP16 `[convKernel - 1, convDim]`
final class GDN {
    private let convDecodePSO: MTLComputePipelineState
    private let convPrefillPSO: MTLComputePipelineState
    private let convTailUpdatePSO: MTLComputePipelineState
    private let convTailVerifyPSO: MTLComputePipelineState?
    private let qkNormPSO: MTLComputePipelineState
    private let deltaDecodePSO: MTLComputePipelineState
    private let qwenDeltaDecodePSO: MTLComputePipelineState?
    private let qwenDeltaGatedDecodePSO: MTLComputePipelineState?
    private let qwenDeltaGatedVerifyPSOs: [Int: MTLComputePipelineState]
    private let deltaPrefillPSO: MTLComputePipelineState
    private let qwenDeltaPrefillPSO: MTLComputePipelineState?
    private let gatedNormPSO: MTLComputePipelineState
    private let inProjPSO: MTLComputePipelineState
    private let inProjSpecializedPSO: MTLComputePipelineState?
    private let stateSnapshotPSO: MTLComputePipelineState?

    let config: LinearAttentionConfig

    /// `specializedHiddenSize` compiles a constant-folded variant of the fused
    /// input projection for the decode shape. Measured elsewhere in this
    /// package: an unspecialized INT4 GEMV runs ~102 GB/s against ~141 GB/s
    /// specialized, so the runtime path must not be the only one available.
    /// The gated norm's activation on `z`, from `Qwen3_5RMSNormGated`'s
    /// `output_gate_type`. Qwen 3.6 and Qwen 3.8 gate with silu; Flash-Next
    /// gates with sigmoid. `silu` builds the pipelines with **no** function
    /// constants, which is exactly what the shipped families already did.
    enum OutputGate {
        case silu
        case sigmoid

        /// `FC_GDN_GATE_SIGMOID` in `gdn.metal`. Unset means silu.
        var constants: [MetalFunctionConstant] {
            switch self {
            case .silu: return []
            case .sigmoid: return [MetalFunctionConstant(index: 99, value: .bool(true))]
            }
        }
    }

    init(context: MetalContext, config: LinearAttentionConfig,
         specializedHiddenSize: Int? = nil,
         useQwenDecodeSpecialization: Bool = true,
         outputGate: OutputGate = .silu) throws {
        precondition(config.keyHeadDim > 0 && config.keyHeadDim % 32 == 0,
                     "keyHeadDim must be a positive multiple of 32")
        precondition(config.keyHeadDim / 32 <= 8,
                     "keyHeadDim register tile exceeds the kernel's bound")
        precondition(config.valueHeadDim % 4 == 0,
                     "valueHeadDim must be a multiple of 4")
        precondition(config.numVHeads % config.numKHeads == 0,
                     "numVHeads must be a multiple of numKHeads")
        self.config = config
        self.convDecodePSO = try context.pipeline("gdn_conv_mix_decode")
        self.convPrefillPSO = try context.pipeline("gdn_conv_mix_prefill")
        self.convTailUpdatePSO = try context.pipeline("gdn_conv_tail_update")
        self.convTailVerifyPSO = try? context.pipeline("gdn_conv_tail_verify")
        self.qkNormPSO = try context.pipeline("gdn_qk_norm")
        self.deltaDecodePSO = try context.pipeline("gdn_delta_step_decode")
        let isQwenGeometry = config.numKHeads == 16
            && config.numVHeads == 32
            && config.keyHeadDim == 128
            && config.valueHeadDim == 128
        let isQwen38Geometry = config.numKHeads == 16
            && config.numVHeads == 48
            && config.keyHeadDim == 128
            && config.valueHeadDim == 128
        self.qwenDeltaDecodePSO = useQwenDecodeSpecialization && isQwenGeometry
            ? try context.pipeline("gdn_delta_step_decode_qwen",
                                   constants: [],
                                   maxTotalThreadsPerThreadgroup: 256)
            : nil
        if useQwenDecodeSpecialization && (isQwenGeometry || isQwen38Geometry) {
            self.qwenDeltaGatedDecodePSO = try context.pipeline(
                isQwenGeometry ? "gdn_delta_gated_decode_qwen"
                               : "gdn_delta_gated_decode_qwen38",
                constants: outputGate.constants,
                maxTotalThreadsPerThreadgroup: 256)
        } else {
            self.qwenDeltaGatedDecodePSO = nil
        }
        if useQwenDecodeSpecialization && isQwenGeometry {
            var verifyMap: [Int: MTLComputePipelineState] = [:]
            for t in 2...4 {
                if let pso = try? context.pipeline(
                    "gdn_delta_gated_verify_qwen_t\(t)",
                    constants: outputGate.constants,
                    maxTotalThreadsPerThreadgroup: 256) {
                    verifyMap[t] = pso
                }
            }
            self.qwenDeltaGatedVerifyPSOs = verifyMap
        } else {
            self.qwenDeltaGatedVerifyPSOs = [:]
        }
        self.deltaPrefillPSO = try context.pipeline("gdn_delta_step_prefill")
        self.qwenDeltaPrefillPSO = useQwenDecodeSpecialization && isQwenGeometry
            ? try context.pipeline("gdn_delta_step_prefill_qwen",
                                   constants: [],
                                   maxTotalThreadsPerThreadgroup: 128)
            : nil
        self.gatedNormPSO = try context.pipeline("gdn_gated_norm",
                                                 constants: outputGate.constants)
        self.inProjPSO = try context.pipeline("gdn_in_proj_gemv_simd",
                                              constants: [],
                                              maxTotalThreadsPerThreadgroup: 512)
        if let n = specializedHiddenSize {
            self.inProjSpecializedPSO = try context.pipeline(
                "gdn_in_proj_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 90, value: .uint32(UInt32(config.qkvDim))),
                    MetalFunctionConstant(index: 91, value: .uint32(UInt32(config.valueDim))),
                    MetalFunctionConstant(index: 92, value: .uint32(UInt32(config.numVHeads))),
                    MetalFunctionConstant(index: 93, value: .uint32(UInt32(n))),
                    MetalFunctionConstant(index: 94, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        } else {
            self.inProjSpecializedPSO = nil
        }
        self.stateSnapshotPSO = try? context.pipeline("gdn_state_snapshot")
    }

    /// Zero-barrier compute copy for GDN recurrent state and conv tail.
    /// Runs directly within the compute queue without opening/closing blit encoders.
    func encodeStateSnapshot(commandBuffer: MTLCommandBuffer,
                             srcState: MTLBuffer, srcStateOffset: Int = 0,
                             dstState: MTLBuffer, dstStateOffset: Int = 0,
                             srcTail: MTLBuffer, srcTailOffset: Int = 0,
                             dstTail: MTLBuffer, dstTailOffset: Int = 0,
                             stateBytes: Int,
                             tailBytes: Int) {
        guard let pso = stateSnapshotPSO,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(srcState, offset: srcStateOffset, index: 0)
        encoder.setBuffer(dstState, offset: dstStateOffset, index: 1)
        encoder.setBuffer(srcTail, offset: srcTailOffset, index: 2)
        encoder.setBuffer(dstTail, offset: dstTailOffset, index: 3)
        var stateVecs = UInt32(stateBytes / 16)
        var tailVecs = UInt32(tailBytes / 8)
        encoder.setBytes(&stateVecs, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&tailVecs, length: MemoryLayout<UInt32>.size, index: 5)
        let totalThreads = max(Int(stateVecs), Int(tailVecs))
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalThreads + 255) / 256, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Fused `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` INT4 GEMV.
    /// One dispatch over the concatenated row space replaces four, of which two
    /// (a and b, `numVHeads` rows each) were near-empty launches. Bit-identical
    /// to the four separate GEMVs — the per-row body and its operand order are
    /// unchanged.
    func encodeInputProjections(commandBuffer: MTLCommandBuffer,
                                x: MTLBuffer, xOffset: Int = 0,
                                qkv: TensorView, qkvOut: MTLBuffer, qkvOutOffset: Int = 0,
                                z: TensorView, zOut: MTLBuffer, zOutOffset: Int = 0,
                                a: TensorView, aOut: MTLBuffer, aOutOffset: Int = 0,
                                b: TensorView, bOut: MTLBuffer, bOutOffset: Int = 0,
                                hiddenSize: Int) {
        precondition(hiddenSize % Quantization.groupSize == 0,
                     "hiddenSize must be a multiple of \(Quantization.groupSize)")
        // The row body reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte.
        precondition(Int(qkv.offset) % 2 == 0 && Int(z.offset) % 2 == 0 &&
                     Int(a.offset) % 2 == 0 && Int(b.offset) % 2 == 0,
                     "gdn_in_proj_gemv_simd needs 2-aligned weights offsets")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(inProjSpecializedPSO ?? inProjPSO)
        for (slot, view) in [qkv, z, a, b].enumerated() {
            encoder.setBuffer(view.buffer, offset: Int(view.offset), index: slot * 3)
            encoder.setBuffer(view.buffer, offset: Int(view.scaleOffset), index: slot * 3 + 1)
            encoder.setBuffer(view.buffer, offset: Int(view.biasOffset), index: slot * 3 + 2)
        }
        encoder.setBuffer(x, offset: xOffset, index: 12)
        encoder.setBuffer(qkvOut, offset: qkvOutOffset, index: 13)
        encoder.setBuffer(zOut, offset: zOutOffset, index: 14)
        encoder.setBuffer(aOut, offset: aOutOffset, index: 15)
        encoder.setBuffer(bOut, offset: bOutOffset, index: 16)
        var qkvRows = UInt32(config.qkvDim)
        var zRows = UInt32(config.valueDim)
        var abRows = UInt32(config.numVHeads)
        var n = UInt32(hiddenSize)
        encoder.setBytes(&qkvRows, length: MemoryLayout<UInt32>.size, index: 17)
        encoder.setBytes(&zRows, length: MemoryLayout<UInt32>.size, index: 18)
        encoder.setBytes(&abRows, length: MemoryLayout<UInt32>.size, index: 19)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 20)
        let totalRows = config.qkvDim + config.valueDim + 2 * config.numVHeads
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalRows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Decode: conv over [tail | current row] with SiLU, shifting the tail in
    /// place. `convWeight` is the BF16 `[convDim, kernel]` tensor view region.
    func encodeConvDecode(commandBuffer: MTLCommandBuffer,
                          tail: MTLBuffer,
                          qkv: MTLBuffer, qkvOffset: Int = 0,
                          convWeight: MTLBuffer, convWeightOffset: Int,
                          out: MTLBuffer, outOffset: Int = 0) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convDecodePSO)
        encoder.setBuffer(tail, offset: 0, index: 0)
        encoder.setBuffer(qkv, offset: qkvOffset, index: 1)
        encoder.setBuffer(convWeight, offset: convWeightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var channels = UInt32(config.qkvDim)
        var taps = UInt32(config.convKernelSize)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
        dispatch1D(encoder, pipeline: convDecodePSO, threads: config.qkvDim)
        encoder.endEncoding()
    }

    /// Prefill: conv over [tail | chunk rows]; the tail is read-only here.
    func encodeConvPrefill(commandBuffer: MTLCommandBuffer,
                           tail: MTLBuffer,
                           qkvRows: MTLBuffer, qkvRowsOffset: Int = 0,
                           convWeight: MTLBuffer, convWeightOffset: Int,
                           out: MTLBuffer, outOffset: Int = 0,
                           rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convPrefillPSO)
        encoder.setBuffer(tail, offset: 0, index: 0)
        encoder.setBuffer(qkvRows, offset: qkvRowsOffset, index: 1)
        encoder.setBuffer(convWeight, offset: convWeightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var channels = UInt32(config.qkvDim)
        var taps = UInt32(config.convKernelSize)
        var rowCount = UInt32(rows)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 6)
        dispatch2D(encoder, pipeline: convPrefillPSO,
                   width: config.qkvDim, height: rows)
        encoder.endEncoding()
    }

    /// After a prefill chunk: tail := last (kernel-1) raw rows of [tail | chunk].
    func encodeConvTailUpdate(commandBuffer: MTLCommandBuffer,
                              tail: MTLBuffer,
                              qkvRows: MTLBuffer, qkvRowsOffset: Int = 0,
                              rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convTailUpdatePSO)
        encoder.setBuffer(tail, offset: 0, index: 0)
        encoder.setBuffer(qkvRows, offset: qkvRowsOffset, index: 1)
        var channels = UInt32(config.qkvDim)
        var taps = UInt32(config.convKernelSize)
        var rowCount = UInt32(rows)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch2D(encoder, pipeline: convTailUpdatePSO,
                   width: config.qkvDim, height: config.convKernelSize - 1)
        encoder.endEncoding()
    }

    /// Per-head no-weight RMS norm over the q and k slices of `convOut`,
    /// in place, with the delta-rule scales folded in. `rows` > 1 for prefill.
    func encodeQKNorm(commandBuffer: MTLCommandBuffer,
                      convOut: MTLBuffer, convOutOffset: Int = 0,
                      rows: Int = 1) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qkNormPSO)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        var kHeads = UInt32(config.numKHeads)
        var keyDim = UInt32(config.keyHeadDim)
        var rowStride = UInt32(config.qkvDim)
        encoder.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 2 * config.numKHeads, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Decode: one gated delta rule step. `state` is FP32 [Hv, Dv, Dk],
    /// updated in place; `y` receives [Hv * Dv] FP16.
    func encodeDeltaStepDecode(commandBuffer: MTLCommandBuffer,
                               convOut: MTLBuffer, convOutOffset: Int = 0,
                               aProj: MTLBuffer, aProjOffset: Int = 0,
                               bProj: MTLBuffer, bProjOffset: Int = 0,
                               aLog: MTLBuffer, aLogOffset: Int,
                               dtBias: MTLBuffer, dtBiasOffset: Int,
                               state: MTLBuffer,
                               y: MTLBuffer, yOffset: Int = 0) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let pipeline = qwenDeltaDecodePSO ?? deltaDecodePSO
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        encoder.setBuffer(aProj, offset: aProjOffset, index: 1)
        encoder.setBuffer(bProj, offset: bProjOffset, index: 2)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 3)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 4)
        encoder.setBuffer(state, offset: 0, index: 5)
        encoder.setBuffer(y, offset: yOffset, index: 6)
        setHeadDims(encoder, startingAt: 7)
        if qwenDeltaDecodePSO != nil {
            encoder.dispatchThreadgroups(
                MTLSize(width: config.numVHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            encoder.dispatchThreadgroups(
                MTLSize(width: config.numVHeads,
                        height: config.valueHeadDim / 4,
                        depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        }
        encoder.endEncoding()
    }

    /// Qwen decode recurrence plus the following per-head gated RMS norm.
    /// The recurrence keeps the same lane mapping and FP16 `y` boundary as
    /// `encodeDeltaStepDecode`; the half values remain in threadgroup memory
    /// for the norm, removing one global round trip and one dispatch.
    /// Returns false for non-Qwen geometries so callers can use the generic
    /// two-dispatch path.
    @discardableResult
    func encodeDeltaGatedDecode(commandBuffer: MTLCommandBuffer,
                                convOut: MTLBuffer, convOutOffset: Int = 0,
                                aProj: MTLBuffer, aProjOffset: Int = 0,
                                bProj: MTLBuffer, bProjOffset: Int = 0,
                                aLog: MTLBuffer, aLogOffset: Int,
                                dtBias: MTLBuffer, dtBiasOffset: Int,
                                state: MTLBuffer,
                                z: MTLBuffer, zOffset: Int = 0,
                                weight: MTLBuffer, weightOffset: Int,
                                out: MTLBuffer, outOffset: Int = 0) -> Bool {
        guard let pipeline = qwenDeltaGatedDecodePSO,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        encoder.setBuffer(aProj, offset: aProjOffset, index: 1)
        encoder.setBuffer(bProj, offset: bProjOffset, index: 2)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 3)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 4)
        encoder.setBuffer(state, offset: 0, index: 5)
        encoder.setBuffer(z, offset: zOffset, index: 6)
        encoder.setBuffer(weight, offset: weightOffset, index: 7)
        encoder.setBuffer(out, offset: outOffset, index: 8)
        setHeadDims(encoder, startingAt: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: config.numVHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    /// Fused multi-token Qwen verify pass (`rows` in `2...4`): updates conv `tail`
    /// while capturing `0..<rows-1` tail snapshots into `tailArena`, then runs
    /// the register-resident multi-step recurrence + gated RMS norm while
    /// writing intermediate states directly into `stateArena` and final state
    /// to `state`. Bit-identical (0 ULP) to sequential decode + snapshots.
    @discardableResult
    func encodeVerifyQwen(commandBuffer: MTLCommandBuffer,
                          tail: MTLBuffer,
                          qkvRows: MTLBuffer, qkvRowsOffset: Int = 0,
                          tailArena: MTLBuffer, tailArenaOffset: Int,
                          tailSlotStrideHalfs: Int,
                          convOut: MTLBuffer, convOutOffset: Int = 0,
                          aProj: MTLBuffer, aProjOffset: Int = 0,
                          bProj: MTLBuffer, bProjOffset: Int = 0,
                          aLog: MTLBuffer, aLogOffset: Int,
                          dtBias: MTLBuffer, dtBiasOffset: Int,
                          state: MTLBuffer,
                          stateArena: MTLBuffer, stateArenaOffset: Int,
                          stateSlotStrideFloats: Int,
                          z: MTLBuffer, zOffset: Int = 0,
                          weight: MTLBuffer, weightOffset: Int,
                          out: MTLBuffer, outOffset: Int = 0,
                          rows: Int) -> Bool {
        guard (2...4).contains(rows),
              config.convKernelSize >= 2 && config.convKernelSize <= 9,
              let tailVerifyPSO = convTailVerifyPSO,
              let deltaVerifyPSO = qwenDeltaGatedVerifyPSOs[rows] else {
            return false
        }
        guard let tailEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }
        tailEncoder.setComputePipelineState(tailVerifyPSO)
        tailEncoder.setBuffer(tail, offset: 0, index: 0)
        tailEncoder.setBuffer(qkvRows, offset: qkvRowsOffset, index: 1)
        tailEncoder.setBuffer(tailArena, offset: tailArenaOffset, index: 2)
        var channels = UInt32(config.qkvDim)
        var taps = UInt32(config.convKernelSize)
        var rowCount = UInt32(rows)
        var tailStride = UInt32(tailSlotStrideHalfs)
        tailEncoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 3)
        tailEncoder.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 4)
        tailEncoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 5)
        tailEncoder.setBytes(&tailStride, length: MemoryLayout<UInt32>.size, index: 6)
        dispatch1D(tailEncoder, pipeline: tailVerifyPSO, threads: config.qkvDim)
        tailEncoder.endEncoding()

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }
        encoder.setComputePipelineState(deltaVerifyPSO)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        encoder.setBuffer(aProj, offset: aProjOffset, index: 1)
        encoder.setBuffer(bProj, offset: bProjOffset, index: 2)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 3)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 4)
        encoder.setBuffer(state, offset: 0, index: 5)
        encoder.setBuffer(stateArena, offset: stateArenaOffset, index: 6)
        encoder.setBuffer(z, offset: zOffset, index: 7)
        encoder.setBuffer(weight, offset: weightOffset, index: 8)
        encoder.setBuffer(out, offset: outOffset, index: 9)
        setHeadDims(encoder, startingAt: 10)
        var rowStride = UInt32(config.qkvDim)
        var stateStride = UInt32(stateSlotStrideFloats)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 14)
        encoder.setBytes(&stateStride, length: MemoryLayout<UInt32>.size, index: 15)
        encoder.dispatchThreadgroups(
            MTLSize(width: config.numVHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    /// Prefill: the recurrence runs sequentially over `rows` inside the
    /// kernel; state persists in registers and is written back once.
    func encodeDeltaStepPrefill(commandBuffer: MTLCommandBuffer,
                                convOut: MTLBuffer, convOutOffset: Int = 0,
                                aProj: MTLBuffer, aProjOffset: Int = 0,
                                bProj: MTLBuffer, bProjOffset: Int = 0,
                                aLog: MTLBuffer, aLogOffset: Int,
                                dtBias: MTLBuffer, dtBiasOffset: Int,
                                state: MTLBuffer,
                                y: MTLBuffer, yOffset: Int = 0,
                                rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qwenDeltaPrefillPSO ?? deltaPrefillPSO)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        encoder.setBuffer(aProj, offset: aProjOffset, index: 1)
        encoder.setBuffer(bProj, offset: bProjOffset, index: 2)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 3)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 4)
        encoder.setBuffer(state, offset: 0, index: 5)
        encoder.setBuffer(y, offset: yOffset, index: 6)
        setHeadDims(encoder, startingAt: 7)
        var rowCount = UInt32(rows)
        var rowStride = UInt32(config.qkvDim)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 11)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 12)
        encoder.dispatchThreadgroups(
            MTLSize(width: config.numVHeads,
                    height: config.valueHeadDim / 4,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        encoder.endEncoding()
    }

    /// out = rmsnorm(y; weight) * silu(z), per value head, `rows` rows.
    func encodeGatedNorm(commandBuffer: MTLCommandBuffer,
                         y: MTLBuffer, yOffset: Int = 0,
                         z: MTLBuffer, zOffset: Int = 0,
                         weight: MTLBuffer, weightOffset: Int,
                         out: MTLBuffer, outOffset: Int = 0,
                         rows: Int = 1) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(gatedNormPSO)
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(z, offset: zOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var vHeads = UInt32(config.numVHeads)
        var valueDim = UInt32(config.valueHeadDim)
        encoder.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: config.numVHeads, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func setHeadDims(_ encoder: MTLComputeCommandEncoder, startingAt index: Int) {
        var kHeads = UInt32(config.numKHeads)
        var vHeads = UInt32(config.numVHeads)
        var keyDim = UInt32(config.keyHeadDim)
        var valueDim = UInt32(config.valueHeadDim)
        encoder.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: index)
        encoder.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: index + 1)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: index + 2)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.size, index: index + 3)
    }

    private func dispatch1D(_ encoder: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState,
                            threads: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func dispatch2D(_ encoder: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState,
                            width: Int, height: Int) {
        let tgWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
    }
}
