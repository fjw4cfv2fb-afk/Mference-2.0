import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct GDNVerifyFusionParityTests {

    private static let qwenCfg = LinearAttentionConfig(
        numKHeads: 16,
        numVHeads: 32,
        keyHeadDim: 128,
        valueHeadDim: 128,
        convKernelSize: 4)

    private static func pseudoFloat(_ idx: Int, seed: UInt64, scale: Float) -> Float {
        var x = UInt64(truncatingIfNeeded: idx) &* 0x9E3779B97F4A7C15 ^ seed
        x = (x ^ (x >> 30)) &* 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) &* 0x94D049BB133111EB
        x ^= x >> 31
        let signed = Int32(truncatingIfNeeded: x & 0xFFFF) - 32768
        return (Float(signed) / 32768.0) * scale
    }

    private static func makeFP16Buffer(_ device: MTLDevice, count: Int, seed: UInt64, scale: Float) -> MTLBuffer {
        var bits = [UInt16](repeating: 0, count: count)
        for i in 0..<count {
            bits[i] = Float16(pseudoFloat(i, seed: seed, scale: scale)).bitPattern
        }
        return device.makeBuffer(bytes: bits, length: count * 2, options: .storageModeShared)!
    }

    private static func makeBF16Buffer(_ device: MTLDevice, count: Int, seed: UInt64, scale: Float) -> MTLBuffer {
        var bits = [UInt16](repeating: 0, count: count)
        for i in 0..<count {
            let f = pseudoFloat(i, seed: seed, scale: scale)
            bits[i] = UInt16(truncatingIfNeeded: f.bitPattern >> 16)
        }
        return device.makeBuffer(bytes: bits, length: count * 2, options: .storageModeShared)!
    }

    private static func fillSentinelU32(_ buffer: MTLBuffer, pattern: UInt32 = 0xCAFEBABE) {
        let count = buffer.length / 4
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count { ptr[i] = pattern }
    }

    private static func fillSentinelU16(_ buffer: MTLBuffer, pattern: UInt16 = 0xBEEF) {
        let count = buffer.length / 2
        let ptr = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count { ptr[i] = pattern }
    }

    private func verifyBitExact(rows: Int, outputGate: GDN.OutputGate) throws {
        let ctx = try MetalContext()
        let gdn = try GDN(context: ctx, config: Self.qwenCfg, useQwenDecodeSpecialization: true, outputGate: outputGate)
        let device = ctx.device

        let C = Self.qwenCfg.qkvDim // 8192
        let history = Self.qwenCfg.convKernelSize - 1 // 3
        let Hv = Self.qwenCfg.numVHeads // 32
        let Dv = Self.qwenCfg.valueHeadDim // 128
        let Dk = Self.qwenCfg.keyHeadDim // 128
        let valueDim = Self.qwenCfg.valueDim // 4096

        let numLinearLayers = 3
        let ordinal = 1
        let stateCount = Hv * Dv * Dk
        let stateBytes = stateCount * 4
        let tailCount = history * C
        let tailBytes = tailCount * 2
        let fp16Stride = 2

        let initialStateBits = (0..<stateCount).map {
            pseudoFloatBits($0, seed: 101, scale: 0.5)
        }
        let initialTailBits = (0..<tailCount).map {
            Float16(Self.pseudoFloat($0, seed: 202, scale: 0.75)).bitPattern
        }

        let qkvBuf = Self.makeFP16Buffer(device, count: rows * C, seed: 303, scale: 0.8)
        let convOutBuf = Self.makeFP16Buffer(device, count: rows * C, seed: 404, scale: 0.25)
        let aProjBuf = Self.makeFP16Buffer(device, count: rows * Hv, seed: 505, scale: 1.5)
        let bProjBuf = Self.makeFP16Buffer(device, count: rows * Hv, seed: 606, scale: 1.5)
        let aLogBuf = Self.makeBF16Buffer(device, count: Hv, seed: 707, scale: 1.0)
        let dtBiasBuf = Self.makeBF16Buffer(device, count: Hv, seed: 808, scale: 0.5)
        let zBuf = Self.makeFP16Buffer(device, count: rows * valueDim, seed: 909, scale: 1.0)
        let weightBuf = Self.makeBF16Buffer(device, count: Dv, seed: 1010, scale: 1.2)

        // Sequential baseline
        let seqState = device.makeBuffer(bytes: initialStateBits, length: stateBytes, options: .storageModeShared)!
        let seqTail = device.makeBuffer(bytes: initialTailBits, length: tailBytes, options: .storageModeShared)!
        let seqStateArena = device.makeBuffer(length: (rows - 1) * numLinearLayers * stateBytes, options: .storageModeShared)!
        let seqTailArena = device.makeBuffer(length: (rows - 1) * numLinearLayers * tailBytes, options: .storageModeShared)!
        Self.fillSentinelU32(seqStateArena)
        Self.fillSentinelU16(seqTailArena)
        let seqOut = device.makeBuffer(length: rows * valueDim * fp16Stride, options: .storageModeShared)!

        let seqCB = ctx.queue.makeCommandBuffer()!
        for i in 0..<rows {
            gdn.encodeConvTailUpdate(
                commandBuffer: seqCB,
                tail: seqTail,
                qkvRows: qkvBuf,
                qkvRowsOffset: i * C * fp16Stride,
                rows: 1)
            let ok = gdn.encodeDeltaGatedDecode(
                commandBuffer: seqCB,
                convOut: convOutBuf,
                convOutOffset: i * C * fp16Stride,
                aProj: aProjBuf,
                aProjOffset: i * Hv * fp16Stride,
                bProj: bProjBuf,
                bProjOffset: i * Hv * fp16Stride,
                aLog: aLogBuf,
                aLogOffset: 0,
                dtBias: dtBiasBuf,
                dtBiasOffset: 0,
                state: seqState,
                z: zBuf,
                zOffset: i * valueDim * fp16Stride,
                weight: weightBuf,
                weightOffset: 0,
                out: seqOut,
                outOffset: i * valueDim * fp16Stride)
            #expect(ok)
            if i < rows - 1 {
                gdn.encodeStateSnapshot(
                    commandBuffer: seqCB,
                    srcState: seqState, srcStateOffset: 0,
                    dstState: seqStateArena, dstStateOffset: (i * numLinearLayers + ordinal) * stateBytes,
                    srcTail: seqTail, srcTailOffset: 0,
                    dstTail: seqTailArena, dstTailOffset: (i * numLinearLayers + ordinal) * tailBytes,
                    stateBytes: stateBytes,
                    tailBytes: tailBytes)
            }
        }
        seqCB.commit()
        seqCB.waitUntilCompleted()
        #expect(seqCB.error == nil)

        // Fused verify path
        let fusedState = device.makeBuffer(bytes: initialStateBits, length: stateBytes, options: .storageModeShared)!
        let fusedTail = device.makeBuffer(bytes: initialTailBits, length: tailBytes, options: .storageModeShared)!
        let fusedStateArena = device.makeBuffer(length: (rows - 1) * numLinearLayers * stateBytes, options: .storageModeShared)!
        let fusedTailArena = device.makeBuffer(length: (rows - 1) * numLinearLayers * tailBytes, options: .storageModeShared)!
        Self.fillSentinelU32(fusedStateArena)
        Self.fillSentinelU16(fusedTailArena)
        let fusedOut = device.makeBuffer(length: rows * valueDim * fp16Stride, options: .storageModeShared)!

        let fusedCB = ctx.queue.makeCommandBuffer()!
        let didFuse = gdn.encodeVerifyQwen(
            commandBuffer: fusedCB,
            tail: fusedTail,
            qkvRows: qkvBuf,
            tailArena: fusedTailArena,
            tailArenaOffset: ordinal * tailBytes,
            tailSlotStrideHalfs: numLinearLayers * tailCount,
            convOut: convOutBuf,
            aProj: aProjBuf,
            bProj: bProjBuf,
            aLog: aLogBuf, aLogOffset: 0,
            dtBias: dtBiasBuf, dtBiasOffset: 0,
            state: fusedState,
            stateArena: fusedStateArena,
            stateArenaOffset: ordinal * stateBytes,
            stateSlotStrideFloats: numLinearLayers * stateCount,
            z: zBuf,
            weight: weightBuf, weightOffset: 0,
            out: fusedOut,
            rows: rows)
        #expect(didFuse)
        fusedCB.commit()
        fusedCB.waitUntilCompleted()
        #expect(fusedCB.error == nil)

        // Compare bit-for-bit (0 ULP tolerance).
        #expect(memcmp(seqOut.contents(), fusedOut.contents(), seqOut.length) == 0,
                "out mismatch for rows=\(rows), gate=\(outputGate)")
        #expect(memcmp(seqState.contents(), fusedState.contents(), stateBytes) == 0,
                "final state mismatch for rows=\(rows), gate=\(outputGate)")
        #expect(memcmp(seqTail.contents(), fusedTail.contents(), tailBytes) == 0,
                "final tail mismatch for rows=\(rows), gate=\(outputGate)")
        #expect(memcmp(seqStateArena.contents(), fusedStateArena.contents(), seqStateArena.length) == 0,
                "stateArena mismatch for rows=\(rows), gate=\(outputGate)")
        #expect(memcmp(seqTailArena.contents(), fusedTailArena.contents(), seqTailArena.length) == 0,
                "tailArena mismatch for rows=\(rows), gate=\(outputGate)")
    }

    private func pseudoFloatBits(_ idx: Int, seed: UInt64, scale: Float) -> UInt32 {
        Self.pseudoFloat(idx, seed: seed, scale: scale).bitPattern
    }

    @Test func qwenVerifyFusedMatchesSequentialBitExactSilu() throws {
        for rows in 2...4 {
            try verifyBitExact(rows: rows, outputGate: .silu)
        }
    }

    @Test func qwenVerifyFusedMatchesSequentialBitExactSigmoid() throws {
        for rows in 2...4 {
            try verifyBitExact(rows: rows, outputGate: .sigmoid)
        }
    }
}
