import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct VisionMetalParityTests {

    private static func pseudoFloat(_ idx: Int, seed: UInt64, scale: Float) -> Float {
        var x = UInt64(truncatingIfNeeded: idx) &* 0x9E3779B97F4A7C15 ^ seed
        x = (x ^ (x >> 30)) &* 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) &* 0x94D049BB133111EB
        x ^= x >> 31
        let signed = Int32(truncatingIfNeeded: x & 0xFFFF) - 32768
        return (Float(signed) / 32768.0) * scale
    }

    private static func bf16Rounded(_ x: Float) -> (bits: UInt16, value: Float) {
        let bits = UInt16(truncatingIfNeeded: x.bitPattern >> 16)
        return (bits, Float(bitPattern: UInt32(bits) << 16))
    }

    private static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Double = 0
        var normA: Double = 0
        var normB: Double = 0
        for i in 0..<a.count {
            let av = Double(a[i])
            let bv = Double(b[i])
            dot += av * bv
            normA += av * av
            normB += bv * bv
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? Float(dot / denom) : 0
    }

    private static func geluTanh(_ v: Float) -> Float {
        let c0: Float = 0.7978845608028654
        let c1: Float = 0.044715
        return 0.5 * v * (1.0 + tanhf(c0 * (v + c1 * v * v * v)))
    }

    private static func geluExact(_ v: Float) -> Float {
        0.5 * v * (1.0 + erff(v * 0.7071067811865475))
    }

    @Test func gemmParityAcrossNonAlignedShapesAndFlags() throws {
        let ctx = try MetalContext()
        guard let pipelines = Qwen36VisionEncoder.makeMetalPipelines(context: ctx) else {
            return
        }

        let cases: [(m: Int, n: Int, k: Int, flags: UInt32)] = [
            // Non-64 M, non-32 N=4304, bias + geluTanh (Qwen36 fc1)
            (60, 4304, 128, 1 | 2),
            // Non-64 M, non-64 K=4304, bias + residualAdd (Qwen36 fc2)
            (60, 96, 4304, 1 | 8),
            // Non-64 K & exact GELU (Qwen36 merger.fc1 activation)
            (36, 128, 192, 1 | 4),
            // Pure matmul without bias or activation
            (17, 65, 130, 0),
        ]

        for (caseIdx, spec) in cases.enumerated() {
            let xVals = (0..<(spec.m * spec.k)).map { Self.pseudoFloat($0, seed: UInt64(10 + caseIdx), scale: 0.5) }
            var wBits = [UInt16](repeating: 0, count: spec.n * spec.k)
            var wVals = [Float](repeating: 0, count: spec.n * spec.k)
            for i in 0..<wVals.count {
                let (b, f) = Self.bf16Rounded(Self.pseudoFloat(i, seed: UInt64(100 + caseIdx), scale: 0.35))
                wBits[i] = b
                wVals[i] = f
            }
            var bBits = [UInt16](repeating: 0, count: spec.n)
            var bVals = [Float](repeating: 0, count: spec.n)
            for i in 0..<spec.n {
                let (b, f) = Self.bf16Rounded(Self.pseudoFloat(i, seed: UInt64(200 + caseIdx), scale: 0.25))
                bBits[i] = b
                bVals[i] = f
            }
            let initY = (0..<(spec.m * spec.n)).map { Self.pseudoFloat($0, seed: UInt64(300 + caseIdx), scale: 0.4) }

            let xBuf = ctx.device.makeBuffer(bytes: xVals, length: xVals.count * 4, options: .storageModeShared)!
            let wBuf = ctx.device.makeBuffer(bytes: wBits, length: wBits.count * 2, options: .storageModeShared)!
            let bBuf = ctx.device.makeBuffer(bytes: bBits, length: bBits.count * 2, options: .storageModeShared)!
            let yBuf = ctx.device.makeBuffer(bytes: initY, length: initY.count * 4, options: .storageModeShared)!

            let cb = ctx.queue.makeCommandBuffer()!
            Qwen36VisionEncoder.encodeGemm(
                commandBuffer: cb,
                pipeline: pipelines.gemm,
                x: xBuf,
                weight: wBuf,
                bias: bBuf,
                y: yBuf,
                m: spec.m, n: spec.n, k: spec.k,
                flags: spec.flags)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)

            let gpuPtr = yBuf.contents().bindMemory(to: Float.self, capacity: spec.m * spec.n)
            let gpuOut = Array(UnsafeBufferPointer(start: gpuPtr, count: spec.m * spec.n))

            var refOut = [Float](repeating: 0, count: spec.m * spec.n)
            for row in 0..<spec.m {
                for col in 0..<spec.n {
                    var sum: Float = 0
                    for kk in 0..<spec.k {
                        sum += xVals[row * spec.k + kk] * wVals[col * spec.k + kk]
                    }
                    if (spec.flags & 1) != 0 { sum += bVals[col] }
                    if (spec.flags & 2) != 0 { sum = Self.geluTanh(sum) }
                    else if (spec.flags & 4) != 0 { sum = Self.geluExact(sum) }
                    let idx = row * spec.n + col
                    refOut[idx] = (spec.flags & 8) != 0 ? (initY[idx] + sum) : sum
                }
            }

            for v in gpuOut { #expect(v.isFinite) }
            let cos = Self.cosineSimilarity(gpuOut, refOut)
            var maxAbs: Float = 0
            for i in 0..<gpuOut.count {
                maxAbs = max(maxAbs, abs(gpuOut[i] - refOut[i]))
            }
            #expect(cos > 0.9999, "cosine \(cos) too low for case \(spec)")
            #expect(maxAbs < 0.08, "maxAbs \(maxAbs) too large for case \(spec)")
        }
    }

    @Test func layerNormRoPEAndAttentionParity() throws {
        let ctx = try MetalContext()
        guard let pipelines = Qwen36VisionEncoder.makeMetalPipelines(context: ctx) else {
            return
        }

        // 1. LayerNorm
        let lnRows = 37
        let dim = 1152
        let xVals = (0..<(lnRows * dim)).map { Self.pseudoFloat($0, seed: 11, scale: 2.0) }
        var wBits = [UInt16](repeating: 0, count: dim)
        var wVals = [Float](repeating: 0, count: dim)
        var bBits = [UInt16](repeating: 0, count: dim)
        var bVals = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            (wBits[i], wVals[i]) = Self.bf16Rounded(Self.pseudoFloat(i, seed: 22, scale: 1.5))
            (bBits[i], bVals[i]) = Self.bf16Rounded(Self.pseudoFloat(i, seed: 33, scale: 0.5))
        }
        let xBuf = ctx.device.makeBuffer(bytes: xVals, length: xVals.count * 4, options: .storageModeShared)!
        let wBuf = ctx.device.makeBuffer(bytes: wBits, length: wBits.count * 2, options: .storageModeShared)!
        let bBuf = ctx.device.makeBuffer(bytes: bBits, length: bBits.count * 2, options: .storageModeShared)!
        let lnOutBuf = ctx.device.makeBuffer(length: xVals.count * 4, options: .storageModeShared)!

        let cbLN = ctx.queue.makeCommandBuffer()!
        Qwen36VisionEncoder.encodeLayerNorm(
            commandBuffer: cbLN,
            pipeline: pipelines.layerNorm,
            x: xBuf, weight: wBuf, bias: bBuf, out: lnOutBuf,
            rows: lnRows, dim: dim, eps: 1e-6)
        cbLN.commit()
        cbLN.waitUntilCompleted()
        #expect(cbLN.error == nil)

        let gpuLN = Array(UnsafeBufferPointer(
            start: lnOutBuf.contents().bindMemory(to: Float.self, capacity: xVals.count),
            count: xVals.count))
        var maxLNErr: Float = 0
        for r in 0..<lnRows {
            var sum: Float = 0
            for c in 0..<dim { sum += xVals[r * dim + c] }
            let mean = sum / Float(dim)
            var sq: Float = 0
            for c in 0..<dim {
                let d = xVals[r * dim + c] - mean
                sq += d * d
            }
            let invStd = 1.0 / sqrtf(sq / Float(dim) + 1e-6)
            for c in 0..<dim {
                let ref = (xVals[r * dim + c] - mean) * invStd * wVals[c] + bVals[c]
                maxLNErr = max(maxLNErr, abs(gpuLN[r * dim + c] - ref))
            }
        }
        #expect(maxLNErr < 1e-4)

        // 2. 2D RoPE + Online Attention (both seqLen=3 < 4 and seqLen=64)
        for seqLen in [3, 64] {
            let numHeads = 16
            let headDim = 72
            let d = numHeads * headDim
            let halfD = headDim / 2
            var qkvVals = (0..<(seqLen * 3 * d)).map { Self.pseudoFloat($0, seed: UInt64(44 + seqLen), scale: 1.0) }
            let cosVals = (0..<(seqLen * halfD)).map { cosf(Self.pseudoFloat($0, seed: 55, scale: 3.14)) }
            let sinVals = (0..<(seqLen * halfD)).map { sinf(Self.pseudoFloat($0, seed: 55, scale: 3.14)) }

            let qkvBuf = ctx.device.makeBuffer(bytes: qkvVals, length: qkvVals.count * 4, options: .storageModeShared)!
            let cosBuf = ctx.device.makeBuffer(bytes: cosVals, length: cosVals.count * 4, options: .storageModeShared)!
            let sinBuf = ctx.device.makeBuffer(bytes: sinVals, length: sinVals.count * 4, options: .storageModeShared)!
            let attnOutBuf = ctx.device.makeBuffer(length: seqLen * d * 4, options: .storageModeShared)!

            let scale = 1.0 / sqrtf(Float(headDim))
            let cb = ctx.queue.makeCommandBuffer()!
            Qwen36VisionEncoder.encodeRoPE2D(
                commandBuffer: cb,
                pipeline: pipelines.rope,
                qkv: qkvBuf, cosTable: cosBuf, sinTable: sinBuf,
                seqLen: seqLen, numHeads: numHeads, headDim: headDim)
            Qwen36VisionEncoder.encodeAttention(
                commandBuffer: cb,
                pipeline: pipelines.attention,
                qkv: qkvBuf, out: attnOutBuf,
                seqLen: seqLen, numHeads: numHeads, headDim: headDim, scale: scale)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)

            // CPU reference
            for s in 0..<seqLen {
                let rowOff = s * 3 * d
                for h in 0..<numHeads {
                    let qOff = rowOff + h * headDim
                    let kOff = rowOff + d + h * headDim
                    for j in 0..<halfD {
                        let cs = cosVals[s * halfD + j]
                        let sn = sinVals[s * halfD + j]
                        let q0 = qkvVals[qOff + j]
                        let q1 = qkvVals[qOff + j + halfD]
                        qkvVals[qOff + j] = q0 * cs - q1 * sn
                        qkvVals[qOff + j + halfD] = q1 * cs + q0 * sn

                        let k0 = qkvVals[kOff + j]
                        let k1 = qkvVals[kOff + j + halfD]
                        qkvVals[kOff + j] = k0 * cs - k1 * sn
                        qkvVals[kOff + j + halfD] = k1 * cs + k0 * sn
                    }
                }
            }
            var refOut = [Float](repeating: 0, count: seqLen * d)
            for h in 0..<numHeads {
                for qr in 0..<seqLen {
                    var scores = [Float](repeating: 0, count: seqLen)
                    var maxScore: Float = -.infinity
                    for kr in 0..<seqLen {
                        var dot: Float = 0
                        let qb = qr * 3 * d + h * headDim
                        let kb = kr * 3 * d + d + h * headDim
                        for c in 0..<headDim { dot += qkvVals[qb + c] * qkvVals[kb + c] }
                        scores[kr] = dot * scale
                        maxScore = max(maxScore, scores[kr])
                    }
                    var denom: Float = 0
                    for kr in 0..<seqLen {
                        scores[kr] = expf(scores[kr] - maxScore)
                        denom += scores[kr]
                    }
                    for c in 0..<headDim {
                        var acc: Float = 0
                        for kr in 0..<seqLen {
                            acc += (scores[kr] / denom) * qkvVals[kr * 3 * d + 2 * d + h * headDim + c]
                        }
                        refOut[qr * d + h * headDim + c] = acc
                    }
                }
            }

            let gpuOut = Array(UnsafeBufferPointer(
                start: attnOutBuf.contents().bindMemory(to: Float.self, capacity: seqLen * d),
                count: seqLen * d))
            let cos = Self.cosineSimilarity(gpuOut, refOut)
            var maxErr: Float = 0
            for i in 0..<gpuOut.count { maxErr = max(maxErr, abs(gpuOut[i] - refOut[i])) }
            #expect(cos > 0.99999)
            #expect(maxErr < 1e-4)
        }
    }

    @Test func installedRealVisionWeightsMetalVsCPUParityIfEnabled() throws {
        guard let rawDir = ProcessInfo.processInfo.environment["MFERENCE_QWEN36_VISION_PARITY_GTURBO"],
              !rawDir.isEmpty else {
            return
        }
        let weightsURL = URL(fileURLWithPath: rawDir).appendingPathComponent("vision_weights.bin")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            return
        }
        let ctx = try MetalContext()
        let encoder = Qwen36VisionEncoder(weightsURL: weightsURL, context: ctx)

        let gridH = 8
        let gridW = 8
        let patches = (0..<(gridH * gridW * Qwen36VisionEncoder.patchDim)).map {
            Self.pseudoFloat($0, seed: 777, scale: 1.0)
        }
        let patchData = Qwen36VisionPatchData(gridT: 1, gridH: gridH, gridW: gridW, patches: patches)

        let cpuOut = try encoder.runEncoder(patchData: patchData, forceCPU: true)
        let metalOut = try encoder.runEncoder(patchData: patchData, forceCPU: false)
        #expect(cpuOut.count == metalOut.count)
        let cos = Self.cosineSimilarity(cpuOut.map(Float.init), metalOut.map(Float.init))
        #expect(cos > 0.9999, "full real-weight Metal vs CPU vision embedding cosine similarity \(cos)")
    }
}
