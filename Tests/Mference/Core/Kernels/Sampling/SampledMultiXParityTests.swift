import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct SampledMultiXParityTests {
    @Test("multi-x logits head matches single-row DequantInt4GEMV bit for bit",
          arguments: [(64, 1024, 1), (256, 1000, 3), (256, 2048, 4)])
    func multixLogitsMatchGEMV(d: Int, vocab: Int, t: Int) throws {
        let ctx = try MetalContext()
        let dev = ctx.device
        var rng: UInt64 = 0xCAFE_BABE &+ UInt64(d &* vocab &+ t)
        func nextByte() -> UInt8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return UInt8(truncatingIfNeeded: rng >> 33)
        }
        let groups = d / 64
        let w = dev.makeBuffer(length: vocab * d / 2, options: .storageModeShared)!
        let s = dev.makeBuffer(length: vocab * groups * 2, options: .storageModeShared)!
        let b = dev.makeBuffer(length: vocab * groups * 2, options: .storageModeShared)!
        let x = dev.makeBuffer(length: t * d * 2, options: .storageModeShared)!
        let wp = w.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<(vocab * d / 2) { wp[i] = nextByte() }
        let sp = s.contents().assumingMemoryBound(to: UInt16.self)
        let bp = b.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<(vocab * groups) {
            sp[i] = Quantization.bf16Bits(0.002 + 0.0005 * Float(i % 11))
            bp[i] = Quantization.bf16Bits(-0.02 + 0.004 * Float(i % 7))
        }
        let xp = x.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<(t * d) { xp[i] = Float16(Float(nextByte()) / 64.0 - 2.0) }

        let yMulti = dev.makeBuffer(length: t * vocab * 2, options: .storageModeShared)!
        let ySingle = dev.makeBuffer(length: t * vocab * 2, options: .storageModeShared)!
        let head = try LMHeadGreedyMultiX(context: ctx, maxVocab: vocab)
        let gemv = try DequantInt4GEMV(context: ctx)

        let cb = ctx.queue.makeCommandBuffer()!
        head.encodeLogits(commandBuffer: cb,
                          xNormed: x,
                          weights: w, scales: s, biases: b,
                          outLogits: yMulti,
                          d: d, vocab: vocab, tokens: t)
        for r in 0..<t {
            gemv.encode(commandBuffer: cb,
                        weights: w, scales: s, biases: b,
                        x: x, xOffset: r * d * 2,
                        y: ySingle, yOffset: r * vocab * 2,
                        m: UInt32(vocab), n: UInt32(d))
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let a = UnsafeBufferPointer(start: yMulti.contents().bindMemory(to: UInt16.self, capacity: t * vocab),
                                    count: t * vocab)
        let c = UnsafeBufferPointer(start: ySingle.contents().bindMemory(to: UInt16.self, capacity: t * vocab),
                                    count: t * vocab)
        #expect(Array(a) == Array(c))
    }

    @Test("SampleTopK64 matches reference Sample for arbitrary topK in 1...64",
          arguments: [1, 20, 40, 64])
    func topK64MatchesReferenceForVariableTopK(topK: Int) throws {
        let vocab = 16_384
        let ctx = try MetalContext()
        let dev = ctx.device
        let ref = try Sample(context: ctx)
        let fast = try SampleTopK64(context: ctx, vocab: vocab)
        let probs = dev.makeBuffer(length: vocab * 2, options: .storageModeShared)!
        let refOut = dev.makeBuffer(length: 4, options: .storageModeShared)!
        let fastOut = dev.makeBuffer(length: 4, options: .storageModeShared)!

        let pp = probs.contents().bindMemory(to: Float16.self, capacity: vocab)
        for i in 0..<vocab {
            let mixed = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
            pp[i] = Float16(Float(UInt32(mixed >> 40) + 1) * (1.0 / 16_777_217.0))
        }

        for topP: Float in [0.95, 1.0] {
            for seed: UInt64 in [1, 42, 0x9E37_79B9_7F4A_7C15] {
                let cb = ctx.queue.makeCommandBuffer()!
                ref.encode(commandBuffer: cb, probs: probs, outToken: refOut,
                           v: UInt32(vocab), temperature: 0.2,
                           topK: UInt32(topK), topP: topP, seed: seed)
                fast.encode(commandBuffer: cb, probs: probs, outToken: fastOut,
                            temperature: 0.2, topK: UInt32(topK), topP: topP, seed: seed)
                cb.commit()
                cb.waitUntilCompleted()
                #expect(cb.error == nil)
                #expect(fastOut.contents().load(as: UInt32.self) == refOut.contents().load(as: UInt32.self))
            }
        }
    }

    @Test("Sampler.sampleBatch with repetitionPenalty matches sequential Sampler.sample")
    func sampleBatchMatchesSequentialWithRepetitionPenalty() throws {
        let vocab = 2048
        let count = 4
        let ctx = try MetalContext()
        let dev = ctx.device
        let sampler = try Sampler(context: ctx, vocab: vocab, logitSoftcap: 0)

        let batchLogits = dev.makeBuffer(length: count * vocab * 2, options: .storageModeShared)!
        let singleLogits = dev.makeBuffer(length: vocab * 2, options: .storageModeShared)!
        let probs = dev.makeBuffer(length: vocab * 2, options: .storageModeShared)!
        let batchOut = dev.makeBuffer(length: count * 4, options: .storageModeShared)!
        let singleOut = dev.makeBuffer(length: 4, options: .storageModeShared)!

        var rawLogits = [UInt16](repeating: 0, count: count * vocab)
        for r in 0..<count {
            for i in 0..<vocab {
                let val = Float(((i * 37 + r * 19) % 101) - 50) * 0.25
                rawLogits[r * vocab + i] = Float16(val).bitPattern
            }
        }
        rawLogits.withUnsafeBytes { ptr in
            batchLogits.contents().copyMemory(from: ptr.baseAddress!, byteCount: ptr.count)
        }

        let baseHistory: [Int32] = [12, 45, 99, 300]
        let drafts: [Int32] = [101, 202, 303]
        let config = GenerationConfig(maxNewTokens: 16, temperature: 0.2, topK: 40, topP: 0.95,
                                      repetitionPenalty: 1.15, seed: 1234)
        let rowStrideBytes = vocab * 2

        let cbBatch = ctx.queue.makeCommandBuffer()!
        sampler.sampleBatch(commandBuffer: cbBatch,
                            logitsBatch: batchLogits,
                            rowStrideBytes: rowStrideBytes,
                            probs: probs,
                            baseHistory: baseHistory,
                            draftTokens: drafts,
                            config: config,
                            basePosition: 10,
                            outTokens: batchOut,
                            count: count)
        cbBatch.commit()
        cbBatch.waitUntilCompleted()
        #expect(cbBatch.error == nil)

        let batchTokens = (0..<count).map {
            batchOut.contents().advanced(by: $0 * 4).load(as: UInt32.self)
        }

        var seqTokens: [UInt32] = []
        for r in 0..<count {
            rawLogits.withUnsafeBytes { ptr in
                singleLogits.contents().copyMemory(from: ptr.baseAddress!.advanced(by: r * rowStrideBytes),
                                                   byteCount: rowStrideBytes)
            }
            let cbSingle = ctx.queue.makeCommandBuffer()!
            _ = sampler.sample(commandBuffer: cbSingle,
                               logits: singleLogits,
                               probs: probs,
                               history: baseHistory + drafts.prefix(r),
                               config: config,
                               position: 10 + r,
                               outToken: singleOut)
            cbSingle.commit()
            cbSingle.waitUntilCompleted()
            #expect(cbSingle.error == nil)
            seqTokens.append(singleOut.contents().load(as: UInt32.self))
        }

        #expect(batchTokens == seqTokens)
    }
}
