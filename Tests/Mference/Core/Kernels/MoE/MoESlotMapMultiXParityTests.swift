import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

@Suite struct MoESlotMapMultiXParityTests {
    private static let dimension = 256
    private static let intermediate = 128

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    private enum SplitScheduleMode {
        case none
        case mixed
        case allHit
        case allMiss
    }

    @Test("batched slot-map MultiX matches single-token encodeSlotMapGuardedFFN bit for bit",
          arguments: [1, 2, 3, 4])
    func batchedMatchesPerTokenBitExact(t: Int) throws {
        try checkParity(t: t, disjoint: false, allHitValue: 1, silu: true)
    }

    @Test("batched slot-map MultiX with disjoint experts and GeLU matches single-token bit for bit")
    func disjointAndGeluMatchBitExact() throws {
        try checkParity(t: 4, disjoint: true, allHitValue: 1, silu: false)
    }

    @Test("batched slot-map MultiX no-ops cleanly when all_hit == 0")
    func guardedNoOpLeavesOutputsUntouched() throws {
        try checkParity(t: 3, disjoint: false, allHitValue: 0, silu: true)
    }

    @Test("split hit/miss MultiX matches single-token encodeSlotMapGuardedFFN bit for bit",
          arguments: [2, 3, 4])
    func splitHitMissMatchesSingleTokenBitExact(t: Int) throws {
        try checkParity(t: t, disjoint: false, allHitValue: 1, silu: true, splitMode: .mixed)
    }

    @Test("split hit/miss MultiX handles GeLU, disjoint, all-hit, and all-miss bit-exact")
    func splitHitMissGeluAndEdgeCasesBitExact() throws {
        try checkParity(t: 4, disjoint: true, allHitValue: 1, silu: false, splitMode: .mixed)
        try checkParity(t: 3, disjoint: false, allHitValue: 1, silu: true, splitMode: .allHit)
        try checkParity(t: 3, disjoint: false, allHitValue: 1, silu: true, splitMode: .allMiss)
    }

    private func checkParity(
        t: Int,
        disjoint: Bool,
        allHitValue: UInt32,
        silu: Bool,
        splitMode: SplitScheduleMode = .none
    ) throws {
        let topK = 8
        let poolCount = 32
        var rng = SeedTree(0x5107_BA7C).key("t\(t)-d\(disjoint)-s\(silu)")

        let blobs: [RoutedBlob] = (0..<poolCount).map { _ in
            let gate = (0..<Self.intermediate).map { _ in
                (0..<Self.dimension).map { _ in rng.uniform(-0.35, 0.35) }
            }
            let up = (0..<Self.intermediate).map { _ in
                (0..<Self.dimension).map { _ in rng.uniform(-0.35, 0.35) }
            }
            let down = (0..<Self.dimension).map { _ in
                (0..<Self.intermediate).map { _ in rng.uniform(-0.35, 0.35) }
            }
            return Self.makeBlob(gate: gate, up: up, down: down)
        }

        let slotStride = (blobs[0].bytes.count + 63) & ~63
        let context = try MetalContext()
        let dev = context.device
        let kernel = try MoE(context: context,
                             siluActivation: silu,
                             specializedD: UInt32(Self.dimension),
                             specializedF: UInt32(Self.intermediate),
                             specializedNumExperts: 256,
                             specializedTopK: UInt32(topK))

        let slab = dev.makeBuffer(length: poolCount * slotStride, options: .storageModeShared)!
        for i in 0..<poolCount {
            blobs[i].bytes.withUnsafeBytes { raw in
                slab.contents().advanced(by: i * slotStride)
                    .copyMemory(from: raw.baseAddress!, byteCount: raw.count)
            }
        }

        var slotOffsetsArr = [UInt32](repeating: 0, count: t * topK)
        for r in 0..<t {
            for k in 0..<topK {
                let slot = disjoint ? (r * topK + k) : ((r * 2 + k) % poolCount)
                slotOffsetsArr[r * topK + k] = UInt32(slot * slotStride)
            }
        }
        let slotOffsetsBuf = dev.makeBuffer(
            bytes: slotOffsetsArr,
            length: slotOffsetsArr.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared)!
        var allHitRaw = allHitValue
        let allHitBuf = dev.makeBuffer(bytes: &allHitRaw, length: 4, options: .storageModeShared)!

        let xVals = (0..<(t * Self.dimension)).map { _ in Float(Float16(rng.uniform(-0.6, 0.6))) }
        let resVals = (0..<(t * Self.dimension)).map { _ in Float(Float16(rng.uniform(-0.4, 0.4))) }
        let hidInit = (0..<(t * Self.dimension)).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let wVals = (0..<(t * topK)).map { Float(Float16(0.05 + Float($0 % topK) * 0.02)) }

        let xBuf = Fp16Buffer.make(dev, values: xVals)!
        let resBuf = Fp16Buffer.make(dev, values: resVals)!
        let wBuf = Fp16Buffer.make(dev, values: wVals)!

        let refActs = Fp16Buffer.make(dev, count: t * topK * Self.intermediate)!
        let refY = Fp16Buffer.make(dev, count: t * Self.dimension)!
        let refHidden = Fp16Buffer.make(dev, values: hidInit)!

        let batActs = Fp16Buffer.make(dev, count: t * topK * Self.intermediate)!
        let batY = Fp16Buffer.make(dev, count: t * Self.dimension)!
        let batHidden = Fp16Buffer.make(dev, values: hidInit)!

        let refCB = context.queue.makeCommandBuffer()!
        for r in 0..<t {
            kernel.encodeSlotMapGuardedFFN(
                commandBuffer: refCB,
                slab: slab,
                slotOffsets: slotOffsetsBuf,
                slotOffsetsOffset: r * topK * MemoryLayout<UInt32>.stride,
                allHit: allHitBuf,
                allHitOffset: 0,
                routedOffsets: blobs[0].offsets,
                x: xBuf,
                xOffset: r * Self.dimension * 2,
                acts: refActs,
                actsOffset: r * topK * Self.intermediate * 2,
                routingWeights: wBuf,
                routingWeightsOffset: r * topK * 2,
                residual: resBuf,
                residualOffset: r * Self.dimension * 2,
                y: refY,
                yOffset: r * Self.dimension * 2,
                hidden: refHidden,
                hiddenOffset: r * Self.dimension * 2,
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(topK))
        }
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let batCB: MTLCommandBuffer
        if splitMode == .none {
            let singleBatCB = context.queue.makeCommandBuffer()!
            kernel.encodeSlotMapGuardedFFNBatch(
                commandBuffer: singleBatCB,
                slab: slab,
                slotOffsets: slotOffsetsBuf,
                allHit: allHitBuf,
                routedOffsets: blobs[0].offsets,
                x: xBuf,
                acts: batActs,
                routingWeights: wBuf,
                residual: resBuf,
                y: batY,
                hidden: batHidden,
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(topK),
                queryCount: t)
            singleBatCB.commit()
            batCB = singleBatCB
        } else {
            let hitSchedBuf = dev.makeBuffer(
                length: MoE.slotmapScheduleStride, options: .storageModeShared)!
            let missSchedBuf = dev.makeBuffer(
                length: MoE.slotmapScheduleStride, options: .storageModeShared)!
            let (hitUnique, missUnique) = MoE.buildSplitSlotMapSchedules(
                queryCount: t,
                topK: topK,
                hitSchedulePtr: hitSchedBuf.contents(),
                missSchedulePtr: missSchedBuf.contents()
            ) { r, k in
                let offset = slotOffsetsArr[r * topK + k]
                let slot = Int(offset) / slotStride
                let isMiss: Bool
                switch splitMode {
                case .mixed:
                    isMiss = (slot % 2) == 1
                case .allHit:
                    isMiss = false
                case .allMiss:
                    isMiss = true
                case .none:
                    isMiss = false
                }
                return (slotOffset: offset, isMiss: isMiss)
            }
            if splitMode == .mixed {
                #expect(hitUnique > 0 && missUnique > 0)
            }
            if hitUnique > 0 && missUnique > 0 {
                let hitCB = context.queue.makeCommandBuffer()!
                kernel.encodeSlotMapScheduledPhase1And2(
                    commandBuffer: hitCB,
                    slab: slab,
                    schedule: hitSchedBuf,
                    uniqueCount: hitUnique,
                    allHit: allHitBuf,
                    routedOffsets: blobs[0].offsets,
                    x: xBuf,
                    acts: batActs,
                    routingWeights: wBuf,
                    d: UInt32(Self.dimension),
                    f: UInt32(Self.intermediate),
                    topK: UInt32(topK),
                    queryCount: t)
                hitCB.commit()
            }
            let fillCB = context.queue.makeCommandBuffer()!
            let useMissSchedule = missUnique > 0
            kernel.encodeSlotMapScheduledMissAndReduce(
                commandBuffer: fillCB,
                slab: slab,
                schedule: useMissSchedule ? missSchedBuf : hitSchedBuf,
                uniqueCount: useMissSchedule ? missUnique : hitUnique,
                allHit: allHitBuf,
                routedOffsets: blobs[0].offsets,
                x: xBuf,
                acts: batActs,
                routingWeights: wBuf,
                residual: resBuf,
                y: batY,
                hidden: batHidden,
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(topK),
                queryCount: t)
            fillCB.commit()
            batCB = fillCB
        }
        batCB.waitUntilCompleted()
        #expect(batCB.error == nil)

        let refActsBits = UnsafeBufferPointer(
            start: refActs.contents().bindMemory(to: UInt16.self, capacity: t * topK * Self.intermediate),
            count: t * topK * Self.intermediate)
        let batActsBits = UnsafeBufferPointer(
            start: batActs.contents().bindMemory(to: UInt16.self, capacity: t * topK * Self.intermediate),
            count: t * topK * Self.intermediate)
        #expect(Array(batActsBits) == Array(refActsBits), "acts bit mismatch for t=\(t)")

        let refYBits = UnsafeBufferPointer(
            start: refY.contents().bindMemory(to: UInt16.self, capacity: t * Self.dimension),
            count: t * Self.dimension)
        let batYBits = UnsafeBufferPointer(
            start: batY.contents().bindMemory(to: UInt16.self, capacity: t * Self.dimension),
            count: t * Self.dimension)
        #expect(Array(batYBits) == Array(refYBits), "y bit mismatch for t=\(t)")

        let refHidBits = UnsafeBufferPointer(
            start: refHidden.contents().bindMemory(to: UInt16.self, capacity: t * Self.dimension),
            count: t * Self.dimension)
        let batHidBits = UnsafeBufferPointer(
            start: batHidden.contents().bindMemory(to: UInt16.self, capacity: t * Self.dimension),
            count: t * Self.dimension)
        #expect(Array(batHidBits) == Array(refHidBits), "hidden bit mismatch for t=\(t)")
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]]) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            return (quantized.flatMap(\.packed),
                    quantized.flatMap(\.scales),
                    quantized.flatMap(\.biases))
        }
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateValues = packed(gate)
        let upValues = packed(up)
        let downValues = packed(down)
        let gateW = UInt32(bytes.count); append(gateValues.weights)
        let gateS = UInt32(bytes.count); append(gateValues.scales)
        let gateB = UInt32(bytes.count); append(gateValues.biases)
        let upW = UInt32(bytes.count); append(upValues.weights)
        let upS = UInt32(bytes.count); append(upValues.scales)
        let upB = UInt32(bytes.count); append(upValues.biases)
        let downW = UInt32(bytes.count); append(downValues.weights)
        let downS = UInt32(bytes.count); append(downValues.scales)
        let downB = UInt32(bytes.count); append(downValues.biases)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
