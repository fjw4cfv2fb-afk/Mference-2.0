import Accelerate
import Darwin
import Foundation
import Metal
import Synchronization

/// Ephemeral Pure-Swift + Accelerate AMX / Metal 4 Vision Transformer encoder
/// for Qwen 3.6 (`vision_weights.bin`).
///
/// Maps `vision_weights.bin` read-only on demand when an uncached image is
/// encountered, executes the 27-layer ViT + `2×2` spatial merger in FP32 using
/// Metal 4 (`matmul2d` Neural Accelerator) when available or Apple's AMX
/// matrix engine (`cblas_sgemm`) otherwise, caches the compact `[N_vis, 2048]`
/// FP16 embeddings in `Qwen36VisionImageStore`, and immediately `munmap`s the
/// weight file and frees all transient scratch buffers before the 40-layer MoE
/// trunk prefill begins.
public final class Qwen36VisionEncoder: @unchecked Sendable {
    public static let depth = 27
    public static let hiddenSize = 1152
    public static let intermediateSize = 4304
    public static let numHeads = 16
    public static let headDim = hiddenSize / numHeads // 72
    public static let patchDim = 1536 // 2 * 16 * 16 * 3
    public static let numPosEmbeddings = 2304
    public static let numGridPerSide = 48
    public static let mergerHiddenSize = hiddenSize * 4 // 4608
    public static let outHiddenSize = 2048
    public static let layerNormEps: Float = 1e-6
    public static let rotaryTheta: Float = 10_000.0

    public let weightsURL: URL
    private let metalContext: MetalContext?
    private var cachedPipelines: MetalPipelines?
    private var didProbePipelines = false
    private let encodeLock = Mutex(())

    struct TensorSpan {
        let offset: Int
        let byteLength: Int
        let shape: [Int]
        let dtype: String
    }

    struct MetalPipelines {
        let gemm: MTLComputePipelineState
        let layerNorm: MTLComputePipelineState
        let rope: MTLComputePipelineState
        let attention: MTLComputePipelineState
    }

    public init(weightsURL: URL, context: MetalContext? = nil) {
        self.weightsURL = weightsURL
        self.metalContext = context
    }

    public func ensureEncoded(imageID: Int) throws -> [Float16] {
        if let cached = Qwen36VisionImageStore.shared.cachedEmbeddings(for: imageID) {
            return cached
        }
        return try encodeLock.withLock { _ in
            if let cached = Qwen36VisionImageStore.shared.cachedEmbeddings(for: imageID) {
                return cached
            }
            guard let registered = Qwen36VisionImageStore.shared.image(for: imageID) else {
                throw Qwen36VisionImageError.missingRegisteredImage(imageID: imageID, sha256: nil)
            }
            let embeddings = try runEncoder(patchData: registered.patchData)
            Qwen36VisionImageStore.shared.storeEmbeddings(embeddings, for: imageID)
            return embeddings
        }
    }

    static func makeMetalPipelines(context: MetalContext) -> MetalPipelines? {
        guard ProcessInfo.processInfo.environment["MFERENCE_VISION_METAL"] != "0",
              context.device.supportsApple10TensorOps else {
            return nil
        }
        guard let lib = try? MetalContext.moduleLibrary(device: context.device, module: "vision"),
              let gemmFn = lib.makeFunction(name: "mpp_vision_bf16_gemm_f32"),
              let lnFn = lib.makeFunction(name: "vision_layernorm_f32"),
              let ropeFn = lib.makeFunction(name: "vision_rope_2d_f32"),
              let attnFn = lib.makeFunction(name: "vision_attention_f32"),
              let gemmPSO = try? context.device.makeComputePipelineState(function: gemmFn),
              let lnPSO = try? context.device.makeComputePipelineState(function: lnFn),
              let ropePSO = try? context.device.makeComputePipelineState(function: ropeFn),
              let attnPSO = try? context.device.makeComputePipelineState(function: attnFn) else {
            return nil
        }
        return MetalPipelines(gemm: gemmPSO, layerNorm: lnPSO, rope: ropePSO, attention: attnPSO)
    }

    private func metalPipelinesIfEnabled() -> MetalPipelines? {
        if ProcessInfo.processInfo.environment["MFERENCE_VISION_METAL"] == "0" {
            return nil
        }
        if didProbePipelines {
            return cachedPipelines
        }
        didProbePipelines = true
        guard let metalContext else { return nil }
        cachedPipelines = Self.makeMetalPipelines(context: metalContext)
        return cachedPipelines
    }

    func runEncoder(patchData: Qwen36VisionPatchData, forceCPU: Bool = false) throws -> [Float16] {
        let path = weightsURL.path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw Qwen36VisionImageError.visionSidecarMissing(path)
        }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)

        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size > 8 else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("cannot stat \(path)")
        }
        let fileSize = Int(st.st_size)
        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != MAP_FAILED else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("mmap failed for \(path) (errno \(errno))")
        }
        defer {
            madvise(mapped, fileSize, MADV_DONTNEED)
            munmap(mapped, fileSize)
        }
        madvise(mapped, fileSize, MADV_SEQUENTIAL)

        let headerLenU64 = UInt64(littleEndian: mapped.loadUnaligned(as: UInt64.self))
        guard headerLenU64 > 0, headerLenU64 <= 32 * 1024 * 1024,
              8 + Int(headerLenU64) <= fileSize else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("invalid safetensors header length")
        }
        let headerLen = Int(headerLenU64)
        let headerData = Data(bytesNoCopy: mapped.advanced(by: 8), count: headerLen, deallocator: .none)
        guard let rawHeader = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("invalid JSON header")
        }

        let payloadOffset = 8 + headerLen
        let payloadLimit = fileSize - payloadOffset
        var spans: [String: TensorSpan] = [:]
        spans.reserveCapacity(rawHeader.count)
        for (name, val) in rawHeader {
            if name == "__metadata__" { continue }
            guard let dict = val as? [String: Any],
                  let dtype = dict["dtype"] as? String,
                  let shapeNums = dict["shape"] as? [NSNumber],
                  let offNums = dict["data_offsets"] as? [NSNumber],
                  offNums.count == 2 else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("malformed tensor entry \(name)")
            }
            let s = offNums[0].intValue
            let e = offNums[1].intValue
            guard s >= 0, e >= s, e <= payloadLimit else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("out-of-bounds offset for \(name)")
            }
            spans[name] = TensorSpan(
                offset: s,
                byteLength: e - s,
                shape: shapeNums.map(\.intValue),
                dtype: dtype)
        }

        let dataBase = mapped.advanced(by: payloadOffset)
        let logPhases = ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1"

        if !forceCPU, let ctx = metalContext, let pipelines = metalPipelinesIfEnabled() {
            let t0 = logPhases ? DispatchTime.now().uptimeNanoseconds : 0
            let out = try runEncoderMetal(
                patchData: patchData,
                spans: spans,
                fd: fd,
                payloadOffset: payloadOffset,
                dataBase: dataBase,
                context: ctx,
                pipelines: pipelines)
            if logPhases {
                let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                let msg = String(format: "[vision]: path=metal4 patches=%d tokens=%d elapsed=%.1fms\n",
                                 patchData.numPatches, patchData.numVisualTokens, elapsedMs)
                FileHandle.standardError.write(Data(msg.utf8))
            }
            return out
        }

        let t0 = logPhases ? DispatchTime.now().uptimeNanoseconds : 0
        let out = try runEncoderCPU(patchData: patchData, spans: spans, dataBase: dataBase)
        if logPhases {
            let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            let msg = String(format: "[vision]: path=cpu-amx patches=%d tokens=%d elapsed=%.1fms\n",
                             patchData.numPatches, patchData.numVisualTokens, elapsedMs)
            FileHandle.standardError.write(Data(msg.utf8))
        }
        return out
    }

    private func runEncoderCPU(
        patchData: Qwen36VisionPatchData,
        spans: [String: TensorSpan],
        dataBase: UnsafeMutableRawPointer
    ) throws -> [Float16] {
        func requireSpan(_ name: String) throws -> TensorSpan {
            guard let span = spans[name] else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("missing tensor \(name)")
            }
            guard span.dtype == "BF16" else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("expected BF16 for \(name), got \(span.dtype)")
            }
            return span
        }

        // Reusable FP32 weight conversion buffer (~19.8 MB covers every ViT block tensor;
        // merger.linear_fc1 grows it transiently only at the end).
        var weightBuf = [Float](repeating: 0, count: Self.intermediateSize * Self.hiddenSize)
        var biasBuf = [Float](repeating: 0, count: Self.mergerHiddenSize)

        @inline(__always)
        func loadBF16(_ span: TensorSpan, into dst: inout [Float]) throws {
            let count = span.byteLength / 2
            guard span.byteLength % 2 == 0 else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("odd byte length \(span.byteLength)")
            }
            if dst.count < count {
                dst = [Float](repeating: 0, count: count)
            }
            let srcPtr = dataBase.advanced(by: span.offset).assumingMemoryBound(to: UInt16.self)
            dst.withUnsafeMutableBufferPointer { out in
                let dstPtr = out.baseAddress!
                for i in 0..<count {
                    dstPtr[i] = Float(bitPattern: UInt32(srcPtr[i]) << 16)
                }
            }
        }

        func linear(
            x: [Float],
            rows n: Int,
            inCols k: Int,
            weightName: String,
            biasName: String?,
            outCols m: Int,
            into y: inout [Float]
        ) throws {
            let wSpan = try requireSpan(weightName)
            guard wSpan.byteLength == m * k * 2 else {
                throw Qwen36VisionImageError.visionWeightsCorrupt(
                    "\(weightName) byte length \(wSpan.byteLength) != expected \(m * k * 2)")
            }
            try loadBF16(wSpan, into: &weightBuf)
            if y.count < n * m {
                y = [Float](repeating: 0, count: n * m)
            }
            x.withUnsafeBufferPointer { xPtr in
                weightBuf.withUnsafeBufferPointer { wPtr in
                    y.withUnsafeMutableBufferPointer { yPtr in
                        cblas_sgemm(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasTrans,
                            Int32(n),
                            Int32(m),
                            Int32(k),
                            1.0,
                            xPtr.baseAddress!,
                            Int32(k),
                            wPtr.baseAddress!,
                            Int32(k),
                            0.0,
                            yPtr.baseAddress!,
                            Int32(m))
                    }
                }
            }
            if let biasName {
                let bSpan = try requireSpan(biasName)
                guard bSpan.byteLength == m * 2 else {
                    throw Qwen36VisionImageError.visionWeightsCorrupt(
                        "\(biasName) byte length \(bSpan.byteLength) != expected \(m * 2)")
                }
                try loadBF16(bSpan, into: &biasBuf)
                biasBuf.withUnsafeBufferPointer { bPtr in
                    y.withUnsafeMutableBufferPointer { yPtr in
                        for r in 0..<n {
                            vDSP_vadd(
                                yPtr.baseAddress! + r * m, 1,
                                bPtr.baseAddress!, 1,
                                yPtr.baseAddress! + r * m, 1,
                                vDSP_Length(m))
                        }
                    }
                }
            }
        }

        let nPatches = patchData.numPatches
        let d = Self.hiddenSize
        let mlpDim = Self.intermediateSize
        let numHeads = Self.numHeads
        let headDim = Self.headDim

        // 1. Patch embedding: [N_patches, 1536] -> [N_patches, 1152]
        var hidden = [Float](repeating: 0, count: nPatches * d)
        try linear(
            x: patchData.patches,
            rows: nPatches,
            inCols: Self.patchDim,
            weightName: "vision_tower.patch_embed.proj.weight",
            biasName: "vision_tower.patch_embed.proj.bias",
            outCols: d,
            into: &hidden)

        // 2. Interpolated 2D position embeddings (48x48 bilinear lookup in (bh, bw, ih, iw) order)
        let posSpan = try requireSpan("vision_tower.pos_embed.weight")
        guard posSpan.byteLength == Self.numPosEmbeddings * d * 2 else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("unexpected pos_embed byte length")
        }
        try loadBF16(posSpan, into: &weightBuf)
        Self.addInterpolatedPosEmbed(
            posTable: weightBuf,
            gridH: patchData.gridH,
            gridW: patchData.gridW,
            into: &hidden)

        // 3. Precompute 2D Vision RoPE cos/sin tables ([N_patches, 36])
        let (rotCos, rotSin) = Self.buildRotaryTables(gridH: patchData.gridH, gridW: patchData.gridW)

        // Reusable per-block activation scratch
        var normBuf = [Float](repeating: 0, count: nPatches * d)
        var qkvBuf = [Float](repeating: 0, count: nPatches * 3 * d)
        var attnOutBuf = [Float](repeating: 0, count: nPatches * d)
        var projOutBuf = [Float](repeating: 0, count: nPatches * d)
        var mlpMidBuf = [Float](repeating: 0, count: nPatches * mlpDim)
        var normW = [Float](repeating: 0, count: d)
        var normB = [Float](repeating: 0, count: d)

        // 4. 27 ViT blocks
        for blockIdx in 0..<Self.depth {
            let pfx = "vision_tower.blocks.\(blockIdx)"
            try loadBF16(try requireSpan("\(pfx).norm1.weight"), into: &normW)
            try loadBF16(try requireSpan("\(pfx).norm1.bias"), into: &normB)
            Self.layerNorm(x: hidden, weight: normW, bias: normB, rows: nPatches, dim: d, eps: Self.layerNormEps, into: &normBuf)

            try linear(
                x: normBuf,
                rows: nPatches,
                inCols: d,
                weightName: "\(pfx).attn.qkv.weight",
                biasName: "\(pfx).attn.qkv.bias",
                outCols: 3 * d,
                into: &qkvBuf)

            Self.applyVisionRoPEAndAttention(
                qkv: &qkvBuf,
                cosTable: rotCos,
                sinTable: rotSin,
                seqLen: nPatches,
                numHeads: numHeads,
                headDim: headDim,
                into: &attnOutBuf)

            try linear(
                x: attnOutBuf,
                rows: nPatches,
                inCols: d,
                weightName: "\(pfx).attn.proj.weight",
                biasName: "\(pfx).attn.proj.bias",
                outCols: d,
                into: &projOutBuf)

            // hidden += projOut
            hidden.withUnsafeMutableBufferPointer { hPtr in
                projOutBuf.withUnsafeBufferPointer { oPtr in
                    vDSP_vadd(hPtr.baseAddress!, 1, oPtr.baseAddress!, 1, hPtr.baseAddress!, 1, vDSP_Length(nPatches * d))
                }
            }

            try loadBF16(try requireSpan("\(pfx).norm2.weight"), into: &normW)
            try loadBF16(try requireSpan("\(pfx).norm2.bias"), into: &normB)
            Self.layerNorm(x: hidden, weight: normW, bias: normB, rows: nPatches, dim: d, eps: Self.layerNormEps, into: &normBuf)

            try linear(
                x: normBuf,
                rows: nPatches,
                inCols: d,
                weightName: "\(pfx).mlp.linear_fc1.weight",
                biasName: "\(pfx).mlp.linear_fc1.bias",
                outCols: mlpDim,
                into: &mlpMidBuf)

            Self.geluTanhInPlace(&mlpMidBuf, count: nPatches * mlpDim)

            try linear(
                x: mlpMidBuf,
                rows: nPatches,
                inCols: mlpDim,
                weightName: "\(pfx).mlp.linear_fc2.weight",
                biasName: "\(pfx).mlp.linear_fc2.bias",
                outCols: d,
                into: &projOutBuf)

            // hidden += mlpOut
            hidden.withUnsafeMutableBufferPointer { hPtr in
                projOutBuf.withUnsafeBufferPointer { oPtr in
                    vDSP_vadd(hPtr.baseAddress!, 1, oPtr.baseAddress!, 1, hPtr.baseAddress!, 1, vDSP_Length(nPatches * d))
                }
            }
        }

        // Release large attention/MLP scratch before allocating merger scratch.
        qkvBuf = []
        attnOutBuf = []
        projOutBuf = []
        mlpMidBuf = []

        // 5. PatchMerger: LayerNorm([N_patches, 1152]) -> reshape [N_vis, 4608] -> fc1 -> exact GELU -> fc2 [N_vis, 2048]
        try loadBF16(try requireSpan("vision_tower.merger.norm.weight"), into: &normW)
        try loadBF16(try requireSpan("vision_tower.merger.norm.bias"), into: &normB)
        Self.layerNorm(x: hidden, weight: normW, bias: normB, rows: nPatches, dim: d, eps: Self.layerNormEps, into: &normBuf)
        hidden = []

        let nVis = patchData.numVisualTokens
        let mDim = Self.mergerHiddenSize
        var mergerMid = [Float](repeating: 0, count: nVis * mDim)
        try linear(
            x: normBuf,
            rows: nVis,
            inCols: mDim,
            weightName: "vision_tower.merger.linear_fc1.weight",
            biasName: "vision_tower.merger.linear_fc1.bias",
            outCols: mDim,
            into: &mergerMid)
        normBuf = []

        Self.geluExactInPlace(&mergerMid, count: nVis * mDim)

        var outF32 = [Float](repeating: 0, count: nVis * Self.outHiddenSize)
        try linear(
            x: mergerMid,
            rows: nVis,
            inCols: mDim,
            weightName: "vision_tower.merger.linear_fc2.weight",
            biasName: "vision_tower.merger.linear_fc2.bias",
            outCols: Self.outHiddenSize,
            into: &outF32)

        for v in outF32 where !v.isFinite {
            throw Qwen36VisionImageError.visionWeightsCorrupt("vision encoder produced non-finite activation")
        }

        var outF16 = [Float16](repeating: 0, count: nVis * Self.outHiddenSize)
        outF16.withUnsafeMutableBufferPointer { dst in
            outF32.withUnsafeBufferPointer { src in
                for i in 0..<(nVis * Self.outHiddenSize) {
                    dst[i] = Float16(src[i])
                }
            }
        }
        return outF16
    }

    // MARK: - Metal 4 Ephemeral Vision Encoder

    private struct BlockSpans {
        let norm1W, norm1B: TensorSpan
        let qkvW, qkvB: TensorSpan
        let projW, projB: TensorSpan
        let norm2W, norm2B: TensorSpan
        let fc1W, fc1B: TensorSpan
        let fc2W, fc2B: TensorSpan
    }

    private struct StagedBlockOffsets {
        let norm1W, norm1B: Int
        let qkvW, qkvB: Int
        let projW, projB: Int
        let norm2W, norm2B: Int
        let fc1W, fc1B: Int
        let fc2W, fc2B: Int
    }

    private static func preadExact(fd: Int32, dst: UnsafeMutableRawPointer, count: Int, fileOffset: Int) -> Bool {
        var total = 0
        while total < count {
            let n = Darwin.pread(fd, dst.advanced(by: total), count - total, off_t(fileOffset + total))
            if n > 0 {
                total += n
            } else if n < 0 && errno == EINTR {
                continue
            } else {
                return false
            }
        }
        return true
    }

    private func runEncoderMetal(
        patchData: Qwen36VisionPatchData,
        spans: [String: TensorSpan],
        fd: Int32,
        payloadOffset: Int,
        dataBase: UnsafeMutableRawPointer,
        context: MetalContext,
        pipelines: MetalPipelines
    ) throws -> [Float16] {
        let nPatches = patchData.numPatches
        let nVis = patchData.numVisualTokens
        let d = Self.hiddenSize
        let mlpDim = Self.intermediateSize
        let numHeads = Self.numHeads
        let headDim = Self.headDim
        let f32Size = MemoryLayout<Float>.stride

        guard nPatches > 0, nVis > 0, nPatches % 4 == 0,
              patchData.patches.count == nPatches * Self.patchDim else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("invalid patch data dimensions")
        }

        func requireCheckedSpan(_ name: String, expectedBytes: Int) throws -> TensorSpan {
            guard let span = spans[name] else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("missing tensor \(name)")
            }
            guard span.dtype == "BF16" else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("expected BF16 for \(name), got \(span.dtype)")
            }
            guard span.byteLength == expectedBytes else {
                throw Qwen36VisionImageError.visionWeightsCorrupt(
                    "\(name) byte length \(span.byteLength) != expected \(expectedBytes)")
            }
            return span
        }

        func makeSharedBuffer(bytes: Int) throws -> MTLBuffer {
            guard let buf = context.device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("failed to allocate \(bytes)-byte Metal buffer")
            }
            return buf
        }

        @inline(__always)
        func stageTensor(_ span: TensorSpan, into buf: MTLBuffer, cursor: inout Int) throws -> Int {
            let aligned = (cursor + 15) & ~15
            guard aligned + span.byteLength <= buf.length else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("staging buffer overflow for \(span.byteLength) bytes")
            }
            memcpy(buf.contents().advanced(by: aligned), dataBase.advanced(by: span.offset), span.byteLength)
            cursor = aligned + span.byteLength
            return aligned
        }

        func waitAndVerify(_ cb: MTLCommandBuffer) throws {
            cb.waitUntilCompleted()
            if cb.status != .completed || cb.error != nil {
                throw Qwen36VisionImageError.visionWeightsCorrupt(
                    "Metal vision execution failed: \(cb.error?.localizedDescription ?? "status \(cb.status.rawValue)")")
            }
        }

        // Pre-validate all block spans up front before staging.
        var blockSpans: [BlockSpans] = []
        blockSpans.reserveCapacity(Self.depth)
        for blockIdx in 0..<Self.depth {
            let pfx = "vision_tower.blocks.\(blockIdx)"
            blockSpans.append(BlockSpans(
                norm1W: try requireCheckedSpan("\(pfx).norm1.weight", expectedBytes: d * 2),
                norm1B: try requireCheckedSpan("\(pfx).norm1.bias", expectedBytes: d * 2),
                qkvW: try requireCheckedSpan("\(pfx).attn.qkv.weight", expectedBytes: 3 * d * d * 2),
                qkvB: try requireCheckedSpan("\(pfx).attn.qkv.bias", expectedBytes: 3 * d * 2),
                projW: try requireCheckedSpan("\(pfx).attn.proj.weight", expectedBytes: d * d * 2),
                projB: try requireCheckedSpan("\(pfx).attn.proj.bias", expectedBytes: d * 2),
                norm2W: try requireCheckedSpan("\(pfx).norm2.weight", expectedBytes: d * 2),
                norm2B: try requireCheckedSpan("\(pfx).norm2.bias", expectedBytes: d * 2),
                fc1W: try requireCheckedSpan("\(pfx).mlp.linear_fc1.weight", expectedBytes: mlpDim * d * 2),
                fc1B: try requireCheckedSpan("\(pfx).mlp.linear_fc1.bias", expectedBytes: mlpDim * 2),
                fc2W: try requireCheckedSpan("\(pfx).mlp.linear_fc2.weight", expectedBytes: d * mlpDim * 2),
                fc2B: try requireCheckedSpan("\(pfx).mlp.linear_fc2.bias", expectedBytes: d * 2)
            ))
        }

        func stageBlock(_ bs: BlockSpans, into buf: MTLBuffer) throws -> StagedBlockOffsets {
            let allSpans = [
                bs.norm1W, bs.norm1B, bs.qkvW, bs.qkvB,
                bs.projW, bs.projB, bs.norm2W, bs.norm2B,
                bs.fc1W, bs.fc1B, bs.fc2W, bs.fc2B,
            ]
            let minStart = allSpans.map(\.offset).min()!
            let maxEnd = allSpans.map { $0.offset + $0.byteLength }.max()!
            let slabBytes = maxEnd - minStart
            if slabBytes <= buf.length,
               allSpans.allSatisfy({ ($0.offset - minStart) % 4 == 0 }),
               Self.preadExact(fd: fd, dst: buf.contents(), count: slabBytes, fileOffset: payloadOffset + minStart) {
                return StagedBlockOffsets(
                    norm1W: bs.norm1W.offset - minStart,
                    norm1B: bs.norm1B.offset - minStart,
                    qkvW: bs.qkvW.offset - minStart,
                    qkvB: bs.qkvB.offset - minStart,
                    projW: bs.projW.offset - minStart,
                    projB: bs.projB.offset - minStart,
                    norm2W: bs.norm2W.offset - minStart,
                    norm2B: bs.norm2B.offset - minStart,
                    fc1W: bs.fc1W.offset - minStart,
                    fc1B: bs.fc1B.offset - minStart,
                    fc2W: bs.fc2W.offset - minStart,
                    fc2B: bs.fc2B.offset - minStart
                )
            }
            var cursor = 0
            return StagedBlockOffsets(
                norm1W: try stageTensor(bs.norm1W, into: buf, cursor: &cursor),
                norm1B: try stageTensor(bs.norm1B, into: buf, cursor: &cursor),
                qkvW: try stageTensor(bs.qkvW, into: buf, cursor: &cursor),
                qkvB: try stageTensor(bs.qkvB, into: buf, cursor: &cursor),
                projW: try stageTensor(bs.projW, into: buf, cursor: &cursor),
                projB: try stageTensor(bs.projB, into: buf, cursor: &cursor),
                norm2W: try stageTensor(bs.norm2W, into: buf, cursor: &cursor),
                norm2B: try stageTensor(bs.norm2B, into: buf, cursor: &cursor),
                fc1W: try stageTensor(bs.fc1W, into: buf, cursor: &cursor),
                fc1B: try stageTensor(bs.fc1B, into: buf, cursor: &cursor),
                fc2W: try stageTensor(bs.fc2W, into: buf, cursor: &cursor),
                fc2B: try stageTensor(bs.fc2B, into: buf, cursor: &cursor)
            )
        }

        let stageBytes = 32 * 1024 * 1024
        let stageBuf0 = try makeSharedBuffer(bytes: stageBytes)
        var stageBuf1: MTLBuffer? = try makeSharedBuffer(bytes: stageBytes)

        let hiddenBuf = try makeSharedBuffer(bytes: nPatches * d * f32Size)
        let normBuf = try makeSharedBuffer(bytes: nPatches * d * f32Size)

        // 1. Patch embedding on GPU overlapped with pos_embed load + block 0 staging.
        let patchW = try requireCheckedSpan("vision_tower.patch_embed.proj.weight",
                                            expectedBytes: d * Self.patchDim * 2)
        let patchB = try requireCheckedSpan("vision_tower.patch_embed.proj.bias",
                                            expectedBytes: d * 2)
        let posSpan = try requireCheckedSpan("vision_tower.pos_embed.weight",
                                             expectedBytes: Self.numPosEmbeddings * d * 2)

        do {
            let patchBuf = try makeSharedBuffer(bytes: patchData.patches.count * f32Size)
            patchData.patches.withUnsafeBytes { raw in
                if let src = raw.baseAddress {
                    memcpy(patchBuf.contents(), src, raw.count)
                }
            }
            var cursor0 = 0
            let pwOff = try stageTensor(patchW, into: stageBuf0, cursor: &cursor0)
            let pbOff = try stageTensor(patchB, into: stageBuf0, cursor: &cursor0)
            guard let cbPatch = context.queue.makeCommandBuffer() else {
                throw Qwen36VisionImageError.visionWeightsCorrupt("failed to allocate Metal command buffer")
            }
            Self.encodeGemm(
                commandBuffer: cbPatch,
                pipeline: pipelines.gemm,
                x: patchBuf,
                weight: stageBuf0, weightOffset: pwOff,
                bias: stageBuf0, biasOffset: pbOff,
                y: hiddenBuf,
                m: nPatches, n: d, k: Self.patchDim,
                flags: 1)
            cbPatch.commit()

            // Overlap CPU pos_embed BF16->F32 + rotary tables + block 0 staging with patch GEMM.
            let posCount = Self.numPosEmbeddings * d
            var posTable = [Float](repeating: 0, count: posCount)
            let posSrc = dataBase.advanced(by: posSpan.offset).assumingMemoryBound(to: UInt16.self)
            posTable.withUnsafeMutableBufferPointer { dst in
                let p = dst.baseAddress!
                for i in 0..<posCount {
                    p[i] = Float(bitPattern: UInt32(posSrc[i]) << 16)
                }
            }

            let block0Offsets = try stageBlock(blockSpans[0], into: stageBuf1!)
            try waitAndVerify(cbPatch)

            posTable.withUnsafeBufferPointer { posPtr in
                Self.addInterpolatedPosEmbed(
                    posPtr: posPtr.baseAddress!,
                    gridH: patchData.gridH,
                    gridW: patchData.gridW,
                    into: hiddenBuf.contents().assumingMemoryBound(to: Float.self))
            }

            // 2. Run all 27 ViT blocks with double-buffered BF16 weight staging.
            var qkvBuf: MTLBuffer? = try makeSharedBuffer(bytes: nPatches * 3 * d * f32Size)
            var attnOutBuf: MTLBuffer? = try makeSharedBuffer(bytes: nPatches * d * f32Size)
            var mlpMidBuf: MTLBuffer? = try makeSharedBuffer(bytes: nPatches * mlpDim * f32Size)

            let (rotCos, rotSin) = Self.buildRotaryTables(gridH: patchData.gridH, gridW: patchData.gridW)
            var rotCosBuf: MTLBuffer? = try makeSharedBuffer(bytes: rotCos.count * f32Size)
            var rotSinBuf: MTLBuffer? = try makeSharedBuffer(bytes: rotSin.count * f32Size)
            rotCos.withUnsafeBytes { raw in
                if let src = raw.baseAddress { memcpy(rotCosBuf!.contents(), src, raw.count) }
            }
            rotSin.withUnsafeBytes { raw in
                if let src = raw.baseAddress { memcpy(rotSinBuf!.contents(), src, raw.count) }
            }

            let scale = 1.0 / sqrtf(Float(headDim))
            var currentOffsets = block0Offsets
            var prevCB: MTLCommandBuffer? = nil

            for blockIdx in 0..<Self.depth {
                let activeBuf = ((blockIdx & 1) == 0) ? stageBuf1! : stageBuf0
                let nextBuf = ((blockIdx & 1) == 0) ? stageBuf0 : stageBuf1!
                let off = currentOffsets

                guard let cb = context.queue.makeCommandBuffer() else {
                    throw Qwen36VisionImageError.visionWeightsCorrupt("failed to allocate Metal command buffer")
                }

                Self.encodeLayerNorm(
                    commandBuffer: cb,
                    pipeline: pipelines.layerNorm,
                    x: hiddenBuf,
                    weight: activeBuf, weightOffset: off.norm1W,
                    bias: activeBuf, biasOffset: off.norm1B,
                    out: normBuf,
                    rows: nPatches, dim: d, eps: Self.layerNormEps)
                Self.encodeGemm(
                    commandBuffer: cb,
                    pipeline: pipelines.gemm,
                    x: normBuf,
                    weight: activeBuf, weightOffset: off.qkvW,
                    bias: activeBuf, biasOffset: off.qkvB,
                    y: qkvBuf!,
                    m: nPatches, n: 3 * d, k: d,
                    flags: 1)
                Self.encodeRoPE2D(
                    commandBuffer: cb,
                    pipeline: pipelines.rope,
                    qkv: qkvBuf!,
                    cosTable: rotCosBuf!,
                    sinTable: rotSinBuf!,
                    seqLen: nPatches, numHeads: numHeads, headDim: headDim)
                Self.encodeAttention(
                    commandBuffer: cb,
                    pipeline: pipelines.attention,
                    qkv: qkvBuf!,
                    out: attnOutBuf!,
                    seqLen: nPatches, numHeads: numHeads, headDim: headDim, scale: scale)
                Self.encodeGemm(
                    commandBuffer: cb,
                    pipeline: pipelines.gemm,
                    x: attnOutBuf!,
                    weight: activeBuf, weightOffset: off.projW,
                    bias: activeBuf, biasOffset: off.projB,
                    y: hiddenBuf,
                    m: nPatches, n: d, k: d,
                    flags: 1 | 8)
                Self.encodeLayerNorm(
                    commandBuffer: cb,
                    pipeline: pipelines.layerNorm,
                    x: hiddenBuf,
                    weight: activeBuf, weightOffset: off.norm2W,
                    bias: activeBuf, biasOffset: off.norm2B,
                    out: normBuf,
                    rows: nPatches, dim: d, eps: Self.layerNormEps)
                Self.encodeGemm(
                    commandBuffer: cb,
                    pipeline: pipelines.gemm,
                    x: normBuf,
                    weight: activeBuf, weightOffset: off.fc1W,
                    bias: activeBuf, biasOffset: off.fc1B,
                    y: mlpMidBuf!,
                    m: nPatches, n: mlpDim, k: d,
                    flags: 1 | 2)
                Self.encodeGemm(
                    commandBuffer: cb,
                    pipeline: pipelines.gemm,
                    x: mlpMidBuf!,
                    weight: activeBuf, weightOffset: off.fc2W,
                    bias: activeBuf, biasOffset: off.fc2B,
                    y: hiddenBuf,
                    m: nPatches, n: d, k: mlpDim,
                    flags: 1 | 8)
                cb.commit()

                if let prev = prevCB {
                    try waitAndVerify(prev)
                }
                if blockIdx + 1 < Self.depth {
                    currentOffsets = try stageBlock(blockSpans[blockIdx + 1], into: nextBuf)
                }
                prevCB = cb
            }
            if let last = prevCB {
                try waitAndVerify(last)
            }

            // Reclaim ViT scratch before allocating merger buffers.
            qkvBuf = nil
            attnOutBuf = nil
            mlpMidBuf = nil
            rotCosBuf = nil
            rotSinBuf = nil
            stageBuf1 = nil
        }

        // 3. PatchMerger: LayerNorm([N_patches, 1152]) -> reshape [N_vis, 4608] -> fc1 (exact GELU) -> fc2 [N_vis, 2048]
        let mDim = Self.mergerHiddenSize
        let mergerNormW = try requireCheckedSpan("vision_tower.merger.norm.weight", expectedBytes: d * 2)
        let mergerNormB = try requireCheckedSpan("vision_tower.merger.norm.bias", expectedBytes: d * 2)
        let mergerFc1W = try requireCheckedSpan("vision_tower.merger.linear_fc1.weight", expectedBytes: mDim * mDim * 2)
        let mergerFc1B = try requireCheckedSpan("vision_tower.merger.linear_fc1.bias", expectedBytes: mDim * 2)
        let mergerFc2W = try requireCheckedSpan("vision_tower.merger.linear_fc2.weight", expectedBytes: Self.outHiddenSize * mDim * 2)
        let mergerFc2B = try requireCheckedSpan("vision_tower.merger.linear_fc2.bias", expectedBytes: Self.outHiddenSize * 2)

        var cursor0 = 0
        let mnwOff = try stageTensor(mergerNormW, into: stageBuf0, cursor: &cursor0)
        let mnbOff = try stageTensor(mergerNormB, into: stageBuf0, cursor: &cursor0)
        let fc2wOff = try stageTensor(mergerFc2W, into: stageBuf0, cursor: &cursor0)
        let fc2bOff = try stageTensor(mergerFc2B, into: stageBuf0, cursor: &cursor0)

        let fc1Min = min(mergerFc1W.offset, mergerFc1B.offset)
        let fc1Max = max(mergerFc1W.offset + mergerFc1W.byteLength, mergerFc1B.offset + mergerFc1B.byteLength)
        let fc1SlabBytes = fc1Max - fc1Min
        let fc1StageBytes = max(
            ((mergerFc1W.byteLength + 15) & ~15) + ((mergerFc1B.byteLength + 15) & ~15),
            fc1SlabBytes)
        let fc1StageBuf = try makeSharedBuffer(bytes: fc1StageBytes)
        let fc1wOff: Int
        let fc1bOff: Int
        if (mergerFc1W.offset - fc1Min) % 4 == 0,
           (mergerFc1B.offset - fc1Min) % 4 == 0,
           fc1SlabBytes <= fc1StageBuf.length,
           Self.preadExact(fd: fd, dst: fc1StageBuf.contents(), count: fc1SlabBytes, fileOffset: payloadOffset + fc1Min) {
            fc1wOff = mergerFc1W.offset - fc1Min
            fc1bOff = mergerFc1B.offset - fc1Min
        } else {
            var cursor1 = 0
            fc1wOff = try stageTensor(mergerFc1W, into: fc1StageBuf, cursor: &cursor1)
            fc1bOff = try stageTensor(mergerFc1B, into: fc1StageBuf, cursor: &cursor1)
        }

        let mergerMidBuf = try makeSharedBuffer(bytes: nVis * mDim * f32Size)
        let outF32Buf = try makeSharedBuffer(bytes: nVis * Self.outHiddenSize * f32Size)

        guard let cbMerger = context.queue.makeCommandBuffer() else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("failed to allocate Metal command buffer")
        }
        Self.encodeLayerNorm(
            commandBuffer: cbMerger,
            pipeline: pipelines.layerNorm,
            x: hiddenBuf,
            weight: stageBuf0, weightOffset: mnwOff,
            bias: stageBuf0, biasOffset: mnbOff,
            out: normBuf,
            rows: nPatches, dim: d, eps: Self.layerNormEps)
        Self.encodeGemm(
            commandBuffer: cbMerger,
            pipeline: pipelines.gemm,
            x: normBuf,
            weight: fc1StageBuf, weightOffset: fc1wOff,
            bias: fc1StageBuf, biasOffset: fc1bOff,
            y: mergerMidBuf,
            m: nVis, n: mDim, k: mDim,
            flags: 1 | 4)
        Self.encodeGemm(
            commandBuffer: cbMerger,
            pipeline: pipelines.gemm,
            x: mergerMidBuf,
            weight: stageBuf0, weightOffset: fc2wOff,
            bias: stageBuf0, biasOffset: fc2bOff,
            y: outF32Buf,
            m: nVis, n: Self.outHiddenSize, k: mDim,
            flags: 1)
        cbMerger.commit()
        try waitAndVerify(cbMerger)

        let outCount = nVis * Self.outHiddenSize
        let outPtr = outF32Buf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<outCount {
            if !outPtr[i].isFinite {
                throw Qwen36VisionImageError.visionWeightsCorrupt("vision encoder produced non-finite activation")
            }
        }
        var outF16 = [Float16](repeating: 0, count: outCount)
        outF16.withUnsafeMutableBufferPointer { dst in
            let dstPtr = dst.baseAddress!
            for i in 0..<outCount {
                dstPtr[i] = Float16(outPtr[i])
            }
        }
        return outF16
    }

    static func encodeGemm(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        x: MTLBuffer, xOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int = 0,
        bias: MTLBuffer, biasOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: Int, n: Int, k: Int,
        flags: UInt32
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(bias, offset: biasOffset, index: 2)
        encoder.setBuffer(y, offset: yOffset, index: 3)
        var mU = UInt32(m)
        var nU = UInt32(n)
        var kU = UInt32(k)
        var fU = flags
        encoder.setBytes(&mU, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nU, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&kU, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&fU, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + 31) / 32, height: (m + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    static func encodeLayerNorm(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        x: MTLBuffer, xOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int = 0,
        bias: MTLBuffer, biasOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        rows: Int, dim: Int, eps: Float
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(bias, offset: biasOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var rU = UInt32(rows)
        var dU = UInt32(dim)
        var epsF = eps
        encoder.setBytes(&rU, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&dU, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&epsF, length: MemoryLayout<Float>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    static func encodeRoPE2D(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        qkv: MTLBuffer, qkvOffset: Int = 0,
        cosTable: MTLBuffer,
        sinTable: MTLBuffer,
        seqLen: Int, numHeads: Int, headDim: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkv, offset: qkvOffset, index: 0)
        encoder.setBuffer(cosTable, offset: 0, index: 1)
        encoder.setBuffer(sinTable, offset: 0, index: 2)
        var sU = UInt32(seqLen)
        var hU = UInt32(numHeads)
        var dU = UInt32(headDim)
        encoder.setBytes(&sU, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&hU, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&dU, length: MemoryLayout<UInt32>.size, index: 5)
        let cols = numHeads * (headDim / 2)
        let tgW = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: cols, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW, height: 1, depth: 1))
        encoder.endEncoding()
    }

    static func encodeAttention(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        qkv: MTLBuffer, qkvOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        seqLen: Int, numHeads: Int, headDim: Int, scale: Float
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkv, offset: qkvOffset, index: 0)
        encoder.setBuffer(out, offset: outOffset, index: 1)
        var sU = UInt32(seqLen)
        var hU = UInt32(numHeads)
        var dU = UInt32(headDim)
        var sc = scale
        encoder.setBytes(&sU, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&hU, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&dU, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&sc, length: MemoryLayout<Float>.size, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: seqLen, height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - Vision Building Blocks

    private static func addInterpolatedPosEmbed(
        posTable: [Float],
        gridH: Int,
        gridW: Int,
        into hidden: inout [Float]
    ) {
        posTable.withUnsafeBufferPointer { posPtr in
            hidden.withUnsafeMutableBufferPointer { hidPtr in
                addInterpolatedPosEmbed(
                    posPtr: posPtr.baseAddress!,
                    gridH: gridH,
                    gridW: gridW,
                    into: hidPtr.baseAddress!)
            }
        }
    }

    private static func addInterpolatedPosEmbed(
        posPtr pBase: UnsafePointer<Float>,
        gridH: Int,
        gridW: Int,
        into hBase: UnsafeMutablePointer<Float>
    ) {
        let side = numGridPerSide
        let d = hiddenSize
        let llmH = gridH / 2
        let llmW = gridW / 2

        let hIdxs: [Double] = (0..<gridH).map { r in
            gridH > 1 ? Double(r) * Double(side - 1) / Double(gridH - 1) : 0.0
        }
        let wIdxs: [Double] = (0..<gridW).map { c in
            gridW > 1 ? Double(c) * Double(side - 1) / Double(gridW - 1) : 0.0
        }

        for bh in 0..<llmH {
            for bw in 0..<llmW {
                for ih in 0..<2 {
                    for iw in 0..<2 {
                        let p = (((bh * llmW + bw) * 2 + ih) * 2) + iw
                        let hr = hIdxs[bh * 2 + ih]
                        let wc = wIdxs[bw * 2 + iw]
                        let h0 = Int(floor(hr))
                        let w0 = Int(floor(wc))
                        let h1 = min(h0 + 1, side - 1)
                        let w1 = min(w0 + 1, side - 1)
                        let dh = Float(hr - Double(h0))
                        let dw = Float(wc - Double(w0))
                        let w00 = (1 - dh) * (1 - dw)
                        let w01 = (1 - dh) * dw
                        let w10 = dh * (1 - dw)
                        let w11 = dh * dw
                        let p00 = pBase + (h0 * side + w0) * d
                        let p01 = pBase + (h0 * side + w1) * d
                        let p10 = pBase + (h1 * side + w0) * d
                        let p11 = pBase + (h1 * side + w1) * d
                        let out = hBase + p * d
                        for j in 0..<d {
                            out[j] += w00 * p00[j] + w01 * p01[j] + w10 * p10[j] + w11 * p11[j]
                        }
                    }
                }
            }
        }
    }

    private static func buildRotaryTables(gridH: Int, gridW: Int) -> (cos: [Float], sin: [Float]) {
        let llmH = gridH / 2
        let llmW = gridW / 2
        let nPatches = gridH * gridW
        let halfD = headDim / 2 // 36
        let quarter = halfD / 2 // 18

        var invFreq = [Float](repeating: 0, count: quarter)
        for k in 0..<quarter {
            invFreq[k] = 1.0 / powf(rotaryTheta, Float(2 * k) / Float(halfD))
        }

        var cosT = [Float](repeating: 0, count: nPatches * halfD)
        var sinT = [Float](repeating: 0, count: nPatches * halfD)
        for bh in 0..<llmH {
            for bw in 0..<llmW {
                for ih in 0..<2 {
                    for iw in 0..<2 {
                        let p = (((bh * llmW + bw) * 2 + ih) * 2) + iw
                        let r = Float(bh * 2 + ih)
                        let c = Float(bw * 2 + iw)
                        let base = p * halfD
                        for k in 0..<quarter {
                            let ah = r * invFreq[k]
                            cosT[base + k] = cosf(ah)
                            sinT[base + k] = sinf(ah)
                            let aw = c * invFreq[k]
                            cosT[base + quarter + k] = cosf(aw)
                            sinT[base + quarter + k] = sinf(aw)
                        }
                    }
                }
            }
        }
        return (cosT, sinT)
    }

    private static func layerNorm(
        x: [Float],
        weight: [Float],
        bias: [Float],
        rows: Int,
        dim: Int,
        eps: Float,
        into out: inout [Float]
    ) {
        if out.count < rows * dim {
            out = [Float](repeating: 0, count: rows * dim)
        }
        x.withUnsafeBufferPointer { xPtr in
            weight.withUnsafeBufferPointer { wPtr in
                bias.withUnsafeBufferPointer { bPtr in
                    out.withUnsafeMutableBufferPointer { oPtr in
                        let xb = xPtr.baseAddress!
                        let wb = wPtr.baseAddress!
                        let bb = bPtr.baseAddress!
                        let ob = oPtr.baseAddress!
                        for r in 0..<rows {
                            let rowIn = xb + r * dim
                            let rowOut = ob + r * dim
                            var mean: Float = 0
                            vDSP_meanv(rowIn, 1, &mean, vDSP_Length(dim))
                            var negMean = -mean
                            vDSP_vsadd(rowIn, 1, &negMean, rowOut, 1, vDSP_Length(dim))
                            var sumSq: Float = 0
                            vDSP_svesq(rowOut, 1, &sumSq, vDSP_Length(dim))
                            var invStd = 1.0 / sqrtf(sumSq / Float(dim) + eps)
                            vDSP_vsmul(rowOut, 1, &invStd, rowOut, 1, vDSP_Length(dim))
                            vDSP_vmul(rowOut, 1, wb, 1, rowOut, 1, vDSP_Length(dim))
                            vDSP_vadd(rowOut, 1, bb, 1, rowOut, 1, vDSP_Length(dim))
                        }
                    }
                }
            }
        }
    }

    private static func applyVisionRoPEAndAttention(
        qkv: inout [Float],
        cosTable: [Float],
        sinTable: [Float],
        seqLen: Int,
        numHeads: Int,
        headDim: Int,
        into out: inout [Float]
    ) {
        let d = numHeads * headDim
        let qkvStride = 3 * d
        let halfD = headDim / 2 // 36
        let scale = 1.0 / sqrtf(Float(headDim))

        // Apply 2D rotary embeddings in place on Q ([0..<d]) and K ([d..<2*d]).
        qkv.withUnsafeMutableBufferPointer { qkvBuf in
            cosTable.withUnsafeBufferPointer { cosBuf in
                sinTable.withUnsafeBufferPointer { sinBuf in
                    let ptr = qkvBuf.baseAddress!
                    let cBase = cosBuf.baseAddress!
                    let sBase = sinBuf.baseAddress!
                    for s in 0..<seqLen {
                        let row = ptr + s * qkvStride
                        let cPtr = cBase + s * halfD
                        let sPtr = sBase + s * halfD
                        for h in 0..<numHeads {
                            let qh = row + h * headDim
                            let kh = row + d + h * headDim
                            for j in 0..<halfD {
                                let cs = cPtr[j]
                                let sn = sPtr[j]
                                let q0 = qh[j]
                                let q1 = qh[j + halfD]
                                qh[j] = q0 * cs - q1 * sn
                                qh[j + halfD] = q1 * cs + q0 * sn

                                let k0 = kh[j]
                                let k1 = kh[j + halfD]
                                kh[j] = k0 * cs - k1 * sn
                                kh[j + halfD] = k1 * cs + k0 * sn
                            }
                        }
                    }
                }
            }
        }

        if out.count < seqLen * d {
            out = [Float](repeating: 0, count: seqLen * d)
        }

        // Compute multi-head scaled dot-product attention per head in parallel across heads.
        // Each head reads disjoint row slices and writes disjoint [h*headDim ..< (h+1)*headDim] columns.
        qkv.withUnsafeBufferPointer { qkvBuf in
            out.withUnsafeMutableBufferPointer { outBuf in
                let qkvRawInt = UInt(bitPattern: qkvBuf.baseAddress!)
                let outRawInt = UInt(bitPattern: outBuf.baseAddress!)

                DispatchQueue.concurrentPerform(iterations: numHeads) { h in
                    let qkvPtr = UnsafePointer<Float>(bitPattern: qkvRawInt)!
                    let outPtr = UnsafeMutablePointer<Float>(bitPattern: outRawInt)!

                    var qh = [Float](repeating: 0, count: seqLen * headDim)
                    var kh = [Float](repeating: 0, count: seqLen * headDim)
                    var vh = [Float](repeating: 0, count: seqLen * headDim)
                    var scores = [Float](repeating: 0, count: seqLen * seqLen)
                    var oh = [Float](repeating: 0, count: seqLen * headDim)

                    qh.withUnsafeMutableBufferPointer { qB in
                        kh.withUnsafeMutableBufferPointer { kB in
                            vh.withUnsafeMutableBufferPointer { vB in
                                for s in 0..<seqLen {
                                    let row = qkvPtr + s * qkvStride
                                    memcpy(qB.baseAddress! + s * headDim, row + h * headDim, headDim * 4)
                                    memcpy(kB.baseAddress! + s * headDim, row + d + h * headDim, headDim * 4)
                                    memcpy(vB.baseAddress! + s * headDim, row + 2 * d + h * headDim, headDim * 4)
                                }
                            }
                        }
                    }

                    qh.withUnsafeBufferPointer { qB in
                        kh.withUnsafeBufferPointer { kB in
                            scores.withUnsafeMutableBufferPointer { sB in
                                cblas_sgemm(
                                    CblasRowMajor,
                                    CblasNoTrans,
                                    CblasTrans,
                                    Int32(seqLen),
                                    Int32(seqLen),
                                    Int32(headDim),
                                    scale,
                                    qB.baseAddress!,
                                    Int32(headDim),
                                    kB.baseAddress!,
                                    Int32(headDim),
                                    0.0,
                                    sB.baseAddress!,
                                    Int32(seqLen))

                                var nInt = Int32(seqLen)
                                for r in 0..<seqLen {
                                    let row = sB.baseAddress! + r * seqLen
                                    var maxVal: Float = 0
                                    vDSP_maxv(row, 1, &maxVal, vDSP_Length(seqLen))
                                    var negMax = -maxVal
                                    vDSP_vsadd(row, 1, &negMax, row, 1, vDSP_Length(seqLen))
                                    vvexpf(row, row, &nInt)
                                    var sum: Float = 0
                                    vDSP_sve(row, 1, &sum, vDSP_Length(seqLen))
                                    var invSum = 1.0 / sum
                                    vDSP_vsmul(row, 1, &invSum, row, 1, vDSP_Length(seqLen))
                                }
                            }
                        }
                    }

                    scores.withUnsafeBufferPointer { sB in
                        vh.withUnsafeBufferPointer { vB in
                            oh.withUnsafeMutableBufferPointer { oB in
                                cblas_sgemm(
                                    CblasRowMajor,
                                    CblasNoTrans,
                                    CblasNoTrans,
                                    Int32(seqLen),
                                    Int32(headDim),
                                    Int32(seqLen),
                                    1.0,
                                    sB.baseAddress!,
                                    Int32(seqLen),
                                    vB.baseAddress!,
                                    Int32(headDim),
                                    0.0,
                                    oB.baseAddress!,
                                    Int32(headDim))

                                for s in 0..<seqLen {
                                    memcpy(outPtr + s * d + h * headDim, oB.baseAddress! + s * headDim, headDim * 4)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static func geluTanhInPlace(_ x: inout [Float], count: Int) {
        let c0: Float = 0.7978845608028654 // sqrt(2 / pi)
        let c1: Float = 0.044715
        x.withUnsafeMutableBufferPointer { buf in
            let p = buf.baseAddress!
            for i in 0..<count {
                let v = p[i]
                let inner = c0 * (v + c1 * v * v * v)
                p[i] = 0.5 * v * (1.0 + tanhf(inner))
            }
        }
    }

    private static func geluExactInPlace(_ x: inout [Float], count: Int) {
        let invSqrt2: Float = 0.7071067811865475
        x.withUnsafeMutableBufferPointer { buf in
            let p = buf.baseAddress!
            for i in 0..<count {
                let v = p[i]
                p[i] = 0.5 * v * (1.0 + erff(v * invSqrt2))
            }
        }
    }
}
