import Accelerate
import Darwin
import Foundation
import Synchronization

/// Ephemeral Pure-Swift + Accelerate AMX Vision Transformer encoder for Qwen 3.6
/// (`vision_weights.bin`).
///
/// Maps `vision_weights.bin` read-only on demand when an uncached image is
/// encountered, executes the 27-layer ViT + `2×2` spatial merger in FP32 using
/// Apple's AMX matrix engine (`cblas_sgemm`), caches the compact `[N_vis, 2048]`
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
    private let encodeLock = Mutex(())

    private struct TensorSpan {
        let offset: Int
        let byteLength: Int
        let shape: [Int]
        let dtype: String
    }

    public init(weightsURL: URL) {
        self.weightsURL = weightsURL
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

    private func runEncoder(patchData: Qwen36VisionPatchData) throws -> [Float16] {
        let path = weightsURL.path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw Qwen36VisionImageError.visionSidecarMissing(path)
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size > 8 else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("cannot stat \(path)")
        }
        let fileSize = Int(st.st_size)
        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != MAP_FAILED else {
            throw Qwen36VisionImageError.visionWeightsCorrupt("mmap failed for \(path) (errno \(errno))")
        }
        defer { munmap(mapped, fileSize) }
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

    // MARK: - Vision Building Blocks

    private static func addInterpolatedPosEmbed(
        posTable: [Float],
        gridH: Int,
        gridW: Int,
        into hidden: inout [Float]
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

        posTable.withUnsafeBufferPointer { posPtr in
            hidden.withUnsafeMutableBufferPointer { hidPtr in
                let pBase = posPtr.baseAddress!
                let hBase = hidPtr.baseAddress!
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
