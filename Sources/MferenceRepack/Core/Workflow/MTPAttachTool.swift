import Foundation

/// Attaches a checkpoint's native MTP (multi-token-prediction) draft layer to
/// an existing Qwen 3.8 `.gturbo` install.
///
/// The BF16 `mtp.*` tensors from the HF shard are appended to
/// `model_weights.bin` as resident tensors: projections are quantized to
/// INT4 affine group-64 in the MLX layout (`Int4AffineEncoder`, bit-exact
/// with the runtime's reference quantizer) so the existing INT4 GEMV
/// kernels consume them; norm vectors stay BF16 with the HF checkpoint's
/// zero-centered convention converted to full form (`w + 1`, mirroring
/// mlx-vlm's `Qwen3_5MTPDraftModel.sanitize`). Tensor names keep the source
/// `mtp.` prefix, which the runtime probes at load; installs without them
/// are unaffected.
///
/// The resident index is rewritten in place when the new entry table still
/// fits the existing page-padded index region (the common case); otherwise
/// the whole weights file is rebuilt with a grown index. `manifest.json`'s
/// size/sha256 entry is refreshed, and any stale verified-install receipt is
/// removed so it cannot bind to the old manifest.
public enum MTPAttachTool {

    public struct Result: Sendable {
        public let tensorCount: Int
        public let appendedBytes: UInt64
        public let weightsFileBytes: UInt64
        public let rewroteWeightsFile: Bool
    }

    /// Surgical scaling parameters applied losslessly to BF16 norms and INT4/INT8
    /// affine group scales/biases in `model_weights.bin`, with optional automatic
    /// co-adaptation of `mtp.norm.weight` and the MTP draft block's internal attention
    /// norms (`mtp.*q_norm*`, `mtp.*k_norm*`), router (`mtp.*mlp.gate.weight`), and
    /// output/down projections (`mtp.*o_proj.weight`, `mtp.*down_proj.weight`).
    public struct SurgeryConfig: Sendable {
        public var finalNormScale: Float
        public var qkNormScale: Float
        public var qkNormStartLayerRatio: Float
        public var routerTemperature: Float
        public var lateLayerResidualScale: Float
        public var lateLayerStartFraction: Float
        public var sharedExpertScale: Float
        public var sharedExpertStartLayerRatio: Float
        public var mtpNormScale: Float
        public var mtpHiddenNormScale: Float
        public var mtpQKNormScale: Float?
        public var mtpRouterTemperature: Float?
        public var mtpResidualScale: Float?
        public var mtpSharedExpertScale: Float?
        public var coAdaptMTP: Bool

        public init(finalNormScale: Float = 1.0,
                    qkNormScale: Float = 1.0,
                    qkNormStartLayerRatio: Float = 0.5,
                    routerTemperature: Float = 1.0,
                    lateLayerResidualScale: Float = 1.0,
                    lateLayerStartFraction: Float = 0.75,
                    sharedExpertScale: Float = 1.0,
                    sharedExpertStartLayerRatio: Float = 0.5,
                    mtpNormScale: Float = 1.0,
                    mtpHiddenNormScale: Float = 1.0,
                    mtpQKNormScale: Float? = nil,
                    mtpRouterTemperature: Float? = nil,
                    mtpResidualScale: Float? = nil,
                    mtpSharedExpertScale: Float? = nil,
                    coAdaptMTP: Bool = true) {
            self.finalNormScale = finalNormScale
            self.qkNormScale = qkNormScale
            self.qkNormStartLayerRatio = qkNormStartLayerRatio
            self.routerTemperature = routerTemperature
            self.lateLayerResidualScale = lateLayerResidualScale
            self.lateLayerStartFraction = lateLayerStartFraction
            self.sharedExpertScale = sharedExpertScale
            self.sharedExpertStartLayerRatio = sharedExpertStartLayerRatio
            self.mtpNormScale = mtpNormScale
            self.mtpHiddenNormScale = mtpHiddenNormScale
            self.mtpQKNormScale = mtpQKNormScale
            self.mtpRouterTemperature = mtpRouterTemperature
            self.mtpResidualScale = mtpResidualScale
            self.mtpSharedExpertScale = mtpSharedExpertScale
            self.coAdaptMTP = coAdaptMTP
        }
    }

    public struct SurgeryResult: Sendable {
        public let modifiedNormTensors: Int
        public let modifiedAffineTensors: Int
        public let coAdaptedMTP: Bool
        public let weightsSha256: String
    }

    private static let headerBytes = 24
    private static let entryBytes = 72
    private static let pageBytes: UInt64 = 16_384

    private struct RawIndexEntry {
        var name: String
        var dtype: UInt8
        var fileOffset: UInt64
        var sizeBytes: UInt64
        var shape: (UInt32, UInt32, UInt32, UInt32)
        var scaleOffset: UInt64
        var scaleSize: UInt64
        var biasOffset: UInt64
        var biasSize: UInt64
    }

    private struct ExpectedTensor {
        let name: String            // without the "mtp." prefix
        let rows: Int
        let cols: Int?              // nil => BF16 norm vector of `rows` elements
        let bits: Int               // 4 or 8 when cols != nil
        let addOne: Bool            // zero-centered norm -> full form

        init(name: String, rows: Int, cols: Int?, bits: Int = 4, addOne: Bool) {
            self.name = name
            self.rows = rows
            self.cols = cols
            self.bits = bits
            self.addOne = addOne
        }
    }

    public static func run(gturboDirectory: String, shardPath: String) throws -> Result {
        let weightsPath = (gturboDirectory as NSString)
            .appendingPathComponent("model_weights.bin")
        let manifestPath = (gturboDirectory as NSString)
            .appendingPathComponent("manifest.json")

        // -- Manifest arch dims drive shape validation.
        let manifestData = try Posix.readBoundedData(manifestPath, maximumBytes: 8 << 20)
        guard var manifestRoot = try JSONSerialization
                .jsonObject(with: manifestData) as? [String: Any],
              let arch = manifestRoot["arch"] as? [String: Any],
              let family = arch["family"] as? String, (family == "qwen38" || family == "qwen36"),
              let d = arch["hiddenSize"] as? Int,
              let f = family == "qwen36"
                ? (arch["moeIntermediateSize"] as? Int)
                : (arch["ffnIntermediate"] as? Int),
              d > 0, f > 0,
              let numHeads = arch["numHeads"] as? Int,
              let numKVHeads = (arch["numFullKVHeads"] as? Int) ?? (arch["numKVHeads"] as? Int),
              let headDim = (arch["fullHeadDim"] as? Int) ?? (arch["headDim"] as? Int) else {
            throw RepackError.configurationInvalid(
                detail: "\(manifestPath) is not a supported Qwen 3.8/3.6 manifest")
        }
        // -- Every manifest file must exist at its recorded size BEFORE the
        // irreversible append: a pre-existing missing/truncated sidecar would
        // otherwise only surface in the post-attach receipt pass, when a
        // retry is already rejected as "already attached".
        if let files = manifestRoot["files"] as? [String: Any] {
            for (name, entry) in files {
                let path = (gturboDirectory as NSString).appendingPathComponent(name)
                let recorded = (entry as? [String: Any])?["size"] as? Int
                let actual = (try? FileManager.default
                    .attributesOfItem(atPath: path)[.size] as? Int) ?? nil
                guard let recorded, let actual, recorded == actual else {
                    throw RepackError.configurationInvalid(detail:
                        "install file \(name) is missing or has size " +
                        "\(actual.map(String.init) ?? "unknown") (manifest records " +
                        "\(recorded.map(String.init) ?? "unknown")); repair the " +
                        "install before attaching MTP tensors")
                }
            }
        }

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        let isMoE = (family == "qwen36")
        let moeF = isMoE ? f : 0
        let sharedF = isMoE
            ? ((arch["sharedExpertIntermediateSize"] as? Int).flatMap { $0 > 0 ? $0 : nil } ?? moeF)
            : 0
        let numExperts = isMoE ? ((arch["numExperts"] as? Int) ?? 0) : 0
        if isMoE {
            guard numExperts > 0, moeF > 0, moeF % 64 == 0,
                  sharedF > 0, sharedF % 64 == 0, d % 64 == 0 else {
                throw RepackError.configurationInvalid(
                    detail: "\(manifestPath) has invalid Qwen 3.6 MoE MTP dimensions")
            }
        }

        var expected: [ExpectedTensor] = [
            ExpectedTensor(name: "fc.weight", rows: d, cols: 2 * d, addOne: false),
            ExpectedTensor(name: "pre_fc_norm_embedding.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "pre_fc_norm_hidden.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "norm.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.input_layernorm.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.post_attention_layernorm.weight", rows: d, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.self_attn.q_proj.weight", rows: 2 * qDim, cols: d, addOne: false),
            ExpectedTensor(name: "layers.0.self_attn.k_proj.weight", rows: kvDim, cols: d, addOne: false),
            ExpectedTensor(name: "layers.0.self_attn.v_proj.weight", rows: kvDim, cols: d, addOne: false),
            ExpectedTensor(name: "layers.0.self_attn.o_proj.weight", rows: d, cols: qDim, addOne: false),
            ExpectedTensor(name: "layers.0.self_attn.q_norm.weight", rows: headDim, cols: nil, addOne: true),
            ExpectedTensor(name: "layers.0.self_attn.k_norm.weight", rows: headDim, cols: nil, addOne: true),
        ]
        if isMoE {
            // Router and scalar gate match trunk precision (INT8 g64); shared expert uses INT4 g64.
            expected.append(contentsOf: [
                ExpectedTensor(name: "layers.0.mlp.gate.weight", rows: numExperts, cols: d, bits: 8, addOne: false),
                ExpectedTensor(name: "layers.0.mlp.shared_expert_gate.weight", rows: 1, cols: d, bits: 8, addOne: false),
                ExpectedTensor(name: "layers.0.mlp.shared_expert.gate_proj.weight", rows: sharedF, cols: d, addOne: false),
                ExpectedTensor(name: "layers.0.mlp.shared_expert.up_proj.weight", rows: sharedF, cols: d, addOne: false),
                ExpectedTensor(name: "layers.0.mlp.shared_expert.down_proj.weight", rows: d, cols: sharedF, addOne: false),
            ])
        } else {
            expected.append(contentsOf: [
                ExpectedTensor(name: "layers.0.mlp.gate_proj.weight", rows: f, cols: d, addOne: false),
                ExpectedTensor(name: "layers.0.mlp.up_proj.weight", rows: f, cols: d, addOne: false),
                ExpectedTensor(name: "layers.0.mlp.down_proj.weight", rows: d, cols: f, addOne: false),
            ])
        }

        // -- Existing resident index.
        let weightsFd = try Posix.openExistingRW(weightsPath)
        defer { close(weightsFd) }
        let oldFileSize = try Posix.fileSize(fd: weightsFd, path: weightsPath)
        var (oldIndexSize, oldResidentSize, oldEntries) =
            try readIndex(fd: weightsFd, path: weightsPath)
        guard oldIndexSize + oldResidentSize == oldFileSize else {
            throw RepackError.configurationInvalid(detail:
                "\(weightsPath) size \(oldFileSize) != index \(oldIndexSize) + resident \(oldResidentSize)")
        }
        guard !oldEntries.contains(where: { $0.name.hasPrefix("mtp.") }) else {
            throw RepackError.configurationInvalid(
                detail: "\(gturboDirectory) already carries mtp.* tensors")
        }

        // -- Source shard.
        let shardFd = try Posix.openRead(shardPath)
        defer { close(shardFd) }
        let shardSize = try Posix.fileSize(fd: shardFd, path: shardPath)
        var headerLenBytes = [UInt8](repeating: 0, count: 8)
        try headerLenBytes.withUnsafeMutableBytes {
            try Posix.preadAll(fd: shardFd, path: shardPath,
                               buf: $0.baseAddress!, count: 8, offset: 0)
        }
        let headerLen = headerLenBytes.withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        guard headerLen > 0, headerLen < Safetensors.maxHeaderBytes else {
            throw RepackError.safetensorsHeaderTooLarge(path: shardPath, size: headerLen)
        }
        var headerData = Data(count: Int(headerLen))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(fd: shardFd, path: shardPath,
                               buf: $0.baseAddress!, count: Int(headerLen), offset: 8)
        }
        let header = try Safetensors.parseHeaderBytes(path: shardPath,
                                                      fileSize: shardSize,
                                                      headerBytes: headerData)
        var byName: [String: SourceTensor] = [:]
        for t in header.tensors where t.name.hasPrefix("mtp.") {
            byName[t.name] = t
        }

        // Validate routed MoE expert tensors before mutating any installed files.
        let routedExperts: (gateUp: SourceTensor, down: SourceTensor)?
        if isMoE {
            let gateUpName = "mtp.layers.0.mlp.experts.gate_up_proj"
            let downName = "mtp.layers.0.mlp.experts.down_proj"
            guard let gateUp = byName[gateUpName] else {
                throw RepackError.missingTensor(name: gateUpName)
            }
            guard let down = byName[downName] else {
                throw RepackError.missingTensor(name: downName)
            }
            guard gateUp.dtype == .bf16 else {
                throw RepackError.dtypeMismatch(name: gateUpName,
                    detail: "expected BF16, got \(gateUp.dtype)")
            }
            guard down.dtype == .bf16 else {
                throw RepackError.dtypeMismatch(name: downName,
                    detail: "expected BF16, got \(down.dtype)")
            }
            guard gateUp.shape == [UInt64(numExperts), UInt64(2 * moeF), UInt64(d)] else {
                throw RepackError.shapeMismatch(name: gateUpName,
                    detail: "expected [\(numExperts), \(2 * moeF), \(d)], got \(gateUp.shape)")
            }
            guard down.shape == [UInt64(numExperts), UInt64(d), UInt64(moeF)] else {
                throw RepackError.shapeMismatch(name: downName,
                    detail: "expected [\(numExperts), \(d), \(moeF)], got \(down.shape)")
            }
            routedExperts = (gateUp, down)
        } else {
            routedExperts = nil
        }

        // -- Quantize / convert into an appended-payload staging file plan.
        // Payload is written straight after the current file end; entries are
        // built against absolute offsets in the (possibly shifted) new file.
        var appended = Data()
        appended.reserveCapacity(isMoE ? (64 << 20) : (1 << 20))
        var newEntries: [RawIndexEntry] = []
        let payloadBase = oldFileSize

        for spec in expected {
            let fullName = "mtp.\(spec.name)"
            guard let tensor = byName[fullName] else {
                throw RepackError.missingTensor(name: fullName)
            }
            guard tensor.dtype == .bf16 else {
                throw RepackError.dtypeMismatch(name: fullName,
                    detail: "expected BF16, got \(tensor.dtype)")
            }
            if let cols = spec.cols {
                guard tensor.shape == [UInt64(spec.rows), UInt64(cols)] else {
                    throw RepackError.shapeMismatch(name: fullName,
                        detail: "expected [\(spec.rows), \(cols)], got \(tensor.shape)")
                }
                guard spec.bits == 4 || spec.bits == 8,
                      cols % Int4AffineEncoder.groupSize == 0 else {
                    throw RepackError.shapeMismatch(name: fullName,
                        detail: "cols \(cols) not a multiple of \(Int4AffineEncoder.groupSize) or invalid bits \(spec.bits)")
                }
                let (weightOffset, scaleOffset, biasOffset) = try appendQuantizedBF16Matrix(
                    shardFd: shardFd, shardPath: shardPath,
                    sourceOffset: tensor.absoluteOffset,
                    rows: spec.rows, cols: cols, bits: spec.bits,
                    payloadBase: payloadBase, into: &appended)
                let groups = cols / Int4AffineEncoder.groupSize
                newEntries.append(RawIndexEntry(
                    name: fullName, dtype: 0,
                    fileOffset: weightOffset,
                    sizeBytes: UInt64(spec.rows * cols * spec.bits / 8),
                    shape: (UInt32(spec.rows), UInt32(cols), 0, 0),
                    scaleOffset: scaleOffset,
                    scaleSize: UInt64(spec.rows * groups * 2),
                    biasOffset: biasOffset,
                    biasSize: UInt64(spec.rows * groups * 2)))
            } else {
                guard tensor.shape == [UInt64(spec.rows)] else {
                    throw RepackError.shapeMismatch(name: fullName,
                        detail: "expected [\(spec.rows)], got \(tensor.shape)")
                }
                var raw = [UInt16](repeating: 0, count: spec.rows)
                try raw.withUnsafeMutableBytes {
                    try Posix.preadAll(fd: shardFd, path: shardPath,
                                       buf: $0.baseAddress!, count: spec.rows * 2,
                                       offset: tensor.absoluteOffset)
                }
                if spec.addOne {
                    for i in 0..<raw.count {
                        let value = Float(bitPattern: UInt32(raw[i]) << 16) + 1.0
                        raw[i] = Int4AffineEncoder.bf16Bits(value)
                    }
                }
                let offset = payloadBase + UInt64(appended.count)
                raw.withUnsafeBufferPointer {
                    appended.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self))
                }
                newEntries.append(RawIndexEntry(
                    name: fullName, dtype: 1,
                    fileOffset: offset,
                    sizeBytes: UInt64(spec.rows * 2),
                    shape: (UInt32(spec.rows), 0, 0, 0),
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0))
            }
        }

        // Write packed_experts_mtp/layer_00.bin (if MoE) before rewriting model_weights.bin.
        var routedExpertPoolInfo: (size: UInt64, sha256: String, expertStride: UInt64)?
        if let routedExperts {
            routedExpertPoolInfo = try writeQwen36MTPRoutedExperts(
                gturboDirectory: gturboDirectory,
                shardFd: shardFd,
                shardPath: shardPath,
                gateUp: routedExperts.gateUp,
                down: routedExperts.down,
                numExperts: numExperts,
                hidden: d,
                intermediate: moeF)
        }

        // -- New index image.
        let allEntries = oldEntries + newEntries
        let rawIndexBytes = headerBytes + allEntries.count * entryBytes
            + allEntries.reduce(0) { $0 + $1.name.utf8.count }
        let fitsInPlace = UInt64(rawIndexBytes) <= oldIndexSize
        let newIndexSize = fitsInPlace
            ? oldIndexSize
            : ((UInt64(rawIndexBytes) + pageBytes - 1) / pageBytes) * pageBytes
        let shift = newIndexSize - oldIndexSize
        let newResidentSize = oldResidentSize + UInt64(appended.count)

        var shifted = allEntries
        if shift > 0 {
            for i in 0..<shifted.count {
                shifted[i].fileOffset += shift
                if shifted[i].scaleSize > 0 { shifted[i].scaleOffset += shift }
                if shifted[i].biasSize > 0 { shifted[i].biasOffset += shift }
            }
        }
        let indexImage = encodeIndex(entries: shifted,
                                     indexSize: newIndexSize,
                                     residentSize: newResidentSize)

        let rewrote = !fitsInPlace
        if fitsInPlace {
            try appended.withUnsafeBytes {
                try Posix.pwriteAll(fd: weightsFd, path: weightsPath,
                                    buf: $0.baseAddress!, count: appended.count,
                                    offset: oldFileSize)
            }
            try indexImage.withUnsafeBytes {
                try Posix.pwriteAll(fd: weightsFd, path: weightsPath,
                                    buf: $0.baseAddress!, count: indexImage.count,
                                    offset: 0)
            }
            try Posix.fsync(weightsFd, path: weightsPath)
        } else {
            // Grown index: rebuild the file with every payload byte shifted.
            let tmpPath = weightsPath + ".mtp-attach.tmp"
            let outFd = try Posix.openCreateRW(tmpPath)
            defer { close(outFd) }
            try indexImage.withUnsafeBytes {
                try Posix.pwriteAll(fd: outFd, path: tmpPath,
                                    buf: $0.baseAddress!, count: indexImage.count,
                                    offset: 0)
            }
            let tile = 8 << 20
            let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: tile, alignment: 16_384)
            defer { buf.deallocate() }
            var copied: UInt64 = 0
            while copied < oldResidentSize {
                let take = Int(min(UInt64(tile), oldResidentSize - copied))
                try Posix.preadAll(fd: weightsFd, path: weightsPath,
                                   buf: buf.baseAddress!, count: take,
                                   offset: oldIndexSize + copied)
                try Posix.pwriteAll(fd: outFd, path: tmpPath,
                                    buf: buf.baseAddress!, count: take,
                                    offset: newIndexSize + copied)
                copied += UInt64(take)
            }
            try appended.withUnsafeBytes {
                try Posix.pwriteAll(fd: outFd, path: tmpPath,
                                    buf: $0.baseAddress!, count: appended.count,
                                    offset: newIndexSize + oldResidentSize)
            }
            try Posix.fsync(outFd, path: tmpPath)
            try Posix.rename(from: tmpPath, to: weightsPath)
        }

        // -- Manifest refresh.
        let newSize = newIndexSize + newResidentSize
        let sha = try Sha256Stream.hashFile(path: weightsPath, tileBytes: 8 << 20)
        guard var files = manifestRoot["files"] as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "manifest.json has no files map")
        }
        files["model_weights.bin"] = ["size": Int(newSize), "sha256": sha]
        if let poolInfo = routedExpertPoolInfo {
            files["packed_experts_mtp/layer_00.bin"] = [
                "size": Int(poolInfo.size),
                "sha256": poolInfo.sha256,
            ]
            var sidecars = manifestRoot["sidecars"] as? [String: Any] ?? [:]
            sidecars["mtp"] = [
                "carried": true,
                "tensorCount": newEntries.count + 2,
            ]
            manifestRoot["sidecars"] = sidecars

            var pools = manifestRoot["auxiliaryExpertPools"] as? [[String: Any]] ?? []
            pools.removeAll { ($0["name"] as? String) == "mtp" }
            pools.append([
                "name": "mtp",
                "directory": "packed_experts_mtp",
                "expertsPerLayer": numExperts,
                "expertStride": Int(poolInfo.expertStride),
                "layers": [["layer": 0, "file": "layer_00.bin"]],
            ])
            manifestRoot["auxiliaryExpertPools"] = pools
        }
        manifestRoot["files"] = files
        let newManifest = try JSONSerialization.data(
            withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try Posix.atomicWrite(newManifest, to: manifestPath,
                              durableIn: gturboDirectory)
        // The attach invalidated the old receipt; regenerate it so installs
        // that gate on verified-install.json (the server's library probe) keep working
        // without a manual --verify-install pass. The attach itself is
        // complete at this point, so a receipt failure must say so and point
        // at the retryable step rather than reading as a failed attach.
        let receiptPath = (gturboDirectory as NSString)
            .appendingPathComponent("verified-install.json")
        try? FileManager.default.removeItem(atPath: receiptPath)
        do {
            _ = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: gturboDirectory))
        } catch {
            throw RepackError.configurationInvalid(detail:
                "MTP tensors were attached, but regenerating the install " +
                "receipt failed (\(error)); repair the install and run " +
                "--verify-install --input-gturbo \(gturboDirectory)")
        }

        return Result(tensorCount: newEntries.count,
                      appendedBytes: UInt64(appended.count),
                      weightsFileBytes: newSize,
                      rewroteWeightsFile: rewrote)
    }

    /// Applies lossless resident weight surgery directly to `model_weights.bin`
    /// (modifying BF16 norm vectors and INT4/INT8 group scales/biases without re-quantizing
    /// nibble payloads), co-adapts `mtp.pre_fc_norm_hidden.weight` when `config.coAdaptMTP`
    /// is enabled, and refreshes `manifest.json` + `verified-install.json`.
    public static func applyResidentWeightSurgery(
        gturboDirectory: String,
        config: SurgeryConfig
    ) throws -> SurgeryResult {
        guard config.routerTemperature > 0, config.finalNormScale > 0 else {
            throw RepackError.configurationInvalid(
                detail: "routerTemperature and finalNormScale must be positive")
        }
        let weightsPath = (gturboDirectory as NSString)
            .appendingPathComponent("model_weights.bin")
        let manifestPath = (gturboDirectory as NSString)
            .appendingPathComponent("manifest.json")

        let manifestData = try Posix.readBoundedData(manifestPath, maximumBytes: 8 << 20)
        guard var manifestRoot = try JSONSerialization
                .jsonObject(with: manifestData) as? [String: Any],
              let arch = manifestRoot["arch"] as? [String: Any],
              var files = manifestRoot["files"] as? [String: Any] else {
            throw RepackError.configurationInvalid(
                detail: "\(manifestPath) is not a valid .gturbo manifest")
        }
        let numLayers = (arch["numLayers"] as? Int) ?? 0
        let qkStartLayer = Int(Float(max(0, numLayers)) * max(0.0, min(1.0, config.qkNormStartLayerRatio)))
        let lateStartLayer = Int(Float(max(0, numLayers)) * max(0.0, min(1.0, config.lateLayerStartFraction)))
        let sharedStartLayer = Int(Float(max(0, numLayers)) * max(0.0, min(1.0, config.sharedExpertStartLayerRatio)))
        let routerScale = 1.0 / config.routerTemperature
        let mtpQKNormScale = config.mtpQKNormScale ?? (config.coAdaptMTP ? config.qkNormScale : 1.0)
        let mtpRouterTemp = config.mtpRouterTemperature ?? (config.coAdaptMTP ? config.routerTemperature : 1.0)
        guard mtpRouterTemp > 0 else {
            throw RepackError.configurationInvalid(detail: "mtpRouterTemperature must be positive")
        }
        let mtpRouterScale = 1.0 / mtpRouterTemp
        let mtpResidualScale = config.mtpResidualScale ?? (config.coAdaptMTP ? config.lateLayerResidualScale : 1.0)
        let mtpSharedExpertScale = config.mtpSharedExpertScale ?? (config.coAdaptMTP ? config.sharedExpertScale : 1.0)

        let weightsFd = try Posix.openExistingRW(weightsPath)
        defer { close(weightsFd) }
        let fileSize = try Posix.fileSize(fd: weightsFd, path: weightsPath)
        let (_, _, entries) = try readIndex(fd: weightsFd, path: weightsPath)

        func scaleBF16Region(offset: UInt64, byteCount: UInt64, factor: Float) throws {
            guard factor != 1.0, byteCount > 0, byteCount % 2 == 0 else { return }
            let count = Int(byteCount / 2)
            var words = [UInt16](repeating: 0, count: count)
            try words.withUnsafeMutableBytes {
                try Posix.preadAll(fd: weightsFd, path: weightsPath,
                                   buf: $0.baseAddress!, count: Int(byteCount), offset: offset)
            }
            for i in 0..<count {
                let v = Float(bitPattern: UInt32(words[i]) << 16) * factor
                words[i] = Int4AffineEncoder.bf16Bits(v)
            }
            try words.withUnsafeBytes {
                try Posix.pwriteAll(fd: weightsFd, path: weightsPath,
                                    buf: $0.baseAddress!, count: Int(byteCount), offset: offset)
            }
        }

        var modifiedNorms = 0
        var modifiedAffine = 0
        var didCoAdaptMTP = false

        for entry in entries {
            let isMtpLayer = entry.name.hasPrefix("mtp.")
            if entry.dtype == 1 {
                // BF16 norm vector
                if entry.name == "final_norm.weight"
                    || entry.name == "model.norm.weight"
                    || entry.name == "language_model.model.norm.weight"
                    || entry.name.hasSuffix(".model.norm.weight") {
                    if config.finalNormScale != 1.0 {
                        try scaleBF16Region(offset: entry.fileOffset,
                                            byteCount: entry.sizeBytes,
                                            factor: config.finalNormScale)
                        modifiedNorms += 1
                    }
                } else if entry.name == "mtp.pre_fc_norm_hidden.weight" {
                    // RMSNorm(alpha * h) == RMSNorm(h), so pre_fc_norm_hidden is already invariant
                    // to uniform final_norm scaling; only apply explicit mtpHiddenNormScale calibration.
                    if config.mtpHiddenNormScale != 1.0 {
                        try scaleBF16Region(offset: entry.fileOffset,
                                            byteCount: entry.sizeBytes,
                                            factor: config.mtpHiddenNormScale)
                        modifiedNorms += 1
                    }
                } else if entry.name == "mtp.norm.weight" {
                    // Synchronize mtp.norm.weight with finalNormScale when coAdaptMTP is enabled
                    // so the MTP head's logit sharpness entering the shared lm_head matches the trunk.
                    let effectiveMtpNormScale = config.coAdaptMTP
                        ? (config.finalNormScale * config.mtpNormScale)
                        : config.mtpNormScale
                    if effectiveMtpNormScale != 1.0 {
                        try scaleBF16Region(offset: entry.fileOffset,
                                            byteCount: entry.sizeBytes,
                                            factor: effectiveMtpNormScale)
                        modifiedNorms += 1
                        if config.coAdaptMTP && config.finalNormScale != 1.0 {
                            didCoAdaptMTP = true
                        }
                    }
                } else if entry.name.hasSuffix(".q_norm.weight") || entry.name.hasSuffix(".k_norm.weight") {
                    if isMtpLayer {
                        if mtpQKNormScale != 1.0 {
                            try scaleBF16Region(offset: entry.fileOffset,
                                                byteCount: entry.sizeBytes,
                                                factor: mtpQKNormScale)
                            modifiedNorms += 1
                            didCoAdaptMTP = true
                        }
                    } else if config.qkNormScale != 1.0 {
                        let inTargetRange: Bool
                        if let layerIdx = parseLayerIndex(from: entry.name) {
                            inTargetRange = layerIdx >= qkStartLayer
                        } else {
                            inTargetRange = true
                        }
                        if inTargetRange {
                            try scaleBF16Region(offset: entry.fileOffset,
                                                byteCount: entry.sizeBytes,
                                                factor: config.qkNormScale)
                            modifiedNorms += 1
                        }
                    }
                }
            } else if entry.dtype == 0, entry.scaleSize > 0, entry.biasSize > 0 {
                // Affine quantized projection: scale both scale and bias BF16 tables in place.
                if entry.name.hasSuffix(".mlp.gate.weight") {
                    if isMtpLayer {
                        if mtpRouterScale != 1.0 {
                            try scaleBF16Region(offset: entry.scaleOffset,
                                                byteCount: entry.scaleSize,
                                                factor: mtpRouterScale)
                            try scaleBF16Region(offset: entry.biasOffset,
                                                byteCount: entry.biasSize,
                                                factor: mtpRouterScale)
                            modifiedAffine += 1
                            didCoAdaptMTP = true
                        }
                    } else if routerScale != 1.0 {
                        let inTargetRange: Bool
                        if let layerIdx = parseLayerIndex(from: entry.name) {
                            inTargetRange = layerIdx >= lateStartLayer
                        } else {
                            inTargetRange = true
                        }
                        if inTargetRange {
                            try scaleBF16Region(offset: entry.scaleOffset,
                                                byteCount: entry.scaleSize,
                                                factor: routerScale)
                            try scaleBF16Region(offset: entry.biasOffset,
                                                byteCount: entry.biasSize,
                                                factor: routerScale)
                            modifiedAffine += 1
                        }
                    }
                } else if entry.name.hasSuffix(".shared_expert.down_proj.weight") {
                    if isMtpLayer {
                        if mtpSharedExpertScale != 1.0 {
                            try scaleBF16Region(offset: entry.scaleOffset,
                                                byteCount: entry.scaleSize,
                                                factor: mtpSharedExpertScale)
                            try scaleBF16Region(offset: entry.biasOffset,
                                                byteCount: entry.biasSize,
                                                factor: mtpSharedExpertScale)
                            modifiedAffine += 1
                            didCoAdaptMTP = true
                        }
                    } else if config.sharedExpertScale != 1.0,
                              let layerIdx = parseLayerIndex(from: entry.name),
                              layerIdx >= sharedStartLayer {
                        try scaleBF16Region(offset: entry.scaleOffset,
                                            byteCount: entry.scaleSize,
                                            factor: config.sharedExpertScale)
                        try scaleBF16Region(offset: entry.biasOffset,
                                            byteCount: entry.biasSize,
                                            factor: config.sharedExpertScale)
                        modifiedAffine += 1
                    }
                } else if entry.name.hasSuffix(".self_attn.o_proj.weight")
                            || entry.name.hasSuffix(".attn.o_proj.weight")
                            || entry.name.hasSuffix(".mlp.down_proj.weight") {
                    if isMtpLayer {
                        if mtpResidualScale != 1.0 {
                            try scaleBF16Region(offset: entry.scaleOffset,
                                                byteCount: entry.scaleSize,
                                                factor: mtpResidualScale)
                            try scaleBF16Region(offset: entry.biasOffset,
                                                byteCount: entry.biasSize,
                                                factor: mtpResidualScale)
                            modifiedAffine += 1
                            didCoAdaptMTP = true
                        }
                    } else if config.lateLayerResidualScale != 1.0,
                              let layerIdx = parseLayerIndex(from: entry.name),
                              layerIdx >= lateStartLayer {
                        try scaleBF16Region(offset: entry.scaleOffset,
                                            byteCount: entry.scaleSize,
                                            factor: config.lateLayerResidualScale)
                        try scaleBF16Region(offset: entry.biasOffset,
                                            byteCount: entry.biasSize,
                                            factor: config.lateLayerResidualScale)
                        modifiedAffine += 1
                    }
                }
            }
        }

        try Posix.fsync(weightsFd, path: weightsPath)
        let sha = try Sha256Stream.hashFile(path: weightsPath, tileBytes: 8 << 20)
        files["model_weights.bin"] = ["size": Int(fileSize), "sha256": sha]
        manifestRoot["files"] = files
        let newManifest = try JSONSerialization.data(
            withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try Posix.atomicWrite(newManifest, to: manifestPath, durableIn: gturboDirectory)

        let receiptPath = (gturboDirectory as NSString)
            .appendingPathComponent("verified-install.json")
        try? FileManager.default.removeItem(atPath: receiptPath)
        _ = try? VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: gturboDirectory))

        return SurgeryResult(modifiedNormTensors: modifiedNorms,
                             modifiedAffineTensors: modifiedAffine,
                             coAdaptedMTP: didCoAdaptMTP,
                             weightsSha256: sha)
    }

    private static func parseLayerIndex(from tensorName: String) -> Int? {
        guard !tensorName.hasPrefix("mtp.") else { return nil }
        guard let range = tensorName.range(of: "layers.") else { return nil }
        let suffix = tensorName[range.upperBound...]
        guard let dot = suffix.firstIndex(of: ".") else { return nil }
        return Int(suffix[..<dot])
    }

    // MARK: - Quantization and auxiliary MoE pool helpers

    private static func appendQuantizedBF16Matrix(
        shardFd: Int32,
        shardPath: String,
        sourceOffset: UInt64,
        rows: Int,
        cols: Int,
        bits: Int,
        payloadBase: UInt64,
        into dst: inout Data
    ) throws -> (weightOffset: UInt64, scaleOffset: UInt64, biasOffset: UInt64) {
        let weightOffset = payloadBase + UInt64(dst.count)
        let groups = cols / Int4AffineEncoder.groupSize
        var scales = [UInt16]()
        var biases = [UInt16]()
        scales.reserveCapacity(rows * groups)
        biases.reserveCapacity(rows * groups)

        let batchRows = min(rows, max(1, (8 << 20) / (cols * 2)))
        var floats = [Float](repeating: 0, count: batchRows * cols)
        var raw = [UInt16](repeating: 0, count: batchRows * cols)
        var row = 0
        while row < rows {
            let take = min(batchRows, rows - row)
            let byteCount = take * cols * 2
            try raw.withUnsafeMutableBytes {
                try Posix.preadAll(fd: shardFd, path: shardPath,
                                   buf: $0.baseAddress!, count: byteCount,
                                   offset: sourceOffset + UInt64(row * cols * 2))
            }
            for i in 0..<(take * cols) {
                floats[i] = Float(bitPattern: UInt32(raw[i]) << 16)
            }
            floats.withUnsafeBufferPointer { buf in
                let slice = UnsafeBufferPointer(rebasing: buf.prefix(take * cols))
                if bits == 8 {
                    let encoded = Int8AffineEncoder.encodeTensor(slice, rowLength: cols)
                    dst.append(contentsOf: encoded.packed)
                    scales.append(contentsOf: encoded.scales)
                    biases.append(contentsOf: encoded.biases)
                } else {
                    let encoded = Int4AffineEncoder.encodeTensor(slice, rowLength: cols)
                    dst.append(contentsOf: encoded.packed)
                    scales.append(contentsOf: encoded.scales)
                    biases.append(contentsOf: encoded.biases)
                }
            }
            row += take
        }

        let scaleOffset = payloadBase + UInt64(dst.count)
        scales.withUnsafeBufferPointer {
            dst.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self))
        }
        let biasOffset = payloadBase + UInt64(dst.count)
        biases.withUnsafeBufferPointer {
            dst.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self))
        }
        return (weightOffset, scaleOffset, biasOffset)
    }

    private static func writeQwen36MTPRoutedExperts(
        gturboDirectory: String,
        shardFd: Int32,
        shardPath: String,
        gateUp: SourceTensor,
        down: SourceTensor,
        numExperts: Int,
        hidden: Int,
        intermediate: Int
    ) throws -> (size: UInt64, sha256: String, expertStride: UInt64) {
        let groupsD = hidden / Int4AffineEncoder.groupSize
        let groupsF = intermediate / Int4AffineEncoder.groupSize
        let gateBytes = intermediate * hidden / 2 + 2 * intermediate * groupsD * 2
        let downBytes = hidden * intermediate / 2 + 2 * hidden * groupsF * 2
        let rawBlobBytes = UInt64(2 * gateBytes + downBytes)
        let pageSize = UInt64(getpagesize())
        let expertStride = ((rawBlobBytes + pageSize - 1) / pageSize) * pageSize

        let auxDir = (gturboDirectory as NSString).appendingPathComponent("packed_experts_mtp")
        try FileManager.default.createDirectory(atPath: auxDir, withIntermediateDirectories: true)
        let finalPath = (auxDir as NSString).appendingPathComponent("layer_00.bin")
        let totalSize = UInt64(numExperts) * expertStride
        if let existingSize = (try? FileManager.default.attributesOfItem(atPath: finalPath)[.size] as? NSNumber)?.uint64Value,
           existingSize == totalSize {
            let sha = try Sha256Stream.hashFile(path: finalPath, tileBytes: 8 << 20)
            return (totalSize, sha, expertStride)
        }
        let tmpPath = finalPath + ".mtp-attach.tmp"
        try? FileManager.default.removeItem(atPath: tmpPath)

        do {
            let outFd = try Posix.openCreateRW(tmpPath)
            defer { close(outFd) }

            let halfSourceBytes = UInt64(intermediate * hidden * 2)
            let gateUpExpertStride = 2 * halfSourceBytes
            let downExpertStride = UInt64(hidden * intermediate * 2)
            var blob = Data(capacity: Int(expertStride))

            for expert in 0..<numExperts {
                blob.removeAll(keepingCapacity: true)
                let gateOffset = gateUp.absoluteOffset + UInt64(expert) * gateUpExpertStride
                let upOffset = gateOffset + halfSourceBytes
                let downOffset = down.absoluteOffset + UInt64(expert) * downExpertStride

                _ = try appendQuantizedBF16Matrix(
                    shardFd: shardFd, shardPath: shardPath,
                    sourceOffset: gateOffset,
                    rows: intermediate, cols: hidden, bits: 4,
                    payloadBase: 0, into: &blob)
                _ = try appendQuantizedBF16Matrix(
                    shardFd: shardFd, shardPath: shardPath,
                    sourceOffset: upOffset,
                    rows: intermediate, cols: hidden, bits: 4,
                    payloadBase: 0, into: &blob)
                _ = try appendQuantizedBF16Matrix(
                    shardFd: shardFd, shardPath: shardPath,
                    sourceOffset: downOffset,
                    rows: hidden, cols: intermediate, bits: 4,
                    payloadBase: 0, into: &blob)

                if blob.count < Int(expertStride) {
                    blob.append(Data(count: Int(expertStride) - blob.count))
                }
                let fileOffset = UInt64(expert) * expertStride
                try blob.withUnsafeBytes {
                    try Posix.pwriteAll(fd: outFd, path: tmpPath,
                                        buf: $0.baseAddress!, count: blob.count,
                                        offset: fileOffset)
                }
            }
            try Posix.fsync(outFd, path: tmpPath)
        } catch {
            try? FileManager.default.removeItem(atPath: tmpPath)
            throw error
        }
        try Posix.rename(from: tmpPath, to: finalPath)
        let sha = try Sha256Stream.hashFile(path: finalPath, tileBytes: 8 << 20)
        return (totalSize, sha, expertStride)
    }

    // MARK: - Index encode/decode

    private static func readIndex(fd: Int32, path: String)
        throws -> (indexSize: UInt64, residentSize: UInt64, entries: [RawIndexEntry]) {
        var header = [UInt8](repeating: 0, count: headerBytes)
        try header.withUnsafeMutableBytes {
            try Posix.preadAll(fd: fd, path: path, buf: $0.baseAddress!,
                               count: headerBytes, offset: 0)
        }
        func u64(_ bytes: [UInt8], _ off: Int) -> UInt64 {
            var v: UInt64 = 0
            for i in (0..<8).reversed() { v = (v << 8) | UInt64(bytes[off + i]) }
            return v
        }
        let indexSize = u64(header, 0)
        let residentSize = u64(header, 8)
        let entryCount = Int(u64(header, 16))
        guard indexSize >= UInt64(headerBytes + entryCount * entryBytes) else {
            throw RepackError.configurationInvalid(detail: "\(path): corrupt index header")
        }
        var region = [UInt8](repeating: 0, count: Int(indexSize))
        try region.withUnsafeMutableBytes {
            try Posix.preadAll(fd: fd, path: path, buf: $0.baseAddress!,
                               count: Int(indexSize), offset: 0)
        }
        var entries: [RawIndexEntry] = []
        entries.reserveCapacity(entryCount)
        try region.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            func u64p(_ p: UnsafeRawPointer, _ off: Int) -> UInt64 {
                var v: UInt64 = 0
                let bytes = p.advanced(by: off).assumingMemoryBound(to: UInt8.self)
                for i in (0..<8).reversed() { v = (v << 8) | UInt64(bytes[i]) }
                return v
            }
            func u32p(_ p: UnsafeRawPointer, _ off: Int) -> UInt32 {
                var v: UInt32 = 0
                let bytes = p.advanced(by: off).assumingMemoryBound(to: UInt8.self)
                for i in (0..<4).reversed() { v = (v << 8) | UInt32(bytes[i]) }
                return v
            }
            for i in 0..<entryCount {
                let p = base.advanced(by: headerBytes + i * entryBytes)
                let nameOffset = Int(u32p(p, 0))
                let nameLength = Int(UInt16(p.advanced(by: 4)
                    .assumingMemoryBound(to: UInt8.self)[0])
                    | (UInt16(p.advanced(by: 5).assumingMemoryBound(to: UInt8.self)[0]) << 8))
                guard nameOffset >= headerBytes,
                      nameOffset + nameLength <= Int(indexSize) else {
                    throw RepackError.configurationInvalid(
                        detail: "\(path): entry \(i) name out of range")
                }
                let name = String(decoding: UnsafeRawBufferPointer(
                    start: base.advanced(by: nameOffset), count: nameLength), as: UTF8.self)
                entries.append(RawIndexEntry(
                    name: name,
                    dtype: p.advanced(by: 6).assumingMemoryBound(to: UInt8.self)[0],
                    fileOffset: u64p(p, 8),
                    sizeBytes: u64p(p, 16),
                    shape: (u32p(p, 24), u32p(p, 28), u32p(p, 32), u32p(p, 36)),
                    scaleOffset: u64p(p, 40),
                    scaleSize: u64p(p, 48),
                    biasOffset: u64p(p, 56),
                    biasSize: u64p(p, 64)))
            }
        }
        return (indexSize, residentSize, entries)
    }

    private static func encodeIndex(entries: [RawIndexEntry],
                                    indexSize: UInt64,
                                    residentSize: UInt64) -> Data {
        var image = Data(count: Int(indexSize))
        image.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexSize,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            let stringTableBase = headerBytes + entries.count * entryBytes
            var nameCursor = 0
            for (i, entry) in entries.enumerated() {
                let nameOffset = UInt32(stringTableBase + nameCursor)
                let nameBytes = Array(entry.name.utf8)
                memcpy(base.advanced(by: stringTableBase + nameCursor),
                       nameBytes, nameBytes.count)
                nameCursor += nameBytes.count
                let dst = base.advanced(by: headerBytes + i * entryBytes)
                var off = 0
                writeU32LE(dst, &off, nameOffset)
                writeU16LE(dst, &off, UInt16(entry.name.utf8.count))
                writeU8(dst, &off, entry.dtype)
                writeU8(dst, &off, 0)
                writeU64LE(dst, &off, entry.fileOffset)
                writeU64LE(dst, &off, entry.sizeBytes)
                writeU32LE(dst, &off, entry.shape.0)
                writeU32LE(dst, &off, entry.shape.1)
                writeU32LE(dst, &off, entry.shape.2)
                writeU32LE(dst, &off, entry.shape.3)
                writeU64LE(dst, &off, entry.scaleOffset)
                writeU64LE(dst, &off, entry.scaleSize)
                writeU64LE(dst, &off, entry.biasOffset)
                writeU64LE(dst, &off, entry.biasSize)
            }
        }
        return image
    }

    private static func writeU64LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt64) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            memcpy(buf.advanced(by: off), src.baseAddress!, 8)
        }
        off += 8
    }

    private static func writeU32LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt32) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            memcpy(buf.advanced(by: off), src.baseAddress!, 4)
        }
        off += 4
    }

    private static func writeU16LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt16) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            memcpy(buf.advanced(by: off), src.baseAddress!, 2)
        }
        off += 2
    }

    private static func writeU8(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt8) {
        buf.advanced(by: off).assumingMemoryBound(to: UInt8.self)[0] = v
        off += 1
    }
}
