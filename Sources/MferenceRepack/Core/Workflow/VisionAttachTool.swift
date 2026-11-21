import Darwin
import Foundation

/// Streams Qwen 3.6's BF16 `vision_tower.*` weights from Hugging Face via HTTP
/// Range requests and writes them as an ephemeral safetensors sidecar
/// (`vision_weights.bin`) inside an existing `qwen36` `.gturbo` directory,
/// updating `manifest.json` (`files` + `sidecars.vision`) and
/// `verified-install.json` without altering `model_weights.bin` or MTP/surgery
/// state.
public enum VisionAttachTool {
    public static let defaultRepoID = "mlx-community/Qwen3.6-35B-A3B-4bit"
    public static let defaultRevision = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
    public static let shardFilename = "model-00001-of-00004.safetensors"
    public static let outputFilename = "vision_weights.bin"
    public static let expectedTensorCount = 333
    private static let maxHeaderBytes: UInt64 = 32 * 1024 * 1024
    private static let maxRangeGapBytes: UInt64 = 1_048_576
    private static let tensorAlignment: UInt64 = 64

    public struct Result: Sendable {
        public let tensorCount: Int
        public let bytesDownloaded: UInt64
        public let visionFileBytes: UInt64
        public let rangeRequestCount: Int
        public let visionSha256: String
        public let verifiedReceiptUpdated: Bool
    }

    private struct SourceTensor {
        let name: String
        let dtype: String
        let shape: [Int]
        let srcStart: UInt64
        let srcEnd: UInt64
        var outStart: UInt64 = 0
        var outEnd: UInt64 = 0

        var byteLength: UInt64 { srcEnd - srcStart }
    }

    private struct CoalescedInterval {
        let srcStart: UInt64
        let srcEnd: UInt64
        let tensorIndices: [Int]

        var byteLength: UInt64 { srcEnd - srcStart }
    }

    public static func run(
        gturboDirectory: String,
        repoID: String = defaultRepoID,
        revision: String = defaultRevision,
        overwrite: Bool = false,
        token: String? = ProcessInfo.processInfo.environment["HF_TOKEN"],
        baseURL: URL = URL(string: "https://huggingface.co")!,
        progress: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> Result {
        let manifestPath = (gturboDirectory as NSString)
            .appendingPathComponent("manifest.json")
        let visionPath = (gturboDirectory as NSString)
            .appendingPathComponent(outputFilename)

        guard FileManager.default.fileExists(atPath: manifestPath) else {
            throw RepackError.configurationInvalid(
                detail: "\(gturboDirectory) does not look like a .gturbo install (missing manifest.json)")
        }

        let manifestData = try Posix.readBoundedData(manifestPath, maximumBytes: 8 << 20)
        guard var manifestRoot = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              let arch = manifestRoot["arch"] as? [String: Any],
              var files = manifestRoot["files"] as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "\(manifestPath) is not a valid .gturbo manifest")
        }
        guard (arch["family"] as? String) == "qwen36" else {
            throw RepackError.configurationInvalid(
                detail: "--attach-vision currently supports qwen36 installs only (found \(arch["family"] ?? "unknown"))")
        }
        if FileManager.default.fileExists(atPath: visionPath), !overwrite {
            throw RepackError.configurationInvalid(
                detail: "\(visionPath) already exists; pass --overwrite to replace it")
        }

        let remote = HuggingFaceRemoteSource(
            repoID: repoID,
            requestedRevision: revision,
            token: token,
            baseURL: baseURL)

        progress("Resolving \(repoID)@\(revision.prefix(8))/\(shardFilename)...")
        let info = try await remote.resolveFileInfo(filename: shardFilename)

        // Read 8-byte safetensors header length.
        guard info.size >= 8 else {
            throw RepackError.remoteProtocolInvalid(detail: "\(shardFilename) too small (\(info.size) bytes)")
        }
        let lenFile = try await remote.downloadRangeToTempFile(
            filename: shardFilename, info: info, offset: 0, length: 8)
        let headerLen: UInt64
        do {
            defer { try? FileManager.default.removeItem(atPath: lenFile.path) }
            let lenData = try Posix.readBoundedData(lenFile.path, maximumBytes: 8)
            guard lenData.count == 8 else {
                throw RepackError.remoteProtocolInvalid(detail: "short safetensors header length")
            }
            headerLen = lenData.withUnsafeBytes {
                UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
            }
        }
        guard headerLen > 0, headerLen <= maxHeaderBytes, 8 + headerLen <= info.size else {
            throw RepackError.remoteProtocolInvalid(
                detail: "invalid safetensors header length \(headerLen) in \(shardFilename)")
        }

        // Read safetensors JSON header.
        let hdrFile = try await remote.downloadRangeToTempFile(
            filename: shardFilename, info: info, offset: 8, length: Int(headerLen))
        let headerData: Data
        do {
            defer { try? FileManager.default.removeItem(atPath: hdrFile.path) }
            headerData = try Posix.readBoundedData(hdrFile.path, maximumBytes: maxHeaderBytes)
        }
        guard let rawHeader = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw RepackError.remoteProtocolInvalid(detail: "invalid safetensors JSON header")
        }

        let sourcePayloadBase = 8 + headerLen
        var tensors: [SourceTensor] = []
        tensors.reserveCapacity(expectedTensorCount)
        for (name, value) in rawHeader {
            if name == "__metadata__" { continue }
            guard name.hasPrefix("vision_tower.") else { continue }
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shapeNums = entry["shape"] as? [NSNumber],
                  let offsetNums = entry["data_offsets"] as? [NSNumber],
                  offsetNums.count == 2 else {
                throw RepackError.remoteProtocolInvalid(detail: "malformed header entry for \(name)")
            }
            guard dtype == "BF16" else {
                throw RepackError.configurationInvalid(
                    detail: "expected BF16 dtype for \(name), got \(dtype)")
            }
            let s = offsetNums[0].uint64Value
            let e = offsetNums[1].uint64Value
            guard e >= s, sourcePayloadBase + e <= info.size else {
                throw RepackError.remoteProtocolInvalid(detail: "invalid byte range for \(name)")
            }
            tensors.append(SourceTensor(
                name: name,
                dtype: dtype,
                shape: shapeNums.map(\.intValue),
                srcStart: s,
                srcEnd: e))
        }

        guard tensors.count == expectedTensorCount else {
            throw RepackError.configurationInvalid(
                detail: "expected \(expectedTensorCount) vision_tower.* tensors in \(shardFilename), found \(tensors.count)")
        }

        tensors.sort { tensorSortKey($0.name) < tensorSortKey($1.name) }

        // Assign 64-byte aligned output offsets inside vision_weights.bin.
        var cursor: UInt64 = 0
        for i in 0..<tensors.count {
            cursor = alignUp(cursor, to: tensorAlignment)
            let len = tensors[i].byteLength
            tensors[i].outStart = cursor
            tensors[i].outEnd = cursor + len
            cursor += len
        }
        let payloadTotalBytes = cursor

        let paddedHeaderData = try encodeSafetensorsHeader(tensors: tensors)
        let outPayloadBase = 8 + UInt64(paddedHeaderData.count)
        let totalFileBytes = outPayloadBase + payloadTotalBytes

        // Coalesce source ranges sorted by source offset to minimize HTTP round-trips.
        let sortedIndicesBySource = tensors.indices.sorted { tensors[$0].srcStart < tensors[$1].srcStart }
        var intervals: [CoalescedInterval] = []
        var curStart: UInt64 = 0
        var curEnd: UInt64 = 0
        var curIndices: [Int] = []
        for idx in sortedIndicesBySource {
            let s = sourcePayloadBase + tensors[idx].srcStart
            let e = sourcePayloadBase + tensors[idx].srcEnd
            if curIndices.isEmpty {
                curStart = s
                curEnd = e
                curIndices = [idx]
            } else if s <= curEnd || s - curEnd <= maxRangeGapBytes {
                curEnd = max(curEnd, e)
                curIndices.append(idx)
            } else {
                intervals.append(CoalescedInterval(srcStart: curStart, srcEnd: curEnd, tensorIndices: curIndices))
                curStart = s
                curEnd = e
                curIndices = [idx]
            }
        }
        if !curIndices.isEmpty {
            intervals.append(CoalescedInterval(srcStart: curStart, srcEnd: curEnd, tensorIndices: curIndices))
        }

        let tmpPath = visionPath + ".attach.tmp"
        try? FileManager.default.removeItem(atPath: tmpPath)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        var bytesDownloaded: UInt64 = 8 + headerLen
        do {
            let outFd = try Posix.openCreateRW(tmpPath)
            defer { close(outFd) }
            try Posix.ftruncate(outFd, path: tmpPath, size: totalFileBytes)

            var lenLE = UInt64(paddedHeaderData.count).littleEndian
            try withUnsafeBytes(of: &lenLE) {
                try Posix.pwriteAll(fd: outFd, path: tmpPath, buf: $0.baseAddress!, count: 8, offset: 0)
            }
            try paddedHeaderData.withUnsafeBytes {
                try Posix.pwriteAll(fd: outFd, path: tmpPath, buf: $0.baseAddress!, count: paddedHeaderData.count, offset: 8)
            }

            let copyTile = 8 << 20
            let copyBuf = UnsafeMutableRawBufferPointer.allocate(byteCount: copyTile, alignment: 16_384)
            defer { copyBuf.deallocate() }

            for (i, interval) in intervals.enumerated() {
                let mib = Double(interval.byteLength) / (1024.0 * 1024.0)
                progress(String(format: "Downloading vision range %d/%d (%.1f MiB)...",
                                i + 1, intervals.count, mib))
                let tmpRange = try await remote.downloadRangeToTempFile(
                    filename: shardFilename,
                    info: info,
                    offset: interval.srcStart,
                    length: Int(interval.byteLength))
                bytesDownloaded += tmpRange.byteCount
                defer { try? FileManager.default.removeItem(atPath: tmpRange.path) }

                let srcFd = try Posix.openRead(tmpRange.path)
                defer { close(srcFd) }

                for tensorIdx in interval.tensorIndices {
                    let tensor = tensors[tensorIdx]
                    let relInRange = (sourcePayloadBase + tensor.srcStart) - interval.srcStart
                    var remaining = tensor.byteLength
                    var copied: UInt64 = 0
                    while remaining > 0 {
                        let take = Int(min(UInt64(copyTile), remaining))
                        try Posix.preadAll(
                            fd: srcFd,
                            path: tmpRange.path,
                            buf: copyBuf.baseAddress!,
                            count: take,
                            offset: relInRange + copied)
                        try Posix.pwriteAll(
                            fd: outFd,
                            path: tmpPath,
                            buf: copyBuf.baseAddress!,
                            count: take,
                            offset: outPayloadBase + tensor.outStart + copied)
                        copied += UInt64(take)
                        remaining -= UInt64(take)
                    }
                }
            }
            try Posix.fsync(outFd, path: tmpPath)
        }

        let visionSha = try Sha256Stream.hashFile(path: tmpPath, tileBytes: 8 << 20)
        try Posix.rename(from: tmpPath, to: visionPath)

        files[outputFilename] = ["size": Int(totalFileBytes), "sha256": visionSha]
        manifestRoot["files"] = files
        var sidecars = manifestRoot["sidecars"] as? [String: Any] ?? [:]
        sidecars["vision"] = [
            "carried": true,
            "tensorCount": tensors.count,
        ]
        manifestRoot["sidecars"] = sidecars

        let newManifest = try JSONSerialization.data(
            withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try Posix.atomicWrite(newManifest, to: manifestPath, durableIn: gturboDirectory)

        let receiptUpdated = try updateVerifiedInstallReceipt(
            gturboDirectory: gturboDirectory,
            manifestPath: manifestPath,
            manifestFiles: files,
            visionFileBytes: totalFileBytes,
            visionSha: visionSha)

        return Result(
            tensorCount: tensors.count,
            bytesDownloaded: bytesDownloaded,
            visionFileBytes: totalFileBytes,
            rangeRequestCount: intervals.count + 2,
            visionSha256: visionSha,
            verifiedReceiptUpdated: receiptUpdated)
    }

    private static func alignUp(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        let rem = value % alignment
        return rem == 0 ? value : (value + alignment - rem)
    }

    private static func tensorSortKey(_ name: String) -> (Int, Int, String) {
        if name.hasPrefix("vision_tower.patch_embed.") {
            return (0, 0, name)
        }
        if name.hasPrefix("vision_tower.pos_embed.") {
            return (1, 0, name)
        }
        if name.hasPrefix("vision_tower.blocks.") {
            let tail = name.dropFirst("vision_tower.blocks.".count)
            if let dot = tail.firstIndex(of: "."), let blockIdx = Int(tail[..<dot]) {
                return (2, blockIdx, name)
            }
            return (2, Int.max, name)
        }
        if name.hasPrefix("vision_tower.merger.") {
            return (3, 0, name)
        }
        return (4, 0, name)
    }

    private static func encodeSafetensorsHeader(tensors: [SourceTensor]) throws -> Data {
        var dict: [String: Any] = [
            "__metadata__": [
                "format": "pt",
                "mference_sidecar": "qwen36_vision_tower",
            ]
        ]
        for t in tensors {
            dict[t.name] = [
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [t.outStart, t.outEnd],
            ] as [String: Any]
        }
        var data = try JSONSerialization.data(
            withJSONObject: dict,
            options: [.sortedKeys, .withoutEscapingSlashes])
        let rem = data.count % Int(tensorAlignment)
        if rem != 0 {
            data.append(contentsOf: repeatElement(UInt8(0x20), count: Int(tensorAlignment) - rem))
        }
        return data
    }

    private struct ExistingReceipt: Decodable {
        struct FileEntry: Decodable {
            let size: UInt64
            let sha256: String
        }
        let schemaVersion: Int
        let modelDirectoryPath: String
        let sourceRepoID: String?
        let sourceRevision: String?
        let files: [String: FileEntry]
    }

    private static func updateVerifiedInstallReceipt(
        gturboDirectory: String,
        manifestPath: String,
        manifestFiles: [String: Any],
        visionFileBytes: UInt64,
        visionSha: String
    ) throws -> Bool {
        let root = URL(fileURLWithPath: gturboDirectory).standardizedFileURL
        let receiptPath = root.appendingPathComponent(VerifiedInstallReceiptWriter.fileName).path

        let manifestSize: UInt64 = try {
            let fd = try Posix.openRead(manifestPath)
            defer { close(fd) }
            return try Posix.fileSize(fd: fd, path: manifestPath)
        }()
        let manifestSha = try Sha256Stream.hashFile(path: manifestPath, noCache: true)

        if let receiptData = try? Posix.readBoundedData(receiptPath, maximumBytes: 8 << 20),
           let existing = try? JSONDecoder().decode(ExistingReceipt.self, from: receiptData),
           existing.schemaVersion == 1,
           existing.modelDirectoryPath == root.path {
            var outputFiles: [RepackAudit.OutputFile] = []
            outputFiles.reserveCapacity(manifestFiles.count)
            var allMatched = true
            for relPath in manifestFiles.keys.sorted() {
                if relPath == outputFilename {
                    outputFiles.append(RepackAudit.OutputFile(
                        relativePath: outputFilename,
                        size: visionFileBytes,
                        sha256: visionSha))
                    continue
                }
                guard let entryDict = manifestFiles[relPath] as? [String: Any],
                      let expectedSize = (entryDict["size"] as? NSNumber)?.uint64Value,
                      let expectedSha = entryDict["sha256"] as? String,
                      let receiptEntry = existing.files[relPath],
                      receiptEntry.size == expectedSize,
                      receiptEntry.sha256.lowercased() == expectedSha.lowercased() else {
                    allMatched = false
                    break
                }
                let filePath = root.appendingPathComponent(relPath).path
                var st = stat()
                guard stat(filePath, &st) == 0,
                      st.st_size >= 0,
                      UInt64(st.st_size) == expectedSize else {
                    allMatched = false
                    break
                }
                outputFiles.append(RepackAudit.OutputFile(
                    relativePath: relPath,
                    size: expectedSize,
                    sha256: expectedSha))
            }
            if allMatched {
                let encoded = try VerifiedInstallReceiptWriter.encode(
                    outputDir: root.path,
                    manifestSha256: manifestSha,
                    manifestSize: manifestSize,
                    sourceRepoID: existing.sourceRepoID,
                    sourceRevision: existing.sourceRevision,
                    toolVersion: "MferenceRepack verify-install",
                    files: outputFiles)
                try Posix.atomicWrite(encoded, to: receiptPath, durableIn: gturboDirectory)
                return true
            }
        }

        try? FileManager.default.removeItem(atPath: receiptPath)
        _ = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: gturboDirectory))
        return true
    }
}
