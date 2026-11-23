import Darwin
import Foundation
import Metal

/// Fixed-size recurrent state for gated-DeltaNet linear-attention layers
/// (`LayerKind.linear`). Unlike KV rows, this state does not grow with
/// context: each linear layer owns
///
///  - a delta-rule state `S`, FP32 `[numVHeads, valueHeadDim, keyHeadDim]`
///    (2 MiB per layer at Qwen 3.6's 32x128x128), and
///  - a causal-conv tail of the last `convKernelSize - 1` pre-activation
///    `mixed_qkv` rows, FP16 `[convKernelSize - 1, convDim]`.
///
/// Buffers are allocated once in `init`; the decode hot path never allocates.
/// `reset()` zero-fills explicitly — the recurrence and the conv both define
/// the empty-context state as zeros, and zeroing 60-odd MiB per generation
/// start is cheap next to a prefill.
public final class GDNStateManager {
    public let config: ArchConfig

    /// Non-nil only at indices whose layer mask is 2.
    private let stateBuffers: [MTLBuffer?]
    private let convTailBuffers: [MTLBuffer?]
    var diagnosticBufferBytes: UInt64 {
        uniqueBufferBytes((stateBuffers + convTailBuffers).compactMap { $0 })
    }

    public let stateBytesPerLayer: Int
    public let convTailBytesPerLayer: Int

    private static let fp32Size = 4
    private static let fp16Size = 2

    public init(device: MTLDevice, config: ArchConfig) throws {
        self.config = config
        let la = config.linearAttention
        let stateBytes = la.numVHeads * la.valueHeadDim * la.keyHeadDim * Self.fp32Size
        let convTailBytes = max(0, la.convKernelSize - 1) * la.qkvDim * Self.fp16Size
        self.stateBytesPerLayer = stateBytes
        self.convTailBytesPerLayer = convTailBytes

        var states: [MTLBuffer?] = []
        var tails: [MTLBuffer?] = []
        states.reserveCapacity(config.numLayers)
        tails.reserveCapacity(config.numLayers)

        for layer in 0..<config.numLayers {
            guard config.layerIsLinear(layer) else {
                states.append(nil)
                tails.append(nil)
                continue
            }
            precondition(stateBytes > 0 && convTailBytes > 0,
                         "linear layer present but linearAttention config is empty")
            guard let state = device.makeBuffer(length: stateBytes,
                                                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            state.label = "gdn.state.layer\(layer)"
            guard let tail = device.makeBuffer(length: convTailBytes,
                                               options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            tail.label = "gdn.convtail.layer\(layer)"
            states.append(state)
            tails.append(tail)
        }
        self.stateBuffers = states
        self.convTailBuffers = tails
        zeroAll()
    }

    /// Delta-rule state `S` for a linear layer.
    public func stateBuffer(layer: Int) -> MTLBuffer {
        guard let buffer = stateBuffers[layer] else {
            preconditionFailure("layer \(layer) is not a linear-attention layer")
        }
        return buffer
    }

    /// Rolling window of the last `convKernelSize - 1` mixed_qkv rows.
    public func convTailBuffer(layer: Int) -> MTLBuffer {
        guard let buffer = convTailBuffers[layer] else {
            preconditionFailure("layer \(layer) is not a linear-attention layer")
        }
        return buffer
    }

    public struct Snapshot: Sendable {
        public var states: [Data?]
        public var convTails: [Data?]

        public init(layerCount: Int) {
            self.states = [Data?](repeating: nil, count: layerCount)
            self.convTails = [Data?](repeating: nil, count: layerCount)
        }

        public init(states: [Data?], convTails: [Data?]) {
            self.states = states
            self.convTails = convTails
        }
    }

    public func makeSnapshot() -> Snapshot {
        let states = stateBuffers.map { buf -> Data? in
            guard let buf else { return nil }
            return Data(bytes: buf.contents(), count: buf.length)
        }
        let tails = convTailBuffers.map { buf -> Data? in
            guard let buf else { return nil }
            return Data(bytes: buf.contents(), count: buf.length)
        }
        return Snapshot(states: states, convTails: tails)
    }

    public func restore(_ snapshot: Snapshot) {
        guard snapshot.states.count == stateBuffers.count,
              snapshot.convTails.count == convTailBuffers.count else { return }
        for i in stateBuffers.indices {
            if let dst = stateBuffers[i], let src = snapshot.states[i], src.count == dst.length {
                src.copyBytes(to: dst.contents().assumingMemoryBound(to: UInt8.self), count: dst.length)
            }
            if let dst = convTailBuffers[i], let src = snapshot.convTails[i], src.count == dst.length {
                src.copyBytes(to: dst.contents().assumingMemoryBound(to: UInt8.self), count: dst.length)
            }
        }
    }

    public func isLinear(layer: Int) -> Bool { stateBuffers[layer] != nil }

    /// Reset all recurrent state to the empty-context value (zeros).
    public func reset() {
        zeroAll()
    }

    private func zeroAll() {
        for buffer in stateBuffers {
            if let buffer { memset(buffer.contents(), 0, buffer.length) }
        }
        for buffer in convTailBuffers {
            if let buffer { memset(buffer.contents(), 0, buffer.length) }
        }
    }
}

/// Bounded ring of evicted turn-boundary GDN snapshots spilled to SSD via
/// uncached POSIX I/O (`F_NOCACHE`), preserving rewind capability across deep
/// multi-turn sessions or >2-chunk prefills at zero steady-state RAM cost.
final class GDNSnapshotSpillStore {
    private static let magic: UInt64 = 0x4D4647444E535031 // "MFGDNSP1"

    let directoryURL: URL
    private let maxEntries: Int
    private var directoryCreated = false
    private var entries: [Int: URL] = [:]

    init(maxEntries: Int = 8, baseDirectory: URL? = nil) {
        self.maxEntries = max(1, maxEntries)
        let root = baseDirectory ?? FileManager.default.temporaryDirectory
        self.directoryURL = root.appendingPathComponent(
            "mference-gdn-spill-\(UUID().uuidString)", isDirectory: true)
    }

    deinit {
        clear(removeDirectory: true)
    }

    var count: Int { entries.count }

    func contains(position: Int) -> Bool {
        entries[position] != nil
    }

    func latestPosition(atOrBefore maxPosition: Int) -> Int? {
        entries.keys.filter { $0 <= maxPosition }.max()
    }

    func spill(_ snapshot: GDNStateManager.Snapshot, at position: Int) {
        guard snapshot.states.count == snapshot.convTails.count else { return }
        if !directoryCreated {
            do {
                try FileManager.default.createDirectory(
                    at: directoryURL, withIntermediateDirectories: true)
                directoryCreated = true
            } catch {
                return
            }
        }
        if entries[position] == nil {
            while entries.count >= maxEntries, let oldest = entries.keys.min() {
                if let oldURL = entries.removeValue(forKey: oldest) {
                    unlink(oldURL.path)
                }
            }
        }
        let finalURL = directoryURL.appendingPathComponent("pos_\(position).gdn")
        let tmpURL = directoryURL.appendingPathComponent("pos_\(position).tmp")
        let fd = open(tmpURL.path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        guard fd >= 0 else { return }
        _ = fcntl(fd, F_NOCACHE, 1)

        var ok = writeUInt64(fd, Self.magic)
            && writeUInt32(fd, UInt32(snapshot.states.count))
        if ok {
            for i in snapshot.states.indices {
                if !writeOptionalData(fd, snapshot.states[i])
                    || !writeOptionalData(fd, snapshot.convTails[i]) {
                    ok = false
                    break
                }
            }
        }
        close(fd)
        guard ok, rename(tmpURL.path, finalURL.path) == 0 else {
            unlink(tmpURL.path)
            return
        }
        entries[position] = finalURL
    }

    func load(at position: Int) -> GDNStateManager.Snapshot? {
        guard let url = entries[position] else { return nil }
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            entries.removeValue(forKey: position)
            return nil
        }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)

        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size >= 12 else { return nil }
        let fileSize = Int(st.st_size)
        var bytesRead = 0

        guard let magic = readUInt64(fd, bytesRead: &bytesRead, limit: fileSize),
              magic == Self.magic,
              let layerCountU32 = readUInt32(fd, bytesRead: &bytesRead, limit: fileSize) else {
            return nil
        }
        let layerCount = Int(layerCountU32)
        guard layerCount >= 0, layerCount <= 4096 else { return nil }

        var states: [Data?] = []
        var tails: [Data?] = []
        states.reserveCapacity(layerCount)
        tails.reserveCapacity(layerCount)
        for _ in 0..<layerCount {
            guard let stateData = readOptionalData(fd, bytesRead: &bytesRead, limit: fileSize),
                  let tailData = readOptionalData(fd, bytesRead: &bytesRead, limit: fileSize) else {
                return nil
            }
            states.append(stateData)
            tails.append(tailData)
        }
        guard bytesRead == fileSize else { return nil }
        return GDNStateManager.Snapshot(states: states, convTails: tails)
    }

    func prune(after maxPosition: Int) {
        let staleKeys = entries.keys.filter { $0 > maxPosition }
        for key in staleKeys {
            if let url = entries.removeValue(forKey: key) {
                unlink(url.path)
            }
        }
    }

    func clear(removeDirectory: Bool) {
        for url in entries.values {
            unlink(url.path)
        }
        entries.removeAll()
        if removeDirectory && directoryCreated {
            try? FileManager.default.removeItem(at: directoryURL)
            directoryCreated = false
        }
    }

    // MARK: - Uncached POSIX I/O Helpers

    private func writeAll(_ fd: Int32, _ ptr: UnsafeRawPointer, count: Int) -> Bool {
        var written = 0
        while written < count {
            let n = Darwin.write(fd, ptr.advanced(by: written), count - written)
            if n > 0 {
                written += n
            } else if n < 0 && errno == EINTR {
                continue
            } else {
                return false
            }
        }
        return true
    }

    private func writeUInt32(_ fd: Int32, _ value: UInt32) -> Bool {
        var le = value.littleEndian
        return withUnsafeBytes(of: &le) { raw in
            writeAll(fd, raw.baseAddress!, count: 4)
        }
    }

    private func writeUInt64(_ fd: Int32, _ value: UInt64) -> Bool {
        var le = value.littleEndian
        return withUnsafeBytes(of: &le) { raw in
            writeAll(fd, raw.baseAddress!, count: 8)
        }
    }

    private func writeOptionalData(_ fd: Int32, _ data: Data?) -> Bool {
        guard let data else {
            return writeUInt32(fd, UInt32.max)
        }
        guard data.count < Int(UInt32.max),
              writeUInt32(fd, UInt32(data.count)) else {
            return false
        }
        if data.isEmpty { return true }
        return data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return false }
            return writeAll(fd, base, count: data.count)
        }
    }

    private func readExact(_ fd: Int32, _ dst: UnsafeMutableRawPointer, count: Int,
                           bytesRead: inout Int, limit: Int) -> Bool {
        guard count >= 0, bytesRead + count <= limit else { return false }
        var got = 0
        while got < count {
            let n = Darwin.read(fd, dst.advanced(by: got), count - got)
            if n > 0 {
                got += n
            } else if n < 0 && errno == EINTR {
                continue
            } else {
                return false
            }
        }
        bytesRead += count
        return true
    }

    private func readUInt32(_ fd: Int32, bytesRead: inout Int, limit: Int) -> UInt32? {
        var raw: UInt32 = 0
        let ok = withUnsafeMutableBytes(of: &raw) { buf in
            readExact(fd, buf.baseAddress!, count: 4, bytesRead: &bytesRead, limit: limit)
        }
        return ok ? UInt32(littleEndian: raw) : nil
    }

    private func readUInt64(_ fd: Int32, bytesRead: inout Int, limit: Int) -> UInt64? {
        var raw: UInt64 = 0
        let ok = withUnsafeMutableBytes(of: &raw) { buf in
            readExact(fd, buf.baseAddress!, count: 8, bytesRead: &bytesRead, limit: limit)
        }
        return ok ? UInt64(littleEndian: raw) : nil
    }

    private func readOptionalData(_ fd: Int32, bytesRead: inout Int, limit: Int) -> Data?? {
        guard let lenU32 = readUInt32(fd, bytesRead: &bytesRead, limit: limit) else {
            return nil
        }
        if lenU32 == UInt32.max {
            return .some(nil)
        }
        let count = Int(lenU32)
        guard count >= 0, bytesRead + count <= limit else { return nil }
        if count == 0 {
            return .some(Data())
        }
        var data = Data(count: count)
        let ok = data.withUnsafeMutableBytes { buf in
            readExact(fd, buf.baseAddress!, count: count, bytesRead: &bytesRead, limit: limit)
        }
        return ok ? .some(data) : nil
    }
}
