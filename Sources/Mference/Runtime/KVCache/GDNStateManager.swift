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
