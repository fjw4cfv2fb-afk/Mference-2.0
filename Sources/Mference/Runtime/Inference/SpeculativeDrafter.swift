import Foundation

/// Unified protocol for speculative decoding drafters.
///
/// Implementations include:
/// - `PromptLookupSpeculator`: Zero-VRAM N-gram pattern matching in context history.
/// - `Qwen36MTPSpeculator`: Checkpoint-native MTP neural draft module.
public protocol SpeculativeDrafter: AnyObject {
    var specStats: Qwen38ForwardRunner.MTPSpecStats { get }
    func reset()
    func prepareForContinuation(expectedPosition: Int) throws
    func consumePending(token: Int32, position: Int) throws -> UInt32?
    func canRunRound(position: Int) -> Bool
    func draft(bonus: Int32, position: Int, count: Int) throws -> [Int32]
    func recordAccepted(drafts: [Int32], emitted: [UInt32], position: Int, verifyNanos: UInt64)
    func notifyTokens(tokens: ArraySlice<Int32>, startPosition: Int)
}

extension SpeculativeDrafter {
    public func notifyTokens(tokens: ArraySlice<Int32>, startPosition: Int) {}
}
