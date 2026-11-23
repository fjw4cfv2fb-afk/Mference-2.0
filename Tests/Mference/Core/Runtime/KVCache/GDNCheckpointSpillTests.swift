import Foundation
import Testing
@testable import Mference

@Suite struct GDNCheckpointSpillTests {

    private static func makeSyntheticSnapshot(layerCount: Int, tag: UInt8) -> GDNStateManager.Snapshot {
        var states = [Data?](repeating: nil, count: layerCount)
        var tails = [Data?](repeating: nil, count: layerCount)
        for i in 0..<layerCount {
            if i % 4 != 3 {
                states[i] = Data((0..<256).map { UInt8(($0 + Int(tag) + i) & 0xFF) })
                tails[i] = Data((0..<64).map { UInt8(($0 * 3 + Int(tag) + i) & 0xFF) })
            }
        }
        return GDNStateManager.Snapshot(states: states, convTails: tails)
    }

    @Test func spillStoreRoundTripRingCapPruneAndCleanup() throws {
        let sandbox = FileManager.default.temporaryDirectory
            .appendingPathComponent("mference-spill-test-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: sandbox) }

        var capturedDir: URL?
        do {
            let store = GDNSnapshotSpillStore(maxEntries: 8, baseDirectory: sandbox)
            capturedDir = store.directoryURL
            // Lazy directory creation: no writes yet -> nothing created on disk.
            #expect(!FileManager.default.fileExists(atPath: store.directoryURL.path))
            #expect(store.latestPosition(atOrBefore: 1000) == nil)

            var expected: [Int: GDNStateManager.Snapshot] = [:]
            for step in 1...10 {
                let pos = step * 100
                let snap = Self.makeSyntheticSnapshot(layerCount: 40, tag: UInt8(step))
                expected[pos] = snap
                store.spill(snap, at: pos)
            }

            #expect(store.count == 8)
            #expect(!store.contains(position: 100))
            #expect(!store.contains(position: 200))
            #expect(store.contains(position: 300))
            #expect(store.contains(position: 1000))
            #expect(store.latestPosition(atOrBefore: 650) == 600)
            #expect(store.latestPosition(atOrBefore: 250) == nil)

            let restored = try #require(store.load(at: 500))
            let reference = try #require(expected[500])
            #expect(restored.states.count == reference.states.count)
            #expect(restored.convTails.count == reference.convTails.count)
            for i in 0..<40 {
                #expect(restored.states[i] == reference.states[i])
                #expect(restored.convTails[i] == reference.convTails[i])
            }

            store.prune(after: 600)
            #expect(store.count == 4)
            #expect(!store.contains(position: 700))
            #expect(!store.contains(position: 1000))
            #expect(store.contains(position: 600))
        }

        if let capturedDir {
            #expect(!FileManager.default.fileExists(atPath: capturedDir.path))
        }
    }
}
