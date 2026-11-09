import Foundation

@frozen
public struct PrefillMoEGroup: Equatable, Sendable {
    public var expert: UInt32
    public var pairStart: UInt32
    public var pairCount: UInt32

    public init(expert: UInt32, pairStart: UInt32, pairCount: UInt32) {
        self.expert = expert
        self.pairStart = pairStart
        self.pairCount = pairCount
    }
}

@frozen
public struct PrefillMoETile: Equatable, Sendable {
    public var groupStart: UInt32
    public var groupCount: UInt32
    public var pairStart: UInt32
    public var pairCount: UInt32

    public init(groupStart: UInt32, groupCount: UInt32, pairStart: UInt32, pairCount: UInt32) {
        self.groupStart = groupStart
        self.groupCount = groupCount
        self.pairStart = pairStart
        self.pairCount = pairCount
    }
}

public struct PrefillMoEGroupedRoutes: Equatable, Sendable {
    public let sortedPairs: [PrefillTokenExpertPair]
    public let perExpertOffsets: [UInt32]
    public let perExpertCounts: [UInt32]
    public let groups: [PrefillMoEGroup]
    public let tiles: [PrefillMoETile]
    public let queryCount: Int

    public var maxPairsPerExpert: Int {
        groups.map { Int($0.pairCount) }.max() ?? 0
    }

    public var maxPairsPerTile: Int {
        tiles.map { Int($0.pairCount) }.max() ?? 0
    }

    public var maxLiveExpertsPerTile: Int {
        tiles.map { Int($0.groupCount) }.max() ?? 0
    }
}

enum PrefillMoEGroupingError: Error, Equatable, CustomStringConvertible {
    case invalidQueryCount(Int)
    case invalidTopK(Int)
    case invalidNumExperts(Int)
    case invalidTileExpertCount(Int)
    case pairCountMismatch(expected: Int, actual: Int)
    case tokenOutOfRange(UInt32)
    case rankOutOfRange(UInt32)
    case expertOutOfRange(UInt32)
    case duplicateTokenRank(token: UInt32, rank: UInt32)
    case expertSortKeyCountMismatch(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .invalidQueryCount(let value):
            return "queryCount must be non-negative, got \(value)"
        case .invalidTopK(let value):
            return "topK must be positive, got \(value)"
        case .invalidNumExperts(let value):
            return "numExperts must be positive, got \(value)"
        case .invalidTileExpertCount(let value):
            return "tileExpertCount must be in 1...16, got \(value)"
        case .pairCountMismatch(let expected, let actual):
            return "expected \(expected) route pairs, got \(actual)"
        case .tokenOutOfRange(let value):
            return "route token \(value) is out of range"
        case .rankOutOfRange(let value):
            return "route rank \(value) is out of range"
        case .expertOutOfRange(let value):
            return "route expert \(value) is out of range"
        case .duplicateTokenRank(let token, let rank):
            return "duplicate route pair for token \(token), rank \(rank)"
        case .expertSortKeyCountMismatch(let expected, let actual):
            return "expected \(expected) expert sort keys, got \(actual)"
        }
    }
}

enum PrefillMoEGrouping {
    static func groupTokenExpertPairs(
        _ pairs: [PrefillTokenExpertPair],
        queryCount: Int,
        topK: Int,
        numExperts: Int,
        tileExpertCount: Int = 16,
        expertSortKeys: [UInt64]? = nil
    ) throws -> PrefillMoEGroupedRoutes {
        guard queryCount >= 0 else {
            throw PrefillMoEGroupingError.invalidQueryCount(queryCount)
        }
        guard topK > 0 else {
            throw PrefillMoEGroupingError.invalidTopK(topK)
        }
        guard numExperts > 0 else {
            throw PrefillMoEGroupingError.invalidNumExperts(numExperts)
        }
        guard (1...16).contains(tileExpertCount) else {
            throw PrefillMoEGroupingError.invalidTileExpertCount(tileExpertCount)
        }
        if let expertSortKeys, expertSortKeys.count != numExperts {
            throw PrefillMoEGroupingError.expertSortKeyCountMismatch(expected: numExperts,
                                                                    actual: expertSortKeys.count)
        }
        let expectedPairs = queryCount * topK
        guard pairs.count == expectedPairs else {
            throw PrefillMoEGroupingError.pairCountMismatch(expected: expectedPairs,
                                                           actual: pairs.count)
        }

        var isCanonicalOrder = true
        let topKU32 = UInt32(topK)
        let queryCountU32 = UInt32(queryCount)
        let numExpertsU32 = UInt32(numExperts)
        var counts = Array(repeating: UInt32(0), count: numExperts)

        for idx in 0..<expectedPairs {
            let pair = pairs[idx]
            guard pair.token < queryCountU32 else {
                throw PrefillMoEGroupingError.tokenOutOfRange(pair.token)
            }
            guard pair.rank < topKU32 else {
                throw PrefillMoEGroupingError.rankOutOfRange(pair.rank)
            }
            guard pair.expert < numExpertsU32 else {
                throw PrefillMoEGroupingError.expertOutOfRange(pair.expert)
            }
            let expectedToken = UInt32(idx / topK)
            let expectedRank = UInt32(idx % topK)
            if pair.token != expectedToken || pair.rank != expectedRank {
                isCanonicalOrder = false
            }
            counts[Int(pair.expert)] &+= 1
        }

        let sortedPairs: [PrefillTokenExpertPair]
        var offsets = Array(repeating: UInt32.max, count: numExperts)
        var groups: [PrefillMoEGroup] = []
        groups.reserveCapacity(min(numExperts, expectedPairs))

        if isCanonicalOrder {
            // Fast O(N + numExperts log numExperts) bucket sort when pairs
            // arrive in canonical (token, rank) order from PrefillRouter.
            var orderedExperts: [Int] = []
            orderedExperts.reserveCapacity(min(numExperts, expectedPairs))
            for expert in 0..<numExperts where counts[expert] > 0 {
                orderedExperts.append(expert)
            }
            if let expertSortKeys {
                orderedExperts.sort { lhs, rhs in
                    let lhsKey = expertSortKeys[lhs]
                    let rhsKey = expertSortKeys[rhs]
                    if lhsKey != rhsKey { return lhsKey < rhsKey }
                    return lhs < rhs
                }
            }

            var writeCursor = Array(repeating: 0, count: numExperts)
            var runningStart = 0
            for expert in orderedExperts {
                let count = Int(counts[expert])
                offsets[expert] = UInt32(runningStart)
                writeCursor[expert] = runningStart
                groups.append(PrefillMoEGroup(expert: UInt32(expert),
                                              pairStart: UInt32(runningStart),
                                              pairCount: UInt32(count)))
                runningStart += count
            }

            sortedPairs = Array(unsafeUninitializedCapacity: expectedPairs) { buffer, initializedCount in
                for idx in 0..<expectedPairs {
                    let pair = pairs[idx]
                    let e = Int(pair.expert)
                    let dst = writeCursor[e]
                    buffer.initializeElement(at: dst, to: pair)
                    writeCursor[e] = dst &+ 1
                }
                initializedCount = expectedPairs
            }
        } else {
            var seenTokenRanks: Set<UInt64> = []
            seenTokenRanks.reserveCapacity(pairs.count)
            for pair in pairs {
                let key = UInt64(pair.token) << 32 | UInt64(pair.rank)
                guard seenTokenRanks.insert(key).inserted else {
                    throw PrefillMoEGroupingError.duplicateTokenRank(token: pair.token,
                                                                    rank: pair.rank)
                }
            }

            sortedPairs = pairs.sorted {
                if let expertSortKeys {
                    let lhsKey = expertSortKeys[Int($0.expert)]
                    let rhsKey = expertSortKeys[Int($1.expert)]
                    if lhsKey != rhsKey { return lhsKey < rhsKey }
                }
                if $0.expert != $1.expert { return $0.expert < $1.expert }
                if $0.token != $1.token { return $0.token < $1.token }
                return $0.rank < $1.rank
            }

            var i = 0
            while i < sortedPairs.count {
                let expert = sortedPairs[i].expert
                let start = i
                while i < sortedPairs.count, sortedPairs[i].expert == expert {
                    i += 1
                }
                let count = i - start
                offsets[Int(expert)] = UInt32(start)
                counts[Int(expert)] = UInt32(count)
                groups.append(PrefillMoEGroup(expert: expert,
                                              pairStart: UInt32(start),
                                              pairCount: UInt32(count)))
            }
        }

        var tiles: [PrefillMoETile] = []
        var groupStart = 0
        while groupStart < groups.count {
            let groupEnd = min(groups.count, groupStart + tileExpertCount)
            let first = groups[groupStart]
            let last = groups[groupEnd - 1]
            let pairStart = first.pairStart
            let pairEnd = last.pairStart + last.pairCount
            tiles.append(PrefillMoETile(groupStart: UInt32(groupStart),
                                        groupCount: UInt32(groupEnd - groupStart),
                                        pairStart: pairStart,
                                        pairCount: pairEnd - pairStart))
            groupStart = groupEnd
        }

        return PrefillMoEGroupedRoutes(sortedPairs: sortedPairs,
                                       perExpertOffsets: offsets,
                                       perExpertCounts: counts,
                                       groups: groups,
                                       tiles: tiles,
                                       queryCount: queryCount)
    }
}
