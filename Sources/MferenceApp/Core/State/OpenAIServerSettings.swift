import Foundation
import Mference

public enum OpenAIServerBindMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case loopback
    case tailnet

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .loopback: "Loopback (127.0.0.1)"
        case .tailnet: "Tailnet (Tailscale)"
        }
    }
}

public enum OpenAIServerPromptCacheMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case singlePrefix = "single-prefix"
    case off = "off"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .singlePrefix: "Single Prefix (2-Slot LRU)"
        case .off: "Off"
        }
    }
}

public enum OpenAIServerVerifyMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case trustedReceipt = "trusted-receipt"
    case fullSha256 = "full-sha256"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .trustedReceipt: "Trusted Receipt (Fast)"
        case .fullSha256: "Full SHA-256 (Strict)"
        }
    }
}

public enum OpenAIServerSpecMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case auto
    case pld
    case off

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .auto: "Auto (MTP + PLD)"
        case .pld: "PLD Only (N-gram)"
        case .off: "Off (Single-Token)"
        }
    }
}

public struct OpenAIServerSettings: Codable, Equatable, Sendable {
    public static let fileName = "mac-app-server-settings.json"
    public static let currentVersion = 1

    public static let allowedContextLengths = AppContextLengthOption.allCases.map(\.tokens)
    public static let allowedSlotCounts = RuntimeConfiguration.allowedExpertCacheSlots
    public static let allowedPrefillChunks = RuntimeConfiguration.allowedPrefillChunkTokens
    public static let allowedQueueLimits = [1, 2, 4, 8, 16]
    public static let allowedMTPTopK = [4, 6, 8]
    public static let defaultModelId = "qwen3.6-35b-a3b"

    public var version: Int = currentVersion
    public var port: Int = 8080
    public var bindMode: OpenAIServerBindMode = .loopback
    public var modelId: String = defaultModelId
    public var maxContext: Int = 32_768
    public var expertCacheSlots: Int = 48
    public var prefillChunk: Int = 2_048
    public var promptCacheMode: OpenAIServerPromptCacheMode = .singlePrefix
    public var queueLimit: Int = 4
    public var verifyMode: OpenAIServerVerifyMode = .trustedReceipt
    public var specMode: OpenAIServerSpecMode = .auto
    public var adaptiveDrafting: Bool = true
    public var mtpTopK: Int = 6
    public var intentAlignerEnabled: Bool = true

    public init(
        version: Int = currentVersion,
        port: Int = 8080,
        bindMode: OpenAIServerBindMode = .loopback,
        modelId: String = defaultModelId,
        maxContext: Int = 32_768,
        expertCacheSlots: Int = 48,
        prefillChunk: Int = 2_048,
        promptCacheMode: OpenAIServerPromptCacheMode = .singlePrefix,
        queueLimit: Int = 4,
        verifyMode: OpenAIServerVerifyMode = .trustedReceipt,
        specMode: OpenAIServerSpecMode = .auto,
        adaptiveDrafting: Bool = true,
        mtpTopK: Int = 6,
        intentAlignerEnabled: Bool = true
    ) {
        self.version = version
        self.port = port
        self.bindMode = bindMode
        self.modelId = modelId
        self.maxContext = maxContext
        self.expertCacheSlots = expertCacheSlots
        self.prefillChunk = prefillChunk
        self.promptCacheMode = promptCacheMode
        self.queueLimit = queueLimit
        self.verifyMode = verifyMode
        self.specMode = specMode
        self.adaptiveDrafting = adaptiveDrafting
        self.mtpTopK = mtpTopK
        self.intentAlignerEnabled = intentAlignerEnabled
    }

    public func isValid() -> Bool {
        let trimmedID = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        return version == Self.currentVersion
            && (1...65_535).contains(port)
            && !trimmedID.isEmpty
            && Self.allowedContextLengths.contains(maxContext)
            && Self.allowedSlotCounts.contains(expertCacheSlots)
            && Self.allowedPrefillChunks.contains(prefillChunk)
            && queueLimit > 0
            && Self.allowedMTPTopK.contains(mtpTopK)
    }

    public func normalized() -> OpenAIServerSettings {
        var copy = self
        copy.version = Self.currentVersion
        copy.port = min(max(port, 1), 65_535)
        let trimmedID = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        copy.modelId = trimmedID.isEmpty ? Self.defaultModelId : trimmedID
        if !Self.allowedContextLengths.contains(copy.maxContext) {
            copy.maxContext = 32_768
        }
        if !Self.allowedSlotCounts.contains(copy.expertCacheSlots) {
            copy.expertCacheSlots = 48
        }
        if !Self.allowedPrefillChunks.contains(copy.prefillChunk) {
            copy.prefillChunk = 2_048
        }
        if copy.queueLimit <= 0 {
            copy.queueLimit = 4
        }
        if !Self.allowedMTPTopK.contains(copy.mtpTopK) {
            copy.mtpTopK = 6
        }
        return copy
    }
}

public enum OpenAIServerSettingsFileStore {
    public static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(OpenAIServerSettings.fileName, isDirectory: false)
    }

    public static func loadOrCreate(
        forModelDirectory modelDirectory: URL,
        fileManager: FileManager = .default
    ) -> OpenAIServerSettings {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                let decoded = try JSONDecoder().decode(OpenAIServerSettings.self, from: data)
                guard decoded.isValid() else { throw InvalidSettings() }
                return decoded
            } catch {
                try? fileManager.removeItem(at: fileURL)
            }
        }

        let defaults = OpenAIServerSettings()
        try? save(defaults, forModelDirectory: modelDirectory, fileManager: fileManager)
        return defaults
    }

    public static func save(
        _ settings: OpenAIServerSettings,
        forModelDirectory modelDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        let normalized = settings.normalized()
        guard normalized.isValid() else { throw InvalidSettings() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(normalized)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private struct InvalidSettings: Error {}
}
