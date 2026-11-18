import Foundation
import Testing
@testable import MferenceAppCore

@Suite struct OpenAIServerManagerTests {
    @MainActor
    @Test func initialStateIsStopped() {
        let manager = OpenAIServerManager()
        #expect(manager.status == .stopped)
        #expect(!manager.isRunning)
        #expect(!manager.isStarting)
        #expect(manager.port == 8080)
        #expect(manager.modelId == "qwen3.6-35b-a3b")
        #expect(manager.maxContext == 32768)
        #expect(manager.prefillChunk == 2048)
        #expect(manager.promptCacheMode == "single-prefix")
        #expect(manager.expertCacheSlots == 48)
        #expect(manager.verifyMode == "trusted-receipt")
        #expect(manager.serverURLString == "http://127.0.0.1:8080")
        #expect(manager.openAIBaseURLString == "http://127.0.0.1:8080/v1")
    }

    @MainActor
    @Test func openAIBaseURLAndPortUpdatesEndpointStrings() {
        let manager = OpenAIServerManager()
        manager.port = 8090
        #expect(manager.serverURLString == "http://127.0.0.1:8090")
        #expect(manager.openAIBaseURLString == "http://127.0.0.1:8090/v1")
        #expect(manager.latestCommandPreview.contains("--port 8090"))
        #expect(OpenAIServerManager.clampPort(0) == 1)
        #expect(OpenAIServerManager.clampPort(70000) == 65535)
        #expect(OpenAIServerManager.clampPort(18080) == 18080)
    }

    @MainActor
    @Test func latestCommandPreviewContainsAllRequiredFlags() {
        let manager = OpenAIServerManager()
        let cmd = manager.latestCommandPreview
        #expect(cmd.contains("--port 8080"))
        #expect(cmd.contains("--bind loopback"))
        #expect(cmd.contains("--model-id qwen3.6-35b-a3b"))
        #expect(cmd.contains("--max-context 32768"))
        #expect(cmd.contains("--prefill-chunk 2048"))
        #expect(cmd.contains("--prompt-cache-mode single-prefix"))
        #expect(cmd.contains("--expert-cache-slots 48"))
        #expect(cmd.contains("--verify trusted-receipt"))
    }

    @MainActor
    @Test func locateServerBinaryFindsExecutable() {
        let binary = OpenAIServerManager.locateServerBinary()
        #expect(binary != nil)
        if let binary {
            #expect(FileManager.default.isExecutableFile(atPath: binary.path))
        }
    }

    @MainActor
    @Test func resolveModelPathFallsBackToDefaultScratch() {
        let path = OpenAIServerManager.resolveModelPath(candidate: nil)
        #expect(!path.isEmpty)
        #expect(path.contains("qwen36-mtp.gturbo") || path.contains("qwen36.gturbo"))
    }

    @MainActor
    @Test func appModelCoordinatesWithServer() {
        let model = AppModel()
        #expect(!model.openAIServer.isRunning)
        #expect(!model.openAIServer.isStarting)

        // Stopping when already stopped is safe
        model.stopOpenAIServer()
        #expect(!model.openAIServer.isRunning)
    }

    @MainActor
    @Test func commandAndEnvironmentReflectEditedServerSettingsAndReset() {
        let manager = OpenAIServerManager()
        #expect(manager.environmentOverrides.isEmpty)

        manager.apply(OpenAIServerSettings(
            port: 9090,
            modelId: "custom-qwen",
            maxContext: 65_536,
            expertCacheSlots: 96,
            prefillChunk: 4_096,
            promptCacheMode: .off,
            queueLimit: 2,
            verifyMode: .fullSha256,
            speculationEnabled: false,
            intentAlignerEnabled: false))

        let cmd = manager.latestCommandPreview
        #expect(cmd.contains("MFERENCE_INTENT_ALIGN=0"))
        #expect(cmd.contains("MFERENCE_SPEC=off"))
        #expect(cmd.contains("--port 9090"))
        #expect(cmd.contains("--bind loopback"))
        #expect(cmd.contains("--model-id custom-qwen"))
        #expect(cmd.contains("--max-context 65536"))
        #expect(cmd.contains("--queue-limit 2"))
        #expect(cmd.contains("--prefill-chunk 4096"))
        #expect(cmd.contains("--prompt-cache-mode off"))
        #expect(cmd.contains("--expert-cache-slots 96"))
        #expect(cmd.contains("--verify full-sha256"))
        #expect(manager.openAIBaseURLString == "http://127.0.0.1:9090/v1")

        manager.resetToDefaults()
        #expect(manager.currentSettings == OpenAIServerSettings())
        #expect(manager.environmentOverrides.isEmpty)
        #expect(manager.openAIBaseURLString == "http://127.0.0.1:8080/v1")
    }

    @Test func serverSettingsNormalizeInvalidValuesAndPersist() throws {
        let invalid = OpenAIServerSettings(
            port: 999_999,
            modelId: "   ",
            maxContext: 1234,
            expertCacheSlots: 17,
            prefillChunk: 99,
            queueLimit: 0)
        #expect(!invalid.isValid())
        let normalized = invalid.normalized()
        #expect(normalized.isValid())
        #expect(normalized.port == 65_535)
        #expect(normalized.modelId == OpenAIServerSettings.defaultModelId)
        #expect(normalized.maxContext == 32_768)
        #expect(normalized.expertCacheSlots == 48)
        #expect(normalized.prefillChunk == 2_048)
        #expect(normalized.queueLimit == 4)

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("OpenAIServerSettingsTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDir = root.appendingPathComponent("qwen36-mtp.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let defaults = OpenAIServerSettingsFileStore.loadOrCreate(forModelDirectory: modelDir)
        #expect(defaults == OpenAIServerSettings())

        let custom = OpenAIServerSettings(
            port: 8181,
            modelId: "qwen-custom",
            maxContext: 16_384,
            expertCacheSlots: 64,
            prefillChunk: 1_024,
            promptCacheMode: .off,
            queueLimit: 8,
            verifyMode: .fullSha256,
            speculationEnabled: false,
            intentAlignerEnabled: false)
        try OpenAIServerSettingsFileStore.save(custom, forModelDirectory: modelDir)
        let reloaded = OpenAIServerSettingsFileStore.loadOrCreate(forModelDirectory: modelDir)
        #expect(reloaded == custom)
    }

    @MainActor
    @Test func appModelPersistsAndResetsServerSettings() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppModelServerSettingsTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDir = root.appendingPathComponent("qwen36-mtp.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let model = AppModel(modelDirectory: modelDir, settingsPersistenceEnabled: true)
        model.openAIServer.port = 8282
        model.openAIServer.prefillChunk = 4_096
        model.openAIServer.speculationEnabled = false
        model.saveOpenAIServerSettings()

        let relaunched = AppModel(modelDirectory: modelDir, settingsPersistenceEnabled: true)
        #expect(relaunched.openAIServer.port == 8282)
        #expect(relaunched.openAIServer.prefillChunk == 4_096)
        #expect(relaunched.openAIServer.speculationEnabled == false)

        relaunched.resetOpenAIServerSettings()
        let afterReset = AppModel(modelDirectory: modelDir, settingsPersistenceEnabled: true)
        #expect(afterReset.openAIServer.currentSettings == OpenAIServerSettings())
    }
}
