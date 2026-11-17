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
}
