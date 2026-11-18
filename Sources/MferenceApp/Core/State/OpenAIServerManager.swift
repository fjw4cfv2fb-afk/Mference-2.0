import Foundation
import Observation

public enum OpenAIServerStatus: Equatable, Sendable {
    case stopped
    case starting(port: Int)
    case running(port: Int, pid: Int32, modelId: String)
    case failed(String)

    public var isRunning: Bool {
        if case .running = self { return true }
        return false
    }

    public var isStarting: Bool {
        if case .starting = self { return true }
        return false
    }
}

@MainActor
@Observable
public final class OpenAIServerManager {
    public private(set) var status: OpenAIServerStatus = .stopped
    public private(set) var lastError: String?

    public var port: Int = 8080
    public var modelId: String = OpenAIServerSettings.defaultModelId
    public var maxContext: Int = 32768
    public var prefillChunk: Int = 2048
    public var promptCacheMode: String = OpenAIServerPromptCacheMode.singlePrefix.rawValue
    public var expertCacheSlots: Int = 48
    public var queueLimit: Int = 4
    public var verifyMode: String = OpenAIServerVerifyMode.trustedReceipt.rawValue
    public var speculationEnabled: Bool = true
    public var intentAlignerEnabled: Bool = true

    private var process: Process?
    private var healthCheckTask: Task<Void, Never>?
    private var terminationObserver: Any?

    public init() {
        // Observe app termination to ensure the server subprocess is terminated cleanly.
        terminationObserver = NotificationCenter.default.addObserver(
            forName: NSNotification.Name("NSApplicationWillTerminateNotification"),
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.stopSync()
            }
        }
    }

    isolated deinit {
        if let observer = terminationObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        stopSync()
    }

    public static let validPortRange = 1...65535

    public var isRunning: Bool {
        status.isRunning
    }

    public var isStarting: Bool {
        status.isStarting
    }

    public var resolvedPromptCacheMode: OpenAIServerPromptCacheMode {
        OpenAIServerPromptCacheMode(rawValue: promptCacheMode) ?? .singlePrefix
    }

    public var resolvedVerifyMode: OpenAIServerVerifyMode {
        OpenAIServerVerifyMode(rawValue: verifyMode) ?? .trustedReceipt
    }

    public var effectiveModelId: String {
        let trimmed = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? OpenAIServerSettings.defaultModelId : trimmed
    }

    public var serverURLString: String {
        "http://127.0.0.1:\(port)"
    }

    public var openAIBaseURLString: String {
        "http://127.0.0.1:\(port)/v1"
    }

    public var currentSettings: OpenAIServerSettings {
        OpenAIServerSettings(
            port: port,
            modelId: modelId,
            maxContext: maxContext,
            expertCacheSlots: expertCacheSlots,
            prefillChunk: prefillChunk,
            promptCacheMode: resolvedPromptCacheMode,
            queueLimit: queueLimit,
            verifyMode: resolvedVerifyMode,
            speculationEnabled: speculationEnabled,
            intentAlignerEnabled: intentAlignerEnabled
        ).normalized()
    }

    public func apply(_ settings: OpenAIServerSettings) {
        let clean = settings.normalized()
        port = clean.port
        modelId = clean.modelId
        maxContext = clean.maxContext
        expertCacheSlots = clean.expertCacheSlots
        prefillChunk = clean.prefillChunk
        promptCacheMode = clean.promptCacheMode.rawValue
        queueLimit = clean.queueLimit
        verifyMode = clean.verifyMode.rawValue
        speculationEnabled = clean.speculationEnabled
        intentAlignerEnabled = clean.intentAlignerEnabled
    }

    public func resetToDefaults() {
        apply(OpenAIServerSettings())
    }

    public static func clampPort(_ candidate: Int) -> Int {
        min(max(candidate, validPortRange.lowerBound), validPortRange.upperBound)
    }

    public var environmentOverrides: [String: String] {
        var env: [String: String] = [:]
        if !speculationEnabled {
            env["MFERENCE_SPEC"] = "off"
        }
        if !intentAlignerEnabled {
            env["MFERENCE_INTENT_ALIGN"] = "0"
        }
        return env
    }

    public var statusDescription: String {
        switch status {
        case .stopped:
            return "OpenAI Server Stopped"
        case .starting(let p):
            return "Starting on :\(p)…"
        case .running(let p, _, _):
            return "Running on :\(p)"
        case .failed(let err):
            return "Server Failed: \(err)"
        }
    }

    public func fullCommandString(
        modelPath: String? = nil,
        maxContext: Int? = nil,
        expertCacheSlots: Int? = nil
    ) -> String {
        let resolved = Self.resolveModelPath(candidate: modelPath)
        let ctx = maxContext ?? self.maxContext
        let slots = expertCacheSlots ?? self.expertCacheSlots
        let envPrefix = environmentOverrides
            .sorted(by: { $0.key < $1.key })
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: " ")
        let prefix = envPrefix.isEmpty ? "" : "\(envPrefix) "
        return "\(prefix).build/release/MferenceServer --model \(Self.shellQuote(resolved)) --port \(Self.clampPort(port)) --bind loopback --model-id \(Self.shellQuote(effectiveModelId)) --max-context \(ctx) --queue-limit \(max(1, queueLimit)) --prefill-chunk \(prefillChunk) --prompt-cache-mode \(resolvedPromptCacheMode.rawValue) --expert-cache-slots \(slots) --verify \(resolvedVerifyMode.rawValue)"
    }

    private static func shellQuote(_ value: String) -> String {
        let safe = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "/._-:@"))
        if !value.isEmpty && value.unicodeScalars.allSatisfy(safe.contains) {
            return value
        }
        return "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }

    public var latestCommandPreview: String {
        fullCommandString(modelPath: nil)
    }

    /// Toggles the OpenAI server between running and stopped states.
    public func toggle(modelPath: String?, onBeforeStart: (() -> Void)? = nil) {
        if isRunning || isStarting {
            stop()
        } else {
            start(modelPath: modelPath, onBeforeStart: onBeforeStart)
        }
    }

    /// Starts the OpenAI-compatible server with the latest command flags.
    public func start(modelPath: String?, onBeforeStart: (() -> Void)? = nil) {
        guard !isRunning && !isStarting else { return }

        // Give caller a chance to unload internal model or coordinate state
        onBeforeStart?()

        apply(currentSettings)
        status = .starting(port: port)
        lastError = nil

        let resolvedModelPath = Self.resolveModelPath(candidate: modelPath)
        guard FileManager.default.fileExists(atPath: resolvedModelPath) else {
            status = .failed("Model directory not found at \(resolvedModelPath)")
            lastError = "Model directory not found at \(resolvedModelPath)"
            return
        }

        guard let executableURL = Self.locateServerBinary() else {
            status = .failed("MferenceServer executable not found")
            lastError = "MferenceServer executable not found. Build it with swift build -c release."
            return
        }

        let proc = Process()
        proc.executableURL = executableURL
        proc.arguments = [
            "--model", resolvedModelPath,
            "--port", "\(port)",
            "--bind", "loopback",
            "--model-id", effectiveModelId,
            "--max-context", "\(maxContext)",
            "--queue-limit", "\(max(1, queueLimit))",
            "--prefill-chunk", "\(prefillChunk)",
            "--prompt-cache-mode", resolvedPromptCacheMode.rawValue,
            "--expert-cache-slots", "\(expertCacheSlots)",
            "--verify", resolvedVerifyMode.rawValue,
        ]
        let overrides = environmentOverrides
        if !overrides.isEmpty {
            var env = ProcessInfo.processInfo.environment
            for (k, v) in overrides {
                env[k] = v
            }
            proc.environment = env
        }

        let errPipe = Pipe()
        proc.standardError = errPipe
        let outPipe = Pipe()
        proc.standardOutput = outPipe

        proc.terminationHandler = { [weak self] p in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.process === p {
                    self.process = nil
                    if case .running = self.status {
                        self.status = .stopped
                    } else if case .starting = self.status {
                        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                        let errString = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
                        let msg = (errString?.isEmpty == false) ? errString! : "Process exited with code \(p.terminationStatus)"
                        self.status = .failed(msg)
                        self.lastError = msg
                    }
                }
            }
        }

        do {
            try proc.run()
            self.process = proc
            let targetPort = self.port
            let currentModelId = self.effectiveModelId
            let targetPID = proc.processIdentifier

            // Launch background task to poll health until server responds
            healthCheckTask?.cancel()
            healthCheckTask = Task { @MainActor [weak self] in
                guard let self else { return }
                var attempts = 0
                while attempts < 75 { // up to 15 seconds (75 * 200ms)
                    try? await Task.sleep(nanoseconds: 200_000_000)
                    if Task.isCancelled { return }
                    if self.process == nil || !proc.isRunning {
                        return
                    }
                    if await Self.checkHealth(port: targetPort) {
                        self.status = .running(port: targetPort, pid: targetPID, modelId: currentModelId)
                        return
                    }
                    attempts += 1
                }
                // If timed out, check if process is still alive
                if proc.isRunning {
                    // Mark as running anyway if process is alive
                    self.status = .running(port: targetPort, pid: targetPID, modelId: currentModelId)
                }
            }
        } catch {
            status = .failed("Failed to launch process: \(error.localizedDescription)")
            lastError = error.localizedDescription
            self.process = nil
        }
    }

    /// Stops the server process.
    public func stop() {
        healthCheckTask?.cancel()
        healthCheckTask = nil

        if let proc = process {
            process = nil
            if proc.isRunning {
                proc.terminate()
                Task {
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    if proc.isRunning {
                        kill(proc.processIdentifier, SIGKILL)
                    }
                }
            }
        }
        status = .stopped
    }

    private func stopSync() {
        healthCheckTask?.cancel()
        healthCheckTask = nil
        let proc = process
        process = nil
        Self.terminateProcess(proc)
        status = .stopped
    }

    private nonisolated static func terminateProcess(_ proc: Process?) {
        guard let proc, proc.isRunning else { return }
        proc.terminate()
    }

    private static func defaultRepoRoots() -> [String] {
        let fm = FileManager.default
        let roots = [
            fm.currentDirectoryPath,
            fm.homeDirectoryForCurrentUser.appendingPathComponent("Documents/Mference").path,
        ]
        var deduped: [String] = []
        for root in roots where !root.isEmpty && !deduped.contains(root) {
            deduped.append(root)
        }
        return deduped
    }

    /// Resolves the model path, preferring the upgraded MTP checkpoint and
    /// falling back to the base Qwen 3.6 directory if needed.
    public static func resolveModelPath(candidate: String?) -> String {
        let fm = FileManager.default
        if let candidate, !candidate.isEmpty, fm.fileExists(atPath: candidate) {
            return candidate
        }
        var firstMtpFallback: String?
        for root in defaultRepoRoots() {
            let mtpFallback = (root as NSString).appendingPathComponent("scratch/qwen36-mtp.gturbo")
            if firstMtpFallback == nil {
                firstMtpFallback = mtpFallback
            }
            if fm.fileExists(atPath: mtpFallback) {
                return mtpFallback
            }
        }
        for root in defaultRepoRoots() {
            let fallback = (root as NSString).appendingPathComponent("scratch/qwen36.gturbo")
            if fm.fileExists(atPath: fallback) {
                return fallback
            }
        }
        return candidate ?? firstMtpFallback ?? "scratch/qwen36-mtp.gturbo"
    }

    /// Probes the server health endpoint.
    public static func checkHealth(port: Int) async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:\(port)/health") else { return false }
        var request = URLRequest(url: url)
        request.timeoutInterval = 0.5
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let status = json["status"] as? String, status == "ok" {
                    return true
                }
            }
        } catch {
            return false
        }
        return false
    }

    /// Locates the `MferenceServer` executable across common bundle and build locations.
    public static func locateServerBinary() -> URL? {
        let fm = FileManager.default

        // 1. Next to the main executable or inside the app bundle
        if let exec = Bundle.main.executableURL {
            let sibling = exec.deletingLastPathComponent().appendingPathComponent("MferenceServer")
            if fm.isExecutableFile(atPath: sibling.path) {
                return sibling
            }
        }

        let bundleHelpers = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/MferenceServer")
        if fm.isExecutableFile(atPath: bundleHelpers.path) {
            return bundleHelpers
        }

        // 2. Local build products
        var candidates: [String] = []
        for root in defaultRepoRoots() {
            candidates.append(contentsOf: [
                (root as NSString).appendingPathComponent(".build/release/MferenceServer"),
                (root as NSString).appendingPathComponent(".build/arm64-apple-macosx/release/MferenceServer"),
                (root as NSString).appendingPathComponent(".build/debug/MferenceServer"),
                (root as NSString).appendingPathComponent(".build/arm64-apple-macosx/debug/MferenceServer"),
            ])
        }
        candidates.append(contentsOf: [
            "/usr/local/bin/MferenceServer",
            "/opt/homebrew/bin/MferenceServer",
        ])
        for candidate in candidates {
            if fm.isExecutableFile(atPath: candidate) {
                return URL(fileURLWithPath: candidate)
            }
        }

        return nil
    }
}
