import AppKit
import MferenceAppCore
import MferenceMacPresentation
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel
    @State private var copiedFieldID: String?
    @State private var portInput: String = ""
    @AppStorage("Mference.serverSettingsExpanded")
    private var showServerSettings = true

    var body: some View {
        Form {
            modelSection
            OpenAIServerInspectorSection(
                model: model,
                server: model.openAIServer,
                copiedFieldID: $copiedFieldID,
                portInput: $portInput,
                showServerSettings: $showServerSettings,
                onCopy: copyToPasteboard
            )
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color.black)
        .tint(MferenceMacTheme.accentColor)
        .accentColor(MferenceMacTheme.accentColor)
        .onAppear {
            portInput = String(model.openAIServer.port)
        }
        .onChange(of: model.openAIServer.port) { _, newValue in
            let text = String(newValue)
            if portInput != text {
                portInput = text
            }
        }
        .onChange(of: model.openAIServer.currentSettings) { _, _ in
            model.saveOpenAIServerSettings()
        }
        .task(id: copiedFieldID) {
            guard let id = copiedFieldID else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled, copiedFieldID == id else { return }
            copiedFieldID = nil
        }
    }

    private func copyToPasteboard(_ value: String, fieldID: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
        copiedFieldID = fieldID
    }

    private var modelSection: some View {
        Section("Model") {
            CopyableValueRow(
                title: "Path",
                value: model.modelPathText,
                monospaced: false,
                isCopied: copiedFieldID == "modelPath",
                onCopy: { copyToPasteboard(model.modelPathText, fieldID: "modelPath") }
            )
            Button("Choose Model Folder…") {
                ModelLocationPicker.choose(for: model)
            }
            .disabled(model.loadState.isLoading)
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
            }
            Divider()
            LabeledContent("State") {
                Text(model.presentation.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.requiresModelInstallation {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(model.installDescriptor.installedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent("Available") {
                        Text(MetricFormat.storage(requirement.availableBytes))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent("Context") {
                Picker("Context", selection: $model.maxContextTokens) {
                    ForEach(AppContextLengthOption.allCases) { option in
                        Text(option.menuLabel).tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            Text("More slots can improve decode speed by keeping more experts in memory, but they also use more RAM. Changes are compared with 4K context and 16 slots and apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var generationSection: some View {
        Section("Generation") {
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                        .tint(MferenceMacTheme.accentColor)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
                .tint(MferenceMacTheme.accentColor)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                    .fixedSize()
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .tint(MferenceMacTheme.accentColor)
                .disabled(!model.topKEnabled)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                            .tint(MferenceMacTheme.accentColor)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
                .toggleStyle(.switch)
                .tint(MferenceMacTheme.accentColor)
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .tint(MferenceMacTheme.accentColor)
                .labelsHidden()
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

}

private struct CopyableValueRow: View {
    let title: String
    let value: String
    var monospaced = true
    let isCopied: Bool
    let onCopy: @MainActor () -> Void

    var body: some View {
        LabeledContent(title) {
            HStack(spacing: 6) {
                Text(value)
                    .font(monospaced ? .caption.monospaced() : .caption)
                    .truncationMode(.middle)
                    .lineLimit(1)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .help(value)
                Button(action: onCopy) {
                    Label(
                        isCopied ? "Copied" : "Copy \(title)",
                        systemImage: isCopied ? "checkmark.circle.fill" : "doc.on.doc"
                    )
                    .labelStyle(.iconOnly)
                    .foregroundStyle(isCopied ? MferenceMacTheme.accentColor : Color.secondary)
                }
                .buttonStyle(.borderless)
                .help(isCopied ? "Copied \(title)" : "Copy \(title)")
            }
        }
    }
}

private struct ExplainedSetting<Content: View>: View {
    let explanation: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            content()
            Text(explanation)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct OpenAIServerInspectorSection: View {
    @Bindable var model: AppModel
    @Bindable var server: OpenAIServerManager
    @Binding var copiedFieldID: String?
    @Binding var portInput: String
    @Binding var showServerSettings: Bool
    let onCopy: @MainActor (String, String) -> Void

    private var serverLocked: Bool {
        server.isRunning || server.isStarting
    }

    var body: some View {
        Section("OpenAI API Server") {
            LabeledContent("Status") {
                HStack(spacing: 4) {
                    Circle()
                        .fill(server.isRunning ? Color.blue : (server.isStarting ? Color.orange : Color.secondary))
                        .frame(width: 8, height: 8)
                    Text(server.statusDescription)
                        .font(.caption)
                }
            }
            CopyableValueRow(
                title: "Base URL",
                value: server.openAIBaseURLString,
                isCopied: copiedFieldID == "baseURL",
                onCopy: { onCopy(server.openAIBaseURLString, "baseURL") }
            )
            CopyableValueRow(
                title: "Model ID",
                value: server.effectiveModelId,
                isCopied: copiedFieldID == "modelID",
                onCopy: { onCopy(server.effectiveModelId, "modelID") }
            )
            Button(server.isRunning ? "Stop OpenAI Server" : "Start OpenAI Server") {
                if !serverLocked {
                    let parsed = Int(portInput) ?? server.port
                    let clamped = OpenAIServerManager.clampPort(parsed)
                    server.port = clamped
                    portInput = String(clamped)
                }
                model.toggleOpenAIServer()
            }
            .disabled(server.isStarting)
            .help(server.latestCommandPreview)
            Button {
                let cmd = server.fullCommandString(modelPath: model.modelPathText)
                onCopy(cmd, "serverCommand")
            } label: {
                Label(
                    copiedFieldID == "serverCommand" ? "Copied Server Command" : "Copy Server Command",
                    systemImage: copiedFieldID == "serverCommand" ? "checkmark.circle.fill" : "terminal"
                )
            }
            .buttonStyle(.borderless)
            .font(.caption)
            .help("Copy exact CLI command: \(server.fullCommandString(modelPath: model.modelPathText))")

            DisclosureGroup("Server Settings & Impact", isExpanded: $showServerSettings) {
                if serverLocked {
                    Text("Stop the server to change settings. Edits take effect on next start.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }

                Group {
                    networkAndIdentityControls
                    memoryAndPrefillControls
                    speculationAndReliabilityControls
                    Button("Reset Server Defaults") {
                        model.resetOpenAIServerSettings()
                        portInput = String(server.port)
                    }
                    .buttonStyle(.borderless)
                    .font(.caption)
                }
                .disabled(serverLocked)
            }

            if let error = server.lastError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
    }

    @ViewBuilder
    private var networkAndIdentityControls: some View {
        ExplainedSetting(explanation: "TCP port clients connect to (`http://<host>:<port>/v1`). Change this if port 8080 is already taken.") {
            LabeledContent("Port") {
                TextField("8080", text: $portInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption.monospacedDigit())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 76)
                    .onChange(of: portInput) { _, newValue in
                        let digits = newValue.filter(\.isNumber)
                        if digits != newValue {
                            portInput = digits
                        }
                        if let parsed = Int(digits), OpenAIServerManager.validPortRange.contains(parsed) {
                            server.port = parsed
                        }
                    }
                    .onSubmit {
                        let parsed = Int(portInput) ?? server.port
                        let clamped = OpenAIServerManager.clampPort(parsed)
                        server.port = clamped
                        portInput = String(clamped)
                    }
            }
        }

        ExplainedSetting(explanation: "Loopback restricts the API to this Mac (127.0.0.1). Tailnet binds your Tailscale IPv4 so your other devices can reach it (requires Tailscale running).") {
            LabeledContent("Bind") {
                Picker("Bind", selection: $server.bindMode) {
                    ForEach(OpenAIServerBindMode.allCases) { mode in
                        Text(mode.label).tag(mode.rawValue)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
        }

        ExplainedSetting(explanation: "Model name advertised at `/v1/models` and matched in API requests. Rename only if a client expects a specific ID.") {
            LabeledContent("API Model ID") {
                TextField(OpenAIServerSettings.defaultModelId, text: $server.modelId)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption.monospaced())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 150)
                    .onSubmit {
                        server.modelId = server.effectiveModelId
                    }
            }
        }
    }

    @ViewBuilder
    private var memoryAndPrefillControls: some View {
        ExplainedSetting(explanation: "Maximum prompt + reply tokens per request. Larger context handles longer files and agent threads, while allocating more full-attention KV cache RAM.") {
            LabeledContent("Context") {
                Picker("Context", selection: $server.maxContext) {
                    ForEach(OpenAIServerSettings.allowedContextLengths, id: \.self) { tokens in
                        Text(tokens == 128_000 ? "128K" : "\(tokens / 1_024)K").tag(tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
        }

        ExplainedSetting(explanation: "Routed MoE experts kept per layer in unified RAM (40 layers). More slots cut SSD reads and speed up decode (32–48 fits 16 GiB Macs; 96–128 shines on 24 GiB+), at higher RAM use.") {
            LabeledContent("Expert Slots") {
                Picker("Expert Slots", selection: $server.expertCacheSlots) {
                    ForEach(OpenAIServerSettings.allowedSlotCounts, id: \.self) { slots in
                        Text("\(slots)").tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
        }

        ExplainedSetting(explanation: "Prompt tokens processed per GPU batch. Larger chunks (1024–4096) speed up prefill and reduce SSD re-reads, using slightly more temporary Metal scratch RAM.") {
            LabeledContent("Prefill Chunk") {
                Picker("Prefill Chunk", selection: $server.prefillChunk) {
                    ForEach(OpenAIServerSettings.allowedPrefillChunks, id: \.self) { chunk in
                        Text("\(chunk)").tag(chunk)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
        }

        ExplainedSetting(explanation: "Single Prefix reuses KV + GDN states across up to 2 LRU conversation slots so unchanged system/tool prompts don't re-prefill each turn. Off saves cache RAM.") {
            LabeledContent("Prompt Cache") {
                Picker("Prompt Cache", selection: $server.promptCacheMode) {
                    ForEach(OpenAIServerPromptCacheMode.allCases) { mode in
                        Text(mode.label).tag(mode.rawValue)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
        }
    }

    @ViewBuilder
    private var speculationAndReliabilityControls: some View {
        ExplainedSetting(explanation: "Max requests allowed to wait while another generation runs (excess requests get HTTP 429). Lower values keep stale client retries from piling up.") {
            LabeledContent("Queue Limit") {
                Picker("Queue Limit", selection: $server.queueLimit) {
                    ForEach(OpenAIServerSettings.allowedQueueLimits, id: \.self) { limit in
                        Text("\(limit)").tag(limit)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
        }

        ExplainedSetting(explanation: "Trusted Receipt checks file sizes for ~1s startup. Full SHA-256 re-hashes every expert slab on load for strict tamper checking, adding startup delay.") {
            LabeledContent("Startup Verify") {
                Picker("Startup Verify", selection: $server.verifyMode) {
                    ForEach(OpenAIServerVerifyMode.allCases) { mode in
                        Text(mode.label).tag(mode.rawValue)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
        }

        ExplainedSetting(explanation: "Auto runs the 19-tensor MTP drafter + zero-cost n-gram PLD at temp=0 for faster multi-token decode with identical greedy output. Off falls back to 1 token/step.") {
            LabeledContent("Speculation") {
                Picker("Speculation", selection: $server.specMode) {
                    ForEach(OpenAIServerSpecMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
        }

        ExplainedSetting(explanation: "Dynamically tunes draft depth K (1–3) by acceptance rate, thinking/code phase, and free RAM. Off locks a fixed draft depth.") {
            Toggle("Adaptive K", isOn: $server.adaptiveDrafting)
                .toggleStyle(.switch)
                .tint(MferenceMacTheme.accentColor)
                .disabled(server.specMode == .off)
        }

        ExplainedSetting(explanation: "Experts routed inside the MTP draft step (4 = fastest draft, 6 = balanced, 8 = highest acceptance). Lossless: verification always uses full top-8.") {
            LabeledContent("MTP Draft Top-K") {
                Picker("MTP Draft Top-K", selection: $server.mtpTopK) {
                    ForEach(OpenAIServerSettings.allowedMTPTopK, id: \.self) { k in
                        Text("Top-\(k)").tag(k)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .disabled(server.specMode != .auto)
            }
        }

        ExplainedSetting(explanation: "Enables repetition loop guards, proportional <think> limits, zero-think fast path on short prompts, and 16 GiB RAM K-caps. Off runs raw unguided generation.") {
            Toggle("Intent Aligner", isOn: $server.intentAlignerEnabled)
                .toggleStyle(.switch)
                .tint(MferenceMacTheme.accentColor)
        }
    }
}
