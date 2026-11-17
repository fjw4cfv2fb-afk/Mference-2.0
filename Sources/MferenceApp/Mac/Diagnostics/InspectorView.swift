import AppKit
import MferenceAppCore
import MferenceMacPresentation
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel
    @State private var copiedFieldID: String?
    @State private var portInput: String = ""

    var body: some View {
        Form {
            modelSection
            openAIServerSection
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

    private func copyableValueRow(
        _ title: String,
        value: String,
        fieldID: String,
        monospaced: Bool = true
    ) -> some View {
        LabeledContent(title) {
            HStack(spacing: 6) {
                Text(value)
                    .font(monospaced ? .caption.monospaced() : .caption)
                    .truncationMode(.middle)
                    .lineLimit(1)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .help(value)
                Button {
                    copyToPasteboard(value, fieldID: fieldID)
                } label: {
                    Label(
                        copiedFieldID == fieldID ? "Copied" : "Copy \(title)",
                        systemImage: copiedFieldID == fieldID ? "checkmark.circle.fill" : "doc.on.doc"
                    )
                    .labelStyle(.iconOnly)
                    .foregroundStyle(copiedFieldID == fieldID ? MferenceMacTheme.accentColor : Color.secondary)
                }
                .buttonStyle(.borderless)
                .help(copiedFieldID == fieldID ? "Copied \(title)" : "Copy \(title)")
            }
        }
    }

    private var modelSection: some View {
        Section("Model") {
            copyableValueRow("Path", value: model.modelPathText, fieldID: "modelPath", monospaced: false)
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

    private var openAIServerSection: some View {
        Section("OpenAI API Server") {
            LabeledContent("Status") {
                HStack(spacing: 4) {
                    Circle()
                        .fill(model.openAIServer.isRunning ? Color.blue : (model.openAIServer.isStarting ? Color.orange : Color.secondary))
                        .frame(width: 8, height: 8)
                    Text(model.openAIServer.statusDescription)
                        .font(.caption)
                }
            }
            copyableValueRow("Base URL", value: model.openAIServer.openAIBaseURLString, fieldID: "baseURL")
            copyableValueRow("Model ID", value: model.openAIServer.modelId, fieldID: "modelID")
            LabeledContent("Port") {
                TextField("8080", text: $portInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption.monospacedDigit())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 72)
                    .disabled(model.openAIServer.isRunning || model.openAIServer.isStarting)
                    .onChange(of: portInput) { _, newValue in
                        let digits = newValue.filter(\.isNumber)
                        if digits != newValue {
                            portInput = digits
                        }
                        if let parsed = Int(digits), OpenAIServerManager.validPortRange.contains(parsed) {
                            model.openAIServer.port = parsed
                        }
                    }
                    .onSubmit {
                        let parsed = Int(portInput) ?? model.openAIServer.port
                        let clamped = OpenAIServerManager.clampPort(parsed)
                        model.openAIServer.port = clamped
                        portInput = String(clamped)
                    }
            }
            Button(model.openAIServer.isRunning ? "Stop OpenAI Server" : "Start OpenAI Server") {
                if !model.openAIServer.isRunning && !model.openAIServer.isStarting {
                    let parsed = Int(portInput) ?? model.openAIServer.port
                    let clamped = OpenAIServerManager.clampPort(parsed)
                    model.openAIServer.port = clamped
                    portInput = String(clamped)
                }
                model.toggleOpenAIServer()
            }
            .disabled(model.openAIServer.isStarting)
            .help(model.openAIServer.latestCommandPreview)
            Button {
                model.openAIServer.expertCacheSlots = model.runtimeOptions.expertCacheSlots
                model.openAIServer.maxContext = model.maxContextTokens
                let cmd = model.openAIServer.fullCommandString(
                    modelPath: model.modelPathText,
                    maxContext: model.maxContextTokens,
                    expertCacheSlots: model.runtimeOptions.expertCacheSlots
                )
                copyToPasteboard(cmd, fieldID: "serverCommand")
            } label: {
                Label(
                    copiedFieldID == "serverCommand" ? "Copied Server Command" : "Copy Server Command",
                    systemImage: copiedFieldID == "serverCommand" ? "checkmark.circle.fill" : "terminal"
                )
            }
            .buttonStyle(.borderless)
            .font(.caption)
            .help("Copy exact CLI command: \(model.openAIServer.fullCommandString(modelPath: model.modelPathText, maxContext: model.maxContextTokens, expertCacheSlots: model.runtimeOptions.expertCacheSlots))")
            if let error = model.openAIServer.lastError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
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
