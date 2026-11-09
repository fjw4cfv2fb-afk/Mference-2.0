import AppKit
import MferenceAppCore
import MferenceMacPresentation
import SwiftUI
import Vision

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var isImportingDocuments = false
    @State private var isExtractingDocuments = false
    @State private var isDropTargeted = false
    @State private var documentImportError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.promptAttachments.isEmpty {
                attachments
            }
            if let documentImportError {
                Text(documentImportError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            editor
            footer
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(MferenceMacTheme.surfaceBackgroundColor)
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(
                            isDropTargeted
                                ? MferenceMacTheme.accentColor
                                : MferenceMacTheme.surfaceBorderColor,
                            lineWidth: isDropTargeted ? 2 : 1)
                }
        }
        .overlay {
            if isDropTargeted {
                dropOverlay
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard !model.isRunning else { return false }
            importDocuments(urls)
            return true
        } isTargeted: { targeted in
            guard !model.isRunning else {
                isDropTargeted = false
                return
            }
            withAnimation(.easeInOut(duration: 0.18)) {
                isDropTargeted = targeted
            }
        }
        .fileImporter(
            isPresented: $isImportingDocuments,
            allowedContentTypes: DocumentTextExtractor.supportedContentTypes + [.folder, .image, .plainText, .sourceCode, .json, .commaSeparatedText],
            allowsMultipleSelection: true,
            onCompletion: handleDocumentSelection)
    }

    private var editor: some View {
        TextEditor(text: $model.promptText)
            .accessibilityLabel("Prompt")
            .font(.body)
            .scrollContentBackground(.hidden)
            .focused($promptFocused)
            .frame(height: editorHeight)
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    // Matches the NSTextView text origin: 5pt line fragment
                    // padding, no vertical inset.
                    Text("Prompt")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
    }

    private var editorHeight: CGFloat {
        model.promptText.isEmpty ? 46 : 84
    }

    private var footer: some View {
        HStack(spacing: 10) {
            attachDocumentAction
            promptTips
            Spacer()
            clearAction
            GenerateControl(model: model)
        }
    }

    private var attachments: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(model.promptAttachments) { attachment in
                    HStack(spacing: 7) {
                        Image(systemName: "doc.text")
                            .foregroundStyle(MferenceMacTheme.accentColor)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(attachment.fileName)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                            Text(attachmentDetail(attachment))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            model.removePromptAttachment(id: attachment.id)
                        } label: {
                            Label("Remove \(attachment.fileName)", systemImage: "xmark.circle.fill")
                                .labelStyle(.iconOnly)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .disabled(model.isRunning)
                    }
                    .padding(.leading, 10)
                    .padding(.trailing, 7)
                    .padding(.vertical, 7)
                    .background(MferenceMacTheme.surfaceBackgroundColor, in: .capsule)
                    .overlay {
                        Capsule().stroke(MferenceMacTheme.surfaceBorderColor, lineWidth: 1)
                    }
                    .help(attachment.wasTruncatedDuringExtraction
                          ? "Text was truncated during local extraction."
                          : "Text extracted locally for this prompt.")
                }
            }
        }
        .scrollIndicators(.hidden)
    }

    private var attachDocumentAction: some View {
        Button {
            documentImportError = nil
            isImportingDocuments = true
        } label: {
            Group {
                if isExtractingDocuments {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Label("Attach documents", systemImage: "paperclip")
                        .labelStyle(.iconOnly)
                }
            }
            .frame(width: 28, height: 28)
            .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .disabled(model.isRunning || isExtractingDocuments)
        .help("Attach PDF, Word, PowerPoint, or Excel files")
        .accessibilityLabel(isExtractingDocuments
                            ? "Extracting document text"
                            : "Attach documents")
    }

    private var promptTips: some View {
        Button {
            showingPromptTips.toggle()
        } label: {
            Label("Prompt tips", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Prompt tips")
        .popover(isPresented: $showingPromptTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            promptGuide
        }
    }

    private var promptGuide: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting this model")
                .font(.headline)

            tipSection("Ask for a clear task",
                       "Say what you want the model to create, explain, plan, or transform. Put the essential context in the same prompt.")
            tipSection("Shape the answer",
                       "Specify a useful length, sections, tone, or output format. Concrete constraints work better than a long list of vague preferences.")
            tipSection("Anchor important facts",
                       "Include facts the answer must preserve and say what should be checked. Generated factual claims can still be wrong or outdated.")
            tipSection("For code and calculations",
                       "Provide types, dimensions, interfaces, edge cases, or a small scaffold. Compile or run the result before relying on it.")
            tipSection("Try a focused revision",
                       "If the answer drifts, shorten the task and make the missing requirement explicit. The default temperature is 0.20 for steadier responses.")
        }
        .font(.callout)
        .frame(width: 390, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .fontWeight(.semibold)
            Text(detail)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func attachmentDetail(_ attachment: AppPromptAttachment) -> String {
        let count = attachment.characterCount.formatted(.number.notation(.compactName))
        let suffix = attachment.wasTruncatedDuringExtraction ? " • truncated" : ""
        return "\(attachment.formatLabel) • \(count) chars\(suffix)"
    }

    private func handleDocumentSelection(_ result: Result<[URL], any Error>) {
        switch result {
        case .success(let urls):
            importDocuments(urls)
        case .failure(let error):
            documentImportError = error.localizedDescription
        }
    }

    private func importDocuments(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        // Only the attach button is disabled while extraction runs, so the
        // selection can move on before it finishes.
        let targetChatID = model.selectedChatID
        isExtractingDocuments = true
        documentImportError = nil

        Task {
            let outcomes = await Task.detached(priority: .userInitiated) {
                var allOutcomes: [DocumentImportOutcome] = []

                for url in urls {
                    let didAccess = url.startAccessingSecurityScopedResource()
                    defer {
                        if didAccess {
                            url.stopAccessingSecurityScopedResource()
                        }
                    }

                    var isDir: ObjCBool = false
                    if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                        let folderFiles = Self.collectSupportedFiles(from: url)
                        if folderFiles.isEmpty {
                            allOutcomes.append(.failure(
                                fileName: url.lastPathComponent,
                                message: "Folder contains no supported documents or text files"))
                        } else {
                            let folderName = url.lastPathComponent
                            let rootPath = url.standardizedFileURL.path
                            for fileURL in folderFiles {
                                let filePath = fileURL.standardizedFileURL.path
                                let relPath: String
                                if filePath.hasPrefix(rootPath) {
                                    var suffix = String(filePath.dropFirst(rootPath.count))
                                    if suffix.hasPrefix("/") { suffix = String(suffix.dropFirst()) }
                                    relPath = "\(folderName)/\(suffix)"
                                } else {
                                    relPath = "\(folderName)/\(fileURL.lastPathComponent)"
                                }

                                do {
                                    allOutcomes.append(.success(try Self.extractDocument(from: fileURL, customFileName: relPath)))
                                } catch {
                                    allOutcomes.append(.failure(fileName: relPath, message: error.localizedDescription))
                                }
                            }
                        }
                    } else {
                        do {
                            allOutcomes.append(.success(try Self.extractDocument(from: url, customFileName: url.lastPathComponent)))
                        } catch {
                            allOutcomes.append(.failure(fileName: url.lastPathComponent, message: error.localizedDescription))
                        }
                    }
                }

                return allOutcomes
            }.value

            var failures: [String] = []
            for outcome in outcomes {
                switch outcome {
                case .success(let document):
                    model.addPromptAttachment(
                        AppPromptAttachment(
                            fileName: document.fileName,
                            formatLabel: document.formatLabel,
                            extractedText: document.text,
                            wasTruncatedDuringExtraction: document.wasTruncated),
                        toChatID: targetChatID)
                case .failure(let fileName, let message):
                    failures.append("\(fileName): \(message)")
                }
            }
            documentImportError = failures.isEmpty
                ? nil
                : failures.joined(separator: "\n")
            isExtractingDocuments = false
        }
    }

    private var dropOverlay: some View {
        RoundedRectangle(cornerRadius: 22)
            .fill(MferenceMacTheme.surfaceBackgroundColor.opacity(0.94))
            .overlay {
                RoundedRectangle(cornerRadius: 22)
                    .strokeBorder(
                        MferenceMacTheme.accentColor,
                        style: StrokeStyle(lineWidth: 2, dash: [7, 5]))
            }
            .overlay {
                VStack(spacing: 5) {
                    Image(systemName: "arrow.down.doc.fill")
                        .font(.title3)
                        .foregroundStyle(MferenceMacTheme.accentColor)
                    Text("Drop documents or folders to attach")
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(.primary)
                    Text("Folders, PDF, Word, Excel, PowerPoint, code, or text files")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .allowsHitTesting(false)
            .transition(.opacity)
    }

    nonisolated private static let supportedExtensions: Set<String> = [
        "pdf", "docx", "pptx", "xlsx",
        "png", "jpg", "jpeg", "heic", "tiff", "webp",
        "txt", "text", "log", "rtf", "md", "markdown", "rst", "tex",
        "json", "jsonl", "csv", "tsv", "xml", "yaml", "yml", "toml", "ini", "conf", "plist",
        "swift", "py", "js", "ts", "jsx", "tsx", "html", "htm", "css", "scss",
        "c", "cpp", "cc", "cxx", "h", "hpp", "m", "mm", "rs", "go", "java", "kt",
        "rb", "php", "sh", "bash", "zsh", "sql", "graphql", "proto"
    ]

    nonisolated private static let imageExtensions: Set<String> = [
        "png", "jpg", "jpeg", "heic", "tiff", "webp"
    ]

    nonisolated private static let ignoredDirectoryNames: Set<String> = [
        "node_modules", "Pods", "Carthage", "DerivedData", "vendor",
        "__pycache__", ".venv", "venv", "dist", "out", "target", "bin", "obj"
    ]

    nonisolated private static func collectSupportedFiles(from folderURL: URL, maxFiles: Int = 60) -> [URL] {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: folderURL.path, isDirectory: &isDir), isDir.boolValue else {
            return [folderURL]
        }

        var collected: [URL] = []
        let enumerator = fm.enumerator(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants])

        while let fileURL = enumerator?.nextObject() as? URL {
            if collected.count >= maxFiles { break }

            let name = fileURL.lastPathComponent
            if ignoredDirectoryNames.contains(name) {
                enumerator?.skipDescendants()
                continue
            }

            let ext = fileURL.pathExtension.lowercased()
            if supportedExtensions.contains(ext) {
                var isFileDir: ObjCBool = false
                if fm.fileExists(atPath: fileURL.path, isDirectory: &isFileDir), !isFileDir.boolValue {
                    collected.append(fileURL)
                }
            }
        }
        return collected
    }

    nonisolated private static func extractDocument(from url: URL, customFileName: String) throws -> ExtractedPromptDocument {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let ext = url.pathExtension.lowercased()
        if imageExtensions.contains(ext) {
            return try extractImageText(from: url, fileName: customFileName)
        }

        do {
            let doc = try DocumentTextExtractor.extract(from: url)
            return ExtractedPromptDocument(
                fileName: customFileName,
                formatLabel: doc.formatLabel,
                text: doc.text,
                wasTruncated: doc.wasTruncated)
        } catch DocumentTextExtractionError.unsupportedFormat {
            return try extractTextDocument(from: url, fileName: customFileName)
        }
    }

    nonisolated private static func extractImageText(from url: URL, fileName: String) throws -> ExtractedPromptDocument {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }

        guard let image = NSImage(contentsOf: url),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw DocumentTextExtractionError.unreadableFile(fileName)
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw DocumentTextExtractionError.unreadableFile(fileName)
        }

        let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
        let text = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)

        guard !text.isEmpty else {
            throw DocumentTextExtractionError.noExtractableText(fileName)
        }

        let maxChars = DocumentTextExtractor.maximumExtractedCharacters
        let clipped = String(text.prefix(maxChars))
        return ExtractedPromptDocument(
            fileName: fileName,
            formatLabel: "Image OCR",
            text: clipped,
            wasTruncated: clipped.count < text.count)
    }

    nonisolated private static func extractTextDocument(from url: URL, fileName: String) throws -> ExtractedPromptDocument {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let ext = url.pathExtension.lowercased()
        let textExtensions: [String: String] = [
            "txt": "Text", "text": "Text", "log": "Log", "rtf": "Rich Text",
            "md": "Markdown", "markdown": "Markdown", "rst": "ReST", "tex": "LaTeX",
            "json": "JSON", "jsonl": "JSONL", "csv": "CSV", "tsv": "TSV",
            "xml": "XML", "yaml": "YAML", "yml": "YAML", "toml": "TOML", "html": "HTML", "htm": "HTML", "css": "CSS", "scss": "SCSS",
            "plist": "Property List", "ini": "Config", "conf": "Config",
            "py": "Python", "swift": "Swift", "js": "JavaScript", "ts": "TypeScript",
            "jsx": "React", "tsx": "React", "c": "C", "cpp": "C++", "cc": "C++", "cxx": "C++", "h": "C Header",
            "hpp": "C++ Header", "m": "Objective-C", "mm": "Objective-C++", "rs": "Rust", "go": "Go", "rb": "Ruby",
            "java": "Java", "kt": "Kotlin", "sh": "Shell", "bash": "Bash", "zsh": "Zsh", "sql": "SQL",
            "graphql": "GraphQL", "proto": "Protobuf"
        ]

        guard let label = textExtensions[ext] else {
            throw DocumentTextExtractionError.unsupportedFormat(fileName)
        }

        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let rawString = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .ascii) else {
            throw DocumentTextExtractionError.unreadableFile(fileName)
        }

        let normalized = rawString
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !normalized.isEmpty else {
            throw DocumentTextExtractionError.noExtractableText(fileName)
        }

        let maxChars = DocumentTextExtractor.maximumExtractedCharacters
        let text = String(normalized.prefix(maxChars))
        return ExtractedPromptDocument(
            fileName: fileName,
            formatLabel: label,
            text: text,
            wasTruncated: text.count < normalized.count)
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused = true
            } label: {
                Label("Clear prompt", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear prompt")
        } else if !model.isRunning && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label("Clear chat history", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear chat history")
        }
    }
}

private enum DocumentImportOutcome: Sendable {
    case success(ExtractedPromptDocument)
    case failure(fileName: String, message: String)
}
