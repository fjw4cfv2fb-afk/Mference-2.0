import AppKit
import MferenceMacPresentation
import SwiftUI

struct CodeBlockCardView: View {
    let language: String
    let code: String

    @State private var isCopied = false
    @State private var copyTask: Task<Void, Never>?

    private var displayLanguage: String {
        let trimmed = language.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "CODE" : trimmed.uppercased()
    }

    private var lineCount: Int {
        normalizedCode.components(separatedBy: "\n").count
    }

    private var normalizedCode: String {
        code.replacingOccurrences(of: "\t", with: "    ")
    }

    private var highlightedCode: AttributedString {
        CodeSyntaxHighlighter().highlight(code: normalizedCode, language: language)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header Bar
            HStack(spacing: 8) {
                HStack(spacing: 5) {
                    Image(systemName: "chevron.left.forwardslash.chevron.right")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(MferenceMacTheme.accentColor)

                    Text(displayLanguage)
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(Color(white: 0.85))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color(white: 0.12), in: Capsule())

                Text("\(lineCount) \(lineCount == 1 ? "line" : "lines")")
                    .font(.system(size: 11))
                    .foregroundStyle(Color(white: 0.45))

                Spacer()

                Button {
                    copyCode()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 11, weight: .medium))
                            .contentTransition(.symbolEffect(.replace))
                        Text(isCopied ? "Copied!" : "Copy")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(isCopied ? MferenceMacTheme.accentColor : Color(white: 0.75))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(white: 0.14), in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .help("Copy code to clipboard")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(red: 0.08, green: 0.08, blue: 0.10))

            Divider()
                .background(Color(white: 0.15))

            // Scrollable Code Body
            ScrollView(.horizontal, showsIndicators: true) {
                Text(highlightedCode)
                    .font(.system(size: 12.5, weight: .regular, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(red: 0.04, green: 0.04, blue: 0.06))
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(white: 0.18), lineWidth: 1)
        }
    }

    private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(normalizedCode, forType: .string)

        copyTask?.cancel()
        withAnimation(.easeIn(duration: 0.15)) {
            isCopied = true
        }

        copyTask = Task {
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            withAnimation(.easeOut(duration: 0.2)) {
                isCopied = false
            }
        }
    }
}
