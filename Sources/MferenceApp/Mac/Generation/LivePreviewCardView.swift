import AppKit
import MferenceMacPresentation
import SwiftUI
import WebKit

struct LivePreviewCardView: View {
    let language: String
    let code: String

    @State private var mode: DisplayMode = .preview
    @State private var isCopied = false
    @State private var copyTask: Task<Void, Never>?

    enum DisplayMode: String, CaseIterable {
        case preview = "Preview"
        case code = "Code"
    }

    private var displayBadge: String {
        language.lowercased() == "svg" ? "SVG" : "HTML"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header Bar
            HStack(spacing: 8) {
                HStack(spacing: 5) {
                    Image(systemName: language.lowercased() == "svg" ? "paintpalette" : "globe")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(MferenceMacTheme.accentColor)

                    Text(displayBadge)
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(Color(white: 0.85))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color(white: 0.12), in: Capsule())

                // Segmented Toggle
                Picker("", selection: $mode) {
                    ForEach(DisplayMode.allCases, id: \.self) { item in
                        Text(item.rawValue).tag(item)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 140)

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

            // Body
            if mode == .preview {
                HTMLWebView(content: code, isSVG: language.lowercased() == "svg")
                    .frame(minHeight: 360)
                    .background(Color(red: 0.04, green: 0.04, blue: 0.06))
            } else {
                ScrollView(.horizontal, showsIndicators: true) {
                    Text(CodeSyntaxHighlighter().highlight(code: code, language: language))
                        .font(.system(size: 12.5, weight: .regular, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .background(Color(red: 0.04, green: 0.04, blue: 0.06))
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(white: 0.18), lineWidth: 1)
        }
    }

    private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)

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

private struct HTMLWebView: NSViewRepresentable {
    let content: String
    let isSVG: Bool

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        let html: String
        if isSVG {
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {
                        margin: 0;
                        padding: 24px;
                        background-color: #0A0A0C;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        box-sizing: border-box;
                    }
                    svg {
                        max-width: 100%;
                        max-height: 80vh;
                        height: auto;
                    }
                </style>
            </head>
            <body>
                \(content)
            </body>
            </html>
            """
        } else {
            html = content.contains("<html") ? content : """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {
                        margin: 0;
                        padding: 16px;
                        background-color: #0A0A0C;
                        color: #E5E7EB;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    }
                </style>
            </head>
            <body>
                \(content)
            </body>
            </html>
            """
        }
        webView.loadHTMLString(html, baseURL: nil)
    }
}
