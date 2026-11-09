import AppKit
import MferenceMacPresentation
import SwiftUI
import WebKit

struct MermaidDiagramCardView: View {
    let diagram: String

    @State private var mode: DisplayMode = .diagram
    @State private var isCopied = false
    @State private var copyTask: Task<Void, Never>?

    enum DisplayMode: String, CaseIterable {
        case diagram = "Diagram"
        case code = "Code"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header Bar
            HStack(spacing: 8) {
                HStack(spacing: 5) {
                    Image(systemName: "arrow.triangle.branch")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(MferenceMacTheme.accentColor)

                    Text("DIAGRAM")
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
                    copyDiagram()
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
                .help("Copy Mermaid definition")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(red: 0.08, green: 0.08, blue: 0.10))

            Divider()
                .background(Color(white: 0.15))

            // Body
            if mode == .diagram {
                MermaidWebView(diagram: diagram)
                    .frame(minHeight: 380)
                    .background(Color(red: 0.04, green: 0.04, blue: 0.06))
            } else {
                ScrollView(.horizontal, showsIndicators: true) {
                    Text(CodeSyntaxHighlighter().highlight(code: diagram, language: "mermaid"))
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

    private func copyDiagram() {
        let textToCopy = MermaidSanitizer.sanitize(diagram)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(textToCopy, forType: .string)

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

private struct MermaidWebView: NSViewRepresentable {
    let diagram: String

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        let sanitized = MermaidSanitizer.sanitize(diagram)
        let escaped = sanitized
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "`", with: "\\`")
            .replacingOccurrences(of: "$", with: "\\$")

        let html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    background-color: #0A0A0C;
                    box-sizing: border-box;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                }
                body {
                    padding: 18px 24px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    box-sizing: border-box;
                    overflow: auto;
                }
                .mermaid {
                    width: 100%;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .mermaid svg {
                    width: 100% !important;
                    height: 100% !important;
                    max-width: 100% !important;
                    max-height: 100% !important;
                    object-fit: contain;
                }
            </style>
        </head>
        <body>
            <div id="diagram-container" class="mermaid">
            \(escaped)
            </div>
            <script>
                window.addEventListener('error', function(e) {
                    var el = document.getElementById('diagram-container');
                    if (el && (!window.mermaid || !el.querySelector('svg'))) {
                        el.innerHTML = '<div style="color:#71717A;font-size:12px;text-align:center;padding:20px;">' +
                                       'Diagram preview unavailable offline.<br>' +
                                       '<span style="color:#3B82F6;">Switch to [Code] tab to view diagram source.</span>' +
                                       '</div>';
                    }
                });

                if (window.mermaid) {
                    try {
                        mermaid.initialize({
                            startOnLoad: true,
                            theme: 'dark',
                            themeVariables: {
                                darkMode: true,
                                background: '#0A0A0C',
                                primaryColor: '#1E3A8A',
                                primaryTextColor: '#F3F4F6',
                                primaryBorderColor: '#3B82F6',
                                lineColor: '#60A5FA',
                                secondaryColor: '#18181B',
                                tertiaryColor: '#111827'
                            }
                        });
                    } catch(err) {
                        console.error(err);
                    }
                } else {
                    setTimeout(function() {
                        var el = document.getElementById('diagram-container');
                        if (el && !el.querySelector('svg')) {
                            el.innerHTML = '<div style="color:#71717A;font-size:12px;text-align:center;padding:20px;">' +
                                           'Diagram preview unavailable.<br>' +
                                           '<span style="color:#3B82F6;">Switch to [Code] tab to view diagram source.</span>' +
                                           '</div>';
                        }
                    }, 1500);
                }
            </script>
        </body>
        </html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }
}
