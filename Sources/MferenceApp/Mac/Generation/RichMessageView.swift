import AppKit
import MferenceMacPresentation
import SwiftUI

struct RichMessageView: View {
    let content: String

    private var blocks: [MarkdownBlock] {
        MarkdownBlockParser().parse(content)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(0..<blocks.count, id: \.self) { index in
                switch blocks[index] {
                case .prose(let text):
                    ProseBlockView(text: text)

                case .code(let language, let code):
                    CodeBlockCardView(language: language, code: code)

                case .table(let headers, let rows, let alignments):
                    MarkdownTableView(headers: headers, rows: rows, alignments: alignments)

                case .mathBlock(let formula):
                    DisplayMathCardView(formula: formula)

                case .mermaid(let diagram):
                    MermaidDiagramCardView(diagram: diagram)

                case .htmlPreview(let code):
                    LivePreviewCardView(language: "html", code: code)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct ProseBlockView: View {
    let text: String

    private var attributedContent: AttributedString {
        // Preprocess inline math $ ... $
        var processed = ""
        let formatter = LaTeXFormatter()
        let nsString = text as NSString
        if let mathRegex = try? NSRegularExpression(pattern: #"(?<!\$)\$(?!\$)([^\$\n]+)\$(?!\$)"#) {
            let matches = mathRegex.matches(in: text, range: NSRange(location: 0, length: nsString.length))
            var lastLocation = 0
            for match in matches {
                let prefix = nsString.substring(with: NSRange(location: lastLocation, length: match.range.location - lastLocation))
                processed += prefix
                let formula = nsString.substring(with: match.range(at: 1))
                processed += formatter.format(latex: formula)
                lastLocation = match.range.location + match.range.length
            }
            if lastLocation < nsString.length {
                processed += nsString.substring(from: lastLocation)
            }
        } else {
            processed = text
        }

        let rendered = ResponseMarkdownRenderer().render(processed).attributedString
        return (try? AttributedString(rendered, including: \.appKit)) ?? AttributedString(rendered.string)
    }

    var body: some View {
        Text(attributedContent)
            .textSelection(.enabled)
            .lineSpacing(3)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}
