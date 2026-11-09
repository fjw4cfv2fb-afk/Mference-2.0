import AppKit
import MferenceMacPresentation
import SwiftUI

struct MarkdownTableView: View {
    let headers: [String]
    let rows: [[String]]
    let alignments: [MarkdownBlock.TableAlignment]

    @State private var isCopied = false
    @State private var copyTask: Task<Void, Never>?

    private func alignment(for index: Int) -> Alignment {
        guard index < alignments.count else { return .leading }
        switch alignments[index] {
        case .left: return .leading
        case .center: return .center
        case .right: return .trailing
        }
    }

    private func textAlignment(for index: Int) -> TextAlignment {
        guard index < alignments.count else { return .leading }
        switch alignments[index] {
        case .left: return .leading
        case .center: return .center
        case .right: return .trailing
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header bar with Copy Table button
            HStack {
                HStack(spacing: 5) {
                    Image(systemName: "tablecells")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(MferenceMacTheme.accentColor)

                    Text("TABLE")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(Color(white: 0.85))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color(white: 0.12), in: Capsule())

                Text("\(rows.count) \(rows.count == 1 ? "row" : "rows")")
                    .font(.system(size: 11))
                    .foregroundStyle(Color(white: 0.45))

                Spacer()

                Button {
                    copyTable()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 11, weight: .medium))
                            .contentTransition(.symbolEffect(.replace))
                        Text(isCopied ? "Copied!" : "Copy Table")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(isCopied ? MferenceMacTheme.accentColor : Color(white: 0.75))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(white: 0.14), in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .help("Copy table as Markdown")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(red: 0.08, green: 0.08, blue: 0.10))

            Divider()
                .background(Color(white: 0.15))

            // Table Content
            ScrollView(.horizontal, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 0) {
                    // Headers
                    HStack(spacing: 0) {
                        ForEach(0..<headers.count, id: \.self) { colIndex in
                            Text(LocalizedStringKey(headers[colIndex]))
                                .font(.system(size: 12.5, weight: .semibold))
                                .foregroundStyle(Color(white: 0.95))
                                .multilineTextAlignment(textAlignment(for: colIndex))
                                .frame(minWidth: 100, alignment: alignment(for: colIndex))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 9)

                            if colIndex < headers.count - 1 {
                                Divider()
                                    .background(Color(white: 0.15))
                            }
                        }
                    }
                    .background(Color(red: 0.10, green: 0.10, blue: 0.13))

                    Divider()
                        .background(Color(white: 0.20))

                    // Rows
                    ForEach(0..<rows.count, id: \.self) { rowIndex in
                        let row = rows[rowIndex]
                        HStack(spacing: 0) {
                            ForEach(0..<headers.count, id: \.self) { colIndex in
                                let cellText = colIndex < row.count ? row[colIndex] : ""
                                Text(LocalizedStringKey(cellText))
                                    .font(.system(size: 12.5))
                                    .foregroundStyle(Color(white: 0.85))
                                    .multilineTextAlignment(textAlignment(for: colIndex))
                                    .frame(minWidth: 100, alignment: alignment(for: colIndex))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)

                                if colIndex < headers.count - 1 {
                                    Divider()
                                        .background(Color(white: 0.12))
                                }
                            }
                        }
                        .background(rowIndex.isMultiple(of: 2)
                                    ? Color(red: 0.04, green: 0.04, blue: 0.06)
                                    : Color(red: 0.06, green: 0.06, blue: 0.08))

                        if rowIndex < rows.count - 1 {
                            Divider()
                                .background(Color(white: 0.10))
                        }
                    }
                }
                .textSelection(.enabled)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(white: 0.18), lineWidth: 1)
        }
    }

    private func copyTable() {
        var markdown = "| " + headers.joined(separator: " | ") + " |\n"
        markdown += "| " + alignments.map { align in
            switch align {
            case .left: return ":---"
            case .center: return ":---:"
            case .right: return "---:"
            }
        }.joined(separator: " | ") + " |\n"

        for row in rows {
            markdown += "| " + row.joined(separator: " | ") + " |\n"
        }

        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(markdown, forType: .string)

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
