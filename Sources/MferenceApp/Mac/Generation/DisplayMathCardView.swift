import AppKit
import MferenceMacPresentation
import SwiftUI

struct DisplayMathCardView: View {
    let formula: String

    @State private var isCopied = false
    @State private var copyTask: Task<Void, Never>?

    private var formattedMath: String {
        LaTeXFormatter().format(latex: formula)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header bar
            HStack {
                HStack(spacing: 5) {
                    Image(systemName: "function")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(MferenceMacTheme.accentColor)

                    Text("EQUATION")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(Color(white: 0.85))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color(white: 0.12), in: Capsule())

                Spacer()

                Button {
                    copyFormula()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 11, weight: .medium))
                            .contentTransition(.symbolEffect(.replace))
                        Text(isCopied ? "Copied!" : "Copy LaTeX")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(isCopied ? MferenceMacTheme.accentColor : Color(white: 0.75))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(white: 0.14), in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .help("Copy raw LaTeX formula")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(red: 0.08, green: 0.08, blue: 0.10))

            Divider()
                .background(Color(white: 0.15))

            // Math Typography Body
            ScrollView(.horizontal, showsIndicators: false) {
                Text(formattedMath)
                    .font(.system(size: 16, weight: .medium, design: .serif))
                    .foregroundStyle(Color(white: 0.95))
                    .multilineTextAlignment(.center)
                    .textSelection(.enabled)
                    .padding(.vertical, 16)
                    .padding(.horizontal, 20)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            .background(Color(red: 0.04, green: 0.04, blue: 0.06))
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(white: 0.18), lineWidth: 1)
        }
    }

    private func copyFormula() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(formula, forType: .string)

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
