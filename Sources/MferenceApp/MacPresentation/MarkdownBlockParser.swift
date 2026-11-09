import Foundation

public enum MarkdownBlock: Equatable, Sendable {
    case prose(String)
    case code(language: String, code: String)
    case table(headers: [String], rows: [[String]], alignments: [TableAlignment])
    case mathBlock(String)
    case mermaid(String)
    case htmlPreview(String)

    public enum TableAlignment: Equatable, Sendable {
        case left
        case center
        case right
    }
}

public struct MarkdownBlockParser: Sendable {
    public init() {}

    public func parse(_ markdown: String) -> [MarkdownBlock] {
        guard !markdown.isEmpty else { return [] }

        var blocks: [MarkdownBlock] = []
        let lines = markdown.components(separatedBy: "\n")
        var currentProseLines: [String] = []

        func flushProse() {
            let joined = currentProseLines.joined(separator: "\n").trimmingCharacters(in: .newlines)
            if !joined.isEmpty {
                blocks.append(.prose(joined))
            }
            currentProseLines.removeAll()
        }

        var i = 0
        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // 1. Check for Code Fence ``` or ~~~
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                let fenceMarker = String(trimmed.prefix(3))
                let rawLang = trimmed.dropFirst(3).trimmingCharacters(in: .whitespaces)
                let lang = rawLang.components(separatedBy: .whitespaces).first?.components(separatedBy: ":").first?.lowercased() ?? ""
                i += 1
                var codeLines: [String] = []
                while i < lines.count {
                    let cLine = lines[i]
                    let cTrimmed = cLine.trimmingCharacters(in: .whitespaces)
                    if cTrimmed.hasPrefix(fenceMarker) {
                        break
                    }
                    codeLines.append(cLine)
                    i += 1
                }

                flushProse()
                let codeContent = codeLines.joined(separator: "\n")

                if lang == "mermaid" {
                    blocks.append(.mermaid(codeContent))
                } else if lang == "html" || lang == "svg" {
                    blocks.append(.htmlPreview(codeContent))
                } else {
                    blocks.append(.code(language: lang, code: codeContent))
                }
                if i < lines.count {
                    i += 1
                }
                continue
            }

            // 2. Check for Display Math $$ ... $$
            if trimmed.hasPrefix("$$") {
                if trimmed.hasSuffix("$$") && trimmed.count > 4 {
                    // Single-line $$ formula $$
                    let formula = String(trimmed.dropFirst(2).dropLast(2)).trimmingCharacters(in: .whitespaces)
                    flushProse()
                    blocks.append(.mathBlock(formula))
                    i += 1
                    continue
                } else {
                    // Multi-line $$ ... $$
                    i += 1
                    var mathLines: [String] = []
                    while i < lines.count {
                        let mLine = lines[i]
                        let mTrimmed = mLine.trimmingCharacters(in: .whitespaces)
                        if mTrimmed.hasSuffix("$$") {
                            let content = String(mTrimmed.dropLast(2)).trimmingCharacters(in: .whitespaces)
                            if !content.isEmpty {
                                mathLines.append(content)
                            }
                            break
                        }
                        mathLines.append(mLine)
                        i += 1
                    }
                    flushProse()
                    let formula = mathLines.joined(separator: "\n")
                    blocks.append(.mathBlock(formula))
                    if i < lines.count {
                        i += 1
                    }
                    continue
                }
            }

            // 3. Check for Markdown Table: starts with '|' and followed by a delimiter line '|---|'
            if trimmed.hasPrefix("|") && trimmed.hasSuffix("|") && i + 1 < lines.count {
                let nextTrimmed = lines[i + 1].trimmingCharacters(in: .whitespaces)
                if isTableDelimiterLine(nextTrimmed) {
                    let headerLine = trimmed
                    let delimiterLine = nextTrimmed
                    let headers = parseTableRow(headerLine)
                    let alignments = parseTableAlignments(delimiterLine, count: headers.count)

                    i += 2
                    var rows: [[String]] = []
                    while i < lines.count {
                        let rowTrimmed = lines[i].trimmingCharacters(in: .whitespaces)
                        guard rowTrimmed.hasPrefix("|") && rowTrimmed.hasSuffix("|") else {
                            break
                        }
                        rows.append(parseTableRow(rowTrimmed))
                        i += 1
                    }

                    flushProse()
                    blocks.append(.table(headers: headers, rows: rows, alignments: alignments))
                    continue
                }
            }

            // Otherwise, regular line in prose
            currentProseLines.append(line)
            i += 1
        }

        flushProse()
        return blocks
    }

    private func isTableDelimiterLine(_ line: String) -> Bool {
        guard line.hasPrefix("|") && line.hasSuffix("|") else { return false }
        let stripped = line.replacingOccurrences(of: "|", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: ":", with: "")
            .replacingOccurrences(of: " ", with: "")
        return stripped.isEmpty && line.contains("-")
    }

    private func parseTableRow(_ line: String) -> [String] {
        var trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("|") { trimmed.removeFirst() }
        if trimmed.hasSuffix("|") { trimmed.removeLast() }
        return trimmed.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    private func parseTableAlignments(_ line: String, count: Int) -> [MarkdownBlock.TableAlignment] {
        let cols = parseTableRow(line)
        var result: [MarkdownBlock.TableAlignment] = []
        for col in cols {
            let leftColon = col.hasPrefix(":")
            let rightColon = col.hasSuffix(":")
            if leftColon && rightColon {
                result.append(.center)
            } else if rightColon {
                result.append(.right)
            } else {
                result.append(.left)
            }
        }
        while result.count < count {
            result.append(.left)
        }
        return result
    }
}
