import Foundation

public struct MermaidSanitizer: Sendable {
    public init() {}

    public static func sanitize(_ diagram: String) -> String {
        let lines = diagram.components(separatedBy: "\n")

        let isXYChart = lines.contains { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            return trimmed.hasPrefix("xychart")
        }

        if isXYChart {
            return sanitizeXYChart(lines: lines)
        } else {
            return sanitizeFlowchart(lines: lines)
        }
    }

    private static func sanitizeXYChart(lines: [String]) -> String {
        var processedLines: [String] = []
        var barCounts: [Int] = []
        var maxBarValue: Double = 0

        // Pass 1: Extract bar values and counts
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("bar [") || trimmed.hasPrefix("line [") {
                if let openBracket = trimmed.firstIndex(of: "["),
                   let closeBracket = trimmed.lastIndex(of: "]") {
                    let numbersStr = trimmed[trimmed.index(after: openBracket)..<closeBracket]
                    let numbers = numbersStr.components(separatedBy: ",").compactMap {
                        Double($0.trimmingCharacters(in: .whitespaces))
                    }
                    if !numbers.isEmpty {
                        barCounts.append(numbers.count)
                        if let maxVal = numbers.max() {
                            maxBarValue = max(maxBarValue, maxVal)
                        }
                    }
                }
            }
        }

        let requiredCategoriesCount = barCounts.first ?? 0
        let yMax = maxBarValue <= 0 ? 100 : (maxBarValue <= 10 ? 10 : (maxBarValue <= 50 ? 50 : (maxBarValue <= 100 ? 100 : ceil(maxBarValue * 1.15))))

        // Pass 2: Clean and fix lines
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // 1. Remove unsupported 'note' lines
            if trimmed.hasPrefix("note ") || trimmed.hasPrefix("note at") {
                continue
            }

            // 2. Fix y-axis: ensure range like '0 --> 100' exists
            if trimmed.hasPrefix("y-axis") {
                if !trimmed.contains("-->") {
                    let fixed = trimmed + " 0 --> \(Int(yMax))"
                    processedLines.append(fixed)
                    continue
                }
            }

            // 3. Fix x-axis: ensure bracketed categories exist
            if trimmed.hasPrefix("x-axis") {
                if !trimmed.contains("[") && requiredCategoriesCount > 0 {
                    let categories = defaultCategories(count: requiredCategoriesCount, context: trimmed)
                    let formattedCats = "[" + categories.map { "\"\($0)\"" }.joined(separator: ", ") + "]"
                    let fixed = "    x-axis \(formattedCats)"
                    processedLines.append(fixed)
                    continue
                }
            }

            processedLines.append(line)
        }

        return processedLines.joined(separator: "\n")
    }

    private static func defaultCategories(count: Int, context: String) -> [String] {
        let lower = context.lowercased()
        if count == 5 {
            if lower.contains("low") || lower.contains("high") {
                return ["Very Low", "Low", "Moderate", "High", "Very High"]
            }
            return ["Seg 1", "Seg 2", "Seg 3", "Seg 4", "Seg 5"]
        } else if count == 3 {
            return ["Low", "Medium", "High"]
        } else if count == 4 {
            return ["Q1", "Q2", "Q3", "Q4"]
        } else {
            return (1...count).map { "Group \($0)" }
        }
    }

    private static func sanitizeFlowchart(lines: [String]) -> String {
        var processedLines: [String] = []

        for line in lines {
            var modified = line
            // Quote node labels with parentheses e.g. A[Title (Info)] -> A["Title (Info)"]
            let regex = try? NSRegularExpression(pattern: #"(\b\w+)\[([^"\]]*\([^\)]*\)[^"\]]*)\]"#)
            if let matches = regex?.matches(in: modified, range: NSRange(location: 0, length: (modified as NSString).length)).reversed() {
                for match in matches {
                    let fullRange = match.range
                    let nodeID = (modified as NSString).substring(with: match.range(at: 1))
                    let label = (modified as NSString).substring(with: match.range(at: 2))
                    let replacement = "\(nodeID)[\"\(label)\"]"
                    if let r = Range(fullRange, in: modified) {
                        modified.replaceSubrange(r, with: replacement)
                    }
                }
            }
            processedLines.append(modified)
        }

        return processedLines.joined(separator: "\n")
    }
}
