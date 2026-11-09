import AppKit
import Foundation

public struct CodeSyntaxHighlighter: Sendable {
    public init() {}

    private static let keywordColor = NSColor(srgbRed: 0.38, green: 0.65, blue: 0.98, alpha: 1.0) // #60A5FA
    private static let stringColor = NSColor(srgbRed: 0.20, green: 0.83, blue: 0.60, alpha: 1.0)  // #34D399
    private static let numberColor = NSColor(srgbRed: 0.98, green: 0.57, blue: 0.24, alpha: 1.0)  // #FB923C
    private static let typeColor = NSColor(srgbRed: 0.22, green: 0.74, blue: 0.97, alpha: 1.0)    // #38BDF8
    private static let commentColor = NSColor(srgbRed: 0.45, green: 0.48, blue: 0.55, alpha: 1.0) // #737B8C
    private static let defaultTextColor = NSColor(srgbRed: 0.90, green: 0.92, blue: 0.94, alpha: 1.0)

    public func highlight(code: String, language: String) -> AttributedString {
        let font = NSFont.monospacedSystemFont(ofSize: 12.5, weight: .regular)
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineSpacing = 3.5

        let baseAttributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: Self.defaultTextColor,
            .paragraphStyle: paragraphStyle,
        ]

        let attributed = NSMutableAttributedString(string: code, attributes: baseAttributes)
        guard !code.isEmpty else { return AttributedString(attributed) }

        let nsString = code as NSString
        let fullRange = NSRange(location: 0, length: nsString.length)

        let lang = language.lowercased()

        // 1. Comments
        let commentRegexes: [String]
        switch lang {
        case "python", "py", "bash", "sh", "zsh", "shell", "yaml", "yml", "ruby", "rb":
            commentRegexes = [#"#.*$"#]
        case "sql":
            commentRegexes = [#"--.*$"#, #"/\*[\s\S]*?\*/"#]
        case "html", "xml", "svg":
            commentRegexes = [#"<!--[\s\S]*?-->"#]
        default:
            commentRegexes = [#"//.*$"#, #"/\*[\s\S]*?\*/"#]
        }

        for pattern in commentRegexes {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .anchorsMatchLines) {
                let matches = regex.matches(in: code, options: [], range: fullRange)
                for match in matches {
                    attributed.addAttribute(.foregroundColor, value: Self.commentColor, range: match.range)
                }
            }
        }

        // 2. Strings
        let stringPatterns = [
            #""(?:\\.|[^"\\])*""#,
            #"'(?:\\.|[^'\\])*'"#,
            #"`(?:\\.|[^`\\])*`"#,
        ]
        for pattern in stringPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
                let matches = regex.matches(in: code, options: [], range: fullRange)
                for match in matches {
                    // Do not override comments
                    let existingColor = attributed.attribute(.foregroundColor, at: match.range.location, effectiveRange: nil) as? NSColor
                    if existingColor != Self.commentColor {
                        attributed.addAttribute(.foregroundColor, value: Self.stringColor, range: match.range)
                    }
                }
            }
        }

        // 3. Numbers
        if let numRegex = try? NSRegularExpression(pattern: #"\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)\b"#, options: []) {
            let matches = numRegex.matches(in: code, options: [], range: fullRange)
            for match in matches {
                let existingColor = attributed.attribute(.foregroundColor, at: match.range.location, effectiveRange: nil) as? NSColor
                if existingColor == nil || existingColor == Self.defaultTextColor {
                    attributed.addAttribute(.foregroundColor, value: Self.numberColor, range: match.range)
                }
            }
        }

        // 4. Keywords
        let keywords = keywordsForLanguage(lang)
        if !keywords.isEmpty {
            let pattern = #"\b("# + keywords.joined(separator: "|") + #")\b"#
            if let kwRegex = try? NSRegularExpression(pattern: pattern, options: []) {
                let matches = kwRegex.matches(in: code, options: [], range: fullRange)
                for match in matches {
                    let existingColor = attributed.attribute(.foregroundColor, at: match.range.location, effectiveRange: nil) as? NSColor
                    if existingColor == nil || existingColor == Self.defaultTextColor {
                        attributed.addAttribute(.foregroundColor, value: Self.keywordColor, range: match.range)
                    }
                }
            }
        }

        // 5. Types & Builtins
        let types = typesForLanguage(lang)
        if !types.isEmpty {
            let pattern = #"\b("# + types.joined(separator: "|") + #")\b"#
            if let typeRegex = try? NSRegularExpression(pattern: pattern, options: []) {
                let matches = typeRegex.matches(in: code, options: [], range: fullRange)
                for match in matches {
                    let existingColor = attributed.attribute(.foregroundColor, at: match.range.location, effectiveRange: nil) as? NSColor
                    if existingColor == nil || existingColor == Self.defaultTextColor {
                        attributed.addAttribute(.foregroundColor, value: Self.typeColor, range: match.range)
                    }
                }
            }
        }

        return AttributedString(attributed)
    }

    private func keywordsForLanguage(_ lang: String) -> [String] {
        switch lang {
        case "swift":
            return [
                "let", "var", "func", "class", "struct", "enum", "protocol", "extension",
                "import", "return", "if", "else", "guard", "switch", "case", "default",
                "for", "while", "repeat", "break", "continue", "fallthrough", "try", "throw",
                "throws", "catch", "async", "await", "public", "private", "fileprivate",
                "internal", "open", "static", "final", "override", "mutating", "nonmutating",
                "init", "deinit", "self", "Self", "super", "where", "in", "as", "is",
                "actor", "macro", "nonisolated", "sendable"
            ]
        case "python", "py":
            return [
                "def", "class", "import", "from", "as", "return", "if", "elif", "else",
                "for", "while", "break", "continue", "pass", "try", "except", "finally",
                "raise", "with", "yield", "lambda", "global", "nonlocal", "assert",
                "async", "await", "match", "case"
            ]
        case "rust", "rs":
            return [
                "fn", "let", "mut", "const", "static", "struct", "enum", "trait", "impl",
                "type", "mod", "use", "pub", "return", "if", "else", "match", "loop",
                "while", "for", "in", "break", "continue", "move", "ref", "where",
                "as", "async", "await", "unsafe", "dyn"
            ]
        case "javascript", "js", "typescript", "ts", "jsx", "tsx":
            return [
                "const", "let", "var", "function", "class", "import", "export", "from",
                "return", "if", "else", "switch", "case", "default", "for", "while",
                "do", "break", "continue", "try", "catch", "finally", "throw", "new",
                "this", "super", "extends", "async", "await", "yield", "typeof", "instanceof",
                "interface", "type", "enum", "implements", "declare", "namespace"
            ]
        case "cpp", "c", "h", "hpp":
            return [
                "int", "float", "double", "char", "void", "bool", "auto", "const", "static",
                "class", "struct", "union", "enum", "namespace", "using", "return", "if",
                "else", "switch", "case", "default", "for", "while", "do", "break",
                "continue", "try", "catch", "throw", "new", "delete", "public", "private",
                "protected", "virtual", "override", "template", "typename", "constexpr",
                "include", "define"
            ]
        case "go", "golang":
            return [
                "func", "package", "import", "return", "var", "const", "type", "struct",
                "interface", "if", "else", "switch", "case", "default", "for", "range",
                "break", "continue", "go", "select", "defer", "chan", "map"
            ]
        case "bash", "sh", "zsh", "shell":
            return [
                "if", "then", "else", "elif", "fi", "case", "esac", "for", "while", "until",
                "do", "done", "in", "function", "select", "return", "exit", "local", "export"
            ]
        case "sql":
            return [
                "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "ON",
                "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "INSERT", "INTO", "VALUES",
                "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "DROP", "ALTER", "ADD",
                "PRIMARY", "KEY", "FOREIGN", "NOT", "NULL", "AND", "OR", "IN", "AS",
                "select", "from", "where", "join", "inner", "left", "right", "full", "on",
                "group", "by", "order", "having", "limit", "insert", "into", "values",
                "update", "set", "delete", "create", "table", "drop", "alter", "and", "or"
            ]
        default:
            return [
                "function", "def", "func", "fn", "var", "let", "const", "class", "return",
                "if", "else", "for", "while", "import", "export", "true", "false", "null", "nil"
            ]
        }
    }

    private func typesForLanguage(_ lang: String) -> [String] {
        switch lang {
        case "swift":
            return [
                "Int", "Int64", "Int32", "UInt", "UInt64", "Float", "Double", "Bool",
                "String", "Character", "Array", "Dictionary", "Set", "Optional", "Result",
                "Any", "AnyObject", "Void", "true", "false", "nil"
            ]
        case "python", "py":
            return [
                "int", "float", "str", "bool", "list", "dict", "set", "tuple", "bytes",
                "object", "None", "True", "False"
            ]
        case "rust", "rs":
            return [
                "i8", "i16", "i32", "i64", "i128", "isize",
                "u8", "u16", "u32", "u64", "u128", "usize",
                "f32", "f64", "bool", "char", "str", "String", "Vec", "Option", "Result",
                "Some", "None", "Ok", "Err", "true", "false"
            ]
        case "javascript", "js", "typescript", "ts":
            return [
                "string", "number", "boolean", "symbol", "bigint", "undefined", "null",
                "true", "false", "NaN", "Infinity", "Array", "Object", "Promise", "Map", "Set"
            ]
        default:
            return ["true", "false", "null", "nil", "None", "True", "False"]
        }
    }
}
