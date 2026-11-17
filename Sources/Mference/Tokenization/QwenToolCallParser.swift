import Foundation

/// Parses the Qwen ChatML tool-call body — the text BETWEEN the `<tool_call>`
/// and `</tool_call>` special tokens:
///
///     \n<function=NAME>\n
///     <parameter=KEY>\nVALUE\n</parameter>\n
///     ...
///     </function>\n
///
/// Each VALUE is everything between the newline following the parameter open
/// tag and the newline before `</parameter>`; multi-line values are allowed.
/// Values that parse as structural JSON (object / array / number / bool /
/// null) become typed `JSONValue`s; everything else is kept as a raw string,
/// unless the tool schema explicitly declares a string. Qwen serializes
/// strings unquoted, so `123`, `true`, and JSON text can all be string payloads.
/// Without an unambiguous string declaration, retain the legacy JSON inference;
/// a declared boolean also accepts case-insensitive true/false spellings seen
/// in Qwen XML output. No arbitrary value is coerced to false. This parser is
/// not a general JSON Schema validator.
public struct QwenToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        try parse(text, allowedTools: allowedTools, id: id, toolSchemas: [:])
    }

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String,
                      toolSchemas: [String: JSONValue]) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        var body = Substring(text)
        trimOuterWhitespace(&body)

        if body.hasPrefix("{") {
            return try parseJSONToolCallFallback(String(body),
                                                 allowedTools: allowedTools,
                                                 id: id,
                                                 toolSchemas: toolSchemas)
        }

        let name = try functionName(&body)
        guard isValidFunctionName(name) else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }

        var arguments: [String: JSONValue] = [:]
        let properties = toolSchemas[name]?.objectValue?["properties"]?.objectValue ?? [:]
        while !body.hasPrefix("</function>") {
            let (key, raw) = try parameter(&body)
            let declaredType = properties[key]?.objectValue?["type"]
            if declaredType == .string("string") {
                arguments[key] = .string(raw)
            } else if declaredType == .string("boolean"),
                      let value = booleanLiteral(raw) {
                arguments[key] = .bool(value)
            } else {
                arguments[key] = try parsedValue(raw)
            }
        }
        body.removeFirst("</function>".count)
        trimOuterWhitespace(&body)
        guard body.isEmpty else { throw ToolCallParserError.malformed }

        let argumentsValue = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: argumentsValue,
                              argumentsJSON: try argumentsValue.encoded())
    }

    private func trimOuterWhitespace(_ body: inout Substring) {
        while let first = body.first, first.isWhitespace { body.removeFirst() }
        while let last = body.last, last.isWhitespace { body.removeLast() }
    }

    private func functionName(_ body: inout Substring) throws -> String {
        guard body.hasPrefix("<function=") else {
            throw ToolCallParserError.malformed
        }
        body.removeFirst("<function=".count)
        guard let close = body.firstIndex(of: ">") else {
            throw ToolCallParserError.malformed
        }
        let name = String(body[..<close])
        body = body[body.index(after: close)...]
        if body.first == "\n" {
            body.removeFirst()
        } else if !body.hasPrefix("<parameter=") {
            throw ToolCallParserError.malformed
        }
        return name
    }

    private func parameter(_ body: inout Substring) throws -> (String, String) {
        guard body.hasPrefix("<parameter=") else {
            throw ToolCallParserError.malformed
        }
        body.removeFirst("<parameter=".count)
        guard let close = body.firstIndex(of: ">") else {
            throw ToolCallParserError.malformed
        }
        let key = String(body[..<close])
        guard !key.isEmpty, !key.contains("\n"), !key.contains("<") else {
            throw ToolCallParserError.malformed
        }
        body = body[body.index(after: close)...]
        let hadLeadingNewline: Bool
        if body.first == "\n" {
            body.removeFirst()
            hadLeadingNewline = true
        } else {
            hadLeadingNewline = false
        }
        // VALUE runs to `\n</parameter>\n`, or for empty values `</parameter>\n`,
        // or under quantized formatting slips inline `</parameter>` followed by
        // optional newline before the next `<parameter=` or `</function>`.
        if hadLeadingNewline,
           let closeRange = body.range(of: "\n</parameter>\n") ?? emptyValueCloseRange(body) {
            let value = String(body[..<closeRange.lowerBound])
            body = body[closeRange.upperBound...]
            return (key, value)
        }
        guard let inlineClose = body.range(of: "</parameter>") else {
            throw ToolCallParserError.malformed
        }
        var rawValue = Substring(body[..<inlineClose.lowerBound])
        if hadLeadingNewline, rawValue.last == "\n" {
            rawValue.removeLast()
        }
        var afterClose = body[inlineClose.upperBound...]
        while afterClose.first == " " || afterClose.first == "\t" || afterClose.first == "\r" {
            afterClose.removeFirst()
        }
        if afterClose.first == "\n" {
            afterClose.removeFirst()
        }
        body = afterClose
        return (key, String(rawValue))
    }

    /// Handles `<parameter=k>\n</parameter>\n` where the single newline both
    /// opens the value and precedes the close tag (empty value, no blank line).
    private func emptyValueCloseRange(_ body: Substring) -> Range<Substring.Index>? {
        guard body.hasPrefix("</parameter>\n") else { return nil }
        return body.startIndex..<body.index(body.startIndex,
                                            offsetBy: "</parameter>\n".count)
    }

    private func isValidFunctionName(_ name: String) -> Bool {
        name.range(of: "^[A-Za-z0-9_-]{1,64}$", options: .regularExpression) != nil
    }

    private func booleanLiteral(_ raw: String) -> Bool? {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "true": return true
        case "false": return false
        default: return nil
        }
    }

    private func parsedValue(_ raw: String) throws -> JSONValue {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = trimmed.first,
              "{[-0123456789tfn".contains(first) else {
            return .string(raw)
        }
        // The decode below bounds depth via `JSONValue.init(from:)`, but a
        // failed decode falls back to `.string`, which would smuggle an
        // over-deep value through as raw text instead of rejecting the call
        // the way the Gemma parser does. Bound it explicitly first — but only
        // for values that really are structural JSON: unquoted string
        // arguments may legitimately start with a bracket and nest deeply
        // (code, minified JSON as string content), and those keep the
        // raw-string fallback. `JSONSerialization` settles which case this is
        // without hazard: its scanner caps nesting at 512, and it never
        // builds a `JSONValue` tree.
        if "{[".contains(first), nestsBeyondLimit(trimmed) {
            if (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8))) != nil {
                throw ToolCallParserError.malformed
            }
            return .string(raw)
        }
        if let value = try? JSONDecoder().decode(JSONValue.self,
                                                 from: Data(trimmed.utf8)) {
            // Quoted strings stay raw text: the template writes string arguments
            // unquoted, so literal quotes belong to the argument itself.
            if case .string = value { return .string(raw) }
            return value
        }
        if "{[".contains(first),
           let repaired = Self.repairMinorJSONSlip(trimmed),
           !nestsBeyondLimit(repaired),
           let value = try? JSONDecoder().decode(JSONValue.self,
                                                 from: Data(repaired.utf8)) {
            if case .string = value { return .string(raw) }
            return value
        }
        return .string(raw)
    }

    /// Repairs minor quantized JSON slips (trailing commas before `}`/`]` or a single
    /// missing closing delimiter at the end of a container).
    private static func repairMinorJSONSlip(_ text: String) -> String? {
        // 1. Remove trailing commas before `}` or `]`, e.g. `{"a": 1, }` -> `{"a": 1 }`
        var cleaned = text.replacingOccurrences(
            of: #",\s*([}\]])"#,
            with: "$1",
            options: .regularExpression)
        if cleaned != text {
            return cleaned
        }
        // 2. Balance a single unclosed trailing `{` or `[` if not inside a string literal.
        var stack: [Character] = []
        var inString = false
        var escaped = false
        for ch in cleaned {
            if inString {
                if escaped { escaped = false }
                else if ch == "\\" { escaped = true }
                else if ch == "\"" { inString = false }
                continue
            }
            switch ch {
            case "\"": inString = true
            case "{": stack.append("}")
            case "[": stack.append("]")
            case "}", "]":
                guard stack.last == ch else { return nil }
                stack.removeLast()
            default: break
            }
        }
        guard !inString, !stack.isEmpty, stack.count <= 2 else { return nil }
        cleaned = cleaned.replacingOccurrences(
            of: #",\s*$"#,
            with: "",
            options: .regularExpression)
        for closer in stack.reversed() {
            cleaned.append(closer)
        }
        return cleaned
    }

    private func parseJSONToolCallFallback(_ text: String,
                                           allowedTools: Set<String>,
                                           id: String,
                                           toolSchemas: [String: JSONValue]) throws -> ParsedToolCall {
        guard !nestsBeyondLimit(text) else { throw ToolCallParserError.malformed }
        let Candidate = (try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)))
            ?? Self.repairMinorJSONSlip(text).flatMap {
                try? JSONDecoder().decode(JSONValue.self, from: Data($0.utf8))
            }
        guard case .object(let root) = Candidate,
              case .string(let name) = root["name"],
              isValidFunctionName(name) else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }
        let rawArgs: [String: JSONValue]
        if let argsField = root["arguments"] ?? root["parameters"] {
            if case .object(let obj) = argsField {
                rawArgs = obj
            } else if case .string(let argStr) = argsField,
                      !nestsBeyondLimit(argStr),
                      case .object(let decodedObj) = try? JSONDecoder().decode(
                        JSONValue.self, from: Data(argStr.utf8)) {
                rawArgs = decodedObj
            } else {
                throw ToolCallParserError.malformed
            }
        } else {
            rawArgs = [:]
        }
        let properties = toolSchemas[name]?.objectValue?["properties"]?.objectValue ?? [:]
        var coerced: [String: JSONValue] = [:]
        for (k, v) in rawArgs {
            let declaredType = properties[k]?.objectValue?["type"]
            if declaredType == .string("boolean"),
               case .string(let s) = v,
               let b = booleanLiteral(s) {
                coerced[k] = .bool(b)
            } else {
                coerced[k] = v
            }
        }
        let argumentsValue = JSONValue.object(coerced)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: argumentsValue,
                              argumentsJSON: try argumentsValue.encoded())
    }

    /// Iteratively counts container nesting, ignoring brackets inside string
    /// literals, so no input can drive recursion — this scan is the depth
    /// guard for the recursive walks downstream of the parse (`JSONValue`
    /// itself, `jinjaSendableValue`), matching `JSONValue.maximumDepth` on
    /// the `Codable` boundary and the Gemma parser's descent guard.
    private func nestsBeyondLimit(_ text: String) -> Bool {
        var depth = 0
        var isInString = false
        var isEscaped = false
        for character in text {
            if isInString {
                if isEscaped {
                    isEscaped = false
                } else if character == "\\" {
                    isEscaped = true
                } else if character == "\"" {
                    isInString = false
                }
                continue
            }
            switch character {
            case "\"": isInString = true
            case "{", "[":
                depth += 1
                if depth > JSONValue.maximumDepth { return true }
            case "}", "]": depth -= 1
            default: break
            }
        }
        return false
    }
}
