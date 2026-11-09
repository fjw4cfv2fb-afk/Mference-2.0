import Foundation

public struct LaTeXFormatter: Sendable {
    public init() {}

    private static let superscripts: [Character: Character] = [
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
        "n": "ⁿ", "i": "ⁱ", "k": "ᵏ", "x": "ˣ", "y": "ʸ"
    ]

    private static let subscripts: [Character: Character] = [
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
        "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
        "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
        "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
        "v": "ᵥ", "x": "ₓ"
    ]

    private static let symbolReplacements: [(String, String)] = [
        // Logic & Sets
        ("\\forall", "∀"),
        ("\\exists", "∃"),
        ("\\nexists", "∄"),
        ("\\land", "∧"),
        ("\\lor", "∨"),
        ("\\neg", "¬"),
        ("\\sim", "∼"),
        ("\\implies", "⇒"),
        ("\\iff", "⇔"),
        ("\\to", "→"),
        ("\\leftarrow", "←"),
        ("\\leftrightarrow", "↔"),
        ("\\in", "∈"),
        ("\\notin", "∉"),
        ("\\subset", "⊂"),
        ("\\subseteq", "⊆"),
        ("\\supset", "⊃"),
        ("\\supseteq", "⊇"),
        ("\\emptyset", "∅"),
        ("\\cup", "∪"),
        ("\\cap", "∩"),
        ("\\setminus", "∖"),

        // Big operators & symbols
        ("\\sum", "∑"),
        ("\\prod", "∏"),
        ("\\int", "∫"),
        ("\\infty", "∞"),
        ("\\partial", "∂"),
        ("\\nabla", "∇"),
        ("\\pm", "±"),
        ("\\mp", "∓"),
        ("\\times", "×"),
        ("\\cdot", "·"),
        ("\\div", "÷"),
        ("\\neq", "≠"),
        ("\\le", "≤"),
        ("\\leq", "≤"),
        ("\\ge", "≥"),
        ("\\geq", "≥"),
        ("\\approx", "≈"),
        ("\\equiv", "≡"),
        ("\\circ", "∘"),

        // Greek Lowercase
        ("\\alpha", "α"),
        ("\\beta", "β"),
        ("\\gamma", "γ"),
        ("\\delta", "δ"),
        ("\\epsilon", "ε"),
        ("\\zeta", "ζ"),
        ("\\eta", "η"),
        ("\\theta", "θ"),
        ("\\iota", "ι"),
        ("\\kappa", "κ"),
        ("\\lambda", "λ"),
        ("\\mu", "μ"),
        ("\\nu", "ν"),
        ("\\xi", "ξ"),
        ("\\pi", "π"),
        ("\\rho", "ρ"),
        ("\\sigma", "σ"),
        ("\\tau", "τ"),
        ("\\upsilon", "υ"),
        ("\\phi", "φ"),
        ("\\chi", "χ"),
        ("\\psi", "ψ"),
        ("\\omega", "ω"),

        // Greek Uppercase
        ("\\Gamma", "Γ"),
        ("\\Delta", "Δ"),
        ("\\Theta", "Θ"),
        ("\\Lambda", "Λ"),
        ("\\Xi", "Ξ"),
        ("\\Pi", "Π"),
        ("\\Sigma", "Σ"),
        ("\\Upsilon", "Υ"),
        ("\\Phi", "Φ"),
        ("\\Psi", "Ψ"),
        ("\\Omega", "Ω"),
    ]

    public func format(latex: String) -> String {
        var result = latex

        // Replace fractions: \frac{a}{b} -> (a)/(b)
        while let match = result.range(of: #"\\frac\{([^{}]+)\}\{([^{}]+)\}"#, options: .regularExpression) {
            let matchedStr = String(result[match])
            if let regex = try? NSRegularExpression(pattern: #"\\frac\{([^{}]+)\}\{([^{}]+)\}"#),
               let firstMatch = regex.firstMatch(in: matchedStr, range: NSRange(location: 0, length: (matchedStr as NSString).length)),
               firstMatch.numberOfRanges == 3 {
                let num = (matchedStr as NSString).substring(with: firstMatch.range(at: 1))
                let den = (matchedStr as NSString).substring(with: firstMatch.range(at: 2))
                result.replaceSubrange(match, with: "(\(num))/(\(den))")
            } else {
                break
            }
        }

        // Replace square roots: \sqrt{x} -> √(x)
        while let match = result.range(of: #"\\sqrt\{([^{}]+)\}"#, options: .regularExpression) {
            let matchedStr = String(result[match])
            if let regex = try? NSRegularExpression(pattern: #"\\sqrt\{([^{}]+)\}"#),
               let firstMatch = regex.firstMatch(in: matchedStr, range: NSRange(location: 0, length: (matchedStr as NSString).length)),
               firstMatch.numberOfRanges == 2 {
                let content = (matchedStr as NSString).substring(with: firstMatch.range(at: 1))
                result.replaceSubrange(match, with: "√(\(content))")
            } else {
                break
            }
        }

        // Replace known LaTeX symbols
        for (command, symbol) in Self.symbolReplacements {
            result = result.replacingOccurrences(of: command, with: symbol)
        }

        // Replace superscripts ^{...} or ^x
        result = replaceSuperscripts(in: result)

        // Replace subscripts _{...} or _x
        result = replaceSubscripts(in: result)

        // Clean up remaining latex formatting markers like \text{...}, \mathbf{...}, \mathit{...}
        result = result.replacingOccurrences(of: #"\\(?:text|mathbf|mathit|mathrm)\{([^{}]+)\}"#,
                                            with: "$1",
                                            options: .regularExpression)

        // Clean up leftover braces { }
        result = result.replacingOccurrences(of: "{", with: "")
                       .replacingOccurrences(of: "}", with: "")
                       .replacingOccurrences(of: "\\,", with: " ")
                       .replacingOccurrences(of: "\\;", with: " ")
                       .replacingOccurrences(of: "\\quad", with: "   ")

        return result.trimmingCharacters(in: .whitespaces)
    }

    private func replaceSuperscripts(in str: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: #"\^\{([^{}]+)\}|\^([0-9a-zA-Z\+\-\=])"#) else {
            return str
        }
        let nsString = str as NSString
        let matches = regex.matches(in: str, range: NSRange(location: 0, length: nsString.length))
        guard !matches.isEmpty else { return str }

        var result = ""
        var lastLocation = 0
        for match in matches {
            let prefixRange = NSRange(location: lastLocation, length: match.range.location - lastLocation)
            result += nsString.substring(with: prefixRange)

            if match.range(at: 1).location != NSNotFound {
                let inner = nsString.substring(with: match.range(at: 1))
                result += String(inner.map { Self.superscripts[$0] ?? $0 })
            } else if match.range(at: 2).location != NSNotFound {
                let inner = nsString.substring(with: match.range(at: 2))
                if let char = inner.first, let sup = Self.superscripts[char] {
                    result += String(sup)
                } else {
                    result += "^" + inner
                }
            }
            lastLocation = match.range.location + match.range.length
        }
        if lastLocation < nsString.length {
            result += nsString.substring(from: lastLocation)
        }
        return result
    }

    private func replaceSubscripts(in str: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: #"_\{([^{}]+)\}|_([0-9a-zA-Z\+\-\=])"#) else {
            return str
        }
        let nsString = str as NSString
        let matches = regex.matches(in: str, range: NSRange(location: 0, length: nsString.length))
        guard !matches.isEmpty else { return str }

        var result = ""
        var lastLocation = 0
        for match in matches {
            let prefixRange = NSRange(location: lastLocation, length: match.range.location - lastLocation)
            result += nsString.substring(with: prefixRange)

            if match.range(at: 1).location != NSNotFound {
                let inner = nsString.substring(with: match.range(at: 1))
                result += String(inner.map { Self.subscripts[$0] ?? $0 })
            } else if match.range(at: 2).location != NSNotFound {
                let inner = nsString.substring(with: match.range(at: 2))
                if let char = inner.first, let sub = Self.subscripts[char] {
                    result += String(sub)
                } else {
                    result += "_" + inner
                }
            }
            lastLocation = match.range.location + match.range.length
        }
        if lastLocation < nsString.length {
            result += nsString.substring(from: lastLocation)
        }
        return result
    }
}
