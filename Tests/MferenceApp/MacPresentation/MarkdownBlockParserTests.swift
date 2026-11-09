import Foundation
import Testing
@testable import MferenceMacPresentation

@Suite struct MarkdownBlockParserTests {
    @Test func parsesCodeFences() {
        let markdown = """
        Here is some code:

        ```swift
        let x = 42
        print(x)
        ```

        And that is all.
        """
        let parser = MarkdownBlockParser()
        let blocks = parser.parse(markdown)

        #expect(blocks.count == 3)
        #expect(blocks[0] == .prose("Here is some code:"))
        #expect(blocks[1] == .code(language: "swift", code: "let x = 42\nprint(x)"))
        #expect(blocks[2] == .prose("And that is all."))
    }

    @Test func parsesMermaidAndHtml() {
        let markdown = """
        ```mermaid
        graph TD
            A --> B
        ```

        ```html
        <button>Click</button>
        ```
        """
        let blocks = MarkdownBlockParser().parse(markdown)

        #expect(blocks.count == 2)
        #expect(blocks[0] == .mermaid("graph TD\n    A --> B"))
        #expect(blocks[1] == .htmlPreview("<button>Click</button>"))
    }

    @Test func parsesMarkdownTables() {
        let markdown = """
        | Header 1 | Header 2 |
        | :--- | ---: |
        | Row 1 Col 1 | Row 1 Col 2 |
        | Row 2 Col 1 | Row 2 Col 2 |
        """
        let blocks = MarkdownBlockParser().parse(markdown)

        #expect(blocks.count == 1)
        if case .table(let headers, let rows, let alignments) = blocks[0] {
            #expect(headers == ["Header 1", "Header 2"])
            #expect(rows.count == 2)
            #expect(rows[0] == ["Row 1 Col 1", "Row 1 Col 2"])
            #expect(alignments == [.left, .right])
        } else {
            Issue.record("Expected table block")
        }
    }

    @Test func parsesDisplayMath() {
        let singleLine = "$$ x^2 + y^2 = z^2 $$"
        let blocks1 = MarkdownBlockParser().parse(singleLine)
        #expect(blocks1.count == 1)
        #expect(blocks1[0] == .mathBlock("x^2 + y^2 = z^2"))

        let multiLine = """
        $$
        \\sum_{i=1}^n i = \\frac{n(n+1)}{2}
        $$
        """
        let blocks2 = MarkdownBlockParser().parse(multiLine)
        #expect(blocks2.count == 1)
        #expect(blocks2[0] == .mathBlock("\\sum_{i=1}^n i = \\frac{n(n+1)}{2}"))
    }

    @Test func parsesCodeFencesWithAttributes() {
        let markdown = """
        ```python title="script.py"
        print("hello")
        ```
        """
        let blocks = MarkdownBlockParser().parse(markdown)
        #expect(blocks.count == 1)
        #expect(blocks[0] == .code(language: "python", code: "print(\"hello\")"))
    }

    @Test func handlesUnclosedCodeFencesWithoutCrashing() {
        let markdown = "```swift\nlet x = 10"
        let blocks = MarkdownBlockParser().parse(markdown)
        #expect(blocks.count == 1)
        #expect(blocks[0] == .code(language: "swift", code: "let x = 10"))
    }
}

@Suite struct LaTeXFormatterTests {
    @Test func formatsFractionsAndRoots() {
        let formatter = LaTeXFormatter()
        let result = formatter.format(latex: "\\frac{a}{b} + \\sqrt{x}")
        #expect(result.contains("(a)/(b)"))
        #expect(result.contains("√(x)"))
    }

    @Test func formatsLogicAndGreekSymbols() {
        let formatter = LaTeXFormatter()
        let result = formatter.format(latex: "\\forall x \\in S, P(x) \\land \\neg Q(x) \\implies \\alpha = \\pi")
        #expect(result.contains("∀"))
        #expect(result.contains("∈"))
        #expect(result.contains("∧"))
        #expect(result.contains("¬"))
        #expect(result.contains("⇒"))
        #expect(result.contains("α"))
        #expect(result.contains("π"))
    }

    @Test func formatsSuperscriptsAndSubscripts() {
        let formatter = LaTeXFormatter()
        let result = formatter.format(latex: "x^2 + a_1 = y^{n+1}")
        #expect(result.contains("x²"))
        #expect(result.contains("a₁"))
        #expect(result.contains("yⁿ⁺¹"))
    }

    @Test func formatsMultipleConsecutiveSubAndSuperScripts() {
        let formatter = LaTeXFormatter()
        let result = formatter.format(latex: "x^{n+1} + x_1 + x_2 + y^2")
        #expect(result.contains("xⁿ⁺¹"))
        #expect(result.contains("x₁"))
        #expect(result.contains("x₂"))
        #expect(result.contains("y²"))
    }
}

@Suite struct MermaidSanitizerTests {
    @Test func sanitizesMalformedXYChart() {
        let malformed = """
        xychart-beta
            title "Global Vulnerability to Misinformation vs. Critical Thinking Levels"
            x-axis "Global Population Segment (from Low to High Education/Critical Thinking)"
            y-axis "Susceptibility to Being Duped by Natural Misinformation (%)"
            bar [85, 65, 45, 25, 10]
            
            note at 0, 85 "Low Critical Thinking"
            note at 2, 45 "Moderate"
            note at 4, 10 "High Critical Thinking"
        """

        let cleaned = MermaidSanitizer.sanitize(malformed)

        #expect(!cleaned.contains("note at"))
        #expect(cleaned.contains("--> 100"))
        #expect(cleaned.contains("x-axis ["))
        #expect(cleaned.contains("bar [85, 65, 45, 25, 10]"))
    }

    @Test func sanitizesFlowchartParentheses() {
        let input = "flowchart TD\n    A[Step One (Important)] --> B"
        let cleaned = MermaidSanitizer.sanitize(input)
        #expect(cleaned.contains("A[\"Step One (Important)\"]"))
    }
}
