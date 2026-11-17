import Foundation
import Testing
@testable import Mference
@testable import MferenceServerCore

@Suite struct SwiftQwenServerTests {
    @Test(arguments: [CheckpointIdentity.swiftQwen38, CheckpointIdentity.baseQwen38])
    func effortAndHistoricalReasoningSurviveValidation(model: String) throws {
        for effort in ["xhigh", "medium", "low", "none"] {
            let data = Data("""
            {"model":"\(model)","reasoning_effort":"\(effort)","messages":[
              {"role":"user","content":"A"},
              {"role":"assistant","content":"B","reasoning_content":"Check A"},
              {"role":"user","content":"C"}]}
            """.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            let validated = try OpenAIRequestValidator.validate(request, modelID: request.model, dialect: .chatml)
            #expect(validated.reasoningEffort?.rawValue == effort)
            #expect(validated.messages[1].reasoningContent == "Check A")
        }
    }

    @Test func aliasCapabilitiesAndBaseExplicitRoleContract() throws {
        let data = Data(#"{"model":"alias","reasoning_effort":"low","messages":[{"role":"user","content":"Hi"}]}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(try OpenAIRequestValidator.validate(request, modelID: "alias", dialect: .chatml,
            swiftQwen: false, qwenReasoning: true).reasoningEffort == .low)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "alias", dialect: .chatml,
                swiftQwen: false, qwenReasoning: false)
        }
        for effort in ["", "\"reasoning_effort\":\"low\","] {
            let body = Data("""
            {"model":"qwen3.8-27b-4bit",\(effort)"messages":[{"role":"developer","content":"Guide"},{"role":"user","content":"Hi"}]}
            """.utf8)
            let base = try JSONDecoder().decode(OpenAIChatRequest.self, from: body)
            if effort.isEmpty {
                #expect(try OpenAIRequestValidator.validate(base, modelID: base.model, dialect: .chatml).reasoningEffort == nil)
            } else {
                #expect(throws: ServerRequestError.self) {
                    try OpenAIRequestValidator.validate(base, modelID: base.model, dialect: .chatml)
                }
            }
        }
    }

    @Test func unsupportedEffortAndDeveloperAreRejected() throws {
        for fragment in ["\"reasoning_effort\":\"high\",", ""] {
            let role = fragment.isEmpty ? "developer" : "system"
            let data = Data("""
            {"model":"swift-qwen3.8-27b-int4g64",\(fragment)"messages":[
              {"role":"\(role)","content":"Guide"},{"role":"user","content":"Hi"}]}
            """.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            #expect(throws: ServerRequestError.self) {
                try OpenAIRequestValidator.validate(request, modelID: request.model, dialect: .chatml)
            }
        }
    }

    @Test func checkpointIdentityIsIndependentOfDirectoryName() throws {
        let root = try ServerLibraryFixture.makeRoot("swift-qwen-identities")
        defer { try? FileManager.default.removeItem(at: root) }
        for (directory, id) in [("renamed", CheckpointIdentity.swiftQwen38), ("qwen38", "qwen3.8-27b-4bit")] {
            let folder = root.appendingPathComponent(directory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: ["modelID": id, "arch": ["family": "qwen38"]])
                .write(to: folder.appendingPathComponent("manifest.json"))
            #expect(try ManifestReader.peekModelID(directoryURL: folder) == id)
        }
        let index = ServerLibraryDiscovery.discover(roots: [root], probe: { directory in
            directory == root ? .notAModelDirectory : .complete(family: .qwen38)
        })
        #expect(Set(index.entries.map(\.modelID)) == [CheckpointIdentity.swiftQwen38, "qwen3.8-27b-4bit"])
        #expect(ServerFamilyModelID.modelID(for: .qwen38, checkpointID: CheckpointIdentity.swiftQwen38) == CheckpointIdentity.swiftQwen38)
    }

    @Test func intentAlignerPrependsActionDirectiveAndConfiguresLoopAndThinkingGuards() {
        let terseUser: [MFTokenizer.Message] = [.init(role: .user, content: "fix bug in sort.swift")]
        let aligned = ServerIntentAligner.alignMessages(terseUser, tools: [], dialect: .chatml, environment: [:])
        #expect(aligned.count == 2)
        #expect(aligned.first?.role == .system)

        // Existing system message is preserved without duplication.
        let withSystem: [MFTokenizer.Message] = [
            .init(role: .system, content: "Custom"),
            .init(role: .user, content: "fix bug")
        ]
        #expect(ServerIntentAligner.alignMessages(withSystem, tools: [], dialect: .chatml, environment: [:]) == withSystem)

        let tuned = ServerIntentAligner.tunedGenerationConfig(
            GenerationConfig(maxNewTokens: 512, temperature: 0, repetitionPenalty: 1.05),
            maxCompletionTokens: 512,
            remainingContext: 2048,
            startsInThinking: true,
            reasoningEffort: .low)
        #expect(tuned.loopGuardRepeats == 3)
        #expect(tuned.repetitionWindow == 128)
        #expect(tuned.maxThinkingTokens == 256)

        // Multi-part algorithmic prompts get a larger thinking budget ratio (4/5 of maxNewTokens).
        let grayPrompt: [MFTokenizer.Message] = [
            .init(role: .user, content: "Trace step-by-step every state transition of a 3-bit Gray code counter, then write a branchless C expression.")
        ]
        let profile = ServerIntentAligner.classifyPromptThinkingProfile(messages: grayPrompt)
        #expect(profile == .multiPartAlgorithmic)
        let tunedMultiPart = ServerIntentAligner.tunedGenerationConfig(
            GenerationConfig(maxNewTokens: 384, temperature: 0),
            maxCompletionTokens: 384,
            remainingContext: 2048,
            startsInThinking: true,
            reasoningEffort: .low,
            promptProfile: profile)
        #expect(tunedMultiPart.maxThinkingTokens == 307)

        // Simple single-step transform requests or reasoningEffort == .none classify as zeroThinkFastPath.
        let fastPrompt: [MFTokenizer.Message] = [.init(role: .user, content: "convert 42 to hex")]
        #expect(ServerIntentAligner.classifyPromptThinkingProfile(messages: fastPrompt) == .zeroThinkFastPath)
        #expect(ServerIntentAligner.classifyPromptThinkingProfile(messages: terseUser, reasoningEffort: .off) == .zeroThinkFastPath)

        // Memory-pressure guard caps maxSlotSupportedK under tight/moderate RAM or long context.
        #expect(ServerIntentAligner.recommendedMaxSlotSupportedK(
            availableMemoryBytes: 1_024 * 1024 * 1024, activeContextTokens: 512, configuredSlotCap: 4) == 1)
        #expect(ServerIntentAligner.recommendedMaxSlotSupportedK(
            availableMemoryBytes: 2_048 * 1024 * 1024, activeContextTokens: 512, configuredSlotCap: 4) == 2)
        #expect(ServerIntentAligner.recommendedMaxSlotSupportedK(
            availableMemoryBytes: 6_144 * 1024 * 1024, activeContextTokens: 512, configuredSlotCap: 3) == 3)

        // Setting MFERENCE_INTENT_ALIGN=0 disables prompt alignment, profiles, loop guard, and thinking cap.
        let disabledEnv = ["MFERENCE_INTENT_ALIGN": "0"]
        #expect(ServerIntentAligner.alignMessages(terseUser, tools: [], dialect: .chatml, environment: disabledEnv) == terseUser)
        #expect(ServerIntentAligner.classifyPromptThinkingProfile(
            messages: grayPrompt, environment: disabledEnv) == .standard)
        #expect(ServerIntentAligner.classifyPromptThinkingProfile(
            messages: fastPrompt, environment: disabledEnv) == .standard)
        let untuned = ServerIntentAligner.tunedGenerationConfig(
            GenerationConfig(maxNewTokens: 512, temperature: 0),
            maxCompletionTokens: 512,
            remainingContext: 2048,
            startsInThinking: true,
            reasoningEffort: .low,
            promptProfile: .standard,
            environment: disabledEnv)
        #expect(untuned.loopGuardRepeats == 0)
        #expect(untuned.maxThinkingTokens == 0)

        let defaultBudget = ServerIntentAligner.tunedGenerationConfig(
            GenerationConfig(maxNewTokens: 512, temperature: 0),
            maxCompletionTokens: 512,
            remainingContext: 2048,
            startsInThinking: true,
            reasoningEffort: nil,
            promptProfile: .standard,
            environment: [:])
        #expect(defaultBudget.maxThinkingTokens == 224)

        let mathBudget = ServerIntentAligner.tunedGenerationConfig(
            GenerationConfig(maxNewTokens: 512, temperature: 0),
            maxCompletionTokens: 512,
            remainingContext: 2048,
            startsInThinking: true,
            reasoningEffort: nil,
            promptProfile: .multiPartAlgorithmic,
            environment: [:])
        #expect(mathBudget.maxThinkingTokens == 320)

        let toolDef = [MFTokenizer.FunctionDefinition(
            name: "patch_file",
            description: "Patch file",
            parameters: JSONValue.object(["type": JSONValue.string("object")]))]
        let multiTurnTool = [
            MFTokenizer.Message(role: .user, content: "Fix test"),
            MFTokenizer.Message(role: .assistant, content: "Reading"),
            MFTokenizer.Message(role: .tool, content: "file content"),
        ]
        #expect(ServerIntentAligner.alignMessages(multiTurnTool, tools: toolDef, dialect: .chatml, environment: [:]) == multiTurnTool)

        // Multi-turn text continuation preserves the exact Turn-1 aligned prefix so ServerPromptCache hits.
        let turn1Aligned = ServerIntentAligner.alignMessages(terseUser, tools: [], dialect: .chatml, environment: [:])
        let turn2Input: [MFTokenizer.Message] = terseUser + [
            .init(role: .assistant, content: "Sorted."),
            .init(role: .user, content: "Explain the complexity"),
        ]
        let turn2Aligned = ServerIntentAligner.alignMessages(turn2Input, tools: [], dialect: .chatml, environment: [:])
        #expect(turn2Aligned.count == turn2Input.count + 1)
        #expect(Array(turn2Aligned.prefix(turn1Aligned.count)) == turn1Aligned)
    }
}
