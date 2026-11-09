import AppKit
import MferenceAppCore
import MferenceMacPresentation
import SwiftUI

struct RichTranscriptView: View {
    let messages: [AppChatMessage]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    ForEach(messages) { message in
                        VStack(alignment: .leading, spacing: 6) {
                            if message.role == .user {
                                Text("You")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundStyle(Color(white: 0.55))

                                Text(message.content)
                                    .font(.system(size: 13.5))
                                    .foregroundStyle(Color(white: 0.95))
                                    .lineSpacing(3)
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.top, 2)
                            } else {
                                Text("Answer")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundStyle(MferenceMacTheme.accentColor)

                                RichMessageView(content: message.content)
                                    .padding(.top, 2)
                            }
                        }
                        .id(message.id)
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
            }
            .onAppear {
                if let lastID = messages.last?.id {
                    proxy.scrollTo(lastID, anchor: .bottom)
                }
            }
            .onChange(of: messages.count) {
                if let lastID = messages.last?.id {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(lastID, anchor: .bottom)
                    }
                }
            }
        }
    }
}
