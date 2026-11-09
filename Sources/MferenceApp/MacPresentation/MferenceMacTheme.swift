import AppKit
import SwiftUI

public enum AppAppearance: String, CaseIterable, Identifiable, Sendable {
    case system
    case light
    case dark

    public static let storageKey = "Mference.appearance"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .system: "System"
        case .light: "Light"
        case .dark: "Dark"
        }
    }

    public var systemImage: String {
        switch self {
        case .system: "circle.lefthalf.filled"
        case .light: "sun.max"
        case .dark: "moon"
        }
    }

    public var preferredColorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light: .light
        case .dark: .dark
        }
    }

    public static func resolve(_ storedValue: String) -> Self {
        Self(rawValue: storedValue) ?? .system
    }
}

public enum MferenceMacTheme {
    /// Vibrant electric blue accent replacing the previous green (#3B82F6)
    public static let accentNSColor = NSColor(
        srgbRed: 59.0 / 255.0,
        green: 130.0 / 255.0,
        blue: 246.0 / 255.0,
        alpha: 1.0)

    public static let accentColor = Color(nsColor: accentNSColor)

    /// Pure pitch black window background (#000000)
    public static let windowBackgroundNSColor = NSColor(srgbRed: 0, green: 0, blue: 0, alpha: 1.0)
    public static let windowBackgroundColor = Color.black

    /// Pure pitch black sidebar background (#000000)
    public static let sidebarBackgroundNSColor = NSColor.black
    public static let sidebarBackgroundColor = Color.black

    /// Dark obsidian surface / card background for prompt composer, HUD, etc. (#141418)
    public static let surfaceBackgroundNSColor = NSColor(
        srgbRed: 20.0 / 255.0,
        green: 20.0 / 255.0,
        blue: 24.0 / 255.0,
        alpha: 1.0)
    public static let surfaceBackgroundColor = Color(nsColor: surfaceBackgroundNSColor)

    /// Subtle card border stroke for crisp contrast against pitch black
    public static let surfaceBorderColor = Color.white.opacity(0.12)
}
