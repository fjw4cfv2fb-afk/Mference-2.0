import Foundation
import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import AppKit
import ImageIO
import UniformTypeIdentifiers

// MARK: - Output Configuration

struct VoiceClip {
    let start: Double
    let wavPath: String
}

struct ReelSpec {
    let filename: String
    let duration: Double
    let boundaries: [Double]
    let sceneCuts: [Double]
    let voiceClips: [VoiceClip]
    let previewTimes: [(label: String, time: Double)]
    let renderer: (_ ctx: CGContext, _ time: Double, _ duration: Double, _ boundaries: [Double]) -> Void
}

private let kWidth = 1080
private let kHeight = 1920
private let kFPS: Int32 = 30
private let kSampleRate: Int = 44_100
private let kChannels: Int = 2
private let kAudioSamplesPerVideoFrame: Int = kSampleRate / Int(kFPS) // 1470 exact

// MARK: - Math & Color Helpers

@inline(__always)
func clamp01(_ x: Double) -> Double { max(0.0, min(1.0, x)) }

@inline(__always)
func easeOutCubic(_ x: Double) -> Double {
    let t = clamp01(x)
    return 1.0 - pow(1.0 - t, 3.0)
}

@inline(__always)
func easeOutBack(_ x: Double) -> Double {
    let t = clamp01(x) - 1.0
    let c1 = 1.70158
    let c3 = c1 + 1.0
    return 1.0 + c3 * t * t * t + c1 * t * t
}

@inline(__always)
func easeInOut(_ x: Double) -> Double {
    let t = clamp01(x)
    return t < 0.5 ? 2.0 * t * t : 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0
}

@inline(__always)
func progressIn(_ time: Double, _ start: Double, _ span: Double) -> Double {
    guard span > 0 else { return time >= start ? 1.0 : 0.0 }
    return clamp01((time - start) / span)
}

func nsColor(_ hex: UInt32, alpha: CGFloat = 1.0) -> NSColor {
    let r = CGFloat((hex >> 16) & 0xFF) / 255.0
    let g = CGFloat((hex >> 8) & 0xFF) / 255.0
    let b = CGFloat(hex & 0xFF) / 255.0
    return NSColor(srgbRed: r, green: g, blue: b, alpha: alpha)
}

func cgColor(_ hex: UInt32, alpha: CGFloat = 1.0) -> CGColor {
    nsColor(hex, alpha: alpha).cgColor
}

// MARK: - Drawing Primitives (Top-Left Flipped Coordinates)

func drawRoundedRect(
    _ ctx: CGContext,
    rect: CGRect,
    radius: CGFloat,
    fill: CGColor?,
    stroke: CGColor? = nil,
    lineWidth: CGFloat = 2.0
) {
    let path = CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil)
    if let fill {
        ctx.saveGState()
        ctx.addPath(path)
        ctx.setFillColor(fill)
        ctx.fillPath()
        ctx.restoreGState()
    }
    if let stroke {
        ctx.saveGState()
        ctx.addPath(path)
        ctx.setStrokeColor(stroke)
        ctx.setLineWidth(lineWidth)
        ctx.strokePath()
        ctx.restoreGState()
    }
}

func drawLinearGradientRect(
    _ ctx: CGContext,
    rect: CGRect,
    radius: CGFloat,
    colors: [CGColor],
    locations: [CGFloat],
    horizontal: Bool = true
) {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let grad = CGGradient(colorsSpace: space, colors: colors as CFArray, locations: locations) else {
        return
    }
    let path = CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil)
    ctx.saveGState()
    ctx.addPath(path)
    ctx.clip()
    let start = CGPoint(x: rect.minX, y: rect.minY)
    let end = horizontal
        ? CGPoint(x: rect.maxX, y: rect.minY)
        : CGPoint(x: rect.minX, y: rect.maxY)
    ctx.drawLinearGradient(grad, start: start, end: end, options: [])
    ctx.restoreGState()
}

func drawText(
    _ text: String,
    rect: CGRect,
    size: CGFloat,
    weight: NSFont.Weight = .regular,
    colorHex: UInt32,
    alpha: CGFloat = 1.0,
    mono: Bool = false,
    align: NSTextAlignment = .left,
    lineSpacing: CGFloat = 4.0,
    letterSpacing: CGFloat = 0.0
) {
    let font: NSFont = mono
        ? NSFont.monospacedSystemFont(ofSize: size, weight: weight)
        : NSFont.systemFont(ofSize: size, weight: weight)
    let style = NSMutableParagraphStyle()
    style.alignment = align
    style.lineSpacing = lineSpacing
    style.lineBreakMode = .byWordWrapping

    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: nsColor(colorHex, alpha: alpha),
        .paragraphStyle: style,
        .kern: letterSpacing,
    ]
    NSAttributedString(string: text, attributes: attrs).draw(in: rect)
}

func drawSingleLineBadge(
    _ ctx: CGContext,
    text: String,
    x: CGFloat,
    y: CGFloat,
    width: CGFloat,
    height: CGFloat = 46,
    bgHex: UInt32,
    bgAlpha: CGFloat = 0.22,
    borderHex: UInt32,
    textHex: UInt32,
    fontSize: CGFloat = 20,
    mono: Bool = false,
    alpha: CGFloat = 1.0
) {
    let r = CGRect(x: x, y: y, width: width, height: height)
    drawRoundedRect(
        ctx,
        rect: r,
        radius: height / 2.0,
        fill: cgColor(bgHex, alpha: bgAlpha * alpha),
        stroke: cgColor(borderHex, alpha: 0.65 * alpha),
        lineWidth: 2.0
    )
    let textRect = CGRect(x: x + 12, y: y + (height - fontSize * 1.25) / 2.0, width: width - 24, height: fontSize * 1.4)
    drawText(text, rect: textRect, size: fontSize, weight: .bold, colorHex: textHex, alpha: alpha, mono: mono, align: .center, lineSpacing: 0, letterSpacing: 0.8)
}

func drawBackground(_ ctx: CGContext, time: Double, duration: Double, subtitle: String) {
    // Base GitHub Dark Canvas
    drawLinearGradientRect(
        ctx,
        rect: CGRect(x: 0, y: 0, width: kWidth, height: kHeight),
        radius: 0,
        colors: [cgColor(0x0D1117), cgColor(0x0E141E), cgColor(0x121926)],
        locations: [0.0, 0.55, 1.0],
        horizontal: false
    )

    // Subtle drifting ambient glow orbs
    if let space = CGColorSpace(name: CGColorSpace.sRGB) {
        let pulse = 0.5 + 0.5 * sin(time * 1.4)
        if let g1 = CGGradient(
            colorsSpace: space,
            colors: [cgColor(0x388BFD, alpha: 0.16 + 0.04 * pulse), cgColor(0x388BFD, alpha: 0.0)] as CFArray,
            locations: [0.0, 1.0]
        ) {
            ctx.drawRadialGradient(g1, startCenter: CGPoint(x: 840, y: 360), startRadius: 0, endCenter: CGPoint(x: 840, y: 360), endRadius: 540, options: [])
        }
        if let g2 = CGGradient(
            colorsSpace: space,
            colors: [cgColor(0x2EA043, alpha: 0.14), cgColor(0x2EA043, alpha: 0.0)] as CFArray,
            locations: [0.0, 1.0]
        ) {
            ctx.drawRadialGradient(g2, startCenter: CGPoint(x: 220, y: 1380), startRadius: 0, endCenter: CGPoint(x: 220, y: 1380), endRadius: 520, options: [])
        }
    }

    // Subtle engineering grid in background
    ctx.saveGState()
    ctx.setStrokeColor(cgColor(0x30363D, alpha: 0.18))
    ctx.setLineWidth(1.0)
    for gx in stride(from: 72, through: 980, by: 88) {
        ctx.move(to: CGPoint(x: gx, y: 140))
        ctx.addLine(to: CGPoint(x: gx, y: 1580))
    }
    for gy in stride(from: 200, through: 1560, by: 88) {
        ctx.move(to: CGPoint(x: 72, y: CGFloat(gy)))
        ctx.addLine(to: CGPoint(x: 952, y: CGFloat(gy)))
    }
    ctx.strokePath()
    ctx.restoreGState()

    // Top rainbow accent stripe
    drawLinearGradientRect(
        ctx,
        rect: CGRect(x: 0, y: 0, width: kWidth, height: 10),
        radius: 0,
        colors: [cgColor(0x2EA043), cgColor(0x388BFD), cgColor(0xA371F7)],
        locations: [0.0, 0.5, 1.0],
        horizontal: true
    )

    // Persistent Safe-Zone Header Pill + Timeline Progress Bar (y: 120..210)
    drawSingleLineBadge(
        ctx,
        text: subtitle,
        x: 72,
        y: 126,
        width: 700,
        height: 44,
        bgHex: 0x1F6FEB,
        bgAlpha: 0.18,
        borderHex: 0x388BFD,
        textHex: 0x58A6FF,
        fontSize: 19,
        mono: true
    )

    // Timeline progress track
    let track = CGRect(x: 72, y: 188, width: 880, height: 6)
    drawRoundedRect(ctx, rect: track, radius: 3, fill: cgColor(0x21262D))
    let frac = CGFloat(clamp01(time / duration))
    if frac > 0.002 {
        drawLinearGradientRect(
            ctx,
            rect: CGRect(x: 72, y: 188, width: max(6, 880 * frac), height: 6),
            radius: 3,
            colors: [cgColor(0x3FB950), cgColor(0x58A6FF), cgColor(0xD2A8FF)],
            locations: [0.0, 0.5, 1.0],
            horizontal: true
        )
    }

    // Safe bottom watermark above IG captions
    drawText(
        "github.com/fjw4cfv2fb-afk/Mference-2.0  ·  Apache-2.0 / MIT",
        rect: CGRect(x: 72, y: 1500, width: 880, height: 32),
        size: 19,
        weight: .semibold,
        colorHex: 0x8B949E,
        alpha: 0.9,
        mono: true,
        align: .center
    )
}

func drawMetallicEmblem(_ ctx: CGContext, x: CGFloat, y: CGFloat, size: CGFloat, alpha: CGFloat = 1.0) {
    let rect = CGRect(x: x, y: y, width: size, height: size)
    drawLinearGradientRect(
        ctx,
        rect: rect,
        radius: size * 0.24,
        colors: [cgColor(0x242B36, alpha: alpha), cgColor(0x11161D, alpha: alpha)],
        locations: [0.0, 1.0],
        horizontal: false
    )
    drawRoundedRect(ctx, rect: rect, radius: size * 0.24, fill: nil, stroke: cgColor(0x30363D, alpha: alpha), lineWidth: 3.0)

    // Vector "M" mark inside tile
    let mPath = CGMutablePath()
    func pt(_ u: CGFloat, _ v: CGFloat) -> CGPoint {
        CGPoint(x: x + size * u, y: y + size * v)
    }
    mPath.move(to: pt(0.24, 0.74))
    mPath.addLine(to: pt(0.24, 0.26))
    mPath.addLine(to: pt(0.38, 0.26))
    mPath.addLine(to: pt(0.50, 0.48))
    mPath.addLine(to: pt(0.62, 0.26))
    mPath.addLine(to: pt(0.76, 0.26))
    mPath.addLine(to: pt(0.76, 0.74))
    mPath.addLine(to: pt(0.635, 0.74))
    mPath.addLine(to: pt(0.635, 0.45))
    mPath.addLine(to: pt(0.50, 0.68))
    mPath.addLine(to: pt(0.365, 0.45))
    mPath.addLine(to: pt(0.365, 0.74))
    mPath.closeSubpath()

    ctx.saveGState()
    ctx.addPath(mPath)
    ctx.clip()
    if let space = CGColorSpace(name: CGColorSpace.sRGB),
       let grad = CGGradient(
           colorsSpace: space,
           colors: [cgColor(0xF0F6FC, alpha: alpha), cgColor(0x9ECBFF, alpha: alpha), cgColor(0x58A6FF, alpha: alpha)] as CFArray,
           locations: [0.0, 0.55, 1.0]
       ) {
        ctx.drawLinearGradient(grad, start: pt(0.2, 0.2), end: pt(0.8, 0.8), options: [])
    }
    ctx.restoreGState()
}

// MARK: - REEL 1: "I ran Qwen 3.6 35B on a 16 GB Mac" (18.5s)

func renderReel1(_ ctx: CGContext, time: Double, duration: Double, boundaries: [Double]) {
    let b = boundaries.count >= 6 ? boundaries : [0.0, 2.5, 7.5, 12.5, 16.0, duration]
    drawBackground(ctx, time: time, duration: duration, subtitle: "MFERENCE 2.0  ·  SWIFT 6.1 + METAL 3")

    if time < b[1] {
        // SCENE 1: HOOK
        let pMyth = easeOutCubic(progressIn(time, b[0], 0.50))
        let pFact = easeOutCubic(progressIn(time, b[0] + min(2.2, (b[1] - b[0]) * 0.32), 0.60))
        let pSpec = easeOutCubic(progressIn(time, b[0] + min(3.4, (b[1] - b[0]) * 0.50), 0.55))

        // Myth Card
        let mythY = 260 + CGFloat(1.0 - pMyth) * 30
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: mythY, width: 880, height: 220), radius: 24, fill: cgColor(0x161B22, alpha: CGFloat(pMyth)), stroke: cgColor(0xF85149, alpha: 0.45 * CGFloat(pMyth)), lineWidth: 2.5)
        drawText("THE MYTH", rect: CGRect(x: 112, y: mythY + 32, width: 400, height: 30), size: 20, weight: .bold, colorHex: 0xF85149, alpha: CGFloat(pMyth), mono: true, letterSpacing: 1.5)
        drawText("“You need 64 GB+ RAM to run\na 35B MoE coding model.”", rect: CGRect(x: 112, y: mythY + 78, width: 800, height: 120), size: 40, weight: .bold, colorHex: 0xC9D1D9, alpha: CGFloat(pMyth), lineSpacing: 8)

        // Strike-through / Arrow
        if pFact > 0.02 {
            drawText("↓", rect: CGRect(x: 72, y: 500, width: 880, height: 56), size: 46, weight: .black, colorHex: 0x3FB950, alpha: CGFloat(pFact), align: .center)
        }

        // Fact Hero Card
        let factY = 580 + CGFloat(1.0 - pFact) * 35
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: factY, width: 880, height: 420), radius: 28, fill: cgColor(0x161B22, alpha: CGFloat(pFact)), stroke: cgColor(0x3FB950, alpha: 0.7 * CGFloat(pFact)), lineWidth: 3.0)
        drawSingleLineBadge(ctx, text: "VERIFIED REALITY", x: 112, y: factY + 36, width: 250, height: 42, bgHex: 0x238636, bgAlpha: 0.25, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 18, mono: true, alpha: CGFloat(pFact))
        drawText("I ran Qwen 3.6\n35B-A3B on a\n16 GB Mac.", rect: CGRect(x: 112, y: factY + 96, width: 800, height: 244), size: 56, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(pFact), lineSpacing: 6)
        drawText("Zero Python in inference  ·  Native NVMe expert streaming", rect: CGRect(x: 112, y: factY + 356, width: 800, height: 34), size: 22, weight: .semibold, colorHex: 0x58A6FF, alpha: CGFloat(pFact), mono: true)

        // Spec Pills Grid
        let pills = [
            ("100% Pass@1 (8/8)", 0x3FB950),
            ("18.36 tok/s Peak", 0x58A6FF),
            ("256 MoE Experts · INT4", 0xD2A8FF),
            ("32-Slot NVMe Cache", 0xE6EDF3),
        ]
        for (i, item) in pills.enumerated() {
            let col = i % 2
            let row = i / 2
            let px: CGFloat = col == 0 ? 72 : 524
            let py: CGFloat = 1048 + CGFloat(row) * 96
            drawRoundedRect(ctx, rect: CGRect(x: px, y: py, width: 428, height: 76), radius: 18, fill: cgColor(0x161B22, alpha: CGFloat(pSpec)), stroke: cgColor(0x30363D, alpha: CGFloat(pSpec)), lineWidth: 2.0)
            drawText(item.0, rect: CGRect(x: px + 18, y: py + 21, width: 392, height: 36), size: 25, weight: .bold, colorHex: UInt32(item.1), alpha: CGFloat(pSpec), mono: true, align: .center)
        }
    } else if time < b[2] {
        // SCENE 2: LIVE CODE STREAM + 16 GB ACTIVITY MONITOR
        let lt = time - b[1]
        let span = max(1.0, b[2] - b[1])
        let enter = easeOutCubic(progressIn(lt, 0.0, 0.40))

        drawText("Streaming 35B MoE from NVMe", rect: CGRect(x: 72, y: 236, width: 880, height: 54), size: 42, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter))
        drawText("Swift Coding (C2)  ·  Green Unified-Memory Pressure", rect: CGRect(x: 72, y: 294, width: 880, height: 34), size: 22, weight: .medium, colorHex: 0x8B949E, alpha: CGFloat(enter))

        // Terminal Window
        let term = CGRect(x: 72, y: 348, width: 880, height: 676)
        drawRoundedRect(ctx, rect: term, radius: 22, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 2.5)
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 348, width: 880, height: 64), radius: 22, fill: cgColor(0x1C2128, alpha: CGFloat(enter)))
        // Traffic dots
        let dotColors: [UInt32] = [0xFF5F56, 0xFFBD2E, 0x27C93F]
        for (i, hex) in dotColors.enumerated() {
            ctx.saveGState()
            ctx.setFillColor(cgColor(hex, alpha: CGFloat(enter)))
            ctx.fillEllipse(in: CGRect(x: 102 + CGFloat(i) * 28, y: 371, width: 16, height: 16))
            ctx.restoreGState()
        }
        drawText("Mference.app · Qwen 3.6 35B-A3B", rect: CGRect(x: 196, y: 367, width: 420, height: 28), size: 19, weight: .semibold, colorHex: 0xE6EDF3, alpha: CGFloat(enter))
        drawText("18.36 tok/s", rect: CGRect(x: 720, y: 367, width: 204, height: 28), size: 20, weight: .bold, colorHex: 0x58A6FF, alpha: CGFloat(enter), mono: true, align: .right)

        // Think banner
        drawRoundedRect(ctx, rect: CGRect(x: 96, y: 428, width: 832, height: 48), radius: 10, fill: cgColor(0x0D1117, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 1.5)
        drawText("<think> 206 tok -> </think>  ·  structuredCodeOrTool", rect: CGRect(x: 116, y: 440, width: 650, height: 28), size: 17.5, weight: .regular, colorHex: 0x8B949E, alpha: CGFloat(enter), mono: true)
        drawSingleLineBadge(ctx, text: "9/9 OK", x: 792, y: 435, width: 118, height: 34, bgHex: 0x238636, bgAlpha: 0.30, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 16, mono: true, alpha: CGFloat(enter))

        let codeLines: [(String, UInt32)] = [
            ("func shortestSubarrayAtLeastK(_ nums: [Int], _ k: Int) -> Int {", 0x79C0FF),
            ("    let n = nums.count", 0xC9D1D9),
            ("    var prefix = Array(repeating: 0, count: n + 1)", 0xC9D1D9),
            ("    for i in 0..<n { prefix[i+1] = prefix[i] + nums[i] }", 0xC9D1D9),
            ("    var dq = Array(repeating: 0, count: n + 1)", 0xC9D1D9),
            ("    var head = 0, tail = 0, ans = Int.max", 0xC9D1D9),
            ("    for j in 0...n {", 0xC9D1D9),
            ("        while head < tail && prefix[j] - prefix[dq[head]] >= k {", 0xD2A8FF),
            ("            ans = min(ans, j - dq[head]); head += 1", 0xC9D1D9),
            ("        }", 0xC9D1D9),
            ("        while head < tail && prefix[dq[tail-1]] >= prefix[j] { tail -= 1 }", 0xD2A8FF),
            ("        dq[tail] = j; tail += 1", 0xC9D1D9),
            ("    }", 0xC9D1D9),
            ("    return ans == Int.max ? -1 : ans  // MTP: 82.1% accept", 0x3FB950),
            ("}", 0x79C0FF),
        ]

        let typeSpan = min(6.2, span * 0.65)
        let revealedLines = Int(floor(clamp01(lt / typeSpan) * Double(codeLines.count))) + 1
        for i in 0..<min(revealedLines, codeLines.count) {
            let lineFade = progressIn(lt, Double(i) * (typeSpan / Double(codeLines.count)), 0.16)
            let yPos = 496 + CGFloat(i) * 31
            drawText(codeLines[i].0, rect: CGRect(x: 100, y: yPos, width: 824, height: 28), size: 15.5, weight: .medium, colorHex: codeLines[i].1, alpha: CGFloat(lineFade * enter), mono: true)
        }

        // Bottom Terminal Telemetry Bar
        drawRoundedRect(ctx, rect: CGRect(x: 96, y: 964, width: 832, height: 42), radius: 10, fill: cgColor(0x0D1117, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 1.5)
        drawText("visible: 249 tok · wall: 27.96s · MTP: 82.1% (2.51 tok/round)", rect: CGRect(x: 114, y: 974, width: 796, height: 24), size: 17, weight: .bold, colorHex: 0x3FB950, alpha: CGFloat(enter), mono: true, align: .center)

        // Activity Monitor Card Below (y: 1052..1452)
        let am = CGRect(x: 72, y: 1052, width: 880, height: 396)
        drawRoundedRect(ctx, rect: am, radius: 22, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 2.5)
        drawText("ACTIVITY MONITOR · 16 GB UNIFIED MEMORY", rect: CGRect(x: 106, y: 1082, width: 560, height: 28), size: 19, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(enter), mono: true, letterSpacing: 1.0)
        drawSingleLineBadge(ctx, text: "GREEN · FLAT", x: 706, y: 1072, width: 214, height: 42, bgHex: 0x238636, bgAlpha: 0.25, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 19, mono: true, alpha: CGFloat(enter))

        // Animated green pressure bars
        drawRoundedRect(ctx, rect: CGRect(x: 106, y: 1134, width: 812, height: 116), radius: 14, fill: cgColor(0x0D1117, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 1.5)
        for b in 0..<18 {
            let phase = lt * 4.5 + Double(b) * 0.45
            let barH = CGFloat(44 + 26 * (0.5 + 0.5 * sin(phase)))
            let bx = 132 + CGFloat(b) * 42
            drawRoundedRect(ctx, rect: CGRect(x: bx, y: 1230 - barH, width: 26, height: barH), radius: 5, fill: cgColor(0x2EA043, alpha: 0.90 * CGFloat(enter)))
        }

        // Spec rows inside Activity Monitor
        let rows: [(String, String, UInt32)] = [
            ("NVMe Expert Cache", "32 Slots (Pread + Tail Warm)", 0xE6EDF3),
            ("Resident Core + MTP", "1.41 GB + MADV_RANDOM Slab", 0x58A6FF),
            ("Inference Stack", "Zero Python · Single Metal Server", 0x3FB950),
        ]
        for (idx, row) in rows.enumerated() {
            let ry = 1278 + CGFloat(idx) * 50
            drawText(row.0, rect: CGRect(x: 106, y: ry, width: 320, height: 32), size: 22, weight: .medium, colorHex: 0x8B949E, alpha: CGFloat(enter))
            drawText(row.1, rect: CGRect(x: 420, y: ry, width: 498, height: 32), size: 22, weight: .bold, colorHex: row.2, alpha: CGFloat(enter), mono: true, align: .right)
        }
    } else if time < b[3] {
        // SCENE 3: LOSSLESS WEIGHT SURGERY REVEAL
        let lt = time - b[2]
        let enter = easeOutCubic(progressIn(lt, 0.0, 0.4))
        let barGrow = easeOutCubic(progressIn(lt, 0.6, 2.2))
        let cardGrow = easeOutCubic(progressIn(lt, 1.8, 0.7))

        drawText("Lossless Weight Surgery", rect: CGRect(x: 72, y: 244, width: 880, height: 56), size: 46, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter))
        drawText("Fixed quantization collapse without changing 1 INT4 nibble", rect: CGRect(x: 72, y: 308, width: 880, height: 34), size: 22, weight: .medium, colorHex: 0x8B949E, alpha: CGFloat(enter))

        // Baseline Card (Red 25%)
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 376, width: 880, height: 220), radius: 22, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 2.5)
        drawText("RAW INT4 BASELINE", rect: CGRect(x: 112, y: 406, width: 420, height: 28), size: 19, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(enter), mono: true)
        drawText("25.0%", rect: CGRect(x: 112, y: 446, width: 240, height: 68), size: 58, weight: .heavy, colorHex: 0xF85149, alpha: CGFloat(enter))
        drawText("2 / 8 passed  (6/8 stuck in <think>)", rect: CGRect(x: 340, y: 468, width: 570, height: 34), size: 24, weight: .semibold, colorHex: 0xC9D1D9, alpha: CGFloat(enter))
        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 534, width: 800, height: 24), radius: 12, fill: cgColor(0x21262D, alpha: CGFloat(enter)))
        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 534, width: max(24, 800 * 0.25 * CGFloat(barGrow)), height: 24), radius: 12, fill: cgColor(0xF85149, alpha: CGFloat(enter)))

        // Mference 2.0 Card (Green -> 100%)
        let pct = 25.0 + 75.0 * barGrow
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 632, width: 880, height: 256), radius: 24, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x3FB950, alpha: 0.75 * CGFloat(enter)), lineWidth: 3.0)
        drawText("MFERENCE 2.0 (SURGERY + HYBRID MTP)", rect: CGRect(x: 112, y: 664, width: 520, height: 28), size: 19, weight: .bold, colorHex: 0x3FB950, alpha: CGFloat(enter), mono: true)
        drawSingleLineBadge(ctx, text: "+75.0 pts", x: 744, y: 656, width: 168, height: 42, bgHex: 0x238636, bgAlpha: 0.30, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 21, mono: true, alpha: CGFloat(barGrow))
        drawText(String(format: "%.1f%%", pct), rect: CGRect(x: 112, y: 706, width: 280, height: 74), size: 64, weight: .heavy, colorHex: 0x3FB950, alpha: CGFloat(enter))
        drawText("8 / 8 Pass@1 Verified", rect: CGRect(x: 376, y: 734, width: 530, height: 36), size: 28, weight: .bold, colorHex: 0xF0F6FC, alpha: CGFloat(enter))
        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 806, width: 800, height: 28), radius: 14, fill: cgColor(0x21262D, alpha: CGFloat(enter)))
        drawLinearGradientRect(
            ctx,
            rect: CGRect(x: 112, y: 806, width: max(28, 800 * CGFloat(pct / 100.0)), height: 28),
            radius: 14,
            colors: [cgColor(0x2EA043, alpha: CGFloat(enter)), cgColor(0x3FB950, alpha: CGFloat(enter))],
            locations: [0.0, 1.0]
        )

        // Equation + Impact Card
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 924, width: 880, height: 504), radius: 24, fill: cgColor(0x161B22, alpha: CGFloat(cardGrow)), stroke: cgColor(0x30363D, alpha: CGFloat(cardGrow)), lineWidth: 2.5)
        drawText("LOSSLESS BF16 SCALE / BIAS TRANSFORM", rect: CGRect(x: 112, y: 958, width: 640, height: 28), size: 19, weight: .bold, colorHex: 0x58A6FF, alpha: CGFloat(cardGrow), mono: true)

        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 1006, width: 800, height: 112), radius: 16, fill: cgColor(0x0D1117, alpha: CGFloat(cardGrow)), stroke: cgColor(0x388BFD, alpha: 0.45 * CGFloat(cardGrow)), lineWidth: 2.0)
        drawText("α · w  =  (α · s) · q  +  (α · b)", rect: CGRect(x: 124, y: 1038, width: 776, height: 52), size: 38, weight: .bold, colorHex: 0xF0F6FC, alpha: CGFloat(cardGrow), mono: true, align: .center)

        // Two metric boxes inside card
        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 1148, width: 384, height: 240), radius: 18, fill: cgColor(0x0D1117, alpha: CGFloat(cardGrow)), stroke: cgColor(0x30363D, alpha: CGFloat(cardGrow)), lineWidth: 1.5)
        drawText("INT4 NIBBLES CHANGED", rect: CGRect(x: 132, y: 1178, width: 344, height: 28), size: 18, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(cardGrow), mono: true, align: .center)
        drawText("0", rect: CGRect(x: 132, y: 1222, width: 344, height: 84), size: 76, weight: .heavy, colorHex: 0x58A6FF, alpha: CGFloat(cardGrow), align: .center)
        drawText("Zero re-quantization loss", rect: CGRect(x: 132, y: 1324, width: 344, height: 32), size: 20, weight: .semibold, colorHex: 0xC9D1D9, alpha: CGFloat(cardGrow), align: .center)

        drawRoundedRect(ctx, rect: CGRect(x: 528, y: 1148, width: 384, height: 240), radius: 18, fill: cgColor(0x0D1117, alpha: CGFloat(cardGrow)), stroke: cgColor(0x30363D, alpha: CGFloat(cardGrow)), lineWidth: 1.5)
        drawText("<THINK> TOKEN WASTE", rect: CGRect(x: 548, y: 1178, width: 344, height: 28), size: 18, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(cardGrow), mono: true, align: .center)
        drawText("-69.5%", rect: CGRect(x: 548, y: 1226, width: 344, height: 80), size: 68, weight: .heavy, colorHex: 0x3FB950, alpha: CGFloat(cardGrow), align: .center)
        drawText("413.4 -> 126.1 avg tokens", rect: CGRect(x: 548, y: 1324, width: 344, height: 32), size: 20, weight: .semibold, colorHex: 0xC9D1D9, alpha: CGFloat(cardGrow), align: .center)
    } else if time < b[4] {
        // SCENE 4: HYBRID MTP + GATED-DELTANET ROLLBACK
        let lt = time - b[3]
        let enter = easeOutCubic(progressIn(lt, 0.0, 0.4))
        let p0 = easeOutBack(progressIn(lt, 0.45, 0.40))
        let p1 = easeOutBack(progressIn(lt, 1.00, 0.40))
        let p2 = easeOutBack(progressIn(lt, 1.55, 0.40))

        drawText("Hybrid MoE MTP + 6-Gram PLD", rect: CGRect(x: 72, y: 244, width: 880, height: 56), size: 42, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter))
        drawText("Speculative decoding across 30 Gated-DeltaNet linear layers", rect: CGRect(x: 72, y: 306, width: 880, height: 34), size: 21, weight: .medium, colorHex: 0x8B949E, alpha: CGFloat(enter))

        // Speculative Token Pills Pipeline Card
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 374, width: 880, height: 224), radius: 22, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 2.5)
        drawText("SINGLE-STEP GPU VERIFY + ROLLBACK", rect: CGRect(x: 112, y: 406, width: 640, height: 28), size: 19, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(enter), mono: true)

        drawSingleLineBadge(ctx, text: "bonus ✓", x: 112, y: 468, width: 220, height: 74, bgHex: 0x1F6FEB, bgAlpha: 0.25, borderHex: 0x58A6FF, textHex: 0x58A6FF, fontSize: 26, mono: true, alpha: CGFloat(clamp01(p0)))
        drawText("+", rect: CGRect(x: 344, y: 482, width: 42, height: 48), size: 34, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(clamp01(p1)), mono: true, align: .center)
        drawSingleLineBadge(ctx, text: "MTP d0 ✓", x: 396, y: 468, width: 230, height: 74, bgHex: 0x238636, bgAlpha: 0.28, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 26, mono: true, alpha: CGFloat(clamp01(p1)))
        drawText("+", rect: CGRect(x: 638, y: 482, width: 42, height: 48), size: 34, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(clamp01(p2)), mono: true, align: .center)
        drawSingleLineBadge(ctx, text: "PLD d1 ✓", x: 690, y: 468, width: 222, height: 74, bgHex: 0x8957E5, bgAlpha: 0.28, borderHex: 0xD2A8FF, textHex: 0xD2A8FF, fontSize: 26, mono: true, alpha: CGFloat(clamp01(p2)))

        // Main Acceptance Hero Card
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 634, width: 880, height: 320), radius: 24, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0xA371F7, alpha: 0.65 * CGFloat(enter)), lineWidth: 3.0)
        drawText("NATIVE 19-TENSOR MoE MTP ACCEPTANCE", rect: CGRect(x: 112, y: 672, width: 640, height: 28), size: 19, weight: .bold, colorHex: 0xD2A8FF, alpha: CGFloat(enter), mono: true)
        drawText("83.1%", rect: CGRect(x: 112, y: 720, width: 320, height: 96), size: 84, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter))
        drawText("Overall Accept Rate\n2.55 tokens / round", rect: CGRect(x: 426, y: 736, width: 480, height: 84), size: 28, weight: .bold, colorHex: 0xC9D1D9, alpha: CGFloat(enter), lineSpacing: 6)
        drawSingleLineBadge(ctx, text: "AIME MATH PEAK: 94.3%  (2.85 tok/round)", x: 112, y: 856, width: 800, height: 56, bgHex: 0x238636, bgAlpha: 0.24, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 23, mono: true, alpha: CGFloat(enter))

        // Rollback + Latency Architecture Cards
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 990, width: 880, height: 210), radius: 22, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 2.5)
        drawText("4-Slot Gated-DeltaNet Rollback Arena", rect: CGRect(x: 112, y: 1026, width: 800, height: 42), size: 32, weight: .bold, colorHex: 0x58A6FF, alpha: CGFloat(enter))
        drawText("Snapshots linear recurrent state on verify passes so partial\nrejections rewind bit-exact with greedy decode.", rect: CGRect(x: 112, y: 1084, width: 800, height: 80), size: 24, weight: .regular, colorHex: 0xC9D1D9, alpha: CGFloat(enter), lineSpacing: 6)

        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 1232, width: 880, height: 200), radius: 22, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 2.5)
        drawText("Exact Top-2 EVICT Margin + Async Accept", rect: CGRect(x: 112, y: 1268, width: 800, height: 42), size: 32, weight: .bold, colorHex: 0x3FB950, alpha: CGFloat(enter))
        drawText("-21.3% total task wall time  ·  17.88 tok/s AIME peak", rect: CGRect(x: 112, y: 1330, width: 800, height: 42), size: 24, weight: .semibold, colorHex: 0xE6EDF3, alpha: CGFloat(enter), mono: true)
    } else {
        // SCENE 5: CTA
        let lt = time - b[4]
        let enter = easeOutCubic(progressIn(lt, 0.0, 0.45))

        drawMetallicEmblem(ctx, x: 422, y: 316, size: 180, alpha: CGFloat(enter))
        drawText("Mference 2.0", rect: CGRect(x: 72, y: 540, width: 880, height: 92), size: 78, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter), align: .center)
        drawText("Run 35B MoE Locally on a 16 GB Mac", rect: CGRect(x: 72, y: 644, width: 880, height: 48), size: 32, weight: .semibold, colorHex: 0xC9D1D9, alpha: CGFloat(enter), align: .center)

        drawRoundedRect(ctx, rect: CGRect(x: 96, y: 742, width: 832, height: 340), radius: 26, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x388BFD, alpha: 0.6 * CGFloat(enter)), lineWidth: 3.0)
        drawSingleLineBadge(ctx, text: "FREE & OPEN SOURCE  ·  MACOS 15+", x: 196, y: 784, width: 632, height: 50, bgHex: 0x238636, bgAlpha: 0.25, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 21, mono: true, alpha: CGFloat(enter))
        drawText("Prebuilt Mference.app (v2.0.0)\n+ OpenAI API Server (:8080)", rect: CGRect(x: 126, y: 866, width: 772, height: 96), size: 34, weight: .bold, colorHex: 0xF0F6FC, alpha: CGFloat(enter), align: .center, lineSpacing: 8)
        drawText("./mference-ui.sh install qwen36-mtp", rect: CGRect(x: 126, y: 996, width: 772, height: 44), size: 25, weight: .bold, colorHex: 0x58A6FF, alpha: CGFloat(enter), mono: true, align: .center)

        drawRoundedRect(ctx, rect: CGRect(x: 96, y: 1134, width: 832, height: 140), radius: 24, fill: cgColor(0x1F6FEB, alpha: 0.20 * CGFloat(enter)), stroke: cgColor(0x58A6FF, alpha: 0.75 * CGFloat(enter)), lineWidth: 2.5)
        drawText("GITHUB REPO  ·  LINK IN BIO", rect: CGRect(x: 126, y: 1158, width: 772, height: 30), size: 19, weight: .bold, colorHex: 0x58A6FF, alpha: CGFloat(enter), mono: true, align: .center, letterSpacing: 1.5)
        drawText("fjw4cfv2fb-afk / Mference-2.0", rect: CGRect(x: 126, y: 1200, width: 772, height: 48), size: 34, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter), mono: true, align: .center)
    }
}

// MARK: - REEL 2: "Changing 0 weight nibbles took it from 25% to 100%" (17.0s)

func renderReel2(_ ctx: CGContext, time: Double, duration: Double, boundaries: [Double]) {
    let b = boundaries.count >= 6 ? boundaries : [0.0, 2.5, 6.5, 11.5, 15.0, duration]
    drawBackground(ctx, time: time, duration: duration, subtitle: "LOSSLESS INT4 WEIGHT SURGERY  ·  MFERENCE 2.0")

    if time < b[1] {
        // SCENE 1: HOOK
        let p1 = easeOutCubic(progressIn(time, b[0], 0.5))
        let p2 = easeOutCubic(progressIn(time, b[0] + min(2.0, (b[1] - b[0]) * 0.32), 0.6))

        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 266, width: 880, height: 290), radius: 26, fill: cgColor(0x161B22, alpha: CGFloat(p1)), stroke: cgColor(0xF85149, alpha: 0.55 * CGFloat(p1)), lineWidth: 2.5)
        drawText("QUANTIZATION COLLAPSE", rect: CGRect(x: 112, y: 302, width: 520, height: 28), size: 19, weight: .bold, colorHex: 0xF85149, alpha: CGFloat(p1), mono: true)
        drawText("My 4-bit Qwen 3.6 35B\nscored just 25% Pass@1.", rect: CGRect(x: 112, y: 350, width: 800, height: 120), size: 46, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(p1), lineSpacing: 8)
        drawText("6 / 8 benchmark tasks trapped in endless <think> loops.", rect: CGRect(x: 112, y: 492, width: 800, height: 34), size: 22, weight: .medium, colorHex: 0x8B949E, alpha: CGFloat(p1))

        drawText("↓", rect: CGRect(x: 72, y: 584, width: 880, height: 56), size: 48, weight: .black, colorHex: 0x3FB950, alpha: CGFloat(p2), align: .center)

        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 668, width: 880, height: 470), radius: 28, fill: cgColor(0x161B22, alpha: CGFloat(p2)), stroke: cgColor(0x3FB950, alpha: 0.75 * CGFloat(p2)), lineWidth: 3.0)
        drawSingleLineBadge(ctx, text: "NO FINE-TUNING · NO RE-QUANTIZATION", x: 112, y: 708, width: 540, height: 42, bgHex: 0x238636, bgAlpha: 0.25, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 18, mono: true, alpha: CGFloat(p2))
        drawText("Changing 0 weight\nnibbles took it from\n25% to 100%.", rect: CGRect(x: 112, y: 780, width: 800, height: 240), size: 62, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(p2), lineSpacing: 8)
        drawText("Pure BF16 scale & norm algebra in resident model_weights.bin", rect: CGRect(x: 112, y: 1068, width: 800, height: 34), size: 21, weight: .semibold, colorHex: 0x58A6FF, alpha: CGFloat(p2), mono: true)
    } else if time < b[2] {
        // SCENE 2: THE PROBLEM (<think> loops)
        let lt = time - b[1]
        let enter = easeOutCubic(progressIn(lt, 0.0, 0.40))
        let countProg = easeOutCubic(progressIn(lt, 0.4, min(3.2, max(1.8, (b[2] - b[1]) * 0.5))))

        drawText("Why Raw INT4 Breaks", rect: CGRect(x: 72, y: 244, width: 880, height: 56), size: 46, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter))
        drawText("Small 4-bit scale drift pushes models into infinite reasoning loops", rect: CGRect(x: 72, y: 308, width: 880, height: 34), size: 21, weight: .medium, colorHex: 0x8B949E, alpha: CGFloat(enter))

        // Red counter card
        let wasteTokens = 413.4 * countProg
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 376, width: 880, height: 420), radius: 26, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0xF85149, alpha: 0.65 * CGFloat(enter)), lineWidth: 3.0)
        drawText("AVERAGE WASTED <THINK> TOKENS PER TASK", rect: CGRect(x: 112, y: 416, width: 760, height: 28), size: 19, weight: .bold, colorHex: 0xF85149, alpha: CGFloat(enter), mono: true)
        drawText(String(format: "%.1f tok", wasteTokens), rect: CGRect(x: 112, y: 472, width: 800, height: 110), size: 92, weight: .heavy, colorHex: 0xF85149, alpha: CGFloat(enter), mono: true)
        drawText("512-token generation cap burned before emitting the answer", rect: CGRect(x: 112, y: 610, width: 800, height: 36), size: 23, weight: .semibold, colorHex: 0xC9D1D9, alpha: CGFloat(enter))

        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 684, width: 800, height: 36), radius: 18, fill: cgColor(0x21262D, alpha: CGFloat(enter)))
        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 684, width: max(36, 800 * CGFloat(0.81 * countProg)), height: 36), radius: 18, fill: cgColor(0xF85149, alpha: CGFloat(enter)))
        drawText("80.7% OF TOKEN BUDGET WASTED IN <think>...</think>", rect: CGRect(x: 112, y: 736, width: 800, height: 28), size: 18, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(enter), mono: true, align: .center)

        // Failing Terminal Card
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 836, width: 880, height: 330), radius: 22, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 2.5)
        drawText("AA BENCHMARK SUITE (RAW INT4 BASELINE)", rect: CGRect(x: 112, y: 868, width: 700, height: 28), size: 19, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(enter), mono: true)
        let lines = [
            ("<think> Wait, let me re-check step 3 again...", 0x8B949E),
            ("Actually let's verify all edge cases once more...", 0x8B949E),
            ("Hold on, let me re-derive the denominator...", 0x8B949E),
            ("[HIT 512 MAX_NEW_TOKENS — 0 VISIBLE ANSWER]", 0xF85149),
        ]
        for (idx, item) in lines.enumerated() {
            let lFade = progressIn(lt, 0.5 + Double(idx) * 0.45, 0.25)
            drawText(item.0, rect: CGRect(x: 112, y: 924 + CGFloat(idx) * 52, width: 800, height: 36), size: 22, weight: .semibold, colorHex: UInt32(item.1), alpha: CGFloat(lFade * enter), mono: true)
        }
    } else if time < b[3] {
        // SCENE 3: THE MATH TRICK + 4 SURGERY KNOBS
        let lt = time - b[2]
        let enter = easeOutCubic(progressIn(lt, 0.0, 0.4))
        let knobs = easeOutCubic(progressIn(lt, 1.4, 0.65))

        drawText("The Lossless Algebra Trick", rect: CGRect(x: 72, y: 244, width: 880, height: 56), size: 46, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter))
        drawText("MLX group-64 affine INT4 weights scale linearly in BF16", rect: CGRect(x: 72, y: 306, width: 880, height: 34), size: 21, weight: .medium, colorHex: 0x8B949E, alpha: CGFloat(enter))

        // Equation Card
        drawRoundedRect(ctx, rect: CGRect(x: 72, y: 374, width: 880, height: 380), radius: 24, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x388BFD, alpha: 0.65 * CGFloat(enter)), lineWidth: 2.5)
        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 412, width: 800, height: 86), radius: 14, fill: cgColor(0x0D1117, alpha: CGFloat(enter)), stroke: cgColor(0x30363D, alpha: CGFloat(enter)), lineWidth: 1.5)
        drawText("w  =  s · q  +  b", rect: CGRect(x: 132, y: 432, width: 420, height: 46), size: 34, weight: .bold, colorHex: 0xC9D1D9, alpha: CGFloat(enter), mono: true)
        drawSingleLineBadge(ctx, text: "INT4 q LOCKED", x: 644, y: 432, width: 244, height: 46, bgHex: 0x1F6FEB, bgAlpha: 0.25, borderHex: 0x58A6FF, textHex: 0x58A6FF, fontSize: 18, mono: true, alpha: CGFloat(enter))

        drawText("↓  Multiply any projection or norm by α", rect: CGRect(x: 112, y: 522, width: 800, height: 38), size: 24, weight: .bold, colorHex: 0x3FB950, alpha: CGFloat(enter), align: .center)

        drawRoundedRect(ctx, rect: CGRect(x: 112, y: 582, width: 800, height: 132), radius: 16, fill: cgColor(0x0D1117, alpha: CGFloat(enter)), stroke: cgColor(0x3FB950, alpha: 0.65 * CGFloat(enter)), lineWidth: 2.0)
        drawText("α · w  =  (α · s) · q  +  (α · b)", rect: CGRect(x: 124, y: 604, width: 776, height: 48), size: 36, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter), mono: true, align: .center)
        drawText("Only BF16 scale (s), bias (b), & norms edited in model_weights.bin", rect: CGRect(x: 124, y: 668, width: 776, height: 28), size: 19, weight: .semibold, colorHex: 0x3FB950, alpha: CGFloat(enter), mono: true, align: .center)

        // 4 Verified Surgery Knobs (2x2 Grid)
        let knobCards: [(String, String, String, UInt32)] = [
            ("LATE Q/K NORMS", "1.03×", "Sharpens late-layer attention", 0x58A6FF),
            ("ROUTER TEMP", "0.96", "Clean MoE expert selection", 0xD2A8FF),
            ("SHARED EXPERT", "1.025×", "Final Norm 1.02× · Residual 0.98×", 0x3FB950),
            ("MTP CO-ADAPT", "ON", "Aligned 19-tensor draft weights", 0xF0F6FC),
        ]
        for (i, item) in knobCards.enumerated() {
            let col = i % 2
            let row = i / 2
            let kx: CGFloat = col == 0 ? 72 : 524
            let ky: CGFloat = 796 + CGFloat(row) * 260
            drawRoundedRect(ctx, rect: CGRect(x: kx, y: ky, width: 428, height: 230), radius: 22, fill: cgColor(0x161B22, alpha: CGFloat(knobs)), stroke: cgColor(0x30363D, alpha: CGFloat(knobs)), lineWidth: 2.5)
            drawText(item.0, rect: CGRect(x: kx + 28, y: ky + 28, width: 372, height: 28), size: 19, weight: .bold, colorHex: 0x8B949E, alpha: CGFloat(knobs), mono: true)
            drawText(item.1, rect: CGRect(x: kx + 28, y: ky + 72, width: 372, height: 74), size: 62, weight: .heavy, colorHex: UInt32(item.3), alpha: CGFloat(knobs), mono: true)
            drawText(item.2, rect: CGRect(x: kx + 28, y: ky + 168, width: 372, height: 36), size: 19, weight: .medium, colorHex: 0xC9D1D9, alpha: CGFloat(knobs))
        }
    } else if time < b[4] {
        // SCENE 4: BEFORE VS AFTER RESULTS
        let lt = time - b[3]
        let c1 = easeOutCubic(progressIn(lt, 0.0, 0.5))
        let c2 = easeOutCubic(progressIn(lt, 0.7, 0.5))
        let c3 = easeOutCubic(progressIn(lt, 1.4, 0.5))

        drawText("Before vs. After Surgery", rect: CGRect(x: 72, y: 244, width: 880, height: 56), size: 46, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(c1))
        drawText("Reproducible 8-Task AA Suite · 16 GB Mac · Temp 0", rect: CGRect(x: 72, y: 308, width: 880, height: 34), size: 21, weight: .medium, colorHex: 0x8B949E, alpha: CGFloat(c1))

        let cards: [(String, String, String, String, UInt32, Double)] = [
            ("PASS@1 ACCURACY", "25.0%  ->  100.0%", "8 / 8 Passed  (+75.0 pts)", "ZERO INT4 nibbles modified", 0x3FB950, c1),
            ("THINK TOKEN WASTE", "413.4  ->  126.1", "-69.5% wasted reasoning", "-21.3% faster task wall time (25.31s)", 0x58A6FF, c2),
            ("PEAK DECODE (16 GB MAC)", "18.36 tok/s", "16.47 tok/s aggregate avg", "83.1% MTP Accept  (94.3% AIME peak)", 0xD2A8FF, c3),
        ]
        for (i, item) in cards.enumerated() {
            let cy = 382 + CGFloat(i) * 290
            let a = CGFloat(item.5)
            drawRoundedRect(ctx, rect: CGRect(x: 72, y: cy, width: 880, height: 256), radius: 24, fill: cgColor(0x161B22, alpha: a), stroke: cgColor(UInt32(item.4), alpha: 0.65 * a), lineWidth: 2.5)
            drawText(item.0, rect: CGRect(x: 112, y: cy + 30, width: 640, height: 28), size: 19, weight: .bold, colorHex: 0x8B949E, alpha: a, mono: true)
            drawText(item.1, rect: CGRect(x: 112, y: cy + 74, width: 800, height: 74), size: 58, weight: .heavy, colorHex: UInt32(item.4), alpha: a)
            drawText(item.2, rect: CGRect(x: 112, y: cy + 162, width: 800, height: 36), size: 26, weight: .bold, colorHex: 0xF0F6FC, alpha: a)
            drawText(item.3, rect: CGRect(x: 112, y: cy + 206, width: 800, height: 30), size: 20, weight: .medium, colorHex: 0xC9D1D9, alpha: a, mono: true)
        }
    } else {
        // SCENE 5: CTA
        let lt = time - b[4]
        let enter = easeOutCubic(progressIn(lt, 0.0, 0.45))

        drawMetallicEmblem(ctx, x: 422, y: 316, size: 180, alpha: CGFloat(enter))
        drawText("Mference 2.0", rect: CGRect(x: 72, y: 540, width: 880, height: 92), size: 78, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter), align: .center)
        drawText("Pure Swift 6.1 + Metal 3  ·  Zero Nibbles Lost", rect: CGRect(x: 72, y: 644, width: 880, height: 48), size: 30, weight: .semibold, colorHex: 0xC9D1D9, alpha: CGFloat(enter), align: .center)

        drawRoundedRect(ctx, rect: CGRect(x: 96, y: 742, width: 832, height: 340), radius: 26, fill: cgColor(0x161B22, alpha: CGFloat(enter)), stroke: cgColor(0x3FB950, alpha: 0.65 * CGFloat(enter)), lineWidth: 3.0)
        drawSingleLineBadge(ctx, text: "RELEASE v2.0.0  ·  PREBUILT MACOS APP", x: 186, y: 784, width: 652, height: 50, bgHex: 0x238636, bgAlpha: 0.25, borderHex: 0x3FB950, textHex: 0x3FB950, fontSize: 21, mono: true, alpha: CGFloat(enter))
        drawText("Surgery Math + Source Code\non GitHub & Hugging Face", rect: CGRect(x: 126, y: 866, width: 772, height: 96), size: 34, weight: .bold, colorHex: 0xF0F6FC, alpha: CGFloat(enter), align: .center, lineSpacing: 8)
        drawText("voodmany / Qwen3.6-35B-A3B-4bit-MTP-Surgery-GTurbo", rect: CGRect(x: 112, y: 996, width: 800, height: 40), size: 20, weight: .bold, colorHex: 0x58A6FF, alpha: CGFloat(enter), mono: true, align: .center)

        drawRoundedRect(ctx, rect: CGRect(x: 96, y: 1134, width: 832, height: 140), radius: 24, fill: cgColor(0x1F6FEB, alpha: 0.20 * CGFloat(enter)), stroke: cgColor(0x58A6FF, alpha: 0.75 * CGFloat(enter)), lineWidth: 2.5)
        drawText("GITHUB REPO  ·  LINK IN BIO", rect: CGRect(x: 126, y: 1158, width: 772, height: 30), size: 19, weight: .bold, colorHex: 0x58A6FF, alpha: CGFloat(enter), mono: true, align: .center, letterSpacing: 1.5)
        drawText("fjw4cfv2fb-afk / Mference-2.0", rect: CGRect(x: 126, y: 1200, width: 772, height: 48), size: 34, weight: .heavy, colorHex: 0xF0F6FC, alpha: CGFloat(enter), mono: true, align: .center)
    }
}

// MARK: - WAV Reader & Procedural Dark-Tech Synthesizer + Voiceover Ducking (44.1 kHz Stereo)

func loadMonoWAVSamples(path: String) -> [Double] {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)), data.count >= 12 else {
        return []
    }
    var offset = 12
    var bitsPerSample = 16
    var numChannels = 1
    var pcmBytes = Data()

    while offset + 8 <= data.count {
        let chunkID = String(bytes: data[offset..<(offset + 4)], encoding: .ascii) ?? ""
        let cSize = Int(
            UInt32(data[offset + 4]) |
            (UInt32(data[offset + 5]) << 8) |
            (UInt32(data[offset + 6]) << 16) |
            (UInt32(data[offset + 7]) << 24)
        )
        let payloadStart = offset + 8
        let payloadEnd = min(data.count, payloadStart + cSize)
        if chunkID == "fmt " && payloadEnd - payloadStart >= 16 {
            numChannels = Int(UInt16(data[payloadStart + 2]) | (UInt16(data[payloadStart + 3]) << 8))
            bitsPerSample = Int(UInt16(data[payloadStart + 14]) | (UInt16(data[payloadStart + 15]) << 8))
        } else if chunkID == "data" && payloadEnd > payloadStart {
            pcmBytes = data[payloadStart..<payloadEnd]
            break
        }
        offset = payloadStart + cSize + (cSize & 1)
    }

    guard bitsPerSample == 16, numChannels >= 1, pcmBytes.count >= 2 * numChannels else {
        return []
    }
    let frameCount = pcmBytes.count / (2 * numChannels)
    var samples = [Double](repeating: 0.0, count: frameCount)
    pcmBytes.withUnsafeBytes { raw in
        guard let ptr = raw.baseAddress?.assumingMemoryBound(to: Int16.self) else { return }
        for i in 0..<frameCount {
            samples[i] = Double(Int16(littleEndian: ptr[i * numChannels])) / 32768.0
        }
    }
    return samples
}

func generateAudioTrack(duration: Double, sceneCuts: [Double], voiceClips: [VoiceClip]) -> [Int16] {
    let totalSamples = Int(round(duration * Double(kSampleRate)))
    var out = [Int16](repeating: 0, count: totalSamples * kChannels)

    // Pre-mix normalized mono voiceover buffer and compute smooth sidechain ducking envelope
    var voiceBuf = [Double](repeating: 0.0, count: totalSamples)
    var rawEnv = [Double](repeating: 0.0, count: totalSamples)

    for clip in voiceClips {
        var samples = loadMonoWAVSamples(path: clip.wavPath)
        guard !samples.isEmpty else { continue }
        let peak = samples.reduce(0.0) { max($0, abs($1)) }
        if peak > 0.02 {
            let normGain = min(2.2, 0.88 / peak)
            for j in 0..<samples.count { samples[j] *= normGain }
        }

        let startSample = max(0, Int(round(clip.start * Double(kSampleRate))))
        let copyCount = min(samples.count, totalSamples - startSample)
        let fadeSamples = min(440, copyCount / 4) // 10ms de-click fade at clip ends
        for j in 0..<copyCount {
            var s = samples[j]
            if j < fadeSamples { s *= Double(j) / Double(fadeSamples) }
            if copyCount - j - 1 < fadeSamples { s *= Double(copyCount - j - 1) / Double(fadeSamples) }
            let idx = startSample + j
            voiceBuf[idx] += s
            rawEnv[idx] = 1.0
        }
    }

    // Smooth ducking envelope (~15ms attack, ~220ms release)
    var duckEnv = [Double](repeating: 0.0, count: totalSamples)
    var envState = 0.0
    for i in 0..<totalSamples {
        let target = rawEnv[i]
        let coeff = target > envState ? 0.004 : 0.00018
        envState += (target - envState) * coeff
        duckEnv[i] = envState
    }

    let bpm = 120.0
    let stepDuration = 60.0 / bpm / 4.0 // 16th note = 0.125s
    let rootsHz = [55.0, 43.65, 65.41, 49.0] // A1, F1, C2, G1
    let arpIntervals = [0.0, 7.0, 12.0, 15.0, 12.0, 7.0, 19.0, 15.0] // minor pentatonic/7th

    var lfsr: UInt32 = 0xACE1
    @inline(__always)
    func nextNoise() -> Double {
        let bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1
        lfsr = (lfsr >> 1) | (bit << 15)
        return (Double(lfsr & 0xFFFF) / 32767.5) - 1.0
    }

    for i in 0..<totalSamples {
        let t = Double(i) / Double(kSampleRate)
        let stepIdx = Int(floor(t / stepDuration))
        let stepT = t - Double(stepIdx) * stepDuration
        let beatStep = stepIdx % 4
        let barIdx = (stepIdx / 16) % rootsHz.count
        let root = rootsHz[barIdx]

        // 1. Punchy sub-kick on quarter beats
        var kick = 0.0
        let beatDuration = stepDuration * 4.0
        let beatIdx = Int(floor(t / beatDuration))
        let beatT = t - Double(beatIdx) * beatDuration
        if beatT < 0.22 {
            let fEnv = 44.0 + 95.0 * exp(-beatT * 38.0)
            let phase = 2.0 * .pi * (44.0 * beatT - (95.0 / 38.0) * (exp(-beatT * 38.0) - 1.0))
            let amp = exp(-beatT * 14.0)
            kick = sin(phase) * amp * 0.40 + sin(2.0 * .pi * fEnv * beatT) * 0.05 * amp
        }

        // 2. Clean minor bass groove on 8ths
        let eighthIdx = stepIdx / 2
        let eighthT = t - Double(eighthIdx) * (stepDuration * 2.0)
        let bassAmp = (stepIdx % 2 == 0 ? 0.22 : 0.14) * exp(-eighthT * 5.2)
        let bass = (sin(2.0 * .pi * root * t) + 0.35 * sin(2.0 * .pi * (root * 2.0) * t)) * bassAmp

        // 3. Soft stereo dark-tech arpeggio
        let semi = arpIntervals[stepIdx % arpIntervals.count]
        let arpFreq = root * 4.0 * pow(2.0, semi / 12.0)
        let arpEnv = exp(-stepT * 16.0) * (1.0 - exp(-stepT * 400.0))
        let panL = (stepIdx % 2 == 0) ? 0.85 : 0.55
        let panR = (stepIdx % 2 == 0) ? 0.55 : 0.85
        let arpCore = (sin(2.0 * .pi * arpFreq * t) + 0.25 * sin(2.0 * .pi * arpFreq * 2.0 * t)) * arpEnv * 0.14

        // 4. Crisp offbeat hi-hat
        var hat = 0.0
        if beatStep == 2 && stepT < 0.035 {
            hat = nextNoise() * exp(-stepT * 95.0) * 0.06
        }

        // 5. Scene cut transition chime
        var chime = 0.0
        for cut in sceneCuts {
            let dt = t - cut
            if dt >= 0.0 && dt < 0.45 {
                let cEnv = exp(-dt * 7.5) * (1.0 - exp(-dt * 220.0))
                chime += (sin(2.0 * .pi * 880.0 * dt) + 0.5 * sin(2.0 * .pi * 1320.0 * dt)) * cEnv * 0.12
            }
        }

        // Sidechain duck music bed cleanly underneath spoken voiceover (~15% while speaking, ~38% in gaps)
        let musicGain = voiceClips.isEmpty ? 0.85 : (0.38 - 0.23 * duckEnv[i])
        let musicFadeIn = clamp01(t / 0.15)
        let musicFadeOut = clamp01((duration - t) / 0.35)
        let mEnv = musicGain * musicFadeIn * musicFadeOut

        let v = voiceBuf[i]
        let mL = (kick + bass + arpCore * panL + hat) * mEnv + chime * 0.65 * musicFadeOut
        let mR = (kick + bass + arpCore * panR - hat * 0.8) * mEnv + chime * 0.65 * musicFadeOut

        let left = tanh((v + mL) * 0.92)
        let right = tanh((v + mR) * 0.92)

        out[i * 2] = Int16(max(-32767, min(32767, Int(left * 29000.0))))
        out[i * 2 + 1] = Int16(max(-32767, min(32767, Int(right * 29000.0))))
    }

    return out
}

// MARK: - PNG Snapshot & AVAssetWriter MP4 Export

func savePNG(_ image: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "MferenceReels", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create PNG destination \(url.path)"])
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "MferenceReels", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to finalize PNG \(url.path)"])
    }
}

func writeWAV(_ pcm: [Int16], sampleRate: Int, channels: Int, to url: URL) throws {
    try? FileManager.default.removeItem(at: url)
    let dataSize = UInt32(pcm.count * MemoryLayout<Int16>.stride)
    let chunkSize = UInt32(36) + dataSize
    let byteRate = UInt32(sampleRate * channels * MemoryLayout<Int16>.stride)
    let blockAlign = UInt16(channels * MemoryLayout<Int16>.stride)
    let bitsPerSample = UInt16(16)

    var header = Data()
    header.append(contentsOf: "RIFF".utf8)
    withUnsafeBytes(of: chunkSize.littleEndian) { header.append(contentsOf: $0) }
    header.append(contentsOf: "WAVE".utf8)
    header.append(contentsOf: "fmt ".utf8)
    withUnsafeBytes(of: UInt32(16).littleEndian) { header.append(contentsOf: $0) }
    withUnsafeBytes(of: UInt16(1).littleEndian) { header.append(contentsOf: $0) } // PCM
    withUnsafeBytes(of: UInt16(channels).littleEndian) { header.append(contentsOf: $0) }
    withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { header.append(contentsOf: $0) }
    withUnsafeBytes(of: byteRate.littleEndian) { header.append(contentsOf: $0) }
    withUnsafeBytes(of: blockAlign.littleEndian) { header.append(contentsOf: $0) }
    withUnsafeBytes(of: bitsPerSample.littleEndian) { header.append(contentsOf: $0) }
    header.append(contentsOf: "data".utf8)
    withUnsafeBytes(of: dataSize.littleEndian) { header.append(contentsOf: $0) }

    var fileData = Data(capacity: header.count + Int(dataSize))
    fileData.append(header)
    pcm.withUnsafeBytes { fileData.append(contentsOf: $0) }
    try fileData.write(to: url)
}

func renderVideoOnly(_ spec: ReelSpec, to videoURL: URL, previewDir: URL) throws {
    try? FileManager.default.removeItem(at: videoURL)

    let writer = try AVAssetWriter(outputURL: videoURL, fileType: .mp4)
    let videoSettings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: kWidth,
        AVVideoHeightKey: kHeight,
        AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: 10_000_000,
            AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            AVVideoExpectedSourceFrameRateKey: Int(kFPS),
            AVVideoMaxKeyFrameIntervalKey: 30,
        ],
    ]
    let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
    videoInput.expectsMediaDataInRealTime = false

    let pbAttrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: kWidth,
        kCVPixelBufferHeightKey as String: kHeight,
    ]
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: pbAttrs)
    guard writer.canAdd(videoInput) else {
        throw NSError(domain: "MferenceReels", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot add videoInput"])
    }
    writer.add(videoInput)

    let totalFrames = Int(round(spec.duration * Double(kFPS)))
    let previewFrameMap: [Int: String] = Dictionary(
        uniqueKeysWithValues: spec.previewTimes.map { (Int(round($0.time * Double(kFPS))), $0.label) }
    )

    guard writer.startWriting() else {
        throw writer.error ?? NSError(domain: "MferenceReels", code: 5, userInfo: [NSLocalizedDescriptionKey: "writer.startWriting failed"])
    }
    writer.startSession(atSourceTime: .zero)

    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    let queue = DispatchQueue(label: "mference.reels.video")
    let feedDone = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var frameIndex = 0
    nonisolated(unsafe) var renderError: Error?
    nonisolated(unsafe) var didSignal = false

    func finishFeeding(_ err: Error? = nil) {
        if let err { renderError = err }
        if !didSignal {
            didSignal = true
            videoInput.markAsFinished()
            feedDone.signal()
        }
    }

    videoInput.requestMediaDataWhenReady(on: queue) {
        while videoInput.isReadyForMoreMediaData {
            if writer.status != .writing {
                finishFeeding(writer.error ?? NSError(domain: "MferenceReels", code: 6, userInfo: [NSLocalizedDescriptionKey: "Writer stopped with status \(writer.status.rawValue)"]))
                return
            }
            if frameIndex >= totalFrames {
                finishFeeding(nil)
                return
            }

            let frame = frameIndex
            frameIndex += 1

            do {
                try autoreleasepool {
                    guard let pool = adaptor.pixelBufferPool else {
                        throw NSError(domain: "MferenceReels", code: 7, userInfo: [NSLocalizedDescriptionKey: "Missing pixelBufferPool"])
                    }
                    var maybePB: CVPixelBuffer?
                    let pbStatus = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &maybePB)
                    guard pbStatus == kCVReturnSuccess, let pb = maybePB else {
                        throw NSError(domain: "MferenceReels", code: 8, userInfo: [NSLocalizedDescriptionKey: "CVPixelBufferPoolCreatePixelBuffer failed: \(pbStatus)"])
                    }

                    let t = Double(frame) / Double(kFPS)
                    CVPixelBufferLockBaseAddress(pb, [])
                    if let ctx = CGContext(
                        data: CVPixelBufferGetBaseAddress(pb),
                        width: kWidth,
                        height: kHeight,
                        bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                        space: colorSpace,
                        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
                    ) {
                        ctx.saveGState()
                        ctx.translateBy(x: 0, y: CGFloat(kHeight))
                        ctx.scaleBy(x: 1, y: -1)
                        let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: true)
                        let prevCtx = NSGraphicsContext.current
                        NSGraphicsContext.current = nsCtx
                        spec.renderer(ctx, t, spec.duration, spec.boundaries)
                        NSGraphicsContext.current = prevCtx
                        ctx.restoreGState()

                        if let label = previewFrameMap[frame], let snap = ctx.makeImage() {
                            try savePNG(snap, to: previewDir.appendingPathComponent("\(label).png"))
                        }
                    }
                    CVPixelBufferUnlockBaseAddress(pb, [])

                    let vPts = CMTime(value: Int64(frame), timescale: kFPS)
                    guard adaptor.append(pb, withPresentationTime: vPts) else {
                        throw writer.error ?? NSError(domain: "MferenceReels", code: 9, userInfo: [NSLocalizedDescriptionKey: "adaptor.append failed at frame \(frame)"])
                    }
                }
            } catch {
                finishFeeding(error)
                return
            }

            if frame % 90 == 0 || frame == totalFrames - 1 {
                print("[reels]   \(spec.filename): frame \(frame + 1)/\(totalFrames)")
                fflush(stdout)
            }
        }
    }

    if feedDone.wait(timeout: .now() + 45.0) == .timedOut {
        writer.cancelWriting()
        throw NSError(domain: "MferenceReels", code: 10, userInfo: [NSLocalizedDescriptionKey: "Video encode timed out after 45s"])
    }
    if let renderError {
        writer.cancelWriting()
        throw renderError
    }

    let finishSema = DispatchSemaphore(value: 0)
    writer.finishWriting { finishSema.signal() }
    if finishSema.wait(timeout: .now() + 30.0) == .timedOut {
        throw NSError(domain: "MferenceReels", code: 11, userInfo: [NSLocalizedDescriptionKey: "writer.finishWriting timed out"])
    }
    if writer.status != .completed {
        throw writer.error ?? NSError(domain: "MferenceReels", code: 12, userInfo: [NSLocalizedDescriptionKey: "finishWriting status=\(writer.status.rawValue)"])
    }
}

func muxVideo(_ videoURL: URL, andAudio wavURL: URL, duration: Double, to finalURL: URL) throws {
    try? FileManager.default.removeItem(at: finalURL)

    let comp = AVMutableComposition()
    let videoAsset = AVURLAsset(url: videoURL)
    let audioAsset = AVURLAsset(url: wavURL)
    let targetDuration = CMTime(seconds: duration, preferredTimescale: 600)

    let loadSema = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var loadErr: Error?
    nonisolated(unsafe) var vTracks: [AVAssetTrack] = []
    nonisolated(unsafe) var aTracks: [AVAssetTrack] = []
    Task {
        do {
            vTracks = try await videoAsset.loadTracks(withMediaType: .video)
            aTracks = try await audioAsset.loadTracks(withMediaType: .audio)
        } catch {
            loadErr = error
        }
        loadSema.signal()
    }
    _ = loadSema.wait(timeout: .now() + 10.0)
    if let loadErr { throw loadErr }

    guard let srcVideo = vTracks.first, let srcAudio = aTracks.first else {
        throw NSError(domain: "MferenceReels", code: 13, userInfo: [NSLocalizedDescriptionKey: "Missing video/audio tracks for muxing"])
    }
    guard let compVideo = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
          let compAudio = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
        throw NSError(domain: "MferenceReels", code: 14, userInfo: [NSLocalizedDescriptionKey: "Failed to add composition tracks"])
    }
    try compVideo.insertTimeRange(CMTimeRange(start: .zero, duration: targetDuration), of: srcVideo, at: .zero)
    try compAudio.insertTimeRange(CMTimeRange(start: .zero, duration: targetDuration), of: srcAudio, at: .zero)

    guard let exporter = AVAssetExportSession(asset: comp, presetName: AVAssetExportPresetHighestQuality) else {
        throw NSError(domain: "MferenceReels", code: 15, userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAssetExportSession"])
    }
    exporter.outputURL = finalURL
    exporter.outputFileType = .mp4
    exporter.shouldOptimizeForNetworkUse = true

    let exportSema = DispatchSemaphore(value: 0)
    exporter.exportAsynchronously { exportSema.signal() }
    if exportSema.wait(timeout: .now() + 30.0) == .timedOut {
        exporter.cancelExport()
        throw NSError(domain: "MferenceReels", code: 16, userInfo: [NSLocalizedDescriptionKey: "AVAssetExportSession timed out after 30s"])
    }
    if exporter.status != .completed {
        throw exporter.error ?? NSError(domain: "MferenceReels", code: 17, userInfo: [NSLocalizedDescriptionKey: "Export failed with status \(exporter.status.rawValue)"])
    }
}

func renderReel(_ spec: ReelSpec, outDir: URL, previewDir: URL) throws {
    let mp4URL = outDir.appendingPathComponent(spec.filename)
    let tmpVideoURL = outDir.appendingPathComponent(".\(spec.filename).video.mp4")
    let tmpWavURL = outDir.appendingPathComponent(".\(spec.filename).audio.wav")
    defer {
        try? FileManager.default.removeItem(at: tmpVideoURL)
        try? FileManager.default.removeItem(at: tmpWavURL)
    }

    try renderVideoOnly(spec, to: tmpVideoURL, previewDir: previewDir)
    let pcm = generateAudioTrack(duration: spec.duration, sceneCuts: spec.sceneCuts, voiceClips: spec.voiceClips)
    try writeWAV(pcm, sampleRate: kSampleRate, channels: kChannels, to: tmpWavURL)
    try muxVideo(tmpVideoURL, andAudio: tmpWavURL, duration: spec.duration, to: mp4URL)
}

// MARK: - Main Entry Point

struct VoiceManifestEntry {
    let duration: Double
    let boundaries: [Double]
    let clips: [VoiceClip]
}

func loadVoiceManifest(from url: URL) -> [String: VoiceManifestEntry] {
    guard let data = try? Data(contentsOf: url),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let reelsDict = obj["reels"] as? [String: [String: Any]] else {
        return [:]
    }
    var result: [String: VoiceManifestEntry] = [:]
    for (filename, info) in reelsDict {
        guard let dur = info["duration"] as? Double,
              let bounds = info["boundaries"] as? [Double],
              bounds.count >= 6 else { continue }
        var clips: [VoiceClip] = []
        if let rawClips = info["clips"] as? [[String: Any]] {
            for rc in rawClips {
                if let st = rc["start"] as? Double, let wav = rc["wav"] as? String {
                    clips.append(VoiceClip(start: st, wavPath: wav))
                }
            }
        }
        result[filename] = VoiceManifestEntry(duration: dur, boundaries: bounds, clips: clips)
    }
    return result
}

func makeSpec(
    filename: String,
    prefix: String,
    defaultBoundaries: [Double],
    manifest: [String: VoiceManifestEntry],
    renderer: @escaping (_ ctx: CGContext, _ time: Double, _ duration: Double, _ boundaries: [Double]) -> Void
) -> ReelSpec {
    let entry = manifest[filename]
    let boundaries = entry?.boundaries ?? defaultBoundaries
    let duration = entry?.duration ?? boundaries.last ?? 18.0
    let sceneCuts = Array(boundaries.dropFirst().dropLast())
    let previewTimes: [(label: String, time: Double)] = (0..<5).map { idx in
        let t = boundaries[idx] + (boundaries[idx + 1] - boundaries[idx]) * 0.65
        return ("\(prefix)_s\(idx + 1)", t)
    }
    return ReelSpec(
        filename: filename,
        duration: duration,
        boundaries: boundaries,
        sceneCuts: sceneCuts,
        voiceClips: entry?.clips ?? [],
        previewTimes: previewTimes,
        renderer: renderer
    )
}

func main() throws {
    _ = NSApplication.shared
    let args = CommandLine.arguments
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let outDir = args.count > 1
        ? URL(fileURLWithPath: args[1], isDirectory: true)
        : cwd.appendingPathComponent("scratch/reels", isDirectory: true)
    let previewDir = outDir.appendingPathComponent("previews", isDirectory: true)
    let manifestURL = outDir.appendingPathComponent("voiceovers/manifest.json")

    try FileManager.default.createDirectory(at: previewDir, withIntermediateDirectories: true)

    let manifest = loadVoiceManifest(from: manifestURL)
    if manifest.isEmpty {
        print("[reels] Note: no voiceovers/manifest.json found; using music-only default timings.")
    } else {
        print("[reels] Loaded synchronized voiceovers from \(manifestURL.path)")
    }

    let specs = [
        makeSpec(
            filename: "mference-reel-1-16gb-mac.mp4",
            prefix: "reel1",
            defaultBoundaries: [0.0, 2.5, 7.5, 12.5, 16.0, 18.5],
            manifest: manifest,
            renderer: renderReel1
        ),
        makeSpec(
            filename: "mference-reel-2-weight-surgery.mp4",
            prefix: "reel2",
            defaultBoundaries: [0.0, 2.5, 6.5, 11.5, 15.0, 17.0],
            manifest: manifest,
            renderer: renderReel2
        ),
    ]

    for spec in specs {
        let start = Date()
        print("[reels] Rendering \(spec.filename) (\(spec.duration)s, \(spec.voiceClips.count) voice clips @ \(kWidth)x\(kHeight) \(kFPS)fps)...")
        try renderReel(spec, outDir: outDir, previewDir: previewDir)
        let elapsed = Date().timeIntervalSince(start)
        let size = (try? FileManager.default.attributesOfItem(atPath: outDir.appendingPathComponent(spec.filename).path)[.size] as? Int64) ?? 0
        print(String(format: "[reels] Done: %@ (%.2f MB in %.2fs)", spec.filename, Double(size) / 1_000_000.0, elapsed))
    }
}

do {
    try main()
} catch {
    fputs("[reels] ERROR: \(error)\n", stderr)
    exit(1)
}
