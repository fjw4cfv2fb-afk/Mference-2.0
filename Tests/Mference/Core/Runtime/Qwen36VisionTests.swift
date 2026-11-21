import CoreGraphics
import Foundation
import ImageIO
import PDFKit
import Testing
@testable import Mference

@Suite("Qwen 3.6 Vision preprocessing & markers")
struct Qwen36VisionTests {
    private static func makeSyntheticPNG(width: Int, height: Int, r: CGFloat, g: CGFloat, b: CGFloat) throws -> Data {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw Qwen36VisionImageError.unreadableImage("failed to create CGContext")
        }
        ctx.setFillColor(red: r, green: g, blue: b, alpha: 1.0)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = ctx.makeImage(),
              let out = CFDataCreateMutable(nil, 0),
              let dest = CGImageDestinationCreateWithData(out, "public.png" as CFString, 1, nil) else {
            throw Qwen36VisionImageError.unreadableImage("failed to encode PNG")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw Qwen36VisionImageError.unreadableImage("failed to finalize PNG")
        }
        return out as Data
    }

    @Test func smartResizeMaintainsFactorDivisibilityAndPixelBounds() throws {
        // Upscale tiny image to satisfy minPixels = 3136 (factor = 32)
        let small = try Qwen36VisionImageStore.smartResizeImage(height: 16, width: 16)
        #expect(small.height % 32 == 0)
        #expect(small.width % 32 == 0)
        #expect(small.height * small.width >= Qwen36VisionImageStore.minPixels)

        // Downscale large landscape image to satisfy defaultMaxPixels = 384 * 384
        let large = try Qwen36VisionImageStore.smartResizeImage(height: 1080, width: 1920)
        #expect(large.height % 32 == 0)
        #expect(large.width % 32 == 0)
        #expect(large.height * large.width <= Qwen36VisionImageStore.defaultMaxPixels)
        #expect(large.width > large.height)
    }

    @Test func visualTokenIDAndMarkerRoundTrip() throws {
        for (imageID, row) in [(1, 0), (1, 143), (42, 511), (32_767, 65_535)] {
            let encoded = Qwen36VisionImageStore.makeVisualTokenID(imageID: imageID, row: row)
            #expect(encoded < 0)
            let decoded = Qwen36VisionImageStore.decodeVisualTokenID(encoded)
            #expect(decoded?.imageID == imageID)
            #expect(decoded?.row == row)
        }
        #expect(Qwen36VisionImageStore.decodeVisualTokenID(0) == nil)
        #expect(Qwen36VisionImageStore.decodeVisualTokenID(248_056) == nil)
        #expect(Qwen36VisionImageStore.decodeVisualTokenID(Int32.min) == nil)

        let marker = Qwen36VisionImageStore.makeMarker(
            imageID: 7,
            sha256: "abc123-147456",
            gridH: 24,
            gridW: 24,
            numVisualTokens: 144)
        let input = "Before \(marker) After"
        #expect(Qwen36VisionImageStore.containsMarker(in: input))
        #expect(Qwen36VisionImageStore.strippingMarkers(from: input) == "Before  After")

        let segments = Qwen36VisionImageStore.splitMarkers(in: input)
        #expect(segments.count == 3)
        guard case .text("Before ") = segments[0],
              case .marker(let parsed) = segments[1],
              case .text(" After") = segments[2] else {
            Issue.record("unexpected segments layout: \(segments)")
            return
        }
        #expect(parsed.imageID == 7)
        #expect(parsed.sha256 == "abc123-147456")
        #expect(parsed.gridH == 24)
        #expect(parsed.gridW == 24)
        #expect(parsed.numVisualTokens == 144)
    }

    @Test func registerAndCrossProcessResolveProducesNormalizedPatches() throws {
        let storeA = Qwen36VisionImageStore()
        let storeB = Qwen36VisionImageStore()

        let png = try Self.makeSyntheticPNG(width: 64, height: 64, r: 1.0, g: 0.0, b: 0.0)
        let first = try storeA.registerImage(data: png, maxPixels: 64 * 64)
        let second = try storeA.registerImage(data: png, maxPixels: 64 * 64)
        #expect(first.imageID == second.imageID)
        #expect(first.marker == second.marker)
        #expect(first.patchData.gridT == 1)
        #expect(first.patchData.gridH == 4)
        #expect(first.patchData.gridW == 4)
        #expect(first.numVisualTokens == 4)
        #expect(first.llmGridH == 2)
        #expect(first.llmGridW == 2)
        #expect(first.patchData.patches.count == first.patchData.numPatches * 1536)

        // Bright red pixel: normalized to [+1, -1, -1]
        #expect(abs(first.patchData.patches[0] - 1.0) < 0.02)
        #expect(abs(first.patchData.patches[1] + 1.0) < 0.02)
        #expect(abs(first.patchData.patches[2] + 1.0) < 0.02)

        guard let parsed = Qwen36VisionImageStore.parseMarker(first.marker) else {
            Issue.record("failed to parse generated marker")
            return
        }
        let resolvedInOtherProcess = try storeB.resolve(marker: parsed)
        #expect(resolvedInOtherProcess.sha256 == first.sha256)
        #expect(resolvedInOtherProcess.numVisualTokens == first.numVisualTokens)
        #expect(resolvedInOtherProcess.patchData.patches == first.patchData.patches)
    }

    private static func makeSyntheticTwoPagePDF() throws -> Data {
        let outData = NSMutableData()
        var box1 = CGRect(x: 0, y: 0, width: 128, height: 128)
        guard let consumer = CGDataConsumer(data: outData as CFMutableData),
              let ctx = CGContext(consumer: consumer, mediaBox: &box1, nil) else {
            throw Qwen36VisionImageError.unreadableImage("failed to create PDF CGContext")
        }

        // Page 1 (128x128, rotation=0): red top-left (y: 64..128), blue bottom-right (x: 64..128, y: 0..64)
        ctx.beginPDFPage(nil)
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(box1)
        ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 64, width: 64, height: 64))
        ctx.setFillColor(red: 0, green: 0, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 64, y: 0, width: 64, height: 64))
        ctx.endPDFPage()

        // Page 2 (128x256 before rotation): green bottom half (y: 0..128), rotated +90 deg -> left half green, width > height
        let box2 = CGRect(x: 0, y: 0, width: 128, height: 256)
        let page2Info: [CFString: Any] = [
            kCGPDFContextMediaBox: withUnsafeBytes(of: box2) { Data($0) },
        ]
        ctx.beginPDFPage(page2Info as CFDictionary)
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(box2)
        ctx.setFillColor(red: 0, green: 1, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: 128, height: 128))
        ctx.endPDFPage()
        ctx.closePDF()

        guard let doc = PDFDocument(data: outData as Data), let p2 = doc.page(at: 1) else {
            throw Qwen36VisionImageError.unreadableImage("failed to open PDFDocument")
        }
        p2.rotation = 90
        guard let rotatedData = doc.dataRepresentation() else {
            throw Qwen36VisionImageError.unreadableImage("failed to serialize rotated PDF")
        }
        return rotatedData
    }

    @Test func registerPDFRasterizesPagesWithRotationPageLimitAndCrossProcessCache() throws {
        let storeA = Qwen36VisionImageStore()
        let storeB = Qwen36VisionImageStore()
        let pdfData = try Self.makeSyntheticTwoPagePDF()
        #expect(Qwen36VisionImageStore.isPDFData(pdfData))
        #expect(!Qwen36VisionImageStore.isPDFData(Data("hello".utf8)))

        let full = try storeA.registerPDF(data: pdfData, maxPixels: 256 * 256, maxPages: 8)
        #expect(full.totalPageCount == 2)
        #expect(full.pages.count == 2)
        #expect(!full.wasPageTruncated)

        // Page 1: top-left pixel must be red (+1, -1, -1); bottom-right pixel must be blue (-1, -1, +1).
        let p1 = full.pages[0].patchData
        #expect(abs(p1.patches[0] - 1.0) < 0.05)
        #expect(abs(p1.patches[1] + 1.0) < 0.05)
        #expect(abs(p1.patches[2] + 1.0) < 0.05)
        let lastPatchBase = (p1.numPatches - 1) * 1536 + (15 * 16 + 15) * 3
        #expect(abs(p1.patches[lastPatchBase + 0] + 1.0) < 0.05)
        #expect(abs(p1.patches[lastPatchBase + 1] + 1.0) < 0.05)
        #expect(abs(p1.patches[lastPatchBase + 2] - 1.0) < 0.05)

        // Page 2 rotated 90 deg CW: landscape (gridW > gridH), top-left green (-1, +1, -1), bottom-right white (+1, +1, +1).
        let p2 = full.pages[1].patchData
        #expect(p2.gridW > p2.gridH)
        #expect(abs(p2.patches[0] + 1.0) < 0.05)
        #expect(abs(p2.patches[1] - 1.0) < 0.05)
        #expect(abs(p2.patches[2] + 1.0) < 0.05)
        let p2LastBase = (p2.numPatches - 1) * 1536 + (15 * 16 + 15) * 3
        #expect(abs(p2.patches[p2LastBase + 0] - 1.0) < 0.05)
        #expect(abs(p2.patches[p2LastBase + 1] - 1.0) < 0.05)
        #expect(abs(p2.patches[p2LastBase + 2] - 1.0) < 0.05)

        // SHA-256 deduplication and maxPages truncation
        let truncated = try storeA.registerPDF(data: pdfData, maxPixels: 256 * 256, maxPages: 1)
        #expect(truncated.totalPageCount == 2)
        #expect(truncated.pages.count == 1)
        #expect(truncated.wasPageTruncated)
        #expect(truncated.pages[0].imageID == full.pages[0].imageID)

        // Cross-process .vpatch resolution
        for page in full.pages {
            let parsed = try #require(Qwen36VisionImageStore.parseMarker(page.marker))
            let resolved = try storeB.resolve(marker: parsed)
            #expect(resolved.sha256 == page.sha256)
            #expect(resolved.patchData.patches == page.patchData.patches)
        }
    }
}
