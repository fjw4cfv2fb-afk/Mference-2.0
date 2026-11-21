import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import Synchronization

public struct Qwen36VisionPatchData: Sendable {
    public let gridT: Int
    public let gridH: Int
    public let gridW: Int
    public let patches: [Float]

    public var numPatches: Int { gridT * gridH * gridW }
    public var numVisualTokens: Int { (gridT * gridH * gridW) / 4 }
    public var llmGridH: Int { gridH / 2 }
    public var llmGridW: Int { gridW / 2 }
}

public struct Qwen36RegisteredImage: Sendable {
    public let imageID: Int
    public let sha256: String
    public let sourcePath: String?
    public let patchData: Qwen36VisionPatchData

    public var numVisualTokens: Int { patchData.numVisualTokens }
    public var llmGridH: Int { patchData.llmGridH }
    public var llmGridW: Int { patchData.llmGridW }
    public var marker: String {
        Qwen36VisionImageStore.makeMarker(
            imageID: imageID,
            sha256: sha256,
            gridH: patchData.gridH,
            gridW: patchData.gridW,
            numVisualTokens: patchData.numVisualTokens)
    }
}

public struct Qwen36RegisteredPDF: Sendable {
    public let pages: [Qwen36RegisteredImage]
    public let totalPageCount: Int

    public var wasPageTruncated: Bool { pages.count < totalPageCount }
}

public struct Qwen36ParsedVisionMarker: Sendable, Equatable {
    public let imageID: Int
    public let sha256: String
    public let gridH: Int
    public let gridW: Int
    public let numVisualTokens: Int
}

public enum Qwen36VisionImageError: Error, CustomStringConvertible {
    case invalidImageSource(String)
    case unreadableImage(String)
    case unsupportedAspectRatio(width: Int, height: Int)
    case missingRegisteredImage(imageID: Int, sha256: String?)
    case visionSidecarMissing(String)
    case visionWeightsCorrupt(String)

    public var description: String {
        switch self {
        case .invalidImageSource(let detail):
            return "invalid image source: \(detail)"
        case .unreadableImage(let detail):
            return "unreadable image: \(detail)"
        case .unsupportedAspectRatio(let w, let h):
            return "unsupported image aspect ratio (\(w)x\(h))"
        case .missingRegisteredImage(let id, let sha):
            return "vision image \(id) (\(sha ?? "unknown")) is not registered"
        case .visionSidecarMissing(let path):
            return "vision sidecar missing at \(path)"
        case .visionWeightsCorrupt(let detail):
            return "vision weights corrupt: \(detail)"
        }
    }
}

/// Thread-safe registry and disk-backed patch cache for Qwen 3.6 visual inputs.
///
/// Images are decoded via `ImageIO`/`CoreGraphics`, resized with Qwen3-VL
/// `smart_resize` (`factor = 32`, configurable `MFERENCE_VISION_MAX_PIXELS`),
/// normalized to `[-1, 1]`, and packed into `(bh, bw, ih, iw)` patch order with
/// `[t=2, ph=16, pw=16, c=3]` inner layout.
///
/// Registering identical image bytes is idempotent (returning the same
/// `imageID` and marker), and a bounded `.vpatch` cache in `Library/Caches`
/// allows `MferenceMac` and `MferenceDecodeService` to share preprocessed
/// patches across processes without changing `MferenceDecodeProtocol`.
public final class Qwen36VisionImageStore: @unchecked Sendable {
    public static let shared = Qwen36VisionImageStore()

    public static let patchSize = 16
    public static let temporalPatchSize = 2
    public static let spatialMergeSize = 2
    public static let resizeFactor = patchSize * spatialMergeSize // 32
    public static let minPixels = 56 * 56
    public static let defaultMaxPixels = 384 * 384 // 144 visual tokens = 576 patches
    public static let absoluteMaxPixels = 16_777_216
    public static let defaultMaxPDFPages = 8
    public static let absoluteMaxPDFPages = 64
    public static let maxImageID = 32_767

    public static let visionStartTokenID: Int32 = 248_053
    public static let visionEndTokenID: Int32 = 248_054
    public static let imagePadTokenID: Int32 = 248_056

    static let markerPrefix = "<|mference_image:"
    static let markerSuffix = "|>"

    private struct State {
        var nextID: Int = 1
        var byID: [Int: Qwen36RegisteredImage] = [:]
        var idBySha: [String: Int] = [:]
        var embeddingsBySha: [String: [Float16]] = [:]
        var shaOrder: [String] = []
    }

    private let state = Mutex(State())
    private static let maxCachedImages = 128

    public init() {}

    public static var configuredMaxPixels: Int {
        if let raw = ProcessInfo.processInfo.environment["MFERENCE_VISION_MAX_PIXELS"],
           let parsed = Int(raw),
           parsed >= resizeFactor * resizeFactor {
            return min(parsed, absoluteMaxPixels)
        }
        return defaultMaxPixels
    }

    public static var configuredMaxPDFPages: Int {
        if let raw = ProcessInfo.processInfo.environment["MFERENCE_VISION_MAX_PDF_PAGES"],
           let parsed = Int(raw),
           parsed >= 1 {
            return min(parsed, absoluteMaxPDFPages)
        }
        return defaultMaxPDFPages
    }

    public static func isPDFData(_ data: Data) -> Bool {
        let prefixCount = min(data.count, 1024)
        guard prefixCount >= 5 else { return false }
        let header = Data([0x25, 0x50, 0x44, 0x46, 0x2D]) // "%PDF-"
        return data.prefix(prefixCount).range(of: header) != nil
    }

    public static func formatMarkers(for pages: [Qwen36RegisteredImage]) -> String {
        guard !pages.isEmpty else { return "" }
        if pages.count == 1 {
            return pages[0].marker
        }
        return pages.enumerated().map { "[Page \($0 + 1)]\n\($1.marker)" }.joined(separator: "\n")
    }

    // MARK: - Marker & Token ID Encoding

    public static func makeMarker(
        imageID: Int,
        sha256: String,
        gridH: Int,
        gridW: Int,
        numVisualTokens: Int
    ) -> String {
        "\(markerPrefix)id=\(imageID):sha=\(sha256):h=\(gridH):w=\(gridW):t=\(numVisualTokens)\(markerSuffix)"
    }

    public static func makeMarker(for image: Qwen36RegisteredImage) -> String {
        image.marker
    }

    public static func parseMarker(_ marker: String) -> Qwen36ParsedVisionMarker? {
        guard marker.hasPrefix(markerPrefix), marker.hasSuffix(markerSuffix) else {
            return nil
        }
        let inner = marker.dropFirst(markerPrefix.count).dropLast(markerSuffix.count)
        var imageID: Int?
        var sha: String?
        var gridH: Int?
        var gridW: Int?
        var tokens: Int?
        for part in inner.split(separator: ":") {
            guard let eq = part.firstIndex(of: "=") else { continue }
            let key = part[..<eq]
            let val = part[part.index(after: eq)...]
            switch key {
            case "id": imageID = Int(val)
            case "sha": sha = String(val)
            case "h": gridH = Int(val)
            case "w": gridW = Int(val)
            case "t": tokens = Int(val)
            default: break
            }
        }
        guard let imageID, let sha, !sha.isEmpty,
              let gridH, let gridW, let tokens,
              imageID > 0, gridH > 0, gridW > 0, tokens > 0 else {
            return nil
        }
        return Qwen36ParsedVisionMarker(
            imageID: imageID,
            sha256: sha,
            gridH: gridH,
            gridW: gridW,
            numVisualTokens: tokens)
    }

    public static func containsMarker(in text: String) -> Bool {
        text.contains(markerPrefix)
    }

    /// Strips internal `<|mference_image:...|>` markers from user-visible strings if needed.
    public static func strippingMarkers(from text: String) -> String {
        var result = ""
        var index = text.startIndex
        while let start = text.range(of: markerPrefix, range: index..<text.endIndex) {
            result += text[index..<start.lowerBound]
            if let end = text.range(of: markerSuffix, range: start.lowerBound..<text.endIndex) {
                index = end.upperBound
            } else {
                index = start.lowerBound
                break
            }
        }
        result += text[index..<text.endIndex]
        return result
    }

    /// Splits `text` into alternating text segments and parsed image markers.
    public enum TextOrMarkerSegment: Sendable {
        case text(String)
        case marker(Qwen36ParsedVisionMarker)
    }

    public static func splitMarkers(in text: String) -> [TextOrMarkerSegment] {
        guard text.contains(markerPrefix) else {
            return text.isEmpty ? [] : [.text(text)]
        }
        var segments: [TextOrMarkerSegment] = []
        var cursor = text.startIndex
        while let start = text.range(of: markerPrefix, range: cursor..<text.endIndex) {
            if start.lowerBound > cursor {
                segments.append(.text(String(text[cursor..<start.lowerBound])))
            }
            guard let end = text.range(of: markerSuffix, range: start.lowerBound..<text.endIndex) else {
                segments.append(.text(String(text[start.lowerBound..<text.endIndex])))
                cursor = text.endIndex
                break
            }
            let rawMarker = String(text[start.lowerBound..<end.upperBound])
            if let parsed = parseMarker(rawMarker) {
                segments.append(.marker(parsed))
            } else {
                segments.append(.text(rawMarker))
            }
            cursor = end.upperBound
        }
        if cursor < text.endIndex {
            segments.append(.text(String(text[cursor..<text.endIndex])))
        }
        return segments
    }

    public static func makeVisualTokenID(imageID: Int, row: Int) -> Int32 {
        precondition(imageID >= 1 && imageID <= maxImageID, "imageID out of range")
        precondition(row >= 0 && row <= 0xFFFF, "visual row out of range")
        let packed = (UInt32(imageID) << 16) | UInt32(row)
        return -Int32(bitPattern: packed)
    }

    public static func decodeVisualTokenID(_ token: Int32) -> (imageID: Int, row: Int)? {
        guard token < 0, token != Int32.min else { return nil }
        let packed = UInt32(bitPattern: -token)
        let imageID = Int(packed >> 16)
        let row = Int(packed & 0xFFFF)
        guard imageID >= 1 && imageID <= maxImageID else { return nil }
        return (imageID, row)
    }

    // MARK: - Registration & Lookup

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func loadFileData(_ fileURL: URL) throws -> Data {
        do {
            return try Data(contentsOf: fileURL, options: [.mappedIfSafe])
        } catch {
            throw Qwen36VisionImageError.unreadableImage("\(fileURL.path): \(error.localizedDescription)")
        }
    }

    private static func loadData(fromURLOrDataURI rawURL: String) throws -> (data: Data, sourcePath: String?) {
        let trimmed = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("data:") {
            guard let comma = trimmed.firstIndex(of: ",") else {
                throw Qwen36VisionImageError.invalidImageSource("malformed data URI")
            }
            let header = trimmed[..<comma].lowercased()
            guard header.contains(";base64") else {
                throw Qwen36VisionImageError.invalidImageSource("only base64 data URIs are supported")
            }
            let payload = String(trimmed[trimmed.index(after: comma)...])
            guard let data = Data(base64Encoded: payload, options: [.ignoreUnknownCharacters]),
                  !data.isEmpty else {
                throw Qwen36VisionImageError.invalidImageSource("invalid base64 payload")
            }
            return (data, nil)
        }
        if trimmed.hasPrefix("file://"), let url = URL(string: trimmed), url.isFileURL {
            return (try loadFileData(url), url.path)
        }
        if !trimmed.contains("://"), FileManager.default.fileExists(atPath: trimmed) {
            let url = URL(fileURLWithPath: trimmed)
            return (try loadFileData(url), url.path)
        }
        throw Qwen36VisionImageError.invalidImageSource(
            "only base64 data: URIs and local file paths are supported")
    }

    public func registerImage(
        data: Data,
        sourcePath: String? = nil,
        maxPixels: Int = Qwen36VisionImageStore.configuredMaxPixels
    ) throws -> Qwen36RegisteredImage {
        let sha = Self.sha256Hex(data)
        let cacheKey = "\(sha)-\(maxPixels)"

        if let existing = lookupBySha(cacheKey) {
            return existing
        }
        if let diskCached = Self.readPatchCacheFile(shaKey: cacheKey) {
            return insertRegistered(shaKey: cacheKey, sourcePath: sourcePath, patchData: diskCached)
        }

        let patchData = try Self.extractPatches(from: data, maxPixels: maxPixels)
        Self.writePatchCacheFile(shaKey: cacheKey, patchData: patchData)
        return insertRegistered(shaKey: cacheKey, sourcePath: sourcePath, patchData: patchData)
    }

    public func registerImage(
        fileURL: URL,
        maxPixels: Int = Qwen36VisionImageStore.configuredMaxPixels
    ) throws -> Qwen36RegisteredImage {
        let data = try Self.loadFileData(fileURL)
        return try registerImage(data: data, sourcePath: fileURL.path, maxPixels: maxPixels)
    }

    public func registerImage(
        fromURLOrDataURI rawURL: String,
        maxPixels: Int = Qwen36VisionImageStore.configuredMaxPixels
    ) throws -> Qwen36RegisteredImage {
        let loaded = try Self.loadData(fromURLOrDataURI: rawURL)
        return try registerImage(data: loaded.data, sourcePath: loaded.sourcePath, maxPixels: maxPixels)
    }

    public func registerPDF(
        data: Data,
        sourcePath: String? = nil,
        maxPixels: Int = Qwen36VisionImageStore.configuredMaxPixels,
        maxPages: Int = Qwen36VisionImageStore.configuredMaxPDFPages
    ) throws -> Qwen36RegisteredPDF {
        guard let provider = CGDataProvider(data: data as CFData),
              let pdf = CGPDFDocument(provider) else {
            throw Qwen36VisionImageError.unreadableImage("CoreGraphics failed to open PDF")
        }
        guard !pdf.isEncrypted || pdf.isUnlocked else {
            throw Qwen36VisionImageError.unreadableImage("password-protected PDFs are not supported")
        }
        let totalPages = pdf.numberOfPages
        guard totalPages > 0 else {
            throw Qwen36VisionImageError.unreadableImage("PDF contains no pages")
        }

        let docSha = Self.sha256Hex(data)
        let renderCount = min(totalPages, max(1, maxPages))
        var registeredPages: [Qwen36RegisteredImage] = []
        registeredPages.reserveCapacity(renderCount)

        for pageNumber in 1...renderCount {
            let cacheKey = "\(docSha)-p\(pageNumber)-\(maxPixels)"
            if let existing = lookupBySha(cacheKey) {
                registeredPages.append(existing)
                continue
            }
            if let diskCached = Self.readPatchCacheFile(shaKey: cacheKey) {
                registeredPages.append(insertRegistered(shaKey: cacheKey, sourcePath: sourcePath, patchData: diskCached))
                continue
            }
            guard let page = pdf.page(at: pageNumber) else {
                throw Qwen36VisionImageError.unreadableImage("failed to load PDF page \(pageNumber)")
            }
            let patchData = try Self.extractPDFPagePatches(page: page, maxPixels: maxPixels)
            Self.writePatchCacheFile(shaKey: cacheKey, patchData: patchData)
            registeredPages.append(insertRegistered(shaKey: cacheKey, sourcePath: sourcePath, patchData: patchData))
        }

        return Qwen36RegisteredPDF(pages: registeredPages, totalPageCount: totalPages)
    }

    public func registerPDF(
        fileURL: URL,
        maxPixels: Int = Qwen36VisionImageStore.configuredMaxPixels,
        maxPages: Int = Qwen36VisionImageStore.configuredMaxPDFPages
    ) throws -> Qwen36RegisteredPDF {
        let data = try Self.loadFileData(fileURL)
        return try registerPDF(data: data, sourcePath: fileURL.path, maxPixels: maxPixels, maxPages: maxPages)
    }

    public func registerVisualInput(
        data: Data,
        sourcePath: String? = nil,
        maxPixels: Int = Qwen36VisionImageStore.configuredMaxPixels,
        maxPDFPages: Int = Qwen36VisionImageStore.configuredMaxPDFPages
    ) throws -> [Qwen36RegisteredImage] {
        if Self.isPDFData(data) {
            return try registerPDF(data: data, sourcePath: sourcePath, maxPixels: maxPixels, maxPages: maxPDFPages).pages
        }
        return [try registerImage(data: data, sourcePath: sourcePath, maxPixels: maxPixels)]
    }

    public func registerVisualInput(
        fileURL: URL,
        maxPixels: Int = Qwen36VisionImageStore.configuredMaxPixels,
        maxPDFPages: Int = Qwen36VisionImageStore.configuredMaxPDFPages
    ) throws -> [Qwen36RegisteredImage] {
        let data = try Self.loadFileData(fileURL)
        if Self.isPDFData(data) || fileURL.pathExtension.lowercased() == "pdf" {
            return try registerPDF(data: data, sourcePath: fileURL.path, maxPixels: maxPixels, maxPages: maxPDFPages).pages
        }
        return [try registerImage(data: data, sourcePath: fileURL.path, maxPixels: maxPixels)]
    }

    public func registerVisualInput(
        fromURLOrDataURI rawURL: String,
        maxPixels: Int = Qwen36VisionImageStore.configuredMaxPixels,
        maxPDFPages: Int = Qwen36VisionImageStore.configuredMaxPDFPages
    ) throws -> [Qwen36RegisteredImage] {
        let loaded = try Self.loadData(fromURLOrDataURI: rawURL)
        return try registerVisualInput(data: loaded.data, sourcePath: loaded.sourcePath, maxPixels: maxPixels, maxPDFPages: maxPDFPages)
    }

    /// Resolves a parsed marker into a process-local `Qwen36RegisteredImage`,
    /// loading from the shared `.vpatch` cache on disk if registered in another
    /// process (e.g. `MferenceMac` -> `MferenceDecodeService`).
    public func resolve(marker: Qwen36ParsedVisionMarker) throws -> Qwen36RegisteredImage {
        if let existing = lookupBySha(marker.sha256) {
            return existing
        }
        if let byID = image(for: marker.imageID), byID.sha256 == marker.sha256 {
            return byID
        }
        if let patchData = Self.readPatchCacheFile(shaKey: marker.sha256) {
            return insertRegistered(shaKey: marker.sha256, sourcePath: nil, patchData: patchData)
        }
        throw Qwen36VisionImageError.missingRegisteredImage(
            imageID: marker.imageID,
            sha256: marker.sha256)
    }

    public func image(for imageID: Int) -> Qwen36RegisteredImage? {
        state.withLock { $0.byID[imageID] }
    }

    public func cachedEmbeddings(for imageID: Int) -> [Float16]? {
        state.withLock { st in
            guard let img = st.byID[imageID] else { return nil }
            return st.embeddingsBySha[img.sha256]
        }
    }

    public func storeEmbeddings(_ embeddings: [Float16], for imageID: Int) {
        state.withLock { st in
            guard let img = st.byID[imageID] else { return }
            st.embeddingsBySha[img.sha256] = embeddings
        }
    }

    private func lookupBySha(_ shaKey: String) -> Qwen36RegisteredImage? {
        state.withLock { st in
            guard let id = st.idBySha[shaKey] else { return nil }
            return st.byID[id]
        }
    }

    private func insertRegistered(
        shaKey: String,
        sourcePath: String?,
        patchData: Qwen36VisionPatchData
    ) -> Qwen36RegisteredImage {
        state.withLock { st in
            if let existingID = st.idBySha[shaKey], let existing = st.byID[existingID] {
                return existing
            }
            var assignedID = st.nextID
            while st.byID[assignedID] != nil {
                assignedID += 1
                if assignedID > Self.maxImageID { assignedID = 1 }
            }
            st.nextID = (assignedID >= Self.maxImageID) ? 1 : (assignedID + 1)

            let record = Qwen36RegisteredImage(
                imageID: assignedID,
                sha256: shaKey,
                sourcePath: sourcePath,
                patchData: patchData)
            st.byID[assignedID] = record
            st.idBySha[shaKey] = assignedID
            st.shaOrder.append(shaKey)

            while st.shaOrder.count > Self.maxCachedImages {
                let oldestSha = st.shaOrder.removeFirst()
                if let oldID = st.idBySha.removeValue(forKey: oldestSha) {
                    st.byID.removeValue(forKey: oldID)
                }
                st.embeddingsBySha.removeValue(forKey: oldestSha)
            }
            return record
        }
    }

    // MARK: - Smart Resize & Patch Extraction

    public static func smartResizeImage(
        height: Int,
        width: Int,
        factor: Int = resizeFactor,
        minPixels: Int = minPixels,
        maxPixels: Int = defaultMaxPixels
    ) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else {
            throw Qwen36VisionImageError.unreadableImage("image dimensions must be positive")
        }
        let maxSide = Double(max(height, width))
        let minSide = Double(min(height, width))
        guard maxSide / minSide <= 200.0 else {
            throw Qwen36VisionImageError.unsupportedAspectRatio(width: width, height: height)
        }

        let f = Double(factor)
        var hBar = max(factor, Int((Double(height) / f).rounded(.toNearestOrEven)) * factor)
        var wBar = max(factor, Int((Double(width) / f).rounded(.toNearestOrEven)) * factor)

        if hBar * wBar > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            hBar = max(factor, Int(floor(Double(height) / beta / f)) * factor)
            wBar = max(factor, Int(floor(Double(width) / beta / f)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            hBar = max(factor, Int(ceil(Double(height) * beta / f)) * factor)
            wBar = max(factor, Int(ceil(Double(width) * beta / f)) * factor)
        }
        return (hBar, wBar)
    }

    private static func extractPatches(from data: Data, maxPixels: Int) throws -> Qwen36VisionPatchData {
        let cfData = data as CFData
        guard let source = CGImageSourceCreateWithData(cfData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Qwen36VisionImageError.unreadableImage("CoreGraphics failed to decode image")
        }

        let orientation: Int = {
            guard let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                  let val = props[kCGImagePropertyOrientation] as? NSNumber else {
                return 1
            }
            return val.intValue
        }()
        let orientedImage: CGImage
        if orientation != 1,
           let transformed = CGImageSourceCreateThumbnailAtIndex(
               source,
               0,
               [
                   kCGImageSourceCreateThumbnailFromImageAlways: true,
                   kCGImageSourceCreateThumbnailWithTransform: true,
                   kCGImageSourceThumbnailMaxPixelSize: max(cgImage.width, cgImage.height),
               ] as CFDictionary
           ) {
            orientedImage = transformed
        } else {
            orientedImage = cgImage
        }

        let target = try smartResizeImage(
            height: orientedImage.height,
            width: orientedImage.width,
            factor: resizeFactor,
            minPixels: minPixels,
            maxPixels: maxPixels)

        return try renderBitmapAndPackPatches(targetW: target.width, targetH: target.height) { ctx, targetW, targetH in
            ctx.draw(orientedImage, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))
        }
    }

    private static func extractPDFPagePatches(page: CGPDFPage, maxPixels: Int) throws -> Qwen36VisionPatchData {
        var box = page.getBoxRect(.cropBox)
        if box.isEmpty || box.width <= 0 || box.height <= 0 {
            box = page.getBoxRect(.mediaBox)
        }
        guard !box.isEmpty, box.width > 0, box.height > 0 else {
            throw Qwen36VisionImageError.unreadableImage("PDF page has empty crop/media box")
        }

        let rotation = ((page.rotationAngle % 360) + 360) % 360
        let rawW = (rotation == 90 || rotation == 270) ? box.height : box.width
        let rawH = (rotation == 90 || rotation == 270) ? box.width : box.height
        let target = try smartResizeImage(
            height: max(1, Int(rawH.rounded())),
            width: max(1, Int(rawW.rounded())),
            factor: resizeFactor,
            minPixels: minPixels,
            maxPixels: maxPixels)

        return try renderBitmapAndPackPatches(targetW: target.width, targetH: target.height) { ctx, targetW, targetH in
            ctx.setShouldAntialias(true)
            ctx.setShouldSmoothFonts(true)
            ctx.saveGState()
            ctx.translateBy(x: CGFloat(targetW) * 0.5, y: CGFloat(targetH) * 0.5)
            ctx.scaleBy(x: CGFloat(targetW) / rawW, y: CGFloat(targetH) / rawH)
            ctx.rotate(by: -CGFloat(rotation) * .pi / 180.0)
            ctx.translateBy(x: -box.midX, y: -box.midY)
            ctx.drawPDFPage(page)
            ctx.restoreGState()
        }
    }

    private static func renderBitmapAndPackPatches(
        targetW: Int,
        targetH: Int,
        draw: (CGContext, Int, Int) -> Void
    ) throws -> Qwen36VisionPatchData {
        let bytesPerRow = targetW * 4
        var rgba = [UInt8](repeating: 255, count: targetH * bytesPerRow)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        let drawn: Bool = rgba.withUnsafeMutableBytes { rawBuf in
            guard let ctx = CGContext(
                data: rawBuf.baseAddress,
                width: targetW,
                height: targetH,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo) else {
                return false
            }
            ctx.setFillColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
            ctx.fill(CGRect(x: 0, y: 0, width: targetW, height: targetH))
            ctx.interpolationQuality = .high
            draw(ctx, targetW, targetH)
            return true
        }
        guard drawn else {
            throw Qwen36VisionImageError.unreadableImage("failed to render resized RGB bitmap")
        }
        return packRGBAPatches(rgba: rgba, width: targetW, height: targetH)
    }

    private static func packRGBAPatches(
        rgba: [UInt8],
        width targetW: Int,
        height targetH: Int
    ) -> Qwen36VisionPatchData {
        // Convert to normalized [-1, 1] float32 RGB image and extract patches in
        // `(bh, bw, ih, iw)` spatial-merge order with `[t=2, ph=16, pw=16, c=3]` inner layout.
        let gridH = targetH / patchSize
        let gridW = targetW / patchSize
        let llmGridH = gridH / spatialMergeSize
        let llmGridW = gridW / spatialMergeSize
        let numPatches = gridH * gridW
        let patchDim = temporalPatchSize * patchSize * patchSize * 3 // 1536
        let halfPatchDim = patchSize * patchSize * 3 // 768

        var patches = [Float](repeating: 0, count: numPatches * patchDim)
        let invScale: Float = 1.0 / 127.5
        rgba.withUnsafeBufferPointer { src in
            patches.withUnsafeMutableBufferPointer { dst in
                for bh in 0..<llmGridH {
                    for bw in 0..<llmGridW {
                        for ih in 0..<spatialMergeSize {
                            for iw in 0..<spatialMergeSize {
                                let patchIdx = (((bh * llmGridW + bw) * spatialMergeSize + ih) * spatialMergeSize) + iw
                                let y0 = (bh * spatialMergeSize + ih) * patchSize
                                let x0 = (bw * spatialMergeSize + iw) * patchSize
                                let base0 = patchIdx * patchDim
                                let base1 = base0 + halfPatchDim
                                for py in 0..<patchSize {
                                    let rowPtr = src.baseAddress! + ((y0 + py) * targetW + x0) * 4
                                    let rowOutOff = py * patchSize * 3
                                    for px in 0..<patchSize {
                                        let r = Float(rowPtr[px * 4 + 0]) * invScale - 1.0
                                        let g = Float(rowPtr[px * 4 + 1]) * invScale - 1.0
                                        let b = Float(rowPtr[px * 4 + 2]) * invScale - 1.0
                                        let pixOff = rowOutOff + px * 3
                                        dst[base0 + pixOff + 0] = r
                                        dst[base0 + pixOff + 1] = g
                                        dst[base0 + pixOff + 2] = b
                                        // Temporal patch duplication for still images (t=1 == t=0).
                                        dst[base1 + pixOff + 0] = r
                                        dst[base1 + pixOff + 1] = g
                                        dst[base1 + pixOff + 2] = b
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return Qwen36VisionPatchData(gridT: 1, gridH: gridH, gridW: gridW, patches: patches)
    }

    // MARK: - Cross-Process Disk Patch Cache

    private static var patchCacheDirectory: URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("Mference", isDirectory: true)
            .appendingPathComponent("vision-patches", isDirectory: true)
    }

    private static func patchCacheURL(shaKey: String) -> URL {
        let safe = shaKey.map { ch -> Character in
            (ch.isASCII && (ch.isLetter || ch.isNumber || ch == "-")) ? ch : "_"
        }
        return patchCacheDirectory.appendingPathComponent(String(safe) + ".vpatch")
    }

    private static func writePatchCacheFile(shaKey: String, patchData: Qwen36VisionPatchData) {
        let dir = patchCacheDirectory
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = patchCacheURL(shaKey: shaKey)
        if FileManager.default.fileExists(atPath: url.path) { return }

        var data = Data()
        data.reserveCapacity(20 + patchData.patches.count * 4)
        data.append(contentsOf: [0x4D, 0x46, 0x56, 0x50]) // "MFVP"
        let header: [UInt32] = [
            UInt32(patchData.gridT).littleEndian,
            UInt32(patchData.gridH).littleEndian,
            UInt32(patchData.gridW).littleEndian,
            UInt32(patchData.patches.count).littleEndian,
        ]
        header.withUnsafeBytes { data.append(contentsOf: $0) }
        patchData.patches.withUnsafeBytes { data.append(contentsOf: $0) }
        try? data.write(to: url, options: .atomic)
        prunePatchCacheDirectory(dir: dir)
    }

    private static func readPatchCacheFile(shaKey: String) -> Qwen36VisionPatchData? {
        let url = patchCacheURL(shaKey: shaKey)
        guard let data = try? Data(contentsOf: url, options: [.mappedIfSafe]),
              data.count >= 20 else {
            return nil
        }
        guard data[0] == 0x4D && data[1] == 0x46 && data[2] == 0x56 && data[3] == 0x50 else {
            return nil
        }
        let ints: [UInt32] = data[4..<20].withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: UInt32.self)
            return [
                UInt32(littleEndian: ptr[0]),
                UInt32(littleEndian: ptr[1]),
                UInt32(littleEndian: ptr[2]),
                UInt32(littleEndian: ptr[3]),
            ]
        }
        let gridT = Int(ints[0])
        let gridH = Int(ints[1])
        let gridW = Int(ints[2])
        let floatCount = Int(ints[3])
        let expectedFloats = gridT * gridH * gridW * 1536
        guard gridT == 1, gridH > 0, gridW > 0,
              floatCount == expectedFloats,
              data.count == 20 + floatCount * 4 else {
            return nil
        }
        var patches = [Float](repeating: 0, count: floatCount)
        patches.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst.bindMemory(to: UInt8.self), from: 20..<data.count)
        }
        return Qwen36VisionPatchData(gridT: gridT, gridH: gridH, gridW: gridW, patches: patches)
    }

    private static func prunePatchCacheDirectory(dir: URL) {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]),
              files.count > maxCachedImages else {
            return
        }
        let sorted = files.sorted { a, b in
            let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return da < db
        }
        for old in sorted.prefix(files.count - maxCachedImages) {
            try? FileManager.default.removeItem(at: old)
        }
    }
}
