import Foundation

public enum AppModelLocation {
    public static let storageKey = "Mference.modelDirectory"

    public static func remember(_ url: URL) {
        remember(url, userDefaults: .standard)
    }

    static func remember(_ url: URL, userDefaults: UserDefaults) {
        userDefaults.set(url.standardizedFileURL.path, forKey: storageKey)
    }

    static func defaultURL(
        descriptor: AppModelInstallDescriptor = .selected,
        userDefaults: UserDefaults = .standard
    ) -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        let documentsRepo = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/Mference", isDirectory: true)
        return resolve(
            explicitURL: preferredURL(userDefaults: userDefaults),
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:),
            installDirectoryName: descriptor.installDirectoryName,
            documentsRepoURL: documentsRepo)
    }

    private static func preferredURL(userDefaults: UserDefaults) -> URL? {
        guard let path = userDefaults.string(forKey: storageKey),
              !path.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private static func preferredModelDirectory(
        in container: URL,
        installDirectoryName: String,
        fileExists: (String) -> Bool
    ) -> URL {
        if installDirectoryName == "qwen36.gturbo" {
            let mtpURL = container.appendingPathComponent("qwen36-mtp.gturbo", isDirectory: true)
            let mtpManifest = mtpURL.appendingPathComponent("manifest.json").path
            if fileExists(mtpManifest) || fileExists(mtpURL.path) {
                return mtpURL.standardizedFileURL
            }
        }
        return container.appendingPathComponent(installDirectoryName, isDirectory: true)
            .standardizedFileURL
    }

    private static func existingModelDirectory(
        in container: URL,
        installDirectoryName: String,
        fileExists: (String) -> Bool
    ) -> URL? {
        let candidate = preferredModelDirectory(
            in: container,
            installDirectoryName: installDirectoryName,
            fileExists: fileExists)
        let manifest = candidate.appendingPathComponent("manifest.json").path
        return fileExists(manifest) ? candidate : nil
    }

    // The default mirrors `AppModelInstallDescriptor.default.installDirectoryName`;
    // callers with a selected descriptor (qwen36.gturbo, deepseekv4flash.gturbo)
    // always pass its directory name explicitly.
    static func resolve(explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool,
                        installDirectoryName: String = "gemma4.gturbo",
                        documentsRepoURL: URL? = nil) -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return preferredModelDirectory(
                in: root.appendingPathComponent("scratch", isDirectory: true),
                installDirectoryName: installDirectoryName,
                fileExists: fileExists)
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return preferredModelDirectory(
                in: root.appendingPathComponent("scratch", isDirectory: true),
                installDirectoryName: installDirectoryName,
                fileExists: fileExists)
        }
        let appSupportMference = applicationSupportURL
            .appendingPathComponent("Mference", isDirectory: true)
        if let existing = existingModelDirectory(
            in: appSupportMference,
            installDirectoryName: installDirectoryName,
            fileExists: fileExists) {
            return existing
        }
        if let documentsRepoURL,
           let existing = existingModelDirectory(
               in: documentsRepoURL.appendingPathComponent("scratch", isDirectory: true),
               installDirectoryName: installDirectoryName,
               fileExists: fileExists) {
            return existing
        }
        return preferredModelDirectory(
            in: appSupportMference,
            installDirectoryName: installDirectoryName,
            fileExists: fileExists)
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let appSources = candidate.appendingPathComponent(
                "Sources/MferenceApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
