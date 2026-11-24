#!/usr/bin/env bash
# Build and package ~/Applications/Mference.app with MferenceMac,
# MferenceDecodeService, MferenceServer, Metal/SwiftPM resource bundles,
# Info.plist, and AppIcon.icns.
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_directory}/.." && pwd)"
build_path="${BUILD_PATH:-${repository_root}/.build}"
app_output="${HOME}/Applications/Mference.app"
skip_build=0
dry_run=0

note() { printf '[build-app] %s\n' "$*" >&2; }
fail() { note "error: $*"; exit 1; }

usage() {
  cat <<'USAGE'
usage: ./Scripts/build-app.sh [options]

Builds release binaries for MferenceMac, MferenceDecodeService, and
MferenceServer and packages them into a signed macOS .app bundle.

  --output <path.app>   Destination .app bundle (default: ~/Applications/Mference.app)
  --build-path <dir>    SwiftPM scratch build path (default: .build)
  --no-build            Skip swift build and reuse existing release binaries
  --dry-run             Print actions without building or writing files
  --help                Show this message
USAGE
}

require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 needs a value; see --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) require_value "$@"; app_output="$2"; shift 2 ;;
    --build-path) require_value "$@"; build_path="$2"; shift 2 ;;
    --no-build) skip_build=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown option $1 (try --help)" ;;
  esac
done

release_dir="${build_path}/release"
icon_png="${repository_root}/Sources/MferenceApp/Mac/Resources/mference-app-icon.png"

if [[ "$dry_run" -eq 1 ]]; then
  echo "would check: Apple Silicon (Darwin arm64)"
  if [[ "$skip_build" -eq 0 ]]; then
    echo "would build: swift build -c release --scratch-path $build_path --product MferenceServer"
    echo "would build: swift build -c release --scratch-path $build_path --product MferenceDecodeService"
    echo "would build: swift build -c release --scratch-path $build_path --product MferenceMac"
  fi
  echo "would write: $app_output/Contents/Info.plist and AppIcon.icns"
  echo "would copy:  MferenceMac, MferenceDecodeService, MferenceServer, and runtime .bundle resources into $app_output"
  echo "would sign:  codesign --force --deep --sign - $app_output"
  exit 0
fi

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || \
  fail "Mference.app requires an Apple Silicon (arm64) Mac."

if pgrep -fl '(^|/)(MferenceMac|MferenceDecodeService|MferenceServer)( |$)' >/dev/null 2>&1; then
  note "warning: a Mference process is currently running; quit Mference.app before replacing it if launch fails."
fi

cd "$repository_root"

if [[ "$skip_build" -eq 0 ]]; then
  for product in MferenceServer MferenceDecodeService MferenceMac; do
    note "Building release ${product}..."
    swift build -c release --scratch-path "$build_path" --product "$product"
  done
fi

for product in MferenceServer MferenceDecodeService MferenceMac; do
  [[ -x "${release_dir}/${product}" ]] || fail "missing release binary ${release_dir}/${product}"
done

contents_dir="${app_output}/Contents"
macos_dir="${contents_dir}/MacOS"
resources_dir="${contents_dir}/Resources"
mkdir -p "$macos_dir" "$resources_dir"

cat > "${contents_dir}/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>MferenceMac</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.mference.MferenceMac</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Mference</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
PLIST

for product in MferenceMac MferenceDecodeService MferenceServer; do
  cp -f "${release_dir}/${product}" "${macos_dir}/${product}"
  chmod +x "${macos_dir}/${product}"
done

for bundle in Mference_Mference.bundle Mference_MferenceAppCore.bundle Mference_MferenceMac.bundle swift-crypto_Crypto.bundle swift-nio_NIOPosix.bundle swift-transformers_Hub.bundle; do
  if [[ -d "${release_dir}/${bundle}" ]]; then
    rm -rf "${resources_dir:?}/${bundle}"
    ditto "${release_dir}/${bundle}" "${resources_dir}/${bundle}"
  fi
done

if [[ -f "$icon_png" ]]; then
  cp -f "$icon_png" "${resources_dir}/mference-app-icon.png"
  if command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
    tmp_icon_dir="$(mktemp -d)"
    iconset_dir="${tmp_icon_dir}/AppIcon.iconset"
    mkdir -p "$iconset_dir"
    for sz in 16 32 128 256 512; do
      sips -z "$sz" "$sz" "$icon_png" --out "${iconset_dir}/icon_${sz}x${sz}.png" >/dev/null
      dbl=$((sz * 2))
      sips -z "$dbl" "$dbl" "$icon_png" --out "${iconset_dir}/icon_${sz}x${sz}@2x.png" >/dev/null
    done
    iconutil -c icns "$iconset_dir" -o "${resources_dir}/AppIcon.icns"
    rm -rf "$tmp_icon_dir"
  fi
fi

note "Signing ${app_output}..."
codesign --force --deep --sign - "$app_output"
codesign --verify --deep --strict "$app_output"

note "Done! Installed signed app bundle at ${app_output}"
note "Open it with: open \"$app_output\""
