#!/usr/bin/env bash
# One-command installer for the Mference 2.0 Qwen 3.6 35B-A3B MTP + Weight Surgery model.
#
# Steps performed:
#   1. Stream & repack mlx-community/Qwen3.6-35B-A3B-4bit into scratch/qwen36-mtp.gturbo
#   2. Download & SHA-256 verify guru87/Qwen3.6-35B-A3B-MTP/mtp.safetensors
#   3. Attach the 19-tensor MTP draft sidecar via MferenceRepack --attach-mtp
#   4. Apply the verified lossless resident weight surgery profile via MferenceRepack --apply-surgery
#   5. Verify the resulting install receipt via MferenceRepack --verify-install
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_directory}/.." && pwd)"
build_path="${BUILD_PATH:-${repository_root}/.build}"
output_dir="${repository_root}/scratch/qwen36-mtp.gturbo"
shard_path="${repository_root}/scratch/guru87_qwen36_mtp.safetensors"

shard_url="https://huggingface.co/guru87/Qwen3.6-35B-A3B-MTP/resolve/main/mtp.safetensors"
shard_sha256="68b2cabe2ff84bf3ba0fde0af082795e175a7ab427cb78e3b53df96e1d4dc929"
model_process_pattern='(^|/)(MferenceServer|MferenceCLI|MferenceRepack|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm)( |$)'

resume=0
overwrite=0
dry_run=0

note() { printf '[install-qwen36-mtp] %s\n' "$*" >&2; }
fail() { note "error: $*"; exit 1; }

usage() {
  cat <<'USAGE'
usage: ./Scripts/install-qwen36-mtp.sh [options]

Installs the upgraded Mference 2.0 Qwen 3.6 35B-A3B checkpoint with the
guru87 MTP draft sidecar and lossless resident weight surgery applied.

  --output <dir>    Destination .gturbo directory (default: scratch/qwen36-mtp.gturbo)
  --shard <path>    Local path for mtp.safetensors (default: scratch/guru87_qwen36_mtp.safetensors)
  --build-path <d>  SwiftPM scratch build path (default: .build)
  --resume          Resume an interrupted base download or shard transfer
  --overwrite       Replace an existing install from scratch
  --dry-run         Print the steps without downloading or modifying files
  --help            Show this message
USAGE
}

require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 needs a value; see --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) require_value "$@"; output_dir="$2"; shift 2 ;;
    --shard) require_value "$@"; shard_path="$2"; shift 2 ;;
    --build-path) require_value "$@"; build_path="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown option $1 (try --help)" ;;
  esac
done

surgery_args=(
  --apply-surgery
  --input-gturbo "$output_dir"
  --surgery-final-norm 1.02
  --surgery-qk-norm 1.03
  --surgery-qk-start-ratio 0.5
  --surgery-router-temp 0.96
  --surgery-late-residual 0.98
  --surgery-late-start-ratio 0.75
  --surgery-shared-expert 1.025
  --surgery-shared-expert-start-ratio 0.5
)

if [[ "$dry_run" -eq 1 ]]; then
  base_extra=""
  [[ "$resume" -eq 1 ]] && base_extra="$base_extra --resume"
  [[ "$overwrite" -eq 1 ]] && base_extra="$base_extra --overwrite"
  echo "would check: Apple Silicon (Darwin arm64) and no running model process ($model_process_pattern)"
  echo "would run:   swift run -c release --scratch-path $build_path MferenceRepack --model qwen36 --output $output_dir$base_extra"
  echo "would fetch: $shard_url -> $shard_path (expected sha256: $shard_sha256)"
  echo "would run:   swift run -c release --scratch-path $build_path MferenceRepack --attach-mtp $shard_path --output $output_dir"
  echo "would run:   swift run -c release --scratch-path $build_path MferenceRepack ${surgery_args[*]}"
  echo "would run:   swift run -c release --scratch-path $build_path MferenceRepack --verify-install --input-gturbo $output_dir"
  exit 0
fi

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || \
  fail "Mference requires an Apple Silicon (arm64) Mac."

if pgrep -fl "$model_process_pattern" >/dev/null 2>&1; then
  pgrep -fl "$model_process_pattern" >&2 || true
  fail "stop existing model processes before installing or patching weights."
fi

# Prevent accidentally double-scaling weights if MTP + surgery were already applied.
if [[ "$overwrite" -eq 0 && -f "$output_dir/manifest.json" ]]; then
  already_mtp="$(python3 - "$output_dir/manifest.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        m = json.load(f)
    mtp = (m.get("sidecars") or {}).get("mtp") or {}
    print("1" if mtp.get("carried") is True else "0")
except Exception:
    print("0")
PY
)"
  if [[ "$already_mtp" == "1" ]]; then
    note "$output_dir already has the MTP sidecar attached (and weight surgery is multiplicative)."
    note "Skipping to avoid double-scaling weights. Pass --overwrite to reinstall cleanly from scratch."
    exit 0
  fi
fi

cd "$repository_root"
mkdir -p "$(dirname "$output_dir")" "$(dirname "$shard_path")"

repack_base_args=(--model qwen36 --output "$output_dir")
if [[ "$overwrite" -eq 1 ]]; then
  repack_base_args+=(--overwrite)
elif [[ "$resume" -eq 1 ]]; then
  repack_base_args+=(--resume)
fi

if [[ "$overwrite" -eq 1 || ! -f "$output_dir/verified-install.json" ]]; then
  note "1/5 Installing base Qwen 3.6 35B-A3B checkpoint into $output_dir..."
  swift run -c release --scratch-path "$build_path" MferenceRepack "${repack_base_args[@]}"
else
  note "1/5 Verified base install already present at $output_dir; verifying before MTP attach..."
  swift run -c release --scratch-path "$build_path" MferenceRepack --verify-install --input-gturbo "$output_dir"
fi

verify_shard_sha() {
  local target="$1"
  [[ -f "$target" ]] || return 1
  local got
  got="$(shasum -a 256 "$target" | awk '{print $1}')"
  [[ "$got" == "$shard_sha256" ]]
}

if verify_shard_sha "$shard_path"; then
  note "2/5 Verified existing MTP shard at $shard_path"
else
  note "2/5 Downloading guru87/Qwen3.6-35B-A3B-MTP shard (~1.69 GB)..."
  part_path="${shard_path}.part"
  curl -L --fail --retry 3 -C - -o "$part_path" "$shard_url"
  verify_shard_sha "$part_path" || {
    rm -f "$part_path"
    fail "SHA-256 mismatch on downloaded MTP shard (expected $shard_sha256)"
  }
  mv "$part_path" "$shard_path"
  note "Verified MTP shard SHA-256: $shard_sha256"
fi

note "3/5 Attaching 19-tensor MTP sidecar to $output_dir..."
swift run -c release --scratch-path "$build_path" MferenceRepack \
  --attach-mtp "$shard_path" --output "$output_dir"

note "4/5 Applying lossless resident weight surgery + MTP co-adaptation..."
swift run -c release --scratch-path "$build_path" MferenceRepack "${surgery_args[@]}"

note "5/5 Verifying final install receipt..."
swift run -c release --scratch-path "$build_path" MferenceRepack \
  --verify-install --input-gturbo "$output_dir"

note "Done! Upgraded model ready at $output_dir"
note "Start the WebUI with ./mference-ui.sh or launch MferenceServer --model $output_dir"
