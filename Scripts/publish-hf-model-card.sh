#!/usr/bin/env bash
# Creates a public Hugging Face repository linked to Qwen/Qwen3.6-35B-A3B and
# uploads docs/launch/HUGGINGFACE_MODEL_CARD.md plus the local surgeried
# resident weights & MTP expert sidecar from scratch/qwen36-mtp.gturbo.
#
# Usage:
#   HF_TOKEN=hf_... ./Scripts/publish-hf-model-card.sh <hf-username>/<repo-name>
# Example:
#   HF_TOKEN=hf_... ./Scripts/publish-hf-model-card.sh fjw4cfv2fb/Qwen3.6-35B-A3B-4bit-MTP-Surgery-GTurbo
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
gturbo_dir="${GTURBO_DIR:-${repo_root}/scratch/qwen36-mtp.gturbo}"
model_card="${repo_root}/docs/launch/HUGGINGFACE_MODEL_CARD.md"

if [[ $# -lt 1 ]]; then
  echo "Usage: HF_TOKEN=hf_... $0 <hf-username/repo-name> [--full-experts]" >&2
  exit 2
fi

hf_repo="$1"
upload_full="${2:-}"

if [[ -z "${HF_TOKEN:-}" && ! -f "${HOME}/.cache/huggingface/token" ]]; then
  echo "Error: set HF_TOKEN=<write-token> or run huggingface login first." >&2
  exit 1
fi
[[ -f "$model_card" ]] || { echo "Missing $model_card" >&2; exit 1; }
[[ -d "$gturbo_dir" ]] || { echo "Missing $gturbo_dir" >&2; exit 1; }

if command -v hf >/dev/null 2>&1; then
  hf_cmd=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  hf_cmd=(huggingface-cli)
elif command -v uvx >/dev/null 2>&1; then
  hf_cmd=(uvx --from huggingface_hub hf)
else
  echo "Error: install huggingface_hub or uv (brew install uv) first." >&2
  exit 1
fi

echo "[hf-publish] Creating/updating model repo https://huggingface.co/${hf_repo} ..."
"${hf_cmd[@]}" repo create "${hf_repo##*/}" --repo-type model --exist-ok 2>/dev/null || true

"${hf_cmd[@]}" upload "$hf_repo" "$model_card" README.md --repo-type model
"${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/manifest.json" manifest.json --repo-type model
"${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/verified-install.json" verified-install.json --repo-type model
"${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/model_weights.bin" model_weights.bin --repo-type model
"${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/packed_experts_mtp/layer_00.bin" packed_experts_mtp/layer_00.bin --repo-type model
"${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/tokenizer" tokenizer --repo-type model

if [[ "$upload_full" == "--full-experts" ]]; then
  echo "[hf-publish] Uploading full packed_experts/ pool..."
  "${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/packed_experts" packed_experts --repo-type model
fi

echo "[hf-publish] Done! View at: https://huggingface.co/${hf_repo}"
