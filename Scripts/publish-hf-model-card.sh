#!/usr/bin/env bash
# Creates a public Hugging Face repository linked to Qwen/Qwen3.6-35B-A3B and
# uploads docs/launch/HUGGINGFACE_MODEL_CARD.md plus the local surgeried
# resident weights & MTP expert sidecar from scratch/qwen36-mtp.gturbo.
#
# Usage:
#   HF_TOKEN=hf_... ./Scripts/publish-hf-model-card.sh [<hf-username>/<repo-name>] [--card-only|--full-experts]
# Defaults to:
#   voodmany/Qwen3.6-35B-A3B-4bit-MTP-Surgery-GTurbo
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
gturbo_dir="${GTURBO_DIR:-${repo_root}/scratch/qwen36-mtp.gturbo}"
model_card="${repo_root}/docs/launch/HUGGINGFACE_MODEL_CARD.md"
default_repo="voodmany/Qwen3.6-35B-A3B-4bit-MTP-Surgery-GTurbo"

hf_repo="$default_repo"
upload_mode="default"
for arg in "$@"; do
  case "$arg" in
    --card-only)
      upload_mode="card-only"
      ;;
    --full-experts)
      upload_mode="full-experts"
      ;;
    -h|--help)
      echo "Usage: HF_TOKEN=hf_... $0 [<hf-username>/<repo-name>] [--card-only|--full-experts]" >&2
      exit 0
      ;;
    *)
      hf_repo="$arg"
      ;;
  esac
done

hf_token_file="${HF_HOME:-${HOME}/.cache/huggingface}/token"
if [[ -z "${HF_TOKEN:-}" && ! -f "$hf_token_file" && ! -f "${HOME}/.huggingface/token" ]]; then
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
  venv_dir="/tmp/mference-hf-venv"
  if [[ ! -x "${venv_dir}/bin/hf" && ! -x "${venv_dir}/bin/huggingface-cli" ]]; then
    echo "[hf-publish] Bootstrapping huggingface_hub CLI in ${venv_dir}..."
    python3 -m venv "$venv_dir"
    "${venv_dir}/bin/pip" install --quiet --upgrade pip huggingface_hub
  fi
  if [[ -x "${venv_dir}/bin/hf" ]]; then
    hf_cmd=("${venv_dir}/bin/hf")
  else
    hf_cmd=("${venv_dir}/bin/huggingface-cli")
  fi
fi

echo "[hf-publish] Creating/updating model repo https://huggingface.co/${hf_repo} ..."
"${hf_cmd[@]}" repo create "${hf_repo##*/}" --repo-type model --exist-ok 2>/dev/null || true

"${hf_cmd[@]}" upload "$hf_repo" "$model_card" README.md --repo-type model
"${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/manifest.json" manifest.json --repo-type model
"${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/verified-install.json" verified-install.json --repo-type model
"${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/tokenizer" tokenizer --repo-type model

if [[ "$upload_mode" != "card-only" ]]; then
  "${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/model_weights.bin" model_weights.bin --repo-type model
  "${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/packed_experts_mtp/layer_00.bin" packed_experts_mtp/layer_00.bin --repo-type model
fi

if [[ "$upload_mode" == "full-experts" ]]; then
  echo "[hf-publish] Uploading full packed_experts/ pool..."
  "${hf_cmd[@]}" upload "$hf_repo" "${gturbo_dir}/packed_experts" packed_experts --repo-type model
fi

echo "[hf-publish] Done! View at: https://huggingface.co/${hf_repo}"
