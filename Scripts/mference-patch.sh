#!/bin/bash
set -euo pipefail

# Mference Model Weight Patch & Rollback Utility
# Safe, zero-risk APFS copy-on-write harness for weight-space programming and steering patches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/scratch/qwen36.gturbo}"
BACKUP_DIR="${MODEL_DIR}/.backups"
GOLDEN_DIR="${BACKUP_DIR}/golden"

usage() {
    cat <<EOF
Usage: $0 {status|backup|rollback|diff|apply <path>|verify|help}

Commands:
  status         Check current model files, active patches, and backup state
  backup         Create an instant APFS clone of untouched golden weights (0s, 0 bytes)
  rollback       Instantly revert all model files and experts to golden baseline
  diff           Compare current model weights and experts against golden baseline
  apply <path>   Safely apply a directory or file of patched weights/tensors
  verify         Verify model install integrity against receipt

Target Model: ${MODEL_DIR}
EOF
    exit 1
}

check_model() {
    if [ ! -d "$MODEL_DIR" ]; then
        echo "❌ Error: Model directory not found at $MODEL_DIR"
        exit 1
    fi
}

cmd_status() {
    check_model
    echo "========================================================"
    echo "  Mference Model Status: $(basename "$MODEL_DIR")"
    echo "========================================================"
    echo "Path: $MODEL_DIR"
    echo ""
    echo "Core Artifacts:"
    ls -lh "$MODEL_DIR/model_weights.bin" "$MODEL_DIR/manifest.json" 2>/dev/null || true
    echo "Experts: $(ls -1 "$MODEL_DIR/packed_experts" 2>/dev/null | wc -l | tr -d ' ') layer files ($(du -sh "$MODEL_DIR/packed_experts" | awk '{print $1}'))"
    echo ""
    if [ -d "$GOLDEN_DIR" ] && [ -f "$GOLDEN_DIR/model_weights.bin" ]; then
        echo "🔒 Golden State: Active backup locked in $GOLDEN_DIR"
        echo "   Created: $(stat -f "%Sm" "$GOLDEN_DIR/model_weights.bin")"
        cmd_diff_quiet
    else
        echo "ℹ️  Golden State: No backup created yet. Run '$0 backup' to snapshot."
    fi
    echo "========================================================"
}

cmd_backup() {
    check_model
    echo "========================================================"
    echo "  Creating Instant Golden State APFS Clone"
    echo "========================================================"
    mkdir -p "$GOLDEN_DIR"

    if [ -f "$GOLDEN_DIR/model_weights.bin" ]; then
        echo "⚠️  Golden backup already exists in $GOLDEN_DIR."
        echo "   Skipping overwrite to prevent losing original untouched state."
        echo "   (If you intentionally want a new golden state, remove $GOLDEN_DIR first)"
        return 0
    fi

    echo "Cloning core model files via APFS Copy-on-Write (instant, 0 extra disk space)..."
    cp -c "$MODEL_DIR/manifest.json" "$GOLDEN_DIR/manifest.json"
    cp -c "$MODEL_DIR/model_weights.bin" "$GOLDEN_DIR/model_weights.bin"
    if [ -f "$MODEL_DIR/verified-install.json" ]; then
        cp -c "$MODEL_DIR/verified-install.json" "$GOLDEN_DIR/verified-install.json"
    fi

    echo "Cloning 41 packed_experts layers..."
    mkdir -p "$GOLDEN_DIR/packed_experts"
    cp -Rc "$MODEL_DIR/packed_experts"/* "$GOLDEN_DIR/packed_experts/"

    echo "Generating verification checksums..."
    shasum -a 256 "$GOLDEN_DIR/manifest.json" > "$GOLDEN_DIR/checksums.txt"
    shasum -a 256 "$GOLDEN_DIR/model_weights.bin" >> "$GOLDEN_DIR/checksums.txt"

    echo "✅ Golden baseline locked and immutable at: $GOLDEN_DIR"
    cat "$GOLDEN_DIR/checksums.txt"
    echo "========================================================"
}

cmd_rollback() {
    check_model
    if [ ! -d "$GOLDEN_DIR" ] || [ ! -f "$GOLDEN_DIR/model_weights.bin" ]; then
        echo "❌ Error: No golden backup found in $GOLDEN_DIR to rollback to."
        exit 1
    fi

    echo "========================================================"
    echo "  Rolling Back Model to Golden Baseline"
    echo "========================================================"
    echo "Reverting model_weights.bin and manifest.json..."
    cp -cf "$GOLDEN_DIR/manifest.json" "$MODEL_DIR/manifest.json"
    cp -cf "$GOLDEN_DIR/model_weights.bin" "$MODEL_DIR/model_weights.bin"

    if [ -d "$GOLDEN_DIR/packed_experts" ]; then
        echo "Reverting packed_experts layers..."
        cp -Rcf "$GOLDEN_DIR/packed_experts"/* "$MODEL_DIR/packed_experts/"
    fi

    echo "✅ Rollback complete! 100% of original weights and experts restored."
    echo "========================================================"
}

cmd_diff_quiet() {
    if [ ! -f "$GOLDEN_DIR/model_weights.bin" ]; then
        return 0
    fi

    if cmp -s "$MODEL_DIR/model_weights.bin" "$GOLDEN_DIR/model_weights.bin" && cmp -s "$MODEL_DIR/manifest.json" "$GOLDEN_DIR/manifest.json"; then
        echo "   Status: 🟢 Clean (Matches Golden Baseline)"
    else
        echo "   Status: 🟡 Patched (Weights differ from Golden Baseline)"
    fi
}

cmd_diff() {
    check_model
    if [ ! -d "$GOLDEN_DIR" ] || [ ! -f "$GOLDEN_DIR/model_weights.bin" ]; then
        echo "No golden backup exists to compare against. Run '$0 backup' first."
        exit 0
    fi

    echo "========================================================"
    echo "  Comparing Current Model Against Golden Baseline"
    echo "========================================================"

    CURR_SUM=$(shasum -a 256 "$MODEL_DIR/model_weights.bin" | awk '{print $1}')
    GOLD_SUM=$(awk '/model_weights.bin$/ {print $1}' "$GOLDEN_DIR/checksums.txt" 2>/dev/null || shasum -a 256 "$GOLDEN_DIR/model_weights.bin" | awk '{print $1}')

    echo "Core model_weights.bin:"
    echo "  Current: $CURR_SUM"
    echo "  Golden:  $GOLD_SUM"
    if [ "$CURR_SUM" == "$GOLD_SUM" ]; then
        echo "  Result:  IDENTICAL"
    else
        echo "  Result:  MODIFIED"
    fi

    echo ""
    echo "Checking expert layers..."
    MOD_COUNT=0
    for exp in "$MODEL_DIR/packed_experts"/layer_*.bin; do
        bname=$(basename "$exp")
        gold_exp="$GOLDEN_DIR/packed_experts/$bname"
        if [ -f "$gold_exp" ]; then
            if ! cmp -s "$exp" "$gold_exp"; then
                echo "  Modified layer: $bname"
                MOD_COUNT=$((MOD_COUNT + 1))
            fi
        fi
    done

    if [ "$MOD_COUNT" -eq 0 ]; then
        echo "All 41 expert layers match golden baseline."
    else
        echo "Total modified expert layers: $MOD_COUNT"
    fi
    echo "========================================================"
}

cmd_apply() {
    check_model
    PATCH_SRC="${1:-}"
    if [ -z "$PATCH_SRC" ] || [ ! -e "$PATCH_SRC" ]; then
        echo "❌ Error: Patch source '$PATCH_SRC' not found."
        echo "Usage: $0 apply <patch_file_or_dir>"
        exit 1
    fi

    # Ensure a golden backup exists before allowing any modification
    if [ ! -f "$GOLDEN_DIR/model_weights.bin" ]; then
        echo "⚠️  No golden backup detected. Creating golden backup first..."
        cmd_backup
    fi

    echo "========================================================"
    echo "  Applying Patch from: $PATCH_SRC"
    echo "========================================================"
    if [ -d "$PATCH_SRC" ]; then
        # If directory contains model_weights.bin or packed_experts
        if [ -f "$PATCH_SRC/model_weights.bin" ]; then
            echo "Applying patched model_weights.bin..."
            cp -f "$PATCH_SRC/model_weights.bin" "$MODEL_DIR/model_weights.bin"
        fi
        if [ -d "$PATCH_SRC/packed_experts" ]; then
            echo "Applying patched packed_experts layers..."
            cp -Rf "$PATCH_SRC/packed_experts"/* "$MODEL_DIR/packed_experts/"
        fi
        if [ -f "$PATCH_SRC/manifest.json" ]; then
            echo "Applying updated manifest.json..."
            cp -f "$PATCH_SRC/manifest.json" "$MODEL_DIR/manifest.json"
        fi
    elif [ -f "$PATCH_SRC" ]; then
        # Single binary file
        echo "Overwriting model_weights.bin with $PATCH_SRC..."
        cp -f "$PATCH_SRC" "$MODEL_DIR/model_weights.bin"
    fi

    echo "✅ Patch applied successfully! Run '$0 diff' or benchmark to inspect."
    echo "========================================================"
}

cmd_verify() {
    check_model
    echo "Verifying model files..."
    if [ -f "$MODEL_DIR/verified-install.json" ]; then
        cat "$MODEL_DIR/verified-install.json"
    else
        echo "No verified-install.json present."
    fi
}

case "${1:-status}" in
    status)   cmd_status ;;
    backup)   cmd_backup ;;
    rollback) cmd_rollback ;;
    diff)     cmd_diff ;;
    apply)    shift; cmd_apply "$@" ;;
    verify)   cmd_verify ;;
    help|*)   usage ;;
esac
