#!/bin/bash
set -euo pipefail

# Mference Objective Benchmark Suite
# Measures exact decode tok/s, prefill latency, and quality across standardized prompts.

DEFAULT_BIN="/Users/vladslugin/Documents/Mference/.build/out/Products/Release/MferenceCLI"
if [ ! -x "$DEFAULT_BIN" ]; then
    DEFAULT_BIN="/Users/vladslugin/Documents/Mference/.build/arm64-apple-macosx/release/MferenceCLI"
fi
CLI_BIN="${CLI_BIN:-$DEFAULT_BIN}"
MODEL_DIR="${MODEL_DIR:-/Users/vladslugin/Documents/Mference/scratch/qwen36.gturbo}"
RDADVISE_MODE="${RDADVISE_MODE:-off}"
BENCHMARK_DIR="/Users/vladslugin/Documents/Mference/benchmarks"
OUT_FILE="${1:-${BENCHMARK_DIR}/baseline-qwen36-$(date +%Y%m%d_%H%M%S).json}"

mkdir -p "$BENCHMARK_DIR"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

if [ ! -x "$CLI_BIN" ]; then
    echo "❌ Error: CLI binary not found at $CLI_BIN. Build it with 'swift build -c release --product MferenceCLI' first."
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found at $MODEL_DIR"
    exit 1
fi

declare -a TEST_NAMES=(
    "Math_Proof"
    "Swift_Systems_Code"
    "Logic_Deduction"
    "MoE_Architecture_Synthesis"
    "Conceptual_Analogy"
)

declare -a TEST_PROMPTS=(
    "Prove that the square root of 2 is irrational using a step-by-step contradiction proof."
    "Write an actor in Swift that manages a thread-safe LRU cache with an async retrieve/insert API and eviction policy."
    "Alice, Bob, and Charlie each wear a colored hat (red or blue). Alice sees Bob and Charlie. Bob sees only Charlie. Charlie sees no hats. Alice says: 'I do not know my hat color.' Bob hears Alice and says: 'I do not know my hat color.' Assuming all are perfect logicians, what is Charlie's hat color and why?"
    "Summarize the key trade-offs between dense Transformers and Mixture-of-Experts (MoE) in 3 concise bullet points, followed by a markdown table comparing memory footprint, FLOPs per token, and routing overhead."
    "Explain the concept of speculative decoding in LLMs using an analogy to a master chef and a sous-chef in a restaurant."
)

declare -a MAX_TOKENS=(
    150
    250
    180
    200
    150
)

echo "========================================================"
echo "  Mference Performance & Quality Benchmark"
echo "========================================================"
echo "Model:   $(basename "$MODEL_DIR")"
echo "Binary:  $CLI_BIN"
echo "Output:  $OUT_FILE"
echo "Date:    $(date)"
echo "========================================================"

TOTAL_TOKENS=0
TOTAL_DECODE_TIME=0
COUNT=0

for i in "${!TEST_NAMES[@]}"; do
    NAME="${TEST_NAMES[$i]}"
    PROMPT="${TEST_PROMPTS[$i]}"
    MAX_TOK="${MAX_TOKENS[$i]}"

    echo ""
    echo "[$((i + 1))/${#TEST_NAMES[@]}] Running: $NAME (max ${MAX_TOK} tokens)..."

    RAW_OUTPUT=$("$CLI_BIN" \
        --model "$MODEL_DIR" \
        --prompt "$PROMPT" \
        --max-new "$MAX_TOK" \
        --expert-cache-slots 96 \
        --rdadvise "$RDADVISE_MODE" \
        --verify trusted-receipt 2>&1)

    # Extract timing metrics from Mference footer
    # Format: [stop=maxTokens prefill=6tok/0.91s new=40tok decode=2.14s tok/s=18.649]
    FOOTER_LINE=$(echo "$RAW_OUTPUT" | grep -E '\[stop=' | tail -n 1 || echo "")
    
    if [ -n "$FOOTER_LINE" ]; then
        TOK_S=$(echo "$FOOTER_LINE" | sed -n 's/.*tok\/s=\([0-9.]*\).*/\1/p')
        PREFILL_INFO=$(echo "$FOOTER_LINE" | sed -n 's/.*prefill=\([0-9]*\)tok\/\([0-9.]*\)s.*/\1tok \2s/p')
        NEW_TOK=$(echo "$FOOTER_LINE" | sed -n 's/.*new=\([0-9]*\)tok.*/\1/p')
        DECODE_S=$(echo "$FOOTER_LINE" | sed -n 's/.*decode=\([0-9.]*\)s.*/\1/p')
    else
        TOK_S="0"
        PREFILL_INFO="N/A"
        NEW_TOK="0"
        DECODE_S="0"
    fi

    echo "  ⚡ Speed:         ${TOK_S} tok/s"
    echo "  ⏱️  Decode Time:   ${DECODE_S} s (${NEW_TOK} tokens)"
    echo "  🚀 Prefill:       ${PREFILL_INFO}"

    CLEAN_TEXT=$(echo "$RAW_OUTPUT" | sed '/^\[stop=/d' | head -n 30)

    # Accumulate
    if [ -n "$TOK_S" ] && [ "$TOK_S" != "0" ]; then
        COUNT=$((COUNT + 1))
        TOTAL_TOKENS=$(echo "$TOTAL_TOKENS + $NEW_TOK" | bc 2>/dev/null || echo "$TOTAL_TOKENS")
        TOTAL_DECODE_TIME=$(echo "$TOTAL_DECODE_TIME + $DECODE_S" | bc 2>/dev/null || echo "$TOTAL_DECODE_TIME")
    fi

    # Write per-test entry to temp file
    echo "$CLEAN_TEXT" > "$TMP_DIR/text_$i.txt"
    cat <<EOF > "$TMP_DIR/entry_$i.json"
{
  "name": "$NAME",
  "prompt": $(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$PROMPT"),
  "tok_per_sec": ${TOK_S:-0},
  "tokens": ${NEW_TOK:-0},
  "decode_seconds": ${DECODE_S:-0}
}
EOF
done

AVG_TOK_S="0"
if [ "$COUNT" -gt 0 ] && [ "$TOTAL_DECODE_TIME" != "0" ]; then
    AVG_TOK_S=$(echo "scale=2; $TOTAL_TOKENS / $TOTAL_DECODE_TIME" | bc 2>/dev/null || echo "0")
fi

# Assemble all entries safely in Python
python3 - <<PYEOF
import json
import os

model_name = "$(basename "$MODEL_DIR")"
tmp_dir = "$TMP_DIR"
out_file = "$OUT_FILE"
avg_speed = float("$AVG_TOK_S")
total_tokens = int("$TOTAL_TOKENS")
total_decode = float("$TOTAL_DECODE_TIME")

tests = []
for i in range(${#TEST_NAMES[@]}):
    meta_path = os.path.join(tmp_dir, f"entry_{i}.json")
    text_path = os.path.join(tmp_dir, f"text_{i}.txt")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(text_path, "r", encoding="utf-8", errors="replace") as f:
        meta["output_preview"] = f.read()
    tests.append(meta)

data = {
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "model": model_name,
    "tests": tests,
    "summary": {
        "average_tok_per_sec": avg_speed,
        "total_tokens": total_tokens,
        "total_decode_seconds": total_decode
    }
}

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

PYEOF

echo ""
echo "========================================================"
echo "  BENCHMARK SUMMARY"
echo "========================================================"
echo "Average Speed:   ${AVG_TOK_S} tok/s"
echo "Total Generated: ${TOTAL_TOKENS} tokens in ${TOTAL_DECODE_TIME}s"
echo "Saved to:        $OUT_FILE"
echo "========================================================"
