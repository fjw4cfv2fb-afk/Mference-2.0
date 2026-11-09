#!/usr/bin/env python3
"""Compare two Mference benchmark JSON runs side-by-side.

Usage:
    ./scripts/mference-benchmark-diff.py baseline.json patched.json
"""
import json
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print("Usage: mference-benchmark-diff.py <baseline.json> <patched.json>")
        sys.exit(1)

    base = load(sys.argv[1])
    patch = load(sys.argv[2])

    b_sum = base.get("summary", {})
    p_sum = patch.get("summary", {})

    b_avg = float(b_sum.get("average_tok_per_sec", 0))
    p_avg = float(p_sum.get("average_tok_per_sec", 0))
    delta_avg = p_avg - b_avg
    pct_avg = (delta_avg / b_avg * 100) if b_avg > 0 else 0

    print("=" * 72)
    print("  MFERENCE BENCHMARK COMPARISON REPORT")
    print("=" * 72)
    print(f"Baseline: {sys.argv[1]} ({base.get('model', 'unknown')})")
    print(f"Patched:  {sys.argv[2]} ({patch.get('model', 'unknown')})")
    print("-" * 72)
    print(f"{'Test Name':<28} | {'Baseline':<10} | {'Patched':<10} | {'Delta':<12}")
    print("-" * 72)

    base_tests = {t["name"]: t for t in base.get("tests", [])}
    patch_tests = {t["name"]: t for t in patch.get("tests", [])}

    for name, b_test in base_tests.items():
        p_test = patch_tests.get(name)
        b_speed = float(b_test.get("tok_per_sec", 0))
        if p_test:
            p_speed = float(p_test.get("tok_per_sec", 0))
            delta = p_speed - b_speed
            pct = (delta / b_speed * 100) if b_speed > 0 else 0
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f} ({sign}{pct:.1f}%)"
            print(f"{name:<28} | {b_speed:>7.2f} t/s | {p_speed:>7.2f} t/s | {delta_str:>12}")
        else:
            print(f"{name:<28} | {b_speed:>7.2f} t/s | {'MISSING':<10} | {'N/A':>12}")

    print("-" * 72)
    sign_avg = "+" if delta_avg >= 0 else ""
    summary_delta = f"{sign_avg}{delta_avg:.2f} ({sign_avg}{pct_avg:.1f}%)"
    print(f"{'OVERALL AVERAGE':<28} | {b_avg:>7.2f} t/s | {p_avg:>7.2f} t/s | {summary_delta:>12}")
    print("=" * 72)

    if delta_avg > 0.5:
        print("🟢 VERDICT: Performance improved! Patches are faster than baseline.")
    elif delta_avg < -0.5:
        print("🔴 VERDICT: Regression detected! Patches degraded tok/s.")
    else:
        print("🟡 VERDICT: Neutral. Performance is parity with baseline (within jitter).")


if __name__ == "__main__":
    main()
