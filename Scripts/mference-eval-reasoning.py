#!/usr/bin/env python3
"""Objective Reasoning and Accuracy Evaluator for Mference models.

Runs verifiable multi-step reasoning problems with exact ground-truth answers
using official chat formatting to measure baseline vs patched model intelligence.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

CLI_BIN = os.environ.get("CLI_BIN", "/Users/vladslugin/Documents/Mference/.build/arm64-apple-macosx/release/MferenceCLI")
MODEL_DIR = os.environ.get("MODEL_DIR", "/Users/vladslugin/Documents/Mference/scratch/qwen36.gturbo")
OUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "/Users/vladslugin/Documents/Mference/benchmarks/baseline-reasoning-accuracy.json"

PROBLEMS = [
    {
        "id": "train_speed_distance",
        "category": "Algebra / Word Problem",
        "prompt": "A train leaves Station A traveling at 60 mph. Two hours later, a second train leaves Station A on a parallel track traveling at 90 mph in the same direction. How many hours after the second train leaves will it overtake the first train? Provide concise step-by-step arithmetic and on the final line write: Final Answer: <number>",
        "target": "4",
        "validate": lambda ans: "4" in ans.split("Final Answer:")[-1] if "Final Answer:" in ans else ("t = 4" in ans or "4 hours" in ans or "= 4" in ans)
    },
    {
        "id": "race_logic_ranking",
        "category": "Deductive Logic",
        "prompt": "Five runners (A, B, C, D, E) finish a race with no ties. A finishes ahead of B. C finishes between A and B. D finishes after B. E finishes ahead of A. Who finished in second place? Provide concise reasoning and on the final line write: Final Answer: <letter>",
        "target": "A",
        "validate": lambda ans: "A" in ans.split("Final Answer:")[-1].upper() if "Final Answer:" in ans else ("second place is A" in ans or "2nd: A" in ans or "Second: A" in ans)
    },
    {
        "id": "combinatorics_permutations",
        "category": "Discrete Math",
        "prompt": "How many distinct positive 3-digit integers can be formed using only the digits 2, 4, 6, and 8, if each digit can be used at most once? Provide concise calculation and on the final line write: Final Answer: <number>",
        "target": "24",
        "validate": lambda ans: "24" in ans.split("Final Answer:")[-1] if "Final Answer:" in ans else "24" in ans
    },
    {
        "id": "recursion_trace",
        "category": "Code Logic / Recursion",
        "prompt": "Consider this recurrence relation:\nf(0) = 0\nf(1) = 1\nf(n) = f(n - 1) + 2 * f(n - 2) for n >= 2\nCalculate f(2), f(3), f(4), and f(5) concisely. On the final line write: Final Answer: <number>",
        "target": "11",
        "validate": lambda ans: "11" in ans.split("Final Answer:")[-1] if "Final Answer:" in ans else ("f(5) = 11" in ans or "= 11" in ans)
    },
    {
        "id": "cube_packing_geometry",
        "category": "Spatial / Geometry",
        "prompt": "A hollow cube has an inner edge length of 10 cm. How many solid 2 cm cubes can fit completely inside it? Provide concise calculation and on the final line write: Final Answer: <number>",
        "target": "125",
        "validate": lambda ans: "125" in ans.split("Final Answer:")[-1] if "Final Answer:" in ans else "125" in ans
    }
]


def run_problem(p):
    messages = [{"role": "user", "content": p["prompt"]}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(messages, f)
        msg_file = f.name

    try:
        cmd = [
            CLI_BIN,
            "--model", MODEL_DIR,
            "--messages-file", msg_file,
            "--max-new", "350",
            "--expert-cache-slots", "96",
            "--verify", "trusted-receipt"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw_out = res.stdout
        raw_err = res.stderr

        is_correct = p["validate"](raw_out)

        tok_s = 0.0
        m = re.search(r'tok/s=([0-9.]+)', raw_err)
        if m:
            tok_s = float(m.group(1))

        return {
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "target": p["target"],
            "output": raw_out.strip(),
            "is_correct": is_correct,
            "tok_per_sec": tok_s
        }
    finally:
        if os.path.exists(msg_file):
            os.remove(msg_file)


def main():
    print("=" * 64)
    print("  Mference Objective Reasoning & Intelligence Benchmark")
    print("=" * 64)
    print(f"Model:  {os.path.basename(MODEL_DIR)}")
    print(f"Binary: {CLI_BIN}")
    print(f"Mode:   Official Chat Template via messages-file")
    print(f"Tests:  {len(PROBLEMS)} verifiable reasoning tasks")
    print("=" * 64)

    results = []
    correct_count = 0

    for i, p in enumerate(PROBLEMS, 1):
        print(f"\n[{i}/{len(PROBLEMS)}] Evaluating: {p['category']} ({p['id']})...")
        r = run_problem(p)
        results.append(r)
        if r["is_correct"]:
            correct_count += 1
            print(f"  Result: 🟢 PASS (Expected: {p['target']}) | {r['tok_per_sec']} tok/s")
        else:
            print(f"  Result: 🔴 FAIL (Expected: {p['target']}) | {r['tok_per_sec']} tok/s")

    pct = (correct_count / len(PROBLEMS)) * 100
    print("\n" + "=" * 64)
    print("  REASONING ACCURACY SUMMARY")
    print("=" * 64)
    print(f"Score:    {correct_count} / {len(PROBLEMS)} ({pct:.1f}%)")
    print(f"Target:   Assess whether future patches preserve or increase this")
    print("=" * 64)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    out_data = {
        "model": os.path.basename(MODEL_DIR),
        "total_tests": len(PROBLEMS),
        "correct": correct_count,
        "accuracy_pct": pct,
        "results": results
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved evaluation report to: {OUT_FILE}")


if __name__ == "__main__":
    main()
