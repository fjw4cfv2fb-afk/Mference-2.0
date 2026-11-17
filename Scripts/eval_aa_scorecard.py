#!/usr/bin/env python3
"""Artificial Analysis (AA) Methodology A/B Scorecard for Qwen 3.6-35B-A3B.

Compares Baseline (temporary APFS zero-copy clone with raw 1.0 weights, static
MTP K=2, no ServerIntentAligner, no loop guard, no <think> ceiling) against
Upgraded v2 (scratch/qwen36-mtp.gturbo + extended surgical weights & co-adapted
MTP block + phase-aware adaptive MTP + clause-aware soft-landing <think> +
RepetitionLoopDetector + ServerIntentAligner + QwenToolCallParser) across all
4 Artificial Analysis Intelligence Index categories (2 tasks per category = 8
tasks total) over MferenceServer HTTP SSE streaming (--max-new 512,
--temperature 0).
"""

import argparse
import dataclasses
import http.client
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_BIN = os.path.join(REPO_ROOT, ".build/release/MferenceServer")
REPACK_BIN = os.path.join(REPO_ROOT, ".build/release/MferenceRepack")
BASE_SOURCE_MODEL = os.path.join(REPO_ROOT, "scratch/qwen36.gturbo")
MTP_SAFETENSORS = os.path.join(REPO_ROOT, "scratch/guru87_qwen36_mtp.safetensors")
BASELINE_CLONE_MODEL = os.path.join(REPO_ROOT, "scratch/qwen36-base-ab.gturbo")
UPGRADED_MODEL = os.path.join(REPO_ROOT, "scratch/qwen36-mtp.gturbo")
DEFAULT_OUT_JSON = os.path.join(REPO_ROOT, "benchmarks/aa-scorecard-v2.json")

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
MAX_CONTEXT = 4096
EXPERT_CACHE_SLOTS = 32
COOLDOWN_SECONDS = 4.0


# ---------------------------------------------------------------------------
# Verifiers (Pass@1 programmatic assertions)
# ---------------------------------------------------------------------------

def _extract_code_block(text: str, lang: str) -> Optional[str]:
    pattern = rf"```{lang}\s*\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    generic = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
    if generic:
        return generic[-1].strip()
    return None


def grade_c1_uncovered_spans(resp: Dict[str, Any]) -> Tuple[bool, str]:
    text = (resp.get("content") or "").strip()
    if resp.get("tool_calls"):
        return False, "unexpected tool_call"
    if not text:
        return False, "no visible answer emitted"
    code = _extract_code_block(text, "python")
    if code is None:
        if "```" in text:
            return False, "code block truncated before closing fence"
        if "def uncovered_spans" in text:
            code = text
    if not code or "def uncovered_spans" not in code:
        return False, "missing uncovered_spans Python implementation"

    ns: Dict[str, Any] = {}
    try:
        exec(compile(code, "<c1>", "exec"), ns, ns)
    except Exception as exc:
        return False, f"compile/exec failed: {exc}"
    fn = ns.get("uncovered_spans")
    if not callable(fn):
        return False, "uncovered_spans is not callable"

    tests = [
        (((0, 10), []), [(0, 10)]),
        (((5, 5), [(0, 10)]), []),
        (((0, 10), [(5, 5), (8, 3), (-5, 0), (10, 15)]), [(0, 10)]),
        (((0, 10), [(-2, 3), (7, 12)]), [(3, 7)]),
        (((0, 10), [(2, 4), (4, 6)]), [(0, 2), (6, 10)]),
        (((0, 10), [(5, 9), (1, 6), (2, 3)]), [(0, 1), (9, 10)]),
        (((0, 10), [(-5, 20)]), []),
        (((-10, -2), [(-12, -8), (-5, -5), (-4, 0)]), [(-8, -4)]),
        (
            ((0, 1_000_000_000), [(100, 500_000_000), (250_000_000, 999_999_990)]),
            [(0, 100), (999_999_990, 1_000_000_000)],
        ),
    ]
    t0 = time.monotonic()
    try:
        for idx, (args, expected) in enumerate(tests):
            got = [tuple(x) for x in fn(args[0], list(args[1]))]
            if got != expected:
                return False, f"assertion {idx} failed: expected {expected}, got {got}"
    except Exception as exc:
        return False, f"runtime exception: {exc}"
    if time.monotonic() - t0 > 0.5:
        return False, "timed out (>0.5s; non O(n log n) implementation)"
    return True, f"passed all {len(tests)} sweep-line assertions"


def grade_c2_shortest_subarray(resp: Dict[str, Any]) -> Tuple[bool, str]:
    text = (resp.get("content") or "").strip()
    if resp.get("tool_calls"):
        return False, "unexpected tool_call"
    code = _extract_code_block(text, "swift")
    if code is None:
        if "```" in text:
            return False, "code block truncated before closing fence"
        if "func shortestSubarrayAtLeastK" in text:
            code = text
    if not code or "func shortestSubarrayAtLeastK" not in code:
        return False, "missing shortestSubarrayAtLeastK Swift implementation"

    harness = """
func check(_ cond: Bool, _ msg: String) {
    if !cond {
        fputs("FAIL: \\(msg)\\n", stderr)
        exit(1)
    }
}

check(shortestSubarrayAtLeastK([1], 1) == 1, "t0")
check(shortestSubarrayAtLeastK([1, 2], 4) == -1, "t1")
check(shortestSubarrayAtLeastK([2, -1, 2], 3) == 3, "t2")
check(shortestSubarrayAtLeastK([84, -37, 32, 40, 95], 167) == 3, "t3")
check(shortestSubarrayAtLeastK([-28, 81, -20, 28, -29], 89) == 3, "t4")
check(shortestSubarrayAtLeastK([48, 99, 37, 4, -31], 140) == 2, "t5")
check(shortestSubarrayAtLeastK([-5, -3, -1], 1) == -1, "t6")
check(shortestSubarrayAtLeastK([], 1) == -1, "t7")
check(shortestSubarrayAtLeastK([10, -5, 10], 10) == 1, "t8")
print("OK")
"""
    cleaned_code = "\n".join(
        line for line in code.splitlines() if not line.strip().startswith("import ")
    )
    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "main.swift")
        bin_path = os.path.join(tmp, "test_bin")
        with open(src_path, "w") as f:
            f.write("import Foundation\n" + cleaned_code + "\n" + harness)
        comp = subprocess.run(
            ["swiftc", src_path, "-o", bin_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if comp.returncode != 0:
            err_line = (comp.stderr or "").strip().splitlines()[:3]
            return False, f"swiftc error: {' | '.join(err_line)}"
        run = subprocess.run(
            [bin_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if run.returncode != 0 or "OK" not in run.stdout:
            return False, f"swift assertion failed: {run.stderr.strip()}"
    return True, "compiled and passed all 9 Swift monotonic-deque assertions"


def _extract_final_answer(text: str) -> Optional[str]:
    matches = re.findall(r"FINAL_ANSWER\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().rstrip(".").strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    return None


def grade_m1_aime_sum_mod(resp: Dict[str, Any]) -> Tuple[bool, str]:
    text = (resp.get("content") or "").strip()
    if not text:
        return False, "no visible answer emitted"
    ans = _extract_final_answer(text)
    if ans is None:
        return False, "missing FINAL_ANSWER marker"
    m = re.search(r"-?\d+", ans)
    if not m:
        return False, f"could not parse integer from '{ans}'"
    val = int(m.group(0))
    if val == 450:
        return True, "exact AIME remainder 450"
    return False, f"wrong answer {val} (expected 450)"


def grade_m2_gpqa_thermo(resp: Dict[str, Any]) -> Tuple[bool, str]:
    text = (resp.get("content") or "").strip()
    if not text:
        return False, "no visible answer emitted"
    ans = _extract_final_answer(text)
    if ans is None:
        return False, "missing FINAL_ANSWER marker"
    m = re.search(r"\b([ABCD])\b", ans.upper())
    if not m:
        m = re.match(r"^\(?([ABCD])\)?", ans.upper())
    if not m:
        return False, f"could not parse option letter from '{ans}'"
    choice = m.group(1)
    if choice == "B":
        return True, "exact choice B (V = 1.875 V_0)"
    return False, f"wrong choice {choice} (expected B)"


def grade_i1_strict_json(resp: Dict[str, Any]) -> Tuple[bool, str]:
    text = (resp.get("content") or "").strip()
    if not text:
        return False, "empty output"
    if "```" in text:
        return False, "violated raw JSON constraint (contains markdown fence)"
    try:
        obj = json.loads(text)
    except Exception as exc:
        return False, f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return False, "top-level value is not JSON object"
    expected_keys = {"service", "rollback_window_minutes", "dry_run", "steps"}
    if set(obj.keys()) != expected_keys:
        return False, f"wrong keys {sorted(obj.keys())}"
    if not isinstance(obj["service"], str) or not re.fullmatch(r"[a-z0-9-]+", obj["service"]):
        return False, f"invalid service slug: {obj['service']!r}"
    rwm = obj["rollback_window_minutes"]
    if not isinstance(rwm, int) or isinstance(rwm, bool) or not (15 <= rwm <= 60):
        return False, f"invalid rollback_window_minutes: {rwm!r}"
    if obj["dry_run"] is not True:
        return False, f"dry_run must be true, got {obj['dry_run']!r}"
    steps = obj["steps"]
    if not isinstance(steps, list) or len(steps) != 4:
        return False, f"steps must be list of len 4, got {type(steps).__name__}"
    forbidden_words = {"backup", "verify", "snapshot"}
    for idx, step in enumerate(steps):
        if not isinstance(step, dict) or set(step.keys()) != {"id", "action"}:
            return False, f"step {idx} has invalid shape"
        if step["id"] != idx + 1 or isinstance(step["id"], bool):
            return False, f"step {idx} id={step['id']!r}, expected {idx + 1}"
        action = step["action"]
        if not isinstance(action, str):
            return False, f"step {idx} action not string"
        words = action.split()
        if not (5 <= len(words) <= 9):
            return False, f"step {idx+1} word count {len(words)} outside [5, 9]: {action!r}"
        if "z" in action.lower():
            return False, f"step {idx+1} contains forbidden letter 'z': {action!r}"
        tokens_lower = set(re.findall(r"[a-z]+", action.lower()))
        hit = forbidden_words & tokens_lower
        if hit:
            return False, f"step {idx+1} contains forbidden word {sorted(hit)}: {action!r}"
    return True, "passed all strict JSON schema & negative constraints"


def grade_i2_constrained_bullets(resp: Dict[str, Any]) -> Tuple[bool, str]:
    text = (resp.get("content") or "").strip()
    if not text:
        return False, "empty output"
    raw_lines = text.splitlines()
    if len(raw_lines) != 3:
        return False, f"expected exact 3 lines with no blanks, got {len(raw_lines)}"
    for i, line in enumerate(raw_lines):
        if not line.startswith("- "):
            return False, f"line {i+1} does not start with '- ': {line!r}"
        if not re.search(r"\[TAG: [A-Z]+\]$", line):
            return False, f"line {i+1} does not end with '[TAG: <WORD>]': {line!r}"
    word_count = len(text.split())
    if not (45 <= word_count <= 65):
        return False, f"total word count {word_count} outside [45, 65]"
    bad_chars = re.search(r"[0-9*_]", text)
    if bad_chars:
        return False, f"contains forbidden character {bad_chars.group(0)!r}"
    bad_words = re.search(r"\b(hash|hashes|hashing|modulo)\b", text, flags=re.IGNORECASE)
    if bad_words:
        return False, f"contains forbidden word {bad_words.group(0)!r}"
    return True, f"passed all 3-bullet formatting, {word_count}-word window, & negative constraints"


def grade_a1_patch_file(resp: Dict[str, Any]) -> Tuple[bool, str]:
    calls = resp.get("tool_calls") or []
    if not calls:
        return False, f"no tool_call emitted (finish={resp.get('finish_reason')})"
    call = calls[0]
    fn = call.get("function") or {}
    if fn.get("name") != "patch_file":
        return False, f"expected patch_file, got {fn.get('name')}"
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception as exc:
        return False, f"invalid JSON arguments from parser: {exc}"
    if args.get("path") != "/srv/app/tests/test_settings.py":
        return False, f"wrong path: {args.get('path')!r}"
    if args.get("create_backup") is not False:
        return False, f"create_backup must be boolean false, got {args.get('create_backup')!r}"

    original = (
        "import pytest\n"
        "from app.settings import parse_config\n\n\n"
        "def test_loads_nested_env():\n"
        '    cfg = parse_config({"APP_ENV": "prod", "PORT": "8080"})\n'
        '    assert cfg["mode"] == "prod" and cfg["port"] == 8080\n'
    )
    expected = (
        "import pytest\n"
        "from app.settings import load_config\n\n\n"
        "def test_loads_nested_env():\n"
        '    cfg = load_config({"APP_ENV": "prod", "PORT": "8080"})\n'
        '    assert cfg["mode"] == "prod" and cfg["port"] == 8080\n'
    )
    old_text = args.get("old_text")
    new_text = args.get("new_text")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return False, "old_text / new_text must be strings"
    if old_text not in original:
        return False, f"old_text does not match file verbatim: {old_text!r}"
    patched = original.replace(old_text, new_text, 1)
    if patched != expected:
        return False, f"patched result mismatch: {patched!r}"
    return True, "exact multi-line patch_file call verified (bool create_backup=False)"


def grade_a2_exec_command(resp: Dict[str, Any]) -> Tuple[bool, str]:
    calls = resp.get("tool_calls") or []
    if not calls:
        return False, f"no tool_call emitted (finish={resp.get('finish_reason')})"
    call = calls[0]
    fn = call.get("function") or {}
    if fn.get("name") != "exec_command":
        return False, f"expected exec_command, got {fn.get('name')}"
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception as exc:
        return False, f"invalid JSON arguments from parser: {exc}"
    expected_argv = ["/opt/worker/bin/worker-check", "--config", "/opt/worker/etc/prod.toml", "--strict"]
    if args.get("argv") != expected_argv:
        return False, f"wrong argv: {args.get('argv')!r}"
    if args.get("cwd") != "/opt/worker":
        return False, f"wrong cwd: {args.get('cwd')!r}"
    t = args.get("timeout_seconds")
    if t != 30 or isinstance(t, bool):
        return False, f"timeout_seconds must be int 30, got {t!r}"
    expected_env = {"WORKER_ENV": "staging", "LOG_LEVEL": "debug"}
    if args.get("env") != expected_env:
        return False, f"wrong env: {args.get('env')!r}"
    return True, "exact typed exec_command call verified (array/int/object)"


# ---------------------------------------------------------------------------
# 8 AA Benchmark Tasks
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AATask:
    task_id: str
    category: str
    title: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    grader: Callable[[Dict[str, Any]], Tuple[bool, str]]


TASKS: List[AATask] = [
    AATask(
        task_id="C1",
        category="Coding",
        title="Interval Uncovered Spans (LiveCodeBench style)",
        messages=[{
            "role": "user",
            "content": (
                "Write a Python function `uncovered_spans(bounds, intervals)` where `bounds` is `(lo, hi)` with "
                "`lo <= hi` and `intervals` is a list of half-open `[start, end)` tuples `(start, end)`. "
                "Return a sorted list of maximal disjoint `(start, end)` tuples inside `[lo, hi)` not covered by "
                "any interval in `intervals`. Ignore empty/invalid intervals where `start >= end` and clamp all "
                "intervals to `[lo, hi)`. It must run in O(n log n) time (no per-integer bitmaps).\n"
                "Output ONLY a single ```python code block defining `uncovered_spans` with no extra text or tests."
            ),
        }],
        tools=None,
        grader=grade_c1_uncovered_spans,
    ),
    AATask(
        task_id="C2",
        category="Coding",
        title="Shortest Subarray >= K w/ Negatives in Swift (SciCode/LCB style)",
        messages=[{
            "role": "user",
            "content": (
                "Write a Swift function with the exact signature:\n"
                "func shortestSubarrayAtLeastK(_ nums: [Int], _ k: Int) -> Int\n"
                "Return the length of the shortest non-empty contiguous subarray of `nums` (which may contain "
                "negative integers, zero, and positive integers) whose sum is at least `k`, or `-1` if none exists. "
                "Because sliding window fails with negatives, use an O(n) prefix-sum monotonic deque using an "
                "array + head index (no `removeFirst` inside a loop).\n"
                "Output ONLY a single ```swift code block containing `shortestSubarrayAtLeastK` (no imports, no tests, no explanation)."
            ),
        }],
        tools=None,
        grader=grade_c2_shortest_subarray,
    ),
    AATask(
        task_id="M1",
        category="Reasoning",
        title="AIME Weighted Power Sum Mod 1000",
        messages=[{
            "role": "user",
            "content": (
                "Find the remainder when S = sum_{k=1}^{100} (k * 2^k) is divided by 1000.\n"
                "Briefly show the clean closed form and modular arithmetic, and end your response with "
                "`FINAL_ANSWER: <integer>` where `<integer>` is an integer from 0 to 999."
            ),
        }],
        tools=None,
        grader=grade_m1_aime_sum_mod,
    ),
    AATask(
        task_id="M2",
        category="Reasoning",
        title="GPQA Thermodynamics Linear P-V Sign Crossover",
        messages=[{
            "role": "user",
            "content": (
                "One mole of a monatomic ideal gas (molar heat capacity at constant volume C_V = (3/2) R) expands "
                "reversibly along a straight line in the P-V diagram from state (P_0, V_0) to state (P_0 / 2, 2 V_0).\n"
                "At what exact volume V along this path does the molar heat capacity of the process equal zero "
                "(the transition point between heat absorption dQ > 0 and heat release dQ < 0)?\n\n"
                "A) V = 1.5 V_0\n"
                "B) V = 1.875 V_0\n"
                "C) V = 1.25 V_0\n"
                "D) V = 2.0 V_0\n\n"
                "Show a brief derivation from the First Law of Thermodynamics and end with `FINAL_ANSWER: <letter>`."
            ),
        }],
        tools=None,
        grader=grade_m2_gpqa_thermo,
    ),
    AATask(
        task_id="I1",
        category="Instruction Following",
        title="IFBench Strict JSON Schema & Negative Constraints",
        messages=[{
            "role": "user",
            "content": (
                "Output a single raw JSON object (no markdown fences, no commentary before or after) describing a "
                "4-step database schema migration checklist matching EVERY rule below:\n"
                "1. Top-level keys MUST be ONLY: \"service\", \"rollback_window_minutes\", \"dry_run\", \"steps\".\n"
                "2. \"service\" must be a lowercase alphanumeric/hyphen slug matching ^[a-z0-9-]+$.\n"
                "3. \"rollback_window_minutes\" must be an integer between 15 and 60 inclusive.\n"
                "4. \"dry_run\" must be the boolean true.\n"
                "5. \"steps\" must be an array of exact length 4; element i (0-indexed) must have ONLY keys "
                "\"id\" (integer i+1) and \"action\" (string).\n"
                "6. Every \"action\" string must contain between 5 and 9 whitespace-separated words inclusive, "
                "and MUST NOT contain the letter 'z' (case-insensitive) or any of the words \"backup\", "
                "\"verify\", or \"snapshot\" (case-insensitive)."
            ),
        }],
        tools=None,
        grader=grade_i1_strict_json,
    ),
    AATask(
        task_id="I2",
        category="Instruction Following",
        title="IFBench Constrained 3-Bullet Partitioning Prose",
        messages=[{
            "role": "user",
            "content": (
                "Explain how consistent key-ring partitioning minimizes data movement when adding a node to a "
                "distributed cache. Follow EVERY constraint strictly:\n"
                "1. Output EXACTLY 3 lines, each starting with \"- \" (no title, no intro, no conclusion, no blank lines).\n"
                "2. Each line MUST end with an uppercase tag matching `[TAG: <WORD>]` where `<WORD>` is a single "
                "uppercase ASCII word ([A-Z]+).\n"
                "3. Total whitespace-separated word count across the entire response MUST be between 45 and 65 words inclusive.\n"
                "4. Do NOT use any digits (0-9), asterisks (*), underscores (_), or the words \"hash\", \"hashes\", "
                "\"hashing\", or \"modulo\" (case-insensitive)."
            ),
        }],
        tools=None,
        grader=grade_i2_constrained_bullets,
    ),
    AATask(
        task_id="A1",
        category="Agentic Tool Use",
        title="Terminal-Bench Code Patch Recovery (patch_file)",
        messages=[
            {
                "role": "user",
                "content": (
                    "Tests are failing in /srv/app after renaming `parse_config` to `load_config` in `app/settings.py`. "
                    "Inspect the failing test file and patch it cleanly using `patch_file` with `create_backup=false`."
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_read_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "/srv/app/tests/test_settings.py"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "content": (
                    "Pytest failure summary:\n"
                    "ImportError: cannot import name 'parse_config' from 'app.settings'\n\n"
                    "Contents of /srv/app/tests/test_settings.py:\n"
                    "import pytest\n"
                    "from app.settings import parse_config\n\n\n"
                    "def test_loads_nested_env():\n"
                    "    cfg = parse_config({\"APP_ENV\": \"prod\", \"PORT\": \"8080\"})\n"
                    "    assert cfg[\"mode\"] == \"prod\" and cfg[\"port\"] == 8080\n"
                ),
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from disk.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "patch_file",
                    "description": "Replace an exact text snippet inside an existing file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                            "create_backup": {"type": "boolean"},
                        },
                        "required": ["path", "old_text", "new_text", "create_backup"],
                    },
                },
            },
        ],
        grader=grade_a1_patch_file,
    ),
    AATask(
        task_id="A2",
        category="Agentic Tool Use",
        title="Terminal-Bench Typed CLI Triage Recovery (exec_command)",
        messages=[
            {
                "role": "user",
                "content": (
                    "Check the worker error log for the startup issue and run the required staging preflight command."
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_logs_1",
                    "type": "function",
                    "function": {
                        "name": "inspect_logs",
                        "arguments": json.dumps({
                            "path": "/var/log/worker/error.log",
                            "pattern": "BindError",
                            "max_lines": 5,
                        }),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_logs_1",
                "content": (
                    "[ERROR] BindError: address 127.0.0.1:8081 already in use\n"
                    "Runbook instruction on BindError:\n"
                    "Call exec_command immediately with:\n"
                    "- argv: [\"/opt/worker/bin/worker-check\", \"--config\", \"/opt/worker/etc/prod.toml\", \"--strict\"]\n"
                    "- cwd: \"/opt/worker\"\n"
                    "- timeout_seconds: 30\n"
                    "- env: {\"WORKER_ENV\": \"staging\", \"LOG_LEVEL\": \"debug\"}"
                ),
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "inspect_logs",
                    "description": "Search lines in a log file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "pattern": {"type": "string"},
                            "max_lines": {"type": "integer"},
                        },
                        "required": ["path", "pattern", "max_lines"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "Execute a non-interactive process with explicit argv, cwd, timeout, and env.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "argv": {"type": "array", "items": {"type": "string"}},
                            "cwd": {"type": "string"},
                            "timeout_seconds": {"type": "integer"},
                            "env": {"type": "object", "additionalProperties": {"type": "string"}},
                        },
                        "required": ["argv", "cwd", "timeout_seconds", "env"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_report",
                    "description": "Write an incident status file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["path", "status"],
                    },
                },
            },
        ],
        grader=grade_a2_exec_command,
    ),
]


# ---------------------------------------------------------------------------
# Baseline APFS Zero-Copy Clone Management
# ---------------------------------------------------------------------------

def create_baseline_clone() -> None:
    remove_baseline_clone()
    print(f"[setup] Creating zero-copy APFS clone at {BASELINE_CLONE_MODEL}...", flush=True)
    subprocess.run(["cp", "-c", "-R", BASE_SOURCE_MODEL, BASELINE_CLONE_MODEL], check=True)
    for extra in (".backups", "model_dossier.json"):
        p = os.path.join(BASELINE_CLONE_MODEL, extra)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    print(f"[setup] Attaching un-tuned MTP shard {MTP_SAFETENSORS} to Baseline clone...", flush=True)
    subprocess.run(
        [REPACK_BIN, "--attach-mtp", MTP_SAFETENSORS, "--output", BASELINE_CLONE_MODEL],
        check=True,
    )


def remove_baseline_clone() -> None:
    if os.path.exists(BASELINE_CLONE_MODEL):
        print(f"[cleanup] Removing temporary baseline clone {BASELINE_CLONE_MODEL}...", flush=True)
        shutil.rmtree(BASELINE_CLONE_MODEL)


# ---------------------------------------------------------------------------
# MferenceServer Runner & SSE Streaming Telemetry Collector
# ---------------------------------------------------------------------------

class ServerRunner:
    def __init__(self, model_dir: str, port: int, extra_env: Dict[str, str]):
        self.model_dir = model_dir
        self.port = port
        self.extra_env = extra_env
        self.proc: Optional[subprocess.Popen] = None
        self.stderr_lines: List[str] = []
        self._stderr_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        env = os.environ.copy()
        env.update(self.extra_env)
        cmd = [
            SERVER_BIN,
            "--model", self.model_dir,
            "--port", str(self.port),
            "--max-context", str(MAX_CONTEXT),
            "--expert-cache-slots", str(EXPERT_CACHE_SLOTS),
            "--prompt-cache-mode", "off",
            "--verify", "trusted-receipt",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        def _drain_stderr() -> None:
            assert self.proc and self.proc.stderr
            for line in self.proc.stderr:
                with self._stderr_lock:
                    self.stderr_lines.append(line.rstrip("\n"))

        self._reader_thread = threading.Thread(target=_drain_stderr, daemon=True)
        self._reader_thread.start()

        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                with self._stderr_lock:
                    tail = "\n".join(self.stderr_lines[-20:])
                raise RuntimeError(f"MferenceServer exited early ({self.proc.returncode}):\n{tail}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1.0) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.25)
        raise TimeoutError("Timed out waiting for MferenceServer /health")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)

    def pop_new_metrics(self, since_idx: int) -> Tuple[Optional[Dict[str, Any]], int]:
        time.sleep(0.05)
        with self._stderr_lock:
            lines = list(self.stderr_lines[since_idx:])
            new_idx = len(self.stderr_lines)
        for line in reversed(lines):
            if line.startswith("[server-metrics] "):
                return _parse_server_metrics(line), new_idx
        return None, new_idx

    def stderr_len(self) -> int:
        with self._stderr_lock:
            return len(self.stderr_lines)


def _parse_server_metrics(line: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m_pre = re.search(r"prefill=(\d+)tok/([0-9.]+)s", line)
    if m_pre:
        out["prefill_tokens"] = int(m_pre.group(1))
        out["prefill_s"] = float(m_pre.group(2))
    m_dec = re.search(r"decode=(\d+)tok/([0-9.]+)s", line)
    if m_dec:
        out["decode_tokens"] = int(m_dec.group(1))
        out["decode_s"] = float(m_dec.group(2))
    m_stop = re.search(r"stop=([^\s]+)", line)
    if m_stop:
        out["raw_stop"] = m_stop.group(1)
    for field in ("mtp_rounds", "mtp_drafted", "mtp_accepted", "mtp_emitted", "mtp_rollbacks"):
        m = re.search(rf"{field}=(\d+)", line)
        if m:
            out[field] = int(m.group(1))
    return out


def stream_chat_completion(port: int, model_id: str, task: AATask) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": task.messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if task.tools is not None:
        payload["tools"] = task.tools

    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    t_start = time.monotonic()
    t_first_answer: Optional[float] = None
    content_parts: List[str] = []
    tool_builders: Dict[int, Dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = {}

    conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        err_body = resp.read().decode("utf-8", errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {err_body}")

    while True:
        raw_line = resp.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            evt = json.loads(data_str)
        except Exception:
            continue
        if evt.get("usage"):
            usage = evt["usage"]
        choices = evt.get("choices") or []
        if not choices:
            continue
        c0 = choices[0]
        if c0.get("finish_reason") is not None:
            finish_reason = c0["finish_reason"]
        delta = c0.get("delta") or {}
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            if t_first_answer is None:
                t_first_answer = time.monotonic() - t_start
            content_parts.append(delta["content"])
        if delta.get("tool_calls"):
            if t_first_answer is None:
                t_first_answer = time.monotonic() - t_start
            for tc in delta["tool_calls"]:
                idx = tc.get("index", 0)
                entry = tool_builders.setdefault(idx, {"id": tc.get("id"), "type": tc.get("type", "function"), "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    entry["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    entry["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    entry["function"]["arguments"] += fn["arguments"]

    t_total = time.monotonic() - t_start
    conn.close()

    tool_calls = [tool_builders[k] for k in sorted(tool_builders.keys())]
    return {
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
        "wall_s": t_total,
        "ttfat_s": t_first_answer,
    }


def evaluate_variant(variant_name: str, model_dir: str, port: int, extra_env: Dict[str, str]) -> Dict[str, Any]:
    print(f"\n=== Running AA Scorecard: {variant_name} ({os.path.basename(model_dir)}) ===", flush=True)
    server = ServerRunner(model_dir=model_dir, port=port, extra_env=extra_env)
    results: List[Dict[str, Any]] = []
    try:
        server.start()
        for idx, task in enumerate(TASKS):
            if idx > 0:
                time.sleep(COOLDOWN_SECONDS)
            idx_before = server.stderr_len()
            resp = stream_chat_completion(port=port, model_id="qwen3.6-35b-a3b", task=task)
            metrics, _ = server.pop_new_metrics(idx_before)
            passed, detail = task.grader(resp)

            usage = resp.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            completion_tokens = int(usage.get("completion_tokens") or (metrics or {}).get("decode_tokens") or 0)
            reasoning_tokens = int(details.get("reasoning_tokens") or 0)
            non_think_tokens = max(0, completion_tokens - reasoning_tokens)
            details_visible = details.get("visible_tokens")

            decode_s = float((metrics or {}).get("decode_s") or 0.0)
            decode_tok_s = (completion_tokens / decode_s) if decode_s > 0 else 0.0
            e2e_tok_s = (completion_tokens / resp["wall_s"]) if resp["wall_s"] > 0 else 0.0

            rounds = int((metrics or {}).get("mtp_rounds") or 0)
            drafted = int((metrics or {}).get("mtp_drafted") or 0)
            accepted = int((metrics or {}).get("mtp_accepted") or 0)
            emitted = int((metrics or {}).get("mtp_emitted") or 0)
            rollbacks = int((metrics or {}).get("mtp_rollbacks") or 0)
            accept_pct = (100.0 * accepted / drafted) if drafted > 0 else 0.0
            avg_k_emitted = (emitted / rounds) if rounds > 0 else 0.0
            avg_k_drafted = (drafted / rounds) if rounds > 0 else 0.0

            rec = {
                "task_id": task.task_id,
                "category": task.category,
                "title": task.title,
                "passed": passed,
                "detail": detail,
                "finish_reason": resp["finish_reason"],
                "wall_s": resp["wall_s"],
                "ttfat_s": resp["ttfat_s"],
                "prefill_s": (metrics or {}).get("prefill_s", 0.0),
                "decode_s": decode_s,
                "decode_tok_s": decode_tok_s,
                "e2e_tok_s": e2e_tok_s,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "visible_tokens": non_think_tokens,
                "details_visible_tokens": details_visible,
                "mtp_rounds": rounds,
                "mtp_drafted": drafted,
                "mtp_accepted": accepted,
                "mtp_emitted": emitted,
                "mtp_rollbacks": rollbacks,
                "mtp_accept_pct": accept_pct,
                "avg_k_emitted": avg_k_emitted,
                "avg_k_drafted": avg_k_drafted,
                "content": resp["content"],
                "tool_calls": resp["tool_calls"],
            }
            results.append(rec)
            ttfat_disp = f"{rec['ttfat_s']:.2f}s" if rec["ttfat_s"] is not None else "N/A(unclosed)"
            status = "PASS" if passed else "FAIL"
            print(
                f"  [{task.task_id} {task.category:21s}] {status:4s} | "
                f"tok={completion_tokens:3d} (think={reasoning_tokens:3d}, ans={non_think_tokens:3d}) | "
                f"dec={decode_tok_s:5.2f} tok/s e2e={e2e_tok_s:5.2f} | "
                f"TTFAT={ttfat_disp:13s} | wall={rec['wall_s']:5.2f}s | "
                f"MTP={accept_pct:5.1f}% avgK={avg_k_emitted:.2f} (draftK={avg_k_drafted:.2f}) | {detail}",
                flush=True,
            )
    finally:
        server.stop()

    return {
        "variant": variant_name,
        "model_dir": model_dir,
        "tasks": results,
        "summary": summarize_variant(results),
    }


def summarize_variant(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(tasks)
    passed = sum(1 for t in tasks if t["passed"])
    total_completion = sum(t["completion_tokens"] for t in tasks)
    total_reasoning = sum(t["reasoning_tokens"] for t in tasks)
    total_visible = sum(t["visible_tokens"] for t in tasks)
    total_decode_s = sum(t["decode_s"] for t in tasks)
    total_wall_s = sum(t["wall_s"] for t in tasks)
    ttfat_vals = [t["ttfat_s"] for t in tasks if t["ttfat_s"] is not None]
    total_rounds = sum(t["mtp_rounds"] for t in tasks)
    total_drafted = sum(t["mtp_drafted"] for t in tasks)
    total_accepted = sum(t["mtp_accepted"] for t in tasks)
    total_emitted = sum(t["mtp_emitted"] for t in tasks)
    return {
        "tasks_total": n,
        "tasks_passed": passed,
        "pass_at_1_pct": 100.0 * passed / n if n else 0.0,
        "decode_tok_s": total_completion / total_decode_s if total_decode_s > 0 else 0.0,
        "e2e_tok_s": total_completion / total_wall_s if total_wall_s > 0 else 0.0,
        "avg_ttfat_answered_s": sum(ttfat_vals) / len(ttfat_vals) if ttfat_vals else None,
        "answered_count": len(ttfat_vals),
        "avg_reasoning_tokens": total_reasoning / n if n else 0.0,
        "avg_visible_tokens": total_visible / n if n else 0.0,
        "avg_total_tokens": total_completion / n if n else 0.0,
        "avg_wall_s": total_wall_s / n if n else 0.0,
        "mtp_accept_pct": 100.0 * total_accepted / total_drafted if total_drafted > 0 else 0.0,
        "avg_k_emitted": total_emitted / total_rounds if total_rounds > 0 else 0.0,
        "avg_k_drafted": total_drafted / total_rounds if total_rounds > 0 else 0.0,
    }


def print_scorecard(reports: Dict[str, Any]) -> None:
    print("\n================ Artificial Analysis Methodology A/B Scorecard ================")
    for name in ("baseline", "upgraded"):
        if name not in reports:
            continue
        s = reports[name]["summary"]
        ttfat_str = (
            f"{s['avg_ttfat_answered_s']:.2f}s ({s['answered_count']}/{s['tasks_total']} emitted)"
            if s["avg_ttfat_answered_s"] is not None else "N/A (0 emitted)"
        )
        print(
            f"{name.upper():10s}: Pass@1={s['pass_at_1_pct']:5.1f}% ({s['tasks_passed']}/{s['tasks_total']}) | "
            f"Decode={s['decode_tok_s']:5.2f} tok/s (E2E={s['e2e_tok_s']:5.2f} tok/s) | "
            f"TTFAT={ttfat_str} | Think={s['avg_reasoning_tokens']:5.1f} tok | "
            f"Visible={s['avg_visible_tokens']:5.1f} tok | AvgTime={s['avg_wall_s']:5.2f}s | "
            f"MTP={s['mtp_accept_pct']:5.1f}% avg_K={s['avg_k_emitted']:.2f} (draft_K={s['avg_k_drafted']:.2f})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Artificial Analysis A/B Scorecard for Qwen3.6")
    parser.add_argument("--variant", choices=["baseline", "upgraded", "both"], default="both")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--keep-baseline", action="store_true", help="Do not delete temporary baseline clone")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    reports: Dict[str, Any] = {}
    if args.variant != "both" and os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                reports.update(existing)
        except Exception:
            pass

    if args.variant in ("baseline", "both"):
        try:
            create_baseline_clone()
            reports["baseline"] = evaluate_variant(
                "baseline",
                BASELINE_CLONE_MODEL,
                port=18421,
                extra_env={
                    "MFERENCE_SPEC": "mtp",
                    "MFERENCE_SPEC_ADAPTIVE": "0",
                    "MFERENCE_SPEC_K": "2",
                    "MFERENCE_INTENT_ALIGN": "0",
                    "MFERENCE_PHASES": "1",
                },
            )
        finally:
            if not args.keep_baseline:
                remove_baseline_clone()

    if args.variant == "both":
        time.sleep(3.0)

    if args.variant in ("upgraded", "both"):
        reports["upgraded"] = evaluate_variant(
            "upgraded",
            UPGRADED_MODEL,
            port=18422,
            extra_env={
                "MFERENCE_PHASES": "1",
            },
        )

    with open(args.out_json, "w") as f:
        json.dump(reports, f, indent=2)
    print_scorecard(reports)
    print(f"\nSaved full JSON results to {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
