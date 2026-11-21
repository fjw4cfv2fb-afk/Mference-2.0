#!/usr/bin/env python3
"""
Synthesizes scene-synced emotional voiceover WAV tracks (44.1 kHz 16-bit mono)
for Mference 2.0 Instagram Reels / TikTok videos and outputs manifest.json with
exact per-scene durations so visual animations stay locked to spoken narration.

Uses Microsoft's neural voice `en-US-AndrewMultilingualNeural` via edge-tts
(auto-bootstrapped into /tmp/mference-tts-venv if needed), with automatic fallback
to macOS `say` (`Samantha` / `Daniel`).
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import sys
import venv
import wave

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUT_DIR = REPO_ROOT / "scratch" / "reels" / "voiceovers"
VENV_DIR = pathlib.Path("/tmp/mference-tts-venv")

REELS = [
    {
        "filename": "mference-reel-1-16gb-mac.mp4",
        "scenes": [
            {
                "id": "reel1_c0",
                "min_duration": 3.0,
                "pad_after": 0.35,
                "text": (
                    "Everyone says you need sixty-four gigs of RAM for a thirty-five billion parameter AI. "
                    "Watch this everyday sixteen gig MacBook."
                ),
            },
            {
                "id": "reel1_c1",
                "min_duration": 5.0,
                "pad_after": 0.35,
                "text": (
                    "That Swift code is streaming live at eighteen tokens a second—zero Python, "
                    "pure Metal, keeping memory completely green from the SSD."
                ),
            },
            {
                "id": "reel1_c2",
                "min_duration": 5.0,
                "pad_after": 0.35,
                "text": (
                    "Raw four-bit models usually overthink and fail. "
                    "Tweaking only the scale math—zero weight nibbles changed—jumped accuracy from twenty-five percent straight to one hundred."
                ),
            },
            {
                "id": "reel1_c3",
                "min_duration": 4.0,
                "pad_after": 0.35,
                "text": (
                    "It predicts up to three tokens ahead at eighty-three percent accuracy, "
                    "and cleanly rolls back whenever a guess misses."
                ),
            },
            {
                "id": "reel1_c4",
                "min_duration": 3.5,
                "pad_after": 0.70,
                "text": (
                    "This is Mference 2.0—free, open source, with a Mac app ready today. Link in bio."
                ),
            },
        ],
    },
    {
        "filename": "mference-reel-2-weight-surgery.mp4",
        "scenes": [
            {
                "id": "reel2_c0",
                "min_duration": 3.0,
                "pad_after": 0.35,
                "text": (
                    "My compressed AI was failing seventy-five percent of its tests—until one tiny math trick took it to one hundred percent."
                ),
            },
            {
                "id": "reel2_c1",
                "min_duration": 4.5,
                "pad_after": 0.35,
                "text": (
                    "Four-bit rounding drift trapped the model inside endless thinking loops, burning over four hundred tokens before even answering."
                ),
            },
            {
                "id": "reel2_c2",
                "min_duration": 5.0,
                "pad_after": 0.35,
                "text": (
                    "Instead of retraining for hours, simple algebra lets us rescale the precision tables directly in seconds—with zero data loss."
                ),
            },
            {
                "id": "reel2_c3",
                "min_duration": 4.0,
                "pad_after": 0.35,
                "text": (
                    "Wasted thinking dropped nearly seventy percent, speeds hit eighteen tokens a second, "
                    "and accuracy jumped from two out of eight to eight out of eight."
                ),
            },
            {
                "id": "reel2_c4",
                "min_duration": 3.0,
                "pad_after": 0.70,
                "text": (
                    "Built in pure Swift and Metal for Apple Silicon. Check Mference 2.0 on GitHub—link in bio."
                ),
            },
        ],
    },
]


def wav_duration(wav_path: pathlib.Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)


def ensure_edge_tts_python() -> pathlib.Path | None:
    venv_py = VENV_DIR / "bin" / "python"
    if venv_py.exists():
        chk = subprocess.run([str(venv_py), "-c", "import edge_tts"], capture_output=True)
        if chk.returncode == 0:
            return venv_py
    try:
        print(f"[tts] Bootstrapping edge-tts in {VENV_DIR}...", flush=True)
        venv.create(VENV_DIR, with_pip=True)
        subprocess.run(
            [str(venv_py), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "edge-tts"],
            check=True,
        )
        return venv_py
    except Exception as exc:
        print(f"[tts] edge-tts bootstrap failed ({exc}); will fall back to macOS say.", flush=True)
        return None


def synthesize_with_edge_tts(venv_py: pathlib.Path, items: list[tuple[str, pathlib.Path]]) -> bool:
    script = """
from __future__ import annotations

import asyncio, json, sys, edge_tts

async def main():
    items = json.loads(sys.argv[1])
    voice = "en-US-AndrewMultilingualNeural"
    for text, mp3_path in items:
        last_err = None
        for attempt in range(4):
            try:
                comm = edge_tts.Communicate(text, voice=voice, rate="+8%")
                await comm.save(mp3_path)
                if os.path.getsize(mp3_path) > 1000:
                    last_err = None
                    break
            except Exception as exc:
                last_err = exc
            await asyncio.sleep(0.8 * (attempt + 1))
        if last_err is not None:
            raise last_err
        await asyncio.sleep(0.35)

import os
asyncio.run(main())
"""
    payload = json.dumps([[text, str(p.with_suffix(".mp3"))] for text, p in items])
    res = subprocess.run([str(venv_py), "-c", script, payload], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[tts] edge-tts synthesis failed: {res.stderr.strip()}", flush=True)
        return False

    for _, wav_path in items:
        mp3_path = wav_path.with_suffix(".mp3")
        subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@44100", "-c", "1", str(mp3_path), str(wav_path)],
            check=True,
        )
        mp3_path.unlink(missing_ok=True)
    return True


def synthesize_with_macos_say(items: list[tuple[str, pathlib.Path]]) -> None:
    for text, wav_path in items:
        aiff_path = wav_path.with_suffix(".aiff")
        subprocess.run(["/usr/bin/say", "-v", "Samantha", "-r", "185", "-o", str(aiff_path), text], check=True)
        subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@44100", "-c", "1", str(aiff_path), str(wav_path)],
            check=True,
        )
        aiff_path.unlink(missing_ok=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_items: list[tuple[str, pathlib.Path]] = []
    for reel in REELS:
        for sc in reel["scenes"]:
            all_items.append((sc["text"], OUT_DIR / f"{sc['id']}.wav"))

    venv_py = ensure_edge_tts_python()
    used_backend = "edge-tts (en-US-AndrewMultilingualNeural)"
    if venv_py is None or not synthesize_with_edge_tts(venv_py, all_items):
        used_backend = "macOS say (Samantha fallback)"
        synthesize_with_macos_say(all_items)

    manifest = {"backend": used_backend, "reels": {}}
    for reel in REELS:
        cursor = 0.0
        boundaries = [0.0]
        clips = []
        for sc in reel["scenes"]:
            wav_path = OUT_DIR / f"{sc['id']}.wav"
            dur = wav_duration(wav_path)
            lead_in = 0.16
            scene_dur = round(max(sc["min_duration"], lead_in + dur + sc["pad_after"]), 2)
            # Align scene duration cleanly to 30 fps frame boundary (multiple of 0.1s is exact 3 frames!)
            scene_dur = round(scene_dur * 10.0) / 10.0
            clips.append({
                "id": sc["id"],
                "wav": str(wav_path),
                "start": round(cursor + lead_in, 3),
                "speech_duration": round(dur, 3),
                "scene_start": round(cursor, 3),
                "scene_end": round(cursor + scene_dur, 3),
            })
            cursor = round(cursor + scene_dur, 3)
            boundaries.append(cursor)

        manifest["reels"][reel["filename"]] = {
            "duration": cursor,
            "boundaries": boundaries,
            "clips": clips,
        }
        print(f"[tts] {reel['filename']}: duration={cursor:.2f}s boundaries={boundaries} ({used_backend})", flush=True)

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[tts] Wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
