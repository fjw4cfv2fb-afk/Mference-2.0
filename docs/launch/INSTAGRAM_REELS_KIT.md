# Mference 2.0 Instagram Reels, TikTok & YouTube Shorts Launch Kit

Both vertical `9:16` (`1080×1920`, `30 fps`, H.264 + synthesized 44.1 kHz stereo AAC audio) Reels are rendered into gitignored `scratch/reels/` using Apple-native `AVFoundation` + `CoreGraphics` via [`Scripts/generate-instagram-reels.swift`](../../Scripts/generate-instagram-reels.swift).

```bash
# Regenerate locally on any macOS 15+ Mac (zero third-party dependencies):
swiftc -O Scripts/generate-instagram-reels.swift -o /tmp/gen-reels && /tmp/gen-reels && rm -f /tmp/gen-reels
```

---

## Reel 1 — `"I ran Qwen 3.6 35B on a 16 GB Mac"` (`scratch/reels/mference-reel-1-16gb-mac.mp4`, `18.5s`)

- **Suggested Cover Frame:** ~`11.2s` (`100.0% 8/8 Pass@1 Verified` + `0 INT4 Nibbles Changed` card) or `1.7s` (Hook card)
- **Built-in Audio:** Royalty-free synthesized 120 BPM dark-tech groove + transition chimes (zero copyright claims on Instagram/TikTok/YouTube)

### Ready-to-Paste Caption
> Everyone says you need 64 GB+ unified memory to run a 35B MoE coding model.
>
> I got **Qwen 3.6 35B-A3B** (4-bit, 256 routed experts) running locally on a **16 GB Mac** at **100% Pass@1 (8/8)** and up to **18.36 tok/s** decode — in pure **Swift 6.1 + Metal 3**, streaming routed experts directly from NVMe with zero Python in inference.
>
> What made it work:
> • Lossless BF16 scale/norm surgery (`0` INT4 nibbles touched): `+75.0 pts` Pass@1 (`25% -> 100%`) & `-69.5%` wasted `<think>` tokens
> • 19-tensor MoE MTP sidecar + 6-gram Hybrid PLD (`83.1%` accept rate, `94.3%` AIME math)
> • 4-slot Gated-DeltaNet rollback arena across 30 linear-attention layers
> • Prebuilt `Mference.app` (v2.0.0) + OpenAI server (`:8080`)
>
> Free & open source on GitHub: `fjw4cfv2fb-afk/Mference-2.0` (link in bio)
>
> #AppleSilicon #LocalLLM #Qwen #SwiftLang #Metal #AI #MachineLearning #OpenSource #MacBook #MoE #SpeculativeDecoding #Quantization

### Pinned First Comment
> Repo + prebuilt macOS `Mference.app` (v2.0.0): https://github.com/fjw4cfv2fb-afk/Mference-2.0
> One-command setup: `./mference-ui.sh install qwen36-mtp`

---

## Reel 2 — `"Changing 0 weight nibbles took it from 25% to 100%"` (`scratch/reels/mference-reel-2-weight-surgery.mp4`, `17.0s`)

- **Suggested Cover Frame:** ~`10.2s` (`The Lossless Algebra Trick: α·w = (α·s)·q + (α·b)` + 4 surgery knobs) or `13.8s` (`25.0% -> 100.0%` results card)
- **Built-in Audio:** Same clean procedural track tuned to scene transitions

### Ready-to-Paste Caption
> Raw 4-bit Qwen 3.6 35B kept getting trapped in endless `<think>` loops on our 8-task suite — burning `413.4` reasoning tokens on average and failing `6/8` tasks (`25% Pass@1`).
>
> Instead of fine-tuning or re-quantizing, **Mference 2.0** exploits linear affine group-64 scaling (`w = s·q + b`): multiplying any projection or norm by `α` transforms directly into `(α·s)·q + (α·b)` inside the `BF16` scale/bias tables in `model_weights.bin`.
>
> Results on a **16 GB Mac** (`32` NVMe cache slots, `temp=0`):
> • **INT4 nibbles modified:** `0`
> • **Pass@1 Accuracy:** `25.0% (2/8) → 100.0% (8/8)` (`+75.0 pts`)
> • **`<think>` Token Waste:** `413.4 → 126.1` (`-69.5%`)
> • **Peak Decode:** `18.36 tok/s` (`83.1%` MTP acceptance)
>
> Full math breakdown + open-source Swift/Metal code on GitHub: `fjw4cfv2fb-afk/Mference-2.0`
> Hugging Face card: `voodmany/Qwen3.6-35B-A3B-4bit-MTP-Surgery-GTurbo`
>
> #AI #LocalLLM #LLMQuantization #AppleSilicon #Qwen3 #Swift #Metal #MachineLearning #DeepLearning #OpenSource #Programming

---

## Fast Upload Checklist (Instagram Reels + TikTok + YouTube Shorts)

1. **AirDrop to iPhone (recommended for Instagram):** Uploading via the iOS Instagram app preserves 60/30p crisp encoding, lets you pick a custom cover frame cleanly from the scrubber, and keeps native metadata.
   - Or upload directly from Mac at https://www.instagram.com/ (`+ Create → Select from computer` → pick either `.mp4` in `scratch/reels/`).
2. **Same MP4s work identically on TikTok & YouTube Shorts:** Vertical `1080×1920`, `< 60s`, safe-zone-padded (right action buttons `x > 952` and bottom caption area `y > 1520` kept clear of critical text).
3. **Add link in Instagram Bio:** Set your Instagram bio website to `https://github.com/fjw4cfv2fb-afk/Mference-2.0` before posting.
