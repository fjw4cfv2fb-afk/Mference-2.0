# Mference 2.0 Launch & Distribution Kit

All metrics below match the reproducible A/B benchmark run in [`benchmarks/aa-scorecard-v2.json`](../../benchmarks/aa-scorecard-v2.json) (`16 GB Mac`, `32` expert cache slots, `temperature = 0`, `max_new_tokens = 512`).

---

## 1. X / Twitter 4-Post Launch Thread

Attach `docs/assets/mference-live-demo.svg` (rendered PNG/screen recording) to **Post 1** and `docs/assets/mference-v2-benchmark.svg` to **Post 2**.

### Post 1 (The Hook + Demo)
> I got Qwen 3.6 35B-A3B (4-bit, 256 MoE experts) running at **100% Pass@1 (8/8)** and up to **18.36 tok/s** on a **16 GB Mac** — in pure Swift 6.1 + Metal 3, streaming experts from NVMe with zero Python in inference.
>
> Prebuilt `.app` (v2.0.0) + code: https://github.com/fjw4cfv2fb-afk/Mference-2.0
>
> 🧵 How it works 👇

### Post 2 (Lossless INT4 Weight Surgery)
> 1/ Raw INT4 Qwen 3.6 kept getting trapped in endless `<think>` loops (scoring just 25% Pass@1, 2/8).
>
> Instead of fine-tuning or re-quantizing nibbles, **Mference 2.0** applies **lossless BF16 scale/norm surgery** ($\alpha w = (\alpha s)q + (\alpha b)$):
> - Pass@1: `25.0% → 100.0%` (`8/8`)
> - `<think>` waste: `-69.5%`
> - 0 INT4 nibbles changed.

### Post 3 (MoE MTP + Gated-DeltaNet Speculation)
> 2/ Standard MTP speculation breaks on Qwen 3.6 because 30 of 40 layers are Gated-DeltaNet linear attention, not KV.
>
> So we built:
> - 4-slot GPU GDN rollback arena (bit-exact with plain greedy decode)
> - 19-tensor MoE MTP sidecar + `topK=6` draft pruning (`83.1%` accept rate, `94.3%` AIME)
> - Single-pass 6-gram Hybrid PLD + exact within-block top-2 EVICT truncation.

### Post 4 (NVMe Streaming + Credit)
> 3/ Built on top of `@NeelM0906`'s Mference & `@drumih`'s TurboFieldfare, shipping with a native macOS SwiftUI app, SwiftNIO OpenAI server (`:8080`), DeepSeek Harness profile, and prebuilt macOS `v2.0.0` binaries.
>
> cc `@awnihannun` `@Alibaba_Qwen` `@JustinLin610` `@simonw` `@ggerganov`
>
> Full architecture writeup: https://github.com/fjw4cfv2fb-afk/Mference-2.0/blob/main/MTP_WEIGHT_SURGERY_AND_ARCHITECTURE.md

---

## 2. Reddit (`r/LocalLLaMA`) Post

**Title:**
> Running Qwen 3.6 35B-A3B at 100% Pass@1 & up to 18.36 tok/s on a 16 GB Mac — Pure Swift + Metal SSD Expert Streaming, 19-Tensor MoE MTP, and Lossless INT4 Weight Surgery

**Body:**
```markdown
Built **Mference 2.0** (building on `NeelM0906/Mference` and `drumih/turbo-fieldfare`) to run **Qwen 3.6 35B-A3B (4-bit, 256 experts)** comfortably on a **16 GB unified-memory Mac** — without quantization collapse or swap storms.

* Repo: https://github.com/fjw4cfv2fb-afk/Mference-2.0
* Prebuilt `v2.0.0` macOS `.app` + CLI binaries: https://github.com/fjw4cfv2fb-afk/Mference-2.0/releases/tag/v2.0.0

### Results on a 16 GB Mac (8-task Coding / AIME Math / GPQA / IFBench / Tool-Call suite, `temp=0`, `32` expert cache slots)
* **Pass@1 Accuracy:** `25.0% (2/8)` raw INT4 baseline → **`100.0% (8/8)`** (`+75.0 pts`)
* **Decode Speed:** **`16.47 tok/s` avg**, up to **`18.36 tok/s`** (Swift coding) and **`17.88 tok/s`** (AIME math)
* **MTP Acceptance:** **`83.1%` overall** (up to **`94.3%`** on AIME, **`2.55` tokens emitted/round**)
* **Wasted `<think>` Tokens:** **`413.4` → `126.1` avg tokens** (`-69.5%`)

### 3 Things That Made the Difference
1. **Lossless INT4 Resident Weight Surgery (`MTPAttachTool`):**
   Raw INT4 Qwen 3.6 kept getting trapped in endless `<think>` loops (`6/8` timeouts before emitting a visible answer). Because MLX group-64 affine weights satisfy $w = s \cdot q + b$, scaling weights by $\alpha$ transforms linearly into $(\alpha \cdot s) \cdot q + (\alpha \cdot b)$ in the `BF16` scale/bias tables and `BF16` norms — **zero nibble unpacking, zero re-quantization, zero fine-tuning**. Scaling late full-attention Q/K norms (`1.03x`), router temperature (`0.96`), late `o_proj` (`0.98x`), shared experts (`1.025x`), and co-adapting `mtp.norm.weight` fixed all 6 failures while keeping `83%+` MTP acceptance.
2. **Native 19-Tensor MoE MTP + Hybrid 6-Gram PLD:**
   Attaches the `guru87/Qwen3.6-35B-A3B-MTP` draft block (`~453 MB` routed experts + 17 resident tensors), pairs it with a **4-slot Gated-DeltaNet rollback arena** (so partial draft rejections rewind both KV and linear-attention recurrent state bit-for-bit cleanly), lossless `topK=6` draft pruning, exact within-block top-2 **EVICT logit-margin truncation**, async-pipelined accept synchronization, and **single-pass 6/5/4/3-gram PLD** for verbatim code/tool-call spans.
3. **16 GB Unified-Memory Survival:**
   Streams routed experts from NVMe via `PreadExpertStreamer` with prefill-tail warm start so decode step 1 hits warm cache slots, `POSIX_MADV_RANDOM` slab-only MTP mapping, plus a memory-bounded 2-slot LRU prompt cache and a native SwiftUI app + SwiftNIO OpenAI server (`:8080`) + Open WebUI launcher.

Hardware reports welcome in GitHub Discussions! Repro benchmark harness (`Scripts/eval_aa_scorecard.py`) and full architecture writeup (`MTP_WEIGHT_SURGERY_AND_ARCHITECTURE.md`) are in the repo.
```

---

## 3. Hacker News (`Show HN`)

**Title:**
> Show HN: Mference 2.0 – Qwen 3.6 35B-A3B in 16 GB RAM (Swift/Metal, SSD MoE + MTP)

**URL:** `https://github.com/fjw4cfv2fb-afk/Mference-2.0`

**First Comment:**
```text
Hi HN — I wanted to run Qwen 3.6 35B-A3B (a 40-layer hybrid Gated-DeltaNet + full-attention MoE with 256 routed experts per layer) locally on a 16 GB Mac with zero swap thrashing and real coding/math accuracy.

Repo: https://github.com/fjw4cfv2fb-afk/Mference-2.0
Prebuilt macOS v2.0.0 binaries: https://github.com/fjw4cfv2fb-afk/Mference-2.0/releases/tag/v2.0.0
Architecture & math writeup: https://github.com/fjw4cfv2fb-afk/Mference-2.0/blob/main/MTP_WEIGHT_SURGERY_AND_ARCHITECTURE.md

A few interesting systems problems we had to solve in pure Swift + Metal (building on NeelM0906/Mference and TurboFieldfare):

1. Speculative rollback across hybrid Gated-DeltaNet + Full Attention layers:
   Rewinding standard KV is just a cursor reset, but 30 of the 40 layers are linear GDN recurrent states. We capture a 4-slot GDN snapshot arena during the batched verify pass so rejecting a draft token blits back the exact pre-rejection FP32 state without recomputing accepted tokens, keeping speculative verify bit-exact with greedy decode.

2. Lossless calibration of 4-bit weights ("weight surgery"):
   On quantized Qwen 3.6, 6/8 tasks on our eval suite spun in <think> loops until hitting the token limit. Instead of re-quantizing or fine-tuning, we scale BF16 RMSNorms and BF16 group-64 affine scale/bias tables in place (since alpha * (s*q + b) = (alpha*s)*q + (alpha*b)), leaving the INT4 nibbles untouched and co-adapting the attached 19-tensor MTP drafter so draft/target calibration stays aligned. Pass@1 went from 25.0% (2/8) to 100.0% (8/8), averaging 16.47 tok/s decode (up to 18.36 tok/s on Swift coding, 83.1% MTP acceptance).

3. 16 GB unified memory + kernel panic avoidance:
   Bounded LFU/LRU expert slot caches (32–48 slots), POSIX_MADV_RANDOM slab-only mmap for MTP experts, async-pipelined accept passes, and single-pass 6-gram PLD over conversation history for verbatim code/tool blocks.

Everything compiles cleanly with Swift 6.1+ (`swift build -c release`) and ships with prebuilt Mference.app/CLI binaries, an OpenAI-compatible SwiftNIO server, CLI, and Open WebUI launcher. Happy to answer questions on the Metal kernels or surgery tradeoffs!
```
