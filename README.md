<p align="center">
  <img src="docs/assets/mference-app-icon.png" alt="Mference app icon" width="160">
</p>

<h1 align="center">Mference 2.0</h1>

<p align="center">
  <strong>Qwen 3.6 35B-A3B on Apple Silicon — with Native MTP, Hybrid PLD Speculation (v5), Lossless Weight Surgery &amp; DeepSeek Harness Integration</strong><br>
  A native Swift + Metal runtime for Qwen 3.6 35B-A3B that streams routed experts from SSD in bounded memory.<br>
  <em>Forked from <a href="https://github.com/NeelM0906/Mference">NeelM0906/Mference</a>, derived from <a href="https://github.com/drumih/turbo-fieldfare">TurboFieldfare</a>.</em>
</p>

<p align="center">
  <img alt="Swift 6.1 or later" src="https://img.shields.io/badge/Swift-6.1%2B-F05138?logo=swift&logoColor=white">
  <img alt="Metal 3 or later" src="https://img.shields.io/badge/Metal-3%2B-5E5CE6">
  <img alt="macOS 15 or later" src="https://img.shields.io/badge/macOS-15%2B-000000?logo=apple&logoColor=white">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-2ea44f"></a>
</p>

<p align="center">
  <a href="#try-it">Quick start</a> ·
  <a href="MTP_WEIGHT_SURGERY_AND_ARCHITECTURE.md"><strong>MTP, Weight Surgery & Full Architecture</strong></a> ·
  <a href="integrations/deepseek-harness/README.md"><strong>DeepSeek Harness Profile</strong></a> ·
  <a href="docs/OPENAI_SERVER.md">Local server</a> ·
  <a href="docs/OPEN_WEBUI.md">The UI</a> ·
  <a href="docs/QWEN36_PERFORMANCE.md">Qwen 3.6 performance</a> ·
  <a href="docs/SYSTEM_DESIGN.md">How it works</a> ·
  <a href="#acknowledgments">Acknowledgments</a>
</p>

> **Deep-Dive Guide:** See **[`MTP_WEIGHT_SURGERY_AND_ARCHITECTURE.md`](MTP_WEIGHT_SURGERY_AND_ARCHITECTURE.md)** for full technical documentation of the Qwen 3.6 MTP sidecar & hybrid PLD speculation stack (`v5`), lossless resident weight surgery, runtime/cache upgrades, and the complete Mference program architecture, plus **[`integrations/deepseek-harness/README.md`](integrations/deepseek-harness/README.md)** for the tuned DeepSeek Harness desktop profile.

<p align="center">
  <strong>Qwen 3.6 35B-A3B AA Scorecard: 25.0% (2/8) → 100.0% (8/8) Pass@1 · 83.1% MTP acceptance · -69.5% think waste</strong><br>
  <strong>23.5–29.3 tok/s decode on a 24 GB M5 · 36.1–42.2 tok/s on a 256 GB M3 Ultra · runs from ~1.45 GB RAM at 16 slots</strong>
</p>

Mixture-of-experts models activate only a fraction of their parameters per token. **Mference 2.0** specializes on **[Qwen 3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** (`35B` total, `~3B` active per token, `256` experts with `top-8` routing across `40` hybrid Gated-DeltaNet + full-attention layers): it keeps the shared core, GDN state, and KV cache resident while streaming only the active routed experts per token from SSD.

## What's New in Mference 2.0 (Qwen 3.6 `v5`)

**Mference 2.0** extends upstream [NeelM0906/Mference](https://github.com/NeelM0906/Mference) (originally derived from [TurboFieldfare](https://github.com/drumih/turbo-fieldfare)) with a complete speculative, surgical, and agentic-workflow stack for **Qwen 3.6 35B-A3B**:

- **Native 19-Tensor MTP Draft Sidecar & Hybrid PLD Speculation (`v5`):**
  - Attaches the 19-tensor [`guru87/Qwen3.6-35B-A3B-MTP`](https://huggingface.co/guru87/Qwen3.6-35B-A3B-MTP) draft layer (`MTPAttachTool`) and executes it natively on Metal (`Qwen36MTPSpeculator`) with a 4-slot Gated-DeltaNet rollback arena (`RealForwardRunner`).
  - Combines **phase-aware adaptive `K`** (`.thinking`, `.visibleProse`, `.structuredCodeOrTool`), **lossless draft expert top-K pruning** (`moe.metal`), **EVICT logit-margin truncation** (`logit.metal` / `LMHeadChainInt4`), **zero-cost Prompt Lookup Decoding (PLD)** 3-gram/4-gram continuation matching for verbatim code/tool-call spans, and **prefill-tail expert cache warm start**.
- **Lossless Resident Weight Surgery (`MferenceRepack --apply-surgery`):**
  - Calibrates `BF16` norms and `BF16` per-group affine `scale`/`bias` tables in `model_weights.bin` in place — **zero weight nibble re-quantization** — with automatic MTP block co-adaptation (`coAdaptMTP`).
- **Agentic Reliability, Alignment & Multi-Slot Prefix Caching (`v5`):**
  - `RepetitionLoopDetector`, clause-aware soft-landing `<think>`, `ServerIntentAligner` (`.zeroThinkFastPath` + 16 GiB RAM-aware `K` caps), hardened `QwenToolCallParser`/`StructuredAssistantDecoder`, and a **2-slot memory-bounded LRU `ServerPromptCache`** with `PrefixSnapshot` KV + GDN state switching across interleaved conversations.
- **Tuned DeepSeek Harness Integration ([`integrations/deepseek-harness/`](integrations/deepseek-harness/README.md)):**
  - Ships [`cordis.patch.yml`](integrations/deepseek-harness/cordis.patch.yml) with `600000` ms `timeoutMs`/`streamIdleTimeoutMs`, `32768` context / `16384` max output tokens, compact tool-spill/pruner limits, and fast macOS agentic rules.
- **Verified AA Scorecard Results** ([`Scripts/eval_aa_scorecard.py`](Scripts/eval_aa_scorecard.py), [`benchmarks/aa-scorecard-v2.json`](benchmarks/aa-scorecard-v2.json), 32 expert slots, `--max-new 512`, `--temperature 0`):
  - **Pass@1 Intelligence:** `25.0% (2/8)` baseline → **`100.0% (8/8)`** upgraded
  - **MTP Acceptance & Efficiency:** `79.1%` → **`83.1%`** acceptance, **`-69.5%`** average `<think>` token waste (`413.4` → `126.1`)
  - **Agentic (`A1`/`A2`) `v5` Speedup (matched-load control → `v5`):** `A1` decode **`+8.6%`** (`13.19 → 14.33 tok/s`, MTP `83.3%`), `A2` decode **`+7.6%`** (`14.09 → 15.16 tok/s`, MTP `88.5%`)

## Try it

This is a **source release for Apple Silicon Macs** (no bundled model weights in Git). You need macOS 15+, Swift 6.1+ (Xcode 16.3+ or matching Command Line Tools), and `uv` (`brew install uv`) if you want the Open WebUI browser interface. The CLI, local OpenAI server, and DeepSeek Harness profile do not need Python or `uv`.

```bash
git clone https://github.com/fjw4cfv2fb-afk/Mference-2.0.git
cd Mference-2.0
./mference-ui.sh doctor               # checks prerequisites without downloading or changing settings
swift build -c release

# Install our Qwen 3.6 35B-A3B checkpoint + guru87 MTP shard + lossless weight surgery
./mference-ui.sh install qwen36-mtp   # (or: ./Scripts/install-qwen36-mtp.sh)

# Launch MferenceServer + Open WebUI at http://127.0.0.1:3000
./mference-ui.sh
```

> **Important:** Running `./mference-ui.sh install qwen36` installs only the unmodified base [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) checkpoint without MTP or surgery. Use **`./mference-ui.sh install qwen36-mtp`** (or [`Scripts/install-qwen36-mtp.sh`](Scripts/install-qwen36-mtp.sh)) to stream-repack the base checkpoint, download and SHA-256 verify [`guru87/Qwen3.6-35B-A3B-MTP`](https://huggingface.co/guru87/Qwen3.6-35B-A3B-MTP), attach the 19-tensor MTP sidecar, and apply the exact Mference 2.0 weight surgery profile automatically. Interrupted downloads can be continued with `--resume`.

### DeepSeek Harness Setup

To wire DeepSeek Harness to `MferenceServer` with the 10-minute prefill timeout, 16K output limit, and compact tool-output profile:

```bash
mkdir -p ~/.dsh/profiles/desktop
cp integrations/deepseek-harness/cordis.patch.yml ~/.dsh/profiles/desktop/cordis.patch.yml
```

See [`integrations/deepseek-harness/README.md`](integrations/deepseek-harness/README.md) for details.

### Command-Line Usage (without UI)

```bash
# Chat interactively with scratch/qwen36-mtp.gturbo
swift run -c release MferenceCLI --model scratch/qwen36-mtp.gturbo --chat

# Or start the OpenAI-compatible server directly (http://127.0.0.1:8080/v1)
.build/release/MferenceServer \
  --model scratch/qwen36-mtp.gturbo \
  --port 8080 \
  --bind loopback \
  --model-id qwen3.6-35b-a3b \
  --max-context 32768 \
  --prefill-chunk 2048 \
  --prompt-cache-mode single-prefix \
  --expert-cache-slots 48 \
  --verify trusted-receipt
```

See [docs/OPENAI_SERVER.md](docs/OPENAI_SERVER.md) and [docs/OPEN_WEBUI.md](docs/OPEN_WEBUI.md) for server flags and UI options.

## At a glance

| Metric | Value |
| --- | --- |
| Target Model | **Qwen 3.6 35B-A3B** (`mlx-community/Qwen3.6-35B-A3B-4bit` + `guru87/Qwen3.6-35B-A3B-MTP` + lossless resident weight surgery) |
| Architecture | 40 layers (3:1 hybrid Gated-DeltaNet linear attention + full attention), 256 routed experts (`top-8`) + scalar-gated shared expert + 19-tensor MTP draft block |
| Quantization | Group-64 INT4 affine projections & routed experts, INT8 routers, BF16 norms/scale/bias (surgery preserves quantized nibbles bit-exactly) |
| Memory Footprint | ~1.45 GB at 16 slots · auto-scales to 32 slots on 16 GiB+ hosts (48 recommended for `MferenceServer` on 16 GiB) and 96 slots on 24 GiB+ hosts |
| Storage | ~19.6 GB installed (`scratch/qwen36-mtp.gturbo`) |
| Hardware & OS | Apple Silicon Mac (`arm64`), macOS 15+, Metal 3 (MSL 3.2; macOS 26 on Apple10 adds Metal 4 tensor-ops prefill), Swift 6.1+ |
| AA Scorecard (16 GiB Mac, 32 slots) | **100.0% (8/8) Pass@1** (up from 25.0%), **83.1% MTP acceptance**, **16.47 tok/s** avg decode (up to **18.36 tok/s** coding), **+8.6% / +7.6%** `v5` agentic speedup |
| Protocol Throughput (Base Qwen 3.6) | **23.5–29.3 tok/s** decode (24 GB M5, 32 slots) · **36.1–42.2 tok/s** decode (256 GB M3 Ultra, 96 slots, 6.8 GB peak) |

## Products

| Product | Purpose |
| --- | --- |
| `Mference` | Swift library containing the runtime, `Qwen36MTPSpeculator` / hybrid PLD drafter, and Metal kernels |
| `MferenceCLI` | Command-line instruction chat and raw completion |
| `MferenceServer` | OpenAI-compatible Chat Completions server (`127.0.0.1:8080`, `--bind tailnet`, `--library`) with `ServerIntentAligner` and 2-slot LRU `ServerPromptCache` |
| `MferenceRepack` | Streaming model installer, `--attach-mtp`, `--apply-surgery`, and `--verify-install` |
| `MferenceMac` (`Mference.app`) | Native SwiftUI macOS chat app + bundled OpenAI server toggle |
| `mference-ui.sh` | Launcher: installs Open WebUI on first run, starts `MferenceServer`, opens the UI (`install qwen36-mtp`, `models`, `doctor`) |
| `Scripts/install-qwen36-mtp.sh` | One-command installer for Qwen 3.6 + SHA-256 verified `guru87` MTP shard + lossless resident weight surgery |
| `integrations/deepseek-harness/` | DeepSeek Harness `cordis.patch.yml` profile (`timeoutMs: 600000`, `maxTokens: 16384`, agentic prompt/spill rules) |

Only one model-owning product should run at a time.

### Requirements

- An Apple Silicon Mac (`arm64` only)
- macOS 15 or later with Metal 3; Xcode 16.3 and Swift 6.1 or newer (Command Line Tools alone also build and run everything)
- For the Open WebUI browser UI only: [`uv`](https://github.com/astral-sh/uv) (used to install pinned Open WebUI `0.11.3` on first launch; CLI, `MferenceServer`, and DeepSeek Harness need no Python)
- ~25 GB free disk space for the `qwen36-mtp.gturbo` install, MTP shard, and build artifacts
- An internet connection for the initial checkpoint and MTP shard download

## How it works

1. **Streaming Repack & `.gturbo` Layout:** `MferenceRepack` streams bounded byte ranges from Hugging Face and writes resident core tensors into `model_weights.bin` and page-aligned routed experts into per-layer files (`packed_experts/`). Only the experts selected by each layer's router are `pread` into a bounded LFU/LRU slot cache at decode time.
2. **19-Tensor MTP + Hybrid PLD (`v5`):** `MTPAttachTool` appends the 17 resident `mtp.*` tensors to `model_weights.bin` and packs the 256 MTP experts into `packed_experts_mtp/layer_00.bin`. At runtime, `Qwen36MTPSpeculator` combines neural MTP drafting, phase-aware adaptive `K`, EVICT margin truncation, lossless draft top-K pruning, and zero-cost n-gram Prompt Lookup Decoding for verbatim tool/code spans, backed by a 4-slot GDN rollback arena in `RealForwardRunner`.
3. **Lossless Resident Weight Surgery:** `MferenceRepack --apply-surgery` scales only `BF16` norms and `BF16` per-group `scale`/`bias` tables in `model_weights.bin` (with `coAdaptMTP` keeping the draft head aligned), leaving all quantized `INT4`/`INT8` nibbles untouched.
4. **Multi-Slot Prompt Caching (`v5`):** `ServerPromptCache` maintains up to 2 LRU prefix slots (dropping to 1 under memory pressure) and parks/restores linear KV cache, GDN recurrent states, and MTP token history via `PrefixSnapshot`.

For the full design, binary formats, and verified benchmarks, see [`MTP_WEIGHT_SURGERY_AND_ARCHITECTURE.md`](MTP_WEIGHT_SURGERY_AND_ARCHITECTURE.md), [System design](docs/SYSTEM_DESIGN.md), and [Qwen 3.6 performance notes](docs/QWEN36_PERFORMANCE.md).

## Acknowledgments

**Mference 2.0** builds directly upon **[Mference](https://github.com/NeelM0906/Mference)**
by [NeelM0906](https://github.com/NeelM0906), which itself is heavily inspired by
— and its foundation derived from —
**[TurboFieldfare](https://github.com/drumih/turbo-fieldfare)** by Andrey
Mikhaylov (which pioneered running MoE models in bounded memory on Apple Silicon
and documented over a hundred experiments behind its design). Those derived
portions are licensed under the [Apache License 2.0](LICENSE-APACHE); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details. New Mference
code is [MIT-licensed](LICENSE).

Model weights and sidecars remain subject to their own terms: the
[Qwen license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE)
for [Qwen 3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
([`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit))
and [`guru87/Qwen3.6-35B-A3B-MTP`](https://huggingface.co/guru87/Qwen3.6-35B-A3B-MTP)
for the MTP draft shard. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
