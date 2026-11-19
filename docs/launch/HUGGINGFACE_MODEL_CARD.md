---
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: quantized
library_name: mference
license: apache-2.0
pipeline_tag: text-generation
tags:
  - apple-silicon
  - metal
  - swift
  - moe
  - speculative-decoding
  - mtp
  - qwen3.6
  - weight-surgery
  - 4-bit
  - gturbo
---

# Qwen3.6-35B-A3B-4bit-MTP-Surgery (.gturbo for Mference 2.0)

*Hugging Face Repository:* **[`voodmany/Qwen3.6-35B-A3B-4bit-MTP-Surgery-GTurbo`](https://huggingface.co/voodmany/Qwen3.6-35B-A3B-4bit-MTP-Surgery-GTurbo)**

Run **[Qwen 3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** (`40` hybrid Gated-DeltaNet + full-attention layers, `256` experts/layer, `top-8` routed + shared expert) on a **16 GB unified-memory Apple Silicon Mac** at **`100.0% (8/8)` Pass@1** and up to **`18.36 tok/s`** decode in pure **Swift 6.1 + Metal 3** via **[Mference 2.0](https://github.com/fjw4cfv2fb-afk/Mference-2.0)**.

## What Makes This Conversion Different?

1. **19-Tensor Native MoE MTP Draft Sidecar (`guru87/Qwen3.6-35B-A3B-MTP`):**
   - Attaches `17` resident `mtp.*` tensors into `model_weights.bin` plus `packed_experts_mtp/layer_00.bin` (`256` page-aligned INT4 experts, `452,984,832` bytes).
   - Paired with Mference's **4-slot Gated-DeltaNet rollback arena**, lossless `topK = 6` draft pruning, exact within-block top-2 EVICT truncation, async-pipelined accept synchronization, and single-pass **6/5/4/3-gram Hybrid PLD**.
2. **Lossless INT4 Resident Weight Surgery (`MTPAttachTool`):**
   - Because MLX group-64 affine weights satisfy $w = s \cdot q + b$, scaling projections by $\alpha$ transforms linearly into $(\alpha \cdot s) \cdot q + (\alpha \cdot b)$ purely inside `BF16` scale/bias tables and `BF16` norms — **never unpacking or re-quantizing INT4 nibbles**.
   - Calibration profile applied (`model_weights.bin`, SHA-256 verified):
     - `--surgery-final-norm 1.02`
     - `--surgery-qk-norm 1.03 --surgery-qk-start-ratio 0.5`
     - `--surgery-router-temp 0.96 --surgery-late-start-ratio 0.75`
     - `--surgery-late-residual 0.98`
     - `--surgery-shared-expert 1.025 --surgery-shared-expert-start-ratio 0.5`
     - Automatic MTP co-adaptation (`coAdaptMTP = true`) on `mtp.norm.weight`, attention norms/output, router, and shared expert.

## Measured A/B Benchmark Results (`16 GB Mac`, `32` Expert Cache Slots, `temp = 0`)

From [`benchmarks/aa-scorecard-v2.json`](https://github.com/fjw4cfv2fb-afk/Mference-2.0/blob/main/benchmarks/aa-scorecard-v2.json) across 8 Coding (Python/Swift), Math/Physics (AIME/GPQA), Instruction Following (IFBench), and Tool-Calling tasks:

| Metric | Raw INT4 Baseline | Mference 2.0 (Surgery + Hybrid MTP) | Impact |
| :--- | ---: | ---: | :--- |
| **Pass@1 Accuracy** | `25.0%` (`2/8`) | **`100.0%` (`8/8`)** | **`+75.0 pts`** |
| **Avg Decode Speed** | `15.96 tok/s` | **`16.47 tok/s`** | Up to **`18.36 tok/s`** (Swift) / **`17.88 tok/s`** (AIME) |
| **MTP Acceptance** | `79.1%` static | **`83.1%` adaptive hybrid** | **`94.3%`** on AIME (`2.55` avg tokens/round) |
| **Avg Time / Task** | `32.14s` | **`25.31s`** | **`-21.3%`** (`13.11s` avg TTFAT) |
| **`<think>` Token Waste** | `413.4` avg | **`126.1` avg** | **`-69.5%`** |

## One-Command Local Install & Run (Mference 2.0)

```bash
git clone https://github.com/fjw4cfv2fb-afk/Mference-2.0.git && cd Mference-2.0
./mference-ui.sh install qwen36-mtp   # streams 4-bit base + guru87 MTP shard + applies lossless surgery
./mference-ui.sh                      # starts OpenAI server (:8080) + Open WebUI (:3000)
```

Or install the prebuilt `Mference.app` from **[Release v2.0.0](https://github.com/fjw4cfv2fb-afk/Mference-2.0/releases/tag/v2.0.0)**.

## Acknowledgments & Licensing

- Base Model: [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Qwen Apache-2.0 / Qwen License) & [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)
- MTP Draft Sidecar: [`guru87/Qwen3.6-35B-A3B-MTP`](https://huggingface.co/guru87/Qwen3.6-35B-A3B-MTP)
- Runtime: [Mference 2.0](https://github.com/fjw4cfv2fb-afk/Mference-2.0) (forked from [`NeelM0906/Mference`](https://github.com/NeelM0906/Mference), derived from [`TurboFieldfare`](https://github.com/drumih/turbo-fieldfare))
