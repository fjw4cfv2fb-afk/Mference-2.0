# DeepSeek Harness Integration (`cordis.patch.yml`)

This directory contains the DeepSeek Harness desktop profile patch ([`cordis.patch.yml`](cordis.patch.yml)) tuned for running **Mference 2.0** (`MferenceServer` on `http://127.0.0.1:8080/v1`) in local agentic workflows.

## What This Profile Modifies

- **`mference` OpenAI-Compatible Provider (`llm-pi-ai`):**
  - Points DeepSeek Harness at `http://127.0.0.1:8080/v1` with model `qwen3.6-35b-a3b` and sets it as `agent-default-model`.
  - Sets `timeoutMs: 600000` and `streamIdleTimeoutMs: 600000` (10 minutes) so long agentic prefills do not hit Node's 60-second client socket/headers timeout.
  - Sets `defaultContextWindow: 32768` and `defaultMaxTokens: 16384` (on both provider and model entries) to prevent premature output-token cutoffs and `continue` loops.
- **Prompt & Tool-Output Token Budget Controls:**
  - Disables background title generation (`session-title-llm`) and unused tool bundles (`tool-jobs`, `tool-schedule`, `tool-skill`, `tool-goal`, `tool-web`, `present`, plugin manager/market tools) to keep per-turn prompt size small and prefill fast.
  - Bounds inline tool output (`spill-policy.maxInlineTokens: 1200`), file reads (`tool-fs`: `readLimit: 120`, `readMaxLineLength: 500`, `readMaxBytes: 6144`), search results (`tool-fs-search`: `globMaxResults: 50`, `grepMaxMatches: 50`), and old tool-result compaction (`tool-result-pruner`: `thresholdChars: 2048`, `headChars: 1200`, `tailChars: 400`).
  - Adds a concise macOS `persona` instruction set that steers directory discovery into single bounded commands and limits `<think>` bloat.

## Installation

1. Back up any existing patch layer, then copy [`cordis.patch.yml`](cordis.patch.yml) into your DeepSeek Harness desktop profile directory:

   ```bash
   mkdir -p ~/.dsh/profiles/desktop
   cp integrations/deepseek-harness/cordis.patch.yml ~/.dsh/profiles/desktop/cordis.patch.yml
   ```

2. *(Optional)* In `~/.dsh/profiles/desktop/cordis.patch.yml`, replace `$HOME` inside the `persona.prefix` text with your actual macOS home directory path if desired, then restart **DeepSeek Harness.app**.
3. Start `MferenceServer` on port `8080` (either via the **Start OpenAI Server** button in `Mference.app` or `.build/release/MferenceServer`).
