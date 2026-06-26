# mlx-bench

Benchmark modern open-source LLMs on the **MLX backend** (Apple Silicon) and
rank them on:

1. **Decode speed** — generation throughput at **1 / 2 / 4 / 8 concurrent requests**.
2. **Prefill speed** — time-to-first-token / prompt-processing throughput across
   **input sizes of ~100 / 500 / 1k / 5k / 10k tokens**.
3. **Features** — **JSON-schema following** (native, no constrained decoding).
4. **Quality** (opt-in) — **IFEval + GSM8K** via lm-evaluation-harness, graded on
   the actual 4-bit artifact (not a full-precision reference).

Built for an Apple M3 Max (36 GB). Results are written to `results/*.json`.

## Results

Apple M3 Max (36 GB), 4-bit MLX, `max_tokens=256`, greedy decoding. Full data in
[`results/models/`](results/models). Regenerate the ranking with
`uv run mlx-bench --rank`.

### Summary

| Model | Params | Backend | Decode 1× (tok/s) | Decode peak (tok/s @conc) | Prefill @1k (tok/s) | JSON-schema |
|---|--:|:--:|--:|--:|--:|:--:|
| Qwen3.5-2B-4bit | 2.0B | lm | 88.6 | 250 @c8 | 1788 | 4/5 (0.8) |
| Qwen3.5-4B-4bit | 4.0B | lm | 69.9 | 149 @c8 | 825 | 5/5 (1.0) |
| Phi-4-mini-instruct-4bit | 3.8B | lm | 67.8 | 208 @c8 | 1081 | 5/5 (1.0) |
| gemma-3-4b-it-qat-4bit | 4.3B | lm | 63.0 | 197 @c8 | 970 | 5/5 (1.0) |
| gemma-4-26b-a4b-it-4bit | 26.0B (MoE) | vlm | 59.0 | 168 @c8 ¹ | 786 | 5/5 (1.0) |
| Qwen3.5-9B-4bit | 9.0B | lm | 27.8 | 73 @c8 | 345 | 5/5 (1.0) |
| gemma-3-12b-it-qat-4bit | 12.0B | lm | 25.9 | 52 @c4 | 336 | 5/5 (1.0) |
| phi-4-4bit | 14.7B | lm | 25.9 | 38 @c4 | 301 | 5/5 (1.0) |
| gemma-4-12B-it-4bit | 12.0B | vlm | 17.8 | 27 @c2 ¹ | 318 | 5/5 (1.0) |
| Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 27.0B | lm | 8.7 | 21 @c8 | 152 | 5/5 (1.0) |
| gemma-3-27b-it-qat-4bit | 27.0B | lm | 7.1 | 20 @c8 | 148 | 5/5 (1.0) |
| Mistral-Small-3.2-24B-Instruct-2506-4bit | 24.0B | lm | 5.7 | 20 @c8 | 183 | 5/5 (1.0) |

¹ gemma-4 runs via the **mlx-vlm** backend (separate server, needs the mlx-vlm
git build). The **e2b / e4b** MatFormer variants currently **fail to load**
(`Received 140 parameters not in model` — `k_eq_v` fusion mismatch).

### Prefill throughput by input size (tok/s)

| Model | 100t | 500t | 1000t | 5000t | 10000t |
|---|--:|--:|--:|--:|--:|
| Qwen3.5-2B-4bit | 639 | 1422 | 1788 | 1974 | 2273 |
| Phi-4-mini-instruct-4bit | 521 | 950 | 1081 | 1080 | 916 |
| gemma-3-4b-it-qat-4bit | 449 | 821 | 970 | 1048 | 1064 |
| Qwen3.5-4B-4bit | 257 | 724 | 825 | 985 | 931 |
| gemma-4-26b-a4b-it-4bit | 435 | 666 | 786 | 898 | 860 |
| Qwen3.5-9B-4bit | 168 | 253 | 345 | 386 | 370 |
| gemma-3-12b-it-qat-4bit | 226 | 308 | 336 | 334 | 327 |
| gemma-4-12B-it-4bit | 226 | 302 | 318 | 323 | 305 |
| phi-4-4bit | 238 | 288 | 301 | 286 | 245 |
| Mistral-Small-3.2-24B-Instruct-2506-4bit | 129 | 170 | 183 | 173 | 143 |
| Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 96 | 142 | 152 | 152 | 121 |
| gemma-3-27b-it-qat-4bit | 119 | 141 | 148 | 141 | 97 |

### Quality — IFEval + GSM8K (lm-eval, direct-answer mode, 40 items/task)

Graded on the actual 4-bit artifacts. **Direct-answer (non-thinking) mode**, so
reasoning models are not shown at their thinking-mode ceiling (see note below).

| Model | Params | GSM8K | IFEval prompt-strict | IFEval inst-loose |
|---|--:|--:|--:|--:|
| phi-4-4bit | 14.7B | 0.90 | 0.53 | 0.73 |
| gemma-3-12b-it-qat-4bit | 12.0B | 0.88 | 0.82 | 0.84 |
| gemma-3-27b-it-qat-4bit | 27.0B | 0.88 | 0.85 | 0.86 |
| gemma-4-26b-a4b-it-4bit | 26.0B | 0.80 | 0.85 | 0.90 |
| Qwen3.5-4B-4bit | 4.0B | 0.78 | 0.80 | 0.86 |
| Mistral-Small-3.2-24B-Instruct-2506-4bit | 24.0B | 0.75 | 0.65 | 0.71 |
| Qwen3.5-9B-4bit | 9.0B | 0.75 | 0.78 | 0.87 |
| gemma-4-12B-it-4bit | 12.0B | 0.75 | 0.82 | 0.86 |
| gemma-3-4b-it-qat-4bit | 4.3B | 0.72 | 0.65 | 0.79 |
| Phi-4-mini-instruct-4bit | 3.8B | 0.65 | 0.50 | 0.70 |
| Qwen3.5-2B-4bit | 2.0B | 0.50 | 0.57 | 0.73 |
| Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 27.0B | 0.05 ⚠️ | 0.47 | 0.60 |

GSM8K = exact-match (flexible-extract). ⚠️ The Claude-Opus *distill*'s 0.05 is a
**measurement artifact, not a real failure**: it's extremely verbose (~370+
tokens/answer) and GSM8K's default 256-token generation cap truncates it before
it writes the final `#### <answer>` (its IFEval 0.48 and schema 5/5 confirm the
model works). The harness now raises GSM8K's generation budget to 1024 tokens to
avoid this; the table value predates that fix — re-run to refresh.

**Takeaways:** Qwen3.5-2B is the throughput leader (decode + prefill) and the
only model below 5/5 on schema (its `<think>` trace occasionally eats the token
budget). Phi-4-mini and gemma-3-4b are the best 4B-class all-rounders. The
**gemma-4-26b-a4b MoE** punches well above its size — ~59 tok/s single-stream
from a 26B model (only ~4B params active) — and is the fastest model ≥12B here.
Dense gemma-4-12B works via mlx-vlm but is slower than gemma-3-12b. Decode scales
~3–4× with concurrency on small models; the dense 24–27B models are
single-stream-bound (~6–9 tok/s) on this hardware. All models that run follow the
JSON schema 5/5 except the 2B.

On **quality**, the standouts are **gemma-3-12b/27b** (best balance: GSM8K ~0.88,
IFEval ~0.82–0.85) and **phi-4** (best GSM8K at 0.90, but weak instruction-
following at 0.53). **Qwen3.5-4B punches far above its weight** (0.78 GSM8K /
0.80 IFEval at 4B). Note these are *direct-answer* numbers — the Qwen3.5 reasoning
models would score higher with thinking enabled, which the current harness can't
capture cleanly (lm-eval reads the response `content`, not the `reasoning` field).

## How it works

For each model the runner does **download → serve → measure → delete**:

1. `snapshot_download` the 4-bit MLX conversion from HF.
2. Launch a server and wait for it to load, then run a **liveness probe** (one
   tiny request) so a model that loads but hangs on generation is skipped in
   ~60s instead of wasting minutes of suite timeouts.
3. **Prefill suite**: for each input size (~100/500/1k/5k/10k tokens) send a
   `max_tokens=1` request and record TTFT; prefill tok/s = `prompt_tokens` /
   TTFT. Prompts are cache-busted (unique header + filler per size) so the
   server's prompt cache doesn't collapse the timing. Sizes that exceed a
   model's context window are recorded as failures, not crashes.
4. **JSON-schema suite**: 5 extraction tasks, each with a JSON Schema. The model
   is asked to emit conforming JSON; we scan its output for any JSON object that
   validates against the schema. No guided decoding, so this measures the
   model's *native* schema-following ability. Robust to markdown fences,
   schema-echoing, special-token leaks, and reasoning-model `<think>` traces.
5. **Speed suite** (last, on purpose): at each concurrency level, fire N
   identical requests at once and record aggregate throughput, per-request
   throughput, and latency. Running it last means a batched-decode crash can't
   cost the already-captured prefill/schema results.
6. The model is deleted from the HF cache before the next one (disk is tight —
   only models downloaded by this run are deleted; pre-existing cache is kept).

### Output: one JSON per model

Each model writes a **self-contained file** to `results/models/<repo>.json`
(`{repo, machine, config, measured_at, result}`), overwritten on re-run — so
`--only <model>` updates just that model's file. A combined `run_<ts>.json`
snapshot and `latest.json` are also written. `--rank` aggregates the per-model
directory by default. Results are flushed incrementally, so an interrupted run
keeps partial data and can be resumed with `--only`.

## Backends

| backend | server | used for | batching |
|---|---|---|---|
| `lm`  | `mlx_lm.server`  | text LLMs (default) | continuous batching (`--decode-concurrency`/`--prompt-concurrency`) |
| `vlm` | `mlx_vlm.server` | multimodal archs `mlx_lm` can't load (e.g. **Gemma-4** / `gemma4_unified`) | none — requests serialize |

The OpenAI-compatible client works against both. Note `mlx_vlm.server`
**requires** the `model` field in the request body (and validates it against the
loaded model); `mlx_lm.server` ignores it. The runner always sends the repo id.

## Usage

```bash
uv run mlx-bench                 # run the default sweep (writes results/run_<ts>.json)
uv run mlx-bench --rank          # print rankings from the latest results
uv run mlx-bench --rank --results results/run_XXXX.json
uv run mlx-bench --only Qwen3.5-9B gemma-3-12b   # run specific models (substring match)
uv run mlx-bench --only Qwen3.5-9B --quality --quality-limit 40   # add IFEval+GSM8K
```

Options: `--levels 1 2 4 8`, `--max-tokens 256`, `--port 8080`,
`--only <substr> ...`, `--quality`, `--quality-limit N`.

### Quality suite (`--quality`)

Runs real **IFEval** and **GSM8K** through `lm-evaluation-harness` pointed at the
running MLX server, so the *4-bit artifact* is graded with the standard graders.
Opt-in because it's **much slower** than the other suites (GSM8K generates a full
chain of thought per item): roughly **1.5 min/model at `--quality-limit 10`** on
a 2B and proportionally more for larger models / higher limits — budget hours for
the full roster.

It runs in **direct-answer mode** (`enable_thinking: false`): reasoning models
(Qwen3.5) otherwise return their answer in the response `reasoning` field, which
lm-eval doesn't read. So quality numbers reflect *non-thinking* performance,
consistent across all models (the flag is a no-op for non-reasoning models).
Reported metrics: GSM8K exact-match (strict + flexible) and IFEval
prompt/instruction-level strict + loose accuracy.

## Models (modern roster, ≤27B, 4-bit MLX)

Default sweep centers on **Qwen3.5** and **Gemma-3**, plus **Phi-4** and
**Mistral-Small-3.2** (2025–2026 families), ordered small→large. See
`src/mlx_bench/models.py`.

### Gemma-4 (experimental, `--only` / `vlm` backend)

Gemma-4 conversions use `model_type: gemma4_unified` (multimodal), which
**`mlx_lm` cannot load** — they run via the `vlm` backend. They are
`default_run=False` (kept out of the default sweep) and reachable explicitly:

```bash
# Gemma-4 needs the mlx-vlm git build (the released 0.6.3 is too old for these conversions):
uv pip install -U "git+https://github.com/Blaizzy/mlx-vlm"
uv run mlx-bench --only gemma-4-12B
```

Status on the current stack: **`gemma-4-12B` works** (loads ~8s, valid output,
schema 5/5). The **`e2b`/`e4b` MatFormer variants fail to load**
(`Received 140 parameters not in model` — k_eq_v fusion mismatch). A sample
result is in `results/gemma-4-12b-vlm-demo.json`.

## Layout

```
src/mlx_bench/
  models.py       # model registry (backend, batch_safe, default_run flags)
  server.py       # mlx_lm / mlx_vlm server launch + OpenAI client
  speed.py        # concurrency throughput suite
  prefill.py      # prefill / TTFT-by-input-size suite
  schema_test.py  # JSON-schema-following suite
  quality.py      # IFEval + GSM8K via lm-evaluation-harness (opt-in)
  runner.py       # orchestration, liveness probe, disk-safe cleanup
  cli.py          # entry point + ranking
results/
  models/         # one self-contained JSON per model (primary output)
  run_*.json      # combined per-run snapshots
  latest.json     # latest combined run
```

## Notes / future work

- Tool-calling and long-context retrieval are natural next feature tests.
- TTFT is approximated via a `max_tokens=1` request; true streaming TTFT could
  be added.
- `vlm`-backed models don't batch, so their concurrency numbers reflect queuing.
