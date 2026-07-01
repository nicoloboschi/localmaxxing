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

Apple M3 Max (36 GB), all models 4-bit MLX. **19 models** benchmarked across decode
throughput (1/2/4/8 concurrency), prefill/TTFT by input size, JSON-schema
following, and quality (IFEval + GSM8K, 40 items, direct-answer mode). Full data in
[`results/models/`](results/models) (one self-contained JSON per model);
`uv run mlx-bench --rank` reprints these tables.

### Summary

| Model | Params | Backend | Decode 1× (tok/s) | Decode peak (tok/s @conc) | Prefill @1k (tok/s) | Schema | GSM8K | IFEval |
|---|--:|:--:|--:|--:|--:|:--:|--:|--:|
| LFM2.5-230M-MLX-4bit | 0.23B | lm | 304.6 | 1597 @c8 | 13996 | 0.20 | 0.23 | 0.53 |
| LFM2.5-1.2B-Instruct-4bit | 1.2B | lm | 235.7 | 460 @c4 | 2342 | 1.00 | 0.55 | 0.72 |
| LFM2.5-8B-A1B-MLX-4bit | 8.0B | lm | 128.9 | 277 @c4 | 1939 | 0.80 | 0.47 | 0.57 |
| Ministral-3-3B-Instruct-2512-4bit | 3.0B | lm | 93.9 | 222 @c4 | 1699 | 1.00 | 0.72 | 0.55 |
| Qwen3.5-2B-4bit | 2.0B | lm | 84.3 | 219 @c4 | 1405 | 0.80 | 0.50 | 0.57 |
| Qwen3.5-4B-4bit | 4.0B | lm | 69.9 | 149 @c8 | 825 | 1.00 | 0.78 | 0.80 |
| Phi-4-mini-instruct-4bit | 3.8B | lm | 67.8 | 208 @c8 | 1081 | 1.00 | 0.65 | 0.50 |
| gemma-3-4b-it-qat-4bit | 4.3B | lm | 63.0 | 197 @c8 | 970 | 1.00 | 0.72 | 0.65 |
| gemma-4-26b-a4b-it-4bit | 26.0B | vlm | 59.0 | 168 @c8 ¹ | 786 | 1.00 | 0.80 | 0.85 |
| Qwen3.6-35B-A3B-4bit | 35.0B | lm | 41.5 | 95 @c8 | 919 | 1.00 | 0.95 | 0.82 |
| Qwen3.5-9B-4bit | 9.0B | lm | 27.8 | 73 @c8 | 345 | 1.00 | 0.75 | 0.78 |
| gemma-3-12b-it-qat-4bit | 12.0B | lm | 25.9 | 52 @c4 | 336 | 1.00 | 0.88 | 0.82 |
| phi-4-4bit | 14.7B | lm | 25.9 | 38 @c4 | 301 | 1.00 | 0.90 | 0.53 |
| gemma-4-12B-it-4bit | 12.0B | vlm | 17.8 | 27 @c2 ¹ | 318 | 1.00 | 0.75 | 0.82 |
| Qwen3.6-27B-4bit | 27.0B | lm | 13.5 | 25 @c8 | 156 | 1.00 | 1.00 | 0.82 |
| Devstral-Small-2-24B-Instruct-2512-4bit | 24.0B | lm | 11.3 | 27 @c8 | 153 | 1.00 | 0.90 | 0.70 |
| Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 27.0B | lm | 8.7 | 21 @c8 | 152 | 1.00 | 0.05 ⚠️ | 0.47 |
| gemma-3-27b-it-qat-4bit | 27.0B | lm | 7.1 | 20 @c8 | 148 | 1.00 | 0.88 | 0.85 |
| Mistral-Small-3.2-24B-Instruct-2506-4bit | 24.0B | lm | 5.7 | 20 @c8 | 183 | 1.00 | 0.75 | 0.65 |

¹ gemma-4 runs via the **mlx-vlm** backend (separate server). ⚠️ the Claude-Opus *distill*'s GSM8K is a truncation artifact (see note); predates the 1024-token fix.

The gemma-4 **e2b / e4b** MatFormer variants are in the registry but **fail to load** on mlx-vlm, so they're excluded.

### Prefill throughput by input size (tok/s)

| Model | 100t | 500t | 1000t | 5000t | 10000t |
|---|--:|--:|--:|--:|--:|
| LFM2.5-230M-MLX-4bit | 6541 | 13732 | 13996 | 12939 | 8346 |
| LFM2.5-1.2B-Instruct-4bit | 2464 | 3015 | 2342 | 2805 | 2764 |
| LFM2.5-8B-A1B-MLX-4bit | 878 | 1805 | 1939 | 2165 | 2102 |
| Ministral-3-3B-Instruct-2512-4bit | 2905 | 2010 | 1699 | 1174 | 978 |
| Qwen3.5-2B-4bit | 513 | 1268 | 1405 | 1888 | 1833 |
| Phi-4-mini-instruct-4bit | 521 | 950 | 1081 | 1080 | 916 |
| gemma-3-4b-it-qat-4bit | 449 | 821 | 970 | 1048 | 1064 |
| Qwen3.6-35B-A3B-4bit | 346 | 698 | 919 | 1092 | 1087 |
| Qwen3.5-4B-4bit | 257 | 724 | 825 | 985 | 931 |
| gemma-4-26b-a4b-it-4bit | 435 | 666 | 786 | 898 | 860 |
| Qwen3.5-9B-4bit | 168 | 253 | 345 | 386 | 370 |
| gemma-3-12b-it-qat-4bit | 226 | 308 | 336 | 334 | 327 |
| gemma-4-12B-it-4bit | 226 | 302 | 318 | 323 | 305 |
| phi-4-4bit | 238 | 288 | 301 | 286 | 245 |
| Mistral-Small-3.2-24B-Instruct-2506-4bit | 129 | 170 | 183 | 173 | 143 |
| Qwen3.6-27B-4bit | 102 | 150 | 156 | 154 | 104 |
| Devstral-Small-2-24B-Instruct-2512-4bit | 99 | 141 | 153 | 145 | 120 |
| Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 96 | 142 | 152 | 152 | 121 |
| gemma-3-27b-it-qat-4bit | 119 | 141 | 148 | 141 | 97 |

**Takeaways:**
- **Speed:** the non-transformer **LiquidAI LFM2.5** models dominate throughput —
  **LFM2.5-230M** hits 305 tok/s single-stream, **1597 @c8**, and ~14k tok/s prefill;
  the 1.2B and 8B-A1B MoE follow. Decode scales ~3–4× with concurrency on small
  models; dense 24–27B models are single-stream-bound (~6–13 tok/s). But the tiny
  edge models trade quality for speed (230M: schema 0.20, GSM8K 0.23).
- **Quality:** **Qwen3.6-27B** tops GSM8K (perfect **1.00**) at 0.82 IFEval; the
  **Qwen3.6-35B-A3B MoE** is the best quality-per-speed pick (0.95 GSM8K / 0.82 IFEval
  at a fast 41 tok/s). **phi-4** and **Devstral-2-24B** also hit 0.90 GSM8K (phi-4 is a
  weak instruction-follower though, 0.53 IFEval). **gemma-3-12b/27b** are the most
  balanced, and **Qwen3.5-4B punches far above its weight** (0.78/0.80 at 4B).
- **Caveat:** quality is measured in *direct-answer* (non-thinking) mode, so the Qwen3.x
  reasoning models would score higher with thinking enabled (lm-eval reads the response
  `content`, not the `reasoning` field).

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

## Models (modern roster, mostly ≤27B, 4-bit MLX)

2025–2026 families: **Qwen3.5**, **Qwen3.6** (incl. the 35B-A3B MoE — over the
≤27B guideline but added by request), **Gemma-3**, **Gemma-4** (vlm backend),
**Phi-4**, **Mistral-Small-3.2**, **Devstral-2**, **Ministral-3**, and the
non-transformer **LFM2.5**. See `src/mlx_bench/models.py`.

### Gemma-4 (experimental, `--only` / `vlm` backend)

Gemma-4 conversions use `model_type: gemma4_unified` (multimodal), which
**`mlx_lm` cannot load** — they run via the `vlm` backend. They are
`default_run=False` (kept out of the default sweep) and reachable explicitly:

```bash
# Gemma-4 needs the mlx-vlm git build (the released 0.6.3 is too old for these conversions):
uv pip install -U "git+https://github.com/Blaizzy/mlx-vlm"
uv run mlx-bench --only gemma-4-12B
```

Status on the current stack: **`gemma-4-12B` and `gemma-4-26b-a4b` work** (valid
output, schema 5/5 — see the Results tables). The **`e2b`/`e4b` MatFormer variants
fail to load** (`Received 140 parameters not in model` — k_eq_v fusion mismatch).

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
  scripts/        # backfill / scheduled-run helpers
results/
  models/         # one self-contained JSON per model (canonical output)
  latest.json     # aggregate of all per-model results
  # run_*.json per-run snapshots are written locally but gitignored
```

## Notes / future work

- Tool-calling and long-context retrieval are natural next feature tests.
- TTFT is approximated via a `max_tokens=1` request; true streaming TTFT could
  be added.
- `vlm`-backed models don't batch, so their concurrency numbers reflect queuing.
