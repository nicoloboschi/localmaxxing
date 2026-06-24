# mlx-bench

Benchmark modern open-source LLMs on the **MLX backend** (Apple Silicon) and
rank them on:

1. **Decode speed** — generation throughput at **1 / 2 / 4 / 8 concurrent requests**.
2. **Prefill speed** — time-to-first-token / prompt-processing throughput across
   **input sizes of ~100 / 500 / 1k / 5k / 10k tokens**.
3. **Features** — **JSON-schema following** (native, no constrained decoding).

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
| Qwen3.5-9B-4bit | 9.0B | lm | 27.8 | 73 @c8 | 345 | 5/5 (1.0) |
| gemma-4-12B-it-4bit | 12.0B | vlm | 26.9 | 29 @c2 ¹ | 320 | 5/5 (1.0) |
| gemma-3-12b-it-qat-4bit | 12.0B | lm | 25.9 | 52 @c4 | 336 | 5/5 (1.0) |
| phi-4-4bit | 14.7B | lm | 25.9 | 38 @c4 | 301 | 5/5 (1.0) |
| Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 27.0B | lm | 8.7 | 21 @c8 | 152 | 5/5 (1.0) |
| gemma-3-27b-it-qat-4bit | 27.0B | lm | 7.1 | 20 @c8 | 148 | 5/5 (1.0) |
| Mistral-Small-3.2-24B-Instruct-2506-4bit | 24.0B | lm | 5.7 | 20 @c8 | 183 | 5/5 (1.0) |

¹ gemma-4 runs via mlx-vlm, which has no continuous batching — its concurrency
numbers reflect queuing, not GPU batching.

### Prefill throughput by input size (tok/s)

| Model | 100t | 500t | 1000t | 5000t | 10000t |
|---|--:|--:|--:|--:|--:|
| Qwen3.5-2B-4bit | 639 | 1422 | 1788 | 1974 | 2273 |
| Phi-4-mini-instruct-4bit | 521 | 950 | 1081 | 1080 | 916 |
| gemma-3-4b-it-qat-4bit | 449 | 821 | 970 | 1048 | 1064 |
| Qwen3.5-4B-4bit | 257 | 724 | 825 | 985 | 931 |
| Qwen3.5-9B-4bit | 168 | 253 | 345 | 386 | 370 |
| gemma-3-12b-it-qat-4bit | 226 | 308 | 336 | 334 | 327 |
| gemma-4-12B-it-4bit | 183 | 260 | 320 | 331 | 302 |
| phi-4-4bit | 238 | 288 | 301 | 286 | 245 |
| Mistral-Small-3.2-24B-Instruct-2506-4bit | 129 | 170 | 183 | 173 | 143 |
| Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 96 | 142 | 152 | 152 | 121 |
| gemma-3-27b-it-qat-4bit | 119 | 141 | 148 | 141 | 97 |

**Takeaways:** Qwen3.5-2B is the throughput leader (decode + prefill) and the
only model below 5/5 on schema (its `<think>` trace occasionally eats the token
budget). Phi-4-mini and gemma-3-4b are the best 4B-class all-rounders. Gemma-4
runs correctly via the mlx-vlm backend with throughput comparable to gemma-3-12b.
Decode scales ~3–4× with concurrency on small models; 24–27B models are usable
but single-stream-bound (~6–9 tok/s) on this hardware.

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
```

Options: `--levels 1 2 4 8`, `--max-tokens 256`, `--port 8080`,
`--only <substr> ...`.

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
