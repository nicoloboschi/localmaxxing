# mlx-bench

Benchmark modern open-source LLMs on the **MLX backend** (Apple Silicon) and
rank them on:

1. **Decode speed** — generation throughput at **1 / 2 / 4 / 8 concurrent requests**.
2. **Prefill speed** — time-to-first-token / prompt-processing throughput across
   **input sizes of ~100 / 500 / 1k / 5k / 10k tokens**.
3. **Features** — **JSON-schema following** (native, no constrained decoding).

Built for an Apple M3 Max (36 GB). Results are written to `results/*.json`.

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
