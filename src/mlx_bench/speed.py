"""Throughput benchmark at varying concurrency levels (1/2/4/8).

For each concurrency level N we fire N identical chat-completion requests
simultaneously and wait for all of them. We report:

  - aggregate_tokens_per_s: total generated tokens / wall-clock time. This is
    the system throughput and is the headline metric -- it should rise with
    concurrency if the server batches effectively.
  - per_request_tokens_per_s: mean of (completion_tokens / request_latency)
    across the N requests -- the speed an individual user perceives.
  - mean_latency_s: mean wall time per request.

Greedy decoding (temperature=0) and a fixed max_tokens keep work comparable
across runs.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from .server import chat_completion

PROMPT = (
    "Write a detailed, technical explanation of how transformer attention "
    "works, including the roles of queries, keys, and values. Be thorough."
)


def _one_request(base_url: str, max_tokens: int, model: str | None = None) -> dict:
    t0 = time.perf_counter()
    resp = chat_completion(
        base_url,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.0,
        model=model,
    )
    dt = time.perf_counter() - t0
    usage = resp.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return {"latency_s": dt, "completion_tokens": completion_tokens}


def run_concurrency_level(base_url: str, concurrency: int, max_tokens: int,
                          model: str | None = None) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(
            lambda _: _one_request(base_url, max_tokens, model=model), range(concurrency)
        ))
    wall = time.perf_counter() - t0

    total_tokens = sum(r["completion_tokens"] for r in results)
    latencies = [r["latency_s"] for r in results]
    per_req_tps = [
        r["completion_tokens"] / r["latency_s"]
        for r in results if r["latency_s"] > 0 and r["completion_tokens"] > 0
    ]
    return {
        "concurrency": concurrency,
        "wall_time_s": round(wall, 3),
        "total_completion_tokens": total_tokens,
        "aggregate_tokens_per_s": round(total_tokens / wall, 2) if wall > 0 else 0.0,
        "per_request_tokens_per_s_mean": round(sum(per_req_tps) / len(per_req_tps), 2) if per_req_tps else 0.0,
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
    }


def run_speed_suite(base_url: str, levels=(1, 2, 4, 8), max_tokens: int = 256,
                    warmup: bool = True, model: str | None = None) -> dict:
    if warmup:
        # Prime the model / prompt cache so the first measured level isn't penalized.
        _one_request(base_url, max_tokens=16, model=model)

    levels_results = []
    for n in levels:
        try:
            levels_results.append(run_concurrency_level(base_url, n, max_tokens, model=model))
        except Exception as e:  # noqa: BLE001 - e.g. gemma-2 crashes under batched decode
            levels_results.append({"concurrency": n, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            # If a concurrency level fails (e.g. batched-decode crash that can
            # leave the server's worker wedged), higher levels will fail too --
            # skip them instead of waiting out a timeout per level.
            if n > 1:
                for higher in (m for m in levels if m > n):
                    levels_results.append({"concurrency": higher, "error": "skipped after lower level failed"})
                break

    ok_levels = [r for r in levels_results if "error" not in r]
    single = next((r for r in ok_levels if r["concurrency"] == 1), None)
    best = max(ok_levels, key=lambda r: r["aggregate_tokens_per_s"], default=None)
    return {
        "max_tokens_per_request": max_tokens,
        "levels": levels_results,
        "single_stream_tokens_per_s": single["aggregate_tokens_per_s"] if single else None,
        "peak_aggregate_tokens_per_s": best["aggregate_tokens_per_s"] if best else None,
        "peak_aggregate_at_concurrency": best["concurrency"] if best else None,
    }
