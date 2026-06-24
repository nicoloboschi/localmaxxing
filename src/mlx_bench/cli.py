"""Command-line entry point.

Usage:
    uv run mlx-bench            # run the full suite
    uv run mlx-bench --rank     # print a ranking from the latest results
"""

import argparse
import json
import time
from pathlib import Path

from .runner import run

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _print_ranking(path: Path):
    report = json.loads(path.read_text())
    ok = [m for m in report["models"] if m.get("status") in ("ok", "partial")]
    if not ok:
        print("No completed models in", path)
        return

    print(f"\nResults from {path.name}  ({report.get('started_at')})\n")

    def _has_speed(m, key):
        return isinstance(m.get("speed"), dict) and m["speed"].get(key) is not None

    single = [m for m in ok if _has_speed(m, "single_stream_tokens_per_s")]
    print("=== Speed: single-stream tok/s (concurrency=1) ===")
    for m in sorted(single, key=lambda x: x["speed"]["single_stream_tokens_per_s"], reverse=True):
        tag = "" if m.get("batch_safe", True) else "  [no batching]"
        print(f"  {m['speed']['single_stream_tokens_per_s']:7.1f}  {m['repo']}{tag}")

    peak = [m for m in ok if _has_speed(m, "peak_aggregate_tokens_per_s")]
    print("\n=== Speed: peak aggregate tok/s (best concurrency) ===")
    for m in sorted(peak, key=lambda x: x["speed"]["peak_aggregate_tokens_per_s"], reverse=True):
        s = m["speed"]
        tag = "" if m.get("batch_safe", True) else "  [no batching]"
        print(f"  {s['peak_aggregate_tokens_per_s']:7.1f}  @c{s['peak_aggregate_at_concurrency']}  {m['repo']}{tag}")

    have_prefill = [m for m in ok if isinstance(m.get("prefill"), dict) and m["prefill"].get("sizes")]
    if have_prefill:
        print("\n=== Prefill: tok/s by input size (TTFT in s) ===")
        sizes = [s["target_tokens"] for s in have_prefill[0]["prefill"]["sizes"]]
        header = "  " + " ".join(f"{s:>8}t" for s in sizes) + "   model"
        print(header)
        for m in sorted(have_prefill, key=lambda x: x["prefill"]["max_prefill_tokens_per_s"], reverse=True):
            cells = []
            for s in m["prefill"]["sizes"]:
                cells.append(f"{s['prefill_tokens_per_s']:>8.0f}" if s["ok"] else f"{'--':>8}")
            print("  " + " ".join(cells) + f"   {m['repo']}")

    have_schema = [m for m in ok if isinstance(m.get("schema"), dict) and "schema_follow_rate" in m["schema"]]
    print("\n=== Feature: JSON-schema follow rate ===")
    for m in sorted(have_schema, key=lambda x: x["schema"]["schema_follow_rate"], reverse=True):
        sc = m["schema"]
        print(f"  {sc['schema_follow_rate']:5.2f}  ({sc['schema_ok']}/{sc['num_tasks']})  {m['repo']}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark OSS LLMs on the MLX backend.")
    ap.add_argument("--rank", action="store_true", help="Print ranking from the latest results and exit.")
    ap.add_argument("--results", type=str, default=None, help="Path to results JSON (for --rank).")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--only", type=str, nargs="+", default=None,
                    help="Only run models whose repo contains any of these substrings.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rank:
        path = Path(args.results) if args.results else _latest_results()
        if path is None:
            print("No results files found in", RESULTS_DIR)
            return
        _print_ranking(path)
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"run_{ts}.json"
    latest = RESULTS_DIR / "latest.json"
    print(f"Writing results to {results_path}")
    report = run(results_path, port=args.port, max_tokens=args.max_tokens,
                 levels=tuple(args.levels), only=args.only)
    latest.write_text(json.dumps(report, indent=2))
    print(f"\nDone. {results_path}")
    _print_ranking(results_path)


def _latest_results():
    files = sorted(RESULTS_DIR.glob("run_*.json"))
    return files[-1] if files else None


if __name__ == "__main__":
    main()
