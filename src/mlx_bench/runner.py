"""Orchestrate the benchmark: for each model, download -> serve -> measure ->
delete. Results are written incrementally so an overnight run survives crashes.

Disk safety: this machine has limited free space, so each model is deleted from
the HF cache after it is benchmarked. We ONLY delete models that were not
already cached before this run started -- pre-existing models are left intact.
"""

import json
import os
import shutil
import time
import traceback
from pathlib import Path

from huggingface_hub import snapshot_download

from .models import MODELS
from .prefill import run_prefill_suite
from .schema_test import run_schema_suite
from .server import MLXServer
from .speed import run_speed_suite

HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))


def _cache_dir_for(repo: str) -> Path:
    return HF_HUB / ("models--" + repo.replace("/", "--"))


def _free_gb() -> float:
    return shutil.disk_usage(HF_HUB if HF_HUB.exists() else Path.home()).free / 1e9


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run(results_path: Path, port: int = 8080, max_tokens: int = 256,
        levels=(1, 2, 4, 8), log_dir: Path | None = None, only=None):
    log_dir = log_dir or results_path.parent / "server_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_path.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if only:
        models = [m for m in MODELS if any(o.lower() in m.repo.lower() for o in only)]
    else:
        models = [m for m in MODELS if m.default_run]

    machine = {
        "chip": "Apple M3 Max",
        "ram_gb": 36,
        "platform": "darwin",
    }
    report = {
        "started_at": _now(),
        "machine": machine,
        "config": {"levels": list(levels), "max_tokens": max_tokens, "port": port},
        "models": [],
    }

    def flush():
        report["updated_at"] = _now()
        results_path.write_text(json.dumps(report, indent=2))

    def write_model(entry):
        """Write one self-contained JSON per model (machine + config + result).

        Overwritten when the model is re-run (e.g. via --only), so each file
        always holds that model's latest measurement.
        """
        doc = {
            "repo": entry["repo"],
            "machine": machine,
            "config": report["config"],
            "measured_at": _now(),
            "result": entry,
        }
        path = models_dir / (entry["repo"].replace("/", "__") + ".json")
        path.write_text(json.dumps(doc, indent=2))

    flush()

    for i, m in enumerate(models, 1):
        entry = {
            "repo": m.repo,
            "family": m.family,
            "params_b": m.params_b,
            "size_gb": m.size_gb,
            "downloads_30d": m.downloads_30d,
            "batch_safe": m.batch_safe,
            "backend": m.backend,
            "status": "pending",
        }
        report["models"].append(entry)
        was_cached = _cache_dir_for(m.repo).exists()
        print(f"\n[{i}/{len(models)}] {m.repo}  (free {_free_gb():.1f}GB, cached={was_cached})", flush=True)

        server = None
        try:
            # 1. Download (skip-cached). Recorded separately from load time.
            t0 = time.perf_counter()
            snapshot_download(m.repo)
            entry["download_s"] = round(time.perf_counter() - t0, 1)

            # 2. Serve + wait for load. Batch-unsafe models (gemma-2) are served
            #    with concurrency=1 so batched-decode crashes are avoided.
            srv_concurrency = max(levels) if m.batch_safe else 1
            log_path = str(log_dir / (m.repo.replace("/", "__") + ".log"))
            server = MLXServer(m.repo, port=port, max_concurrency=srv_concurrency,
                               log_path=log_path, backend=m.backend)
            t1 = time.perf_counter()
            server.start()
            if not server.wait_ready(timeout=900):
                entry["status"] = "load_failed"
                entry["error"] = "server did not become ready (see server log)"
                flush()
                continue
            entry["load_s"] = round(time.perf_counter() - t1, 1)

            # 2b. Liveness probe: some models load but hang on generation (e.g.
            #     gemma-4 MatFormer on mlx_lm 0.31.x). One tiny request with a
            #     short timeout bounds that to ~60s instead of 30min of suite
            #     timeouts.
            try:
                from .server import chat_completion
                chat_completion(server.base_url, [{"role": "user", "content": "hi"}],
                                max_tokens=4, timeout=60, retries=0, model=m.repo)
            except Exception as e:  # noqa: BLE001
                entry["status"] = "gen_failed"
                entry["error"] = f"generation hung/failed: {type(e).__name__}: {str(e)[:120]}"
                print(f"  GEN_FAILED {e}", flush=True)
                flush()
                continue

            # 3-5. Run each suite independently so a failure in one keeps the
            #      others. Speed (concurrency) runs LAST because batched-decode
            #      can crash/wedge the server for some models; by then prefill
            #      and schema are already captured.
            suite_errors = []
            for name, fn in (
                ("prefill", lambda: run_prefill_suite(server.base_url, model=m.repo)),
                ("schema", lambda: run_schema_suite(server.base_url, model=m.repo)),
                ("speed", lambda: run_speed_suite(server.base_url, levels=levels, max_tokens=max_tokens, model=m.repo)),
            ):
                print(f"  {name}...", flush=True)
                try:
                    entry[name] = fn()
                except Exception as e:  # noqa: BLE001
                    suite_errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
                    entry[name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

            entry["status"] = "ok" if not suite_errors else "partial"
            if suite_errors:
                entry["suite_errors"] = suite_errors
            s, sc, pf = entry.get("speed", {}), entry.get("schema", {}), entry.get("prefill", {})
            print(f"  {entry['status'].upper()}  single={s.get('single_stream_tokens_per_s')} tok/s  "
                  f"peak={s.get('peak_aggregate_tokens_per_s')} tok/s @c{s.get('peak_aggregate_at_concurrency')}  "
                  f"prefill={pf.get('max_prefill_tokens_per_s')} tok/s  "
                  f"schema={sc.get('schema_follow_rate')}", flush=True)
        except Exception as e:  # noqa: BLE001 - never let one model abort the run
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()[-2000:]
            print(f"  ERROR {e}", flush=True)
        finally:
            if server is not None:
                server.stop()
                time.sleep(2)  # let the GPU/memory free before next model
            # 6. Disk cleanup: only delete models we downloaded this run.
            if not was_cached:
                cache_dir = _cache_dir_for(m.repo)
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    entry["cache_deleted"] = True
            flush()
            write_model(entry)

    report["finished_at"] = _now()
    flush()
    return report
