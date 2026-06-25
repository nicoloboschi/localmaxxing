"""Backfill the quality suite (IFEval + GSM8K) into existing per-model files.

Quality-only: runs just the quality suite per model and merges the result into
results/models/<repo>.json, preserving the existing speed/prefill/schema numbers.
Resumable — skips models that already have a quality block. Disk-safe:
download -> eval -> delete (only models not already cached).

Run: uv run --no-sync python scripts/backfill_quality.py [limit]
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import snapshot_download

from mlx_bench.models import MODELS
from mlx_bench.quality import run_quality_suite

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"
LOG_DIR = Path(__file__).resolve().parents[1] / "results" / "server_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))

backend_of = {m.repo: m.backend for m in MODELS}


def cache_dir(repo):
    return HF_HUB / ("models--" + repo.replace("/", "--"))


files = sorted(MODELS_DIR.glob("*.json"))
todo = []
for f in files:
    doc = json.loads(f.read_text())
    res = doc.get("result", {})
    if res.get("status") not in ("ok", "partial"):
        continue
    if isinstance(res.get("quality"), dict) and not res["quality"].get("error"):
        print(f"skip (has quality): {res['repo']}", flush=True)
        continue
    todo.append((f, doc))

print(f"\n{len(todo)} models to backfill at limit={LIMIT}\n", flush=True)

for i, (f, doc) in enumerate(todo, 1):
    repo = doc["result"]["repo"]
    backend = backend_of.get(repo, "lm")
    was_cached = cache_dir(repo).exists()
    print(f"[{i}/{len(todo)}] {repo} (backend={backend}, cached={was_cached})", flush=True)
    t0 = time.time()
    try:
        snapshot_download(repo)
        q = run_quality_suite(repo, backend=backend, port=8081, limit=LIMIT,
                              log_path=str(LOG_DIR / (repo.replace('/', '__') + '.quality.log')))
        doc["result"]["quality"] = q
        doc["quality_measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write_text(json.dumps(doc, indent=2))
        g = q.get("gsm8k", {}); ife = q.get("ifeval", {})
        print(f"   done {time.time()-t0:.0f}s  gsm8k={g.get('exact_match_flexible')} "
              f"ifeval={ife.get('prompt_level_strict_acc')}  err={q.get('error')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"   ERROR {type(e).__name__}: {e}", flush=True)
    finally:
        if not was_cached and cache_dir(repo).exists():
            shutil.rmtree(cache_dir(repo), ignore_errors=True)

print("\nBACKFILL DONE", flush=True)
