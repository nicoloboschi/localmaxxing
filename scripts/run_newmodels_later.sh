#!/bin/bash
# Unattended: benchmark the 5 newer models (full suite + quality), rebuild the
# aggregate, and push. Resumable — re-running is safe (per-model files overwrite).
# Launched detached after a delay; does NOT use set -e so the push always runs.

UV=/Users/nicoloboschi/.local/bin/uv
cd /Users/nicoloboschi/dev/localmaxxing || exit 1
LOG=results/newmodels_scheduled.log
echo "=== run started $(date) ===" >> "$LOG"

PYTHONUNBUFFERED=1 "$UV" run --no-sync mlx-bench \
  --only "Ministral-3" "LFM2.5" "Devstral" "Qwen3.6" --quality-limit 40 >> "$LOG" 2>&1

# Rebuild latest.json aggregate from all per-model files.
"$UV" run --no-sync python - <<'PY' >> "$LOG" 2>&1
import json, glob
docs = [json.loads(open(f).read()) for f in sorted(glob.glob("results/models/*.json"))]
r = {"description": "Aggregate of all per-model results in results/models/",
     "machine": docs[0].get("machine", {}), "config": docs[0].get("config", {}),
     "models": [d["result"] for d in docs]}
json.dump(r, open("results/latest.json", "w"), indent=2)
print("latest.json:", len(docs), "models")
PY

rm -f results/*.log.tmp 2>/dev/null
git add -A
git commit -m "Add 5 newer models (Qwen3.6-27B/35B-A3B, Devstral-2-24B, Ministral-3, LFM2.5) — full suite + quality" >> "$LOG" 2>&1 || echo "nothing to commit" >> "$LOG"
git push >> "$LOG" 2>&1
echo "=== run finished $(date) ===" >> "$LOG"
