"""Quality suite: real IFEval + GSM8K via lm-evaluation-harness.

We point `lm_eval`'s `local-chat-completions` model at our running MLX server,
so the *actual 4-bit artifact* is graded (not a full-precision reference) with
the standard, correctly-implemented graders.

Two practical notes that this module handles:

- **Direct-answer mode.** Reasoning models (Qwen3.5) put their answer in the
  response `reasoning` field with `content` empty; lm-eval reads `content`, so
  it would score 0. We serve the quality pass with thinking disabled
  (`enable_thinking: false`) so the answer lands in `content`. This means the
  numbers are *direct-answer* quality, consistent across all models (the flag is
  a no-op for non-reasoning models). The mlx_vlm backend already defaults to
  non-thinking.

- **Cost.** GSM8K generates a full chain of thought per item, so this suite is
  far slower than the others; keep `limit` modest and run it opt-in.

This module manages its own server (the caller passes the repo/backend) so the
quality pass can use a different chat-template config than the main suites.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import time

from .server import MLXServer

TASKS = ["ifeval", "gsm8k"]


def _parse_results(results_dir: str) -> dict:
    files = glob.glob(os.path.join(results_dir, "**", "results_*.json"), recursive=True)
    if not files:
        return {}
    res = json.load(open(max(files, key=os.path.getmtime)))["results"]
    out = {}
    if "ifeval" in res:
        i = res["ifeval"]
        out["ifeval"] = {
            "samples": i.get("sample_len"),
            "prompt_level_strict_acc": i.get("prompt_level_strict_acc,none"),
            "prompt_level_loose_acc": i.get("prompt_level_loose_acc,none"),
            "inst_level_strict_acc": i.get("inst_level_strict_acc,none"),
            "inst_level_loose_acc": i.get("inst_level_loose_acc,none"),
        }
    if "gsm8k" in res:
        g = res["gsm8k"]
        out["gsm8k"] = {
            "samples": g.get("sample_len"),
            "exact_match_strict": g.get("exact_match,strict-match"),
            "exact_match_flexible": g.get("exact_match,flexible-extract"),
        }
    return out


def run_quality_suite(repo: str, backend: str = "lm", port: int = 8081,
                      limit: int = 40, log_path: str | None = None,
                      num_concurrent: int = 8, timeout_s: int = 3600) -> dict:
    """Start a (direct-answer) server for `repo`, run IFEval+GSM8K, return scores."""
    # vlm backend already defaults to non-thinking; lm backend needs the flag.
    cta = None if backend == "vlm" else '{"enable_thinking": false}'
    server = MLXServer(repo, port=port, max_concurrency=num_concurrent,
                       log_path=log_path, backend=backend, chat_template_args=cta)
    result = {"limit": limit, "tasks": TASKS}
    t0 = time.perf_counter()
    try:
        server.start()
        if not server.wait_ready(timeout=900):
            return {**result, "error": "quality server did not become ready"}

        outdir = tempfile.mkdtemp(prefix="mlxbench_lmeval_")
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--model_args",
            # max_gen_toks raises the model's fallback generation length from
            # lm-eval's default 256 -> 1024. GSM8K sets no per-task cap, so a
            # verbose model (e.g. reasoning distills) would otherwise be
            # truncated before writing "#### <answer>" and score ~0. IFEval keeps
            # its own task-level 1280 cap.
            (f"base_url=http://127.0.0.1:{port}/v1/chat/completions,"
             f"model={repo},num_concurrent={num_concurrent},"
             f"max_gen_toks=1024,tokenized_requests=False,timeout=600"),
            "--tasks", ",".join(TASKS),
            "--limit", str(limit),
            "--apply_chat_template",
            "--output_path", outdir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        result["elapsed_s"] = round(time.perf_counter() - t0, 1)
        parsed = _parse_results(outdir)
        if parsed:
            result.update(parsed)
        else:
            result["error"] = "no lm_eval results parsed"
            result["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-8:])
    except subprocess.TimeoutExpired:
        result["error"] = f"lm_eval timed out after {timeout_s}s"
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        server.stop()
    return result
