"""Registry of models to benchmark.

Top-10 popular OSS instruct models on Hugging Face with <=27B parameters that
ship as 4-bit MLX-community conversions. Ordered small -> large so the
download -> benchmark -> delete cycle keeps peak disk usage bounded (only one
model lives on disk at a time).

`downloads_30d` and `size_gb` are snapshots taken at registry-build time
(2026-06) and are informational only.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    repo: str
    family: str
    params_b: float
    size_gb: float
    downloads_30d: int
    # gemma-2 attention in mlx_lm (0.31.x) crashes under batched decoding
    # (mask broadcast error with batch>1). Such models are served with
    # concurrency=1 so the harness measures serialized throughput instead of
    # crashing; their concurrency levels reflect queuing, not GPU batching.
    batch_safe: bool = True
    # Serving backend: "lm" -> mlx_lm.server (continuous batching), "vlm" ->
    # mlx_vlm.server for multimodal archs mlx_lm can't load (e.g. gemma-4 /
    # model_type gemma4_unified). The vlm server does not continuous-batch, so
    # vlm models are also batch_safe=False (concurrency reflects queuing).
    backend: str = "lm"
    # Included in the default `uv run mlx-bench` sweep? Models that are
    # experimental/broken on the current stack are kept in the registry (so
    # `--only` can still target them) but excluded from the default run.
    default_run: bool = True


# Modern roster (2025-2026 model families), <=27B, 4-bit MLX, small -> large.
# Centered on Qwen3.5 and Gemma-3, plus Phi-4 and Mistral-Small-3.2.
#
# NOTE on Gemma-4: the mlx-community gemma-4 conversions use model_type
# "gemma4_unified" (multimodal) which mlx_lm 0.31.3 -- the latest on PyPI --
# cannot load (it would need mlx-vlm). The gemma-4 "e2b" MatFormer variant
# loads but hangs on generation. So we use Gemma-3 (the newest mlx_lm-runnable
# Gemma; its QAT 4-bit conversions are the most downloaded Gemmas on HF).
MODELS = [
    Model("mlx-community/Qwen3.5-2B-4bit", "Qwen3.5", 2.0, 1.7, 11055),
    Model("mlx-community/Phi-4-mini-instruct-4bit", "Phi-4", 3.8, 2.2, 2840),
    Model("mlx-community/gemma-3-4b-it-qat-4bit", "Gemma-3", 4.3, 3.0, 101556),
    Model("mlx-community/Qwen3.5-4B-4bit", "Qwen3.5", 4.0, 3.1, 22347),
    Model("mlx-community/Qwen3.5-9B-4bit", "Qwen3.5", 9.0, 6.0, 35687),
    Model("mlx-community/gemma-3-12b-it-qat-4bit", "Gemma-3", 12.0, 8.1, 86998),
    Model("mlx-community/phi-4-4bit", "Phi-4", 14.7, 8.3, 959),
    Model("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "Mistral-Small-3.2", 24.0, 13.3, 686),
    Model("mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit", "Qwen3.5-Distill", 27.0, 15.2, 25284),
    Model("mlx-community/gemma-3-27b-it-qat-4bit", "Gemma-3", 27.0, 16.9, 41469),
    # Newer popular models (2025Q4-2026Q2), added later. All on the lm backend;
    # several post-date the mlx_lm 0.31.3 release so they may not load -- the
    # harness records that gracefully. Qwen3.6-35B-A3B exceeds the <=27B rule
    # (35B total, ~3B active MoE) but is included by request.
    Model("mlx-community/Ministral-3-3B-Instruct-2512-4bit", "Ministral-3", 3.0, 2.8, 24914),
    Model("mlx-community/LFM2.5-1.2B-Instruct-4bit", "LFM2.5", 1.2, 0.7, 17326),
    Model("mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", "Devstral-2", 24.0, 14.1, 138278),
    Model("mlx-community/Qwen3.6-27B-4bit", "Qwen3.6", 27.0, 16.1, 17813),
    Model("mlx-community/Qwen3.6-35B-A3B-4bit", "Qwen3.6-MoE", 35.0, 20.4, 78196),
    # Qwythos-9B (Claude-Mythos creative distill, Qwen-based). No mlx-community
    # 4bit exists; this is a community mxfp4 conversion -- non-standard quant, so
    # default_run=False until a load smoke-test confirms mlx_lm can load it.
    Model("sahilchachra/Qwythos-9B-Claude-Mythos-5-1M-mxfp4-mlx", "Qwythos", 9.0, 4.8, 4253,
          default_run=False),
    # Gemma-4 (model_type gemma4_unified, multimodal) -- served via the mlx_vlm
    # backend (mlx_lm cannot load it). No continuous batching on the vlm server,
    # so batch_safe=False. These are default_run=False (run them explicitly with
    # `uv run mlx-bench --only gemma-4-...`) because:
    #   - the dense "12B" works via the vlm server (valid output, fast load) but
    #     is the only one verified;
    #   - the "e2b"/"e4b" MatFormer variants currently FAIL to load on
    #     mlx_vlm 0.6.3 ("Received 140 parameters not in model" -- k_eq_v
    #     fusion mismatch);
    #   - keeping them out of the default sweep keeps it clean and fast.
    Model("mlx-community/gemma-4-12B-it-4bit", "Gemma-4", 12.0, 6.8, 38258,
          batch_safe=False, backend="vlm", default_run=False),
    Model("mlx-community/gemma-4-26b-a4b-it-4bit", "Gemma-4-MoE", 26.0, 15.6, 33371,
          batch_safe=False, backend="vlm", default_run=False),
    Model("mlx-community/gemma-4-e2b-it-4bit", "Gemma-4", 5.0, 3.6, 83321,
          batch_safe=False, backend="vlm", default_run=False),
    Model("mlx-community/gemma-4-e4b-it-4bit", "Gemma-4", 8.0, 5.2, 45392,
          batch_safe=False, backend="vlm", default_run=False),
]
