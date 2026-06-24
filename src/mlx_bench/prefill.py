"""Prefill (prompt-processing) benchmark across input sizes.

Prefill is the time the model spends ingesting the prompt before it can emit
the first token -- i.e. time-to-first-token (TTFT). It scales with input
length, so we sweep prompt sizes: ~100 / 500 / 1k / 5k / 10k tokens.

We measure it with a non-streaming request capped at max_tokens=1: the wall
latency is then prefill + a single decode step (negligible vs. prefill for
large prompts), which is exactly the user-facing TTFT. We normalize by the
server's *reported* prompt_tokens (usage.prompt_tokens), so the exact prompt
length we built doesn't have to be perfect.

Cache-busting: mlx_lm.server keeps an LRU prompt cache, so each prompt is given
a unique header and unique filler to prevent prefix-cache hits from collapsing
the measured prefill time to ~0.
"""

import time

from .server import chat_completion

INPUT_SIZES = [100, 500, 1000, 5000, 10000]

# These common single words tokenize to ~1.05 tokens each, so words-per-token
# is ~0.95. The server's reported prompt_tokens is the source of truth either
# way; this just makes the prompts land near their size labels.
_WORDS_PER_TOKEN = 0.95

_LEXICON = (
    "system latency throughput memory bandwidth tensor kernel attention cache "
    "pipeline gradient inference quantization parallel scheduler benchmark vector "
    "matrix decode prefill context window embedding transformer residual softmax "
    "rotary position weight activation buffer register stream batch token logits"
).split()


def build_prompt(target_tokens: int, salt: int) -> str:
    """Build a prompt of approximately `target_tokens` tokens, unique per salt."""
    n_words = max(8, int(target_tokens * _WORDS_PER_TOKEN))
    words = []
    # Deterministic but salt-varied word sequence (no RNG -> reproducible).
    for i in range(n_words):
        words.append(_LEXICON[(i * 31 + salt * 7) % len(_LEXICON)])
    body = " ".join(words)
    header = (
        f"[doc-{salt}-{target_tokens}] Read the following technical notes and, "
        f"in one short sentence, name the single most frequent word.\n\n"
    )
    return header + body


def run_one_size(base_url: str, target_tokens: int, salt: int, model: str | None = None) -> dict:
    prompt = build_prompt(target_tokens, salt)
    result = {"target_tokens": target_tokens, "ok": False, "error": None}
    try:
        t0 = time.perf_counter()
        resp = chat_completion(
            base_url,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
            timeout=300.0,
            model=model,
        )
        ttft = time.perf_counter() - t0
        prompt_tokens = resp.get("usage", {}).get("prompt_tokens", 0)
        result.update({
            "ok": True,
            "prompt_tokens": prompt_tokens,
            "ttft_s": round(ttft, 4),
            "prefill_tokens_per_s": round(prompt_tokens / ttft, 1) if ttft > 0 else 0.0,
        })
    except Exception as e:  # noqa: BLE001 - e.g. context-length exceeded (gemma-2 @ 10k)
        result["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return result


def run_prefill_suite(base_url: str, sizes=INPUT_SIZES, warmup: bool = True,
                      model: str | None = None) -> dict:
    if warmup:
        # Compile the graph / warm the GPU so the first measured size isn't
        # penalized by one-time cold-start cost.
        run_one_size(base_url, 64, salt=0, model=model)
    # Distinct salt per size avoids shared prefixes / cache hits between sizes.
    sizes_results = [run_one_size(base_url, sz, salt=i + 1, model=model) for i, sz in enumerate(sizes)]
    ok = [r for r in sizes_results if r["ok"]]
    return {
        "sizes": sizes_results,
        "max_prefill_tokens_per_s": max((r["prefill_tokens_per_s"] for r in ok), default=0.0),
    }
