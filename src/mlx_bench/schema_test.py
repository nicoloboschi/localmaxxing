"""Feature test: JSON-schema following.

The mlx_lm server does not do constrained/guided decoding, so this measures
each model's *native* ability to emit schema-conforming JSON when instructed --
a real capability differentiator. For each task we prompt the model with the
schema + an instruction to return ONLY JSON, then:

  - parse_ok:  the output (after stripping markdown fences) parses as JSON
  - schema_ok: the parsed object validates against the JSON Schema

The score is the fraction of tasks that are schema_ok.
"""

import json

import jsonschema

from .server import chat_completion

TASKS = [
    {
        "name": "person_extract",
        "instruction": (
            "Extract the person's info from this text: "
            "'Dr. Maria Chen is 42 years old and works as a cardiologist in Boston.'"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "occupation", "city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sentiment_enum",
        "instruction": (
            "Classify the sentiment of this review: "
            "'The battery life is terrible but the screen is gorgeous.'"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["sentiment", "confidence"],
            "additionalProperties": False,
        },
    },
    {
        "name": "product_list",
        "instruction": (
            "List the products mentioned: 'We sell the AeroBook laptop ($1299), "
            "the Pulse smartwatch ($199), and the Nimbus earbuds ($79).'"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price_usd": {"type": "number"},
                        },
                        "required": ["name", "price_usd"],
                        "additionalProperties": False,
                    },
                    "minItems": 3,
                }
            },
            "required": ["products"],
            "additionalProperties": False,
        },
    },
    {
        "name": "nested_event",
        "instruction": (
            "Structure this event: 'The AI Summit happens on 2026-09-14 in San Francisco. "
            "Speakers include Ada Lovelace (keynote) and Alan Turing (workshop).'"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "location": {"type": "string"},
                "speakers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string", "enum": ["keynote", "workshop", "panel"]},
                        },
                        "required": ["name", "role"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["title", "date", "location", "speakers"],
            "additionalProperties": False,
        },
    },
    {
        "name": "bool_and_types",
        "instruction": (
            "Summarize this support ticket: 'User cannot log in after the update. "
            "This is urgent and affects 1500 users. Ticket #4471.'"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "integer"},
                "urgent": {"type": "boolean"},
                "affected_users": {"type": "integer"},
                "summary": {"type": "string"},
            },
            "required": ["ticket_id", "urgent", "affected_users", "summary"],
            "additionalProperties": False,
        },
    },
]


def _candidate_objects(text: str) -> list:
    """Extract every top-level JSON object from a model response.

    Models wrap JSON in markdown fences, sometimes echo the schema first, leak
    special tokens (e.g. Phi-3.5's <|end|>), or append prose. Scanning for all
    `{...}` objects via raw_decode (which stops at the end of each valid object
    and ignores trailing junk) is robust to all of these. We later accept the
    response if ANY extracted object conforms to the schema.
    """
    objs = []
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        j = text.find("{", i)
        if j == -1:
            break
        try:
            obj, end = dec.raw_decode(text, j)
            if isinstance(obj, dict):
                objs.append(obj)
            i = end
        except json.JSONDecodeError:
            i = j + 1
    return objs


def run_one_task(base_url: str, task: dict, model: str | None = None) -> dict:
    schema_str = json.dumps(task["schema"], indent=2)
    # Note: instruction is folded into a single USER message -- some chat
    # templates (gemma-2, Mistral-v0.3) reject a system role, which the mlx
    # server surfaces as a misleading HTTP 404.
    messages = [
        {
            "role": "user",
            "content": (
                "You are a precise data-extraction engine. Respond with ONLY a "
                "single JSON object and nothing else -- no prose, no markdown "
                "fences, no explanation.\n\n"
                f"{task['instruction']}\n\n"
                f"Return a JSON object that strictly conforms to this JSON Schema:\n"
                f"{schema_str}"
            ),
        },
    ]
    result = {"task": task["name"], "parse_ok": False, "schema_ok": False, "error": None}
    try:
        # Reasoning models (e.g. Qwen3.5) may emit a long <think> trace and put
        # the answer under "reasoning" with "content" empty/missing, so we scan
        # both and allow extra tokens for the thinking budget.
        resp = chat_completion(base_url, messages, max_tokens=2048, temperature=0.0, model=model)
        msg = resp["choices"][0]["message"]
        content = (msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")
        result["raw"] = content.strip()[:800]
        candidates = _candidate_objects(content)
        if not candidates:
            result["error"] = "unparseable"
            return result
        result["parse_ok"] = True
        last_err = None
        for obj in candidates:
            try:
                jsonschema.validate(obj, task["schema"])
                result["schema_ok"] = True
                break
            except jsonschema.ValidationError as e:
                last_err = e
        if not result["schema_ok"] and last_err is not None:
            result["error"] = f"schema: {last_err.message[:160]}"
    except Exception as e:  # noqa: BLE001 - record any failure, keep going
        result["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return result


def run_schema_suite(base_url: str, model: str | None = None) -> dict:
    tasks = [run_one_task(base_url, t, model=model) for t in TASKS]
    n = len(tasks)
    parse_ok = sum(1 for t in tasks if t["parse_ok"])
    schema_ok = sum(1 for t in tasks if t["schema_ok"])
    return {
        "num_tasks": n,
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "parse_rate": round(parse_ok / n, 3) if n else 0.0,
        "schema_follow_rate": round(schema_ok / n, 3) if n else 0.0,
        "tasks": tasks,
    }
