"""Backend: the Gemini API, direct and tool-free.

Like the Anthropic backend and unlike the codex CLI, this is a plain model call:
no tools, no shell, no filesystem. The model sees the dossier and nothing else,
which is what makes the arm evaluable — a harness with disk access cannot be
shown not to have read the generator.
"""

import os

from google import genai
from google.genai import types

from . import cached, with_backoff

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_EFFORT = "medium"

# Thinking budgets standing in for the effort ladder; 0 disables thinking.
# Bounded on purpose. With thinking_budget=-1 the model spent 30.7k
# of a 32k output budget on thoughts and truncated the JSON mid-answer, so the
# answer must be given a guaranteed share of the budget.
EFFORT_BUDGET = {
    "none": 0, "low": 2048, "medium": 8192,
    "high": 16384, "max": 24576, "default": 8192,
}
ANSWER_TOKENS = 16000  # headroom for TOP_K entries with evidence paths
# gemini-3.6-flash treats thinking_budget as a hint, not a cap: at budget 8192 it
# spent 23k thinking tokens and truncated the answer. Size the ceiling for what
# it actually does, since a truncated response wastes every token it consumed.
MIN_OUTPUT_TOKENS = 48000


TEMPERATURE = 0.0
SEED = 12345


def describe(model: str | None = None, effort: str | None = None) -> dict:
    """The exact configuration a row was produced with, for the results file."""
    eff = effort or DEFAULT_EFFORT
    return {"backend": "gemini", "model": model or DEFAULT_MODEL,
            "effort": eff, "thinking_budget": EFFORT_BUDGET.get(eff, -1),
            "tools": "none", "isolated": True,
            # twin worlds compare two completions; without pinned sampling the
            # difference between them is confounded with inference randomness
            "temperature": TEMPERATURE, "seed": SEED}


# Keywords Gemini's response_schema rejects with a bare 400 INVALID_ARGUMENT.
# Stripping them costs nothing: the top-K cap is enforced in score_predictions
# and the numeric bounds in ldb.backends._validate, both of which apply to every
# backend rather than relying on any provider to honour the schema.
_UNSUPPORTED = ("additionalProperties", "maxItems")


def _clean(schema):
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k not in _UNSUPPORTED}
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


_CLIENT = None


def _client():
    """One long-lived client. Constructing it inline let it be collected
    mid-request ("client has been closed"), and it is thread-safe to share."""
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY (get one at aistudio.google.com/apikey)")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


@cached
@with_backoff
def complete(*, system: str, task: str, schema: dict,
             model: str | None = None, effort: str | None = None) -> str:
    budget = EFFORT_BUDGET.get(effort or DEFAULT_EFFORT, -1)
    resp = _client().models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=task,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=_clean(schema),
            thinking_config=types.ThinkingConfig(thinking_budget=budget),
            max_output_tokens=max(MIN_OUTPUT_TOKENS,
                                  (budget if budget > 0 else 8192) + ANSWER_TOKENS),
            temperature=TEMPERATURE,
            seed=SEED,
        ),
    )
    fin = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
    if fin is not None and "MAX_TOKENS" in str(fin):
        raise RuntimeError(
            f"truncated: thinking used {resp.usage_metadata.thoughts_token_count} tokens; "
            f"lower effort or raise ANSWER_TOKENS")
    if not resp.text:
        raise RuntimeError(
            f"empty response (finish_reason="
            f"{getattr(resp.candidates[0], 'finish_reason', '?') if resp.candidates else 'none'})"
        )
    return resp.text
