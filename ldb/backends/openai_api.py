"""Backend: the OpenAI Responses API, direct and tool-free.

Isolated in the sense the benchmark requires: a plain model call with no tools,
no shell, no filesystem. That is what separates this from the `codex` backend,
which reaches the same model family through an agentic CLI that can read the
generator and is therefore unusable for a publishable row.
"""

import os

import openai

from . import cached, with_backoff

DEFAULT_MODEL = "gpt-5.6-luna"  # cheapest of the family: $0.20 / $1.20 per 1M
DEFAULT_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 48000

# Reasoning effort maps straight through; the ladder axis for this family is
# model tier (luna -> terra -> sol) rather than effort, because a budget hint
# does not reliably bind.
EFFORTS = {"none": "minimal", "low": "low", "medium": "medium",
           "high": "high", "max": "high", "default": "medium"}

TEMPERATURE = 0.0

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("set OPENAI_API_KEY")
        _CLIENT = openai.OpenAI()
    return _CLIENT


def describe(model: str | None = None, effort: str | None = None) -> dict:
    """The exact configuration a row was produced with, for the results file."""
    eff = effort or DEFAULT_EFFORT
    return {"backend": "openai", "model": model or DEFAULT_MODEL,
            "effort": eff, "reasoning_effort": EFFORTS.get(eff, "medium"),
            "temperature": TEMPERATURE, "tools": "none", "isolated": True}


def _strict(schema):
    """Structured outputs require every property to be required and closed."""
    if isinstance(schema, dict):
        out = {k: _strict(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["required"] = sorted((out.get("properties") or {}).keys())
        return out
    if isinstance(schema, list):
        return [_strict(v) for v in schema]
    return schema


@cached
@with_backoff
def complete(*, system: str, task: str, schema: dict,
             model: str | None = None, effort: str | None = None) -> str:
    eff = EFFORTS.get(effort or DEFAULT_EFFORT, "medium")
    resp = _client().responses.create(
        model=model or DEFAULT_MODEL,
        instructions=system,
        input=task,
        reasoning={"effort": eff},
        text={"format": {"type": "json_schema", "name": "ldb_predictions",
                         "schema": _strict(schema), "strict": True}},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    if resp.status == "incomplete":
        reason = getattr(resp.incomplete_details, "reason", "unknown")
        raise RuntimeError(
            f"incomplete response ({reason}); reasoning likely consumed the "
            f"output budget - lower effort or raise MAX_OUTPUT_TOKENS")
    if not resp.output_text:
        raise RuntimeError(f"empty response (status={resp.status})")
    return resp.output_text
