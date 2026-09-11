"""Backend: the Anthropic Messages API."""

import anthropic

from . import cached, with_backoff

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 64000


def describe(model: str | None = None, effort: str | None = None) -> dict:
    """The exact configuration a row was produced with, for the results file."""
    return {"backend": "anthropic", "model": model or DEFAULT_MODEL,
            "effort": effort or DEFAULT_EFFORT, "tools": "none", "isolated": True,
            # This model family rejects sampling parameters outright, so
            # neither temperature nor a seed is sent. Twin comparisons on this
            # backend must use --replicates rather than pinned decoding.
            "temperature": "rejected by this model family", "seed": None}


@cached
@with_backoff
def complete(*, system: str, task: str, schema: dict,
             model: str | None = None, effort: str | None = None) -> str:
    client = anthropic.Anthropic()
    # 177 candidate pairs is a lot of composition; give thinking room and stream
    with client.messages.stream(
        model=model or DEFAULT_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        output_config={
            "effort": effort or DEFAULT_EFFORT,
            "format": {"type": "json_schema", "schema": schema},
        },
        messages=[{"role": "user", "content": task}],
    ) as stream:
        resp = stream.get_final_message()

    if resp.stop_reason == "max_tokens":
        raise RuntimeError("hit max_tokens before emitting JSON; raise the budget")
    return next(b.text for b in resp.content if b.type == "text")
