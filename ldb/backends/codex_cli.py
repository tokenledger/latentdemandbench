"""Backend: the `codex exec` CLI.

Run read-only and ephemeral with no writable workspace, so this is a single
constrained model call rather than an agentic session.
"""

import json
import os
import subprocess
import tempfile

from . import cached, with_backoff

# Pinned rather than inherited: without --ignore-user-config a run picks up
# ~/.codex/config.toml (model, reasoning effort, plugins), so results would not
# reproduce on another machine. Anything varied for the paper is passed here
# explicitly and recorded in the results file.
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_EFFORT = "max"
TIMEOUT_S = 1800


def describe(model: str | None = None, effort: str | None = None) -> dict:
    """The exact configuration a row was produced with, for the results file."""
    return {"backend": "codex", "model": model or DEFAULT_MODEL,
            "reasoning_effort": effort or DEFAULT_EFFORT, "config": "pinned",
            # An agent with shell tools and full-disk read access. `read-only`
            # blocks writes, not reads, so it can inspect the generator, the
            # cached responses, and other surfaces' outputs. Rows from this
            # backend are not evaluable as clean model measurements.
            "isolated": False,
            "isolation_note": "agentic CLI with filesystem read access"}


@cached
@with_backoff
def complete(
    *, system: str, task: str, schema: dict,
    model: str | None = None, effort: str | None = None,
) -> str:
    prompt = f"{system}\n\n===\n\n{task}"
    with tempfile.TemporaryDirectory() as d:
        schema_path = os.path.join(d, "schema.json")
        out_path = os.path.join(d, "last.txt")
        with open(schema_path, "w") as f:
            json.dump(schema, f)

        cmd = [
            "codex", "exec",
            "--model", model or DEFAULT_MODEL,
            "-c", f"model_reasoning_effort={effort or DEFAULT_EFFORT!r}".replace("'", '"'),
            "--ignore-user-config",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-schema", schema_path,
            "--output-last-message", out_path,
            "--color", "never",
            "-",  # prompt on stdin
        ]

        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT_S
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"codex exec failed ({proc.returncode}): {proc.stderr[-800:]}"
            )
        with open(out_path) as f:
            return f.read()
