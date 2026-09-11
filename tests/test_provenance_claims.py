"""Regression checks for provenance claims made in the camera-ready paper."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_openai_request_omits_temperature_and_artifacts_omit_timing():
    source = (ROOT / "ldb/backends/openai_api.py").read_text()
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create"
    ]
    assert len(calls) == 1
    assert "temperature" not in {kw.arg for kw in calls[0].keywords}

    for arm in ("luna_full", "terra_full"):
        artifact = json.loads(
            (ROOT / f"results/{arm}/surface_finance.json").read_text()
        )
        backend = artifact["config"]["backend"]
        assert backend["backend"] == "openai"
        assert backend["reasoning_effort"] == "medium"
        assert "wall_clock_s" not in backend
        assert "requested_model" not in backend


def test_open_weight_artifact_records_decoding_and_timing():
    artifact = json.loads(
        (ROOT / "results/openweight/model_finance.json").read_text()
    )
    backend = artifact["config"]["backend"]
    assert backend["model"] == "Qwen/Qwen3-32B"
    assert backend["temperature"] == 0.0
    assert backend["seed"] == 0
    assert backend["max_tokens"] == 24000
    assert backend["max_model_len"] == 32768
    assert backend["wall_clock_s"] > 0
