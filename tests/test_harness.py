"""Backend plumbing and the guards that keep bad artifacts out of the paper.

No test here calls a real model; the point is that the harness fails loudly
rather than emitting something that reads like a legitimate result.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import make_tables
from ldb.backends import (InfrastructureError, backend_names, cached,
                          get_backend, with_backoff)

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------- backends

@pytest.mark.parametrize("name", backend_names())
def test_every_backend_declares_its_configuration(name):
    d = get_backend(name).describe(None, None)
    assert d["model"] and "isolated" in d


@pytest.mark.parametrize("name", backend_names())
def test_decorators_preserve_the_backend_module(name):
    """Regression: with_backoff lacked functools.wraps, so cached() looked for
    describe() in the wrong module and EVERY model call raised - which the
    runner then scored as a chance-level result."""
    assert get_backend(name).complete.__module__ == f"ldb.backends.{ _mod(name) }"


def _mod(name):
    return {"codex": "codex_cli", "anthropic": "anthropic_api",
            "gemini": "gemini_api", "openai": "openai_api"}[name]


def test_only_tool_free_backends_are_marked_isolated():
    assert get_backend("codex").describe()["isolated"] is False
    for n in ("gemini", "anthropic", "openai"):
        assert get_backend(n).describe()["isolated"] is True


def test_missing_describe_is_an_infrastructure_error_not_a_model_failure():
    import types
    mod = types.ModuleType("ldb.backends.broken")
    sys.modules["ldb.backends.broken"] = mod

    def fn(*, system, task, schema, model=None, effort=None):
        return "{}"
    fn.__module__ = "ldb.backends.broken"
    with pytest.raises(InfrastructureError):
        cached(fn)(system="s", task="t", schema={}, model=None, effort=None)


def test_backoff_retries_transient_errors_and_reraises_others(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 rate limit exceeded")
        return "ok"
    assert with_backoff(flaky, base=0.0)(x=1) == "ok" and calls["n"] == 3

    def fatal(**kw):
        raise ValueError("schema is wrong")
    with pytest.raises(ValueError):
        with_backoff(fatal, base=0.0)(x=1)


def test_cache_refuses_to_store_unparseable_responses(tmp_path, monkeypatch):
    """Regression: a truncated response was cached verbatim and then failed
    identically on every retry, permanently poisoning that universe."""
    monkeypatch.setattr("ldb.backends.CACHE_DIR", str(tmp_path))
    import types
    mod = types.ModuleType("ldb.backends.trunc")
    mod.describe = lambda m=None, e=None: {"backend": "t", "model": "m"}
    sys.modules["ldb.backends.trunc"] = mod

    def fn(*, system, task, schema, model=None, effort=None):
        return '{"predictions": [' # truncated
    fn.__module__ = "ldb.backends.trunc"
    with pytest.raises(json.JSONDecodeError):
        cached(fn)(system="s", task="t", schema={}, model=None, effort=None)
    assert not list(tmp_path.glob("*.json")), "invalid response was cached"


# --------------------------------------------------------------- table guards

def _artifact(tmp_path, name="a.json", **cfg):
    """A fixture that is current unless the test deliberately makes it stale, so
    each test exercises its own guard rather than whichever fires first."""
    sys.path.insert(0, str(ROOT))
    from run_v0 import _code_hash
    base = json.load(open(ROOT / "results" / "baselines.json"))
    base["config"]["code_hash"] = _code_hash()
    base["config"].update(cfg)
    p = tmp_path / name
    json.dump(base, open(p, "w"))
    return p


def _run_tables(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "make_tables.py", "--out", str(tmp_path / "t.tex"), *args],
        capture_output=True, text=True, cwd=ROOT)


def test_current_artifacts_build_tables(tmp_path):
    r = _run_tables(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


def test_stale_artifacts_are_refused(tmp_path):
    """Regression: tables were built from artifacts whose source no longer
    existed, so published numbers came from unavailable code."""
    p = _artifact(tmp_path, code_hash="deadbeef")
    r = _run_tables(tmp_path, "--main", str(p))
    assert r.returncode != 0 and "predate current source" in r.stdout + r.stderr


def test_unisolated_backend_is_refused(tmp_path):
    p = _artifact(tmp_path, backend={"backend": "codex", "isolated": False,
                                     "isolation_note": "fs access"})
    r = _run_tables(tmp_path, "--main", str(p))
    assert r.returncode != 0 and "not isolated" in r.stdout + r.stderr


def test_high_failure_rate_is_refused(tmp_path):
    """Regression: an all-failure arm scores 0.500 and would otherwise be
    published as a legitimate n=30 model row."""
    p = _artifact(tmp_path, llm_seeds=[0, 1, 2], failures=[{"seed": 0}, {"seed": 1}])
    r = _run_tables(tmp_path, "--main", str(p))
    assert r.returncode != 0 and "failed" in r.stdout + r.stderr


def test_missing_required_artifact_is_refused(tmp_path):
    r = _run_tables(tmp_path, "--twins", str(tmp_path / "nope.json"))
    assert r.returncode != 0 and "missing required artifacts" in r.stdout + r.stderr


def test_null_metric_cells_render_as_dashes():
    """Strict-JSON artifacts carry null where a metric was NaN."""
    assert make_tables.cell({"mean": None, "lo": None, "hi": None}) == "--"
    assert make_tables.cell({"mean": 0.5, "lo": None, "hi": None}) == "0.500"


def test_ladder_rejects_a_rung_whose_recorded_effort_disagrees(tmp_path):
    """Regression: rungs were labelled from the filename, so a mislabelled file
    would be published under the wrong effort."""
    for eff in ("low", "medium", "high", "max"):
        _artifact(tmp_path, f"L_{eff}.json",
                  backend={"backend": "gemini", "model": "m", "isolated": True,
                           "effort": "medium"},  # every rung claims medium
                  llm_seeds=[0], surface="finance", assist="none")
    r = _run_tables(tmp_path, "--ladder", str(tmp_path / "L_{}.json"))
    assert r.returncode != 0 and "effort" in r.stdout + r.stderr


def test_partial_ladder_is_refused(tmp_path):
    _artifact(tmp_path, "L_low.json",
              backend={"backend": "gemini", "model": "m", "isolated": True,
                       "effort": "low"}, llm_seeds=[0])
    r = _run_tables(tmp_path, "--ladder", str(tmp_path / "L_{}.json"))
    assert r.returncode != 0 and "partial" in (r.stdout + r.stderr).lower()


def test_generated_tables_match_reference_digests(tmp_path):
    """Reproduce all tables and numerical macros from the included artifacts."""
    import hashlib

    expected = {'tables.tex': '3f224b6e81d7c2f2238b4c6df6f5747616b0c9e4d854165e9a28ab75e4687c28', 'tables_appendix.tex': 'ae360ead78b96d7ba4f31a1e311628d29c1a534a433c8a7134f1473bad5f02d3', 'numbers.tex': '99695eea5eff23772695681347502a3bdaacb0fb8d18972c1db977022943bf49'}
    r = subprocess.run(
        [sys.executable, "tools/paper_tables.py",
         "--out", str(tmp_path / "tables.tex"),
         "--appendix", str(tmp_path / "tables_appendix.tex"),
         "--numbers", str(tmp_path / "numbers.tex")],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    for name, digest in expected.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest, name
