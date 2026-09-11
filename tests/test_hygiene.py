"""Guards against leftovers: dead references, uncollected metrics, and
artifacts or cache entries that no longer correspond to the current code.

Ten rounds of refactoring left several of these behind, and none of them are
visible from a passing functional suite.
"""

import ast
import glob
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted(ROOT.glob("ldb/**/*.py")) + sorted(ROOT.glob("*.py"))

# names removed during refactors; a live reference means something was missed
REMOVED = ["brier_vs_p*", "_git_rev", "_UNRANKED_P", "code_version",
           "evidence_oracle", "path_metrics_v1"]


def _text(paths):
    return {p: p.read_text() for p in paths}


def test_no_references_to_removed_names():
    offenders = []
    for path, body in _text(SOURCES).items():
        for name in REMOVED:
            for i, line in enumerate(body.splitlines(), 1):
                # a string mentioning the old name for back-compat is fine;
                # an identifier reference is not
                if name in line and "stale naming" not in line and '"' not in line:
                    offenders.append(f"{path.name}:{i} {name}")
    assert not offenders, offenders


def test_no_todo_markers_left():
    offenders = [f"{p.name}: {line.strip()[:60]}"
                 for p, body in _text(SOURCES).items()
                 for line in body.splitlines()
                 if re.search(r"\b(TODO|FIXME|XXX|HACK)\b", line)]
    assert not offenders, offenders


def test_every_computed_metric_is_reported_somewhere():
    """A metric that lands in every artifact but appears in no table is cruft:
    it costs storage and implies a claim nobody checks."""
    import run_v0
    from ldb.baselines import structured_evidence
    from ldb.metrics import evaluate
    from ldb.task import build_instance

    inst = build_instance(0)
    scores = structured_evidence(inst)
    produced = set(evaluate(inst, scores, probs=scores))
    reported = set(run_v0.COLS) | set(run_v0.PATH_COLS) | set(run_v0.ELICITED_COLS)
    assert not produced - reported, f"computed but never shown: {produced - reported}"


def test_no_unused_imports_in_package():
    offenders = []
    for path in ROOT.glob("ldb/**/*.py"):
        tree = ast.parse(path.read_text())
        body = path.read_text()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = (alias.asname or alias.name).split(".")[0]
                    if name == "*":
                        continue
                    uses = len(re.findall(rf"\b{re.escape(name)}\b", body))
                    if uses <= 1:
                        offenders.append(f"{path.name}:{node.lineno} {name}")
    assert not offenders, offenders


def test_requirements_agree_with_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text()
    reqs = (ROOT / "requirements.txt").read_text().splitlines()
    for line in [r for r in reqs if r.strip()]:
        pkg = re.split(r"[<>=]", line)[0].strip()
        assert pkg in pyproject, f"{pkg} in requirements.txt but not pyproject.toml"


def test_published_results_all_match_current_source():
    """results/ is what gets published; results/stale/ is the archive."""
    from run_v0 import _code_hash
    current = _code_hash()
    bad = []
    for f in glob.glob(str(ROOT / "results" / "*.json")):
        cfg = json.load(open(f)).get("config") or {}
        if cfg.get("code_hash") and cfg["code_hash"] != current:
            bad.append(Path(f).name)
    assert not bad, f"stale artifacts left in results/: {bad} (move to results/stale/)"


def test_orphaned_cache_entries_can_be_identified(tmp_path):
    """Changing the prompt orphans every cached response. The pruner must find
    them, and must derive keys through the real code path so it cannot drift."""
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.prune_cache import prune

    (tmp_path / "live.json").write_text("{}")
    (tmp_path / "dead.json").write_text("{}")
    total, dead = prune(str(tmp_path), {"live"}, dry_run=True)
    assert (total, dead) == (2, 1)
    assert len(list(tmp_path.glob("*.json"))) == 2, "dry run must not delete"

    prune(str(tmp_path), {"live"}, dry_run=False)
    assert [f.name for f in tmp_path.glob("*.json")] == ["live.json"]


def test_code_hash_is_captured_before_the_run(tmp_path):
    """Regression (P0): the hash was computed at write time, certifying whatever
    was on disk when a long run finished rather than the code that produced the
    results. An artifact shipped claiming a hash it was not built with."""
    src = (ROOT / "run_v0.py").read_text()
    assert "code_hash_at_start = _code_hash()" in src
    assert '"code_hash": code_hash_at_start' in src
    # and it must refuse to write if the source moved underneath it
    assert "source changed during the run" in src


def test_billing_faults_are_not_scored_as_model_failures():
    """Regression: depleted-credit errors became zero rankings, understating
    the model and biasing any row that tolerated a nonzero failure rate."""
    from run_v0 import _is_billing_fault
    assert _is_billing_fault("429 RESOURCE_EXHAUSTED prepayment credit")
    assert _is_billing_fault("AuthenticationError: invalid api key")
    assert not _is_billing_fault("truncated: thinking used 45760 tokens")


def test_replicates_are_distinct_cache_entries():
    """Regression: --replicates repeated identical calls that were served from
    cache, so they averaged nothing."""
    import hashlib
    import json as _json
    seen = set()
    for r in (0, 1, 2):
        seen.add(hashlib.sha256(_json.dumps(
            ["m", "sys", "task", {}, {"model": "x"}, r], sort_keys=True
        ).encode()).hexdigest()[:32])
    assert len(seen) == 3
    src = (ROOT / "ldb" / "backends" / "__init__.py").read_text()
    assert "replicate" in src and "resolved, replicate" in src


def test_pruner_and_writer_agree_on_cache_keys(tmp_path, monkeypatch):
    """Regression (P0): the pruner reimplemented key derivation, then fell out
    of step when `replicate` was added to the writer's key - reporting every
    live response as dead, one --delete from erasing paid work.

    This writes through the real cache and asserts the pruner considers that
    exact entry live, so the two can never drift again.
    """
    import sys
    import types
    sys.path.insert(0, str(ROOT))
    import ldb.backends as be
    from tools.prune_cache import prune

    monkeypatch.setattr(be, "CACHE_DIR", str(tmp_path))
    mod = types.ModuleType("ldb.backends.probe")
    mod.describe = lambda m=None, e=None: {"backend": "probe", "model": "m"}
    sys.modules["ldb.backends.probe"] = mod

    def fn(*, system, task, schema, model=None, effort=None):
        return '{"ok": 1}'
    fn.__module__ = "ldb.backends.probe"

    for rep in (0, 1):
        be.cached(fn)(system="S", task="T", schema={}, model=None,
                      effort=None, replicate=rep)
    assert len(list(tmp_path.glob("*.json"))) == 2

    live = {be.cache_key("ldb.backends.probe", "S", "T", {},
                         mod.describe(), r) for r in (0, 1)}
    total, dead = prune(str(tmp_path), live, dry_run=True)
    assert (total, dead) == (2, 0), "writer's own entries reported as dead"
