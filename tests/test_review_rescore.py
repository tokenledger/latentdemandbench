"""Offline migration preserves paid responses and distinguishes provenance."""

import json
from pathlib import Path

import pytest

from run_v0 import _code_hash
from tools import rescore_review

ROOT = Path(__file__).resolve().parents[1]


def small_artifact(tmp_path, name="surface_finance"):
    art = json.loads((ROOT / f"results/terra_full/{name}.json").read_text())
    art["config"]["universe_seeds"] = [0]
    art["config"]["llm_seeds"] = [0]
    art["model_outputs"] = {"0": art["model_outputs"]["0"]}
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(art))
    return path, art


def test_offline_rescore_preserves_raw_responses_and_original_provenance(tmp_path):
    path, original = small_artifact(tmp_path, "replicates_3")
    before = path.read_bytes()
    result = rescore_review.rescore(path)
    assert path.read_bytes() == before  # dry run
    assert rescore_review.response_digest(result) == rescore_review.response_digest(original)
    assert result["config"]["generation"] == original["config"]["generation"]
    assert result["config"]["generation"]["code_hash"] == rescore_review.GENERATION_HASH
    assert result["config"]["code_hash"] == _code_hash()
    model = next(k for k in result["per_universe"] if k.startswith("openai:"))
    assert result["per_universe"][model][0]["n_ranked"] == 40


def test_rescore_refuses_source_change_before_overwriting(tmp_path, monkeypatch):
    path, _ = small_artifact(tmp_path)
    before = path.read_bytes()
    hashes = iter(["before", "after"])
    monkeypatch.setattr(rescore_review, "_code_hash", lambda: next(hashes))
    with pytest.raises(ValueError, match="source changed"):
        rescore_review.rescore(path, write=True)
    assert path.read_bytes() == before


def test_rescore_refuses_unsupported_generation(tmp_path):
    path, art = small_artifact(tmp_path)
    art["config"]["generation"]["code_hash"] = "unknown"
    path.write_text(json.dumps(art))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported generation"):
        rescore_review.rescore(path, write=True)
    assert path.read_bytes() == before


def test_migration_cli_keeps_current_native_reruns_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "native.json"
    path.write_text(json.dumps({"config": {"code_hash": _code_hash()}}))
    before = path.read_bytes()
    monkeypatch.setattr(rescore_review, "ROOT", tmp_path)
    monkeypatch.setattr(rescore_review, "verify_generation_contract", lambda: None)
    monkeypatch.setattr("sys.argv", ["rescore_review.py", "--write", "--paths", "native.json"])
    rescore_review.main()
    assert path.read_bytes() == before


def test_paper_refuses_an_arm_marked_for_model_rerun(tmp_path):
    from tools.paper_tables import check

    path, art = small_artifact(tmp_path)
    art["config"]["requires_model_rerun"] = "schema capped at 40 instead of 90"
    path.write_text(json.dumps(art))
    with pytest.raises(SystemExit, match="rerun"):
        check(str(path), art, _code_hash())

    from tools.figures import load
    with pytest.raises(SystemExit, match="rerun"):
        load(path)


def test_transport_workaround_changes_only_compression_and_restores_client(monkeypatch):
    from tools import run_uncompressed

    prior = object()
    monkeypatch.setattr(run_uncompressed.openai_api, "_CLIENT", prior)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"default_headers": {"Accept-Encoding": "identity"}}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def runner():
        assert isinstance(run_uncompressed.openai_api._CLIENT, Client)
        raise RuntimeError("runner error")

    monkeypatch.setattr(run_uncompressed.openai, "OpenAI", Client)
    monkeypatch.setattr(run_uncompressed.run_v0, "main", runner)
    with pytest.raises(RuntimeError, match="runner error"):
        run_uncompressed.main()
    assert run_uncompressed.openai_api._CLIENT is prior


@pytest.mark.parametrize("arm", ["luna_full", "terra_full"])
def test_corrected_large_artifacts_match_archived_responses(arm):
    import math

    from ldb.llm import score_response
    from ldb.metrics import evaluate
    from ldb.task import build_instance
    from tools.paper_tables import model_key

    art = json.loads((ROOT / f"results/{arm}/size_large.json").read_text())
    cfg = art["config"]
    assert cfg["top_k"] == cfg["review_rerun"]["response_schema_max_items"] == 90
    assert not cfg.get("requires_model_rerun") and not cfg["failures"]
    assert cfg["llm_seeds"] == list(range(30))
    lengths = []
    for idx, seed in enumerate(cfg["llm_seeds"]):
        raw = art["model_outputs"][str(idx)]["raw_response"]
        lengths.append(len(json.loads(raw)["predictions"]))
        inst = build_instance(seed, n_products=18, n_needs=22)
        scores, probs, meta = score_response(inst, raw, 90)
        expected = evaluate(inst, scores, probs, meta["ev_scores"],
                            meta["cap_scores"], meta["probe_scores"])
        stored = art["per_universe"][model_key(art)][idx]
        assert stored["n_ranked"] == meta["n_ranked"] <= 90
        for metric in ("auc", "ndcg_ev@10", "capture_auc", "probe_auc"):
            if math.isnan(expected[metric]):
                assert stored[metric] is None
            else:
                assert stored[metric] == pytest.approx(expected[metric])
    assert 40 < max(lengths) <= 90  # rules out the original capped artifacts
