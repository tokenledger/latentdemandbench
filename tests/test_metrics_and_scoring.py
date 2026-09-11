"""Metric definitions, scoring contracts, and evidence-path validation."""

import json
import math

import numpy as np
import pytest

from ldb.baselines import hidden_oracle, random_rank, structured_evidence
from ldb.llm import TOP_K, build_system, score_predictions
from ldb.metrics import (aggregate, auc, evaluate, ndcg_at_k, paired_diff,
                         prob_mse, recognition_gap_auc)
from ldb.paths import _cites
from ldb.task import build_instance

SEEDS = range(20)


# ------------------------------------------------------------------ ranking metrics

def test_constant_scores_give_auc_half_not_one():
    assert auc([1, 0, 1, 0], [1.0, 1.0, 1.0, 1.0]) == 0.5


def test_p_star_is_the_ndcg_upper_bound_in_expectation():
    """Regression: nDCG was weighted by materiality, which is latent and never
    rendered, so ranking by p* was NOT an upper bound on it (0.428 vs 0.693).

    The bound is over the distribution, not per universe: nDCG scores one
    realization, so on any single draw another ranker can get lucky. Asserting
    it per universe fails on ~1 seed in 20 for that reason alone.
    """
    p_star, ev = [], []
    for s in SEEDS:
        inst = build_instance(s)
        p_star.append(ndcg_at_k(inst, hidden_oracle(inst)))
        ev.append(ndcg_at_k(inst, {k: inst.p_star[k] * inst.materiality[k]
                                   for k in inst.candidates}))
    assert np.nanmean(p_star) > np.nanmean(ev)


def test_gap_auc_is_restricted_to_unmentioned_pairs():
    """Regression: the below-median subset admitted ~76% of pairs, making the
    metric a restatement of AUC."""
    inst = build_instance(0)
    unmentioned = [k for k in inst.candidates if inst.recognized[k] == 0]
    assert 0.2 < len(unmentioned) / len(inst.candidates) < 0.7
    # a ranker with no signal there must sit at chance
    flat = {k: 0.0 for k in inst.candidates}
    assert recognition_gap_auc(inst, flat) == 0.5


def test_prob_mse_is_zero_for_the_generating_probability():
    inst = build_instance(0)
    assert prob_mse(inst, dict(inst.p_star)) == pytest.approx(0.0)


def test_evaluate_does_not_emit_a_brier_key():
    """It is MSE against latent p*, not the outcome Brier score."""
    inst = build_instance(0)
    out = evaluate(inst, structured_evidence(inst), probs=structured_evidence(inst))
    assert "prob_mse_vs_p*" in out and "brier_vs_p*" not in out


# --------------------------------------------------------------------- aggregation

def test_aggregate_uses_the_union_of_keys():
    """Regression: keying off rows[0] dropped path metrics when the first call
    failed, or raised when a later row lacked them."""
    rows = [{"auc": 0.7, "path_cap_cited": 0.5}, {"auc": 0.6}]
    out = aggregate(rows)
    assert set(out) == {"auc", "path_cap_cited"}
    assert out["auc"]["n"] == 2 and out["path_cap_cited"]["n"] == 1


def test_aggregate_key_order_is_stable():
    rows = [{"b": 1.0, "a": 2.0}, {"a": 1.0, "b": 3.0}]
    assert list(aggregate(rows)) == sorted(aggregate(rows))


def test_paired_diff_tolerates_missing_keys():
    rows_a = [{"auc": 0.7}, {}]
    m, _, _ = paired_diff(rows_a, rows_a, "auc")
    assert m == pytest.approx(0.0)


def test_artifact_is_strict_json():
    """NaN is not JSON; strict parsers reject it."""
    from run_v0 import _finite
    payload = _finite({"a": float("nan"), "b": (float("inf"), 1.0), "c": [float("nan")]})
    text = json.dumps(payload, allow_nan=False)
    json.loads(text, parse_constant=lambda x: pytest.fail(f"non-strict {x}"))
    assert payload["a"] is None and payload["b"][0] is None


# ------------------------------------------------------------- top-K contract

def _pred(key, p=0.5, path=""):
    return {"product_id": key[0], "need_id": key[1],
            "probability": p, "evidence_path": path}


def test_returning_every_candidate_still_ranks_only_top_k():
    """Regression: an unenforced cap let one model rank 177 pairs while others
    ranked 40, making their AUCs incomparable."""
    inst = build_instance(0)
    scores, _, meta = score_predictions(inst, [_pred(k) for k in inst.candidates])
    assert sum(v > 0 for v in scores.values()) == TOP_K
    assert meta["n_ranked"] == TOP_K


def test_duplicates_keep_their_best_position():
    inst = build_instance(0)
    a, b = inst.candidates[0], inst.candidates[1]
    scores, _, _ = score_predictions(inst, [_pred(a), _pred(b), _pred(a)])
    assert scores[a] > scores[b]


def test_hallucinated_pairs_are_ignored():
    inst = build_instance(0)
    scores, _, meta = score_predictions(inst, [_pred(("P999", "N999")),
                                               _pred(inst.candidates[0])])
    assert meta["n_ranked"] == 1 and ("P999", "N999") not in scores


# ---------------------------------------------------------------- evidence paths

@pytest.mark.parametrize("text,alias,want", [
    ("the C10 capability", "C1", False),   # prefix
    ("the C1 capability", "C1", True),
    ("uses Cavex-57", "Cavex", False),     # hyphen is part of the token
    ("uses Cavex-57", "Cavex-57", True),
    ("ZIRUN aligns", "Zirun", True),       # case-insensitive
])
def test_citation_matching_is_whole_token(text, alias, want):
    assert _cites(text, {alias}) is want


def test_path_metrics_reward_correct_citations_only():
    inst = build_instance(0)
    pid, nid = inst.candidates[0]
    true_cap = inst.universe.capabilities[inst.observed.prod_caps[pid][0]].name
    wrong = next(c.name for cid, c in inst.universe.capabilities.items()
                 if cid not in inst.observed.prod_caps[pid])
    _, _, good = score_predictions(inst, [_pred((pid, nid), path=true_cap)])
    _, _, bad = score_predictions(inst, [_pred((pid, nid), path=wrong)])
    assert good["path_cap_cited"] == 1.0 and good["path_cap_false"] == 0.0
    assert bad["path_cap_cited"] == 0.0 and bad["path_cap_false"] == 1.0


# ---------------------------------------------------------------------- prompt

def test_opaque_prompt_carries_no_finance_vocabulary():
    """Regression: the corpus dropped financial framing while the instructions
    still said 'sell-side commentary' and 'quarters'."""
    text = build_system("opaque")
    assert not any(w in text.lower()
                   for w in ("sell-side", "quarter", "product-need", "market"))


@pytest.mark.parametrize("horizon", [4, 8])
def test_prompt_horizon_matches_the_instance(horizon):
    """Regression: the system prompt hardcoded 8 while the task used the real
    horizon, so the two contradicted each other."""
    assert f"{horizon} quarters" in build_system("finance", horizon)


def test_ranking_baselines_are_ordered_as_expected():
    """The benchmark is only interesting if this ordering holds."""
    aucs = {}
    for name, fn in (("random", random_rank), ("evidence", structured_evidence),
                     ("hidden", hidden_oracle)):
        vals = [evaluate(i := build_instance(s), fn(i))["auc"] for s in SEEDS]
        aucs[name] = np.nanmean(vals)
    assert aucs["random"] < aucs["evidence"] < aucs["hidden"]
    assert aucs["random"] == pytest.approx(0.5, abs=0.05)


# ------------------------------------------------------- shared top-K protocol

def test_baselines_are_held_to_the_models_top_k_contract():
    """Regression: baselines scored all ~172 candidates while models returned
    40, so AUC and probability MSE partly measured truncation and an imputed
    constant rather than ranking quality."""
    from ldb.llm import UNRANKED_P, apply_top_k_protocol
    inst = build_instance(0)
    raw = hidden_oracle(inst)
    sc, pr = apply_top_k_protocol(raw, raw, TOP_K)
    assert sum(v > 0 for v in sc.values()) == TOP_K
    assert sum(v == UNRANKED_P for v in pr.values()) == len(inst.candidates) - TOP_K


def test_oracle_under_the_protocol_is_not_perfectly_calibrated():
    """The ceiling is the ceiling *as the task is posed*: even a perfect top-K
    answer imputes the rest, so nonzero MSE is correct rather than a defect."""
    from ldb.llm import apply_top_k_protocol
    inst = build_instance(0)
    raw = hidden_oracle(inst)
    _, pr = apply_top_k_protocol(raw, raw, TOP_K)
    assert 0.0 < prob_mse(inst, pr) < 0.05


def test_predictions_are_returned_for_archival():
    """Regression: artifacts stored only derived metrics, so published results
    could not be re-scored without the local response cache."""
    inst = build_instance(0)
    key = inst.candidates[0]
    _, _, meta = score_predictions(inst, [_pred(key, p=0.9, path="because X")])
    got = meta["predictions"][0]
    assert got["product_id"] == key[0] and got["need_id"] == key[1]
    assert got["probability"] == 0.9 and got["evidence_path"] == "because X"
    # the four idea.md outputs are archived too, even when absent from a response
    assert {"materiality", "capture", "recognized", "prerequisites"} <= set(got)


def test_metric_rows_reject_non_numeric_payloads():
    """Regression: the verbatim model response leaked into a metric row and the
    bootstrap tried to call isnan() on a string."""
    from ldb.metrics import bootstrap_ci
    with pytest.raises(TypeError, match="non-numeric"):
        bootstrap_ci([0.5, "{\"predictions\": []}"])
