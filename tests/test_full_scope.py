"""The parts of idea.md beyond ranking: value capture, materiality, recognition
prediction, unresolved prerequisites, and the five narrative surfaces."""

import numpy as np
import pytest

from ldb.baselines import BASELINES, expected_value_rank, hidden_oracle
from ldb.llm import LEXICON, build_system, score_predictions
from ldb.metrics import capture_auc, ndcg_ev_at_k, pair_auc
from ldb.render import SURFACES
from ldb.task import build_instance

SEEDS = range(20)


# ----------------------------------------------------------- value capture

def test_capture_is_separable_from_activation():
    """idea.md's third question: a pairing can be served without the holder
    keeping the value. If capture tracked activation it would be redundant."""
    p_true, p_cap = [], []
    for s in SEEDS:
        u = build_instance(s).universe
        for pr in u.pairs.values():
            p_true.append(pr.p_true)
            p_cap.append(pr.p_capture)
    assert abs(np.corrcoef(p_true, p_cap)[0, 1]) < 0.2


def test_capture_depends_on_company_factors_only():
    inst = build_instance(0)
    u = inst.universe
    for (pid, _), pr in list(u.pairs.items())[:20]:
        p = u.products[pid]
        expected = min(1.0, 0.15 + 0.55 * p.supply * p.distribution + 0.30 * p.moat)
        assert pr.p_capture == pytest.approx(expected)


def test_capture_auc_separates_captured_pairings():
    aucs = [capture_auc(i := build_instance(s), hidden_oracle(i)) for s in SEEDS]
    assert np.nanmean(aucs) > 0.6


# ------------------------------------------------- the four questions differ

def test_different_questions_have_different_best_rankers():
    """Ranking by p* wins on realization; ranking by expected value wins when
    materiality and capture are what matter. If one ranker won both, the extra
    questions would carry no information."""
    ev_on_ev, ps_on_ev, ev_on_auc, ps_on_auc = [], [], [], []
    for s in SEEDS:
        i = build_instance(s)
        ev, ps = expected_value_rank(i), hidden_oracle(i)
        ev_on_ev.append(ndcg_ev_at_k(i, ev)); ps_on_ev.append(ndcg_ev_at_k(i, ps))
        ev_on_auc.append(pair_auc(i, ev)); ps_on_auc.append(pair_auc(i, ps))
    assert np.nanmean(ev_on_ev) > np.nanmean(ps_on_ev)
    assert np.nanmean(ps_on_auc) > np.nanmean(ev_on_auc)


# ------------------------------------------------------------- observability

def test_materiality_and_company_factors_are_rendered():
    """Regression: nDCG was weighted by latent materiality. Anything scored must
    be visible in the dossier."""
    inst = build_instance(0)
    o = inst.observed
    assert o.need_materiality and o.prod_supply and o.prod_moat
    assert f"{o.need_materiality['N0']:.2f}" in inst.corpus
    assert f"{o.prod_supply['P0']:.2f}" in inst.corpus


# ----------------------------------------------------------------- surfaces

def test_all_five_surfaces_exist_and_share_ground_truth():
    assert len(SURFACES) == 5
    base = build_instance(3, surface="finance")
    for name in SURFACES:
        other = build_instance(3, surface=name)
        assert other.label == base.label and other.p_star == base.p_star
        assert 0.8 < len(other.corpus) / len(base.corpus) < 1.25


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_every_surface_has_matching_prompt_vocabulary(surface):
    """A surface whose prompt still spoke finance would confound the comparison."""
    assert surface in LEXICON
    text = build_system(surface).lower()
    if surface != "finance":
        assert LEXICON[surface]["unit"] in text
        assert "sell-side" not in text


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_baselines_are_invariant_across_every_surface(surface):
    a = BASELINES["structured_evidence"](build_instance(3, surface="finance"))
    b = BASELINES["structured_evidence"](build_instance(3, surface=surface))
    assert a == b


# -------------------------------------------------------- elicited outputs

def _perfect(inst, key):
    o = inst.observed
    return {"product_id": key[0], "need_id": key[1], "probability": inst.p_star[key],
            "materiality": inst.materiality[key],
            "capture": inst.universe.pairs[key].p_capture,
            "recognized": inst.recognized[key] > 0,
            "prerequisites": [k for k in o.need_comps[key[1]]
                              if o.comp_ready[k] > o.cutoff + o.horizon],
            "evidence_path": ""}


def test_perfect_elicitation_scores_perfectly():
    from ldb.paths import elicited_metrics
    inst = build_instance(0)
    m = elicited_metrics(inst, [(k, _perfect(inst, k)) for k in inst.probes])
    assert m["capture_mae"] == pytest.approx(0.0, abs=1e-9)
    assert m["recognition_acc"] == 1.0
    assert m["prereq_f1"] == pytest.approx(1.0)
    assert m["materiality_rho"] > 0.99


def test_wrong_elicitation_is_penalised():
    from ldb.paths import elicited_metrics
    inst = build_instance(0)
    wrong = []
    for k in inst.probes:
        p = _perfect(inst, k)
        p["capture"] = 1.0 - p["capture"]
        p["recognized"] = not p["recognized"]
        p["prerequisites"] = ["K999"]
        wrong.append((k, p))
    m = elicited_metrics(inst, wrong)
    assert m["capture_mae"] > 0.1
    assert m["recognition_acc"] < 0.5
    assert m["prereq_f1"] < 0.5


# ------------------------------------------- review round 11 regressions

def test_probe_set_is_stratified_and_fixed():
    """Regression: auxiliary metrics were scored on each model's self-selected
    top-40, which contained almost no blocked pairs - so always answering
    `prerequisites: []` scored a perfect 1.0."""
    from ldb.compose import observed_ready
    inst = build_instance(0)
    o = inst.observed
    cells = {}
    for p, n in inst.probes:
        cells.setdefault((observed_ready(o, n), o.mention_counts[(p, n)] > 0), 0)
        cells[(observed_ready(o, n), o.mention_counts[(p, n)] > 0)] += 1
    assert len(cells) == 4 and all(v >= 2 for v in cells.values())
    assert inst.probes == build_instance(0).probes  # deterministic


def test_empty_prerequisites_no_longer_scores_perfectly():
    from ldb.paths import elicited_metrics
    inst = build_instance(0)
    lazy = [(k, {"prerequisites": [], "capture": 0.5, "recognized": False,
                 "materiality": 1.0}) for k in inst.probes]
    assert elicited_metrics(inst, lazy)["prereq_f1"] < 0.9


def test_value_metrics_are_graded_on_the_expected_value_order():
    """Regression: one ranking was graded against realization AND value-weighted
    objectives, which cannot be jointly optimised."""
    from ldb.metrics import evaluate
    inst = build_instance(0)
    probs = hidden_oracle(inst)
    ev = {k: probs[k] * inst.materiality[k] for k in inst.candidates}
    assert (evaluate(inst, probs, ev_scores=ev)["ndcg_ev@10"]
            != evaluate(inst, probs)["ndcg_ev@10"])


def test_twin_percentiles_use_midranks():
    """Regression: the large tied unranked group all scored percentile 0."""
    from ldb.twins import build_twin, direction_score
    tp = next(t for s in range(10) if (t := build_twin(s)))
    flat = {k: 0.0 for k in tp.blocked.candidates}
    out = direction_score(tp, flat, dict(flat))
    assert out["direction_acc"] == 0.5 and out["mean_delta_pct"] == 0.0


def test_pruner_refuses_to_delete_without_explicit_confirmation():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "tools/prune_cache.py", "--delete"],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode != 0 and "refusing to delete" in r.stdout + r.stderr
