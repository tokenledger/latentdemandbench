"""Behavioral regressions from the September 2026 full-repository review."""

import pytest

from ldb.llm import UNRANKED_P, average_replicates


def test_ensemble_has_one_final_budget_and_preserves_ties():
    # Each completion obeys K=2, but their union contains four pairs.
    first = ({"a": 2., "b": 1., "c": 0., "d": 0.},
             {"a": .9, "b": .7, "c": .02, "d": .02},
             {"predictions": ["a", "b"], "raw_response": "first"})
    second = ({"a": 0., "b": 0., "c": 2., "d": 1.},
              {"a": .02, "b": .02, "c": .8, "d": .6},
              {"predictions": ["c", "d"], "raw_response": "second"})
    scores, probs, meta = average_replicates([first, second], 2)
    assert scores == {"a": 1., "b": 0., "c": 1., "d": 0.}
    assert probs["a"] == pytest.approx(.46)
    assert probs["b"] == probs["d"] == UNRANKED_P
    assert meta["n_ranked"] == 2
    assert meta["raw_response"] == ["first", "second"]
    assert meta["predictions"] == [["a", "b"], ["c", "d"]]


def test_ensemble_does_not_promote_pairs_no_completion_returned():
    rep = ({"a": 1., "b": 0.}, {"a": .9, "b": UNRANKED_P}, {})
    scores, _, meta = average_replicates([rep, rep], 40)
    assert scores == {"a": 1., "b": 0.}
    assert meta["n_ranked"] == 1


@pytest.mark.parametrize("top_k", [25, 40, 90])
def test_backend_receives_the_requested_schema_budget(monkeypatch, top_k):
    import json
    from types import SimpleNamespace

    from ldb.backends import _validate
    from ldb.llm import SCHEMA, llm_rank
    from ldb.task import build_instance

    inst = build_instance(0, n_products=18, n_needs=22)

    def complete(**request):
        assert request["schema"]["properties"]["predictions"]["maxItems"] == top_k
        assert f"Rank the {top_k} pairs" in request["task"]
        predictions = [dict(product_id=p, need_id=n, probability=.5,
                            materiality=1., capture=.5, recognized=False,
                            prerequisites=[], evidence_path="")
                       for p, n in inst.candidates[:top_k]]
        response = {"predictions": predictions, "probes": []}
        _validate(response, request["schema"])
        with pytest.raises(ValueError, match="maxItems"):
            _validate({**response, "predictions": predictions + predictions[:1]},
                      request["schema"])
        return json.dumps(response)

    monkeypatch.setattr("ldb.llm.get_backend", lambda _: SimpleNamespace(complete=complete))
    scores, _, meta = llm_rank(inst, top_k=top_k)
    assert sum(v > 0 for v in scores.values()) == top_k
    assert meta["n_ranked"] == top_k
    assert SCHEMA["properties"]["predictions"]["maxItems"] == 40


@pytest.mark.parametrize("transport", ["shared", "modal", "tokenrouter"])
def test_probe_auc_uses_probe_probabilities_without_main_ranking(transport):
    import json
    from ldb.llm import score_response
    from ldb.metrics import evaluate
    from ldb.task import build_instance
    from tools.run_open_weight import rank_from_raw as modal_score
    from tools.run_tokenrouter import rank_from_raw as router_score

    inst = build_instance(0)
    scorer = {"shared": score_response, "modal": modal_score,
              "tokenrouter": router_score}[transport]
    probes = [dict(product_id=p, need_id=n, probability=inst.label[p, n],
                   materiality=1., capture=.5, recognized=False,
                   prerequisites=[], evidence_path="") for p, n in inst.probes]
    raw = json.dumps({"predictions": [], "probes": probes})
    scores, probs, meta = scorer(inst, raw)
    assert all(v == 0 for v in scores.values())
    assert evaluate(inst, scores, probs, probe_scores=meta["probe_scores"])["probe_auc"] == 1.
    assert evaluate(inst, scores, probs)["probe_auc"] == .5


def test_value_metrics_use_separate_orders_and_only_validated_shortlist():
    import json
    from ldb.llm import score_response
    from ldb.task import build_instance

    inst = build_instance(0)
    a, b, c = inst.candidates[:3]

    def pred(key, materiality, capture):
        return dict(product_id=key[0], need_id=key[1], probability=.5,
                    materiality=materiality, capture=capture,
                    recognized=False, prerequisites=[], evidence_path="")

    raw = json.dumps({"predictions": [pred(a, 10, .1), pred(b, 1, .9),
                                     pred(a, 100, 1), pred(c, 100, 1)], "probes": []})
    _, _, meta = score_response(inst, raw, top_k=2)
    assert meta["ev_scores"][a] == 5.
    assert meta["ev_scores"][b] == .5
    assert meta["cap_scores"][a] == .05
    assert meta["cap_scores"][b] == .45
    assert meta["ev_scores"][c] == meta["cap_scores"][c] == 0.


def test_ev_baseline_uses_objective_specific_orders_on_its_shortlist():
    from ldb.baselines import expected_value_rank
    from ldb.compose import evidence_score, observed_capture
    from ldb.task import build_instance
    from run_v0 import _value_orders

    inst = build_instance(0)
    raw = expected_value_rank(inst)
    selected = set(sorted(raw, key=lambda k: -raw[k])[:10])
    ev, cap = _value_orders(inst, raw, "ev", 10)
    for key in inst.candidates:
        probability = evidence_score(inst.observed, *key) if key in selected else 0.
        assert ev[key] == pytest.approx(probability * inst.observed.need_materiality[key[1]])
        assert cap[key] == pytest.approx(probability * observed_capture(inst.observed, key[0]))


def test_observation_decomposition_holds_formula_and_labels_fixed():
    from ldb.task import build_instance
    from tools.observation_loss import decompose, latent_inputs

    inst = build_instance(0)
    corpus = inst.corpus
    latent = latent_inputs(inst)
    assert latent.label is inst.label
    assert latent.candidates is inst.candidates
    assert latent.observed is not inst.observed
    row = decompose(inst)
    assert row["observation_loss"] + row["formula_difference"] == pytest.approx(row["total_gap"])
    # With latent inputs already supplied, the observation term vanishes but
    # the exact conditional probability can still differ from the reference.
    assert decompose(latent)["observation_loss"] == 0.
    assert inst.corpus == corpus


def test_materiality_correlation_uses_average_ties_and_is_order_invariant():
    import numpy as np
    from ldb.metrics import average_ranks
    from ldb.paths import elicited_metrics
    from ldb.task import build_instance

    assert average_ranks([2, 1, 2, 3]).tolist() == [2.5, 1., 2.5, 4.]
    inst = build_instance(0)
    answers = [(k, {"materiality": round(inst.materiality[k]), "capture": .5,
                    "recognized": False, "prerequisites": []}) for k in inst.probes]
    expected = float(np.corrcoef(
        average_ranks([p["materiality"] for _, p in answers]),
        average_ranks([inst.materiality[k] for k, _ in answers]))[0, 1])
    for seed in range(10):
        np.random.default_rng(seed).shuffle(answers)
        assert elicited_metrics(inst, answers)["materiality_rho"] == pytest.approx(expected)


def test_twin_runner_refuses_changed_source_before_touching_output(tmp_path, monkeypatch):
    import run_twins
    from ldb.twins import build_twin

    twin = next(t for s in range(10) if (t := build_twin(s)) is not None)
    hashes = iter(["start", "changed"])
    monkeypatch.setattr(run_twins, "_code_hash", lambda: next(hashes))
    monkeypatch.setattr(run_twins, "build_twin", lambda *a, **kw: twin)
    monkeypatch.setattr(run_twins, "BASELINES", {
        "flat": lambda inst, seed=0: {k: 0. for k in inst.candidates}})
    out = tmp_path / "twins.json"
    out.write_text("preserve existing artifact")
    monkeypatch.setattr("sys.argv", ["run_twins.py", "--seeds", "1", "--out", str(out)])
    with pytest.raises(SystemExit, match="source changed during the run"):
        run_twins.main()
    assert out.read_text() == "preserve existing artifact"


def test_cache_retention_covers_sweeps_and_historical_schema():
    from ldb.backends import cache_key, get_backend
    from ldb.llm import SCHEMA, build_system, build_task, response_schema
    from ldb.task import build_instance
    from tools.prune_cache import live_keys

    live = live_keys("openai", range(1), replicates=3)
    resolved = get_backend("openai").describe()
    configs = [({"horizon": 4}, 40), ({"horizon": 12}, 40),
               ({"slope": 6}, 40), ({"slope": 14}, 40),
               ({"n_products": 8, "n_needs": 10}, 25),
               ({"n_products": 18, "n_needs": 22}, 90)]
    for config, k in configs:
        inst = build_instance(0, **config)
        for schema in (SCHEMA, response_schema(k)):
            for replicate in range(3):
                key = cache_key("ldb.backends.openai_api", build_system(inst.surface, inst.horizon),
                                build_task(inst, k), schema, resolved, replicate)
                assert key in live, (config, k, replicate)


def test_unlisted_cache_entries_are_recoverable(tmp_path):
    from tools.prune_cache import prune

    (tmp_path / "keep.json").write_text("paid response one")
    (tmp_path / "unknown.json").write_text("paid response two")
    assert prune(str(tmp_path), {"keep"}, dry_run=False) == (2, 1)
    archived = list((tmp_path / "quarantine").glob("*/unknown.json"))
    assert len(archived) == 1
    assert archived[0].read_text() == "paid response two"
    assert (tmp_path / "keep.json").read_text() == "paid response one"


def test_all_replicates_contribute_to_value_probe_and_path_metrics():
    first = ({"a": 2., "b": 1., "c": 0.}, {"a": .8, "b": .6, "c": .02},
             {"ev_scores": {"a": 2., "b": 1., "c": 0.},
              "cap_scores": {"a": .1, "b": .2, "c": 0.},
              "probe_scores": {"a": .2, "c": .8},
              "path_cap_cited": 0., "n_returned": 2, "n_ranked": 2})
    second = ({"a": 1., "b": 0., "c": 2.}, {"a": .6, "b": .02, "c": .8},
              {"ev_scores": {"a": 6., "b": 0., "c": 10.},
               "cap_scores": {"a": .9, "b": 0., "c": .8},
               "probe_scores": {"a": .8, "c": .2},
               "path_cap_cited": 1., "n_returned": 4, "n_ranked": 2})
    _, _, meta = average_replicates([first, second], 1)
    assert meta["ev_scores"] == {"a": 4., "b": 0., "c": 0.}
    assert meta["cap_scores"] == {"a": .5, "b": 0., "c": 0.}
    # Probes stay untruncated even when the ensemble shortlist has only one pair.
    assert meta["probe_scores"] == {"a": .5, "c": .5}
    assert meta["path_cap_cited"] == .5
    assert meta["n_returned"] == 3.
    assert meta["n_ranked"] == 1
    assert meta["mean_n_ranked"] == 2.


def test_empty_probe_answers_remain_in_aggregate_with_worst_case_penalties():
    from ldb.llm import failed_response, score_response
    from ldb.metrics import aggregate
    from ldb.task import build_instance

    inst = build_instance(0)
    _, _, missing = score_response(inst, '{"predictions": [], "probes": []}')
    assert missing["probe_coverage"] == 0.
    good = {"capture_mae": 0., "recognition_acc": 1., "prereq_f1": 1.}
    penalized = {k: missing[k] for k in good}
    assert penalized == {"capture_mae": 1., "recognition_acc": 0., "prereq_f1": 0.}
    result = aggregate([good, penalized])
    for metric in good:
        assert result[metric]["n"] == 2
        assert result[metric]["mean"] == .5
    _, _, failed = failed_response(inst)
    assert failed["raw_response"] is None
    assert {k: failed[k] for k in good} == penalized
