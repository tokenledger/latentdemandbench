"""Counterfactual twin invariants.

The twin arm is the paper's causal claim, and it is the part that broke most
often: three separate RNG-desynchronisation bugs let the intervention leak into
pairs, mentions, and observations it should never have touched.
"""

import numpy as np
import pytest

from ldb.baselines import BASELINES
from ldb.twins import build_twin, direction_score

TWINS = [t for s in range(30) if (t := build_twin(s)) is not None]


def test_some_seeds_yield_twins():
    assert len(TWINS) >= 15, "too few pivotal complements to test with"


@pytest.mark.parametrize("tp", TWINS, ids=lambda t: f"seed{t.seed}")
def test_only_affected_pairs_change_outcome(tp):
    """Regression: activation draws came from a shared stream whose length
    depended on p_true, so one flipped complement re-rolled every later pair."""
    unrelated = [k for k in tp.blocked.candidates
                 if k in tp.enabled.label and k not in set(tp.affected)]
    diff = [k for k in unrelated if tp.blocked.label[k] != tp.enabled.label[k]]
    assert not diff, f"{len(diff)} unrelated pairs changed outcome"


@pytest.mark.parametrize("tp", TWINS, ids=lambda t: f"seed{t.seed}")
def test_mentions_are_identical_across_twins(tp):
    """Regression: mention probability included p_true, so the intervention
    leaked into the very channel the twins exist to show is uninformed."""
    diff = [k for k in tp.blocked.candidates
            if k in tp.enabled.recognized
            and tp.blocked.recognized[k] != tp.enabled.recognized[k]]
    assert not diff, f"{len(diff)} pairs changed mention count"


@pytest.mark.parametrize("tp", TWINS, ids=lambda t: f"seed{t.seed}")
def test_observations_outside_the_pivot_are_identical(tp):
    """Regression: a conditional readiness draw consumed a different number of
    random values for a NEVER pivot, shifting every later complement's noise."""
    ob, oe = tp.blocked.observed, tp.enabled.observed
    assert [k for k in ob.comp_ready if k != tp.pivot
            and ob.comp_ready[k] != oe.comp_ready[k]] == []
    assert all(np.array_equal(ob.cap_vecs[k], oe.cap_vecs[k]) for k in ob.cap_vecs)
    assert all(np.array_equal(ob.need_reqs[k], oe.need_reqs[k]) for k in ob.need_reqs)
    assert all(ob.prod_caps[k] == oe.prod_caps[k] for k in ob.prod_caps)


def test_pivot_itself_does_change():
    """The counterfactual must actually be a counterfactual."""
    tp = TWINS[0]
    assert (tp.blocked.observed.comp_ready[tp.pivot]
            != tp.enabled.observed.comp_ready[tp.pivot])


def test_uninformed_rankers_score_exactly_chance():
    """Random and recognition cannot know about complement readiness, so their
    direction accuracy must be exactly 0.5 - not merely close to it."""
    for name in ("random", "recognition"):
        fn = BASELINES[name]
        accs = [direction_score(t, fn(t.blocked, seed=i), fn(t.enabled, seed=i))
                ["direction_acc"] for i, t in enumerate(TWINS)]
        assert np.allclose(accs, 0.5), f"{name} is not at chance: {set(accs)}"


def test_evidence_ranker_detects_the_intervention():
    fn = BASELINES["structured_evidence"]
    accs = [direction_score(t, fn(t.blocked), fn(t.enabled))["direction_acc"]
            for t in TWINS]
    assert np.mean(accs) > 0.8, f"structured evidence only reached {np.mean(accs):.3f}"
