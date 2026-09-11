"""Invariants of the universe generator and the rendered corpus.

Every test here corresponds to a defect that actually reached the results and
had to be caught in review. The regression each one guards is named, because a
test whose purpose is forgotten is a test that gets deleted during the next
refactor.
"""

import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pytest

from ldb.baselines import structured_evidence
from ldb.render import PRECISION
from ldb.task import build_instance
from ldb.universe import NEVER, T_MAX, generate_universe

SEEDS = range(30)


def corpus_hash(inst):
    return hashlib.sha256(inst.corpus.encode()).hexdigest()


# --------------------------------------------------------------- reproducibility

def test_corpus_is_reproducible_in_process():
    assert corpus_hash(build_instance(7)) == corpus_hash(build_instance(7))


def test_corpus_is_reproducible_across_processes():
    """Regression: stream ids derived from hash(str), which Python salts per
    process, so every corpus differed between runs and no artifact reproduced."""
    prog = (
        "import hashlib;"
        "from ldb.task import build_instance;"
        "print(hashlib.sha256(build_instance(7).corpus.encode()).hexdigest())"
    )
    out = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                          text=True, cwd=ROOT).stdout.strip() for _ in range(2)}
    assert len(out) == 1, "corpus differs between processes"
    assert out.pop() == corpus_hash(build_instance(7))


# ------------------------------------------------------------------- sentinel

def test_horizon_past_timeline_is_rejected():
    """Regression: T_MAX+1 sat close enough to the horizon that a later cutoff
    swept every never-event into the positive window."""
    with pytest.raises(ValueError, match="timeline"):
        generate_universe(0, cutoff=T_MAX - 2, horizon=8)


@pytest.mark.parametrize("cutoff", [12, 20, 28])
def test_never_events_are_never_positive(cutoff):
    inst = build_instance(0, cutoff=cutoff, horizon=8)
    never = [k for k in inst.candidates
             if inst.universe.pairs[k].activated_at == NEVER]
    assert never, "expected some never-activating pairs"
    assert not any(inst.label[k] for k in never)
    assert 0 < sum(inst.label.values()) < len(inst.candidates)


def test_never_complement_has_zero_scored_probability_despite_residual_gate():
    """The 0.03 gate is an internal propensity, not positive horizon mass.

    A required NEVER complement makes the feasible arrival window empty.  The
    generator must therefore assign p_horizon/p* and the realised label exactly
    zero even though its internal p_true remains positive.
    """
    n_blocked = 0
    for seed in SEEDS:
        inst = build_instance(seed)
        for key in inst.candidates:
            pair = inst.universe.pairs[key]
            need = inst.universe.needs[key[1]]
            if any(inst.universe.complements[k].ready_at == NEVER
                   for k in need.complement_ids):
                n_blocked += 1
                assert 0.0 < pair.p_true <= 0.03
                assert pair.p_horizon == 0.0
                assert inst.p_star[key] == 0.0
                assert inst.label[key] == 0
    assert n_blocked > 0, "expected at least one NEVER-blocked candidate"


# ------------------------------------------------- information symmetry (LLM vs baseline)

def test_entity_names_are_unique_within_a_universe():
    """Regression: names sampled with replacement made dossiers ambiguous for a
    reader while the structured baseline still held unambiguous ids."""
    for s in SEEDS:
        u = generate_universe(s)
        names = ([c.name for c in u.capabilities.values()]
                 + [k.name for k in u.complements.values()]
                 + [p.name for p in u.products.values()]
                 + [n.name for n in u.needs.values()])
        dupes = [n for n, c in Counter(names).items() if c > 1]
        assert not dupes, f"seed {s}: duplicate names {dupes}"


def test_baseline_sees_only_the_precision_the_corpus_prints():
    """Regression: the corpus printed 2dp while the baseline used full
    precision, so the baseline solved a strictly easier problem."""
    for s in SEEDS:
        inst = build_instance(s)
        before = structured_evidence(inst)
        o = inst.observed
        for d in (o.cap_vecs, o.need_reqs):
            for k in d:
                d[k] = np.round(d[k], PRECISION)
        for k in o.need_thresholds:
            o.need_thresholds[k] = round(o.need_thresholds[k], PRECISION)
        assert structured_evidence(inst) == before, f"seed {s} precision-sensitive"


# --------------------------------------------------------------------- surfaces

def test_surfaces_share_ground_truth_and_observations():
    a, b = build_instance(3, surface="finance"), build_instance(3, surface="opaque")
    assert a.label == b.label and a.p_star == b.p_star
    assert a.recognized == b.recognized
    for k in a.observed.cap_vecs:
        assert np.array_equal(a.observed.cap_vecs[k], b.observed.cap_vecs[k])


def test_surfaces_have_comparable_length():
    """A shorter opaque corpus would confound surface with prompt length."""
    a, b = build_instance(3, surface="finance"), build_instance(3, surface="opaque")
    assert 0.8 < len(b.corpus) / len(a.corpus) < 1.25


def test_baselines_are_surface_invariant():
    """The baseline reads Observed, not text, so it is the fixed control against
    which any model difference between surfaces is measured."""
    a, b = build_instance(3, surface="finance"), build_instance(3, surface="opaque")
    assert structured_evidence(a) == structured_evidence(b)


# ------------------------------------------------------------------- corpus content

def test_corpus_never_asserts_a_future_realization():
    """Mentions may name a pair - that is the recognition signal - but no
    document may state that an unrealized pair will be realized."""
    inst = build_instance(0)
    banned = ("will be realized", "will activate", "is expected to succeed",
              "confirmed for", "guaranteed")
    low = inst.corpus.lower()
    assert not any(b in low for b in banned)


def test_base_rate_is_in_a_sane_band():
    rates = [np.mean(list(build_instance(s).label.values())) for s in SEEDS]
    assert 0.03 <= np.mean(rates) <= 0.30, f"base rate {np.mean(rates):.3f}"


def test_already_realized_pairs_are_disclosed_not_silently_filtered():
    """Regression: eligibility removed already-realized pairs behind the scenes,
    so the baseline got a filtered candidate list the model could not derive."""
    inst = build_instance(0)
    realized = [k for k, p in inst.universe.pairs.items()
                if p.activated_at <= inst.cutoff]
    assert realized, "seed 0 should have some realized pairs to disclose"
    assert not any(k in inst.label for k in realized)
    for pid, nid in realized:
        name = inst.universe.products[pid].name
        assert any(line.startswith("[RZ]") and name in line
                   for line in inst.corpus.split("\n")), f"{pid},{nid} undisclosed"
