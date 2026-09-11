"""Baselines. Each returns {pair: score}, higher = more likely to activate."""

import numpy as np

from .compose import evidence_score, expected_value
from .task import Instance


def random_rank(inst: Instance, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {k: float(rng.random()) for k in inst.candidates}


def recognition_rank(inst: Instance, seed: int = 0) -> dict:
    """What the market already talks about. The consensus baseline to beat."""
    rng = np.random.default_rng(seed)
    return {k: inst.recognized[k] + 1e-6 * rng.random() for k in inst.candidates}


def structured_evidence(inst: Instance, seed: int = 0) -> dict:
    """Multi-hop composition over the NOISY observed evidence only.

    product -> claimed capabilities -> measured attribute profile, matched against
    need -> bottleneck -> required profile, then gated on announced complement
    readiness.

    PRIVILEGED, and not an achievable ceiling. It knows the simulator's own
    functional form - max-pooling, cosine alignment, a slope-9 sigmoid and the
    0.03 gate multiplier - none of which appears in the model's prompt, which
    says only that profiles must "align" above a threshold. It is therefore a
    reference for how much signal the evidence carries, NOT a fair competitor,
    and calibration comparisons against it are not like-for-like.

    It is also not optimal: a posterior over the same observables could beat it,
    since it uses noisy point estimates, ignores mention evidence, and collapses
    readiness timing to a boolean.
    """
    return {k: evidence_score(inst.observed, *k) for k in inst.candidates}


def hidden_oracle(inst: Instance, seed: int = 0) -> dict:
    """Ranks by the exact simulator probability. Sanity upper bound."""
    return {k: inst.p_star[k] for k in inst.candidates}


def expected_value_rank(inst: Instance, seed: int = 0) -> dict:
    """Structured evidence x materiality x capture - the full chain idea.md
    describes, including the company-side leg."""
    return {k: expected_value(inst.observed, *k) for k in inst.candidates}


# What each ranker's score means. EV-weighted metrics need a probability to
# multiply; multiplying an already-expected-value score by materiality and
# capture again squares both factors, which is what made `expected_value` look
# worse than `structured_evidence` on nDCG-EV.
SCORE_KIND = {
    "random": "rank",              # arbitrary order, no probability
    "recognition": "rank",         # mention counts, not probabilities
    "structured_evidence": "prob",
    "fair_reference": "prob",
    "expected_value": "ev",        # already probability x materiality x capture
    "hidden_oracle": "prob",
}

_FAIR = None


def fair_reference(inst: Instance, seed: int = 0) -> dict:
    """Learned from observables alone, on universes disjoint from evaluation.

    Fitted once on seeds 1000-1099, which no experiment evaluates, so it can
    never have seen the instance it scores. Unlike `structured_evidence` it is
    given no part of the simulator's functional form, which is what makes a
    shortfall against it interpretable as achievable headroom.
    """
    global _FAIR
    if _FAIR is None:
        from .fair import fit
        from .task import build_instance
        _FAIR = fit([build_instance(s) for s in range(1000, 1100)])
    return _FAIR(inst)


BASELINES = {
    "random": random_rank,
    "recognition": recognition_rank,
    "structured_evidence": structured_evidence,
    "fair_reference": fair_reference,
    "expected_value": expected_value_rank,
    "hidden_oracle": hidden_oracle,
}
