"""Counterfactual twin worlds.

Two universes identical in every respect except that one necessary complement
becomes available inside the horizon in one and not the other. A model that has
actually understood the readiness gate must move its predictions for the affected
pairs in the right direction; a model pattern-matching on surface plausibility
has no reason to move at all. Historical financial data cannot supply this
contrast, which is the point.
"""

from dataclasses import dataclass

from .task import Instance, build_instance
from .universe import NEVER, generate_universe

CUTOFF, HORIZON = 20, 8


@dataclass
class TwinPair:
    seed: int
    pivot: str  # the complement whose availability was flipped
    blocked: Instance  # pivot arrives after the horizon
    enabled: Instance  # pivot arrives inside the horizon
    affected: list[tuple[str, str]]  # pairs whose only blocker was the pivot


def _find_pivot(u, cutoff=CUTOFF, horizon=HORIZON) -> str | None:
    """A complement that blocks at least one otherwise-viable need."""
    end = cutoff + horizon
    best, best_n = None, 0
    for kid, k in u.complements.items():
        if k.ready_at <= end:
            continue  # not currently blocking
        needs = [
            n for n in u.needs.values()
            if kid in n.complement_ids
            and all(u.complements[o].ready_at <= end for o in n.complement_ids if o != kid)
        ]
        if len(needs) > best_n:
            best, best_n = kid, len(needs)
    return best


def build_twin(seed: int, surface: str = "finance",
               cutoff: int = CUTOFF, horizon: int = HORIZON) -> TwinPair | None:
    """Returns None when this seed has no cleanly pivotal complement."""
    base = generate_universe(seed, cutoff=cutoff, horizon=horizon)
    pivot = _find_pivot(base, cutoff, horizon)
    if pivot is None:
        return None

    # Commentary in BOTH worlds is generated from the base world's probabilities,
    # so the intervention cannot reach the recognition channel.
    pre = {k: p.p_true for k, p in base.pairs.items()}
    kw = dict(surface=surface, cutoff=cutoff, horizon=horizon,
              mention_p_override=pre)
    blocked = build_instance(seed, comp_ready_override={pivot: NEVER}, **kw)
    enabled = build_instance(seed, comp_ready_override={pivot: cutoff + 1}, **kw)

    end = cutoff + horizon
    affected = [
        (pid, nid)
        for (pid, nid) in blocked.candidates
        if pivot in base.needs[nid].complement_ids
        and all(base.complements[o].ready_at <= end
                for o in base.needs[nid].complement_ids if o != pivot)
        and (pid, nid) in enabled.label
    ]
    if not affected:
        return None
    return TwinPair(seed, pivot, blocked, enabled, affected)


def direction_score(tp: TwinPair, scores_blocked: dict, scores_enabled: dict) -> dict:
    """Did the ranker move affected pairs UP when the blocker was removed?

    Scores are rank-scale and not comparable across worlds in absolute terms, so
    we compare each pair against its own world: its percentile among that world's
    candidates. Chance is 0.5.
    """
    def pct(scores, keys):
        """Midrank percentile. Counting only strictly-lower values put the whole
        tied unranked group at 0.0, which flattered any movement out of it and
        inflated the reported deltas (structured 0.190 -> 0.108 once corrected).
        """
        vals = list(scores.values())
        n = len(vals)
        out = {}
        for k in keys:
            lower = sum(v < scores[k] for v in vals)
            equal = sum(v == scores[k] for v in vals)
            out[k] = (lower + (equal - 1) / 2) / max(n - 1, 1)
        return out

    pb = pct(scores_blocked, tp.affected)
    pe = pct(scores_enabled, tp.affected)
    moved_up = [pe[k] > pb[k] for k in tp.affected]
    ties = [pe[k] == pb[k] for k in tp.affected]
    return {
        "n_affected": len(tp.affected),
        # ties (both worlds unranked) count as chance, not as a win
        "direction_acc": float(
            sum(moved_up) + 0.5 * sum(ties)
        ) / len(tp.affected),
        "mean_delta_pct": float(sum(pe[k] - pb[k] for k in tp.affected) / len(tp.affected)),
    }
