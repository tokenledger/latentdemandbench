"""Ranking, calibration and recognition-gap metrics."""

import numpy as np

from .task import Instance


def _order(scores: dict, keys: list) -> list:
    return sorted(keys, key=lambda k: -scores[k])


def recall_at_k(inst: Instance, scores: dict, k: int = 10) -> float:
    pos = sum(inst.label.values())
    if pos == 0:
        return float("nan")
    top = _order(scores, inst.candidates)[:k]
    return sum(inst.label[p] for p in top) / pos


def ndcg_at_k(inst: Instance, scores: dict, k: int = 10) -> float:
    """Unweighted: gain 1 for a pair that activates, 0 otherwise.

    This was weighted by each need's materiality, but materiality is sampled
    latently and never rendered into the dossier, so the metric graded models on
    information they were never given - and ranking by p* was consequently NOT
    an upper bound on it (0.428 vs 0.693 for a p*-by-materiality ranker).
    Restoring materiality as a weight requires rendering it in the corpus and
    asking for expected-value ranking; until then the honest metric is unweighted.
    """
    gains = {p: float(inst.label[p]) for p in inst.candidates}
    top = _order(scores, inst.candidates)[:k]
    dcg = sum(gains[p] / np.log2(i + 2) for i, p in enumerate(top))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else float("nan")


def average_ranks(values) -> np.ndarray:
    """One-based ranks, assigning the average occupied rank to every tie."""
    values = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2)[inverse]


def auc(labels: list, scores: list) -> float:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    ranks = average_ranks(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_auc(inst: Instance, scores: dict) -> float:
    ks = inst.candidates
    return auc([inst.label[k] for k in ks], [scores[k] for k in ks])


def prob_mse(inst: Instance, probs: dict) -> float:
    """Mean squared error against the latent probability p*.

    NOT the Brier score, which is squared error against the realized 0/1
    outcome. Calling it Brier overstated comparability with the literature.
    """
    ks = inst.candidates
    d = [(probs[k] - inst.p_star[k]) ** 2 for k in ks]
    return float(np.mean(d))


def mean_lead_time(inst: Instance, scores: dict, k: int = 10) -> float:
    top = _order(scores, inst.candidates)[:k]
    hits = [inst.lead_time[p] for p in top if inst.label[p] == 1]
    return float(np.mean(hits)) if hits else float("nan")


def recognition_gap_auc(inst: Instance, scores: dict) -> float:
    """Can the ranker find winners the market is NOT talking about at all?

    Restricted to pairs with *zero* mentions (~40% of candidates). An earlier
    version used the below-median subset, but the median count is 1, so that
    admitted ~76% of pairs and the metric was nearly a restatement of AUC.
    """
    sub = [k for k in inst.candidates if inst.recognized[k] == 0]
    if not sub:
        return float("nan")
    return auc([inst.label[k] for k in sub], [scores[k] for k in sub])


def consensus_rho(inst: Instance, scores: dict) -> float:
    """Spearman correlation between the ranker's scores and mention counts.

    Distinct from gap-AUC: it asks whether the ranker is *reproducing* consensus
    rather than whether it is accurate where consensus is silent. The recognition
    baseline is ~1 by construction; a ranker reading evidence should be near 0.
    """
    ks = inst.candidates
    x = np.array([scores[k] for k in ks], dtype=float)
    y = np.array([inst.recognized[k] for k in ks], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")

    rx, ry = average_ranks(x), average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def ndcg_ev_at_k(inst: Instance, scores: dict, k: int = 10) -> float:
    """Materiality-weighted nDCG. Fair only because materiality is now rendered
    into the dossier; when it was latent this graded models on hidden state."""
    gains = {p: inst.label[p] * inst.materiality[p] for p in inst.candidates}
    top = _order(scores, inst.candidates)[:k]
    dcg = sum(gains[p] / np.log2(i + 2) for i, p in enumerate(top))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else float("nan")


def capture_auc(inst: Instance, scores: dict) -> float:
    """Does the ranking also separate pairings whose holder captures the value?

    Distinct from `auc`: a pairing can be served without the originating company
    keeping the value, which is the third of idea.md's four questions.
    """
    if not inst.captured:
        return float("nan")
    ks = inst.candidates
    return auc([inst.captured[k] for k in ks], [scores[k] for k in ks])


def probe_auc(inst: Instance, scores: dict) -> float:
    """AUC restricted to the fixed probe set.

    A human cannot rank 175 pairs, so the human-comparable task is the 16 probes
    every model already answers. Scoring models and baselines on the same 16
    makes the three directly comparable; scoring humans on a shortlist while
    models rank everything would not.
    """
    if not inst.probes:
        return float("nan")
    y = [inst.label[k] for k in inst.probes]
    return auc(y, [scores[k] for k in inst.probes])


def evaluate(inst: Instance, scores: dict, probs: dict | None = None,
             ev_scores: dict | None = None, cap_scores: dict | None = None,
             probe_scores: dict | None = None) -> dict:
    """Three objectives, three orders.

    `scores`     ranks by realization probability      -> recall/nDCG/AUC/gap
    `ev_scores`  ranks by realization x materiality    -> nDCG-EV
    `cap_scores` ranks by realization x capture        -> capture-AUC

    nDCG-EV's gain is (realized x materiality) and capture-AUC's label is
    captured/not; a single p x materiality x capture order optimises neither.
    """
    ev = ev_scores if ev_scores is not None else scores
    cap = cap_scores if cap_scores is not None else scores
    out = {
        "recall@10": recall_at_k(inst, scores, 10),
        "ndcg@10": ndcg_at_k(inst, scores, 10),
        "auc": pair_auc(inst, scores),
        "lead_time@10": mean_lead_time(inst, scores, 10),
        "ndcg_ev@10": ndcg_ev_at_k(inst, ev, 10),
        "capture_auc": capture_auc(inst, cap),
        "gap_auc": recognition_gap_auc(inst, scores),
        "probe_auc": probe_auc(inst, probe_scores if probe_scores is not None else scores),
        "consensus_rho": consensus_rho(inst, scores),
    }
    if probs is not None:
        out["prob_mse_vs_p*"] = prob_mse(inst, probs)
    return out


N_BOOT = 10000


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap over universes. Returns (mean, lo, hi)."""
    bad = [x for x in values if not isinstance(x, (int, float, np.floating))]
    if bad:
        raise TypeError(
            f"non-numeric value in a metric row: {bad[0]!r}. Payloads such as "
            "raw responses belong in the artifact's archive, not in metrics.")
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (
        float(v.mean()),
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def aggregate(rows: list[dict], seed: int = 0) -> dict:
    """metric -> {mean, lo, hi, n}. The universe is the resampling unit.

    Keys are the union across rows, not row 0's: a failed call contributes a
    row without path metrics, and keying off the first row would either drop
    those metrics entirely or raise when a later row lacked them.
    """
    keys = sorted({k for r in rows for k in r})  # sorted: set order is not stable
    out = {}
    for k in keys:
        vals = [r.get(k, float("nan")) for r in rows]
        mean, lo, hi = bootstrap_ci(vals, seed=seed)
        out[k] = {"mean": mean, "lo": lo, "hi": hi,
                  "n": int(sum(not np.isnan(x) for x in vals))}
    return out


def paired_diff(rows_a: list[dict], rows_b: list[dict], metric: str, seed: int = 0):
    """Paired bootstrap on per-universe differences (a - b), same universes.

    Paired rather than comparing two independent CIs: universes vary a lot in
    difficulty, and that shared variance cancels in the difference.
    """
    nan = float("nan")
    d = [a.get(metric, nan) - b.get(metric, nan) for a, b in zip(rows_a, rows_b)]
    return bootstrap_ci(d, seed=seed)
