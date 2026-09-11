"""Benchmark instance construction: evidence in, ranked-pair task out."""

from dataclasses import dataclass

from .compose import assist_block
from .render import Observed, observe, to_text
from .universe import NEVER, Universe, generate_universe


@dataclass
class Instance:
    universe: Universe
    observed: Observed
    corpus: str
    cutoff: int
    horizon: int
    candidates: list[tuple[str, str]]
    label: dict[tuple[str, str], int]  # activated within (cutoff, cutoff+H]
    p_star: dict[tuple[str, str], float]  # exact conditional probability
    materiality: dict[tuple[str, str], float]
    lead_time: dict[tuple[str, str], int]  # quarters from cutoff to activation
    recognized: dict[tuple[str, str], int]  # mention count at cutoff
    probes: list = None  # fixed stratified pairs for the auxiliary questions
    captured: dict[tuple[str, str], int] = None      # realized value capture
    p_capture: dict[tuple[str, str], float] = None   # exact capture probability
    surface: str = "finance"


def _probe_set(o, cands, n_per_cell: int = 4) -> list:
    """Pairs the auxiliary questions are asked about, fixed in advance.

    Scoring materiality/capture/recognition/prerequisites on whatever a model
    chose to rank makes those numbers selection-dependent and trivially gamed:
    a top-40 by realization probability contains almost no blocked pairs, so
    answering `prerequisites: []` every time scored a perfect 1.0. Stratifying
    by (blocked, mentioned) forces every cell to be answered.
    """
    from .compose import observed_ready

    cells: dict[tuple[bool, bool], list] = {}
    for key in cands:
        cell = (observed_ready(o, key[1]), o.mention_counts[key] > 0)
        cells.setdefault(cell, []).append(key)
    probes = []
    for cell in sorted(cells, key=lambda c: (c[0], c[1])):
        probes.extend(sorted(cells[cell])[:n_per_cell])  # deterministic
    return probes


def build_instance(
    seed: int, cutoff: int = 20, horizon: int = 8, surface: str = "finance",
    assist: str = "none", **kw
) -> Instance:
    u = generate_universe(seed, cutoff=cutoff, horizon=horizon, **kw)
    # `observe` is seeded independently of universe content, so twin worlds share
    # their observation noise as well as their generative stream
    o = observe(u, cutoff, horizon, seed)

    cands, label, p_star, mat, lead, rec = [], {}, {}, {}, {}, {}
    cap, p_cap = {}, {}
    for key, pr in u.pairs.items():
        if pr.activated_at <= cutoff:
            continue  # already realized; not a forecasting candidate
        cands.append(key)
        hit = int(pr.activated_at != NEVER and cutoff < pr.activated_at <= cutoff + horizon)
        label[key] = hit
        p_star[key] = pr.p_horizon
        mat[key] = u.needs[key[1]].materiality
        lead[key] = (pr.activated_at - cutoff) if hit else -1
        rec[key] = o.mention_counts[key]
        # value capture is scored only where the pairing is actually realized
        cap[key] = int(hit and pr.captured)
        p_cap[key] = pr.p_capture

    probes = _probe_set(o, cands, n_per_cell=4)
    corpus = to_text(u, o, surface) + assist_block(o, cands, assist)
    return Instance(u, o, corpus, cutoff, horizon, cands, label, p_star,
                    mat, lead, rec, probes=probes, captured=cap,
                    p_capture=p_cap, surface=surface)
