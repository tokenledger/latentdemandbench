"""Multi-hop composition over the OBSERVED evidence.

Single source of truth for "what can be worked out from the dossier". The
evidence oracle and the composition-assist ablation both call this, so the
ablation is guaranteed to hand the model exactly the quantity the oracle uses
rather than a re-derivation that might drift from it.
"""

import numpy as np

from .render import Observed
from .universe import D


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))


def observed_profile(o: Observed, pid: str) -> np.ndarray:
    """product -> claimed capabilities -> combined attribute profile."""
    cvs = [o.cap_vecs[c] for c in o.prod_caps[pid]]
    return np.max(np.stack(cvs), axis=0) if cvs else np.zeros(D)


def observed_match(o: Observed, pid: str, nid: str) -> float:
    """Alignment between a product's profile and a need's requirement."""
    return _cos(observed_profile(o, pid), o.need_reqs[nid])


def observed_ready(o: Observed, nid: str) -> bool:
    """Are all of the need's claimed complements available within the horizon?"""
    return all(o.comp_ready[k] <= o.cutoff + o.horizon for k in o.need_comps[nid])


ASSISTS = ("none", "composition", "gate", "both")


def assist_block(o: Observed, candidates, assist: str) -> str:
    """Hand the model one hop of the chain, to localise where it fails.

    `composition` resolves product->capability->profile against need->requirement,
    leaving only the readiness gate. `gate` resolves complement availability,
    leaving only the composition. Whatever is handed over is computed by the same
    functions the evidence oracle uses, so an assisted model is being given the
    oracle's own intermediate quantity, not an approximation of it.
    """
    if assist not in ASSISTS:
        raise ValueError(f"unknown assist {assist!r}; have {ASSISTS}")
    if assist == "none":
        return ""

    L = []
    if assist in ("composition", "both"):
        L.append("\n--- PRECOMPUTED PROFILE ALIGNMENT (provided; no need to recompute) ---")
        L.append("Alignment of each product's combined profile against each need's "
                 "requirement, with that need's sufficiency threshold.")
        for pid, nid in candidates:
            m = observed_match(o, pid, nid)
            thr = o.need_thresholds[nid]
            L.append(f"[AL] ({pid}, {nid}): alignment {m:.3f} vs threshold {thr:.3f}"
                     f" -> {'above' if m >= thr else 'below'}")
    if assist in ("gate", "both"):
        L.append("\n--- PRECOMPUTED DEPLOYMENT READINESS (provided) ---")
        L.append("Whether every complement a need depends on is available within "
                 "the horizon.")
        for nid in sorted(o.need_comps):
            ok = observed_ready(o, nid)
            L.append(f"[RD] {nid}: {'all complements available' if ok else 'blocked'}")
    return "\n".join(L)


def observed_capture(o: Observed, pid: str) -> float:
    """Company-side value capture from the reported supply/distribution/moat."""
    return float(min(1.0, max(0.0,
        0.15 + 0.55 * o.prod_supply[pid] * o.prod_distribution[pid]
        + 0.30 * o.prod_moat[pid])))


def expected_value(o: Observed, pid: str, nid: str) -> float:
    """Realization x materiality x capture: what the holder expects to earn."""
    return (evidence_score(o, pid, nid) * o.need_materiality[nid]
            * observed_capture(o, pid))


def evidence_score(o: Observed, pid: str, nid: str, slope: float = 9.0) -> float:
    m = observed_match(o, pid, nid)
    tech = 1.0 / (1.0 + np.exp(-slope * (m - o.need_thresholds[nid])))
    return float(tech if observed_ready(o, nid) else tech * 0.03)
