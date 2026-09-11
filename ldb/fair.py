"""A reference that is not privileged.

`structured_evidence` knows the simulator's own functional form - max-pooling,
cosine alignment, a slope-9 sigmoid, a 0.03 gate multiplier - none of which is
stated anywhere a model can read. Headroom measured against it is therefore not
headroom against anything achievable, which makes the paper's central number
unquotable.

This reference is given exactly what a model is given: the observable
quantities, with no knowledge of how they combine. It fits that combination from
*other* universes and is evaluated on held-out ones, so it never sees the
answers for the instance it is scoring. It is the honest answer to "how much of
this evidence is actually usable?"

Deliberately a plain logistic model on interpretable features rather than
anything larger: the claim is about what the evidence supports, so the reference
should be the simplest thing that extracts it, not the strongest.
"""

from dataclasses import dataclass

import numpy as np

from .compose import observed_capture, observed_profile
from .task import Instance


def features(inst: Instance, pid: str, nid: str) -> np.ndarray:
    """Observables only. No cosine, no sigmoid, no gate constant.

    Every term here is read straight off the dossier; how they combine is what
    the fit has to discover.
    """
    o = inst.observed
    prof, req = observed_profile(o, pid), o.need_reqs[nid]
    both = (prof > 0) & (req > 0)
    horizon_end = o.cutoff + o.horizon
    waits = [o.comp_ready[k] - o.cutoff for k in o.need_comps[nid]]
    return np.array([
        float(np.dot(prof, req)),                    # raw overlap
        float(np.sum(np.minimum(prof, req))),        # covered requirement
        float(np.sum(req[~both])),                   # requirement left unmet
        float(both.sum()),                           # attributes in common
        float(o.need_thresholds[nid]),               # stated sufficiency bar
        float(np.dot(prof, req) - o.need_thresholds[nid]),
        float(all(o.comp_ready[k] <= horizon_end for k in o.need_comps[nid])),
        float(min(waits)), float(max(w for w in waits)),
        float(len(o.need_comps[nid])),
        float(len(o.prod_caps[pid])),
        float(o.need_materiality[nid]),
        float(observed_capture(o, pid)),
        float(o.mention_counts[(pid, nid)]),         # the market's own view
        1.0,
    ])


@dataclass
class FairReference:
    w: np.ndarray

    def __call__(self, inst: Instance, seed: int = 0) -> dict:
        X = np.stack([features(inst, *k) for k in inst.candidates])
        z = X @ self.w
        return {k: float(1 / (1 + np.exp(-v)))
                for k, v in zip(inst.candidates, z)}


def fit(train: list[Instance], iters: int = 400, lr: float = 0.5,
        l2: float = 1e-3) -> FairReference:
    """Logistic regression by gradient descent on standardised features."""
    X = np.stack([features(i, *k) for i in train for k in i.candidates])
    y = np.array([i.label[k] for i in train for k in i.candidates], dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    sd[-1] = 1.0
    mu[-1] = 0.0  # keep the intercept
    Xs = (X - mu) / sd

    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xs @ w)))
        w -= lr * (Xs.T @ (p - y) / len(y) + l2 * w)

    # fold standardisation back in so the returned ranker takes raw features
    raw = w / sd
    raw[-1] = w[-1] - float(np.sum(w[:-1] * mu[:-1] / sd[:-1]))
    return FairReference(raw)


def fit_holdout(seeds: range, held: set, **kw) -> FairReference:
    """Fit on every seed except the held-out ones."""
    from .task import build_instance
    return fit([build_instance(s) for s in seeds if s not in held], **kw)
