"""Partial validation of the evidence paths a model reports.

SCOPE, because the name invites overclaiming: these metrics check *capability
attribution* only - whether a cited capability is one the dossier attributes to
that product, and whether a blocked need's blocker is named. They do NOT verify
that the capability supports the selected need, nor the bottleneck, attribute
profile, threshold arithmetic, or the chain end to end. They are evidence that a
citation is not fabricated, not that the reasoning is correct.


The schema has always required `evidence_path`, but nothing checked it, so a
model could earn full ranking credit while citing entities that have nothing to
do with the pair. These metrics ask whether the stated reasoning actually
matches the dossier: a right answer for a cited-wrong reason is visible here and
invisible in AUC.

Matching is by the entity's *rendered* name, i.e. the string the model actually
saw, which differs by surface (a pseudo-word under `finance`, a bare id under
`opaque`).
"""

from .compose import observed_ready
from .metrics import average_ranks
from .task import Instance


def rendered_names(inst: Instance) -> tuple[dict, dict]:
    """(capability id -> alias set, complement id -> alias set) as rendered."""
    u, opaque = inst.universe, inst.surface == "opaque"
    caps = {
        cid: {cid} | (set() if opaque else {c.name})
        for cid, c in u.capabilities.items()
    }
    comps = {
        kid: {kid} | (set() if opaque else {k.name})
        for kid, k in u.complements.items()
    }
    return caps, comps


def _cites(text: str, aliases: set[str]) -> bool:
    """Whole-token match. Substring matching made 'C1' a citation of 'C10',
    and every id shares a prefix with nine others."""
    import re
    low = text.lower()
    # hyphen counts as part of the token: "Cavex" must not match "Cavex-57"
    return any(re.search(rf"(?<![0-9a-z-]){re.escape(a.lower())}(?![0-9a-z-])", low)
               for a in aliases if a)


def path_metrics(inst: Instance, kept: list[tuple[tuple[str, str], dict]]) -> dict:
    """Score the evidence paths of the predictions that were actually ranked."""
    if not kept:
        return {"path_cap_cited": float("nan"),
                "path_cap_false": float("nan"),
                "path_gate_cited": float("nan")}

    caps, comps = rendered_names(inst)
    o = inst.observed
    cited, false_cited, gate_ok, gate_n = 0, 0, 0, 0

    for (pid, nid), pred in kept:
        text = pred.get("evidence_path") or ""
        true_caps = set(o.prod_caps[pid])
        # does it cite a capability the dossier actually attributes to this item?
        if any(_cites(text, caps[c]) for c in true_caps):
            cited += 1
        # does it cite one the dossier does NOT attribute to it?
        if any(_cites(text, caps[c]) for c in caps if c not in true_caps):
            false_cited += 1
        # for a need the dossier shows as blocked, is the blocker named?
        if not observed_ready(o, nid):
            gate_n += 1
            blockers = [k for k in o.need_comps[nid]
                        if o.comp_ready[k] > o.cutoff + o.horizon]
            if any(_cites(text, comps[k]) for k in blockers):
                gate_ok += 1

    n = len(kept)
    return {
        "path_cap_cited": cited / n,
        "path_cap_false": false_cited / n,
        "path_gate_cited": (gate_ok / gate_n) if gate_n else float("nan"),
    }


def elicited_metrics(inst: Instance, kept: list, n_missing: int = 0) -> dict:
    """Score the four outputs beyond ranking that `idea.md` asks for.

    Each is a separate question about the same pairing, and a model can be right
    about one and wrong about the others: whether the need gets served, how much
    it is worth, whether the holder keeps the value, and whether the market has
    noticed already.
    """
    import numpy as np

    from .compose import observed_ready

    if not kept:
        nan = float("nan")
        if n_missing:
            return {"materiality_rho": nan, "capture_mae": 1.0,
                    "recognition_acc": 0.0, "prereq_f1": 0.0}
        return {k: nan for k in ("materiality_rho", "capture_mae",
                                 "recognition_acc", "prereq_f1")}
    total = len(kept) + n_missing
    o, u = inst.observed, inst.universe

    # materiality: rank correlation against the true latent value
    pred_m = [p.get("materiality") for _, p in kept]
    true_m = [inst.materiality[key] for key, _ in kept]
    ok = [(a, b) for a, b in zip(pred_m, true_m) if isinstance(a, (int, float))]
    if len(ok) > 2 and len({a for a, _ in ok}) > 1 and len({b for _, b in ok}) > 1:
        a = average_ranks([x for x, _ in ok])
        b = average_ranks([y for _, y in ok])
        materiality_rho = float(np.corrcoef(a, b)[0, 1])
    else:
        materiality_rho = float("nan")

    # value capture: absolute error against the exact simulator probability
    cap_err = [abs(p.get("capture", 0.0) - u.pairs[key].p_capture)
               for key, p in kept if isinstance(p.get("capture"), (int, float))]
    # an unanswered probe takes the worst possible error, so omitting the hard
    # ones cannot improve the score
    capture_mae = (float((np.sum(cap_err) + n_missing) / total)
                   if cap_err or n_missing else float("nan"))

    # recognition: is this pairing already discussed?
    rec = [(bool(p.get("recognized")), inst.recognized[key] > 0) for key, p in kept]
    recognition_acc = (float(sum(a == b for a, b in rec) / total)
                       if rec or n_missing else float("nan"))

    # prerequisites: the complements the dossier shows as still blocking
    f1s = []
    for key, p in kept:
        nid = key[1]
        truth = {k for k in o.need_comps[nid]
                 if o.comp_ready[k] > o.cutoff + o.horizon}
        said = {str(x).strip().upper() for x in (p.get("prerequisites") or [])}
        said = {x for x in said if x}
        if not truth and not said:
            f1s.append(1.0)
            continue
        tp = len(truth & said)
        prec = tp / len(said) if said else 0.0
        rec_ = tp / len(truth) if truth else 0.0
        f1s.append(0.0 if prec + rec_ == 0 else 2 * prec * rec_ / (prec + rec_))
    prereq_f1 = float(sum(f1s) / total) if f1s or n_missing else float("nan")

    return {"materiality_rho": materiality_rho, "capture_mae": capture_mae,
            "recognition_acc": recognition_acc, "prereq_f1": prereq_f1}
