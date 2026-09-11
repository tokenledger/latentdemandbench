"""Hidden graph -> observable evidence at a cutoff.

Two views of the same corpus are produced:
  * `Observed` - a structured, noisy transcription used by the evidence oracle.
  * `to_text()` - the finance-flavored document corpus handed to an LLM.

Both are derived from the SAME noisy draws, so the oracle and the LLM see
exactly the same information. The hidden graph is never exposed, and no
document states a product->need link for a pair that has not already activated.
"""

from dataclasses import dataclass, field

import numpy as np

from .universe import D, NEVER, Universe

PRECISION = 2  # decimals shown in the corpus AND stored in Observed

ATTR = ["thermal", "coherence", "yield", "latency", "density",
        "purity", "throughput", "stability"]


@dataclass
class Observed:
    cutoff: int
    horizon: int
    # product -> claimed capability ids (noisy: drops and false additions)
    prod_caps: dict[str, list[str]] = field(default_factory=dict)
    # capability -> noisy attribute vector
    cap_vecs: dict[str, np.ndarray] = field(default_factory=dict)
    # need -> noisy requirement vector
    need_reqs: dict[str, np.ndarray] = field(default_factory=dict)
    need_thresholds: dict[str, float] = field(default_factory=dict)
    need_materiality: dict[str, float] = field(default_factory=dict)
    prod_supply: dict[str, float] = field(default_factory=dict)
    prod_distribution: dict[str, float] = field(default_factory=dict)
    prod_moat: dict[str, float] = field(default_factory=dict)
    # need -> claimed required complements
    need_comps: dict[str, list[str]] = field(default_factory=dict)
    # complement -> claimed ready-by quarter (T_MAX+1 == no announced date)
    comp_ready: dict[str, int] = field(default_factory=dict)
    mention_counts: dict[tuple[str, str], int] = field(default_factory=dict)


def observe(
    u: Universe,
    cutoff: int,
    horizon: int,
    seed: int,
    drop_p: float = 0.15,
    false_p: float = 0.10,
    attr_noise: float = 0.18,
) -> Observed:
    # One stream per entity, never a shared one. With a shared stream any
    # data-dependent branch (e.g. "don't misreport an undated complement")
    # consumes a different number of draws in a counterfactual world and
    # silently shifts every later entity's noise. That is not hypothetical: it
    # desynchronised 16 of 45 twin pairs before this change.
    # Fixed integers, never hash(): Python salts str hashes per process, so
    # hash(kind) silently made every corpus irreproducible across runs.
    KIND = {"cap": 1, "prod": 2, "need": 3, "comp": 4}

    def s(kind: str, idx: int) -> np.random.Generator:
        return np.random.default_rng([seed * 7919 + 13, KIND[kind], idx])

    o = Observed(cutoff=cutoff, horizon=horizon)

    # The dossier prints two decimals. Store what is printed, so the structured
    # baseline reasons over exactly the numbers a reader gets rather than the
    # unrounded originals - that asymmetry moved the baseline's top-10 in most
    # universes and inflated the gap it appeared to open over a model.
    def q(x):
        return np.round(x, PRECISION)

    for i, (cid, c) in enumerate(u.capabilities.items()):
        rng = s("cap", i)
        v = np.clip(c.vec + rng.normal(0, attr_noise, D), 0, None)
        v[v < 0.12] = 0.0
        o.cap_vecs[cid] = q(v)

    for i, (pid, p) in enumerate(u.products.items()):
        rng = s("prod", i)
        claimed = [c for c in p.cap_ids if rng.random() > drop_p]
        for cid in u.capabilities:
            if cid not in p.cap_ids and rng.random() < false_p / len(u.capabilities) * 3:
                claimed.append(cid)
        o.prod_caps[pid] = claimed or list(p.cap_ids[:1])
        # company-side factors, reported with the same instrument error
        for attr, dst in (("supply", o.prod_supply), ("distribution", o.prod_distribution),
                          ("moat", o.prod_moat)):
            dst[pid] = float(q(np.clip(getattr(p, attr) + rng.normal(0, 0.08), 0, 1)))

    for i, (nid, n) in enumerate(u.needs.items()):
        rng = s("need", i)
        r = np.clip(n.req + rng.normal(0, attr_noise, D), 0, None)
        r[r < 0.12] = 0.0
        o.need_reqs[nid] = q(r)
        o.need_thresholds[nid] = float(q(np.clip(
            n.threshold + rng.normal(0, 0.04), 0.2, 0.9)))
        claimed = [k for k in n.complement_ids if rng.random() > drop_p]
        o.need_comps[nid] = claimed or list(n.complement_ids[:1])
        o.need_materiality[nid] = float(q(max(
            0.01, n.materiality * (1 + rng.normal(0, 0.10)))))

    for i, (kid, k) in enumerate(u.complements.items()):
        rng = s("comp", i)
        # Draw unconditionally, then decide how to use it, so a NEVER pivot
        # consumes exactly the same randomness as a dated one.
        misreport, jitter = rng.random() < 0.12, int(rng.integers(-4, 5))
        o.comp_ready[kid] = (
            int(max(0, k.ready_at + jitter))
            if misreport and k.ready_at != NEVER else k.ready_at
        )

    o.mention_counts = {key: len(v) for key, v in u.mentions.items()}
    return o


def _vec_str(v: np.ndarray, attr: list[str]) -> str:
    parts = [f"{attr[i]} {v[i]:.{PRECISION}f}" for i in range(D) if v[i] > 0]
    return ", ".join(parts) if parts else "no measurable profile"


# Narrative surfaces over the SAME hidden graph and the SAME noisy observations.
# Both surfaces use identical sentence templates and emit the same number of
# statements - only the lexicon changes. That keeps corpus length comparable, so a
# performance delta isolates reliance on familiar framing rather than on prompt
# length or document structure.
def _surface(attr, unit, heads, sources, holder, size_label, opaque=False):
    return {"attr": attr, "unit": unit, "head": heads, "sources": sources,
            "holder": holder, "size_label": size_label, "opaque": opaque}


SURFACES = {
    "finance": {
        "attr": ATTR,
        "unit": "quarter",
        "head": ["SECTOR DOSSIER", "TECHNICAL PUBLICATIONS: capability characterisation",
                 "PRODUCT TEARDOWNS: which capabilities each unit embodies",
                 "INDUSTRY PROBLEM REPORTS: unmet needs and their bottlenecks",
                 "SUPPLY / READINESS NOTES: complement availability",
                 "FINANCIAL NARRATIVES: what the market currently discusses",
                 "REALIZED DEPLOYMENTS: pairings already in commercial service"],
        "sources": "sell-side sources",
        "holder": "Issuer",
        "size_label": "Addressable value",
        "opaque": False,
    },
    "opaque": {
        "attr": [f"A{i}" for i in range(D)],
        "unit": "step",
        "head": ["RECORD SET", "TABLE T: characterisation of each C-item",
                 "TABLE P: which C-items each P-item embodies",
                 "TABLE N: requirements and constraints of each N-item",
                 "TABLE K: availability of each K-item",
                 "TABLE M: reference counts per item pair",
                 "TABLE R: pairs already deployed before the cutoff"],
        "sources": "independent references",
        "holder": "Holder",
        "size_label": "Value if served",
        "opaque": True,
    },
    # Same hidden economy, different narrative costume. Cross-surface transfer is
    # the point: a model relying on the register rather than the evidence should
    # move between these.
    "space_opera": _surface(
        ["heat", "resonance", "yield", "lag", "mass", "purity", "flux", "stability"],
        "cycle",
        ["FLEET SURVEY", "ARCHIVE: relic characterisation",
         "SALVAGE MANIFESTS: what each artifact carries",
         "COLONY PETITIONS: unmet needs and their obstacles",
         "SUPPLY LANES: substrate availability",
         "COUNCIL CHATTER: what the fleets currently debate",
         "STANDING DEPLOYMENTS: pairings already in service"],
        "envoy dispatches", "House", "Tribute value"),
    "anime_tech": _surface(
        ["thermal", "sync", "output", "delay", "density", "clarity", "flow", "balance"],
        "arc",
        ["ACADEMY DOSSIER", "LAB NOTES: core characterisation",
         "UNIT SCHEMATICS: which cores each frame carries",
         "FIELD REPORTS: unmet needs and their limiters",
         "LOGISTICS: catalyst availability",
         "GUILD RUMOURS: what pilots currently discuss",
         "ACTIVE SORTIES: pairings already deployed"],
        "guild sources", "Syndicate", "Contract value"),
    "cyberpunk": _surface(
        ["thermal", "signal", "yield", "latency", "density", "integrity",
         "bandwidth", "uptime"],
        "quarter",
        ["ARCOLOGY BRIEF", "TEARDOWN: wetware characterisation",
         "RIGS: which modules each deck runs",
         "STREET DEMAND: unmet needs and their chokepoints",
         "BLACK SUPPLY: component availability",
         "FIXER TRAFFIC: what the runners currently move",
         "LIVE RUNS: pairings already fielded"],
        "fixer channels", "Zaibatsu", "Street value"),
}


def to_text(u: Universe, o: Observed, surface: str = "finance") -> str:
    """Render the observed evidence as a document corpus in a given surface."""
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r}; have {sorted(SURFACES)}")
    cfg = SURFACES[surface]
    attr, op, unit, head = cfg["attr"], cfg["opaque"], cfg["unit"], cfg["head"]

    def cap_name(cid):
        return f"item {cid}" if op else u.capabilities[cid].name

    def prod_name(pid):
        p = u.products[pid]
        return f"item {pid} (holder {pid}H)" if op else f"{p.name} ({p.company})"

    def need_name(nid):
        return f"item {nid}" if op else u.needs[nid].name

    def comp_name(kid):
        return f"item {kid}" if op else u.complements[kid].name

    def effect(cid):
        return f"outcome E{cid}" if op else u.capabilities[cid].effect

    def bottleneck(nid):
        return f"limiter B{nid}" if op else u.needs[nid].bottleneck

    L = [f"=== {head[0]}, {unit.upper()} {o.cutoff} ===",
         f"All figures below are as reported at the cutoff {unit}. Attribute "
         f"measurements are instrument readings and carry error."]

    L.append(f"\n--- {head[1]} ---")
    for cid in u.capabilities:
        L.append(
            f"[TP-{cid}] Characterisation of {cap_name(cid)}. Measured attribute profile: "
            f"{_vec_str(o.cap_vecs[cid], attr)}. Observed downstream effect: {effect(cid)}."
        )

    L.append(f"\n--- {head[2]} ---")
    for pid in u.products:
        names = ", ".join(cap_name(c) for c in o.prod_caps[pid])
        L.append(
            f"[PT-{pid}] {prod_name(pid)}. Teardown attributes capability: {names}. "
            f"{cfg['holder']} position: supply capacity {o.prod_supply[pid]:.2f}, "
            f"distribution reach {o.prod_distribution[pid]:.2f}, "
            f"defensibility {o.prod_moat[pid]:.2f} (0-1 scale)."
        )

    L.append(f"\n--- {head[3]} ---")
    for nid in u.needs:
        comps = ", ".join(comp_name(k) for k in o.need_comps[nid])
        L.append(
            f"[PR-{nid}] {need_name(nid)}. Governing bottleneck: {bottleneck(nid)}. Relief "
            f"requires an attribute profile of {_vec_str(o.need_reqs[nid], attr)}, with a "
            f"sufficiency threshold near {o.need_thresholds[nid]:.{PRECISION}f} on profile alignment. "
            f"Deployment additionally depends on: {comps}. "
            f"{cfg['size_label']} if served: "
            f"{o.need_materiality[nid]:.{PRECISION}f} units."
        )

    L.append(f"\n--- {head[4]} ---")
    for kid in u.complements:
        r = o.comp_ready[kid]
        when = (
            f"available since {unit} {r}" if r <= o.cutoff
            else (f"guided to {unit} {r}" if r != NEVER
                  else "no announced availability")
        )
        L.append(f"[SR-{kid}] {comp_name(kid)}: {when}.")

    L.append(f"\n--- {head[6]} ---")
    realized = sorted(k for k, pr in u.pairs.items() if pr.activated_at <= o.cutoff)
    if realized:
        for pid, nid in realized:
            L.append(
                f"[RZ] {prod_name(pid) if op else u.products[pid].name} is already "
                f"deployed against {need_name(nid)}; this pairing is history, not a "
                f"forecast candidate."
            )
    else:
        L.append("[RZ] No pairing has been realized before the cutoff.")

    L.append(f"\n--- {head[5]} ---")
    any_m = False
    for (pid, nid), cnt in sorted(o.mention_counts.items(), key=lambda x: -x[1]):
        if cnt == 0:
            continue
        any_m = True
        L.append(
            f"[FN] {prod_name(pid) if op else u.products[pid].name} is discussed by {cnt} "
            f"{cfg['sources']} in connection with {need_name(nid)}."
        )
    if not any_m:
        L.append(f"[FN] No pairing draws sustained commentary from {cfg['sources']}.")

    return "\n".join(L)
