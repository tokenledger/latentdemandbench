"""Synthetic economy generator for LatentDemandBench.

A universe is a latent economy in R^d. Products own capabilities; needs demand
capability profiles; complements gate commercial realization. Everything the
model ever sees is a lossy, noisy view of the structures built here.
"""

from dataclasses import dataclass, field

import numpy as np

D = 8  # latent capability-attribute dimension
T_MAX = 40  # universe timeline length, in quarters

# "Never activates" sentinel. Deliberately far outside any real timeline: the
# old value (T_MAX + 1) sat close enough to the horizon that a later cutoff
# swept it into the positive window, turning every never-event into a positive.
NEVER = 10 ** 9

_ONSET = ["kel", "vor", "tam", "zir", "quon", "bre", "nal", "sith", "dro", "phae",
          "mor", "iks", "lun", "cav", "reth", "obb", "yss", "tarn", "vex", "hul"]
_CODA = ["ith", "ax", "un", "orr", "el", "yne", "ash", "ost", "ir", "um",
         "ade", "ock", "ent", "ilt", "ov", "arn", "ex", "ule", "ip", "oth"]


def _name(rng: np.random.Generator, suffix: str = "", used: set | None = None) -> str:
    """Unique within a universe. Sampling with replacement produced duplicate
    capability and complement names, and since relationships render by name the
    dossier became ambiguous for a reader while the structured baseline still
    held the underlying ids - an information asymmetry in the baseline's favour.
    """
    for _ in range(200):
        s = rng.choice(_ONSET) + rng.choice(_CODA)
        if rng.random() < 0.4:
            s += "-" + str(rng.integers(2, 90))
        out = (s.capitalize() + (" " + suffix if suffix else "")).strip()
        if used is None or out not in used:
            if used is not None:
                used.add(out)
            return out
    raise RuntimeError("name space exhausted")


@dataclass
class Capability:
    id: str
    name: str
    vec: np.ndarray
    effect: str  # human-readable effect label, rendered in publications


@dataclass
class Product:
    id: str
    name: str
    company: str
    company_size: float  # drives recognition bias, NOT true activation
    cap_ids: list[str]
    profile: np.ndarray
    # Company -> Supply / Distribution / Moat -> Value Capture. A pairing can
    # activate commercially and still yield the originating company nothing:
    # technical possibility, commercial readiness and value capture are three
    # separate questions, and the benchmark must be able to separate them.
    supply: float = 1.0        # capacity to serve the need at volume
    distribution: float = 1.0  # reach into the population that has the need
    moat: float = 1.0          # ability to hold the position against entrants


@dataclass
class Need:
    id: str
    name: str
    bottleneck: str  # the capability-attribute story a publication describes
    req: np.ndarray
    threshold: float
    materiality: float
    complement_ids: list[str]


@dataclass
class Complement:
    id: str
    name: str
    ready_at: int  # quarter at which it becomes available; NEVER == never


@dataclass
class Pair:
    product_id: str
    need_id: str
    match: float
    p_true: float  # exact simulator probability of activation at any point
    p_horizon: float  # exact P(activate in (cutoff, cutoff+H] | not active at cutoff)
    activated_at: int  # NEVER if it never activates
    p_capture: float = 0.0  # exact P(originating company captures the value | active)
    captured: bool = False  # realized value capture


@dataclass
class Universe:
    seed: int
    capabilities: dict[str, Capability]
    products: dict[str, Product]
    needs: dict[str, Need]
    complements: dict[str, Complement]
    pairs: dict[tuple[str, str], Pair]
    mentions: dict[tuple[str, str], list[int]] = field(default_factory=dict)


def _sparse_vec(rng: np.random.Generator, k: int) -> np.ndarray:
    v = np.zeros(D)
    idx = rng.choice(D, size=k, replace=False)
    v[idx] = rng.uniform(0.4, 1.0, size=k)
    return v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def generate_universe(
    seed: int,
    n_products: int = 12,
    n_needs: int = 15,
    n_caps: int = 10,
    n_complements: int = 8,
    horizon: int = 8,
    cutoff: int = 20,
    slope: float = 9.0,
    comp_ready_override: dict[str, int] | None = None,
    mention_p_override: dict[tuple[str, str], float] | None = None,
) -> Universe:
    if cutoff + horizon >= T_MAX:
        raise ValueError(
            f"cutoff+horizon ({cutoff + horizon}) must stay inside the timeline "
            f"(T_MAX={T_MAX}); events after it are unobservable, not negative")

    rng = np.random.default_rng(seed)
    used: set[str] = set()  # every rendered name is unique within a universe

    caps = {}
    for i in range(n_caps):
        cid = f"C{i}"
        caps[cid] = Capability(
            id=cid,
            name=_name(rng, used=used),
            vec=_sparse_vec(rng, int(rng.integers(2, 4))),
            effect=_name(rng, "response", used),
        )

    comps = {}
    for i in range(n_complements):
        kid = f"K{i}"
        # roughly half are ready before the cutoff, half arrive later or never
        r = rng.random()
        if r < 0.45:
            ready = int(rng.integers(0, cutoff))
        elif r < 0.85:
            ready = int(rng.integers(cutoff, cutoff + horizon))
        else:
            ready = NEVER
        comps[kid] = Complement(id=kid, name=_name(rng, "substrate", used), ready_at=ready)

    # Counterfactual twin: override AFTER the rng has drawn every name and date, so
    # the twin shares its entire random stream with the base world (common random
    # numbers) and differs only through this one availability date.
    if comp_ready_override:
        for kid, ready in comp_ready_override.items():
            comps[kid].ready_at = int(ready)

    prods = {}
    for i in range(n_products):
        pid = f"P{i}"
        cids = list(rng.choice(list(caps), size=int(rng.integers(1, 4)), replace=False))
        profile = np.max(np.stack([caps[c].vec for c in cids]), axis=0)
        prods[pid] = Product(
            id=pid,
            name=_name(rng, "unit", used),
            company=_name(rng, "Consortium", used),
            company_size=float(np.exp(rng.normal(0, 1.0))),
            cap_ids=cids,
            profile=profile,
            supply=float(rng.uniform(0.1, 1.0)),
            distribution=float(rng.uniform(0.1, 1.0)),
            moat=float(rng.uniform(0.1, 1.0)),
        )

    needs = {}
    for j in range(n_needs):
        nid = f"N{j}"
        needs[nid] = Need(
            id=nid,
            name=_name(rng, "shortfall", used),
            bottleneck=_name(rng, "constraint", used),
            req=_sparse_vec(rng, int(rng.integers(2, 5))),
            threshold=float(rng.uniform(0.45, 0.72)),
            materiality=float(np.exp(rng.normal(0, 0.8))),
            complement_ids=list(
                rng.choice(list(comps), size=int(rng.integers(1, 3)), replace=False)
            ),
        )

    pairs = {}
    for pid, p in prods.items():
        for nid, n in needs.items():
            m = _cos(p.profile, n.req)
            tech = 1.0 / (1.0 + np.exp(-slope * (m - n.threshold)))
            # commercial gate: every complement must be ready by end of horizon
            ready = all(comps[k].ready_at <= cutoff + horizon for k in n.complement_ids)
            p_true = float(tech) if ready else float(tech) * 0.03

            # Exact conditional target for calibration: uniform arrival over the
            # feasible window, conditioned on the pair not being live at cutoff.
            lo0 = max(cutoff - 8, max(comps[k].ready_at for k in n.complement_ids))
            if lo0 > cutoff + horizon:
                p_h = 0.0
            else:
                span = cutoff + horizon - lo0 + 1
                n_before = max(0, cutoff - lo0 + 1)
                p_before = p_true * n_before / span
                p_after = p_true * (span - n_before) / span
                p_h = p_after / (1.0 - p_before) if p_before < 1.0 else 0.0

            # Common random numbers, drawn from a stream private to THIS pair.
            # Using the shared stream made the number of draws depend on p_true,
            # so a counterfactual that changed one complement shifted the stream
            # and silently re-rolled every later pair. Twin worlds must differ
            # only where the intervention reaches.
            prng = np.random.default_rng([seed, _pair_key(pid, nid)])
            u_act, u_arr = prng.random(), prng.random()

            activated = u_act < p_true
            if activated:
                # arrival is bounded below by the last complement to land; some
                # pairs realize before the cutoff and become the market's history
                lo = max(cutoff - 8, max(comps[k].ready_at for k in n.complement_ids))
                at = (
                    lo + int(u_arr * (cutoff + horizon + 1 - lo))
                    if lo <= cutoff + horizon
                    else NEVER
                )
            else:
                at = NEVER
            # value capture is conditional on activation and driven entirely by
            # company-side factors, so it is separable from demand discovery
            p_cap = float(np.clip(
                0.15 + 0.55 * p.supply * p.distribution + 0.30 * p.moat, 0.0, 1.0))
            captured = bool(at != NEVER and prng.random() < p_cap)
            pairs[(pid, nid)] = Pair(pid, nid, m, p_true, float(p_h), at,
                                     p_cap, captured)

    u = Universe(seed, caps, prods, needs, comps, pairs)
    u.mentions = _generate_mentions(u, seed, cutoff, mention_p_override)
    return u


def _pair_key(pid: str, nid: str) -> int:
    """Stable per-pair stream id (no PYTHONHASHSEED dependence)."""
    return int(pid[1:]) * 10_000 + int(nid[1:])


def _generate_mentions(
    u: Universe, seed: int, cutoff: int,
    p_override: dict[tuple[str, str], float] | None = None,
) -> dict[tuple[str, str], list[int]]:
    """Financial recognition: lagging and size-biased, only weakly tied to truth.

    A pair gets mentioned mostly because its company is large or because it has
    ALREADY activated. True future probability contributes only a little. This is
    what makes the recognition gap a real, exploitable signal rather than a
    relabelling of the answer.

    `p_override` supplies the p_true used for that weak term. Twin worlds pass the
    PRE-intervention values, so commentary reflects only what the market could
    have known before the counterfactual was applied. Without it, moving a
    complement's availability moved the mention counts too, and the intervention
    leaked into the very channel the twins are meant to show is uninformed.
    Mentions also draw from per-pair streams, so one pair's history cannot shift
    another's.
    """
    mentions: dict[tuple[str, str], list[int]] = {}
    for key, pr in u.pairs.items():
        size = u.products[pr.product_id].company_size
        p_for_mentions = (p_override or {}).get(key, pr.p_true)
        base = 0.02 + 0.10 * np.tanh(size) + 0.05 * p_for_mentions
        mrng = np.random.default_rng([seed, 999, _pair_key(*key)])
        qs = []
        for q in range(max(0, cutoff - 8), cutoff + 1):
            rate = base + (0.45 if pr.activated_at <= q else 0.0)
            if mrng.random() < min(rate, 0.9):
                qs.append(q)
        mentions[key] = qs
    return mentions
