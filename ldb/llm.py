"""LLM baseline: rendered corpus in, ranked product-need pairs out.

The task definition lives here — prompt, output schema, and the mapping from a
returned ranking to scores/probabilities. Which model actually answers is a
backend concern; see `ldb.backends`. Every backend sees the identical prompt and
schema, so backend rows in the results table are directly comparable.
"""

import json
from copy import deepcopy

import numpy as np

from .backends import get_backend
from .paths import elicited_metrics, path_metrics
from .task import Instance

TOP_K = 40

# Surface-specific vocabulary. The prompt must not reintroduce the financial
# framing that the opaque surface strips out, or the surface comparison would be
# testing the corpus while the instructions still said "sell-side" and "quarter".
LEXICON = {
    "finance": {"unit": "quarter", "units": "quarters",
                "commentary": "sell-side commentary", "pairing": "product-need"},
    "opaque": {"unit": "step", "units": "steps",
               "commentary": "reference counts", "pairing": "item-pair"},
    "space_opera": {"unit": "cycle", "units": "cycles",
                    "commentary": "council chatter", "pairing": "artifact-need"},
    "anime_tech": {"unit": "arc", "units": "arcs",
                   "commentary": "guild rumours", "pairing": "frame-need"},
    "cyberpunk": {"unit": "quarter", "units": "quarters",
                  "commentary": "fixer traffic", "pairing": "deck-need"},
}

SYSTEM = """You are an analyst working inside a synthetic economy.

You are given a dossier of evidence available at a cutoff {unit}. Your job is to
predict which {pairing} pairs will become COMMERCIALLY REALIZED within the next
{horizon} {units}.

No document tells you whether a pairing will be realized. Some pairings are named
in {commentary}; that records attention already paid to them, not evidence that
they will be realized, and attention tracks the size of the holder and things that
have already happened. Ranking by how much a pairing is discussed will not work.

What the evidence does support is a chain you must assemble yourself:

  item -> its capabilities -> those capabilities' measured attribute profile
  need -> its bottleneck -> the attribute profile that relieves it

A pairing is technically viable when the first item's combined attribute profile
aligns with the need's required profile above that need's stated sufficiency
threshold. Technical viability is not sufficient: a need also cannot be served
until every complement it depends on is available. A complement with no announced
availability, or one arriving after the horizon, blocks realization no matter how
good the profile alignment.

Pairings listed as already deployed are history and are not candidates; every
other pairing is. Do not rank a pairing that the dossier records as already
deployed.

Return TWO things.

(1) `predictions`: your ranking, ordered by probability of commercial
realization within the horizon. Rank by that probability alone - not by value,
not by who captures it.

(2) `probes`: answers for the specific pairings listed under PROBES below.
These are fixed in advance and are not a ranking; answer every one.

For each entry in either list, report:
  probability   - that it becomes commercially realized within the horizon
  materiality   - the value if served, on the scale the dossier reports
  capture       - probability the holder captures that value rather than losing
                  it to entrants, given it is realized. This depends on supply
                  capacity, distribution reach and defensibility, not on whether
                  the need gets served at all
  recognized    - whether this pairing is ALREADY discussed in {commentary}
  prerequisites - the ids of any complements still blocking it, [] if none

Materiality, capture and recognition do not change your ranking order in (1);
they are separate judgements about the same pairing.

Answer from the dossier text alone. Do not read files, run commands, or search."""

TASK = """{corpus}

===

Cutoff {unit}: {cutoff}. Horizon: the next {horizon} {units}.

Rank the {k} pairs most likely to become commercially realized in
({unit} {cutoff}, {unit} {end}]. Use the [PT-*] and [PR-*] bracket tags as ids
(e.g. product_id "P3", need_id "N7").

For each, give a calibrated probability and the evidence path you composed.

PROBES (answer all of these, in `probes`; they are not part of the ranking):
{probes}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "predictions": {
            "type": "array",
            "maxItems": TOP_K,
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "need_id": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "materiality": {"type": "number", "minimum": 0.0},
                    "capture": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "recognized": {"type": "boolean"},
                    "prerequisites": {"type": "array", "items": {"type": "string"}},
                    "evidence_path": {"type": "string"},
                },
                "required": ["product_id", "need_id", "probability", "materiality",
                             "capture", "recognized", "prerequisites", "evidence_path"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["predictions", "probes"],
    "additionalProperties": False,
}
SCHEMA["properties"]["probes"] = {
    "type": "array",
    "items": SCHEMA["properties"]["predictions"]["items"],
}

UNRANKED_P = 0.02  # probability imputed for anything outside the top-K


def response_schema(top_k: int = TOP_K) -> dict:
    """Return an independent schema with the requested ranking budget."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    schema = deepcopy(SCHEMA)
    schema["properties"]["predictions"]["maxItems"] = top_k
    return schema


def apply_top_k_protocol(scores: dict, probs: dict, k: int = TOP_K):
    """Hold a ranker to the same top-K contract the model is given.

    A model returns K pairs; a baseline scoring all ~172 was being measured
    under a different contract, so AUC and probability MSE partly reflected
    truncation and the imputed constant rather than ranking quality. Every row
    in a table must be produced under one protocol.
    """
    top = sorted(scores, key=lambda x: -scores[x])[:k]
    rank = {key: float(k - i) for i, key in enumerate(top)}
    return ({key: rank.get(key, 0.0) for key in scores},
            {key: probs[key] if key in rank else UNRANKED_P for key in probs})


def build_system(surface: str = "finance", horizon: int = 8) -> str:
    """Horizon comes from the instance; hardcoding 8 made the system prompt
    contradict the task text whenever a non-default horizon was used."""
    return SYSTEM.format(horizon=horizon, **LEXICON[surface])


def average_replicates(reps: list, top_k: int = TOP_K) -> tuple[dict, dict, dict]:
    """Average completions, then enforce the final ranking's shared budget.

    Only pairs surfaced by a completion can enter the ensemble. Ties in the
    averaged scores are retained; candidate order breaks a tie at the K boundary.
    The archive keeps all completions, including pairs outside the final list.
    """
    if not reps or top_k < 1:
        raise ValueError("at least one completion and a positive top_k are required")
    if len(reps) == 1:
        return reps[0]
    scores = {k: float(np.mean([r[0][k] for r in reps])) for k in reps[0][0]}
    probs = {k: float(np.mean([r[1][k] for r in reps])) for k in reps[0][1]}
    selected = set(sorted((k for k in scores if scores[k] > 0),
                          key=lambda k: -scores[k])[:top_k])
    scores = {k: v if k in selected else 0.0 for k, v in scores.items()}
    probs = {k: v if k in selected else UNRANKED_P for k, v in probs.items()}
    meta = {}
    # Scalar diagnostics describe the mean completion; undefined diagnostics
    # are excluded, as in universe aggregation. n_ranked instead describes the
    # final ensemble, while mean_n_ranked preserves completion-level coverage.
    numeric = {k for r in reps for k, v in r[2].items()
               if isinstance(v, (int, float)) and k != "n_ranked"}
    for key in sorted(numeric):
        vals = [r[2][key] for r in reps if isinstance(r[2].get(key), (int, float))
                and np.isfinite(r[2][key])]
        meta[key] = float(np.mean(vals)) if vals else float("nan")
    for key in ("ev_scores", "cap_scores", "probe_scores"):
        keys = {k for r in reps for k in r[2].get(key, {})}
        if keys:
            meta[key] = {
                k: float(np.mean([r[2].get(key, {}).get(k, 0.0) for r in reps]))
                if key == "probe_scores" or k in selected else 0.0
                for k in sorted(keys)
            }
    meta["mean_n_ranked"] = float(np.mean([r[2].get("n_ranked", 0) for r in reps]))
    meta["n_ranked"] = len(selected)
    meta["replicates"] = len(reps)
    meta["predictions"] = [r[2].get("predictions") for r in reps]
    meta["raw_response"] = [r[2].get("raw_response") for r in reps]
    return scores, probs, meta


def build_task(inst: Instance, top_k: int = TOP_K) -> str:
    lex = LEXICON[inst.surface]
    return TASK.format(
        corpus=inst.corpus, cutoff=inst.cutoff, horizon=inst.horizon,
        end=inst.cutoff + inst.horizon, k=top_k,
        unit=lex["unit"], units=lex["units"],
        probes="\n".join(f"  - product_id {p}, need_id {n}" for p, n in inst.probes),
    )


def score_predictions(inst: Instance, preds: list[dict],
                      top_k: int = TOP_K) -> tuple[dict, dict, dict]:
    """Ranked predictions -> (scores, probs). Unranked candidates get a floor.

    The top-K contract is enforced here, not merely requested in the prompt: a
    model that returned every candidate would be ranking the whole set while
    others rank K, and the resulting AUC would not be comparable across models.
    Duplicates keep their first (best) position rather than being overwritten.
    """
    seen, kept = set(), []
    for p in preds:
        key = (p["product_id"], p["need_id"])
        if key in seen or key not in inst.label:
            continue  # duplicate, hallucinated, or already-realized pair
        seen.add(key)
        kept.append((key, p))
        if len(kept) == top_k:
            break

    scores = {k: 0.0 for k in inst.candidates}
    probs = {k: UNRANKED_P for k in inst.candidates}
    # rank position carries the signal; probability is only the calibration target
    for i, (key, p) in enumerate(kept):
        scores[key] = float(len(kept) - i)
        probs[key] = min(max(float(p["probability"]), 0.0), 1.0)
    meta = dict(path_metrics(inst, kept))
    meta["n_ranked"] = len(kept)
    meta["n_returned"] = len(preds)
    # the model's own output, so a published artifact can be re-scored and
    # audited without access to the local response cache
    meta["predictions"] = [
        {"product_id": key[0], "need_id": key[1],
         "probability": p.get("probability"),
         "materiality": p.get("materiality"),
         "capture": p.get("capture"),
         "recognized": p.get("recognized"),
         "prerequisites": p.get("prerequisites", []),
         "evidence_path": p.get("evidence_path", "")}
        for key, p in kept
    ]
    return scores, probs, meta


def value_rankings(inst: Instance, preds: list[dict]) -> tuple[dict, dict]:
    """Separate orders on validated top-K predictions, one per value objective."""
    ev = {k: 0.0 for k in inst.candidates}
    capture = dict(ev)
    for p in preds:
        key = (p.get("product_id"), p.get("need_id"))
        if key in ev:
            probability = min(max(float(p.get("probability") or 0.0), 0.0), 1.0)
            ev[key] = probability * float(p.get("materiality") or 0.0)
            capture[key] = probability * float(p.get("capture") or 0.0)
    return ev, capture


def score_response(inst: Instance, raw: str, top_k: int = TOP_K):
    """Score an archived or fresh completion independently of its transport."""
    parsed = json.loads(raw)
    scores, probs, meta = score_predictions(inst, parsed["predictions"], top_k)
    # auxiliary questions are scored on the fixed probe set, never on the
    # model's self-selected top-K
    probes = {}
    for p in parsed.get("probes", []):
        probes.setdefault((p.get("product_id"), p.get("need_id")), p)
    # Omitted probes are scored as worst-case, not skipped: evaluating only the
    # intersection let a model drop the hard ones and keep a strong average.
    answered = [(k, probes[k]) for k in inst.probes if k in probes]
    missing = [k for k in inst.probes if k not in probes]
    meta.update(elicited_metrics(inst, answered, n_missing=len(missing)))
    meta["probe_coverage"] = len(answered) / max(len(inst.probes), 1)
    # Probe AUC measures the separately requested probabilities, not whether
    # these pairs appeared in the main ranking. Missing answers receive zero,
    # matching the human-task scorer; coverage is reported independently.
    meta["probe_scores"] = {
        k: min(max(float(probes[k]["probability"]), 0.0), 1.0) if k in probes else 0.0
        for k in inst.probes
    }
    meta["ev_scores"], meta["cap_scores"] = value_rankings(inst, meta["predictions"])
    meta["raw_response"] = raw
    return scores, probs, meta


def failed_response(inst: Instance):
    """A failed call has no archived response and answers none of the probes."""
    scores, probs, meta = score_response(inst, '{"predictions": [], "probes": []}')
    meta["raw_response"] = None
    return scores, probs, meta


def llm_rank(
    inst: Instance, backend: str = "codex", model: str | None = None,
    effort: str | None = None, replicate: int = 0, top_k: int = TOP_K,
) -> tuple[dict, dict, dict]:
    raw = get_backend(backend).complete(
        system=build_system(inst.surface, inst.horizon),
        task=build_task(inst, top_k), schema=response_schema(top_k),
        model=model, effort=effort, replicate=replicate,
    )
    return score_response(inst, raw, top_k)
