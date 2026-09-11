"""Two mechanistic claims restated as TESTED INTERACTIONS, not as a pair of
one-sided tests.

A reviewer objected, correctly, that "significant in model A, not significant in
model B" is not evidence that A and B differ, and that "the deficit is on the
mentioned pairings, not the silent ones" is not evidence that the deficit
differs between the two subsets. Both objections are answered by differencing
the two effects and bootstrapping the difference, which is what this tool does.
No API calls: everything is read off artifacts already on disk.

(1) ASSIST x MODEL. The paper reports, per model, the paired AUC gain from
    supplying an assist (composed alignments / resolved readiness / both) over
    the base finance task. Both models ran the SAME 30 universe seeds
    (`config.llm_seeds`), so the assist delta is formed per universe within each
    model and the two delta vectors are then differenced universe by universe:

        ( assist_luna[u] - base_luna[u] ) - ( assist_terra[u] - base_terra[u] )

    and bootstrapped with `ldb.metrics.paired_diff`. Only if that interval
    excludes zero may we say the assist HELPS ONE MODEL MORE THAN THE OTHER.

(2) SUBSET x MODEL, within a model. The paper reports each model's paired
    deficit vs `fair_reference` on the zero-mention subset (`gap_auc`, from the
    main finance artifact) and on the mentioned subset (`mentioned_auc`, from
    `results/probe_subset.json`). The localisation claim needs the difference of
    those two deficits, per model:

        ( model_gap_auc[u]  - fair_gap_auc[u] )
      - ( model_ment_auc[u] - fair_ment_auc[u] )

    PAIRING IS BY UNIVERSE, NOT BY CANDIDATE. The zero-mention subset and the
    mentioned subset are disjoint candidate sets inside the same universe, so
    the two AUCs are not computed on comparable items; what they share is the
    universe (its products, needs, latent structure and difficulty), and the
    universe is therefore the resampling unit, exactly as elsewhere in the
    repo. A universe contributing NaN on either subset (e.g. no mentioned
    pairing exists) is dropped by `bootstrap_ci`, and the surviving count is
    reported as `n_used`.

Baseline rows are stored for every evaluation universe while model rows are
stored only for the seeds the model ran, so baselines are restricted to
`config.llm_seeds` before differencing - the same restriction
`tools/paper_tables.py:paired()` applies.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ldb.metrics import paired_diff

REFERENCE = "fair_reference"
ASSIST_ARMS = ["assist_composition", "assist_gate", "assist_both"]
ASSIST_METRIC = "auc"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def model_key(art: dict) -> str:
    """The single non-baseline ranker in an artifact."""
    base = set(art["results"]) & {"random", "recognition", "structured_evidence",
                                  "fair_reference", "expected_value",
                                  "hidden_oracle"}
    keys = [k for k in art["results"] if k not in base]
    if len(keys) != 1:
        raise SystemExit(f"expected exactly one model row, found {keys}")
    return keys[0]


def model_rows(art: dict) -> list[dict]:
    """Per-universe rows for the model arm, in `llm_seeds` order."""
    return art["per_universe"][model_key(art)]


def baseline_rows(art: dict, name: str) -> list[dict]:
    """Baseline rows restricted to the seeds the model arm actually ran."""
    seeds = (art.get("config") or {}).get("llm_seeds") or []
    rows = art["per_universe"][name]
    return [rows[s] for s in seeds] if seeds else rows


def wrap(values: list[float], key: str = "d") -> list[dict]:
    """Per-universe scalars as metric rows, so `paired_diff` can consume them."""
    return [{key: v} for v in values]


def ci(mean, lo, hi, n_used: int, n: int) -> dict:
    return {"mean": mean, "lo": lo, "hi": hi, "n": n, "n_used": n_used,
            "excludes_zero": bool(lo == lo and (lo > 0 or hi < 0))}


def n_finite(*vecs) -> int:
    return sum(1 for t in zip(*vecs) if all(x == x for x in t))


# ------------------------------------------------------------------ (1)

def assist_interaction(dir_a: str, dir_b: str) -> dict:
    base_a, base_b = load(f"{dir_a}/surface_finance.json"), load(f"{dir_b}/surface_finance.json")
    seeds_a = base_a["config"]["llm_seeds"]
    seeds_b = base_b["config"]["llm_seeds"]
    if seeds_a != seeds_b:
        raise SystemExit("the two models ran different universe seeds; the "
                         "difference of assist deltas would not be paired")
    ba, bb = model_rows(base_a), model_rows(base_b)

    out = {"models": [base_a["config"]["backend"]["model"],
                      base_b["config"]["backend"]["model"]],
           "metric": ASSIST_METRIC, "seeds": seeds_a, "arms": {}}
    for arm in ASSIST_ARMS:
        aa, ab = load(f"{dir_a}/{arm}.json"), load(f"{dir_b}/{arm}.json")
        if aa["config"]["llm_seeds"] != seeds_a or ab["config"]["llm_seeds"] != seeds_a:
            raise SystemExit(f"{arm}: seed set differs from the base task")
        ra = model_rows(aa)
        rb = model_rows(ab)
        da = [x[ASSIST_METRIC] - y[ASSIST_METRIC] for x, y in zip(ra, ba)]
        db = [x[ASSIST_METRIC] - y[ASSIST_METRIC] for x, y in zip(rb, bb)]
        within_a = paired_diff(ra, ba, ASSIST_METRIC)
        within_b = paired_diff(rb, bb, ASSIST_METRIC)
        dd = paired_diff(wrap(da), wrap(db), "d")
        out["arms"][arm] = {
            "within_" + out["models"][0]: ci(*within_a, n_finite(da), len(da)),
            "within_" + out["models"][1]: ci(*within_b, n_finite(db), len(db)),
            "difference_in_differences": ci(*dd, n_finite(da, db), len(da)),
        }
    return out


# ------------------------------------------------------------------ (2)

def subset_interaction(fin_rel: str, subset: dict) -> dict:
    """(gap deficit) - (mentioned deficit) for one model, paired by universe."""
    art = load(fin_rel)
    mk = model_key(art)
    model = art["config"]["backend"]["model"]
    seeds = art["config"]["llm_seeds"]

    mg = [r["gap_auc"] for r in model_rows(art)]
    fg = [r["gap_auc"] for r in baseline_rows(art, REFERENCE)]

    sm = subset["results"][model]["per_universe"]
    sf = subset["results"][REFERENCE]["per_universe"]
    missing = [s for s in seeds if str(s) not in sm or str(s) not in sf]
    if missing:
        raise SystemExit(f"{model}: probe_subset lacks universes {missing}")
    mm = [sm[str(s)]["mentioned_auc"] for s in seeds]
    fm = [sf[str(s)]["mentioned_auc"] for s in seeds]

    gap = [a - b for a, b in zip(mg, fg)]          # deficit on silent pairings
    ment = [a - b for a, b in zip(mm, fm)]         # deficit on discussed pairings
    inter = paired_diff(wrap(gap), wrap(ment), "d")
    return {
        "model": model, "arm": fin_rel, "ranker": mk, "seeds": seeds,
        "gap_auc_deficit": ci(*paired_diff(model_rows(art),
                                           baseline_rows(art, REFERENCE), "gap_auc"),
                              n_finite(gap), len(gap)),
        "mentioned_auc_deficit": ci(*paired_diff(wrap(mm), wrap(fm), "d"),
                                    n_finite(ment), len(ment)),
        "interaction_gap_minus_mentioned": ci(*inter, n_finite(gap, ment), len(gap)),
        "vs": REFERENCE,
        "pairing": "by universe; the two subsets are disjoint candidate sets "
                   "within the same universe, so items are not comparable but "
                   "the universe is",
    }


def fmt(d: dict) -> str:
    s = f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  n={d['n_used']}"
    return s + ("  EXCLUDES 0" if d["excludes_zero"] else "  includes 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", default="results/luna_full")
    ap.add_argument("--dir-b", default="results/terra_full")
    ap.add_argument("--subset", default="results/probe_subset.json")
    ap.add_argument("--arms", nargs="+",
                    default=["results/luna_full/surface_finance.json",
                             "results/terra_full/surface_finance.json",
                             "results/openweight/model_finance.json"])
    ap.add_argument("--out", default=str(ROOT / "results" / "interactions.json"))
    args = ap.parse_args()

    from run_v0 import _code_hash
    code_hash = _code_hash()

    subset = load(args.subset)
    if subset["config"]["code_hash"] != code_hash:
        print(f"WARNING: {args.subset} was written at code_hash "
              f"{subset['config']['code_hash']}, now {code_hash}")

    assist = assist_interaction(args.dir_a, args.dir_b)
    localisation = {}
    for rel in args.arms:
        r = subset_interaction(rel, subset)
        localisation[r["model"]] = r

    a, b = assist["models"]
    print(f"\n(1) assist x model, metric={assist['metric']}, "
          f"n={len(assist['seeds'])} shared universes")
    print(f"    difference in differences = ({a} assist gain) - ({b} assist gain)")
    for arm, d in assist["arms"].items():
        print(f"  {arm}")
        print(f"    within {a:16s} {fmt(d['within_' + a])}")
        print(f"    within {b:16s} {fmt(d['within_' + b])}")
        print(f"    DiD    {'':16s} {fmt(d['difference_in_differences'])}")

    print(f"\n(2) subset x model, within model, vs {REFERENCE}, paired by universe")
    print("    interaction = (zero-mention deficit) - (mentioned deficit)")
    for m, r in localisation.items():
        print(f"  {m}")
        print(f"    gap_auc deficit        {fmt(r['gap_auc_deficit'])}")
        print(f"    mentioned_auc deficit  {fmt(r['mentioned_auc_deficit'])}")
        print(f"    INTERACTION            {fmt(r['interaction_gap_minus_mentioned'])}")

    if _code_hash() != code_hash:
        raise SystemExit("source changed during the run; refusing to write")
    Path(args.out).write_text(json.dumps(
        {"assist_by_model": assist, "subset_by_model": localisation,
         "config": {"code_hash": code_hash, "reference": REFERENCE,
                    "assist_metric": ASSIST_METRIC,
                    "dirs": [args.dir_a, args.dir_b], "arms": args.arms,
                    "subset_artifact": args.subset,
                    "subset_artifact_code_hash": subset["config"]["code_hash"],
                    "pairing": "universe is the resampling unit for every "
                               "bootstrap; NaN universes dropped, n_used given"}},
        indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
