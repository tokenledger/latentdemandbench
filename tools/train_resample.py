"""Is one training universe enough, or was *this* one training universe enough?

`tools/train_curve.py` refits the fair reference on the first n seeds of the
training block and reports a bootstrap interval over EVALUATION universes. That
interval says nothing about the choice of training set: at n=1 it describes one
particular universe, seed 1000, and a reviewer is right that the resulting claim
is "this single universe sufficed", not "a single universe suffices".

This script resamples the training set itself. For each size n it draws R random
training sets of n universes from a held-out pool, refits `ldb.fair.fit` on each
draw, and scores every draw on the same 100 evaluation universes under the same
top-K protocol. The spread reported here is ACROSS DRAWS - a training-set
interval - and is a different quantity from the evaluation bootstrap in Figure 2.
Both are needed; neither substitutes for the other.

Seed pool. The deployed reference (`ldb.baselines.fair_reference`) is fitted on
seeds 1000-1099. Ten disjoint draws of n=100 need more than 100 seeds, so the
pool is extended to 1000-1999. Every evaluation set in the paper uses seeds
0..99 (see `universe_seeds` in any results config), so the whole extended pool
remains disjoint from evaluation by construction, exactly as 1000-1099 was: no
fitted ranker can ever have seen an instance it is scored on.

Matched-universe comparison. The 100-universe evaluation above is a stability
check on the fit, but it is NOT a fair comparison against the model arms: those
arms were only run on the first 30 universes (`config.llm_seeds` is 0..29 in
every base-task artifact), so comparing a 100-universe reference mean against a
30-universe model mean mixes a real difference with a difference in which
universes were scored. Between-universe difficulty dominates the variance here,
which is exactly why the rest of the paper pairs. So each draw is ALSO scored on
the same 30 universes the models saw, and compared to each model arm with
`ldb.metrics.paired_diff` on those matched universes. Reported per size n:
median paired difference, 5th/95th percentiles of it ACROSS DRAWS, and the
fraction of draws whose paired mean difference is positive. The three
single-completion base-task arms and the three-completion terra arm
(`results/terra_full/replicates_3.json`) are all included, because "beats every
model arm" and "beats every single-completion arm" are different claims.

Model-free, so it costs nothing to re-run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ldb.fair import fit
from ldb.llm import TOP_K, apply_top_k_protocol
from ldb.metrics import evaluate, paired_diff
from ldb.task import build_instance

ROOT = Path(__file__).resolve().parent.parent

# held-out training pool, disjoint from every evaluation set (which uses 0..99)
POOL_LO, POOL_HI = 1000, 2000

# draw counts: more draws where a single draw is noisiest
DEFAULT_DRAWS = {1: 50, 2: 30, 5: 20, 10: 20, 30: 20, 100: 10}

# the three fixed model scores the fits are compared against, read from disk
MODEL_SOURCES = {
    "luna": "results/luna_full/surface_finance.json",
    "terra": "results/terra_full/surface_finance.json",
    "qwen": "results/openweight/model_finance.json",
}
# the three single-completion arms above plus the stronger three-completion terra
# row: "beats every model arm" is false against it, so it is scored separately
# and reported alongside rather than folded into the all-three fraction.
EXTRA_MODEL_SOURCES = {
    "terra_rep3": "results/terra_full/replicates_3.json",
}
BASELINE_KEYS = {"random", "recognition", "structured_evidence",
                 "fair_reference", "expected_value", "hidden_oracle"}


def _model_key(doc: dict, rel: str) -> str:
    keys = [k for k in doc["results"] if k not in BASELINE_KEYS]
    if len(keys) != 1:
        raise SystemExit(f"{rel}: expected one model row, found {keys}")
    return keys[0]


def read_model_scores(sources: dict) -> dict:
    """AUC of each model arm; the model row is the one key that is not a baseline.

    Also carries the arm's per-universe rows, restricted to the universes the
    arm actually ran on (`config.llm_seeds`), for the paired comparison.
    """
    out = {}
    for tag, rel in sources.items():
        doc = json.loads((ROOT / rel).read_text())
        key = _model_key(doc, rel)
        seeds = [int(s) for s in doc["config"]["llm_seeds"]]
        rows = doc["per_universe"][key]
        matched = [rows[s] for s in seeds]
        bad = [s for s, r in zip(seeds, matched)
               if not isinstance(r.get("auc"), (int, float))]
        if bad:
            raise SystemExit(f"{rel}: no AUC on llm seeds {bad}")
        out[tag] = {
            "key": key,
            "auc": float(doc["results"][key]["auc"]["mean"]),
            "source": rel,
            "llm_seeds": seeds,
            "matched_rows": matched,
            "matched_auc": float(np.mean([r["auc"] for r in matched])),
            "replicates": int(doc["config"].get("replicates", 1)),
        }
    return out


def draw_sets(rng: np.random.Generator, pool: np.ndarray, n: int, reps: int):
    """`reps` training sets of `n` seeds, disjoint across draws where possible.

    If reps * n fits in the pool the draws are consecutive blocks of one shuffle,
    hence pairwise disjoint. Otherwise each draw is an independent sample without
    replacement - disjoint within a draw, necessarily overlapping between them.
    """
    if reps * n <= len(pool):
        perm = rng.permutation(pool)
        return [perm[i * n:(i + 1) * n].tolist() for i in range(reps)]
    return [rng.choice(pool, size=n, replace=False).tolist() for _ in range(reps)]


def per_universe_auc(ref, ev, top_k: int) -> list[float]:
    """AUC of a fitted reference on each evaluation universe, in order."""
    aucs = []
    for i, inst in enumerate(ev):
        raw = ref(inst, seed=i)
        sc, pr = apply_top_k_protocol(raw, raw, top_k)
        aucs.append(float(evaluate(inst, sc, probs=pr)["auc"]))
    return aucs


def score(ref, ev, top_k: int) -> float:
    """Mean AUC of a fitted reference over the evaluation universes."""
    return float(np.mean(per_universe_auc(ref, ev, top_k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[1, 2, 5, 10, 30, 100])
    ap.add_argument("--draws", type=int, nargs="+", default=None,
                    help="draws per size, aligned with --sizes; "
                         "default 50/30/20/20/20/10")
    ap.add_argument("--eval-universes", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--seed", type=int, default=20260810,
                    help="seed for the numpy Generator that draws training sets")
    ap.add_argument("--pool", type=int, nargs=2, default=[POOL_LO, POOL_HI],
                    metavar=("LO", "HI"), help="half-open held-out seed range")
    ap.add_argument("--out", default=str(ROOT / "results" / "train_resample.json"))
    args = ap.parse_args()

    if args.draws is not None and len(args.draws) != len(args.sizes):
        raise SystemExit("--draws must have one entry per --sizes entry")
    reps = (dict(zip(args.sizes, args.draws)) if args.draws
            else {n: DEFAULT_DRAWS.get(n, 20) for n in args.sizes})

    from run_v0 import _code_hash
    code_hash = _code_hash()

    lo, hi = args.pool
    if lo < args.eval_universes:
        raise SystemExit("training pool overlaps the evaluation seeds")
    pool = np.arange(lo, hi)
    rng = np.random.default_rng(args.seed)
    models = read_model_scores(MODEL_SOURCES)
    extra = read_model_scores(EXTRA_MODEL_SOURCES)
    all_models = {**models, **extra}

    # every arm must have run on the same universes, or "paired" means nothing
    matched_seeds = models["luna"]["llm_seeds"]
    for tag, m in all_models.items():
        if m["llm_seeds"] != matched_seeds:
            raise SystemExit(f"{tag}: llm_seeds {m['llm_seeds'][:5]}... differ "
                             f"from luna's; cannot pair")
    if max(matched_seeds) >= args.eval_universes:
        raise SystemExit(f"--eval-universes {args.eval_universes} does not cover "
                         f"the model seeds 0..{max(matched_seeds)}")

    print(f"training-set resampling: pool {lo}-{hi - 1}, rng seed {args.seed}")
    print("model bars: " + ", ".join(
        f"{t} {m['auc']:.4f}" for t, m in all_models.items()))
    print(f"matched universes (config.llm_seeds): {len(matched_seeds)} seeds "
          f"{matched_seeds[0]}..{matched_seeds[-1]}")

    ev = [build_instance(s) for s in range(args.eval_universes)]
    cache: dict[int, object] = {}

    def inst(s: int):
        if s not in cache:
            cache[s] = build_instance(int(s))
        return cache[s]

    per_n = {}
    headline = None
    headline_matched = None
    single_tags = list(MODEL_SOURCES)
    for n in args.sizes:
        r = reps[n]
        sets = draw_sets(rng, pool, n, r)
        disjoint = r * n <= len(pool)
        curves = [per_universe_auc(fit([inst(s) for s in ts]), ev, args.top_k)
                  for ts in sets]
        aucs = np.array([float(np.mean(c)) for c in curves])

        # --- matched-universe paired comparison (seeds the models actually ran)
        matched_rows = [[{"auc": c[s]} for s in matched_seeds] for c in curves]
        matched_auc = np.array([float(np.mean([x["auc"] for x in rows_]))
                                for rows_ in matched_rows])
        paired = {}
        beats = {}
        for tag, m in all_models.items():
            # paired_diff returns (mean, lo, hi) of the reference-minus-model
            # per-universe difference; the mean is the per-draw paired statistic
            stats = [paired_diff(rows_, m["matched_rows"], "auc", seed=d)
                     for d, rows_ in enumerate(matched_rows)]
            diffs = np.array([s[0] for s in stats])
            beats[tag] = diffs > 0
            paired[tag] = {
                "model_key": m["key"],
                "model_source": m["source"],
                "model_matched_auc": m["matched_auc"],
                "median": float(np.median(diffs)),
                "p5": float(np.percentile(diffs, 5)),
                "p95": float(np.percentile(diffs, 95)),
                "mean": float(diffs.mean()),
                "min": float(diffs.min()), "max": float(diffs.max()),
                "frac_positive": float(np.mean(diffs > 0)),
                "diffs": [float(x) for x in diffs],
                "ci_lo": [float(s[1]) for s in stats],
                "ci_hi": [float(s[2]) for s in stats],
                "frac_ci_excludes_zero_positive": float(
                    np.mean([s[1] > 0 for s in stats])),
            }
        all_three = np.logical_and.reduce([beats[t] for t in single_tags])
        matched_block = {
            "eval_seeds": list(matched_seeds),
            "n_eval": len(matched_seeds),
            "median_auc": float(np.median(matched_auc)),
            "p5_auc": float(np.percentile(matched_auc, 5)),
            "p95_auc": float(np.percentile(matched_auc, 95)),
            "aucs": [float(a) for a in matched_auc],
            "paired": paired,
            "frac_beats_all_single_completion": float(np.mean(all_three)),
            "frac_beats_all_single_completion_and_rep3": float(
                np.mean(all_three & beats["terra_rep3"])),
            "single_completion_tags": single_tags,
        }

        row = {
            "matched": matched_block,
            "draws": r, "n_train": n, "disjoint_draws": bool(disjoint),
            "median": float(np.median(aucs)),
            "p5": float(np.percentile(aucs, 5)),
            "p95": float(np.percentile(aucs, 95)),
            "min": float(aucs.min()), "max": float(aucs.max()),
            "mean": float(aucs.mean()), "std": float(aucs.std(ddof=1)) if r > 1 else 0.0,
            "exceed": {t: float(np.mean(aucs > m["auc"])) for t, m in models.items()},
            "exceed_both_frontier": float(np.mean(
                (aucs > models["luna"]["auc"]) & (aucs > models["terra"]["auc"]))),
            "aucs": [float(a) for a in aucs],
            "seed_sets": [[int(s) for s in ts] for ts in sets],
        }
        per_n[str(n)] = row
        if n == 1:
            headline = row["exceed_both_frontier"]
        if n == 5:
            headline_matched = matched_block

    hdr = (f"{'n':>4} {'R':>4} {'median':>8} {'p5':>8} {'p95':>8} "
           f"{'min':>8} {'max':>8} " + " ".join(f"{'>' + t:>8}" for t in models)
           + f" {'>both':>8}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for n in args.sizes:
        d = per_n[str(n)]
        print(f"{n:>4} {d['draws']:>4} {d['median']:>8.4f} {d['p5']:>8.4f} "
              f"{d['p95']:>8.4f} {d['min']:>8.4f} {d['max']:>8.4f} "
              + " ".join(f"{d['exceed'][t]:>8.2f}" for t in models)
              + f" {d['exceed_both_frontier']:>8.2f}")
    print("\np5/p95 are percentiles ACROSS DRAWS (training-set spread), not an "
          "evaluation bootstrap.")
    if headline is not None:
        print(f"single-universe fits outranking both frontier arms: "
              f"{100 * headline:.1f}%")

    print(f"\nMATCHED UNIVERSES ({len(matched_seeds)} seeds "
          f"{matched_seeds[0]}..{matched_seeds[-1]}, the ones the models ran on)")
    print("paired per-universe difference, reference minus model; median and "
          "p5/p95 ACROSS DRAWS; frac+ = draws with a positive paired mean.")
    for tag in list(models) + list(extra):
        m = all_models[tag]
        print(f"\nvs {tag}  ({m['key']}, matched AUC {m['matched_auc']:.4f}, "
              f"replicates={m['replicates']})")
        h = (f"{'n':>4} {'R':>4} {'median':>9} {'p5':>9} {'p95':>9} "
             f"{'min':>9} {'max':>9} {'frac+':>7}")
        print(h)
        print("-" * len(h))
        for n in args.sizes:
            p = per_n[str(n)]["matched"]["paired"][tag]
            print(f"{n:>4} {per_n[str(n)]['draws']:>4} {p['median']:>9.4f} "
                  f"{p['p5']:>9.4f} {p['p95']:>9.4f} {p['min']:>9.4f} "
                  f"{p['max']:>9.4f} {p['frac_positive']:>7.2f}")

    h = (f"\n{'n':>4} {'R':>4} {'beats all 3 single':>20} "
         f"{'beats 3-completion terra':>26} {'beats all 4':>12}")
    print(h)
    for n in args.sizes:
        mb = per_n[str(n)]["matched"]
        print(f"{n:>4} {per_n[str(n)]['draws']:>4} "
              f"{mb['frac_beats_all_single_completion']:>20.2f} "
              f"{mb['paired']['terra_rep3']['frac_positive']:>26.2f} "
              f"{mb['frac_beats_all_single_completion_and_rep3']:>12.2f}")
    if headline_matched is not None:
        print(f"\nfive-universe draws beating all three single-completion arms "
              f"on matched universes: "
              f"{100 * headline_matched['frac_beats_all_single_completion']:.1f}%")
        print(f"five-universe draws beating the three-completion terra arm "
              f"({all_models['terra_rep3']['matched_auc']:.4f}): "
              f"{100 * headline_matched['paired']['terra_rep3']['frac_positive']:.1f}%")

    if _code_hash() != code_hash:
        raise SystemExit("source changed during the run; refusing to write")
    drop = ("matched_rows",)  # 30 full metric rows per arm; already on disk
    payload = {
        "per_n": per_n,
        "model_scores": {t: {k: v for k, v in m.items() if k not in drop}
                         for t, m in models.items()},
        "extra_model_scores": {t: {k: v for k, v in m.items() if k not in drop}
                               for t, m in extra.items()},
        "config": {
            "matched_eval_seeds": list(matched_seeds),
            "matched_eval_universes": len(matched_seeds),
            "matched_seed_source": "config.llm_seeds of each model artifact",
            "extra_model_sources": EXTRA_MODEL_SOURCES,
            "sizes": args.sizes,
            "draws": {str(k): v for k, v in reps.items()},
            "eval_universes": args.eval_universes,
            "eval_seeds": [0, args.eval_universes],
            "top_k": args.top_k,
            "seed_pool": [lo, hi],
            "rng_seed": args.seed,
            "pool_disjoint_from_eval": True,
            "code_hash": code_hash,
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
