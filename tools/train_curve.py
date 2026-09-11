"""How much of the fair reference's advantage comes from seeing many universes?

The obvious objection to comparing a fitted ranker against a model reading one
dossier cold is that the ranker has been shown labelled worlds. This bounds that
advantage directly: refit the reference on 1, 2, 5, 10, 30 and 100 training
universes (all disjoint from evaluation) and re-score the same evaluation set.

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
from ldb.metrics import bootstrap_ci, evaluate
from ldb.task import build_instance

ROOT = Path(__file__).resolve().parent.parent
# the seed block the deployed reference is fitted on; disjoint from every
# evaluation set by construction (evaluation uses 0..n-1)
TRAIN_BASE = 1000


def curve(sizes: list[int], n_eval: int, top_k: int) -> dict:
    ev = [build_instance(s) for s in range(n_eval)]
    out = {}
    for n in sizes:
        ref = fit([build_instance(TRAIN_BASE + i) for i in range(n)])
        rows = []
        for i, inst in enumerate(ev):
            raw = ref(inst, seed=i)
            sc, pr = apply_top_k_protocol(raw, raw, top_k)
            rows.append(evaluate(inst, sc, probs=pr))
        mean, lo, hi = bootstrap_ci([r["auc"] for r in rows])
        out[str(n)] = {"mean": mean, "lo": lo, "hi": hi, "n": len(rows)}
        print(f"  train universes {n:4d}: AUC {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    return out


def mention_ablation(n_eval: int, top_k: int, n_train: int = 100) -> dict:
    """Refit the reference with individual features zeroed out.

    Two features invite an objection. One is commentary volume, which the prompt
    tells a solver to discount. The other is the formula-derived capture proxy,
    which is the only feature computed by applying a simulator formula to dossier numbers rather
    than read straight off the page, and is therefore partially privileged. If
    either carried the reference's edge the comparison would be unfair in a way
    training size does not capture, so each is removed and the reference refitted
    from scratch.
    """
    import ldb.fair as fairmod
    original = fairmod.features
    train = [build_instance(TRAIN_BASE + i) for i in range(n_train)]
    ev = [build_instance(s) for s in range(n_eval)]
    out = {}

    def drop(*idx):
        def f(inst, pid, nid):
            v = original(inst, pid, nid).copy()
            for i in idx:
                v[i] = 0.0
            return v
        return f

    for tag, feat in (("with_mentions", original),
                      ("without_mentions", drop(MENTION_FEATURE)),
                      ("without_capture", drop(CAPTURE_FEATURE)),
                      ("without_either", drop(MENTION_FEATURE, CAPTURE_FEATURE))):
        fairmod.features = feat
        ref = fairmod.fit(train)
        rows = []
        for i, inst in enumerate(ev):
            raw = ref(inst, seed=i)
            sc, pr = apply_top_k_protocol(raw, raw, top_k)
            rows.append(evaluate(inst, sc, probs=pr))
        mean, lo, hi = bootstrap_ci([r["auc"] for r in rows])
        out[tag] = {"mean": mean, "lo": lo, "hi": hi, "n": len(rows)}
        print(f"  {tag:16s}: AUC {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    fairmod.features = original
    return out


# indices into ldb.fair.features
MENTION_FEATURE = 13   # commentary volume, which the prompt tells a solver to discount
CAPTURE_FEATURE = 12   # formula-derived capture proxy; applies a simulator formula


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[1, 2, 5, 10, 30, 100])
    ap.add_argument("--eval-universes", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--out", default=str(ROOT / "results" / "train_curve.json"))
    args = ap.parse_args()

    from run_v0 import _code_hash
    code_hash = _code_hash()

    print(f"fair-reference training curve on {args.eval_universes} universes")
    res = curve(args.sizes, args.eval_universes, args.top_k)
    print("commentary-feature ablation")
    abl = mention_ablation(args.eval_universes, args.top_k)

    if _code_hash() != code_hash:
        raise SystemExit("source changed during the run; refusing to write")
    payload = {"curve": res, "mention_ablation": abl,
               "config": {"sizes": args.sizes, "eval_universes": args.eval_universes,
                          "top_k": args.top_k, "train_base": TRAIN_BASE,
                          "code_hash": code_hash}}
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
