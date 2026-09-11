"""Calibration on the fixed probe set, where the top-K contract does not apply.

The probability error reported in the main table is measured over every
candidate, and anything a ranker leaves outside its top-K is imputed a constant.
That number therefore mixes two things: which pairings a ranker chose to score
and how well it scored them. The hidden oracle giving a non-zero error is the
tell.

The probe set has no such problem. It is sixteen pairings fixed in advance,
every ranker answers all of them, and no truncation is involved, so squared
error against the exact latent probability there is calibration and nothing
else. Model answers are read from the archived responses, so this costs no API
calls.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from ldb.baselines import BASELINES, SCORE_KIND
from ldb.metrics import bootstrap_ci
from ldb.task import build_instance

# rank-only baselines emit no probability and cannot be scored for calibration
PROB_BASELINES = [n for n, k in SCORE_KIND.items() if k == "prob"]


def probe_errors(inst, probs: dict) -> tuple[float, float]:
    """(mean squared error, mean absolute error) against p* on the probes."""
    d = [probs[k] - inst.p_star[k] for k in inst.probes if k in probs]
    if not d:
        return float("nan"), float("nan")
    return float(np.mean(np.square(d))), float(np.mean(np.abs(d)))


def baseline_rows(n_eval: int) -> dict:
    out = {}
    for name in PROB_BASELINES:
        mses, maes = [], []
        for i in range(n_eval):
            inst = build_instance(i)
            probs = BASELINES[name](inst, seed=i)
            mse, mae = probe_errors(inst, probs)
            mses.append(mse)
            maes.append(mae)
        out[name] = {"mse": bootstrap_ci(mses), "mae": bootstrap_ci(maes),
                     "coverage": 1.0, "n": len(mses)}
    return out


def model_row(path: str) -> tuple[str, dict]:
    art = json.load(open(path))
    seeds = art["config"]["llm_seeds"]
    model = art["config"]["backend"]["model"]
    mses, maes, cov = [], [], []
    for idx, seed in enumerate(seeds):
        inst = build_instance(seed, surface=art["config"].get("surface", "finance"))
        parsed = json.loads(art["model_outputs"][str(idx)]["raw_response"])
        answered = {}
        for p in parsed.get("probes", []):
            key = (p.get("product_id"), p.get("need_id"))
            if key in inst.p_star and key not in answered:
                answered[key] = min(max(float(p.get("probability") or 0.0), 0.0), 1.0)
        mse, mae = probe_errors(inst, answered)
        mses.append(mse)
        maes.append(mae)
        cov.append(len([k for k in inst.probes if k in answered]) / len(inst.probes))
    return model, {"mse": bootstrap_ci(mses), "mae": bootstrap_ci(maes),
                   "coverage": float(np.mean(cov)), "n": len(mses)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["results/luna_full/surface_finance.json",
                             "results/terra_full/surface_finance.json"])
    ap.add_argument("--eval-universes", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "results" / "probe_calibration.json"))
    args = ap.parse_args()

    from run_v0 import _code_hash
    code_hash = _code_hash()

    res = baseline_rows(args.eval_universes)
    for path in args.arms:
        name, row = model_row(str(ROOT / path) if not Path(path).is_absolute() else path)
        res[name] = row
    for name, r in res.items():
        print(f"  {name:22s} MSE {r['mse'][0]:.4f} [{r['mse'][1]:.4f}, "
              f"{r['mse'][2]:.4f}]  MAE {r['mae'][0]:.4f}  coverage {r['coverage']:.3f}")

    if _code_hash() != code_hash:
        raise SystemExit("source changed during the run; refusing to write")
    Path(args.out).write_text(json.dumps(
        {"results": res, "config": {"code_hash": code_hash,
                                    "eval_universes": args.eval_universes}}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
