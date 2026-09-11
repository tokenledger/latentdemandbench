"""Two reviewer objections, both answerable from artifacts already on disk.

(1) MENTIONED-SUBSET AUC. The paper reports gap-AUC (AUC restricted to pairings
    with ZERO mentions) and overall AUC. A model can match the reference on the
    zero-mention subset and trail overall without its shortfall being localised
    on the mentioned pairings, because overall AUC also ranks zero-mention
    pairings against mentioned ones. The complement subset - `recognized > 0` -
    is the number that actually decides the localisation claim, so it is
    computed here exactly as `recognition_gap_auc` is, with the selector
    flipped, and differenced against `fair_reference` with a PAIRED bootstrap
    over the universes the model actually ran on.

    Every ranker is put under the paper's top-K contract first
    (`apply_top_k_protocol` for baselines, `score_predictions` for models, which
    is the same contract enforced at parse time), and model scores are rank
    positions rather than reported probabilities - the same construction the
    main table uses. Otherwise these AUCs would not be comparable to the paper's.

(2) BRIER vs p*-MSE. Squared error against the latent p* is not calibration:
    the dossier is deliberately lossy, so the Bayes-optimal probability given
    the OBSERVED evidence need not equal the hidden-world p*. This reports the
    genuine calibration number - Brier score against the REALISED binary
    outcome `inst.label` - on the same fixed probe set, alongside the p*-MSE,
    so the two can be printed side by side and the ranking of systems compared.

Probe handling matches `tools/probe_calibration.py` exactly: omitted probe
answers are EXCLUDED and coverage is recorded, so the two artifacts agree.
No API calls; model answers come from the archived `raw_response`.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from ldb.baselines import BASELINES, SCORE_KIND
from ldb.llm import TOP_K, apply_top_k_protocol, score_predictions
from ldb.metrics import auc, bootstrap_ci, paired_diff
from ldb.task import build_instance

# rank-only baselines emit no probability and cannot be scored for calibration
PROB_BASELINES = [n for n, k in SCORE_KIND.items() if k == "prob"]

REFERENCE = "fair_reference"


def mentioned_subset_auc(inst, scores: dict) -> float:
    """AUC restricted to pairings the market ALREADY discusses.

    The exact complement of `ldb.metrics.recognition_gap_auc`: same candidate
    set, same tie-averaged AUC, `recognized > 0` instead of `== 0`.
    """
    sub = [k for k in inst.candidates if inst.recognized[k] > 0]
    if not sub:
        return float("nan")
    return auc([inst.label[k] for k in sub], [scores[k] for k in sub])


def probe_scores(inst, probs: dict) -> dict:
    """Brier vs the realised 0/1 label and MSE/MAE vs latent p*, on the probes.

    Omitted probes are excluded (not imputed) and coverage is reported, which is
    what `tools/probe_calibration.py` does; imputing here would make the two
    artifacts disagree on the same quantity.
    """
    ks = [k for k in inst.probes if k in probs]
    if not ks:
        nan = float("nan")
        return {"brier": nan, "prob_mse_vs_p*": nan, "prob_mae_vs_p*": nan,
                "coverage": 0.0}
    d = np.array([probs[k] - inst.p_star[k] for k in ks], dtype=float)
    b = np.array([probs[k] - inst.label[k] for k in ks], dtype=float)
    return {"brier": float(np.mean(np.square(b))),
            "prob_mse_vs_p*": float(np.mean(np.square(d))),
            "prob_mae_vs_p*": float(np.mean(np.abs(d))),
            "coverage": len(ks) / len(inst.probes)}


def baseline_rows(insts: dict, top_k: int) -> dict:
    """{name: {seed: row}} for every probability-emitting baseline."""
    out = {}
    for name in PROB_BASELINES:
        out[name] = {}
        for i, (seed, inst) in enumerate(sorted(insts.items())):
            raw = BASELINES[name](inst, seed=i)
            # held to the model's top-K contract, exactly as run_v0 does
            sc, _ = apply_top_k_protocol(raw, raw, top_k)
            row = {"mentioned_auc": mentioned_subset_auc(inst, sc)}
            # calibration uses the untruncated probabilities: the probe set is
            # fixed in advance and the top-K contract does not apply to it
            row.update(probe_scores(inst, raw))
            out[name][seed] = row
    return out


def model_rows(path: Path, insts: dict, top_k: int) -> tuple[str, dict]:
    art = json.loads(path.read_text())
    cfg = art["config"]
    label = f"{cfg['backend']['model']}"
    rows = {}
    for idx, seed in enumerate(cfg["llm_seeds"]):
        inst = insts[seed]
        parsed = json.loads(art["model_outputs"][str(idx)]["raw_response"])
        # rank position as score, never the reported probability - the same
        # construction `ldb.llm.score_predictions` uses for the main table
        sc, _, _ = score_predictions(inst, parsed["predictions"], top_k)
        row = {"mentioned_auc": mentioned_subset_auc(inst, sc)}
        answered = {}
        for p in parsed.get("probes", []):
            key = (p.get("product_id"), p.get("need_id"))
            if key in inst.p_star and key not in answered:
                answered[key] = min(max(float(p.get("probability") or 0.0), 0.0), 1.0)
        row.update(probe_scores(inst, answered))
        rows[seed] = row
    return label, rows


def summarise(rows: dict, seeds: list | None = None) -> dict:
    use = sorted(rows) if seeds is None else [s for s in seeds if s in rows]
    out = {"seeds": use, "n": len(use)}
    for m in ("mentioned_auc", "brier", "prob_mse_vs_p*", "prob_mae_vs_p*"):
        mean, lo, hi = bootstrap_ci([rows[s][m] for s in use])
        out[m] = {"mean": mean, "lo": lo, "hi": hi}
    out["coverage"] = float(np.mean([rows[s]["coverage"] for s in use]))
    out["per_universe"] = {str(s): rows[s] for s in use}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["results/luna_full/surface_finance.json",
                             "results/terra_full/surface_finance.json",
                             "results/openweight/model_finance.json"])
    ap.add_argument("--eval-universes", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--out", default=str(ROOT / "results" / "probe_subset.json"))
    args = ap.parse_args()

    from run_v0 import _code_hash
    code_hash = _code_hash()

    paths = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.arms]
    arts = {p: json.loads(p.read_text()) for p in paths}

    # every artifact must describe the same generator, or baseline rows and
    # model rows would be scored on different universes and the paired
    # difference would be meaningless
    gen_keys = ["surface", "assist", "cutoff", "horizon", "n_products",
                "n_needs", "n_complements", "slope", "top_k"]
    gens = {p: {k: a["config"][k] for k in gen_keys} for p, a in arts.items()}
    first = next(iter(gens.values()))
    for p, g in gens.items():
        if g != first:
            raise SystemExit(f"generator config mismatch in {p}: {g} != {first}")
    if first["top_k"] != args.top_k:
        raise SystemExit(f"artifacts ran at top_k={first['top_k']}, "
                         f"scoring at {args.top_k}")

    seeds = sorted(set(range(args.eval_universes))
                   | {s for a in arts.values() for s in a["config"]["llm_seeds"]})
    insts = {s: build_instance(s, surface=first["surface"], assist=first["assist"],
                               cutoff=first["cutoff"], horizon=first["horizon"],
                               n_products=first["n_products"],
                               n_needs=first["n_needs"],
                               n_complements=first["n_complements"],
                               slope=first["slope"])
             for s in seeds}

    braw = baseline_rows(insts, args.top_k)
    mraw = {}
    for p in paths:
        name, rows = model_rows(p, insts, args.top_k)
        mraw[name] = rows

    res = {n: summarise(r, list(range(args.eval_universes))) for n, r in braw.items()}
    for n, r in mraw.items():
        res[n] = summarise(r)

    # (b) the paired difference that decides the localisation claim
    paired = {}
    for n, r in mraw.items():
        sd = sorted(r)
        a = [r[s] for s in sd]
        b = [braw[REFERENCE][s] for s in sd]
        for metric in ("mentioned_auc", "brier", "prob_mse_vs_p*"):
            mean, lo, hi = paired_diff(a, b, metric)
            paired.setdefault(n, {})[metric] = {"mean": mean, "lo": lo, "hi": hi,
                                                "n": len(sd), "vs": REFERENCE}

    print("\nmentioned-subset AUC (recognized > 0)")
    for n, r in res.items():
        a = r["mentioned_auc"]
        print(f"  {n:22s} {a['mean']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}]  n={r['n']}")
    print(f"\npaired difference in mentioned-subset AUC (model - {REFERENCE})")
    for n, d in paired.items():
        a = d["mentioned_auc"]
        print(f"  {n:22s} {a['mean']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]  n={a['n']}")
    print("\nprobe set: Brier (vs realised 0/1) vs p*-MSE (vs latent p*)")
    for n, r in res.items():
        print(f"  {n:22s} Brier {r['brier']['mean']:.4f} "
              f"[{r['brier']['lo']:.4f}, {r['brier']['hi']:.4f}]   "
              f"p*-MSE {r['prob_mse_vs_p*']['mean']:.4f} "
              f"[{r['prob_mse_vs_p*']['lo']:.4f}, {r['prob_mse_vs_p*']['hi']:.4f}]"
              f"   coverage {r['coverage']:.3f}")
    for m in ("brier", "prob_mse_vs_p*"):
        order = sorted(res, key=lambda n: res[n][m]["mean"])
        print(f"  order by {m:16s} (best first): {' < '.join(order)}")

    if _code_hash() != code_hash:
        raise SystemExit("source changed during the run; refusing to write")
    Path(args.out).write_text(json.dumps(
        {"results": res, "paired_vs_reference": paired,
         "config": {"code_hash": code_hash, "top_k": args.top_k,
                    "eval_universes": args.eval_universes,
                    "reference": REFERENCE, "generator": first,
                    "arms": [str(p.relative_to(ROOT)) for p in paths],
                    "probe_missing_policy": "excluded; coverage reported "
                                            "(matches tools/probe_calibration.py)"}},
        indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
