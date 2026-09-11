"""Do the structural conclusions survive different generator settings?

The benchmark's claims rest on numbers produced under one configuration of
noise, gating and slope. A reviewer is entitled to ask whether the ordering
"consensus < evidence < oracle", and the recognition gap in particular, are
properties of the task or of the knobs. This sweeps each knob independently and
reports whether the qualitative conclusions hold at every setting.

Model-free by construction: baselines only, so it costs nothing to re-run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ldb.baselines import BASELINES
from ldb.llm import TOP_K, apply_top_k_protocol
from ldb.metrics import evaluate
from ldb.render import observe
from ldb.task import Instance
from ldb.universe import generate_universe

ROOT = Path(__file__).resolve().parent.parent

# each knob swept independently around the default configuration
GRID = {
    "attr_noise": [0.06, 0.12, 0.18, 0.26, 0.34],   # instrument error
    "drop_p": [0.00, 0.08, 0.15, 0.25, 0.35],       # missing attributions
    "false_p": [0.00, 0.05, 0.10, 0.20, 0.30],      # spurious attributions
    "slope": [4.0, 6.5, 9.0, 12.0, 16.0],           # how sharp the threshold is
}
DEFAULTS = {"attr_noise": 0.18, "drop_p": 0.15, "false_p": 0.10, "slope": 9.0}


def run(seeds: int, **over) -> dict:
    cfg = {**DEFAULTS, **over}
    rows = {name: [] for name in BASELINES}
    base_rates = []
    for s in range(seeds):
        u = generate_universe(s, slope=cfg["slope"])
        o = observe(u, 20, 8, s, drop_p=cfg["drop_p"],
                    false_p=cfg["false_p"], attr_noise=cfg["attr_noise"])
        cands, label, p_star, mat, lead, rec, cap, p_cap = [], {}, {}, {}, {}, {}, {}, {}
        for key, pr in u.pairs.items():
            if pr.activated_at <= 20:
                continue
            cands.append(key)
            hit = int(20 < pr.activated_at <= 28)
            label[key] = hit
            p_star[key] = pr.p_horizon
            mat[key] = u.needs[key[1]].materiality
            lead[key] = (pr.activated_at - 20) if hit else -1
            rec[key] = o.mention_counts[key]
            cap[key] = int(hit and pr.captured)
            p_cap[key] = pr.p_capture
        inst = Instance(u, o, "", 20, 8, cands, label, p_star, mat, lead, rec,
                        captured=cap, p_capture=p_cap)
        base_rates.append(np.mean(list(label.values())))
        for name, fn in BASELINES.items():
            raw = fn(inst, seed=s)
            sc, _ = apply_top_k_protocol(raw, raw, TOP_K)
            rows[name].append(evaluate(inst, sc))
    out = {n: float(np.nanmean([r["auc"] for r in v])) for n, v in rows.items()}
    out["gap_recognition"] = float(np.nanmean(
        [r["gap_auc"] for r in rows["recognition"]]))
    out["base_rate"] = float(np.mean(base_rates))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default="results/sensitivity.json")
    args = ap.parse_args()

    report, violations = {}, []
    for knob, values in GRID.items():
        report[knob] = {}
        for v in values:
            r = run(args.seeds, **{knob: v})
            report[knob][str(v)] = r
            # the three structural claims, checked at every setting
            if not (r["random"] < r["recognition"] < r["structured_evidence"]
                    < r["hidden_oracle"]):
                violations.append(f"{knob}={v}: ordering broke {r}")
            if abs(r["gap_recognition"] - 0.5) > 0.05:
                violations.append(
                    f"{knob}={v}: consensus not at chance on unmentioned pairs "
                    f"({r['gap_recognition']:.3f})")
            if not 0.03 <= r["base_rate"] <= 0.35:
                violations.append(f"{knob}={v}: base rate {r['base_rate']:.3f}")

    print(f"{'knob':<12}{'value':>7}{'random':>9}{'recog':>9}{'evidence':>10}"
          f"{'oracle':>9}{'gap':>8}{'rate':>7}")
    for knob, vals in report.items():
        for v, r in vals.items():
            print(f"{knob:<12}{v:>7}{r['random']:>9.3f}{r['recognition']:>9.3f}"
                  f"{r['structured_evidence']:>10.3f}{r['hidden_oracle']:>9.3f}"
                  f"{r['gap_recognition']:>8.3f}{r['base_rate']:>7.3f}")

    print()
    if violations:
        print(f"{len(violations)} setting(s) break a structural claim:")
        for v in violations:
            print(f"  {v}")
    else:
        print("all structural claims hold across every setting swept")

    (ROOT / args.out).write_text(json.dumps(
        {"grid": GRID, "defaults": DEFAULTS, "seeds": args.seeds,
         "report": report, "violations": violations}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
