"""Separate observation degradation from the reference's probability formula."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldb.baselines import hidden_oracle, structured_evidence
from ldb.llm import apply_top_k_protocol
from ldb.metrics import aggregate, pair_auc
from ldb.task import build_instance
from run_v0 import _code_hash, _finite


def latent_inputs(inst):
    """Replace only the reference's observed inputs, keeping labels fixed."""
    u = inst.universe
    observed = replace(
        inst.observed,
        cap_vecs={k: c.vec for k, c in u.capabilities.items()},
        prod_caps={k: p.cap_ids for k, p in u.products.items()},
        need_reqs={k: n.req for k, n in u.needs.items()},
        need_thresholds={k: n.threshold for k, n in u.needs.items()},
        need_comps={k: n.complement_ids for k, n in u.needs.items()},
        comp_ready={k: c.ready_at for k, c in u.complements.items()},
    )
    return replace(inst, observed=observed)


def decompose(inst, top_k=40):
    raw = {
        "observed_formula_auc": structured_evidence(inst),
        "latent_formula_auc": structured_evidence(latent_inputs(inst)),
        "oracle_auc": hidden_oracle(inst),
    }
    row = {name: pair_auc(inst, apply_top_k_protocol(scores, scores, top_k)[0])
           for name, scores in raw.items()}
    row["observation_loss"] = row["latent_formula_auc"] - row["observed_formula_auc"]
    row["formula_difference"] = row["oracle_auc"] - row["latent_formula_auc"]
    row["total_gap"] = row["oracle_auc"] - row["observed_formula_auc"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--out", default="results/observation_loss.json")
    args = ap.parse_args()
    source_hash = _code_hash()
    rows = [decompose(build_instance(seed), args.top_k)
            for seed in range(args.universes)]
    payload = _finite({
        "config": {"code_hash": source_hash, "top_k": args.top_k,
                   "universe_seeds": list(range(args.universes)),
                   "path": "observed reference -> latent-input reference -> conditional oracle"},
        "results": aggregate(rows), "per_universe": rows,
    })
    if _code_hash() != source_hash:
        raise SystemExit("source changed during the run; refusing to write")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
