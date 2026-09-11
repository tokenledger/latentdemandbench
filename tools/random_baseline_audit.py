"""Recompute the random control with an RNG namespace independent of worlds.

The evaluated runner historically passed universe index ``i`` directly to
``random_rank`` while universe ``i`` was generated from the same integer. A
random control must not share that seed namespace. This model-free sidecar
preserves the archived model artifacts and their code hash while supplying an
independently seeded control for paper tables and paired comparisons.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ldb.baselines import random_rank
from ldb.llm import apply_top_k_protocol
from ldb.metrics import aggregate, evaluate
from ldb.task import build_instance
from run_v0 import _code_hash, _finite

RNG_NAMESPACE = 20260810


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/baselines.json")
    ap.add_argument("--out", default="results/random_baseline_audit.json")
    args = ap.parse_args()

    base = json.loads((ROOT / args.base).read_text())
    cfg = base["config"]
    if cfg.get("code_hash") != _code_hash():
        raise SystemExit("baseline artifact does not match the evaluated code hash")

    rows = []
    for universe_seed in cfg["universe_seeds"]:
        inst = build_instance(
            universe_seed,
            surface=cfg["surface"],
            assist=cfg["assist"],
            cutoff=cfg["cutoff"],
            horizon=cfg["horizon"],
            n_products=cfg["n_products"],
            n_needs=cfg["n_needs"],
            n_complements=cfg["n_complements"],
            slope=cfg["slope"],
        )
        independent_seed = np.random.SeedSequence(
            [universe_seed, RNG_NAMESPACE]
        )
        raw = random_rank(inst, seed=independent_seed)
        scores, _ = apply_top_k_protocol(raw, raw, cfg["top_k"])
        row = evaluate(inst, scores, probe_scores=raw)
        row["ndcg_ev@10"] = float("nan")
        row["capture_auc"] = float("nan")
        rows.append(row)

    payload = _finite({
        "config": {
            "code_hash": _code_hash(),
            "source_artifact": args.base,
            "universe_seeds": cfg["universe_seeds"],
            "rng_namespace": RNG_NAMESPACE,
            "seed_rule": "SeedSequence([universe_seed, rng_namespace])",
            "top_k": cfg["top_k"],
        },
        "results": {"random": aggregate(rows)},
        "per_universe": {"random": rows},
    })
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    a = payload["results"]["random"]["auc"]
    g = payload["results"]["random"]["gap_auc"]
    print(
        f"wrote {out}: AUC_K {a['mean']:.3f} [{a['lo']:.3f},{a['hi']:.3f}], "
        f"Gap-AUC_K {g['mean']:.3f} [{g['lo']:.3f},{g['hi']:.3f}]"
    )


if __name__ == "__main__":
    main()
