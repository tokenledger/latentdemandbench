"""Rebuild reviewed artifacts from unchanged archived model responses.

Generation provenance is retained under config.generation. config.code_hash
identifies the code that computes the new metric rows, never a new model call.
Only the reviewed generation revision is supported; older experiments remain
explicitly historical. The large-economy rows are blocked from paper generation
until their model calls have been repeated with a schema allowing K=90.
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ldb.baselines import BASELINES, SCORE_KIND
from ldb.llm import apply_top_k_protocol, average_replicates, score_response
from ldb.metrics import aggregate, bootstrap_ci, evaluate
from ldb.task import build_instance
from ldb.twins import build_twin, direction_score
from run_v0 import _code_hash, _finite, _value_orders
from tools.random_baseline_audit import RNG_NAMESPACE

GENERATION_HASH = "4f96dc85f1cee7ee"
GENERATION_COMMIT = "fef91460badd865c771c81a668893be6cab6d6e5"
CONTRACT_FILES = ("ldb/universe.py", "ldb/render.py", "ldb/task.py", "ldb/compose.py")
CONTRACT_HASH = "4503a322422e0d799b3446d53174bb702da40745782d02b27f4ebf375d8f1d06"
GEN_KEYS = ("cutoff", "horizon", "n_products", "n_needs", "n_complements", "slope")


def verify_generation_contract():
    digest = hashlib.sha256()
    for name in CONTRACT_FILES:
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    if digest.hexdigest() != CONTRACT_HASH:
        raise ValueError("generation/observation contract changed; archived outputs cannot be rescored")


def response_digest(art):
    # Include every raw response, including the two worlds in twin artifacts.
    raw = {idx: {k: v for k, v in output.items() if "raw" in k}
           for idx, output in art.get("model_outputs", {}).items()}
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def generation_config(cfg):
    return deepcopy(cfg.get("generation", cfg))


@lru_cache(maxsize=32)
def baseline_rows(config_json):
    cfg = json.loads(config_json)
    insts = [build_instance(s, **cfg["generator"]) for s in cfg["seeds"]]
    rows = {}
    for name, fn in BASELINES.items():
        rows[name] = []
        for seed, inst in zip(cfg["seeds"], insts):
            rng_seed = (np.random.SeedSequence([seed, RNG_NAMESPACE])
                        if name == "random" else seed)
            raw = fn(inst, seed=rng_seed)
            scores, probs = apply_top_k_protocol(raw, raw, cfg["top_k"])
            kind = SCORE_KIND[name]
            ev, capture = _value_orders(inst, raw, kind, cfg["top_k"])
            row = evaluate(inst, scores, probs if kind == "prob" else None,
                           ev, capture, probe_scores=raw)
            if kind == "rank":
                row["ndcg_ev@10"] = row["capture_auc"] = float("nan")
            rows[name].append(row)
    return rows, float(np.mean([np.mean(list(i.label.values())) for i in insts]))


def score_ranking(art):
    cfg = art["config"]
    gen = {k: cfg[k] for k in GEN_KEYS}
    key = json.dumps({"generator": gen, "seeds": cfg["universe_seeds"],
                      "top_k": cfg["top_k"]}, sort_keys=True)
    rows, base_rate = baseline_rows(key)
    rows = deepcopy(rows)
    model_keys = [k for k in art["per_universe"] if k not in BASELINES]
    for model in model_keys:
        rows[model] = []
        for idx, seed in enumerate(cfg["llm_seeds"]):
            inst = build_instance(seed, surface=cfg["surface"], assist=cfg["assist"], **gen)
            original = art["model_outputs"][str(idx)]["raw_response"]
            raw = [original] if isinstance(original, str) else original
            if not raw or any(not isinstance(r, str) for r in raw):
                raise ValueError(f"missing raw response at model index {idx}")
            reps = [score_response(inst, r, cfg["top_k"]) for r in raw]
            scores, probs, meta = average_replicates(reps, cfg["top_k"])
            ev = meta.pop("ev_scores")
            cap = meta.pop("cap_scores")
            probe = meta.pop("probe_scores")
            art["model_outputs"][str(idx)] = {
                "predictions": meta.pop("predictions"),
                "raw_response": meta.pop("raw_response"),
            }
            rows[model].append({**evaluate(inst, scores, probs, ev, cap, probe), **meta})
    art["per_universe"] = rows
    art["results"] = {name: aggregate(r) for name, r in rows.items()}
    art["base_rate"] = base_rate
    if model_keys and cfg["top_k"] > 40:
        cfg["requires_model_rerun"] = "historical response schema capped predictions at 40"


def score_twins(art):
    cfg = art["config"]
    twins = {s: build_twin(s, surface=cfg["surface"]) for s in cfg["twin_seeds"]}
    if any(t is None for t in twins.values()):
        raise ValueError("recorded twin no longer exists")
    rows = {}
    for name, fn in BASELINES.items():
        rows[name] = []
        for idx, twin in enumerate(twins.values()):
            blocked = fn(twin.blocked, seed=idx)
            enabled = fn(twin.enabled, seed=idx)
            b = apply_top_k_protocol(blocked, blocked)[0]
            e = apply_top_k_protocol(enabled, enabled)[0]
            rows[name].append(direction_score(twin, b, e))
    for model in (k for k in art["per_twin"] if k not in BASELINES):
        rows[model] = []
        for seed in cfg["llm_seeds"]:
            twin = twins[seed]
            reps = []
            for rep in range(cfg["replicates"]):
                archived = art["model_outputs"][f"{seed}#{rep}"]
                b = score_response(twin.blocked, archived["blocked_raw"])[0]
                e = score_response(twin.enabled, archived["enabled_raw"])[0]
                reps.append(direction_score(twin, b, e))
            rows[model].append({k: float(np.mean([r[k] for r in reps])) for k in reps[0]})
    art["per_twin"] = rows
    art["results"] = {
        name: {"direction_acc": bootstrap_ci([r["direction_acc"] for r in rs]),
               "mean_delta_pct": bootstrap_ci([r["mean_delta_pct"] for r in rs]),
               "n_twins": len(rs)} for name, rs in rows.items()
    }


def rescore(path, *, write=False):
    verify_generation_contract()
    scoring_hash = _code_hash()
    payload = path.read_bytes()
    art = json.loads(payload)
    cfg = art["config"]
    original = generation_config(cfg)
    if original["code_hash"] != GENERATION_HASH:
        raise ValueError(f"unsupported generation revision in {path}")
    if cfg.get("failures"):
        raise ValueError(f"unresolved calls in {path}")
    before = response_digest(art)
    if "per_twin" in art:
        score_twins(art)
    else:
        score_ranking(art)
    assert response_digest(art) == before, "rescoring must preserve every raw response"
    cfg["generation"] = original
    cfg["code_hash"] = scoring_hash
    cfg["rescoring"] = {
        "source_commit": GENERATION_COMMIT,
        "source_artifact_sha256": cfg.get("rescoring", {}).get(
            "source_artifact_sha256", hashlib.sha256(payload).hexdigest()),
        "raw_responses_sha256": before,
        "generation_contract_sha256": CONTRACT_HASH,
        "scoring_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "offline from unchanged archived responses; no model calls",
    }
    if write:
        if _code_hash() != scoring_hash:
            raise ValueError("source changed during rescoring; refusing to write")
        path.write_text(json.dumps(_finite(art), indent=2, allow_nan=False) + "\n")
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--paths", nargs="+")
    args = ap.parse_args()
    verify_generation_contract()
    paths = ([ROOT / p for p in args.paths] if args.paths else
             [ROOT / "results/baselines.json", ROOT / "results/twins.json",
              *sorted((ROOT / "results/luna_full").glob("*.json")),
              *sorted((ROOT / "results/terra_full").glob("*.json")),
              ROOT / "results/openweight/model_finance.json"])
    for path in paths:
        cfg = json.loads(path.read_text())["config"]
        if cfg.get("code_hash") == _code_hash() and "generation" not in cfg:
            if cfg.get("requires_model_rerun"):
                raise ValueError(f"current native artifact still requires a rerun: {path}")
            print(f"{path.relative_to(ROOT)}: already current (native run)", flush=True)
            continue
        art = rescore(path, write=args.write)
        status = "MODEL RERUN REQUIRED" if art["config"].get("requires_model_rerun") else "rescored"
        print(f"{path.relative_to(ROOT)}: {status}", flush=True)


if __name__ == "__main__":
    main()
