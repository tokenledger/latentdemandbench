"""Open-weight arm: one Modal batch job per run, scored like any other arm.

Why this is not a `ldb.backends` backend: `run_v0._code_hash` hashes every .py
under `ldb/**` and every .py at the repo root, and every artifact in results/
records that hash. Adding a module there would change the hash and invalidate
every published artifact at once. So this file reimplements the two things
`run_v0.main` does that a backend cannot reach from `tools/`:

  * `ldb.llm.llm_rank`, minus its `get_backend(...)` call — the prompts, the
    schema and the scoring are imported from `ldb.llm` unchanged, and only the
    transport is replaced;
  * `run_v0.main`'s baseline sweep and artifact layout — imported from
    `run_v0` where importable (`_value_orders`, `_finite`, `_code_hash`) so the
    two runners cannot drift.

The whole run is ONE batch call: every universe's prompt is sent to a single
Modal container that boots vLLM once and generates them together.

    python3 tools/run_open_weight.py --llm-universes 2 \
        --model Qwen/Qwen2.5-7B-Instruct --out results/openweight/smoke.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ldb.baselines import BASELINES, SCORE_KIND               # noqa: E402
from ldb.llm import (TOP_K, apply_top_k_protocol,            # noqa: E402
                     build_system, build_task, failed_response, response_schema,
                     score_response)
from ldb.metrics import aggregate, evaluate, paired_diff      # noqa: E402
from ldb.task import build_instance                           # noqa: E402
from run_v0 import (COLS, CONTRASTS, ELICITED_COLS, PATH_COLS,  # noqa: E402
                    _code_hash, _finite, _value_orders, fmt)

# run_v0 scores calibration only for the rankers that emit probabilities
PROB_SCORERS = {"structured_evidence", "fair_reference", "hidden_oracle"}


def rank_from_raw(inst, raw: str, top_k: int = TOP_K):
    """Use the shared scorer after the Modal transport has completed."""
    return score_response(inst, raw, top_k)


def _failure_row(inst):
    """Scored as a failure, not dropped — see run_v0.main."""
    return failed_response(inst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", type=int, default=100)
    ap.add_argument("--llm-universes", type=int, default=30)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--surface", default="finance")
    ap.add_argument("--assist", default="none")
    ap.add_argument("--n-products", type=int, default=12)
    ap.add_argument("--n-needs", type=int, default=15)
    ap.add_argument("--n-complements", type=int, default=8)
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--slope", type=float, default=9.0)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--gpu", default="H100", help="modal gpu spec, e.g. H100:2")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--max-model-len", type=int, default=32768,
                    help="prompt is ~7k tokens; leave room for --max-tokens. "
                         "Must not exceed the model's own context window.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--enable-thinking", default=None,
                    choices=[None, "true", "false"])
    ap.add_argument("--no-guided", action="store_true",
                    help="free-form decoding; only for debugging a schema")
    ap.add_argument("--out", default="results/openweight/model_finance.json")
    args = ap.parse_args()

    # Captured BEFORE the run, exactly as run_v0 does: an artifact must certify
    # the code that produced it, not the code on disk when it finished.
    code_hash_at_start = _code_hash()

    gen = dict(n_products=args.n_products, n_needs=args.n_needs,
               n_complements=args.n_complements, slope=args.slope)
    insts = [build_instance(seed, surface=args.surface, assist=args.assist,
                            cutoff=args.cutoff, horizon=args.horizon, **gen)
             for seed in range(args.universes)]
    rates = [np.mean(list(i.label.values())) for i in insts]
    print(f"universes: {len(insts)}  surface: {args.surface}  "
          f"assist: {args.assist}  candidates/universe: {len(insts[0].candidates)}")
    print(f"activation base rate: {np.mean(rates):.3f} "
          f"(min {min(rates):.3f}, max {max(rates):.3f})")

    # ---- baselines, over every universe, under the model's top-K contract ----
    rows: dict[str, list[dict]] = {}
    for name, fn in BASELINES.items():
        rows[name] = []
        for i, inst in enumerate(insts):
            raw = fn(inst, seed=i)
            sc, pr = apply_top_k_protocol(raw, raw, args.top_k)
            kind = SCORE_KIND.get(name, "rank")
            ev, cap = _value_orders(inst, raw, kind, args.top_k)
            row = evaluate(inst, sc, probs=pr if name in PROB_SCORERS else None,
                           ev_scores=ev, cap_scores=cap, probe_scores=raw)
            if kind == "rank":
                row["ndcg_ev@10"] = row["capture_auc"] = float("nan")
            rows[name].append(row)

    # ---- one Modal batch call for every universe ----
    subset = insts[: args.llm_universes]
    items = [{"id": str(i),
              "system": build_system(inst.surface, inst.horizon),
              "task": build_task(inst, args.top_k)}
             for i, inst in enumerate(subset)]

    import modal

    from tools.modal_batch import APP_NAME, VLLM_VERSION, app, generate

    thinking = ({"true": True, "false": False}[args.enable_thinking]
                if args.enable_thinking else None)
    print(f"dispatching {len(items)} prompts to modal app {APP_NAME} "
          f"({args.model} on {args.gpu}, vllm {VLLM_VERSION})...")
    t0 = time.time()
    with modal.enable_output():
        with app.run():
            out = generate.with_options(gpu=args.gpu).remote(
                items, model=args.model,
                schema=None if args.no_guided else response_schema(args.top_k),
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel,
                gpu_memory_utilization=args.gpu_mem_util,
                enable_thinking=thinking)
    wall = time.time() - t0
    by_id = {r["id"]: r for r in out}
    remote_meta = json.loads(by_id.pop("__meta__", {"text": "{}"})["text"])
    print(f"modal batch returned in {wall:.1f}s wall clock; remote: {remote_meta}")

    # ---- score each completion exactly as llm_rank would ----
    failures, results_per_universe = [], []
    for i, inst in enumerate(subset):
        r = by_id.get(str(i))
        if r is None or not r.get("text"):
            failures.append({"seed": inst.universe.seed,
                             "error": (r or {}).get("error", "no completion")})
            results_per_universe.append(_failure_row(inst))
            continue
        try:
            results_per_universe.append(rank_from_raw(inst, r["text"], args.top_k))
        except Exception as e:  # noqa: BLE001 - a completion that will not parse
            print(f"  ! seed {inst.universe.seed}: {type(e).__name__}: "
                  f"{str(e)[:200]} (finish_reason={r.get('finish_reason')})")
            failures.append({"seed": inst.universe.seed,
                             "error": f"{type(e).__name__}: {str(e)[:200]}"})
            results_per_universe.append(_failure_row(inst))

    llm_label = f"modal-vllm:{args.model}@batch"
    if args.assist != "none":
        llm_label += f"+{args.assist}"

    model_outputs: dict = {}
    rows[llm_label] = []
    for idx, (inst, (s, p, m)) in enumerate(zip(subset, results_per_universe)):
        m = dict(m)
        ev = m.pop("ev_scores", None)
        cap = m.pop("cap_scores", None)
        probe_scores = m.pop("probe_scores", None)
        model_outputs[str(idx)] = {"predictions": m.pop("predictions", []),
                                   "raw_response": m.pop("raw_response", None)}
        rows[llm_label].append({**evaluate(inst, s, probs=p, ev_scores=ev, cap_scores=cap,
                                        probe_scores=probe_scores), **m})
    llm_universes = list(range(len(subset)))

    if failures:
        print(f"  {len(failures)}/{len(subset)} completions unusable; "
              f"scored as failures (not dropped)")
    if len(failures) == len(subset):
        raise SystemExit(
            "every model call failed; refusing to write an artifact that would "
            "look like a legitimate chance-level result")

    results = {name: aggregate(r) for name, r in rows.items()}
    print("\n" + f"{'baseline':<22}" + "".join(f"{c:>22}" for c in COLS))
    for name, r in results.items():
        print(f"{name:<22}" +
              "".join(fmt(r[c]) if c in r else f"{'-':>22}" for c in COLS))
    r = results[llm_label]
    for title, cols in (("elicited outputs", ELICITED_COLS),
                        ("evidence-path validity", PATH_COLS)):
        shown = [c for c in cols if c in r]
        if shown:
            print(f"\n{title} (model rows only):")
            for c in shown:
                print(f"  {c:20s} {fmt(r[c]).strip()}")

    print("\npaired differences on AUC (95% bootstrap CI over universes):")
    for a, b, why in list(CONTRASTS) + [
            (llm_label, "recognition", "model beats commentary-count control"),
            ("structured_evidence", llm_label, "headroom remains")]:
        idx = llm_universes if llm_label in (a, b) else range(len(insts))
        ra = rows[a] if a == llm_label else [rows[a][i] for i in idx]
        rb = rows[b] if b == llm_label else [rows[b][i] for i in idx]
        m, lo, hi = paired_diff(ra, rb, "auc")
        sig = "" if np.isnan(lo) else ("  significant" if lo > 0
                                       else "  NOT significant")
        ci = "" if np.isnan(lo) else f" [{lo:+.3f},{hi:+.3f}]"
        print(f"  {a} - {b}: {m:+.3f}{ci}{sig}   ({why})")

    if _code_hash() != code_hash_at_start:
        raise SystemExit(
            f"source changed during the run ({code_hash_at_start} -> "
            f"{_code_hash()}); refusing to write an artifact that would certify "
            "code which did not produce these results. Re-run.")

    provenance = {
        "backend": "modal-vllm", "model": args.model, "effort": "none",
        "tools": "none",
        # a dedicated container per run, no shared state, no network tools
        "isolated": True,
        "temperature": 0.0, "seed": 0,
        "vllm": VLLM_VERSION, "gpu": args.gpu,
        "tensor_parallel_size": args.tensor_parallel,
        "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
        "guided_json": not args.no_guided,
        "enable_thinking": thinking,
        "cli_version": f"modal {modal.__version__}",
        "batch": True, "wall_clock_s": round(wall, 1),
        "remote": remote_meta,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_finite({
            "n_universes": len(insts),
            "base_rate": float(np.mean(rates)),
            "config": {
                "surface": args.surface, "assist": args.assist,
                "cutoff": args.cutoff, "horizon": args.horizon,
                "top_k": args.top_k, "replicates": 1,
                "n_products": args.n_products, "n_needs": args.n_needs,
                "n_complements": args.n_complements, "slope": args.slope,
                "universe_seeds": list(range(args.universes)),
                "llm_seeds": llm_universes,
                "backend": provenance, "failures": failures,
                "code_hash": code_hash_at_start,
            },
            "results": results, "per_universe": rows,
            "model_outputs": model_outputs}), f, indent=2, allow_nan=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
