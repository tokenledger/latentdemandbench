"""Counterfactual twin worlds: does the ranker respond to the readiness gate?"""

import argparse
import json
import os

import numpy as np

from ldb.backends import InfrastructureError, backend_names, get_backend
from run_v0 import _backend_version, _code_hash, _finite, _is_billing_fault
from ldb.baselines import BASELINES
from ldb.metrics import bootstrap_ci
from ldb.render import SURFACES
from ldb.twins import build_twin, direction_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llm-twins", type=int, default=15)
    ap.add_argument("--backend", default="gemini", choices=backend_names())
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--surface", default="finance", choices=sorted(SURFACES))
    ap.add_argument("--replicates", type=int, default=1,
                    help="completions per world; >1 averages out sampling noise "
                         "on backends that expose no seed")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/twins.json")  # canonical name
    args = ap.parse_args()
    code_hash_at_start = _code_hash()

    twins = [t for s in range(args.seeds)
             if (t := build_twin(s, surface=args.surface)) is not None]
    n_aff = [len(t.affected) for t in twins]
    print(f"{len(twins)}/{args.seeds} seeds yielded a pivotal complement; "
          f"affected pairs/twin: mean {np.mean(n_aff):.1f}, total {sum(n_aff)}")

    from ldb.llm import TOP_K, apply_top_k_protocol  # noqa: F811
    provenance, failures, outputs = None, [], {}
    rows: dict[str, list[dict]] = {}
    for name, fn in BASELINES.items():
        rows[name] = []
        for i, t in enumerate(twins):
            b, e = fn(t.blocked, seed=i), fn(t.enabled, seed=i)
            # same top-K contract the model arm is held to
            b, _ = apply_top_k_protocol(b, b, TOP_K)
            e, _ = apply_top_k_protocol(e, e, TOP_K)
            rows[name].append(direction_score(t, b, e))

    if args.llm:
        from concurrent.futures import ThreadPoolExecutor

        from ldb.llm import TOP_K, apply_top_k_protocol, llm_rank

        subset = twins[: args.llm_twins]
        label = f"{args.backend}:{args.model or 'default'}@{args.effort or 'default'}"
        provenance = get_backend(args.backend).describe(args.model, args.effort)
        print(f"querying {label} on {len(subset)} twin pairs ({2 * len(subset)} calls)...")

        failures, outputs = [], {}

        def one(t):
            last = None
            for _ in range(args.retries + 1):
                try:
                    sb, _p, mb = llm_rank(t.blocked, backend=args.backend,
                                          model=args.model, effort=args.effort)
                    se, _q, me = llm_rank(t.enabled, backend=args.backend,
                                          model=args.model, effort=args.effort)
                    outputs[f"{t.seed}#0"] = {
                        "blocked": mb.get("predictions"), "enabled": me.get("predictions"),
                        "blocked_raw": mb.get("raw_response"),
                        "enabled_raw": me.get("raw_response"),
                    }
                    reps = [direction_score(t, sb, se)]
                    for r in range(1, args.replicates):
                        sb2, _a, mb2 = llm_rank(t.blocked, backend=args.backend,
                                                model=args.model,
                                                effort=args.effort, replicate=r)
                        se2, _c, me2 = llm_rank(t.enabled, backend=args.backend,
                                                model=args.model,
                                                effort=args.effort, replicate=r)
                        outputs[f"{t.seed}#{r}"] = {
                            "blocked": mb2.get("predictions"),
                            "enabled": me2.get("predictions"),
                            "blocked_raw": mb2.get("raw_response"),
                            "enabled_raw": me2.get("raw_response"),
                        }
                        reps.append(direction_score(t, sb2, se2))
                    # average across replicates so one sampling draw does not
                    # stand in for the effect of the intervention
                    return {k: sum(r[k] for r in reps) / len(reps) for k in reps[0]}
                except InfrastructureError:
                    raise  # our bug: abort rather than fake a chance-level row
                except (AttributeError, TypeError, NameError, ImportError) as e:
                    raise InfrastructureError(f"{type(e).__name__}: {e}") from e
                except Exception as e:
                    last = f"{type(e).__name__}: {str(e)[:200]}"
                    if _is_billing_fault(last):
                        raise InfrastructureError(
                            f"account/quota fault, not a model failure: {last}") from e
            # scored, not dropped: failures may correlate with twin difficulty
            print(f"  ! unresolved after {args.retries + 1} attempts: {last}")
            failures.append({"seed": t.seed, "error": last})
            flat = {k: 0.0 for k in t.blocked.candidates}
            return direction_score(t, flat, dict(flat))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            out = list(pool.map(one, subset))
        rows[label] = out
        if failures:
            print(f"  {len(failures)}/{len(subset)} unresolved; scored as failures")
        if len(failures) == len(subset):
            raise SystemExit(
                "every twin call failed; refusing to write an artifact that "
                "would look like a legitimate chance-level result")

    print(f"\n{'ranker':<22}{'direction_acc':>24}{'mean_delta_pct':>24}{'n':>6}")
    results = {}
    for name, rs in rows.items():
        acc = bootstrap_ci([r["direction_acc"] for r in rs])
        dlt = bootstrap_ci([r["mean_delta_pct"] for r in rs])
        results[name] = {"direction_acc": acc, "mean_delta_pct": dlt, "n_twins": len(rs)}
        print(f"{name:<22}{acc[0]:>10.3f} [{acc[1]:.3f},{acc[2]:.3f}]"
              f"{dlt[0]:>12.3f} [{dlt[1]:+.3f},{dlt[2]:+.3f}]{len(rs):>6}")
    print("\nchance = 0.500. hidden_oracle is the ceiling; recognition should sit at "
          "chance, since mention counts do not encode complement readiness.")

    if _code_hash() != code_hash_at_start:
        raise SystemExit("source changed during the run; refusing to write")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_finite({
            "n_twins": len(twins), "results": results,
            "config": {
                "surface": args.surface, "seeds_scanned": args.seeds,
                "twin_seeds": [t.seed for t in twins],
                # the seeds the MODEL ran on, so failure rate is checkable
                "llm_seeds": [t.seed for t in twins[:args.llm_twins]] if args.llm else [],
                "pivots": {str(t.seed): t.pivot for t in twins},
                "replicates": args.replicates,
                "backend": provenance,
                "backend_version": _backend_version(args.backend) if args.llm else None,
                "code_hash": code_hash_at_start,
                "hash_capture": "before_execution_with_end_check",
                "failures": failures if args.llm else [],
            },
            "per_twin": rows,
            "model_outputs": outputs if args.llm else {}}), f, indent=2,
            allow_nan=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
