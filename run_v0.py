"""LatentDemandBench v0 runner: go/no-go table with bootstrap CIs."""

import argparse
import json
import math
import os

import numpy as np

from ldb.backends import InfrastructureError, backend_names, get_backend
from ldb.baselines import BASELINES, SCORE_KIND
from ldb.compose import ASSISTS, evidence_score, observed_capture
from ldb.metrics import aggregate, evaluate, paired_diff
from ldb.render import SURFACES
from ldb.llm import TOP_K, apply_top_k_protocol
from ldb.task import build_instance

COLS = ["recall@10", "ndcg@10", "ndcg_ev@10", "auc", "capture_auc",
        "gap_auc", "probe_auc", "consensus_rho", "lead_time@10",
        "prob_mse_vs_p*"]
PATH_COLS = ["path_cap_cited", "path_cap_false", "path_gate_cited"]
# the four outputs idea.md asks for beyond a ranking
ELICITED_COLS = ["materiality_rho", "capture_mae", "recognition_acc",
                 "prereq_f1", "probe_coverage"]

# The pre-registered decision rule, as paired comparisons on shared universes.
CONTRASTS = [
    ("fair_reference", "random", "signal exists in observable evidence"),
    ("structured_evidence", "fair_reference",
     "privileged form adds little over learning it (should be small)"),
    ("structured_evidence", "recognition", "task is not just market consensus"),
    ("hidden_oracle", "structured_evidence", "metrics sane (should be >= 0)"),
]


def _code_hash() -> str:
    """Deterministic hash of the source that produced a result.

    git rev is useless here: the tree is not a repo, so it recorded "untracked"
    on every artifact. Hashing the modules that define the task means two
    artifacts are comparable iff this string matches.
    """
    import hashlib
    import pathlib
    h = hashlib.sha256()
    root = pathlib.Path(__file__).parent
    for f in sorted(root.glob("ldb/**/*.py")) + sorted(root.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _backend_version(backend: str) -> str:
    """Version of whatever actually served the call, per backend."""
    import subprocess
    try:
        if backend == "codex":
            return subprocess.run(["codex", "--version"], capture_output=True,
                                  text=True).stdout.strip()
        if backend == "anthropic":
            import anthropic
            return f"anthropic-sdk {anthropic.__version__}"
        if backend == "gemini":
            import google.genai as g
            return f"google-genai {g.__version__}"
    except Exception:
        pass
    return "unknown"


_BILLING = ("credit", "quota", "billing", "prepayment", "payment",
            "insufficient", "authentication", "permission denied", "api key")


def _is_billing_fault(msg: str) -> bool:
    low = msg.lower()
    return any(t in low for t in _BILLING)


def _value_orders(inst, raw: dict, kind: str, top_k: int = TOP_K):
    """Build the two value-weighted orders from a ranker's own score.

    Both are formed from the ranker's probability-ranked top-K first, matching
    the constraint a model is under: it can only re-weight pairs it surfaced.
    """
    o = inst.observed
    if kind == "rank":
        return None, None  # no probability to weight; metrics stay undefined
    top = set(sorted(raw, key=lambda k: -raw[k])[:top_k])
    if kind == "ev":
        # The EV baseline selects by holder value (p * materiality * capture),
        # but each evaluation objective needs its own order on that shortlist.
        raw = {k: evidence_score(o, *k) for k in raw}
    ev = {k: (raw[k] * o.need_materiality[k[1]] if k in top else 0.0) for k in raw}
    cap = {k: (raw[k] * observed_capture(o, k[0]) if k in top else 0.0) for k in raw}
    return ev, cap


def _finite(o):
    """NaN/Infinity are not JSON. Emit null so the artifact parses strictly."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        # tuples too: twin bootstrap intervals are tuples, and json would
        # otherwise emit their NaNs verbatim
        return [_finite(v) for v in o]
    return o


def fmt(cell: dict) -> str:
    if np.isnan(cell["mean"]):
        return f"{'-':>22}"
    if np.isnan(cell["lo"]):
        return f"{cell['mean']:>22.3f}"
    return f"{cell['mean']:>10.3f} [{cell['lo']:.3f},{cell['hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", type=int, default=100)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llm-universes", type=int, default=30,
                    help="LLM runs on the first N universes (cost control)")
    ap.add_argument("--backend", default="gemini", choices=backend_names())
    ap.add_argument("--model", default=None, help="omit for the backend's default")
    ap.add_argument("--effort", default=None,
                    help="reasoning effort; the capability ladder axis")
    ap.add_argument("--surface", default="finance", choices=sorted(SURFACES))
    ap.add_argument("--assist", default="none", choices=list(ASSISTS),
                    help="hand the model one hop, to localise its failure")
    # generator knobs, so difficulty can be varied with a model in the loop
    ap.add_argument("--n-products", type=int, default=12)
    ap.add_argument("--n-needs", type=int, default=15)
    ap.add_argument("--n-complements", type=int, default=8)
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--slope", type=float, default=9.0)
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help="scale with candidate count when sweeping economy size, "
                         "or a larger economy silently faces a tighter filter")
    ap.add_argument("--replicates", type=int, default=1,
                    help="completions per universe, averaged; >1 estimates "
                         "within-condition variance")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="results/baselines.json")
    ap.add_argument("--dump-corpus", type=str, default=None)
    args = ap.parse_args()

    # Captured BEFORE any model call. Computing it at write time certified the
    # code that happened to be on disk when a long run finished, not the code
    # that produced the results - and that mis-certification already shipped.
    code_hash_at_start = _code_hash()

    gen = dict(n_products=args.n_products, n_needs=args.n_needs,
               n_complements=args.n_complements, slope=args.slope)
    insts = [build_instance(seed, surface=args.surface, assist=args.assist,
                            cutoff=args.cutoff, horizon=args.horizon, **gen)
             for seed in range(args.universes)]

    rates = [np.mean(list(i.label.values())) for i in insts]
    print(f"universes: {len(insts)}  surface: {args.surface}  assist: {args.assist}  "
          f"candidates/universe: {len(insts[0].candidates)}")
    print(f"activation base rate: {np.mean(rates):.3f} "
          f"(min {min(rates):.3f}, max {max(rates):.3f})")
    if not 0.03 <= np.mean(rates) <= 0.30:
        print("!! base rate outside sane band - metrics below are not trustworthy")

    if args.dump_corpus:
        with open(args.dump_corpus, "w") as f:
            f.write(insts[0].corpus)
        print(f"wrote sample corpus -> {args.dump_corpus}")

    # per-universe rows are kept so comparisons can be paired
    # these two output calibrated probabilities, so they get a Brier score;
    # random/recognition are rank-only and must not be scored for calibration
    PROB_SCORERS = {"structured_evidence", "fair_reference", "hidden_oracle"}
    rows: dict[str, list[dict]] = {}
    for name, fn in BASELINES.items():
        rows[name] = []
        for i, inst in enumerate(insts):
            raw = fn(inst, seed=i)
            # every ranker is held to the model's top-K contract
            sc, pr = apply_top_k_protocol(raw, raw, args.top_k)
            kind = SCORE_KIND.get(name, "rank")
            ev, cap = _value_orders(inst, raw, kind, args.top_k)
            row = evaluate(inst, sc, probs=pr if name in PROB_SCORERS else None,
                           ev_scores=ev, cap_scores=cap, probe_scores=raw)
            if kind == "rank":
                # no probability to weight, so these objectives are undefined
                # rather than computed from an arbitrary order
                row["ndcg_ev@10"] = row["capture_auc"] = float("nan")
            rows[name].append(row)

    llm_label, provenance, failures, llm_universes = None, None, [], []
    model_outputs: dict = {}
    if args.llm:
        from concurrent.futures import ThreadPoolExecutor

        from ldb.llm import average_replicates, failed_response, llm_rank

        subset = insts[: args.llm_universes]
        llm_label = f"{args.backend}:{args.model or 'default'}@{args.effort or 'default'}"
        if args.assist != "none":
            llm_label += f"+{args.assist}"
        # record the RESOLVED backend configuration, not the flags as typed
        provenance = get_backend(args.backend).describe(args.model, args.effort)
        provenance["cli_version"] = _backend_version(args.backend)
        if not provenance.get("isolated", False):
            print(f"  !! {args.backend} is NOT isolated "
                  f"({provenance.get('isolation_note', '')}); this row cannot be "
                  f"reported as a clean model measurement")
        print(f"querying {llm_label} on {len(subset)} universes...")

        failures = []

        def one(inst):
            last = None
            for attempt in range(args.retries + 1):
                try:
                    reps = [llm_rank(inst, backend=args.backend,
                                     model=args.model, effort=args.effort,
                                     replicate=r, top_k=args.top_k)
                            for r in range(args.replicates)]
                    return average_replicates(reps, args.top_k)
                except InfrastructureError:
                    raise  # our bug: abort loudly rather than fake a row
                except (AttributeError, TypeError, NameError, ImportError) as e:
                    raise InfrastructureError(f"{type(e).__name__}: {e}") from e
                except Exception as e:
                    last = f"{type(e).__name__}: {str(e)[:200]}"
                    if _is_billing_fault(last):
                        # a depleted balance is not a wrong prediction; scoring
                        # it as one would understate the model
                        raise InfrastructureError(
                            f"account/quota fault, not a model failure: {last}")
            # Scored as a failure, not dropped: failures may correlate with
            # difficulty, so silently excluding them biases the row upward.
            print(f"  ! unresolved after {args.retries + 1} attempts: {last}")
            failures.append({"seed": inst.universe.seed, "error": last})
            return failed_response(inst)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            out = list(pool.map(one, subset))

        # predictions are archived separately: they are the raw record, not a
        # metric, and must not enter bootstrap aggregation
        model_outputs = {}
        rows[llm_label] = []
        for idx, (i, (s, p, m)) in enumerate(zip(subset, out)):
            m = dict(m)
            ev = m.pop("ev_scores", None)
            cap = m.pop("cap_scores", None)
            probe_scores = m.pop("probe_scores", None)
            # everything non-numeric belongs in the archive, not the metric row
            model_outputs[str(idx)] = {
                "predictions": m.pop("predictions", []),
                "raw_response": m.pop("raw_response", None),
            }
            rows[llm_label].append(
                {**evaluate(i, s, probs=p, ev_scores=ev, cap_scores=cap,
                            probe_scores=probe_scores), **m})
        llm_universes = list(range(len(subset)))
        if failures:
            print(f"  {len(failures)}/{len(subset)} calls unresolved; "
                  f"scored as failures (not dropped)")
        if len(failures) == len(subset):
            raise SystemExit(
                "every model call failed; refusing to write an artifact that "
                "would look like a legitimate chance-level result")

    results = {name: aggregate(r) for name, r in rows.items()}

    print("\n" + f"{'baseline':<22}" + "".join(f"{c:>22}" for c in COLS))
    for name, r in results.items():
        print(f"{name:<22}" + "".join(fmt(r[c]) if c in r else f"{'-':>22}" for c in COLS))

    if llm_label:
        r = results[llm_label]
        for title, cols in (("elicited outputs", ELICITED_COLS),
                            ("evidence-path validity", PATH_COLS)):
            shown = [c for c in cols if c in r]
            if shown:
                print(f"\n{title} (model rows only):")
                for c in shown:
                    print(f"  {c:20s} {fmt(r[c]).strip()}")

    print("\npaired differences on AUC (95% bootstrap CI over universes):")
    contrasts = list(CONTRASTS)
    if llm_label:
        contrasts += [
            (llm_label, "recognition", "model beats consensus"),
            ("structured_evidence", llm_label, "headroom remains"),
        ]
    for a, b, why in contrasts:
        # restrict both sides to the universes the LLM actually ran on
        idx = llm_universes if llm_label in (a, b) else range(len(insts))
        ra = rows[a] if a == llm_label else [rows[a][i] for i in idx]
        rb = rows[b] if b == llm_label else [rows[b][i] for i in idx]
        m, lo, hi = paired_diff(ra, rb, "auc")
        sig = "" if np.isnan(lo) else ("  significant" if lo > 0 else "  NOT significant")
        ci = "" if np.isnan(lo) else f" [{lo:+.3f},{hi:+.3f}]"
        print(f"  {a} - {b}: {m:+.3f}{ci}{sig}   ({why})")

    if _code_hash() != code_hash_at_start:
        raise SystemExit(
            f"source changed during the run ({code_hash_at_start} -> "
            f"{_code_hash()}); refusing to write an artifact that would certify "
            "code which did not produce these results. Re-run.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_finite({
            "n_universes": len(insts),
            "base_rate": float(np.mean(rates)),
            "config": {
                "surface": args.surface, "assist": args.assist,
                "cutoff": args.cutoff, "horizon": args.horizon,
                "top_k": args.top_k, "replicates": args.replicates,
                "n_products": args.n_products, "n_needs": args.n_needs,
                "n_complements": args.n_complements, "slope": args.slope,
                "universe_seeds": list(range(args.universes)),
                "llm_seeds": llm_universes if llm_label else [],
                "backend": provenance, "failures": failures if llm_label else [],
                "code_hash": code_hash_at_start,
            },
            "results": results, "per_universe": rows,
            "model_outputs": model_outputs if llm_label else {}}), f, indent=2,
            allow_nan=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
