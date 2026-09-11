"""Emit paper-ready LaTeX tables straight from the results JSON.

Numbers in the paper are generated, never transcribed: a stale table is the
easiest way to publish a wrong number, and re-running this after any experiment
keeps the manuscript consistent with the artifacts on disk.
"""

import argparse
import json
import os

PRETTY = {
    "random": r"Random",
    "recognition": r"Recognition (market consensus)",
    "structured_evidence": r"Structured evidence \textit{(privileged)}",
    "fair_reference": r"Fair reference \textit{(learned, held-out)}",
    "expected_value": r"Expected value \textit{(privileged)}",
    "hidden_oracle": r"Hidden oracle \textit{(simulator upper bound)}",
    # rows produced before the backend recorded its resolved config
    "evidence_oracle": r"Structured-evidence baseline \textit{(stale naming)}",
}
METRICS = [("recall@10", "R@10"), ("ndcg@10", "nDCG@10"),
           ("ndcg_ev@10", "nDCG$_{EV}$"), ("auc", "AUC"),
           ("capture_auc", "Capture"), ("gap_auc", "Gap-AUC"),
           ("consensus_rho", r"$\rho_{\text{consensus}}$"),
           ("prob_mse_vs_p*", "Prob.\\ MSE"), ("probe_coverage", "Probes")]


def pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    return name.replace("_", r"\_")


def cell(c: dict, nd: int = 3) -> str:
    """Null-safe: strict-JSON artifacts carry `null` where a metric was NaN."""
    if not isinstance(c, dict):
        return "--"
    m, lo, hi = c.get("mean"), c.get("lo"), c.get("hi")
    if m is None or m != m:
        return "--"
    if lo is None or lo != lo or hi is None or hi != hi:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} \\ci{{{lo:.{nd}f}}}{{{hi:.{nd}f}}}"


def main_table(res: dict, caption: str, label: str) -> str:
    cols = "l" + "c" * (len(METRICS) + 1)
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
         "Ranker & $n$ & " + " & ".join(h for _, h in METRICS) + r" \\", r"\midrule"]
    for name, row in res.items():
        cells = [cell(row[k]) if k in row else "--" for k, _ in METRICS]
        # rows differ in n: model arms run on a subset, and failed calls are dropped
        n = row.get("auc", {}).get("n", "--")
        L.append(f"{pretty(name)} & {n} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


def twin_table(res: dict, caption: str, label: str) -> str:
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Ranker & $n$ & Direction acc. & $\Delta$ percentile \\", r"\midrule"]
    for name, row in res.items():
        a, d = row["direction_acc"], row["mean_delta_pct"]
        n = row.get("n_twins", "--")
        fa = f"{a[0]:.3f} \\ci{{{a[1]:.3f}}}{{{a[2]:.3f}}}"
        fd = f"{d[0]:+.3f} \\ci{{{d[1]:+.3f}}}{{{d[2]:+.3f}}}"
        L.append(f"{pretty(name)} & {n} & {fa} & {fd} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


def model_id(d: dict) -> str:
    """Model identity WITHOUT effort. The ladder sweeps effort on purpose, so
    including it made every rung look like a different model."""
    return ((d.get("config") or {}).get("backend") or {}).get("model") or "unknown"


def model_name(d: dict) -> str:
    """Model identity from the artifact's own provenance, never hardcoded."""
    b = (d.get("config") or {}).get("backend") or {}
    m, e = b.get("model"), b.get("effort") or b.get("reasoning_effort")
    if not m:
        return "unknown model"
    return rf"\textsc{{{m}}}" + (f" ({e})" if e and e != "default" else "")


def ladder_table(files: dict, caption: str, label: str,
                 allow_partial: bool = False) -> str:
    """Capability ladder: one model, reasoning effort swept.

    The column that matters is the gap to the structured-evidence baseline. If it fails to
    shrink as effort rises, the shortfall is not a matter of spending more
    test-time compute.
    """
    names = set()
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Reasoning effort & $n$ & AUC & Gap-AUC & Baseline $-$ model \\", r"\midrule"]
    present = [e for e, p in files.items() if os.path.exists(p)]
    if len(present) < len(files) and not allow_partial:
        raise SystemExit(
            f"ladder has only {present} of {list(files)}; a partial sweep read "
            "as a full one. Pass --allow-partial to publish it anyway.")

    shared = {}
    for eff, path in files.items():
        if not os.path.exists(path):
            continue
        art = json.load(open(path))
        cfg = art.get("config") or {}
        b = cfg.get("backend") or {}
        # the filename must not be the source of truth for what was run
        rec = b.get("effort") or b.get("reasoning_effort")
        if rec and rec != eff:
            raise SystemExit(
                f"{path}: filename says effort={eff} but the artifact records "
                f"effort={rec}; rename or regenerate")
        for key, val in (("backend", b.get("backend")), ("surface", cfg.get("surface")),
                         ("assist", cfg.get("assist")),
                         ("seeds", tuple(cfg.get("llm_seeds") or []))):
            if key in shared and shared[key] != val:
                raise SystemExit(
                    f"ladder rungs differ in {key}: {shared[key]!r} vs {val!r}; "
                    "rungs must vary only in effort")
            shared[key] = val
        d = art["results"]
        # derived from the registry, so a new baseline cannot be mistaken for
        # the model row (expected_value was, producing four identical rungs)
        from ldb.baselines import BASELINES
        baselines = set(BASELINES) | {"evidence_oracle"}
        model_rows = [k for k in d if k not in baselines]
        if len(model_rows) != 1:
            raise SystemExit(
                f"{path}: expected exactly one model row, found {model_rows}")
        row = d[model_rows[0]]
        if row is None:
            continue
        # Baselines run on every universe, model arms on a subset. Subtracting
        # their aggregate means compared different universe sets, so restrict
        # the baseline to the seeds the model actually saw.
        seeds = (art.get("config") or {}).get("llm_seeds")
        per = art.get("per_universe", {})
        base_rows = per.get("structured_evidence") or per.get("evidence_oracle")
        if base_rows is None:
            raise SystemExit(f"{path}: no per-universe baseline rows; regenerate it")
        if seeds:
            base_rows = [base_rows[i] for i in seeds]
        vals = [r["auc"] for r in base_rows if r.get("auc") is not None]
        orc = sum(vals) / len(vals)
        names.add(model_id(art))
        L.append(f"{eff} & {row['auc']['n']} & {cell(row['auc'])} & "
                 f"{cell(row.get('gap_auc'))} & "
                 f"{orc - row['auc']['mean']:+.3f} \\\\")
    if len(names) > 1:
        raise SystemExit(f"ladder mixes models: {sorted(names)}; one model per ladder")
    label_name = rf"\textsc{{{names.pop()}}}" if names else "unknown"
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption.format(model=label_name)}}}",
          rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


PREAMBLE = r"""% requires: \usepackage{booktabs}
% \newcommand{\ci}[2]{{\tiny\,[#1,\,#2]}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/baselines.json")
    ap.add_argument("--opaque", default="results/opaque.json")
    ap.add_argument("--twins", default="results/twins.json")
    ap.add_argument("--ladder", default="results/ladder_{}.json",
                    help="pattern with {} for the effort level")
    ap.add_argument("--out", default="paper/tables.tex")
    ap.add_argument("--max-failure-rate", type=float, default=0.0,
                    help="reject a model artifact with more failures than this")
    ap.add_argument("--allow-stale", action="store_true",
                    help="permit artifacts older than the current source")
    ap.add_argument("--allow-unisolated", action="store_true",
                    help="permit rows from a backend with filesystem access")
    ap.add_argument("--allow-partial", action="store_true",
                    help="emit whatever artifacts exist instead of erroring")
    args = ap.parse_args()

    missing = [p for p in (args.main, args.twins) if not os.path.exists(p)]
    if missing and not args.allow_partial:
        raise SystemExit(
            "missing required artifacts: " + ", ".join(missing) +
            "\n(generate them, or pass --allow-partial to build what exists)")

    parts = [PREAMBLE]

    # A table assembled from artifacts built by different code is not a table of
    # one experiment. Refuse rather than silently publish incomparable rows.
    hashes = {}
    for name in (args.main, args.opaque, args.twins,
                 *(args.ladder.format(e) for e in ("low", "medium", "high", "max"))):
        if os.path.exists(name):
            cfg = json.load(open(name)).get("config", {})
            hashes[name] = cfg.get("code_hash", "MISSING")
    from run_v0 import _code_hash
    current = _code_hash()
    stale = {k: v for k, v in hashes.items() if v != current}
    if stale and not args.allow_stale:
        for k, v in sorted(stale.items()):
            print(f"  {v}  {k}")
        raise SystemExit(
            f"artifacts predate current source ({current}); regenerate them "
            "(or pass --allow-stale)")

    for name in hashes:
        cfg = json.load(open(name)).get("config") or {}
        fails, seeds = cfg.get("failures") or [], cfg.get("llm_seeds") or []
        if fails and not seeds:
            raise SystemExit(
                f"{name}: records {len(fails)} failures but no llm_seeds, so the "
                "failure rate cannot be checked; regenerate with current code")
        if seeds and len(fails) / len(seeds) > args.max_failure_rate:
            raise SystemExit(
                f"{name}: {len(fails)}/{len(seeds)} model calls failed "
                f"(> --max-failure-rate {args.max_failure_rate}); such a row "
                "scores as chance and would read as a real measurement")
        b = cfg.get("backend") or {}
        if b and not b.get("isolated", False) and not args.allow_unisolated:
            raise SystemExit(
                f"{name}: backend {b.get('backend')} is not isolated "
                f"({b.get('isolation_note', '')}); such rows are exploratory. "
                "Pass --allow-unisolated to publish anyway.")

    distinct = set(hashes.values())
    if len(distinct) > 1:
        for k, v in sorted(hashes.items()):
            print(f"  {v}  {k}")
        raise SystemExit(
            f"refusing to build tables from {len(distinct)} different code versions; "
            "regenerate all arms under one version")

    if os.path.exists(args.main):
        d = json.load(open(args.main))
        nf = len((d.get("config") or {}).get("failures") or [])
        parts.append(main_table(
            d["results"],
            f"Ranking and calibration on the finance surface "
            f"(base rate {d['base_rate']:.3f}). Model arms run on a subset of "
            f"universes; $n$ is per row. "
            r"Brackets are 95\% bootstrap CIs over universes. The "
            r"structured-evidence row is privileged: it applies the simulator's "
            r"own functional form, which the prompt does not state, so it bounds "
            r"the signal in the evidence rather than achievable performance."
            + (f" {nf} model call(s) failed and are scored as failures." if nf else ""),
            "tab:main"))

    if os.path.exists(args.opaque):
        d = json.load(open(args.opaque))
        parts.append(main_table(
            d["results"],
            "Same hidden graphs rendered with opaque entity labels. Oracles are "
            "surface-invariant by construction, so any model change isolates "
            "reliance on familiar framing.",
            "tab:opaque"))

    files = {e: args.ladder.format(e) for e in ("low", "medium", "high", "max")}
    if any(os.path.exists(v) for v in files.values()):
        parts.append(ladder_table(
            files,
            "Capability ladder: {model} with reasoning effort swept, same prompt "
            "and universes throughout. The final column is the shortfall against "
            "the structured-evidence baseline.",
            "tab:ladder", allow_partial=args.allow_partial))

    if os.path.exists(args.twins):
        d = json.load(open(args.twins))
        parts.append(twin_table(
            d["results"],
            f"Counterfactual twin worlds. Model rows run on a subset; $n$ is per "
            f"row. One complement's "
            "availability is moved across the horizon; everything else is held "
            "fixed under common random numbers. Chance is 0.500.",
            "tab:twins"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n\n".join(parts) + "\n")
    print(f"wrote {args.out}  ({len(parts) - 1} tables)")


if __name__ == "__main__":
    main()
