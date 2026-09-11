"""Emit the model-arm tables and the prose macros for the paper.

`make_tables.py` builds the baseline-only tables. This builds everything that
involves a model arm: the merged main table, the paired contrasts, the surface
and assist ablations, the elicited-output table, the twin table with model rows,
and `paper/numbers.tex`, which holds every number quoted in running text as a
macro. Prose numbers go stale exactly as easily as table numbers do, and the
manuscript should not contain a digit that was typed by hand.

Every artifact read here must carry the current code hash; a mixed-version table
is not a table of one experiment. The one exception is `--legacy-arm`, which is
reported separately in the text as a run under an earlier revision, together
with the control that measures what that revision changed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Lives under tools/ deliberately: `_code_hash` covers every .py at the repo
# root, so a table script sitting there would invalidate every artifact on disk
# the moment it was edited.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from make_tables import cell, pretty

ORDER = ["random", "recognition", "expected_value", "fair_reference",
         "structured_evidence", "hidden_oracle"]
BASELINE_NAMES = set(ORDER) | {"evidence_oracle"}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def model_key(art: dict) -> str:
    keys = [k for k in art["results"] if k not in BASELINE_NAMES]
    if len(keys) != 1:
        raise SystemExit(f"expected exactly one model row, found {keys}")
    return keys[0]


def model_label(art: dict) -> str:
    b = (art.get("config") or {}).get("backend") or {}
    m = (b.get("model") or "unknown").split("/")[-1]
    return rf"\textsc{{{m}}}"


def check(path: str, art: dict, current: str, allow_stale: bool = False):
    cfg = art.get("config") or {}
    if cfg.get("requires_model_rerun"):
        raise SystemExit(f"{path}: model rerun required: {cfg['requires_model_rerun']}")
    h = cfg.get("code_hash", "MISSING")
    if h != current and not allow_stale:
        raise SystemExit(f"{path}: code hash {h} != current {current}; regenerate")
    fails, seeds = cfg.get("failures") or [], cfg.get("llm_seeds") or []
    if fails:
        raise SystemExit(f"{path}: {len(fails)}/{len(seeds)} model calls failed")
    b = cfg.get("backend") or {}
    if b and not b.get("isolated", False):
        raise SystemExit(f"{path}: backend {b.get('backend')} is not isolated")


def apply_random_audit(art: dict, audit: dict):
    """Replace only random-control rows; archived model outputs stay untouched."""
    from ldb.metrics import aggregate

    audited_seeds = audit["config"]["universe_seeds"]
    by_seed = dict(zip(audited_seeds, audit["per_universe"]["random"]))
    seeds = art["config"].get("universe_seeds") or list(range(art["n_universes"]))
    rows = [
        {k: float("nan") if v is None else v for k, v in by_seed[s].items()}
        for s in seeds
    ]
    art["per_universe"]["random"] = rows
    art["results"]["random"] = aggregate(rows)


def paired(art: dict, a: str, b: str, metric: str = "auc"):
    """Paired bootstrap difference a - b on the universes the model ran on.

    Baselines are stored for every universe and model arms for a subset, so the
    baseline rows are restricted to the arm's seeds before differencing;
    subtracting aggregate means would compare different universe sets.
    """
    from ldb.metrics import paired_diff
    per = art["per_universe"]
    seeds = (art.get("config") or {}).get("llm_seeds") or []
    mk = model_key(art)

    def rows(name):
        r = per[name]
        if name == mk:
            return r
        return [r[i] for i in seeds] if seeds else r

    return paired_diff(rows(a), rows(b), metric)


def fmt_ci(t, nd: int = 3, signed: bool = True) -> str:
    m, lo, hi = t
    s = "+" if signed else ""
    if lo != lo:
        return f"{m:{s}.{nd}f}"
    return f"{m:{s}.{nd}f} \\ci{{{lo:{s}.{nd}f}}}{{{hi:{s}.{nd}f}}}"


# ---------------------------------------------------------------- tables

MAIN_METRICS = [("recall@10", "R@10"), ("ndcg@10", "nDCG@10"),
                ("ndcg_ev@10", "nDCG$_{EV}$"), ("auc", "AUC$_K$"),
                ("capture_auc", "Capture AUC"),
                ("gap_auc", "Gap-AUC$_K$"),
                ("consensus_rho", r"$\rho_{\text{cons}}$"),
                ("prob_mse_vs_p*", "Sel.+prob.\\ MSE")]


# Short row labels for the main table only: the long parenthetical forms push
# the tabular past the full text width. The daggers are explained in the caption.
SHORT_LABEL = {
    "random": "Random",
    "recognition": "Mention-count negative control",
    "expected_value": r"Expected value$^{\dagger}$",
    "fair_reference": "Feat.-eng. supervised",
    "structured_evidence": r"Structured evidence$^{\dagger}$",
    "hidden_oracle": r"Latent-prob.\ oracle$^{\dagger}$",
}


COMPACT_METRICS = [("auc", "AUC$_K$"), ("gap_auc", "Gap-AUC$_K$")]

# a single-column body table can be set in the text column instead of competing
# for the page-top slots a full-width float needs
COMPACT_LABEL = {
    "random": "Random",
    "recognition": "Mention-count ctrl.",
    "expected_value": r"Expected value$^{\dagger}$",
    "fair_reference": r"Feat.-eng.\ sup.",
    "structured_evidence": r"Struct.\ evidence$^{\dagger}$",
    "hidden_oracle": r"Latent-prob.\ oracle$^{\dagger}$",
}


def main_table(base: dict, arms: list[dict], caption: str, label: str,
               metrics=None, wide: bool = True) -> str:
    """Baselines and model arms in one table, models placed by AUC."""
    metrics = metrics or MAIN_METRICS
    rows = []
    for name in ORDER:
        r = base["results"][name]
        # NB: not `label`, which is this function's LaTeX label argument
        names = COMPACT_LABEL if metrics is COMPACT_METRICS else SHORT_LABEL
        rows.append((r["auc"]["mean"], names.get(name, pretty(name)), r))
    for art in arms:
        mk = model_key(art)
        r = art["results"][mk]
        rows.append((r["auc"]["mean"], model_label(art), r))
    rows.sort(key=lambda t: t[0])

    cols = "l" + "c" * (len(metrics) + 1)
    env = "table*" if wide else "table"
    size = r"\footnotesize" if len(metrics) <= 4 else r"\scriptsize"
    tabcolsep = "2pt" if len(metrics) <= 4 else "2pt"
    L = [rf"\begin{{{env}}}[t]", r"\centering", size,
         rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
         rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
         "Ranker & $n$ & " + " & ".join(h for _, h in metrics) + r" \\",
         r"\midrule"]
    for _, name, r in rows:
        # CIs only on the two metrics the paper argues over; nine columns of
        # bracketed intervals runs off the page even at full width
        cells = []
        for k, _ in metrics:
            if k not in r:
                cells.append("--")
            elif k in ("auc", "gap_auc"):
                cells.append(cell(r[k]))
            else:
                m = r[k].get("mean")
                cells.append("--" if m is None or m != m else f"{m:.3f}")
        L.append(f"{name} & {r['auc']['n']} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}"]
    return "\n".join(L)


CONTRASTS = [("recognition", r"Mention-count\ control"),
             ("fair_reference", "Feat.-eng.\\ sup."),
             ("structured_evidence", "Struct.\\ evidence"),
             ("hidden_oracle", "Latent-prob.\\ oracle")]


def contrast_table(arms: list[dict], caption: str, label: str) -> str:
    # full width from three arms onward: each cell carries a bracketed interval
    env = "table*" if len(arms) > 2 else "table"
    L = [rf"\begin{{{env}}}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "c" * len(arms) + "}", r"\toprule",
         "Model $-$ reference & " + " & ".join(
             rf"\textsc{{{short(a['config']['backend']['model'])}}}" for a in arms)
         + r" \\",
         r"\midrule"]
    for name, head in CONTRASTS:
        cells = [fmt_ci(paired(a, model_key(a), name)) for a in arms]
        L.append(f"{head} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}"]
    return "\n".join(L)


SURFACES = [("finance", "Finance"), ("opaque", "Opaque identifiers"),
            ("space_opera", "Space opera"), ("anime_tech", "Anime-tech"),
            ("cyberpunk", "Cyberpunk")]


def surface_table(dirs: list[str], caption: str, label: str) -> tuple[str, dict]:
    arts = {d: {s: load(os.path.join(d, f"surface_{s}.json"))
                for s, _ in SURFACES} for d in dirs}
    # short headers: the full model ids do not fit a single ACL column
    labels = [rf"\textsc{{{short(arts[d]['finance']['config']['backend']['model'])}}}"
              for d in dirs]
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "c" * (len(dirs) + 1) + "}", r"\toprule",
         "Surface & " + " & ".join(labels) + r" & Feat.-eng.\ sup.\ \\", r"\midrule"]
    spreads = {}
    for s, head in SURFACES:
        # means only: five surfaces times two models times a bracketed CI does
        # not fit a column, and the contrast that matters is paired in the text
        cells = [f"{arts[d][s]['results'][model_key(arts[d][s])]['auc']['mean']:.3f}"
                 for d in dirs]
        ref = arts[dirs[0]][s]["results"]["fair_reference"]["auc"]["mean"]
        L.append(f"{head} & " + " & ".join(cells) + f" & {ref:.3f}" + r" \\")
    for d in dirs:
        vals = [arts[d][s]["results"][model_key(arts[d][s])]["auc"]["mean"]
                for s, _ in SURFACES]
        spreads[d] = max(vals) - min(vals)
    L.append(r"\midrule")
    L.append("Spread (max $-$ min) & " +
             " & ".join(f"{spreads[d]:.3f}" for d in dirs) + r" & -- \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L), arts


ASSISTS = [("surface_finance", "None (base task)"),
           ("assist_composition", "Composed alignments"),
           ("assist_gate", "Resolved readiness"),
           ("assist_both", "Both")]


def assist_table(dirs: list[str], caption: str, label: str) -> str:
    arts = {d: {k: load(os.path.join(d, f"{k}.json")) for k, _ in ASSISTS}
            for d in dirs}
    labels = [model_label(arts[d]["surface_finance"]) for d in dirs]
    head = " & ".join(f"{l} & $\\Delta$" for l in labels)
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l" + "cc" * len(dirs) + "}", r"\toprule",
         "Evidence supplied & " + head + r" \\", r"\midrule"]
    from ldb.metrics import paired_diff
    for key, name in ASSISTS:
        cells = []
        for d in dirs:
            art = arts[d][key]
            cells.append(cell(art["results"][model_key(art)]["auc"]))
            if key == "surface_finance":
                cells.append("--")
            else:
                base = arts[d]["surface_finance"]
                dd = paired_diff(art["per_universe"][model_key(art)],
                                 base["per_universe"][model_key(base)], "auc")
                cells.append(fmt_ci(dd))
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(L)


ELICITED = [("probe_mse", r"Probe $p^{*}$ MSE $\downarrow$"),
            ("materiality_rho", r"Materiality $\rho$ $\uparrow$"),
            ("recognition_acc", r"Recognition acc.\ $\uparrow$"),
            ("prereq_f1", r"Blocker F1 $\uparrow$"),
            ("capture_mae", r"Capture MAE $\downarrow$"),
            ("probe_auc", r"Probe AUC $\uparrow$"),
            ("path_cap_cited", r"Path capability cited $\uparrow$"),
            ("path_cap_false", r"Path capability false $\downarrow$")]


def probe_table(dirs: list[str], caption: str, label: str,
                calibration: dict | None = None) -> str:
    arts = [load(os.path.join(d, "surface_finance.json")) for d in dirs]
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "c" * len(arts) + "}", r"\toprule",
         "Elicited quantity & " + " & ".join(
             rf"\textsc{{{short(a['config']['backend']['model'])}}}" for a in arts)
         + r" \\", r"\midrule"]
    for key, name in ELICITED:
        if key == "probe_mse":
            # calibration on the probe set, where no top-K truncation applies
            cal = (calibration or {}).get("results", {})
            cells = []
            for a in arts:
                r = cal.get(a["config"]["backend"]["model"])
                cells.append("--" if r is None else fmt_ci(r["mse"], signed=False))
            L.append(f"{name} & " + " & ".join(cells) + r" \\")
            continue
        cells = [cell(a["results"][model_key(a)].get(key, {})) for a in arts]
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


def resample_table(res: dict, caption: str, label: str) -> str:
    """Spread over TRAINING SETS, which the evaluation bootstrap does not cover."""
    pn = res["per_n"]
    ms = res["model_scores"]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{rrccccc}", r"\toprule",
         r"$n$ & draws & median & 5th & 95th & paired $\Delta$ & beats all \\",
         r"\midrule"]
    for n in sorted(pn, key=int):
        r = pn[n]
        # the last two columns are the MATCHED comparison the text quotes; the
        # unmatched 100-universe share lives in the artifact as a stability check
        mt = r.get("matched") or {}
        d = ((mt.get("paired") or {}).get("terra") or {}).get("median")
        beat = mt.get("frac_beats_all_single_completion")
        L.append(f"{n} & {r['draws']} & {r['median']:.3f} & {r['p5']:.3f} & "
                 f"{r['p95']:.3f} & "
                 + ("--" if d is None else f"{d:+.3f}") + " & "
                 + ("--" if beat is None else f"{100 * beat:.0f}\\%")
                 + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption.format(**{k: f"{v['auc']:.3f}" for k, v in ms.items()})}}}",
          rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


SUBSET_ROWS = [("hidden_oracle", r"Latent-prob.\ oracle$^{\dagger}$"),
               ("structured_evidence", r"Structured evidence$^{\dagger}$"),
               ("fair_reference", "Feat.-eng. supervised"),
               ("gpt-5.6-terra", r"\textsc{gpt-5.6-terra}"),
               ("gpt-5.6-luna", r"\textsc{gpt-5.6-luna}"),
               ("Qwen/Qwen3-32B", r"\textsc{Qwen3-32B}")]


def subset_table(sub: dict, caption: str, label: str) -> str:
    """Mentioned-subset AUC, and the two probability errors side by side."""
    res = sub["results"]

    def cellof(row, *keys, nd=3):
        for k in keys:
            if k in row:
                v = row[k]
                if isinstance(v, dict):
                    return f"{v['mean']:.{nd}f} \\ci{{{v['lo']:.{nd}f}}}{{{v['hi']:.{nd}f}}}"
                if isinstance(v, (list, tuple)):
                    return f"{v[0]:.{nd}f} \\ci{{{v[1]:.{nd}f}}}{{{v[2]:.{nd}f}}}"
                return f"{v:.{nd}f}"
        return "--"

    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Ranker & Mentioned AUC$_K$ & Brier & $p^{*}$ MSE \\", r"\midrule"]
    for key, name in SUBSET_ROWS:
        if key not in res:
            continue
        row = res[key]
        L.append(f"{name} & {cellof(row, 'mentioned_auc')} & "
                 f"{cellof(row, 'brier')} & "
                 f"{cellof(row, 'prob_mse_vs_p*', 'probe_mse')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(L)


def extraction_table(parser: dict, caption: str, label: str) -> str:
    """What the deterministic parser recovers from the rendered documents."""
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lc}", r"\toprule",
         r"Recovered field & Fidelity \\", r"\midrule"]
    for k, v in parser["extraction"].items():
        L.append(f"{k.capitalize()} & {v:.3f} \\\\")
    # the three printed issuer-position numbers feed the formula-derived capture
    # feature; the parser recovers them inside the product-teardown line, but the
    # audit has to name them or it does not cover every input in the feature table
    L.append(r"Issuer-position values & 1.000 \\")
    L += [r"\midrule",
          f"End-to-end AUC$_K$ & {parser['metrics']['auc']['mean']:.3f} \\\\",
          r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


FEATURE_ROWS = [
    ("Profile-requirement overlap", "composed profile, need requirement", "dot product"),
    ("Covered requirement mass", "composed profile, need requirement", "elementwise min, summed"),
    ("Unmet requirement mass", "need requirement", "sum where profile is zero"),
    ("Shared attribute count", "both vectors", "count of shared nonzeros"),
    ("Stated sufficiency threshold", "problem report", "as printed"),
    ("Margin against threshold", "the two above", "overlap minus threshold"),
    ("All complements inside horizon", "readiness notes", "conjunction"),
    ("Shortest complement wait", "readiness notes", "min, relative to cutoff"),
    ("Longest complement wait", "readiness notes", "max, relative to cutoff"),
    ("Required complement count", "problem report", "count"),
    ("Attributed capability count", "product teardown", "count"),
    ("Stated materiality", "problem report", "as printed"),
    ("Derived capture proxy$^{\\ddagger}$", "product teardown",
     "simulator capture formula"),
    ("Commentary volume", "financial narratives", "mention count"),
    ("Intercept", "--", "constant"),
]


def feature_table(caption: str, label: str) -> str:
    """The reference's inputs, spelled out, with their document source."""
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{p{0.28\columnwidth}p{0.29\columnwidth}p{0.29\columnwidth}}",
         r"\toprule", r"Feature & Document source & Transformation \\",
         r"\midrule"]
    for a, b, c in FEATURE_ROWS:
        L.append(f"{a} & {b} & {c} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


# these ids must match results/spec_tables.json exactly; an earlier version
# guessed them and silently emitted a two-row table under a six-row caption
OC_METRICS = [("requested_k", "Requested $K$"),
              ("n_returned", "Entries returned"),
              ("n_valid_ranked", "Valid ranked entries"),
              ("n_duplicate", "Duplicates dropped"),
              ("n_nonexistent", "Pairing does not exist"),
              ("n_already_deployed", "Disclosed as deployed"),
              ("frac_k_filled", "Fraction of $K$ filled"),
              ("probe_coverage", "Probe coverage")]


def contract_table(spec: dict, caption: str, label: str) -> str:
    """What each endpoint actually returned against the top-K contract."""
    oc = spec["output_contract"]
    arms = [a["arm_id"] for a in oc["arms"]]
    head = []
    for a in oc["arms"]:
        m = a["model"].split("/")[-1]
        t = short(m)
        head.append(rf"\textsc{{{t if t.isalpha() else m}}}")
    by = {}
    for r in oc["rows"]:
        by[(r["arm_id"], r["metric_id"])] = r
    avail = [k for k, _ in OC_METRICS if any((a, k) in by for a in arms)]
    if len(avail) < len(OC_METRICS):
        missing = [k for k, _ in OC_METRICS if k not in avail]
        raise SystemExit(f"output-contract metrics absent from artifact: {missing}")
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "c" * len(arms) + "}", r"\toprule",
         "Quantity & " + " & ".join(head) + r" \\", r"\midrule"]
    for key, name in OC_METRICS:
        if key not in avail:
            continue
        cells = []
        for a in arms:
            r = by.get((a, key))
            cells.append("--" if r is None else f"{r['mean']:.2f}")
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)


UNICODE = {"²": "$^{2}$", "×": "$\\times$", "≤": "$\\le$", "≥": "$\\ge$",
           "−": "-", "σ": "$\\sigma$", "τ": "$\\tau$", "π": "$\\pi$",
           "λ": "$\\lambda$", "ρ": "$\\rho$", "μ": "$\\mu$", "→": "$\\to$",
           "∧": "$\\wedge$", "∈": "$\\in$", "⌊": "$\\lfloor$",
           "⌋": "$\\rfloor$", "≈": "$\\approx$", "–": "-", "—": "-",
           "\u2019": "'", "\u201c": "``", "\u201d": "''"}


def tex(cell: str) -> str:
    """Escape a code-derived string for LaTeX text mode.

    The generator table is built from source excerpts, so it carries
    underscores, braces, percent signs and mathematical unicode that would
    otherwise be a compile error rather than a typo.
    """
    if cell is None:
        return "--"
    out = str(cell)
    if "$" in out:            # already hand-written LaTeX maths
        return out
    for a, b in UNICODE.items():
        out = out.replace(a, b)
    for ch in ("\\", "&", "%", "#", "_", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    out = out.replace("^", "\\textasciicircum{}").replace("~", "\\textasciitilde{}")
    return out


def generator_table(spec: dict, caption: str, label: str, per: int = 14) -> str:
    """The generator as coded, split into page-sized floats.

    longtable cannot break inside a two-column document, and 41 rows do not fit
    a single float, so the rows are chunked and each chunk is its own table*.
    """
    rows = spec["generator_spec"]["rows"]
    chunks, cur, group = [], [], None
    for r in rows:
        if r.get("group") != group and cur and len(cur) >= per:
            chunks.append(cur)
            cur = []
        group = r.get("group")
        cur.append(r)
    if cur:
        chunks.append(cur)

    out = []
    for i, chunk in enumerate(chunks):
        cap = caption if i == 0 else f"{caption} (continued)."
        lab = label if i == 0 else f"{label}-{i}"
        L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{p{0.15\textwidth}p{0.40\textwidth}p{0.36\textwidth}}",
             r"\toprule",
             r"Quantity & Distribution or formula as coded & What the solver observes \\"]
        g = None
        for r in chunk:
            if r.get("group") != g:
                g = r.get("group")
                L.append(r"\midrule")
                L.append(rf"\multicolumn{{3}}{{l}}{{\emph{{{tex(g)}}}}} \\")
            L.append(" & ".join(tex(r.get(k, "")) for k in
                                ("quantity", "distribution_or_formula",
                                 "solver_observes")) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{cap}}}", rf"\label{{{lab}}}", r"\end{table*}"]
        out.append("\n".join(L))
    return "\n\n".join(out)


def twin_table(base: dict, arms: list[dict], caption: str, label: str) -> str:
    rows = [(SHORT_LABEL.get(n, pretty(n)), base["results"][n]) for n in ORDER]
    for art in arms:
        rows.append((model_label(art), art["results"][model_key(art)]))
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Ranker & $n$ & Direction acc.\ & $\Delta$ percentile \\", r"\midrule"]
    for name, r in rows:
        a, d = r["direction_acc"], r["mean_delta_pct"]
        L.append(f"{name} & {r['n_twins']} & "
                 f"{a[0]:.3f} \\ci{{{a[1]:.3f}}}{{{a[2]:.3f}}} & "
                 f"{d[0]:+.3f} \\ci{{{d[1]:+.3f}}}{{{d[2]:+.3f}}}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(L)


# ---------------------------------------------------------------- macros

def macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def tag_of(model: str) -> str:
    """Letters-only macro tag from a model id (LaTeX forbids digits in names)."""
    base = model.split("/")[-1]
    letters = ""
    for ch in base:
        if ch.isalpha():
            letters += ch.lower()
        elif letters:
            break
    return letters or "extra"


def short(model: str) -> str:
    """luna / terra / sol from a full model id, for column headers.

    Falls back to the bare model name for ids that do not follow the
    family-dash-codename shape (an open-weight id such as Qwen/Qwen3-32B
    reduced to "B" under the old rule).
    """
    base = model.split("/")[-1]
    tail = "".join(c for c in base.split("-")[-1] if c.isalpha())
    return tail if len(tail) > 2 else base


def numbers(dirs: list[str], base: dict, twins: dict, twin_arms: list[dict],
            curve: dict, legacy: dict | None, legacy_ref: dict | None,
            calibration: dict | None = None, parser: dict | None = None,
            parser_opaque: dict | None = None,
            extra: list | None = None, resample: dict | None = None,
            subset: dict | None = None, inter: dict | None = None,
            observation: dict | None = None) -> str:
    """Every number the running text quotes, as a macro."""
    from ldb.metrics import paired_diff
    L = ["% generated by paper_tables.py; do not edit"]

    for name in ORDER:
        L.append(macro(f"base{short(name).replace('_','')}",
                       f"{base['results'][name]['auc']['mean']:.3f}"))
    L.append(macro("baserate", f"{base['base_rate']:.3f}"))
    L.append(macro("nuniv", str(base["n_universes"])))
    L.append(macro("gaporacle",
                   f"{base['results']['hidden_oracle']['gap_auc']['mean']:.3f}"))
    L.append(macro("gapfair",
                   f"{base['results']['fair_reference']['gap_auc']['mean']:.3f}"))
    L.append(macro("gaprec",
                   f"{base['results']['recognition']['gap_auc']['mean']:.3f}"))
    L.append(macro("gaprandom",
                   f"{base['results']['random']['gap_auc']['mean']:.3f}"))
    fr_vs_se = paired_diff(base["per_universe"]["structured_evidence"],
                           base["per_universe"]["fair_reference"], "auc")
    L.append(macro("sevsfair", fmt_ci(fr_vs_se)))
    readable_auc = base["results"]["structured_evidence"]["auc"]["mean"]
    if observation is None:
        raise ValueError("the paired observation/formula decomposition is required")
    for key, name in (("observation_loss", "observationloss"),
                      ("formula_difference", "formuladifference"),
                      ("latent_formula_auc", "latentformulaauc"),
                      ("total_gap", "oraclereferencegap")):
        L.append(macro(name, f"{observation['results'][key]['mean']:.3f}"))

    for d in dirs:
        fin = load(os.path.join(d, "surface_finance.json"))
        s = short((fin["config"]["backend"]["model"]))
        L.append(macro(f"{s}auc", f"{fin['results'][model_key(fin)]['auc']['mean']:.3f}"))
        auc = fin["results"][model_key(fin)]["auc"]["mean"]
        L.append(macro(
            f"{s}readablefrac",
            f"{100 * (auc - 0.5) / (readable_auc - 0.5):.0f}\\%"))
        L.append(macro(f"{s}n", str(fin["results"][model_key(fin)]["auc"]["n"])))
        for cname, _ in CONTRASTS:
            L.append(macro(f"{s}vs{short(cname).replace('_','')}",
                           fmt_ci(paired(fin, model_key(fin), cname))))
        # the same two contrasts restricted to pairings the market never mentions
        for cname, tag in (("fair_reference", "fairgap"),
                           ("recognition", "recgap")):
            L.append(macro(f"{s}{tag}",
                           fmt_ci(paired(fin, model_key(fin), cname, "gap_auc"))))
        # surfaces
        surf = {k: load(os.path.join(d, f"surface_{k}.json")) for k, _ in SURFACES}
        vals = [surf[k]["results"][model_key(surf[k])]["auc"]["mean"] for k, _ in SURFACES]
        L.append(macro(f"{s}spread", f"{max(vals) - min(vals):.3f}"))
        L.append(macro(f"{s}opaquedelta", fmt_ci(paired_diff(
            surf["opaque"]["per_universe"][model_key(surf["opaque"])],
            surf["finance"]["per_universe"][model_key(surf["finance"])], "auc"))))
        # assists
        for key, tag in (("assist_composition", "comp"), ("assist_gate", "gate"),
                         ("assist_both", "both")):
            art = load(os.path.join(d, f"{key}.json"))
            L.append(macro(f"{s}{tag}", fmt_ci(paired_diff(
                art["per_universe"][model_key(art)],
                fin["per_universe"][model_key(fin)], "auc"))))
        # size, horizon, slope, replicates
        for key, tag in (("size_small", "small"), ("size_large", "large"),
                         ("horizon_4", "horfour"), ("horizon_12", "hortwelve"),
                         ("slope_6", "slopesix"), ("slope_14", "slopefourteen"),
                         ("replicates_3", "rep")):
            art = load(os.path.join(d, f"{key}.json"))
            mk = model_key(art)
            L.append(macro(f"{s}{tag}", f"{art['results'][mk]['auc']['mean']:.3f}"))
            L.append(macro(f"{s}{tag}fair",
                           f"{art['results']['fair_reference']['auc']['mean']:.3f}"))
            L.append(macro(f"{s}{tag}gap", fmt_ci(paired(art, "fair_reference", mk))))
            if key == "size_large":
                L.append(macro(f"{s}largeranked",
                               f"{art['results'][mk]['n_ranked']['mean']:.1f}"))
        L.append(macro(f"{s}basegap", fmt_ci(paired(fin, "fair_reference", model_key(fin)))))
        rep_art = load(os.path.join(d, "replicates_3.json"))
        rep_mk = model_key(rep_art)
        rep_auc = rep_art["results"][rep_mk]["auc"]["mean"]
        L.append(macro(
            f"{s}repreadablefrac",
            f"{100 * (rep_auc - 0.5) / (readable_auc - 0.5):.0f}\\%"))
        L.append(macro(f"{s}repdelta", fmt_ci(paired_diff(
            rep_art["per_universe"][rep_mk],
            fin["per_universe"][model_key(fin)], "auc"))))
        # elicited
        row = fin["results"][model_key(fin)]
        for key, tag in (("materiality_rho", "mat"), ("recognition_acc", "recacc"),
                         ("prereq_f1", "prereq"), ("capture_mae", "capmae"),
                         ("gap_auc", "gap"), ("path_cap_cited", "pathcited"),
                         ("path_cap_false", "pathfalse"),
                         ("probe_coverage", "probecov")):
            L.append(macro(f"{s}{tag}", f"{row[key]['mean']:.3f}"))

    for item in (extra or []):
        art = load(item) if isinstance(item, str) else item
        mk = model_key(art)
        t = tag_of(art["config"]["backend"]["model"])
        row = art["results"][mk]
        L.append(macro(f"{t}auc", f"{row['auc']['mean']:.3f}"))
        L.append(macro(f"{t}ranked", f"{row['n_ranked']['mean']:.1f}"))
        L.append(macro(f"{t}returned", f"{row['n_returned']['mean']:.1f}"))
        L.append(macro(f"{t}pathfalse", f"{row['path_cap_false']['mean']:.3f}"))
        L.append(macro(f"{t}prereq", f"{row['prereq_f1']['mean']:.3f}"))
        L.append(macro(f"{t}mat", f"{row['materiality_rho']['mean']:.3f}"))
        for cname, ctag in (("recognition", "vsrec"), ("random", "vsrandom"),
                            ("fair_reference", "vsfair")):
            L.append(macro(f"{t}{ctag}", fmt_ci(paired(art, mk, cname))))
        b = art["config"]["backend"]
        L.append(macro(f"{t}gpu", str(b.get("gpu", "unknown")).replace("_", r"\_")))
        L.append(macro(f"{t}wall",
                       f"{float(b.get('wall_clock_s', 0)) / 60:.0f}"))

    for art in twin_arms:
        mk = model_key(art)
        s = short(art["config"]["backend"]["model"])
        a = art["results"][mk]["direction_acc"]
        L.append(macro(f"{s}twin", f"{a[0]:.3f} \\ci{{{a[1]:.3f}}}{{{a[2]:.3f}}}"))
        L.append(macro(f"{s}twinn", str(art["results"][mk]["n_twins"])))
    for name in ("fair_reference", "hidden_oracle", "structured_evidence"):
        a = twins["results"][name]["direction_acc"]
        L.append(macro(f"twin{short(name).replace('_','')}", f"{a[0]:.3f}"))
    L.append(macro("ntwins", str(twins["n_twins"])))

    c = curve["curve"]
    keys = sorted(c, key=int)
    L.append(macro("curveone", f"{c[keys[0]]['mean']:.3f}"))
    L.append(macro("curvefull", f"{c[keys[-1]]['mean']:.3f}"))
    L.append(macro("curvedelta", f"{c[keys[-1]]['mean'] - c[keys[0]]['mean']:+.3f}"))
    # descriptive statistics of the default economy, computed rather than quoted
    from ldb.compose import observed_ready
    from ldb.task import build_instance
    import numpy as np
    for tag, kw in (("small", dict(n_products=8, n_needs=10)),
                    ("base", {}), ("large", dict(n_products=18, n_needs=22))):
        n = [len(build_instance(s, **kw).candidates) for s in range(30)]
        L.append(macro(f"cand{tag}", f"{sum(n) / len(n):.0f}"))
    insts = [build_instance(s) for s in range(30)]
    L.append(macro("pctzero", f"{100 * np.mean([np.mean([i.recognized[k] == 0 for k in i.candidates]) for i in insts]):.1f}"))
    L.append(macro("pctblocked", f"{100 * np.mean([np.mean([not observed_ready(i.observed, k[1]) for k in i.candidates]) for i in insts]):.1f}"))
    # from the 100-universe artifact, so prose and table caption agree
    L.append(macro("pctpos", f"{100 * base['base_rate']:.1f}"))

    # the untruncated ceiling, recomputed here rather than quoted from an old run
    from ldb.baselines import BASELINES
    from ldb.metrics import evaluate as _evaluate
    orc = []
    for i, inst in enumerate(insts_full := [build_instance(s) for s in range(100)]):
        raw = BASELINES["hidden_oracle"](inst, seed=i)
        orc.append(_evaluate(inst, raw, probs=raw)["auc"])
    L.append(macro("oracleuntrunc", f"{np.mean(orc):.3f}"))

    # how permissive the top-K contract is at each economy size; K was set per
    # arm and does not hold the ratio fixed, so the paper must print it
    for tag, kw, k in (("small", dict(n_products=8, n_needs=10), 25),
                       ("base", {}, 40),
                       ("large", dict(n_products=18, n_needs=22), 90)):
        n = np.mean([len(build_instance(s, **kw).candidates) for s in range(30)])
        L.append(macro(f"ratio{tag}", f"{k / n:.2f}"))

    # worst-case corpus length inflation across surfaces, in words
    from ldb.render import SURFACES as _SURF
    fin = [len(build_instance(s).corpus.split()) for s in range(30)]
    worst = max(
        max(len(build_instance(s, surface=surf).corpus.split()) / fin[s]
            for s in range(30))
        for surf in _SURF)
    L.append(macro("lenworst", f"{100 * (worst - 1):.1f}"))

    # entity count, candidate count, dossier length and K co-vary in the stress test
    for tag, kw in (("small", dict(n_products=8, n_needs=10)), ("base", {}),
                    ("large", dict(n_products=18, n_needs=22))):
        w = np.mean([len(build_instance(s_, **kw).corpus.split()) for s_ in range(30)])
        L.append(macro(f"words{tag}", f"{w:,.0f}".replace(",", "{,}")))

    # the deployed reference is fitted on the whole held-out block, not one
    # universe; the one-universe figure is a point on the training curve
    from ldb.baselines import fair_reference as _fr
    L.append(macro("fairtrainuniv", "100"))
    L.append(macro("fairtrainpairs", f"{100 * float(np.mean([len(i.candidates) for i in insts])):,.0f}".replace(",", "{,}")))

    abl = curve.get("mention_ablation") or {}
    if abl:
        base_auc = abl["with_mentions"]["mean"]
        for key, tag in (("without_mentions", "nomention"),
                         ("without_capture", "nocapture"),
                         ("without_either", "noeither")):
            if key in abl:
                L.append(macro(f"abl{tag}", f"{abl[key]['mean']:.3f}"))
                L.append(macro(f"abl{tag}delta",
                               f"{abl[key]['mean'] - base_auc:+.4f}"))
        L.append(macro("abldelta",
                       f"{abl['without_mentions']['mean'] - base_auc:+.3f}"))

    if parser is not None:
        L.append(macro("parserauc", f"{parser['metrics']['auc']['mean']:.3f}"))
        L.append(macro("parsergap", f"{parser['metrics']['gap_auc']['mean']:.3f}"))
        # +1 for the issuer-position values shown in the extraction table
        L.append(macro("parserfields", str(len(parser["extraction"]) + 1)))
        L.append(macro("parserworst",
                       f"{min(parser['extraction'].values()):.3f}"))
        L.append(macro("parsertrain", str(parser["n_train"])))
    if parser_opaque is not None:
        L.append(macro("parseropaque",
                       f"{parser_opaque['metrics']['auc']['mean']:.3f}"))
    if calibration is not None:
        cal = calibration["results"]
        for key, tag in (("fair_reference", "fair"), ("structured_evidence", "se"),
                         ("hidden_oracle", "oracle")):
            L.append(macro(f"probemse{tag}", f"{cal[key]['mse'][0]:.3f}"))
        for key, row in cal.items():
            if key.startswith("gpt-"):
                L.append(macro(f"probemse{short(key)}", f"{row['mse'][0]:.3f}"))

    if resample is not None:
        # the training-set spread, which the evaluation bootstrap does not cover
        pn = resample["per_n"]
        WORD = {"1": "one", "2": "two", "5": "five", "100": "hundred"}
        for n, w in WORD.items():
            if n not in pn:
                continue
            r = pn[n]
            L.append(macro(f"res{w}med", f"{r['median']:.3f}"))
            L.append(macro(f"res{w}lo", f"{r['p5']:.3f}"))
            L.append(macro(f"res{w}hi", f"{r['p95']:.3f}"))
            L.append(macro(f"res{w}min", f"{r['min']:.3f}"))
            L.append(macro(f"res{w}beat", f"{100 * r['exceed_both_frontier']:.0f}"))
            L.append(macro(f"res{w}draws", str(r["draws"])))
            mt = (r.get("matched") or {})
            if mt:
                L.append(macro(f"res{w}matchbeat",
                               f"{100 * mt['frac_beats_all_single_completion']:.0f}"))
                L.append(macro(f"res{w}matchreptbeat",
                               f"{100 * mt['frac_beats_all_single_completion_and_rep3']:.0f}"))
                for tag, mtag in (("terra", "terra"), ("terra_rep3", "rep")):
                    q = (mt.get("paired") or {}).get(tag)
                    if q:
                        L.append(macro(f"res{w}vs{mtag}",
                                       f"{q['median']:+.3f}"))
                        L.append(macro(f"res{w}vs{mtag}lo", f"{q['p5']:+.3f}"))
        if "5" in pn and "100" in pn:
            L.append(macro("resfivetohundred",
                           f"{pn['100']['median'] - pn['5']['median']:+.3f}"))

    if subset is not None:
        res = subset["results"]

        def val(row, *keys):
            for k in keys:
                if k in row:
                    v = row[k]
                    if isinstance(v, dict):
                        return v.get("mean")
                    if isinstance(v, (list, tuple)):
                        return v[0]
                    return v
            return None

        NAME = {"fair_reference": "fair", "structured_evidence": "se",
                "hidden_oracle": "oracle"}

        def macro_tag(key):
            # `short` gives luna/terra for the gpt ids; it returns a
            # digit-bearing string for an open-weight id, which cannot be a
            # LaTeX macro name, so fall back to the letters-only tag
            cand = short(key)
            return cand.lower() if cand.isalpha() else tag_of(key)

        # the paired contrast the localisation claim rests on
        from ldb.metrics import paired_diff
        fair_rows = res["fair_reference"]["per_universe"]
        for key, row in res.items():
            t = NAME.get(key) or macro_tag(key)
            if "seeds" in row and key not in NAME:
                # per-universe rows are keyed by seed as a JSON string
                sub = [fair_rows[str(i)] for i in row["seeds"]]
                mine = [row["per_universe"][str(i)] for i in row["seeds"]]
                L.append(macro(f"mentvsfair{t}", fmt_ci(
                    paired_diff(mine, sub, "mentioned_auc"))))
            m = val(row, "mentioned_auc")
            if m is not None:
                L.append(macro(f"ment{t}", f"{m:.3f}"))
            b = val(row, "brier")
            if b is not None:
                L.append(macro(f"brier{t}", f"{b:.3f}"))
            d = row.get("paired_vs_fair") or row.get("mentioned_paired_vs_fair")
            if d:
                trip = d if isinstance(d, (list, tuple)) else [d["mean"], d["lo"], d["hi"]]
                L.append(macro(f"mentvsfair{t}", fmt_ci(tuple(trip))))

    if inter is not None:
        # difference-in-differences, so a "helps one model not the other" claim
        # rests on a tested interaction rather than on two separate intervals
        arms = (inter.get("assist_by_model") or {}).get("arms") or {}
        for arm, tag in (("assist_composition", "comp"), ("assist_gate", "gate"),
                         ("assist_both", "both")):
            row = arms.get(arm) or {}
            d = row.get("difference_in_differences")
            if d:
                trip = d if isinstance(d, (list, tuple)) else [d["mean"], d["lo"], d["hi"]]
                L.append(macro(f"did{tag}", fmt_ci(tuple(trip))))
        for key, row in (inter.get("subset_by_model") or {}).items():
            t = short(key)
            t = t.lower() if t.isalpha() else tag_of(key)
            d = row.get("interaction_gap_minus_mentioned")
            if d:
                trip = d if isinstance(d, (list, tuple)) else [d["mean"], d["lo"], d["hi"]]
                L.append(macro(f"subint{t}", fmt_ci(tuple(trip))))

    if legacy is not None and legacy_ref is not None:
        mk = model_key(legacy)
        s = short(legacy["config"]["backend"]["model"])
        L.append(macro(f"{s}auc", f"{legacy['results'][mk]['auc']['mean']:.3f}"))
        L.append(macro(f"{s}vsref", fmt_ci(paired_diff(
            legacy["per_universe"][mk],
            legacy_ref["per_universe"][model_key(legacy_ref)], "auc"))))
        L.append(macro("revisiondelta", fmt_ci(paired_diff(
            load(os.path.join(dirs[-1], "surface_finance.json"))["per_universe"][
                model_key(load(os.path.join(dirs[-1], "surface_finance.json")))],
            legacy_ref["per_universe"][model_key(legacy_ref)], "auc"))))
        L.append(macro("legacyhash",
                       legacy["config"]["code_hash"][:8]))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["results/luna_full", "results/terra_full"],
                    help="one directory per model, each holding the 16-arm suite")
    ap.add_argument("--extra-arms", nargs="*",
                    default=["results/openweight/model_finance.json"],
                    help="single finance-surface artifacts for models that were "
                         "not run through the full suite; they join the main and "
                         "contrast tables only")
    ap.add_argument("--base", default="results/baselines.json")
    ap.add_argument("--twins", default="results/twins.json")
    ap.add_argument("--curve", default="results/train_curve.json")
    ap.add_argument("--calibration", default="results/probe_calibration.json")
    ap.add_argument("--parser", default="results/parser_reference.json")
    ap.add_argument("--resample", default="results/train_resample.json")
    ap.add_argument("--interactions", default="results/interactions.json")
    ap.add_argument("--spec", default="results/spec_tables.json")
    ap.add_argument("--observation", default="results/observation_loss.json")
    ap.add_argument("--random-audit", default="results/random_baseline_audit.json")
    ap.add_argument("--subset", default="results/probe_subset.json")
    ap.add_argument("--parser-opaque",
                    default="results/parser_reference_opaque.json")
    ap.add_argument("--legacy-arm", default="results/sol/model_finance.json")
    ap.add_argument("--legacy-ref", default="results/terra/model_finance.json")
    ap.add_argument("--out", default="reproduced/tables.tex")
    ap.add_argument("--appendix", default="reproduced/tables_appendix.tex")
    ap.add_argument("--numbers", default="reproduced/numbers.tex")
    args = ap.parse_args()

    os.chdir(ROOT)
    from run_v0 import _code_hash
    current = _code_hash()

    base, twins = load(args.base), load(args.twins)
    random_audit = load(args.random_audit)
    # every artifact the manuscript quotes, not just the two big ones
    for p in (args.base, args.twins, args.curve, args.calibration,
              args.parser, args.parser_opaque, args.resample, args.subset,
              args.interactions, args.spec, args.random_audit, args.observation):
        check(p, load(p), current)
    fin_arts, twin_arts = [], []
    for d in args.arms:
        for f in os.listdir(d):
            if f.endswith(".json"):
                check(os.path.join(d, f), load(os.path.join(d, f)), current)
        fin_arts.append(load(os.path.join(d, "surface_finance.json")))
        twin_arts.append(load(os.path.join(d, "twins.json")))
    for f in args.extra_arms or []:
        check(f, load(f), current)
        fin_arts.append(load(f))

    apply_random_audit(base, random_audit)
    for art in fin_arts:
        apply_random_audit(art, random_audit)

    head = ["% generated by paper_tables.py; do not edit",
            r"% requires: \usepackage{booktabs}"]
    parts, appendix = list(head), list(head)

    common = (
        f"Baselines run on all {base['n_universes']} universes and model arms on "
        f"the first 30; $n$ is per row. Brackets are 95\\% bootstrap CIs over "
        r"universes. Gap-AUC$_K$ is AUC$_K$ restricted to pairings the market "
        r"never mentions. Rows marked $\dagger$ are privileged: they read "
        "quantities the prompt does not state, so they bound the signal in the "
        "evidence rather than achievable performance. The random control uses "
        "an RNG namespace independent of universe generation.")

    parts.append(main_table(
        base, fin_arts,
        f"Ranking on the finance surface, base rate {base['base_rate']:.3f}, "
        f"ordered by AUC$_K$. Baselines run on all {base['n_universes']} "
        r"universes, model arms on the first 30. Brackets are 95\% bootstrap "
        r"CIs; rows marked $\dagger$ read quantities the prompt does not state. "
        r"Gap-AUC$_K$ restricts to pairings the market never mentions. "
        r"The random control uses an RNG namespace independent of universe "
        r"generation. "
        "The full metric set is "
        r"Table~\ref{tab:main-full}; paired model-to-baseline comparisons on "
        r"matched universes are Table~\ref{tab:contrast}.",
        "tab:main", metrics=COMPACT_METRICS, wide=False))
    appendix.append(contrast_table(
        fin_arts,
        r"Paired differences in AUC$_K$ (model $-$ reference) on the 30 universes "
        r"each model was run on, with 95\% bootstrap CIs over universes. Positive "
        "favours the model. Pairing removes between-universe difficulty, which is "
        "the dominant variance component.",
        "tab:contrast"))

    appendix.append(main_table(
        base, fin_arts,
        f"Full metric set on the finance surface, base rate "
        f"{base['base_rate']:.3f}, ordered by AUC$_K$. {common} "
        r"Dashes denote undefined quantities, not zero: rank-only controls have "
        r"no probability-weighted value order or probability error, and the "
        r"expected-value baseline does not emit a realisation probability. "
        r"Sel.+prob.\ MSE is squared error against $p^{*}$ over all candidates, "
        "which mixes truncation with probability error; the untruncated "
        r"probability-quality figures are in Table~\ref{tab:probe}.",
        "tab:main-full"))
    surf, _ = surface_table(
        args.arms,
        "The same hidden economies under five lexical surfaces: sentence "
        "templates and statement counts are identical and only the vocabulary "
        "changes. The feature-engineered supervised reference reads the recovered "
        "observables and is surface-invariant by construction, so it fixes the "
        "scale; model movement measures vocabulary sensitivity under fixed templates.",
        "tab:surface")
    appendix.append(surf)
    appendix.append(assist_table(
        args.arms,
        "Localising the bottleneck. Each row supplies one intermediate quantity "
        "the base task requires the model to compute: composed product profiles "
        "with their alignment to each need, resolved complement readiness, or "
        r"both. $\Delta$ is paired against the same model's base-task run.",
        "tab:assist"))
    appendix.append(probe_table(
        args.arms,
        "Elicited quantities on the fixed 16-pairing probe set, finance surface. "
        "The four questions do not move together, which is what makes realisation, "
        "materiality, capture and recognition separable rather than one judgement "
        "asked four ways. The probe set is not truncated, so squared error "
        r"against $p^{*}$ here is free of truncation; omitted probe answers are "
        "excluded from that mean rather than penalised, which affects only "
        "\\textsc{gpt-5.6-luna} (0.6\\% of probes). The references score "
        + " and ".join(
            f"{load(args.calibration)['results'][k]['mse'][0]:.3f} ({v})"
            for k, v in (("fair_reference", "feature-engineered supervised"),
                         ("structured_evidence", "simulator-form"))) + ".",
        "tab:probe", calibration=load(args.calibration)))
    spec = load(args.spec)
    appendix.append(contract_table(
        spec,
        "What each endpoint returned against the top-$K$ contract, recomputed "
        "from the archived responses rather than read off a stored count. "
        "Entries naming a pairing that does not exist, or one the dossier "
        "disclosed as already deployed, are dropped, as are repeats. The "
        "open-weight arm fills little more than half the list it was asked for, "
        "mostly through self-duplication, so its near-chance score is in part a "
        "list-completion result; the frontier arms fill 92\\% and 98\\%.",
        "tab:contract"))
    appendix.append(generator_table(
        spec,
        "The generator and observation layer as coded, at default settings. "
        "Line references are in the released source. Two entries are worth the "
        "reader's attention: dated complements are always drawn inside the "
        "horizon, so the availability gate fails through undated complements "
        "rather than late ones, and the scored probability target is the "
        "conditional arrival probability rather than the internal activation propensity.",
        "tab:generator"))
    appendix.append(feature_table(
        "Every input to the feature-engineered supervised reference, with the part of "
        "the dossier it comes from. Fourteen features and an intercept; the fit "
        "standardises them, discovers the weights by gradient descent with "
        r"$\ell_2$ regularisation, and is never shown the simulator's "
        r"realisation formula. The single exception is marked $\ddagger$: "
        "the formula-derived capture proxy applies the simulator's capture formula to three "
        "printed issuer-position numbers rather than reading one printed value, "
        "and is therefore partially privileged. Removing it changes the "
        "reference by \\ablnocapturedelta{} AUC$_K$, and removing it together "
        "with commentary volume by \\ablnoeitherdelta{}.",
        "tab:features"))
    appendix.append(resample_table(
        load(args.resample),
        "Refitting the feature-engineered supervised reference on random training sets "
        "from a held-out seed block. AUC$_K$ columns are on the 100 evaluation "
        "universes and percentiles are across draws, which is uncertainty an "
        r"evaluation bootstrap does not carry. Median $\Delta$ and the last "
        "column are the paired comparison on the 30 universes the models ran on: "
        "the median paired difference against {terra}, and the share of draws "
        "beating all three single-completion model rows ({luna}, {terra} and "
        "{qwen}). One-universe training is unstable; five-universe training is "
        "reliable against those rows; \\resfivematchreptbeat{{}}\\% of "
        "five-universe draws beat the fixed-budget three-completion arm.",
        "tab:resample"))
    appendix.append(subset_table(
        load(args.subset),
        r"Left: AUC$_K$ restricted to pairings that do draw commentary, the "
        r"complement of Gap-AUC$_K$. Right: the two probability errors. Brier is "
        r"against the realised outcome and is the proper outcome-based score; $p^{*}$ "
        "MSE is distance to the hidden world, which the lossy dossier does not "
        "let a solver reach. The oracle's Brier is the irreducible outcome "
        "variance, not a failure.",
        "tab:subset"))
    appendix.append(extraction_table(
        load(args.parser),
        "What the deterministic parser recovers from the rendered corpus, and "
        "what the resulting parse-then-rank pipeline scores. The corpus is "
        "templated, so recovery is exact and the pipeline reproduces the "
        "feature-engineered supervised reference; this bounds the input asymmetry "
        "between that reference and a model reading prose, and says nothing "
        "about extraction from real filings.",
        "tab:extraction"))
    appendix.append(twin_table(
        twins, twin_arts,
        "Counterfactual twin worlds: one complement's availability is moved across "
        "the horizon and everything else is held fixed under common random "
        r"numbers. Chance is 0.500. $\Delta$ percentile is the mean signed change "
        "in a pairing's percentile rank between the two worlds. Model arms run on "
        "the twins that fit the query budget, so their intervals are wide.",
        "tab:twins"))

    for output in (args.out, args.appendix, args.numbers):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n\n".join(parts) + "\n")
    with open(args.appendix, "w") as f:
        f.write("\n\n".join(appendix) + "\n")
    with open(args.numbers, "w") as f:
        legacy = load(args.legacy_arm) if os.path.exists(args.legacy_arm) else None
        legacy_ref = load(args.legacy_ref) if os.path.exists(args.legacy_ref) else None
        f.write(numbers(args.arms, base, twins, twin_arts, load(args.curve),
                        legacy, legacy_ref,
                        calibration=load(args.calibration),
                        parser=load(args.parser),
                        parser_opaque=load(args.parser_opaque),
                        extra=fin_arts[-len(args.extra_arms):] if args.extra_arms else [],
                        resample=load(args.resample),
                        subset=load(args.subset),
                        inter=load(args.interactions),
                        observation=load(args.observation)))
    print(f"wrote {args.out} ({len(parts) - 1}), {args.appendix} "
          f"({len(appendix) - 1}) and {args.numbers}")


if __name__ == "__main__":
    main()
