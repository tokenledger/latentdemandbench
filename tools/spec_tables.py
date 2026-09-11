"""Two appendix tables, derived from code and artifacts already on disk.

TABLE 1 (`output_contract`): per-model-endpoint accounting of what the model was
asked for versus what it actually returned, and why entries were dropped. The
reviewer's worry is that the headline ranking metric partly measures list
completion. This table separates the two: requested K, returned entries, valid
ranked entries, and every drop reason, recomputed from the archived raw
responses rather than trusted from the stored summary fields.

TABLE 2 (`generator_spec`): the ACTUAL default distributions and parameters of
the universe generator and the observation layer, read off `ldb/universe.py` and
`ldb/render.py`, with parents and what the solver observes for each quantity.

No API calls and no model runs: everything comes from `results/*.json` plus a
re-derivation of each instance with `ldb.task.build_instance`.

Output: `results/spec_tables.json`, structured as lists of row dicts with
explicit field names so a LaTeX table can be generated from it downstream.

Run:  python3 tools/spec_tables.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldb.metrics import bootstrap_ci  # noqa: E402
from ldb.task import build_instance  # noqa: E402
from run_v0 import _code_hash  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spec_tables.json")

ARMS = [
    {"arm_id": "luna", "label": "Luna (frontier, closed)",
     "artifact": "results/luna_full/surface_finance.json"},
    {"arm_id": "terra", "label": "Terra (frontier, closed)",
     "artifact": "results/terra_full/surface_finance.json"},
    {"arm_id": "openweight", "label": "Open-weight",
     "artifact": "results/openweight/model_finance.json"},
]

# What each accounting quantity means. `ci` marks the rows where a bootstrap
# over universes is meaningful; requested K is a constant of the protocol, so a
# CI on it would be an artefact of resampling a degenerate sample.
METRICS = [
    {"metric_id": "requested_k", "label": "Requested $K$", "ci": False,
     "definition": "config.top_k: the cap the prompt asks for and the cap "
                   "score_predictions() enforces. Constant across universes."},
    {"metric_id": "n_returned", "label": "Returned entries", "ci": True,
     "definition": "len(json.loads(raw_response)['predictions']) - every entry "
                   "the model emitted, BEFORE any filtering."},
    {"metric_id": "n_valid_ranked", "label": "Valid ranked entries", "ci": True,
     "definition": "entries that survive score_predictions(): key in "
                   "inst.label, first occurrence, within the top-K cap. These "
                   "are the only entries that receive a rank score."},
    {"metric_id": "n_duplicate", "label": "Duplicates dropped", "ci": True,
     "definition": "a valid candidate pairing repeated after its first "
                   "occurrence; the first (best) position is kept."},
    {"metric_id": "n_nonexistent", "label": "Non-candidate: no such pairing", "ci": True,
     "definition": "(product_id, need_id) is not a pair of the universe at all "
                   "(unknown or malformed ids) - a hallucinated pairing."},
    {"metric_id": "n_already_deployed", "label": "Non-candidate: already deployed", "ci": True,
     "definition": "the pairing exists but activated_at <= cutoff, so the "
                   "dossier's REALIZED DEPLOYMENTS block disclosed it and "
                   "build_instance() excluded it from candidates."},
    {"metric_id": "n_over_cap", "label": "Dropped by top-K cap", "ci": True,
     "definition": "valid, unique entries positioned after the K-th kept "
                   "entry; score_predictions() breaks out of the loop there."},
    {"metric_id": "n_malformed", "label": "Malformed entries", "ci": True,
     "definition": "entries lacking a product_id/need_id field."},
    {"metric_id": "frac_returned_valid", "label": "Valid / returned", "ci": True,
     "definition": "n_valid_ranked / n_returned per universe."},
    {"metric_id": "frac_k_filled", "label": "List completion (valid / $K$)", "ci": True,
     "definition": "n_valid_ranked / requested K per universe - the list "
                   "completion the reviewer asks about."},
    {"metric_id": "probe_coverage", "label": "Probe coverage", "ci": True,
     "definition": "|{fixed probe pairings answered}| / |inst.probes|, "
                   "recomputed from raw_response['probes'].",},
    {"metric_id": "n_probes", "label": "Probe set size", "ci": True,
     "definition": "len(inst.probes): up to 4 per (blocked, mentioned) cell."},
]


def _classify(inst, preds, top_k):
    """Recompute the drop reasons, mirroring ldb.llm.score_predictions.

    score_predictions() keeps an entry iff its key is in inst.label (i.e. it is
    a live candidate), it has not been kept before, and fewer than top_k have
    been kept. Everything else is silently skipped, so the reasons have to be
    rebuilt here. Note the loop in score_predictions BREAKS once top_k entries
    are kept; entries after that point are never inspected by the scorer. We
    still classify them (and record where the scorer stopped) so the accounting
    covers the whole returned list.
    """
    u, cutoff = inst.universe, inst.cutoff
    deployed = {k for k, pr in u.pairs.items() if pr.activated_at <= cutoff}
    seen, kept = set(), 0
    c = {"n_returned": len(preds), "n_valid_ranked": 0, "n_duplicate": 0,
         "n_nonexistent": 0, "n_already_deployed": 0, "n_over_cap": 0,
         "n_malformed": 0, "n_examined_by_scorer": len(preds)}
    for i, p in enumerate(preds):
        if not isinstance(p, dict) or "product_id" not in p or "need_id" not in p:
            c["n_malformed"] += 1
            continue
        key = (p["product_id"], p["need_id"])
        if key in inst.label:
            if key in seen:
                c["n_duplicate"] += 1
            elif kept < top_k:
                seen.add(key)
                kept += 1
                c["n_valid_ranked"] += 1
                if kept == top_k:
                    c["n_examined_by_scorer"] = i + 1
            else:
                c["n_over_cap"] += 1
        elif key in deployed:
            c["n_already_deployed"] += 1
        else:
            c["n_nonexistent"] += 1
    return c


def _arm_rows(arm):
    path = os.path.join(ROOT, arm["artifact"])
    with open(path) as f:
        art = json.load(f)
    cfg = art["config"]
    top_k = cfg["top_k"]
    model_key = [k for k in art["per_universe"] if ":" in k][0]
    stored = art["per_universe"][model_key]
    outputs = art["model_outputs"]
    # model_outputs is keyed by the index into insts, and insts are built from
    # seed = range(n_universes), so index == universe seed.
    seeds = sorted(int(k) for k in outputs)

    per = {m["metric_id"]: [] for m in METRICS}
    mismatches = []
    for pos, seed in enumerate(seeds):
        inst = build_instance(
            seed, cutoff=cfg["cutoff"], horizon=cfg["horizon"],
            surface=cfg["surface"], assist=cfg["assist"],
            n_products=cfg["n_products"], n_needs=cfg["n_needs"],
            n_complements=cfg["n_complements"], slope=cfg["slope"])
        raw = outputs[str(seed)]["raw_response"]
        parsed = json.loads(raw)
        preds = parsed.get("predictions") or []
        c = _classify(inst, preds, top_k)

        probes = {(p.get("product_id"), p.get("need_id"))
                  for p in (parsed.get("probes") or []) if isinstance(p, dict)}
        answered = sum(1 for k in inst.probes if k in probes)
        cov = answered / max(len(inst.probes), 1)

        per["requested_k"].append(float(top_k))
        for k in ("n_returned", "n_valid_ranked", "n_duplicate", "n_nonexistent",
                  "n_already_deployed", "n_over_cap", "n_malformed"):
            per[k].append(float(c[k]))
        per["frac_returned_valid"].append(
            c["n_valid_ranked"] / c["n_returned"] if c["n_returned"] else float("nan"))
        per["frac_k_filled"].append(c["n_valid_ranked"] / top_k)
        per["probe_coverage"].append(cov)
        per["n_probes"].append(float(len(inst.probes)))

        # cross-check the artifact's own summary fields against the recomputation
        srow = stored[pos]
        for field, mine in (("n_returned", c["n_returned"]),
                            ("n_ranked", c["n_valid_ranked"]),
                            ("probe_coverage", cov)):
            theirs = srow.get(field)
            if theirs is None or abs(float(theirs) - float(mine)) > 1e-9:
                mismatches.append({"arm_id": arm["arm_id"], "seed": seed,
                                   "field": field, "stored": theirs,
                                   "recomputed": mine})

    rows = []
    for m in METRICS:
        vals = per[m["metric_id"]]
        mean, lo, hi = bootstrap_ci(vals)
        rows.append({
            "arm_id": arm["arm_id"], "metric_id": m["metric_id"],
            "mean": mean,
            "ci_lo": lo if m["ci"] else None,
            "ci_hi": hi if m["ci"] else None,
            "ci_meaningful": m["ci"],
            "min": min(vals), "max": max(vals), "total": sum(vals),
            "n_universes": len(vals),
            "per_universe": vals,
        })
    meta = {
        "arm_id": arm["arm_id"], "label": arm["label"], "artifact": arm["artifact"],
        "backend": cfg["backend"].get("backend"),
        "model": cfg["backend"].get("model"),
        "effort": cfg["backend"].get("effort"),
        "surface": cfg["surface"], "cutoff": cfg["cutoff"],
        "horizon": cfg["horizon"], "top_k": top_k,
        "replicates": cfg["replicates"],
        "n_universes_scored": len(seeds),
        "universe_seeds": seeds,
        "artifact_code_hash": cfg.get("code_hash"),
        "failures": cfg.get("failures", []),
    }
    return meta, rows, mismatches


# --------------------------------------------------------------------------
# TABLE 2: generator specification, read directly off the source.
# --------------------------------------------------------------------------
def _spec_rows():
    U, R = "ldb/universe.py", "ldb/render.py"
    T = "ldb/task.py"

    def row(group, quantity, symbol, form, parents, observed, source, notes=""):
        return {"group": group, "quantity": quantity, "symbol": symbol,
                "distribution_or_formula": form, "parents": parents,
                "solver_observes": observed, "source": source, "notes": notes}

    return [
        # ---------------- defaults ----------------
        row("Defaults", "Latent dimension", "$D$", "8 (module constant D)", "-",
            "attribute names only: 8 named channels per surface",
            f"{U}:12", "opaque surface renames them A0..A7"),
        row("Defaults", "Timeline length", "$T_{\\max}$", "40 quarters", "-",
            "not stated in the dossier", f"{U}:13",
            "generate_universe raises if cutoff+horizon >= T_MAX"),
        row("Defaults", "Never-activates sentinel", "NEVER", "$10^{9}$", "-",
            "rendered as `no announced availability'", f"{U}:18",
            "deliberately far outside the timeline so no later cutoff sweeps it in"),
        row("Defaults", "Economy size", "-",
            "n_products=12, n_needs=15, n_caps=10, n_complements=8",
            "-", "12 teardowns, 15 problem reports, 10 characterisations, "
                 "8 readiness notes", f"{U}:124-135",
            "180 product-need pairs before the already-deployed exclusion"),
        row("Defaults", "Cutoff / horizon", "$t_c$ / $H$", "cutoff=20, horizon=8",
            "-", "both stated verbatim in the prompt", f"{U}:130-131; {T}:51"),
        row("Defaults", "Logistic slope", "$a$", "9.0", "-",
            "never stated; must be inferred", f"{U}:132"),
        row("Defaults", "Ranking cap", "$K$", "TOP_K = 40", "-",
            "stated in the task text and enforced in scoring", "ldb/llm.py:15"),

        # ---------------- latent structure ----------------
        row("Capabilities", "Capability attribute vector", "$v_c \\in \\mathbb{R}^{8}$",
            "$k \\sim \\mathcal{U}\\{2,3\\}$ nonzero coordinates "
            "(rng.integers(2,4)), positions sampled without replacement from "
            "the 8; each nonzero $\\sim \\mathcal{U}(0.4,1.0)$; all other "
            "coordinates exactly 0",
            "-",
            "$\\mathrm{round}_2(\\max(0,\\, v_c+\\mathcal{N}(0,0.18^2)))$ with "
            "values $<0.12$ zeroed; printed as `attr 0.xx' pairs",
            f"{U}:110-114,144-152; {R}:76-80",
            "sparsity is 2-3 of 8, not a fixed density"),
        row("Capabilities", "Capability / effect names", "-",
            "pseudo-words: onset x coda from two 20-item lists, plus a "
            "`-NN' suffix with prob. 0.4, $NN \\sim \\mathcal{U}\\{2..89\\}$; "
            "resampled until unique within the universe",
            "-", "the name string itself (bare id under the opaque surface)",
            f"{U}:20-41"),
        row("Products", "Capability ownership", "-",
            "$|caps(p)| \\sim \\mathcal{U}\\{1,2,3\\}$ (rng.integers(1,4)), "
            "sampled without replacement from the 10 capabilities",
            "capability set",
            "noisy claimed list: each owned capability kept with prob. "
            "$1-0.15$; each NON-owned capability added with prob. "
            "$0.10/10\\times 3 = 0.03$; if the list empties, the first owned "
            "capability is restored",
            f"{U}:174-189; {R}:82-88",
            "the effective false-add rate is 0.03 per non-owned capability, "
            "not the nominal false_p=0.10"),
        row("Products", "Product profile", "$\\pi_p$",
            "$\\pi_p = \\max_{c \\in caps(p)} v_c$ (element-wise maximum)",
            "capability vectors",
            "never printed; the solver must compose it from the teardown plus "
            "the characterisations",
            f"{U}:178"),
        row("Products", "Company size", "$s_p$",
            "$\\exp(\\mathcal{N}(0,1^2))$ (lognormal, $\\sigma=1$)", "-",
            "NOT rendered anywhere; only its effect via mention counts is visible",
            f"{U}:183"),
        row("Products", "Supply / distribution / moat", "$u,d,m$",
            "each i.i.d. $\\mathcal{U}(0.1,1.0)$", "-",
            "$\\mathrm{round}_2(\\mathrm{clip}(x+\\mathcal{N}(0,0.08^2),0,1))$, "
            "printed on a 0-1 scale in the teardown",
            f"{U}:186-188; {R}:90-92"),
        row("Needs", "Requirement vector", "$r_n \\in \\mathbb{R}^{8}$",
            "$k \\sim \\mathcal{U}\\{2,3,4\\}$ nonzero coordinates "
            "(rng.integers(2,5)); each nonzero $\\sim \\mathcal{U}(0.4,1.0)$",
            "-",
            "$\\mathrm{round}_2(\\max(0,\\, r_n+\\mathcal{N}(0,0.18^2)))$, "
            "values $<0.12$ zeroed",
            f"{U}:198; {R}:96-98",
            "needs are denser than capabilities (2-4 vs 2-3 nonzeros)"),
        row("Needs", "Sufficiency threshold", "$\\tau_n$",
            "$\\mathcal{U}(0.45,0.72)$", "-",
            "$\\mathrm{round}_2(\\mathrm{clip}(\\tau_n+\\mathcal{N}(0,0.04^2),"
            "0.2,0.9))$, printed as `sufficiency threshold near x.xx'",
            f"{U}:199; {R}:99-100"),
        row("Needs", "Materiality", "$M_n$",
            "$\\exp(\\mathcal{N}(0,0.8^2))$ (lognormal, $\\sigma=0.8$)", "-",
            "$\\mathrm{round}_2(\\max(0.01,\\, M_n(1+\\mathcal{N}(0,0.10^2))))$ "
            "- MULTIPLICATIVE 10\\% error, printed as `value if served'",
            f"{U}:200; {R}:103-104"),
        row("Needs", "Complements per need", "-",
            "$|K(n)| \\sim \\mathcal{U}\\{1,2\\}$ (rng.integers(1,3)), sampled "
            "without replacement from the 8 complements",
            "-",
            "each required complement is dropped from the claimed list with "
            "prob. 0.15 (drop_p); if the list empties, the first is restored",
            f"{U}:201-203; {R}:101-102",
            "a dropped complement makes a truly blocked need look unblocked"),
        row("Complements", "Availability date", "$\\rho_k$",
            "$r\\sim\\mathcal{U}(0,1)$: $r<0.45 \\Rightarrow \\rho_k \\sim "
            "\\mathcal{U}\\{0,\\dots,t_c-1\\}$; $0.45\\le r<0.85 \\Rightarrow "
            "\\rho_k \\sim \\mathcal{U}\\{t_c,\\dots,t_c+H-1\\}$; "
            "$r\\ge 0.85 \\Rightarrow \\rho_k = \\text{NEVER}$",
            "-",
            "with prob. 0.12 (and only if dated) the printed date is "
            "$\\max(0,\\rho_k+j)$, $j\\sim\\mathcal{U}\\{-4..4\\}$; rendered as "
            "`available since q', `guided to q', or `no announced availability'",
            f"{U}:154-165; {R}:106-114,260-268",
            "45/40/15 split. The dated branches are ALL <= t_c+H-1, so under "
            "the defaults the only true blocker is NEVER"),

        # ---------------- pair mechanics ----------------
        row("Pairs", "Alignment", "$m_{pn}$",
            "$\\cos(\\pi_p, r_n)$ (0 if either norm is 0)",
            "product profile, need requirement",
            "not printed; recomputable only from the noisy, rounded vectors",
            f"{U}:117-121,209"),
        row("Pairs", "Technical term", "$\\sigma(a(m-\\tau))$",
            "$1/(1+e^{-9.0\\,(m_{pn}-\\tau_n)})$",
            "$m_{pn}$, $\\tau_n$, slope $a=9$",
            "not printed", f"{U}:210"),
        row("Pairs", "Commercial gate", "$g$",
            "$g=1$ if $\\rho_k \\le t_c+H$ for EVERY $k \\in K(n)$, else "
            "$g=0.03$",
            "complement dates, cutoff, horizon",
            "inferable only from the printed (possibly misreported, possibly "
            "dropped) complement dates",
            f"{U}:212-213",
            "the 0.03 residual affects p_true only; a failed gate leaves no "
            "feasible arrival window, so p* and the realised label are zero"),
        row("Pairs", "Internal activation propensity", "$p^{true}_{pn}$",
            "$p^{true} = \\sigma(a(m-\\tau))\\cdot g$",
            "alignment, threshold, gate", "not printed", f"{U}:210-213"),
        row("Pairs", "Activation draw", "$u_{act}$",
            "$u_{act}\\sim\\mathcal{U}(0,1)$ from the private stream "
            "default\\_rng([seed, 10000\\,p+n]); activated iff "
            "$u_{act}<p^{true}$",
            "$p^{true}$",
            "only realised-before-cutoff pairings are disclosed",
            f"{U}:227-235,260-262",
            "one stream per pair, so a counterfactual on one pair cannot "
            "re-roll another"),
        row("Pairs", "Activation quarter", "$t_{pn}$",
            "$lo=\\max(t_c-8,\\ \\max_{k\\in K(n)}\\rho_k)$; "
            "$t = lo + \\lfloor u_{arr}(t_c+H+1-lo)\\rfloor$ if "
            "$lo\\le t_c+H$, else NEVER; NEVER if not activated",
            "$u_{arr}$, complement dates",
            "disclosed only if $t \\le t_c$ (REALIZED DEPLOYMENTS block)",
            f"{U}:236-246",
            "arrival is uniform over a window that starts at most 8 quarters "
            "before the cutoff"),
        row("Pairs", "Exact conditional target", "$p^{*}$",
            "$lo_0=\\max(t_c-8,\\max_k\\rho_k)$; if $lo_0>t_c+H$ then "
            "$p^{*}=0$; else with $span=t_c+H-lo_0+1$, "
            "$n_b=\\max(0,t_c-lo_0+1)$, $p_b=p^{true}n_b/span$, "
            "$p_a=p^{true}(span-n_b)/span$: "
            "$p^{*}=p_a/(1-p_b)$ (0 if $p_b\\ge 1$)",
            "$p^{true}$, complement dates",
            "not observable; it is the latent-probability target, not the realised label",
            f"{U}:215-225",
            "this - NOT $\\sigma(a(m-\\tau))g$ - is what inst.p\\_star holds "
            "and what prob_mse scores against"),
        row("Pairs", "Label", "$y_{pn}$",
            "$y=1$ iff $t_{pn}\\neq$ NEVER and $t_c < t_{pn} \\le t_c+H$",
            "activation quarter", "the quantity to be predicted",
            f"{T}:65"),
        row("Pairs", "Candidate set", "-",
            "every pair with $t_{pn} > t_c$; pairs with $t_{pn}\\le t_c$ are "
            "excluded",
            "activation quarter",
            "the excluded ones are listed explicitly as already deployed",
            f"{T}:61-64; {R}:270-280"),
        row("Capture", "Capture probability", "$p^{cap}_p$",
            "$\\mathrm{clip}(0.15+0.55\\,u_p d_p+0.30\\,m_p,\\,0,\\,1)$",
            "TRUE supply, distribution, moat",
            "only the noisy rounded $u,d,m$ are printed; the coefficients are "
            "not stated",
            f"{U}:249-250",
            "depends on the product only - identical across all 15 needs; "
            "note the supply-distribution PRODUCT term"),
        row("Capture", "Realised capture", "$captured$",
            "$captured = (t_{pn}\\neq \\text{NEVER}) \\wedge (u_{cap} < "
            "p^{cap}_p)$, $u_{cap}$ the third draw of the pair stream",
            "$p^{cap}$, activation", "scored via the capture probe",
            f"{U}:251"),

        # ---------------- commentary ----------------
        row("Commentary", "Per-quarter mention rate", "$\\lambda_{pn}(q)$",
            "$base = 0.02 + 0.10\\tanh(s_p) + 0.05\\,p^{true}_{pn}$; "
            "$\\lambda(q) = \\min(base + 0.45\\cdot\\mathbb{1}[t_{pn}\\le q],"
            "\\ 0.9)$",
            "company size, $p^{true}$, activation quarter",
            "only the total count is printed",
            f"{U}:284-295",
            "SIZE term saturates at 0.10 via tanh; the truth term has "
            "coefficient 0.05, so $p^{true}$ moves the base rate by at most "
            "0.05 while a prior deployment adds 0.45 - nine times larger"),
        row("Commentary", "Mention count", "$M^{cnt}_{pn}$",
            "$|\\{q \\in [\\max(0,t_c-8),\\,t_c] : U_q < \\lambda(q)\\}|$, "
            "one Bernoulli per quarter, private stream "
            "default\\_rng([seed, 999, 10000\\,p+n])",
            "$\\lambda(q)$",
            "printed as `discussed by N sources' only when $N>0$; "
            "zero-mention pairings are simply absent",
            f"{U}:289-295; {R}:282-293",
            "9 quarters, so the count is bounded by 9"),

        # ---------------- observation layer ----------------
        row("Observation", "Capability-attribution drop rate", "drop\\_p",
            "0.15, i.i.d. per owned capability (and per required complement)",
            "-", "the claimed list only", f"{R}:51,84,101"),
        row("Observation", "Spurious-attribution rate", "false\\_p",
            "nominal 0.10, but applied as $false\\_p / n_{caps} \\times 3 = "
            "0.03$ per non-owned capability",
            "-", "the claimed list only", f"{R}:52,86",
            "expected spurious additions per product $= 0.03\\times 9 = 0.27$"),
        row("Observation", "Instrument noise", "attr\\_noise",
            "additive $\\mathcal{N}(0,0.18^2)$ per coordinate on capability "
            "AND need vectors, clipped at 0, then coordinates $<0.12$ set to 0",
            "-", "the printed profile", f"{R}:53,78-79,96-97",
            "the 0.12 floor deletes small true signal outright"),
        row("Observation", "Threshold noise", "-", "$\\mathcal{N}(0,0.04^2)$, "
            "clipped to $[0.2,0.9]$", "-", "printed threshold", f"{R}:99-100"),
        row("Observation", "Materiality noise", "-",
            "multiplicative $\\times(1+\\mathcal{N}(0,0.10^2))$, floored at 0.01",
            "-", "printed value if served", f"{R}:103-104"),
        row("Observation", "Company-factor noise", "-",
            "$\\mathcal{N}(0,0.08^2)$ on supply/distribution/moat, clipped "
            "to $[0,1]$", "-", "printed 0-1 scores", f"{R}:90-92"),
        row("Observation", "Complement-date misreport rate", "-",
            "0.12; the jitter $j\\sim\\mathcal{U}\\{-4,\\dots,4\\}$ is drawn "
            "unconditionally and applied as $\\max(0,\\rho_k+j)$ only when "
            "misreported AND dated",
            "-", "printed availability line", f"{R}:106-114",
            "a NEVER complement is never re-dated, but a real date can be "
            "pushed past $t_c+H$ and read as a blocker"),
        row("Observation", "Printed precision", "PRECISION",
            "2 decimals, applied to the STORED Observed values as well as the "
            "rendered text",
            "-", "every number in the dossier", f"{R}:18,73-74",
            "the structured references consume the same rounded values"),
        row("Observation", "Corpus sections", "-",
            "7 blocks: header, characterisations, teardowns, problem reports, "
            "readiness notes, already-deployed list, commentary",
            "-", "the whole dossier", f"{R}:227-295",
            "identical templates and statement counts across all 5 surfaces"),
        row("Task", "Probe set", "-",
            "candidates bucketed by (observed-blocked, mentioned$>$0); the "
            "first 4 of each cell in sorted order",
            "observed gate, mention counts",
            "the 16 pairings are listed verbatim in the prompt",
            f"{T}:29-47",
            "up to 16; fewer when a cell is empty"),
    ]


def _generator_audit(n_seeds: int = 100):
    """Empirical checks of the claims made in `_discrepancies`, over the 100
    default universes. Cheap, deterministic, and no model involved."""
    import numpy as np

    from ldb.universe import NEVER

    n_comp = n_never = n_late_true = n_late_obs = 0
    n_needs = n_blocked_true = n_blocked_obs = 0
    probe_sizes, n_cands = [], []
    for s in range(n_seeds):
        inst = build_instance(s)
        u, o, end = inst.universe, inst.observed, inst.cutoff + inst.horizon
        for kid, k in u.complements.items():
            n_comp += 1
            if k.ready_at == NEVER:
                n_never += 1
            elif k.ready_at > end:
                n_late_true += 1
            if o.comp_ready[kid] != NEVER and o.comp_ready[kid] > end:
                n_late_obs += 1
        for nid, n in u.needs.items():
            n_needs += 1
            if not all(u.complements[k].ready_at <= end for k in n.complement_ids):
                n_blocked_true += 1
            if not all(o.comp_ready[k] <= end for k in o.need_comps[nid]):
                n_blocked_obs += 1
        probe_sizes.append(len(inst.probes))
        n_cands.append(len(inst.candidates))
    return {
        "n_seeds": n_seeds,
        "complements_total": n_comp,
        "complements_never": n_never,
        "complements_dated_after_horizon_true": n_late_true,
        "complements_dated_after_horizon_observed": n_late_obs,
        "needs_total": n_needs,
        "needs_blocked_true": n_blocked_true,
        "needs_blocked_as_observed": n_blocked_obs,
        "probe_size_mean": float(np.mean(probe_sizes)),
        "probe_size_min": int(min(probe_sizes)),
        "probe_size_max": int(max(probe_sizes)),
        "probe_size_frac_equal_16": float(np.mean([p == 16 for p in probe_sizes])),
        "candidates_mean": float(np.mean(n_cands)),
    }


def _discrepancies():
    """Historical paper audit, with resolved review corrections marked explicitly."""
    return [
        {"id": "horizon_late_complements", "severity": "high",
         "paper_claim": "The gate is described as failing when a complement is "
                        "not available inside the horizon, and the system "
                        "prompt says `a complement with no announced "
                        "availability, OR ONE ARRIVING AFTER THE HORIZON, "
                        "blocks realization'.",
         "code_behaviour": "The date generator draws either "
                           "$\\mathcal{U}\\{0..t_c-1\\}$ (45\\%), "
                           "$\\mathcal{U}\\{t_c..t_c+H-1\\}$ (40\\%) or NEVER "
                           "(15\\%). Every dated draw is $\\le t_c+H-1$, so "
                           "under the default cutoff/horizon the gate can only "
                           "fail through NEVER. `Dated but after the horizon' "
                           "does not occur in the true universe; it appears "
                           "only in the OBSERVED corpus, via the 12\\% "
                           "misreport jitter of up to +4.",
         "source": "ldb/universe.py:156-164,212; ldb/render.py:110-114",
         "impact": "The blocking mechanism the paper and prompt describe as "
                   "two-sided is effectively one-sided. Any observed "
                   "`guided to a quarter after the horizon' is a "
                   "misreporting artefact, so a solver that trusts it is "
                   "penalised for following the stated rule."},
        {"id": "probe_omissions_penalised", "severity": "high",
         "status": "resolved: prose now states worst-case omission penalties; "
                   "the scorer also penalises an entirely empty probe set",
         "paper_claim": "Sec 3.5: probe coverage is reported `with the few "
                        "omissions excluded from the mean rather than "
                        "penalised'.",
         "code_behaviour": "elicited_metrics() divides by "
                           "total = len(kept) + n_missing and adds 1.0 of "
                           "error per missing probe to capture_mae, so "
                           "capture_mae, recognition_acc and prereq_f1 all "
                           "penalise omissions at worst case. Only "
                           "materiality_rho is computed on answered probes "
                           "alone. An entirely empty probe set is penalised "
                           "on the same three metrics.",
         "source": "ldb/paths.py:elicited_metrics; ldb/llm.py:score_response",
         "impact": "The paper understates the penalty a model pays for an "
                   "incomplete probe list."},
        {"id": "false_attribution_rate", "severity": "medium",
         "paper_claim": "Sec 3.3 says capability attributions are `dropped and "
                        "spuriously added', with no rate given; the parameter "
                        "is named false_p=0.10.",
         "code_behaviour": "The per-capability spurious-add probability is "
                           "false_p / n_caps * 3 = 0.10/10*3 = 0.03, giving "
                           "0.27 expected false attributions per product, not "
                           "0.10 of anything.",
         "source": "ldb/render.py:86",
         "impact": "The nominal parameter name overstates the noise by more "
                   "than 3x per capability; anyone reading false_p=0.10 as the "
                   "spurious rate will mis-model the observation channel."},
        {"id": "drop_fallback", "severity": "medium",
         "paper_claim": "Attributions and complement requirements are dropped "
                        "i.i.d. at 15\\%.",
         "code_behaviour": "If dropping empties a list, the first element is "
                           "restored (`claimed or list(...)[:1]'), so the drop "
                           "process is not i.i.d. for single-capability "
                           "products or single-complement needs - exactly the "
                           "cases where a drop would matter most.",
         "source": "ldb/render.py:88,102",
         "impact": "Understates the reliability of single-item claims and "
                   "makes the effective drop rate depend on list length."},
        {"id": "capture_form", "severity": "medium",
         "paper_claim": "Sec 3.2: value capture is `drawn separately from the "
                        "company's supply, distribution and defensibility'.",
         "code_behaviour": "$p^{cap} = \\mathrm{clip}(0.15 + 0.55\\,u\\,d + "
                           "0.30\\,m, 0, 1)$: supply and distribution enter as "
                           "a PRODUCT, not additively, and the intercept is "
                           "0.15. It is a per-product constant, identical "
                           "across all needs, so capture-AUC has at most 12 "
                           "distinct score levels per universe.",
         "source": "ldb/universe.py:249-250",
         "impact": "The multiplicative u*d term and the per-product constancy "
                   "are not implied by the prose."},
        {"id": "commentary_weights", "severity": "medium",
         "paper_claim": "Sec 3.2: commentary depends on size, on prior "
                        "deployment, and `only weakly' on $p^{*}$.",
         "code_behaviour": "base = 0.02 + 0.10*tanh(size) + 0.05*p_true, plus "
                           "0.45 if already deployed, capped at 0.9. The truth "
                           "term uses p_true, NOT the $p^{*}$ the paper names, "
                           "its coefficient is 0.05 (max 5\\% per quarter over "
                           "9 quarters), the size term saturates at 0.10 via "
                           "tanh, and the deployment term is 0.45 - 9x the "
                           "truth term.",
         "source": "ldb/universe.py:286-294",
         "impact": "`Weak' is quantitatively 0.05 vs 0.45; and the term is "
                   "p_true, not the conditional $p^{*}$."},
        {"id": "attr_zero_floor", "severity": "low",
         "paper_claim": "Sec 3.3: `attribute readings carry instrument error'.",
         "code_behaviour": "After adding $\\mathcal{N}(0,0.18^2)$ and clipping "
                           "at 0, any coordinate below 0.12 is set to exactly "
                           "0. This is censoring, not additive error: a true "
                           "attribute near the 0.4 lower bound can vanish from "
                           "the dossier entirely.",
         "source": "ldb/render.py:78-79,96-97",
         "impact": "The observation channel deletes signal as well as blurring "
                   "it, which the prose does not convey."},
        {"id": "size_never_rendered", "severity": "low",
         "paper_claim": "Sec 3.1: recognition is `generated from firm size'; "
                        "firm size is presented as an economic quantity.",
         "code_behaviour": "Product.company_size is lognormal(0,1) and is "
                           "never rendered into any surface. It is observable "
                           "to the solver only through its effect on mention "
                           "counts.",
         "source": "ldb/universe.py:183; ldb/render.py:238-246",
         "impact": "A solver cannot de-bias commentary by size, because size "
                   "is not disclosed."},
        {"id": "probe_count_16", "severity": "low",
         "paper_claim": "Sec 3.4: `a probe set of sixteen pairings fixed in "
                        "advance'.",
         "code_behaviour": "_probe_set takes up to 4 per (blocked, mentioned) "
                           "cell across 4 cells, so 16 is an upper bound that "
                           "holds only when all four cells are populated with "
                           "at least 4 candidates. The realised counts are "
                           "recorded in the n_probes row of the "
                           "output-contract table.",
         "source": "ldb/task.py:29-47"},
        {"id": "materiality_unrendered_note", "severity": "low",
         "status": "resolved: separate materiality and capture ranking objectives",
         "paper_claim": "Sec 3.5 reports a materiality-weighted nDCG$_{EV}$.",
         "code_behaviour": "ndcg_at_k is explicitly UNWEIGHTED (gain 1 for a "
                           "realised pair); its docstring records that "
                           "materiality weighting was removed. The "
                           "materiality-weighted objective survives only as "
                           "ndcg_ev@10, scored on the model's probability x "
                           "materiality order. Capture-AUC separately uses "
                           "probability x capture. Both orders are restricted "
                           "to the validated top-K shortlist.",
         "source": "ldb/metrics.py:evaluate; ldb/llm.py:value_rankings"},
    ]


def main():
    code_hash_at_start = _code_hash()

    arms, contract_rows, mismatches = [], [], []
    for arm in ARMS:
        meta, rows, mm = _arm_rows(arm)
        arms.append(meta)
        contract_rows.extend(rows)
        mismatches.extend(mm)

    payload = {
        "config": {
            "code_hash": code_hash_at_start,
            "generator": "tools/spec_tables.py",
            "bootstrap": {"n_boot": 10000, "seed": 0, "alpha": 0.05,
                          "resampling_unit": "universe",
                          "function": "ldb.metrics.bootstrap_ci"},
            "note": "Derived entirely from on-disk artifacts plus a "
                    "re-derivation of each instance via ldb.task.build_instance. "
                    "No model was queried.",
        },
        "output_contract": {
            "caption": "Output-contract accounting per model endpoint, "
                       "finance surface, recomputed from archived raw "
                       "responses.",
            "field_semantics": {
                "n_ranked": "artifact field: len(kept) in "
                            "ldb.llm.score_predictions - unique, in-candidate "
                            "entries surviving the top-K cap; the entries that "
                            "actually receive a rank score.",
                "n_returned": "artifact field: len(preds), the full length of "
                              "raw_response['predictions'] before any "
                              "filtering.",
                "probe_coverage": "artifact field: answered / len(inst.probes), "
                                  "where an answer counts only if its "
                                  "(product_id, need_id) is one of the fixed "
                                  "probe pairings.",
            },
            "arms": arms,
            "metrics": METRICS,
            "rows": contract_rows,
            "stored_vs_recomputed_mismatches": mismatches,
        },
        "generator_spec": {
            "caption": "Default generator and observation-layer specification, "
                       "as coded.",
            "fields": ["group", "quantity", "symbol", "distribution_or_formula",
                       "parents", "solver_observes", "source", "notes"],
            "rows": _spec_rows(),
        },
        "code_vs_paper": {
            "caption": "Historical code-versus-paper audit, not a list of "
                       "current findings. Original paper_claim text is retained; "
                       "review-related resolutions are marked in status.",
            "fields": ["id", "severity", "paper_claim", "code_behaviour",
                       "source", "impact"],
            "rows": _discrepancies(),
            "empirical_audit": _generator_audit(),
        },
    }

    if _code_hash() != code_hash_at_start:
        raise SystemExit(
            f"source changed during the run ({code_hash_at_start} -> "
            f"{_code_hash()}); refusing to write an artifact that would "
            "certify code which did not produce it.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"wrote {OUT}  (code_hash {code_hash_at_start})")
    if mismatches:
        print(f"!! {len(mismatches)} stored/recomputed mismatches")
    return payload


if __name__ == "__main__":
    main()
