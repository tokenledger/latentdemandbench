"""A reference that reads the dossier text, not the generator's structures.

The structured supervised reference is handed observables the harness computed;
a language model is handed prose. That asymmetry is the fairest objection to the
comparison, and this closes it: a deterministic parser recovers the same fifteen
quantities from the rendered corpus alone, using nothing but the bracket tags and
the sentence templates, and the identical logistic model is fitted on top.

Two numbers come out. Extraction fidelity says how much of the dossier the
parser recovers (it should be near-perfect, because the corpus is templated), and
end-to-end AUC says what a pipeline of parse-then-rank achieves reading the same
text a model reads. A model below this line is not losing to privileged inputs.

Model-free, so it costs nothing to re-run.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from ldb.fair import features, fit
from ldb.llm import TOP_K, apply_top_k_protocol
from ldb.metrics import bootstrap_ci, evaluate
from ldb.render import ATTR, PRECISION
from ldb.task import build_instance
from ldb.universe import D, NEVER

ATTR_INDEX = {a: i for i, a in enumerate(ATTR)}
# the opaque surface prints A0..A7 instead of attribute words
ATTR_INDEX.update({f"A{i}": i for i in range(D)})

TRAIN_BASE = 1000


@dataclass
class ParsedObserved:
    """Same field names as ldb.render.Observed, recovered from the text."""
    cutoff: int
    horizon: int
    prod_caps: dict = field(default_factory=dict)
    cap_vecs: dict = field(default_factory=dict)
    need_reqs: dict = field(default_factory=dict)
    need_thresholds: dict = field(default_factory=dict)
    need_materiality: dict = field(default_factory=dict)
    prod_supply: dict = field(default_factory=dict)
    prod_distribution: dict = field(default_factory=dict)
    prod_moat: dict = field(default_factory=dict)
    need_comps: dict = field(default_factory=dict)
    comp_ready: dict = field(default_factory=dict)
    mention_counts: dict = field(default_factory=dict)


@dataclass
class ParsedInstance:
    """Duck-types the parts of ldb.task.Instance that the features touch."""
    observed: ParsedObserved
    candidates: list
    label: dict


def _vec(text: str) -> np.ndarray:
    """`thermal 0.39, coherence 0.26, ...` -> the 8-vector it prints."""
    v = np.zeros(D)
    if "no measurable profile" in text:
        return v
    for name, val in re.findall(r"([A-Za-z]\w*)\s+(\d+\.\d+)", text):
        if name in ATTR_INDEX:
            v[ATTR_INDEX[name]] = float(val)
    return v


def parse(corpus: str, horizon: int) -> ParsedObserved:
    """Recover the structured view from the rendered documents.

    Names are resolved through the bracket tags rather than the prose names, the
    same handles the prompt tells a solver to use. Nothing here consults the
    universe: if a fact was dropped or misreported by the observation layer, the
    parser inherits the error, which is the point.
    """
    head = re.search(r"===\s+.*?,\s+\w+\s+(\d+)\s+===", corpus)
    o = ParsedObserved(cutoff=int(head.group(1)), horizon=horizon)

    name_to_cap, name_to_comp = {}, {}

    for cid, rest in re.findall(r"\[TP-(C\d+)\] Characterisation of (.+)", corpus):
        name = rest.split(". Measured attribute profile:")[0].strip()
        prof = rest.split("Measured attribute profile:")[1].split(". Observed")[0]
        o.cap_vecs[cid] = _vec(prof)
        name_to_cap[name] = cid
        name_to_cap[f"item {cid}"] = cid

    for pid, rest in re.findall(r"\[PT-(P\d+)\] (.+)", corpus):
        caps = rest.split("capability:")[1].split(". ")[0]
        o.prod_caps[pid] = [name_to_cap[c.strip()] for c in caps.split(",")
                            if c.strip() in name_to_cap]
        for label, dst in (("supply capacity", o.prod_supply),
                           ("distribution reach", o.prod_distribution),
                           ("defensibility", o.prod_moat)):
            dst[pid] = float(re.search(rf"{label} (\d+\.\d+)", rest).group(1))

    for kid, rest in re.findall(r"\[SR-(K\d+)\] (.+)", corpus):
        name = rest.split(":")[0].strip()
        name_to_comp[name] = kid
        name_to_comp[f"item {kid}"] = kid
        when = re.search(r"(?:since|to) \w+ (\d+)", rest)
        o.comp_ready[kid] = int(when.group(1)) if when else NEVER

    for nid, rest in re.findall(r"\[PR-(N\d+)\] (.+)", corpus):
        req = rest.split("attribute profile of")[1].split(", with a sufficiency")[0]
        o.need_reqs[nid] = _vec(req)
        o.need_thresholds[nid] = float(
            re.search(r"threshold near (\d+\.\d+)", rest).group(1))
        o.need_materiality[nid] = float(
            re.search(r"if served: (\d+\.\d+)", rest).group(1))
        comps = rest.split("depends on:")[1].split(".")[0]
        o.need_comps[nid] = [name_to_comp[c.strip()] for c in comps.split(",")
                             if c.strip() in name_to_comp]

    # commentary volume, and the deployments that are disclosed as history
    counts, realized = {}, set()
    prod_by_name = {}
    for pid, rest in re.findall(r"\[PT-(P\d+)\] (.+)", corpus):
        prod_by_name[rest.split(".")[0].split(" (")[0].strip()] = pid
        prod_by_name[f"item {pid} (holder {pid}H)"] = pid
    need_by_name = {}
    for nid, rest in re.findall(r"\[PR-(N\d+)\] (.+)", corpus):
        need_by_name[rest.split(".")[0].strip()] = nid
        need_by_name[f"item {nid}"] = nid

    for line in corpus.splitlines():
        m = re.match(r"\[FN\] (.+?) is discussed by (\d+) .+? in connection with (.+)\.",
                     line)
        if m:
            pid = prod_by_name.get(m.group(1).strip())
            nid = need_by_name.get(m.group(3).strip())
            if pid and nid:
                counts[(pid, nid)] = int(m.group(2))
        r = re.match(r"\[RZ\] (.+?) is already deployed against (.+?);", line)
        if r:
            pid = prod_by_name.get(r.group(1).strip())
            nid = need_by_name.get(r.group(2).strip())
            if pid and nid:
                realized.add((pid, nid))

    o.mention_counts = {(p, n): counts.get((p, n), 0)
                        for p in o.prod_caps for n in o.need_reqs}
    o._realized = realized  # candidates are everything not disclosed as history
    return o


def parsed_instance(inst) -> ParsedInstance:
    o = parse(inst.corpus, inst.horizon)
    cands = [(p, n) for p in o.prod_caps for n in o.need_reqs
             if (p, n) not in o._realized]
    return ParsedInstance(o, cands, {k: inst.label.get(k, 0) for k in cands})


def fidelity(inst, o: ParsedObserved) -> dict:
    """How much of the harness-side observation the parser recovers."""
    t = inst.observed
    out = {}
    out["capability profiles"] = float(np.mean([
        np.allclose(o.cap_vecs[c], np.round(t.cap_vecs[c], PRECISION), atol=5e-3)
        for c in t.cap_vecs]))
    out["product attributions"] = float(np.mean([
        sorted(o.prod_caps[p]) == sorted(t.prod_caps[p]) for p in t.prod_caps]))
    out["need requirements"] = float(np.mean([
        np.allclose(o.need_reqs[n], np.round(t.need_reqs[n], PRECISION), atol=5e-3)
        for n in t.need_reqs]))
    out["thresholds"] = float(np.mean([
        abs(o.need_thresholds[n] - t.need_thresholds[n]) < 5e-3 for n in t.need_reqs]))
    out["materiality"] = float(np.mean([
        abs(o.need_materiality[n] - t.need_materiality[n]) < 5e-3 for n in t.need_reqs]))
    out["complement dates"] = float(np.mean([
        o.comp_ready[k] == t.comp_ready[k] for k in t.comp_ready]))
    out["required complements"] = float(np.mean([
        sorted(o.need_comps[n]) == sorted(t.need_comps[n]) for n in t.need_reqs]))
    out["mention counts"] = float(np.mean([
        o.mention_counts.get(k, 0) == v for k, v in t.mention_counts.items()]))
    out["candidate set"] = float(
        set(parsed_instance(inst).candidates) == set(inst.candidates))
    return out


def run(n_eval: int, n_train: int, top_k: int, surface: str) -> dict:
    train = [parsed_instance(build_instance(TRAIN_BASE + i, surface=surface))
             for i in range(n_train)]
    ref = fit(train)

    rows, fids = [], []
    for i in range(n_eval):
        inst = build_instance(i, surface=surface)
        pi = parsed_instance(inst)
        fids.append(fidelity(inst, pi.observed))
        raw = {k: float(1 / (1 + np.exp(-(features(pi, *k) @ ref.w))))
               for k in pi.candidates}
        # every ranker is held to the model's top-K contract
        sc, pr = apply_top_k_protocol(raw, raw, top_k)
        rows.append(evaluate(inst, {k: sc.get(k, 0.0) for k in inst.candidates},
                             probs={k: pr.get(k, 0.02) for k in inst.candidates}))

    out = {"n_eval": n_eval, "n_train": n_train, "surface": surface,
           "extraction": {k: float(np.mean([f[k] for f in fids])) for k in fids[0]},
           "metrics": {}}
    for m in ("auc", "gap_auc", "recall@10", "ndcg@10", "prob_mse_vs_p*"):
        mean, lo, hi = bootstrap_ci([r[m] for r in rows])
        out["metrics"][m] = {"mean": mean, "lo": lo, "hi": hi, "n": len(rows)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-universes", type=int, default=100)
    ap.add_argument("--train-universes", type=int, default=100)
    ap.add_argument("--surface", default="finance")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--out", default=str(ROOT / "results" / "parser_reference.json"))
    args = ap.parse_args()

    from run_v0 import _code_hash
    code_hash = _code_hash()

    res = run(args.eval_universes, args.train_universes, args.top_k, args.surface)
    print(f"extraction fidelity ({args.surface}):")
    for k, v in res["extraction"].items():
        print(f"  {k:24s} {v:.4f}")
    print("end-to-end:")
    for k, v in res["metrics"].items():
        print(f"  {k:16s} {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")

    if _code_hash() != code_hash:
        raise SystemExit("source changed during the run; refusing to write")
    res["config"] = {"code_hash": code_hash, "top_k": args.top_k,
                     "train_base": TRAIN_BASE}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
