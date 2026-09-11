"""Generate and score the human baseline.

Why this exists: every number in the paper compares models to other programs.
Nothing yet shows the task is solvable by reasoning at all, which is the first
thing a reviewer asks of a synthetic benchmark. A human baseline answers it, and
it is the one remaining result that cannot be bought.

Design constraint: nobody will rank 175 pairs. The fixed probe set is the
human-comparable task - 16 pairs, chosen before any respondent sees the dossier,
stratified by (blocked, mentioned). Models already answer exactly these, and
baselines are re-scored on them via `probe_auc`, so all three are comparable.

    python tools/human_task.py make --seeds 0 1 2 --out human/
    # respondents fill in the CSV next to each dossier
    python tools/human_task.py score --dir human/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ldb.baselines import fair_reference, hidden_oracle, recognition_rank
from ldb.metrics import probe_auc
from ldb.paths import elicited_metrics
from ldb.task import build_instance

INSTRUCTIONS = """\
HOW TO DO THIS TASK

You are reading a dossier about a fictional economy, as it stood at the cutoff.
Your job: for each of the 16 pairings listed in answers.csv, judge whether that
product will be commercially deployed against that need within the next 8
quarters, and answer four further questions about it.

The dossier never tells you the answer. You have to work it out:

  a product -> the capabilities it contains -> the attribute profile of those
  capabilities, matched against
  a need -> its bottleneck -> the attribute profile that would relieve it

A pairing is technically viable when the product's combined attribute profile
lines up with what the need requires, above the sufficiency threshold stated for
that need. But viability is not enough: a need also cannot be served until every
complement it depends on is available. A complement with no announced
availability, or one arriving after the horizon, blocks the pairing however good
the profile match.

Commentary mentions some pairings. That reflects attention already paid, mostly
driven by how large the holder is and what has already happened - not evidence
about the future. Do not rank by how much something is discussed.

FILL IN answers.csv:
  probability    0-1, that it is realized within the horizon
  materiality    the value if served, on the scale the dossier uses
  capture        0-1, that the holder keeps that value rather than losing it
                 to entrants (supply, distribution, defensibility)
  recognized     yes/no, is this pairing ALREADY discussed in commentary
  prerequisites  ids of complements still blocking it (e.g. K3;K5), blank if none
  evidence_path  one line: which capability and which requirement you matched

There is no time limit. Please answer all 16 - skipping the hard ones makes the
result unusable. Do not use a search engine or an AI assistant.
"""

FIELDS = ["product_id", "need_id", "probability", "materiality", "capture",
          "recognized", "prerequisites", "evidence_path"]


def make(seeds, outdir: Path, surface: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        inst = build_instance(s, surface=surface)
        d = outdir / f"universe_{s:03d}"
        d.mkdir(exist_ok=True)
        (d / "instructions.txt").write_text(INSTRUCTIONS)
        (d / "dossier.txt").write_text(inst.corpus)
        with open(d / "answers.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for pid, nid in inst.probes:
                w.writerow({"product_id": pid, "need_id": nid})
        print(f"  {d}/  ({len(inst.probes)} probes, "
              f"{len(inst.corpus)} chars of dossier)")


def _read(path: Path) -> list[dict]:
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("probability", "").strip()]
    out = []
    for r in rows:
        pre = [x.strip().upper() for x in (r.get("prerequisites") or "").split(";")]
        out.append({
            "product_id": r["product_id"].strip(),
            "need_id": r["need_id"].strip(),
            "probability": float(r["probability"]),
            "materiality": float(r["materiality"] or 0),
            "capture": float(r["capture"] or 0),
            "recognized": str(r.get("recognized", "")).strip().lower()
                          in ("y", "yes", "true", "1"),
            "prerequisites": [x for x in pre if x],
            "evidence_path": r.get("evidence_path", ""),
        })
    return out


def score(d: Path, surface: str):
    rows, per_seed = [], []
    for sub in sorted(d.glob("universe_*")):
        seed = int(sub.name.split("_")[1])
        answers = _read(sub / "answers.csv")
        inst = build_instance(seed, surface=surface)
        if not answers:
            print(f"  {sub.name}: no completed rows, skipped")
            continue
        got = {(a["product_id"], a["need_id"]): a for a in answers}
        missing = [k for k in inst.probes if k not in got]
        # unanswered probes are scored, not skipped - same rule models face
        scores = {k: got[k]["probability"] if k in got else 0.0
                  for k in inst.candidates}
        m = elicited_metrics(inst, [(k, got[k]) for k in inst.probes if k in got],
                             n_missing=len(missing))
        m["probe_auc"] = probe_auc(inst, scores)
        m["coverage"] = 1 - len(missing) / len(inst.probes)
        per_seed.append((seed, m))
        rows.append(m)
        print(f"  {sub.name}: probe_auc {m['probe_auc']:.3f} "
              f"coverage {m['coverage']:.2f}")

    if not rows:
        raise SystemExit("no completed responses found")

    print(f"\nhuman (n={len(rows)} universes)")
    keys = ["probe_auc", "materiality_rho", "capture_mae", "recognition_acc",
            "prereq_f1"]
    human = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    for k in keys:
        print(f"  {k:18s} {human[k]:.3f}")

    print("\nsame universes, same 16 probes:")
    seeds = [s for s, _ in per_seed]
    for name, fn in (("recognition", recognition_rank),
                     ("fair_reference", fair_reference),
                     ("hidden_oracle", hidden_oracle)):
        v = [probe_auc(i := build_instance(s, surface=surface), fn(i)) for s in seeds]
        print(f"  {name:18s} probe_auc {np.nanmean(v):.3f}")
    print("  (model rows: read probe_auc from that arm's artifact)")

    (d / "human_scores.json").write_text(json.dumps(
        {"per_seed": {str(s): m for s, m in per_seed}, "mean": human,
         "n_universes": len(rows)}, indent=2, allow_nan=False))
    print(f"\nwrote {d}/human_scores.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dir", default="human")
    ap.add_argument("--out", default="human")
    ap.add_argument("--surface", default="finance")
    args = ap.parse_args()
    if args.cmd == "make":
        make(args.seeds, Path(args.out), args.surface)
    else:
        score(Path(args.dir), args.surface)


if __name__ == "__main__":
    main()
