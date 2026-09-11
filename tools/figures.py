"""Build the paper's two data figures straight from the result artifacts.

Same rule as the tables: nothing is transcribed. Both figures answer an
objection rather than decorate a claim: one bounds the feature-engineered supervised
reference's training advantage, the other reports a coupled problem-size stress
test. Arms that ran on a single configuration appear as an
unconnected marker, never a line.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 200,
})
STYLE = {"luna": ("#1f77b4", "o"), "terra": ("#d62728", "s")}
EXTRA = ("#7f7f7f", "D")   # arms run on one configuration only


def _log_axis(ax, ticks, labels):
    """Log x-axis showing only the ticks we asked for.

    Matplotlib's minor formatter relabels a log axis with 1.2 x 10^2 style
    entries, which collided with the real ticks in both figures.
    """
    from matplotlib.ticker import NullFormatter, NullLocator
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def load(p):
    from run_v0 import _code_hash
    from tools.paper_tables import check

    with open(ROOT / p) as f:
        art = json.load(f)
    check(str(p), art, _code_hash())
    return art


def model_key(art):
    base = {"random", "recognition", "structured_evidence", "fair_reference",
            "expected_value", "hidden_oracle"}
    return [k for k in art["results"] if k not in base][0]


def short(model):
    return "".join(c for c in model.split("-")[-1] if c.isalpha())


def train_curve(out: Path, arms: list[str], extra: list[str] | None = None):
    """Paired advantage of the reference over the strongest model, by training size.

    An earlier version plotted the reference's absolute AUC over 100 evaluation
    universes against horizontal model lines measured on 30, so the crossings
    looked like paired comparisons and were not. This plots the paired
    difference directly, on the 30 universes the models actually ran, with zero
    as the only reference line that matters.
    """
    r = load("results/train_resample.json")["per_n"]
    ms = load("results/train_resample.json")["model_scores"]
    xs = sorted((int(k) for k in r), key=int)
    tag = "terra"
    med, lo, hi = [], [], []
    for x in xs:
        q = r[str(x)]["matched"]["paired"][tag]
        med.append(q["median"])
        lo.append(q["p5"])
        hi.append(q["p95"])

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.fill_between(xs, lo, hi, color="#4c72b0", alpha=0.15, linewidth=0,
                    label="5th to 95th pct. over training sets")
    ax.plot(xs, med, color="#4c72b0", marker="o", markersize=3.5,
            label="median paired advantage")
    name = ms[tag]["key"].split(":")[-1].split("@")[0]
    ax.text(xs[-1], 0.004, f"parity with {name}", ha="right", va="bottom",
            fontsize=6.5, color="#333333")
    _log_axis(ax, xs, [str(x) for x in xs])
    ax.set_xlabel("labelled training universes")
    ax.set_ylabel("paired $\\Delta$ AUC$_K$")
    ax.legend(loc="lower right", fontsize=6.0)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


SIZES = [("size_small", 8, 10), ("surface_finance", 12, 15),
         ("size_large", 18, 22)]


def size_curve(out: Path, arms: list[str], extra: list[str] | None = None):
    from ldb.task import build_instance
    xs = []
    for _, npr, nnd in SIZES:
        # averaged over the universes the model arms actually ran on, so the
        # Candidate count is one of several co-varying quantities in this stress test.
        n = [len(build_instance(s, n_products=npr, n_needs=nnd).candidates)
             for s in range(30)]
        xs.append(sum(n) / len(n))

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ref = [load(f"{arms[0]}/{k}.json")["results"]["fair_reference"]["auc"]["mean"]
           for k, _, _ in SIZES]
    ax.plot(xs, ref, color="#4c72b0", marker="^", markersize=3.5,
            label="feature-eng. supervised")
    for d in arms:
        ys, name = [], None
        for k, _, _ in SIZES:
            art = load(f"{d}/{k}.json")
            name = art["config"]["backend"]["model"]
            ys.append(art["results"][model_key(art)]["auc"]["mean"])
        colour, marker = STYLE.get(short(name), ("#555555", "o"))
        ax.plot(xs, ys, color=colour, marker=marker, markersize=3.5, label=name)
    # arms that ran at one economy size only: a marker, never a line, because a
    # single point says nothing about the slope this figure is about
    for f in (extra or []):
        art = load(f)
        name = art["config"]["backend"]["model"].split("/")[-1]
        ax.plot([xs[1]], [art["results"][model_key(art)]["auc"]["mean"]],
                color=EXTRA[0], marker=EXTRA[1], markersize=4, linestyle="none",
                label=f"{name} (one size)")
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":")
    ax.text(xs[-1], 0.505, "chance", fontsize=6.5, color="#777777", ha="right")
    _log_axis(ax, xs, [f"{x:.0f}" for x in xs])
    ax.set_ylim(0.48, 0.80)
    ax.set_xlabel("candidate pairings per universe")
    ax.set_ylabel("AUC$_K$")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["results/luna_full", "results/terra_full"])
    ap.add_argument("--extra-arms", nargs="*",
                    default=["results/openweight/model_finance.json"],
                    help="artifacts for models run on one configuration only")
    ap.add_argument("--outdir", default="reproduced")
    args = ap.parse_args()
    out = ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    train_curve(out / "fig_train_curve.pdf", args.arms, args.extra_arms)
    size_curve(out / "fig_size.pdf", args.arms, args.extra_arms)


if __name__ == "__main__":
    main()
