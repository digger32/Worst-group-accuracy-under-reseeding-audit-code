#!/usr/bin/env python3
"""Figures for the audit.

Conventions applied throughout, because a reader compares figures with tables and
inconsistency between them reads as carelessness:

  * methods appear in the manuscript's order, baseline first, not alphabetically;
  * datasets appear in the manuscript's order, Waterbirds first;
  * legends sit outside the axes -- inside they covered the very bars that carry the
    result, which is what an earlier version did;
  * every quantity is computed the same way the tables compute it. In particular
    $P(A>B)$ is the PAIRED estimate over seeds, matching the table; an earlier
    version plotted the all-pairs variant and disagreed with its own table by eight
    percentage points.

Greyscale-legible, colourblind-safe, vector output, fonts matched to the IEEE
two-column body text, and no PDF metadata (the backend otherwise records the
toolchain and a time-zoned creation date, which under blind review narrows the
authors geographically).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# All figure text is 8 pt Times (Times New Roman, else the metric-compatible Liberation
# Serif), embedded as TrueType. Figures are drawn at their PRINTED size -- 3.5 in for
# one column, 7.16 in for two -- and included without scaling, so 8 pt stays 8 pt.
plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "Times"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Serif",
                     "mathtext.it": "Liberation Serif:italic",
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
                     "pdf.fonttype": 42})
COL_W, TEXT_W = 3.45, 7.1   # inches, a hair under IEEE column and text widths
PDF_META = {"Creator": "", "Producer": "", "CreationDate": None}

M = "acc_worst_group"
ORDER = ["erm", "rw", "gdro", "adv", "dfr"]
LABEL = {"erm": "ERM", "rw": "RW", "gdro": "group-DRO", "adv": "Adv", "dfr": "DFR"}
DS_ORDER = ["waterbirds", "celeba"]
DS_LABEL = {"waterbirds": "Waterbirds", "celeba": "CelebA"}
MARKERS = ["o", "s", "^", "D", "v"]
DARK, LIGHT = "0.25", "0.72"


def _order(df, key, ref):
    present = [x for x in ref if x in set(df[key])]
    return present


def fig_seed_spread(df, outdir):
    """Every run as a point. This is the paper's subject in one view."""
    ds_list = _order(df, "dataset", DS_ORDER)
    fig, axes = plt.subplots(1, len(ds_list), figsize=(COL_W, 2.0), squeeze=False)
    for j, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        d = df[df["dataset"] == ds]
        methods = _order(d, "method", ORDER)
        for i, m in enumerate(methods):
            v = d[d["method"] == m][M].to_numpy()
            jitter = np.random.default_rng(i).normal(0, .055, len(v))
            ax.scatter(np.full_like(v, i, dtype=float) + jitter, v, s=6,
                       marker=MARKERS[i % len(MARKERS)], facecolors="none",
                       edgecolors="k", linewidths=.6)
            ax.hlines(v.mean(), i - .3, i + .3, colors="k", linewidth=1.6)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABEL[m] for m in methods], rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_title(DS_LABEL[ds], fontsize=8)
        ax.set_xlim(-.6, len(methods) - .4)
        if j == 0:
            ax.set_ylabel("Worst-group accuracy")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(Path(outdir) / "fig_seed_spread.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_win_rate(df, outdir):
    """P(A>B), paired over seeds, with both reference levels drawn."""
    ds_list = _order(df, "dataset", DS_ORDER)
    methods = [m for m in ORDER if m != "erm"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(methods))
    for k, ds in enumerate(ds_list):
        piv = df[df["dataset"] == ds].pivot_table(index="seed", columns="method",
                                                  values=M)
        wins = [float(np.mean(piv[m].to_numpy() > piv["erm"].to_numpy()))
                for m in methods]
        ax.bar(x + (k - .5) * .38, wins, .34, label=DS_LABEL[ds],
               color=DARK if k == 0 else LIGHT, edgecolor="k", linewidth=.5)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=.9)
    ax.axhline(0.75, color="k", linestyle="--", linewidth=.9)
    ax.set_xlim(-.75, len(methods) - .25)
    ax.text(-.68, 0.775, "decision threshold", fontsize=8, ha="left", va="bottom")
    ax.text(-.68, 0.525, "coin flip", fontsize=8, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in methods])
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_ylabel("$P(A>B)$ vs ERM")
    # legend above the axes: inside, it covered the bars it describes
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.005))
    fig.savefig(Path(outdir) / "fig_win_rate.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_rank_churn(df, outdir):
    """Rank of each method on each individual seed.

    This replaces a critical-difference diagram. That diagram rests on an omnibus
    rank test whose blocks are supposed to be datasets; substituting correlated
    seeds inflates its power, so the manuscript declines to use it. The churn plot
    makes the same point descriptively and without an invalid test: where lines cross
    repeatedly, a single-run ranking is a coin flip.
    """
    ds_list = _order(df, "dataset", DS_ORDER)
    fig, axes = plt.subplots(1, len(ds_list), figsize=(TEXT_W, 2.3), squeeze=False)
    for j, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        piv = df[df["dataset"] == ds].pivot_table(index="seed", columns="method",
                                                  values=M)
        methods = _order(df, "method", ORDER)
        ranks = piv[methods].rank(axis=1, ascending=False)
        for i, m in enumerate(methods):
            ax.plot(ranks.index, ranks[m], marker=MARKERS[i % len(MARKERS)],
                    markersize=3.2, linewidth=.9,
                    color=["k", "0.35", "0.55", "0.15", "0.7"][i % 5],
                    markerfacecolor="none", label=LABEL[m])
        ax.set_yticks(range(1, len(methods) + 1))
        ax.invert_yaxis()
        ax.set_xticks(range(0, len(ranks.index), 4))
        ax.set_xlabel("Seed")
        ax.set_title(DS_LABEL[ds], fontsize=8)
        if j == 0:
            ax.set_ylabel("Rank (1 = best)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels),
               loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=2.0)
    fig.savefig(Path(outdir) / "fig_rank_churn.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_deltas(post, outdir):
    """Effect against ERM with its bootstrap interval; the zero line is the claim."""
    ds_list = [d for d in DS_ORDER if d in post["per_dataset"]]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(7.0, 2.1), squeeze=False)
    for j, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        ent = post["per_dataset"][ds]["methods"]
        ms = [m for m in ORDER if m in ent and "delta_vs_erm" in ent[m]]
        for i, m in enumerate(ms):
            lo, hi = ent[m]["delta_ci95"]
            ax.plot([lo, hi], [i, i], color="k", linewidth=1.2)
            ax.plot(ent[m]["delta_vs_erm"], i, marker=MARKERS[i % len(MARKERS)],
                    color="k", markersize=4.5, markerfacecolor="none")
        ax.axvline(0, color="k", linestyle="--", linewidth=.9)
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([LABEL[m] for m in ms])
        ax.set_ylim(-.6, len(ms) - .4)
        ax.set_xlabel("worst-group accuracy vs ERM")
        ax.set_title(DS_LABEL[ds], fontsize=8)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(Path(outdir) / "fig_delta_ci.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_ranking_survival(df, post, outdir):
    """How often the averaged ordering is recovered, from one seed and by bootstrap."""
    ds_list = [d for d in DS_ORDER if d in post["per_dataset"]]
    single, boot = [], []
    for ds in ds_list:
        piv = df[df["dataset"] == ds].pivot_table(index="seed", columns="method",
                                                  values=M)
        full = tuple(piv.mean().sort_values(ascending=False).index)
        orders = [tuple(piv.loc[s].sort_values(ascending=False).index)
                  for s in piv.index]
        single.append(sum(o == full for o in orders) / len(orders))
        boot.append(post["per_dataset"][ds]["ranking"]["exact_order_survival"])
    x = np.arange(len(ds_list))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.bar(x - .18, single, .34, color=DARK, edgecolor="k", linewidth=.5,
           label="from one seed")
    ax.bar(x + .18, boot, .34, color=LIGHT, edgecolor="k", linewidth=.5,
           label="bootstrap over seeds")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in ds_list])
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_ylabel("ordering recovered")
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.005))
    fig.savefig(Path(outdir) / "fig_ranking_survival.pdf", metadata=PDF_META)
    plt.close(fig)


def main(outdir):
    outdir = Path(outdir)
    fdir = outdir / "figures"
    fdir.mkdir(exist_ok=True)
    df = pd.read_csv(outdir / "results.csv")
    post = json.loads((outdir / "stats" / "posthoc.json").read_text())
    fig_seed_spread(df, fdir)
    fig_win_rate(df, fdir)
    fig_rank_churn(df, fdir)
    fig_deltas(post, fdir)
    fig_ranking_survival(df, post, fdir)
    print(f"[figures] -> {fdir}")
    for p in sorted(fdir.glob("*.pdf")):
        print(f"           {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/full"))
