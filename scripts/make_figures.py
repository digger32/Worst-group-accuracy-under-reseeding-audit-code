#!/usr/bin/env python3
"""Figures for the audit. Each carries information the prose cannot: the seed
spread itself, the effect sizes with their intervals, how often the ranking
survives resampling, and the critical-difference position of every method.
Greyscale-legible, colourblind-safe, vector output, fonts matched to the IEEE
two-column body text.
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

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 200, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})

# Triple-blind: the PDF backend otherwise records the toolchain and a creation
# timestamp WITH A TIME ZONE, which narrows the authors geographically. Suppressing
# it also makes the figures byte-reproducible across machines.
PDF_META = {"Creator": "", "Producer": "", "CreationDate": None}
MARKERS = ["o", "s", "^", "D", "v", "P"]


def fig_seed_spread(df, outdir):
    """Every seed as a point, next to the interval a single-seed paper would report."""
    ds_list = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.4 * len(ds_list), 2.6), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        d = df[df["dataset"] == ds]
        methods = sorted(d["method"].unique())
        for i, m in enumerate(methods):
            v = d[d["method"] == m]["acc_worst_group"].to_numpy()
            ax.scatter(np.full_like(v, i, dtype=float) + np.random.default_rng(i).normal(0, .06, len(v)),
                       v, s=9, marker=MARKERS[i % len(MARKERS)], facecolors="none",
                       edgecolors="k", linewidths=.6)
            ax.hlines(v.mean(), i - .3, i + .3, colors="k", linewidth=1.4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_title(ds)
        ax.set_ylabel("worst-group accuracy")
    fig.savefig(Path(outdir) / "fig_seed_spread.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_deltas(posthoc, outdir):
    """Effect versus ERM with its bootstrap interval; the zero line is the claim."""
    ds_list = sorted(posthoc["per_dataset"])
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.4 * len(ds_list), 2.4), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        ent = posthoc["per_dataset"][ds]["methods"]
        ms = [m for m in ent if "delta_vs_erm" in ent[m]]
        y = np.arange(len(ms))
        for i, m in enumerate(ms):
            lo, hi = ent[m]["delta_ci95"]
            ax.plot([lo, hi], [i, i], color="k", linewidth=1.1)
            ax.plot(ent[m]["delta_vs_erm"], i, marker=MARKERS[i % len(MARKERS)],
                    color="k", markersize=4)
        ax.axvline(0, color="k", linestyle="--", linewidth=.8)
        ax.set_yticks(y); ax.set_yticklabels(ms)
        ax.set_xlabel("worst-group accuracy vs ERM")
        ax.set_title(ds)
    fig.savefig(Path(outdir) / "fig_delta_ci.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_cd(omnibus, outdir):
    """Critical-difference diagram over mean ranks."""
    if omnibus.get("status") == "degenerate":
        return
    ranks = omnibus["mean_ranks"]
    cd = omnibus["nemenyi_cd_0.05"]
    ms = sorted(ranks, key=ranks.get)
    fig, ax = plt.subplots(figsize=(4.6, 1.5 + .18 * len(ms)))
    lo, hi = min(ranks.values()) - .4, max(ranks.values()) + .4
    ax.hlines(0, lo, hi, colors="k", linewidth=1)
    for r in np.arange(np.floor(lo), np.ceil(hi) + .5, .5):
        ax.vlines(r, 0, .06, colors="k", linewidth=.7)
        ax.text(r, .12, f"{r:g}", ha="center", fontsize=6)
    for i, m in enumerate(ms):
        ax.plot([ranks[m], ranks[m]], [0, -.25 - .18 * i], color="k", linewidth=.7)
        ax.text(ranks[m], -.30 - .18 * i, f" {m} ({ranks[m]:.2f})", va="center", fontsize=7)
    ax.hlines(.45, lo, lo + cd, colors="k", linewidth=2)
    ax.text(lo + cd / 2, .55, f"CD = {cd:.2f}", ha="center", fontsize=7)
    ax.axis("off")
    fig.savefig(Path(outdir) / "fig_cd.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_ranking_survival(posthoc, outdir):
    """How often the observed ordering reappears when the seeds are resampled."""
    ds_list = sorted(posthoc["per_dataset"])
    exact = [posthoc["per_dataset"][d]["ranking"]["exact_order_survival"] for d in ds_list]
    top = [posthoc["per_dataset"][d]["ranking"]["winner_survival"] for d in ds_list]
    x = np.arange(len(ds_list))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(x - .18, exact, .34, color="0.25", label="full ordering")
    ax.bar(x + .18, top, .34, color="0.7", edgecolor="k", label="winner only")
    ax.set_xticks(x); ax.set_xticklabels(ds_list)
    ax.set_ylim(0, 1); ax.set_ylabel("survival under seed resampling")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(Path(outdir) / "fig_ranking_survival.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_single_seed(df, outdir):
    """The paper's headline: win rate against ERM over all seed pairs, per method.

    A bar at 100% means the published gain is reproducible from any single run; a
    bar near 50% means the published result depends on which seed was drawn.
    """
    ds_list = sorted(df["dataset"].unique())
    methods = [m for m in ["rw", "gdro", "adv", "dfr"] if m in set(df["method"])]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(methods))
    for k, ds in enumerate(ds_list):
        piv = df[df["dataset"] == ds].pivot_table(index="seed", columns="method",
                                                  values="acc_worst_group")
        wins = [float(np.mean(piv[m].values[:, None] > piv["erm"].values[None, :]))
                for m in methods]
        ax.bar(x + (k - .5) * .38, wins, .36, label=ds,
               color="0.25" if k == 0 else "0.72", edgecolor="k", linewidth=.5)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of seed pairs\nbeating ERM")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(Path(outdir) / "fig_single_seed.pdf", metadata=PDF_META)
    plt.close(fig)


def main(outdir):
    outdir = Path(outdir)
    fdir = outdir / "figures"
    fdir.mkdir(exist_ok=True)
    df = pd.read_csv(outdir / "results.csv")
    omnibus = json.loads((outdir / "stats" / "omnibus.json").read_text())
    posthoc = json.loads((outdir / "stats" / "posthoc.json").read_text())
    fig_seed_spread(df, fdir)
    fig_deltas(posthoc, fdir)
    fig_cd(omnibus, fdir)
    fig_ranking_survival(posthoc, fdir)
    fig_single_seed(df, fdir)
    print(f"[figures] -> {fdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/full"))
