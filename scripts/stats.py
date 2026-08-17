#!/usr/bin/env python3
"""Statistics for a seed-stability audit.

Three things the paper needs and single-seed reporting cannot give: the paired test
of each method against a fairly tuned ERM across seeds with a multiple-comparison
correction; how often the observed RANKING survives resampling the seeds; and how
many seeds would have been needed to detect the effect the literature reports.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRIC = "acc_worst_group"
B = 5000
RNG = np.random.default_rng(20260723)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def perm_test(diff, max_exact=22):
    """Exact paired sign-flip test by full enumeration.

    Wilcoxon's exact distribution assumes no ties in |d|, and accuracies measured on
    a fixed test set produce ties constantly. Worse, scipy's `auto` silently switches
    between the exact and the tie-corrected normal approximation depending on version
    and data, so the same code gave p = 1.9e-06 on the run host and p = 8.8e-05 here
    for the identical numbers. A reproducibility audit cannot ship a version-dependent
    p-value. Full enumeration of the 2^n sign flips is exact, valid under ties, and
    identical everywhere; at n = 20 it is 1.05M rows and takes under a second.
    """
    d = np.asarray(diff, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    if n <= max_exact:
        signs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8) * 2 - 1
        means = (signs @ d) / n
        return float((np.abs(means) >= obs - 1e-15).mean()), "exact_enumeration"
    rng = np.random.default_rng(20260723)
    signs = rng.choice([-1, 1], size=(200000, n))
    means = (signs @ d) / n
    return float((np.abs(means) >= obs - 1e-15).mean()), "monte_carlo_200k"


def boot_ci(x, n=B):
    draws = RNG.choice(np.asarray(x), size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def seeds_for_power(delta, sd, power=0.8, alpha=0.05):
    """Paired-design normal approximation: n >= ((z_a/2 + z_b) * sd / delta)^2."""
    if not np.isfinite(delta) or abs(delta) < 1e-9 or sd <= 0:
        return None
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    # floored at 2: a paired test needs at least two observations, and the normal
    # approximation returns 1 whenever the effect dwarfs the spread, which reads as
    # nonsense in a table.
    return max(2, int(np.ceil((z * sd / abs(delta)) ** 2)))


def ranking_survival(df, methods):
    """Fraction of seed-bootstrap resamples whose method ranking equals the full-sample
    ranking, and the fraction preserving only the winner."""
    piv = df.pivot_table(index="seed", columns="method", values=METRIC)[methods]
    full = tuple(piv.mean().sort_values(ascending=False).index)
    same, same_top = 0, 0
    idx = np.arange(len(piv))
    for _ in range(B):
        r = piv.iloc[RNG.choice(idx, size=len(idx), replace=True)]
        order = tuple(r.mean().sort_values(ascending=False).index)
        same += order == full
        same_top += order[0] == full[0]
    return {"full_sample_order": list(full), "exact_order_survival": same / B,
            "winner_survival": same_top / B}


def main(outdir):
    outdir = Path(outdir)
    df = pd.read_csv(outdir / "results.csv")
    sdir = outdir / "stats"
    sdir.mkdir(exist_ok=True)
    methods = sorted(df["method"].unique())
    baseline = "erm"

    posthoc, omnibus = {"per_dataset": {}}, {}
    for ds, d in df.groupby("dataset"):
        piv = d.pivot_table(index="seed", columns="method", values=METRIC).dropna()
        if baseline not in piv.columns or piv.shape[0] < 3:
            continue
        others = [m for m in piv.columns if m != baseline]
        raw_p, entries = [], {}
        for m in others:
            a, b = piv[m].to_numpy(), piv[baseline].to_numpy()
            diff = a - b
            p, p_method = perm_test(diff)
            try:  # kept for comparison only; NOT the reported test
                p_wil = float(stats.wilcoxon(a, b, zero_method="zsplit",
                                             method="approx").pvalue)
            except ValueError:
                p_wil = float("nan")
            raw_p.append(p)
            entries[m] = {
                "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                "ci95": boot_ci(a), "delta_vs_erm": float(diff.mean()),
                "delta_ci95": boot_ci(diff), "perm_p": p, "perm_method": p_method,
                "wilcoxon_p_approx_reference": p_wil,
                "n_ties_in_abs_diff": int(len(diff) - len(np.unique(np.abs(diff)))),
                "seeds_for_80pct_power": seeds_for_power(diff.mean(), diff.std(ddof=1)),
                "n_seeds": int(len(a)),
            }
        for m, padj in zip(others, holm(np.array(raw_p))):
            entries[m]["perm_p_holm"] = float(padj)
            entries[m]["significant_holm_0.05"] = bool(padj < 0.05)
        entries[baseline] = {"mean": float(piv[baseline].mean()),
                             "sd": float(piv[baseline].std(ddof=1)),
                             "ci95": boot_ci(piv[baseline].to_numpy()),
                             "n_seeds": int(piv.shape[0])}
        posthoc["per_dataset"][ds] = {
            "methods": entries,
            "ranking": ranking_survival(d, [m for m in methods if m in piv.columns]),
        }

    # Omnibus across datasets x seeds treated as blocks, methods as treatments.
    piv = df.pivot_table(index=["dataset", "seed"], columns="method", values=METRIC).dropna()
    if piv.shape[0] >= 3 and piv.shape[1] >= 3:
        chi2, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
        ranks = piv.rank(axis=1, ascending=False).mean().to_dict()
        k, n = piv.shape[1], piv.shape[0]
        cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * \
            np.sqrt(k * (k + 1) / (6 * n))
        omnibus = {"test": "friedman", "chi2": float(chi2), "p": float(p),
                   "n_blocks": int(n), "k_methods": int(k),
                   "mean_ranks": {m: float(v) for m, v in ranks.items()},
                   "nemenyi_cd_0.05": float(cd),
                   "pairs_beyond_cd": [f"{a} vs {b}" for a, b in combinations(ranks, 2)
                                       if abs(ranks[a] - ranks[b]) > cd]}
    else:
        omnibus = {"test": "friedman", "status": "degenerate",
                   "reason": f"{piv.shape[0]} blocks x {piv.shape[1]} methods"}

    import scipy, sys as _sys
    omnibus["environment"] = {"python": _sys.version.split()[0],
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "pandas": pd.__version__}
    omnibus["paired_test"] = ("exact enumerated sign-flip permutation test; "
                              "Wilcoxon is recorded only as a reference because its "
                              "exact null is invalid under ties in |d|")
    (sdir / "omnibus.json").write_text(json.dumps(omnibus, indent=2))
    (sdir / "posthoc.json").write_text(json.dumps(posthoc, indent=2))
    print(f"[stats] -> {sdir}/omnibus.json, {sdir}/posthoc.json")
    print(json.dumps(omnibus, indent=2)[:900])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/full"))
