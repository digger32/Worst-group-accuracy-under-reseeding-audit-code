"""Fairness / robustness metrics. Pure numpy so the aggregate and gate stages
never need torch."""
from __future__ import annotations

import numpy as np


def per_group_accuracy(y_true, y_pred, groups, n_groups=4):
    accs, counts = [], []
    for g in range(n_groups):
        m = groups == g
        counts.append(int(m.sum()))
        accs.append(float((y_pred[m] == y_true[m]).mean()) if m.any() else float("nan"))
    return accs, counts


def equalised_odds_gap(y_true, y_pred, spurious):
    """max over {TPR, FPR} of the absolute gap between the two spurious-attribute
    groups. Binary y, binary a -- the standard setting for both benchmarks."""
    gaps = []
    for y_ref in (1, 0):
        rates = []
        for a in (0, 1):
            m = (y_true == y_ref) & (spurious == a)
            rates.append(float((y_pred[m] == 1).mean()) if m.any() else float("nan"))
        gaps.append(abs(rates[0] - rates[1]))
    return float(np.nanmax(gaps))


def score_all(y_true, y_pred, groups, n_groups=4):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    groups = np.asarray(groups).astype(np.int64)
    spurious = groups % 2
    accs, counts = per_group_accuracy(y_true, y_pred, groups, n_groups)
    return {
        "acc_mean": float((y_pred == y_true).mean()),
        "acc_worst_group": float(np.nanmin(accs)),
        "acc_per_group": accs,
        "n_per_group": counts,
        "eo_gap": equalised_odds_gap(y_true, y_pred, spurious),
        "acc_balanced_group": float(np.nanmean(accs)),
    }
