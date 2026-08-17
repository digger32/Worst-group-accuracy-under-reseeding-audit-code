#!/usr/bin/env python3
"""Equal-budget hyperparameter search, so "debiasing beats ERM" cannot be an
artefact of a well-tuned method meeting a badly-tuned baseline.

Every (dataset, method) pair gets the SAME number of trials over the SAME space,
selected on validation worst-group accuracy at a reduced epoch budget, on seed 0
only. The reduction is identical for all methods and is disclosed in Methods; the
resulting configs are frozen into configs/tuned.yaml before the seed grid starts,
so no tuning decision can leak into the 20-seed comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TUNE_EPOCH_FRAC = 1 / 3


# Each method's own decisive knob, searched alongside the shared optimiser settings.
# Leaving these fixed would under-tune the debiasers, and that error points STRAIGHT
# AT THIS PAPER'S CONCLUSION: a weakened debiaser makes "the reported gains vanish"
# easier to conclude. The baseline and the methods have to be tuned with equal care,
# in both directions.
#   gdro -- eta, the exponentiated-gradient step on the group weights. `adj` stays 0,
#           which is Sagawa et al.'s setting when the epoch is chosen by worst-group
#           validation accuracy, as it is here.
#   adv  -- lambda_adv, the weight of the reversed adversarial gradient. Too large and
#           it destroys the features; 1.0 was an arbitrary default.
#   dfr  -- C, the regularisation of the refitted head.
# lr upper bound raised from 10^-2.5 to 10^-1.5 after the 26 Jul run: Waterbirds/ERM
# selected the LARGEST learning rate it was offered (97% of the range, correlation
# +0.93 between log-lr and score), i.e. the optimum lay outside the box. 3.2e-3 is
# simply low for fine-tuning a ResNet-50 with SGD. An under-tuned ERM attacks this
# paper's own promise -- that the gains dissolve against a FAIRLY tuned baseline --
# so the box has to contain the optimum rather than merely touch it.
SHARED_SPACE = {"lr": (-4.5, -1.5), "weight_decay": (-5.0, -2.5)}

METHOD_SPACE = {
    "gdro": ("eta", -3.0, -0.5),
    "adv": ("lambda_adv", -2.0, 0.7),
    "dfr": ("C", -3.0, 1.0),
}


def space(rng, method):
    sp = {"lr": float(10 ** rng.uniform(*SHARED_SPACE["lr"])),
          "weight_decay": float(10 ** rng.uniform(*SHARED_SPACE["weight_decay"]))}
    knob = METHOD_SPACE.get(method)
    if knob:
        name, lo, hi = knob
        sp[name] = float(10 ** rng.uniform(lo, hi))
    return sp


def resolution(dataset, grid):
    """Granularity of the worst-group metric: 1 / size of the smallest group in the
    validation selection half. Score differences below this are not measurable."""
    import numpy as np

    from wgaudit import data as D

    _, meta, _ = D.load_prepared(dataset)
    va = D.split_indices(meta, "val")
    g = D.group_of(meta["y"], meta["a"])[va]
    smallest = int(np.bincount(g, minlength=4).min()) // 2   # the selection half
    return 1.0 / max(1, smallest)


def choose(trials, tol):
    """Plain argmax, plus a count of how many trials are indistinguishable from it.

    An earlier version broke ties toward a "central" configuration, on the theory
    that the argmax can land on a stability shoulder and turn a tuning artefact into
    seed variance. The pilot refuted it: Waterbirds/ERM ran at the suspect learning
    rate and produced the SECOND SMALLEST seed spread of the five methods (0.026,
    against 0.049 for adversarial and 0.063 for group-DRO). Worse, any tie-break
    picks along one dimension and drags the others: on Waterbirds/group-DRO three
    trials tied with near-identical learning rates, so the rule swung `eta` by a
    factor of 38 as a side effect. A tie is a tie; breaking it by a rule of my own
    invention is not safer than argmax, only differently arbitrary. The tie count is
    still reported, because a large one is worth knowing when writing Methods.
    """
    best = max(trials, key=lambda t: t["score"])
    tie = [t for t in trials if best["score"] - t["score"] < tol]
    return best, len(tie)


def main():
    from wgaudit.train import run_unit

    grid = yaml.safe_load((ROOT / "configs" / "grid.yaml").read_text())
    mcfg = yaml.safe_load((ROOT / "configs" / "methods.yaml").read_text())
    n_trials = int(mcfg["hpo_trials"])
    warnings = []
    out = ROOT / "runs" / "tune"
    out.mkdir(parents=True, exist_ok=True)
    best_path = ROOT / "configs" / "tuned.yaml"
    tuned = yaml.safe_load(best_path.read_text()) if best_path.exists() else {}

    for ds, dcfg in grid["datasets"].items():
        for method in mcfg["methods"]:
            if tuned.get(ds, {}).get(method):
                print(f"[tune] {ds}/{method}: already tuned -- skipping")
                continue
            rng = np.random.default_rng(abs(hash((ds, method))) % 2**32)
            trials = []
            for t in range(n_trials):
                trial = space(rng, method)
                trial["epochs"] = max(1, int(round(dcfg["epochs"] * TUNE_EPOCH_FRAC)))
                over = {ds: {method: trial}}
                tag = out / f"{ds}__{method}__trial{t}"
                tag.mkdir(exist_ok=True)
                try:
                    r = run_unit(ds, method, 0, tag, grid, mcfg, over)
                except Exception as e:  # a bad corner of the space must not stop the search
                    print(f"[tune] {ds}/{method} trial {t} failed: {str(e)[:120]}")
                    continue
                score = r["val_worst_group"]
                extra = "".join(f" {k}={trial[k]:.3g}" for k in trial
                                if k not in ("lr", "weight_decay", "epochs"))
                print(f"[tune] {ds}/{method} trial {t}: lr={trial['lr']:.2e} "
                      f"wd={trial['weight_decay']:.2e}{extra} val_worst={score:.4f}")
                trials.append({**trial, "score": score})
            if not trials:
                raise SystemExit(f"[tune] every trial failed for {ds}/{method}")
            with (out / "trials.jsonl").open("a") as fh:
                for t in trials:
                    fh.write(json.dumps({"dataset": ds, "method": method, **t}) + "\n")
            tol = resolution(ds, grid)
            picked, n_tie = choose(trials, tol)
            if n_tie > 1:
                print(f"[tune] {ds}/{method}: {n_tie} of {len(trials)} trials are "
                      f"indistinguishable at the metric's resolution ({tol:.4f}); "
                      f"the selection among them is not meaningful and Methods says so")
            best = (picked["score"], {k: v for k, v in picked.items() if k != "score"})
            # Boundary audit. A selected value in the outer decile of its range means
            # the search box may be cutting off the optimum, which is how the 26 Jul
            # run under-tuned ERM. Reported per pair so it cannot go unnoticed again.
            ranges = dict(SHARED_SPACE)
            if method in METHOD_SPACE:
                knob, klo, khi = METHOD_SPACE[method]
                ranges[knob] = (klo, khi)
            for name, (lo, hi) in ranges.items():
                if name not in best[1]:
                    continue
                pos = (np.log10(best[1][name]) - lo) / (hi - lo)
                if pos < 0.10 or pos > 0.90:
                    warnings.append(f"{ds}/{method}: {name}={best[1][name]:.3g} sits at "
                                    f"{pos * 100:.0f}% of its search range -- widen the "
                                    f"range and re-tune this pair")
            keep = {k: v for k, v in best[1].items() if k != "epochs"}
            tuned.setdefault(ds, {})[method] = keep
            best_path.write_text(yaml.safe_dump(tuned, sort_keys=True))
            print(f"[tune] {ds}/{method} -> {keep} (val_worst={best[0]:.4f})")

    counts = {ds: {m: n_trials for m in mcfg["methods"]} for ds in grid["datasets"]}
    (out / "tuning_budget.json").write_text(json.dumps(
        {"trials_per_pair": counts, "tune_epoch_fraction": TUNE_EPOCH_FRAC,
         "selection": "val_worst_group_accuracy", "seed": 0}, indent=2))
    if warnings:
        print("\n[tune] BOUNDARY WARNINGS -- the search box may be cutting the optimum:")
        for w in warnings:
            print("  - " + w)
        print("[tune] tuned.yaml was still written; decide whether to widen and re-run.")
    else:
        print("[tune] boundary audit: every selected value lies inside its range.")
    print(f"[tune] frozen -> {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
