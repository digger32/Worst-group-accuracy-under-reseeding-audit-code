#!/usr/bin/env python3
"""DIAGNOSTIC: how much of a tune log's selection is actually meaningful.

For each (dataset, method) it reports how many trials are indistinguishable from the
argmax at the metric's own resolution -- 1/66 on Waterbirds, 1/91 on CelebA, because
the worst-group accuracy is measured on that many images. A large tie count means
the selection among those trials carries no information and Methods should say so.

    python scripts/reselect_tuned.py logs/tune_<stamp>.log

`--write` exists but is NOT the default path. It rewrites configs/tuned.yaml with a
central-instead-of-argmax choice, which was tried and dropped: the pilot showed the
baseline was not destabilised by its argmax learning rate, and any tie-break picks
along one dimension while dragging the others (on Waterbirds/group-DRO it swung
`eta` 38-fold as a side effect). Argmax is what the paper reports.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRIAL = re.compile(
    r"\[tune\] (\w+)/(\w+) trial (\d+): lr=([\d.eE+-]+) wd=([\d.eE+-]+)"
    r"((?: \w+=[\d.eE+-]+)*) val_worst=([\d.]+)")


def parse(path):
    trials = defaultdict(list)
    for ds, method, t, lr, wd, extra, score in TRIAL.findall(Path(path).read_text()):
        rec = {"trial": int(t), "lr": float(lr), "weight_decay": float(wd),
               "score": float(score)}
        for kv in extra.split():
            k, v = kv.split("=")
            rec[k] = float(v)
        trials[(ds, method)].append(rec)
    return trials


def resolution(dataset):
    """1 / size of the smallest group in the validation SELECTION half."""
    from a5 import data as D

    _, meta, _ = D.load_prepared(dataset)
    va = D.split_indices(meta, "val")
    g = D.group_of(meta["y"], meta["a"])[va]
    return 1.0 / max(1, int(np.bincount(g, minlength=4).min()) // 2)


def choose(trials, tol):
    best = max(t["score"] for t in trials)
    tie = [t for t in trials if best - t["score"] < tol]
    med = np.exp(np.median(np.log([t["lr"] for t in tie])))
    return min(tie, key=lambda t: abs(np.log(t["lr"]) - np.log(med))), len(tie)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--resolution", type=float, default=None,
                    help="override, e.g. when the prepared corpus is not on this host")
    args = ap.parse_args()

    trials = parse(args.log)
    if not trials:
        raise SystemExit(f"[reselect] no trials parsed from {args.log}")
    path = ROOT / "configs" / "tuned.yaml"
    tuned = yaml.safe_load(path.read_text()) if path.exists() else {}

    res_cache, changed = {}, 0
    print(f"{'pair':20s}{'ties':>6s}{'lr before':>12s}{'lr after':>12s}{'score':>18s}")
    for (ds, method), ts in sorted(trials.items()):
        if ds not in res_cache:
            res_cache[ds] = args.resolution or resolution(ds)
        picked, n_tie = choose(ts, res_cache[ds])
        argmax = max(ts, key=lambda t: t["score"])
        mark = ""
        if picked["trial"] != argmax["trial"]:
            changed += 1
            mark = "  <- changed"
        print(f"{ds + '/' + method:20s}{n_tie:6d}{argmax['lr']:12.2e}{picked['lr']:12.2e}"
              f"{f'{argmax[chr(115)+chr(99)+chr(111)+chr(114)+chr(101)]:.4f} -> {picked[chr(115)+chr(99)+chr(111)+chr(114)+chr(101)]:.4f}':>18s}{mark}")
        tuned.setdefault(ds, {})[method] = {
            k: v for k, v in picked.items() if k not in ("score", "trial")}

    print(f"\n{changed} of {len(trials)} pairs changed; "
          f"metric resolution used: { {k: round(v, 4) for k, v in res_cache.items()} }")
    if args.write:
        path.write_text(yaml.safe_dump(tuned, sort_keys=True))
        print(f"[reselect] written -> {path}")
    else:
        print("[reselect] dry run; pass --write to update configs/tuned.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
