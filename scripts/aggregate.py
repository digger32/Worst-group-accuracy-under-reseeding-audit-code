#!/usr/bin/env python3
"""Merge per-unit JSONs into results.csv and print the coverage the gate will check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main(outdir):
    outdir = Path(outdir)
    rows = []
    for p in sorted(outdir.glob("*__seed*.json")):
        u = json.loads(p.read_text())
        m = u["metrics"]
        rows.append({
            "dataset": u["dataset"], "method": u["method"], "seed": u["seed"],
            "acc_worst_group": m["acc_worst_group"], "acc_mean": m["acc_mean"],
            "acc_balanced_group": m["acc_balanced_group"], "eo_gap": m["eo_gap"],
            "val_worst_group": u.get("val_worst_group"),
            **{f"hp_{k}": v for k, v in (u.get("hyperparams") or {}).items()},
            "backbone": u.get("backbone"), "epochs": u.get("epochs"),
            "batch_size": u.get("batch_size"), "hpo_trials": u.get("hpo_trials"),
            "wall_s": u.get("wall_s"), "peak_rss_gb": u.get("peak_rss_gb"),
            "peak_vram_gb": u.get("peak_vram_gb"), "eval_sha256": u.get("eval_sha256"),
            "oom_backoffs": u.get("oom_backoffs", 0),
        })
    if not rows:
        print(f"[aggregate] no unit JSONs in {outdir}")
        return 1
    df = pd.DataFrame(rows).sort_values(["dataset", "method", "seed"])
    df.to_csv(outdir / "results.csv", index=False)
    cov = df.groupby(["dataset", "method"]).size().rename("n_seeds").reset_index()
    print(f"[aggregate] {len(df)} units -> {outdir/'results.csv'}")
    print(cov.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/full"))
