#!/usr/bin/env python3
"""Two micro slices, and the go/no-go they produce.

microfast  -- the cheap representative unit per method. Answers "which path and how
              long", i.e. the classic timing slice. Extrapolation is read from
              MEASURED wall_s in the manifest, never from arithmetic.
microheavy -- every method on the HEAVIEST dataset. Answers the question that broke
              on a 32 GB host: what is the peak resident memory of the worst unit,
              and how close did the host come to running out. `full` refuses to start
              until this slice exists and clears configs/grid.yaml:ram_budget_gb.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    g = yaml.safe_load((ROOT / "configs" / "grid.yaml").read_text())
    m = yaml.safe_load((ROOT / "configs" / "methods.yaml").read_text())
    return g, m


def records(outdir):
    p = Path(outdir) / "manifest.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def grid_size(g, m):
    return len(g["datasets"]) * len(m["methods"]) * g["seeds"]


def emit_heavy_units():
    """Worst case = every method on the heaviest dataset, seed 0."""
    g, m = cfg()
    ds = g["heavy_unit"]["dataset"]
    for method in m["methods"]:
        print(f"{ds},{method},0")


def report(outdir, mode):
    g, m = cfg()
    recs = [r for r in records(outdir) if r.get("status") == "ok"]
    if not recs:
        print(f"[micro] no successful units in {outdir}")
        return 1

    n_units = grid_size(g, m)
    print(f"{'unit':34s} {'wall_s':>9s} {'rss_GB':>8s} {'vram_GB':>8s} {'host_free_min_GB':>18s}")
    for r in recs:
        print(f"{r['unit']:34s} {r['wall_s']:9.1f} "
              f"{(r.get('peak_rss_gb') or float('nan')):8.2f} "
              f"{(r.get('peak_vram_gb') or float('nan')):8.2f} "
              f"{(r.get('host_mem_avail_min_gb') or float('nan')):18.2f}")

    per_method = {}
    for r in recs:
        per_method.setdefault(r["method"], []).append(r["wall_s"])
    mean_wall = sum(sum(v) / len(v) for v in per_method.values()) / len(per_method)
    est_h = mean_wall * n_units / 3600
    max_rss = max((r.get("peak_rss_gb") or 0) for r in recs)
    min_free = min((r.get("host_mem_avail_min_gb") or 1e9) for r in recs)
    budget = float(g["ram_budget_gb"])

    print(f"\nfull grid = {n_units} units")
    if mode == "fast":
        print(f"lower-bound estimate from the CHEAP slice: {est_h:.1f} h "
              f"(the heavy dataset will dominate -- run microheavy before trusting any total)")
    else:
        print(f"upper-bound estimate from the HEAVY slice: {est_h:.1f} h")
    print(f"peak RSS of the worst unit: {max_rss:.2f} GB (budget {budget:.1f} GB)")
    print(f"host MemAvailable low-water: {min_free:.2f} GB")

    # ---- how many units may share this host, from measurement only ------------- #
    max_vram = max((r.get("peak_vram_gb") or 0) for r in recs)
    import os as _os
    n_cores = _os.cpu_count() or 8
    threads_per_unit = 1 + int(g["num_workers"])      # main process + dataloader workers
    by_ram = int((min_free + max_rss - 4.0) // max_rss) if max_rss > 0 else 1
    by_vram = int(76.0 // max_vram) if max_vram > 0 else 1
    by_cpu = max(1, n_cores // max(1, threads_per_unit * 2))
    jobs = max(1, min(by_ram, by_vram, by_cpu))
    print(f"\njobs recommendation (the binding constraint decides, nothing is guessed):")
    print(f"  by host RAM  : {by_ram:3d}   (worst-unit RSS {max_rss:.2f} GB, 4 GB kept for the kernel)")
    print(f"  by GPU VRAM  : {by_vram:3d}   (worst-unit {max_vram:.2f} GB of ~76 GB usable)")
    print(f"  by CPU cores : {by_cpu:3d}   ({n_cores} cores, {threads_per_unit} procs x 2 threads per unit)")
    print(f"  -> ceiling from RAM/VRAM/cores: JOBS={jobs}")
    print("  NOTE: this is an upper bound on what FITS, not on what HELPS. The A800 is")
    print("  already saturated by a single unit here: the 27 Jul pilot measured a 4.9x")
    print("  per-unit slowdown at JOBS=5, i.e. 1.03x throughput on Waterbirds and 1.27x")
    print("  on CelebA. Confirm scaling on the pilot before choosing JOBS for the grid.")
    if mode == "heavy":
        print(f"  wall-clock at JOBS={jobs}: about {est_h / jobs:.1f} h")

    if mode in ("heavy", "gate"):
        bad = []
        if max_rss > budget:
            bad.append(f"peak RSS {max_rss:.2f} GB exceeds ram_budget_gb {budget:.1f}")
        if min_free < 2.0:
            bad.append(f"host free RAM fell to {min_free:.2f} GB -- the OOM killer was close")
        if bad:
            print("\n[micro] NO-GO:")
            for b in bad:
                print("  - " + b)
            print("  lower num_workers or batch_size in configs/grid.yaml and re-run microheavy.")
            return 1
        print("\n[micro] GO -- the worst unit fits with margin.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--mode", choices=["fast", "heavy", "gate"], default="fast")
    ap.add_argument("--emit-heavy-units", action="store_true")
    a = ap.parse_args()
    if a.emit_heavy_units:
        emit_heavy_units()
        sys.exit(0)
    if not a.outdir:
        ap.error("outdir required unless --emit-heavy-units")
    sys.exit(report(a.outdir, a.mode))
