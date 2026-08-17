#!/usr/bin/env python3
"""Unit-test the gate at Stage 0, not at the end.

Builds a synthetic CLEAN run (fresh dir, resume disabled, full seed coverage,
matched compute, identical input fingerprints, stats artifacts present) and a
synthetic DIRTY run (resume on, a skipped unit, a missing seed, a mismatched
fingerprint, an unmatched epoch count, missing stats), then proves the gate exits
0 and 1 respectively. If this ever stops holding, the gate is broken -- not the run.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "gate_config.yaml").read_text())


def unit(ds, method, seed, fp, epochs=30):
    return {
        "dataset": ds, "method": method, "seed": seed, "device": "cuda",
        "backbone": "resnet50", "epochs": epochs, "batch_size": 64, "lr": 1e-3,
        "hpo_trials": 8, "selection_rule": "val_worst_group_accuracy",
        "metrics": {"acc_mean": 0.9, "acc_worst_group": 0.7 + 0.01 * (seed % 5),
                    "acc_per_group": [0.95, 0.9, 0.85, 0.7], "n_per_group": [100] * 4,
                    "eo_gap": 0.1, "acc_balanced_group": 0.85},
        "hyperparams": {"lr": 1e-3, "weight_decay": 1e-4},
        "val_worst_group": 0.72, "oom_backoffs": 0, "peak_vram_gb": 8.0,
        "peak_rss_gb": 6.0, "wall_s": 120.0,
        "eval_sha256": fp, "n_probe": 64, "min_probe_std": 0.9,
        "train_sha256": "t" + fp[:8], "train_group_hist": [100, 100, 100, 100],
        "n_train": 400,
    }


def build(root, dirty):
    root.mkdir(parents=True, exist_ok=True)
    run = CFG["run"]
    manifest = []
    for ds in run["required_datasets"]:
        fp = "f" * 63 + ("a" if ds == "waterbirds" else "b")
        for m in run["required_methods"]:
            for s in range(run["required_seeds"]):
                if dirty and ds == "celeba" and m == "gdro" and s == 3:
                    manifest.append({"unit": f"{ds}__{m}__seed{s}", "dataset": ds,
                                     "method": m, "seed": s, "status": "skip",
                                     "wall_s": 0.0, "no_resume": False})
                    continue
                bad_fp = fp
                bad_ep = 30
                if dirty and ds == "waterbirds" and m == "adv" and s == 0:
                    bad_fp = "c" * 64            # different pixels than the other methods
                if dirty and ds == "waterbirds" and m == "rw":
                    bad_ep = 60                  # not compute-matched
                u_train = "t" + fp[:8]
                if dirty and ds == "celeba" and m == "dfr" and s == 7:
                    u_train = "DIFFERENT_TRAINING_SET"   # must trip C1
                u = unit(ds, m, s, bad_fp, bad_ep)
                u["train_sha256"] = u_train
                if dirty and ds == "waterbirds" and m == "erm" and s == 5:
                    u["hyperparams"] = {"lr": 9.9e-9, "weight_decay": 1e-4}  # trips G1
                (root / f"{ds}__{m}__seed{s}.json").write_text(json.dumps(u))
                manifest.append({"unit": f"{ds}__{m}__seed{s}", "dataset": ds,
                                 "method": m, "seed": s, "status": "ok",
                                 "wall_s": 120.0, "no_resume": not dirty,
                                 "peak_rss_gb": 6.0, "host_mem_avail_min_gb": 20.0,
                                 "eval_sha256": bad_fp})
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest) + "\n")
    (root / "run_meta.json").write_text(json.dumps({
        "run_started": "2026-07-23T00:00:00+00:00", "no_resume": not dirty,
        "n_units": len(manifest), "timeout_s": 7200, "config_sha256": "selftest",
        "hpo_trials": 8}))
    sdir = root / "stats"
    sdir.mkdir(exist_ok=True)
    (sdir / "omnibus.json").write_text(json.dumps({"test": "friedman", "p": 0.01}))
    if not dirty:
        (sdir / "posthoc.json").write_text(json.dumps({"per_dataset": {}}))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="a5_gatetest_"))
    # G1 compares against configs/tuned.yaml; stand up a matching one for the test
    tuned = ROOT / "configs" / "tuned.yaml"
    saved = tuned.read_text() if tuned.exists() else None
    tuned.write_text(yaml.safe_dump(
        {ds: {m: {"lr": 1e-3, "weight_decay": 1e-4}
              for m in CFG["run"]["required_methods"]}
         for ds in CFG["run"]["required_datasets"]}))
    try:
        codes = {}
        for name, dirty in (("clean", False), ("dirty", True)):
            d = tmp / name
            build(d, dirty)
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "review_gate.py"), str(d)],
                               capture_output=True, text=True)
            codes[name] = r.returncode
            if dirty:
                print(r.stdout.strip())
        print(f"\n[gatetest] clean exit={codes['clean']} (want 0) | "
              f"dirty exit={codes['dirty']} (want 1)")
        ok = codes["clean"] == 0 and codes["dirty"] == 1
        print("[gatetest] " + ("PASS -- the gate separates clean from dirty."
                               if ok else "FAIL -- the gate is broken."))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        tuned.write_text(saved) if saved is not None else tuned.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
