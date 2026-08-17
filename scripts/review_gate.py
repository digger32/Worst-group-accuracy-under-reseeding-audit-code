#!/usr/bin/env python3
"""Review-proofing gate. Exit 0 = the numbers may be frozen; exit 1 = they may not.

Clean reproduction and external validity are the minimum. Input integrity, seed
coverage and compute matching are this study's own failure modes: an audit whose
claim concerns seed variance is destroyed by a silent input bug, by uneven seed
coverage, or by methods that were not in fact compute-matched -- so the gate refuses
those as hard as it refuses a dirty rerun.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(outdir):
    outdir = Path(outdir)
    meta = json.loads((outdir / "run_meta.json").read_text())
    manifest = [json.loads(l) for l in (outdir / "manifest.jsonl").read_text().splitlines()
                if l.strip()] if (outdir / "manifest.jsonl").exists() else []
    units = [json.loads(p.read_text()) for p in sorted(outdir.glob("*__seed*.json"))]
    cfg = yaml.safe_load((ROOT / "configs" / "gate_config.yaml").read_text())
    return outdir, meta, manifest, units, cfg


def gate(outdir):
    outdir, meta, manifest, units, cfg = load(outdir)
    run, checks = cfg["run"], cfg["checks"]
    exceptions = set(run.get("audited_exceptions") or [])
    fails, notes = [], []

    # ---- clean-reproduction: clean reproduction -------------------------------------------- #
    if checks.get("a1_clean_reproduction"):
        if not meta.get("no_resume"):
            fails.append("clean-reproduction: run_meta.no_resume is false -- the final pass must run with resume DISABLED")
        skipped = [r["unit"] for r in manifest if r.get("status") == "skip"]
        if skipped:
            fails.append(f"clean-reproduction: {len(skipped)} unit(s) were skipped, e.g. {skipped[:3]}")
        bad = [r["unit"] for r in manifest
               if r.get("status") != "ok" and r["unit"] not in exceptions]
        if bad:
            fails.append(f"clean-reproduction: {len(bad)} unit(s) did not complete: {bad[:5]}")
        undocumented = [u for u in exceptions
                        if not (run.get("exception_justifications") or {}).get(u)]
        if undocumented:
            fails.append(f"clean-reproduction: audited exceptions without a written justification: {undocumented}")

    # ---- external-validity: external validity --------------------------------------------- #
    present_ds = {u["dataset"] for u in units}
    if checks.get("b1_external_validity"):
        for c in cfg["claims"]:
            if c.get("waive"):
                if not c.get("waiver_justification"):
                    fails.append(f"external-validity: claim {c['id']} is waived without a justification")
                else:
                    notes.append(f"external-validity: claim {c['id']} waived -- {c['waiver_justification']}")
                continue
            missing = [d for d in c["independent_datasets"] if d not in present_ds]
            if missing:
                fails.append(f"external-validity: claim {c['id']} needs {missing}, absent from this run")

    # ---- input-integrity: input integrity ------------------------------------------------ #
    if checks.get("c1_input_integrity"):
        by_ds, tr_ds = defaultdict(set), defaultdict(set)
        for u in units:
            by_ds[u["dataset"]].add(u.get("eval_sha256"))
            tr_ds[u["dataset"]].add(u.get("train_sha256"))
            if not u.get("min_probe_std", 1) > 0:
                fails.append(f"input-integrity: {u['dataset']}/{u['method']}/seed{u['seed']} scored a "
                             f"constant eval probe -- the model was not fed images")
        for ds, hs in by_ds.items():
            if len(hs) > 1:
                fails.append(f"input-integrity: {ds} has {len(hs)} distinct eval fingerprints -- "
                             f"methods were evaluated on different pixels")
        for ds, hs in tr_ds.items():
            if len(hs) > 1:
                fails.append(f"input-integrity: {ds} has {len(hs)} distinct training fingerprints -- "
                             f"the training set must be identical across seeds and "
                             f"methods, or the seed-variance claim is unattributable")

    # ---- seed-coverage: seed coverage -------------------------------------------------- #
    if checks.get("d1_seed_coverage"):
        seen = defaultdict(set)
        for u in units:
            seen[(u["dataset"], u["method"])].add(u["seed"])
        for ds in run["required_datasets"]:
            for m in run["required_methods"]:
                got = seen.get((ds, m), set())
                if len(got) < run["required_seeds"]:
                    fails.append(f"seed-coverage: {ds}/{m} has {len(got)}/{run['required_seeds']} seeds")

    # ---- compute-matching: compute matching ----------------------------------------------- #
    if checks.get("e1_compute_matching"):
        by_ds = defaultdict(list)
        for u in units:
            by_ds[u["dataset"]].append(u)
        for ds, us in by_ds.items():
            for field in ("backbone", "epochs", "batch_size", "hpo_trials"):
                vals = {u.get(field) for u in us}
                if len(vals) > 1:
                    fails.append(f"compute-matching: {ds} has non-matching {field} across methods: {vals} "
                                 f"-- a gain measured against an under-tuned baseline is not a gain")
        oom = [f"{u['dataset']}/{u['method']}/seed{u['seed']}" for u in units
               if u.get("oom_backoffs")]
        if oom:
            fails.append(f"compute-matching: {len(oom)} unit(s) hit an OOM backoff, so their effective "
                         f"batch differed: {oom[:5]}")

    # ---- frozen-hyperparams: the run used the frozen hyperparameters ------------------------ #
    if checks.get("g1_frozen_hyperparams", True):
        tuned_path = ROOT / "configs" / "tuned.yaml"
        tuned = yaml.safe_load(tuned_path.read_text()) if tuned_path.exists() else None
        by_pair = defaultdict(list)
        for u in units:
            by_pair[(u["dataset"], u["method"])].append(u)
        for (ds, m), us in sorted(by_pair.items()):
            seen = {json.dumps(u.get("hyperparams"), sort_keys=True) for u in us}
            if len(seen) > 1:
                fails.append(f"frozen-hyperparams: {ds}/{m} ran with {len(seen)} different "
                             f"hyperparameter sets across seeds")
            if tuned is None:
                continue
            want = (tuned.get(ds) or {}).get(m)
            got = us[0].get("hyperparams")
            if want is None:
                fails.append(f"frozen-hyperparams: configs/tuned.yaml has no entry for {ds}/{m}")
            elif got is None:
                fails.append(f"frozen-hyperparams: {ds}/{m} units record no hyperparams -- rerun with "
                             f"the current build")
            else:
                diff = {k: (want[k], got.get(k)) for k in want
                        if abs(float(want[k]) - float(got.get(k, float("nan")))) > 1e-12}
                if diff:
                    fails.append(f"frozen-hyperparams: {ds}/{m} ran with hyperparameters that differ "
                                 f"from configs/tuned.yaml: {diff}")

    # ---- stats-artifacts: stats artifacts ------------------------------------------------ #
    if checks.get("f1_stats_artifacts"):
        for name, rel in cfg["stats_artifacts"].items():
            if not (outdir / rel).exists():
                fails.append(f"stats-artifacts: missing stats artifact {name} at {rel}")

    print(f"[gate] {outdir} | units={len(units)} | no_resume={meta.get('no_resume')} "
          f"| config={meta.get('config_sha256')}")
    for n in notes:
        print("  note: " + n)
    if fails:
        print(f"[gate] FAIL ({len(fails)})")
        for f in fails:
            print("  - " + f)
        return 1
    print("[gate] PASS -- numbers may be frozen.")
    return 0


if __name__ == "__main__":
    sys.exit(gate(sys.argv[1] if len(sys.argv) > 1 else "runs/final"))
