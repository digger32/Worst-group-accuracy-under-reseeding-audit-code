#!/usr/bin/env python3
"""Job-based runner. unit = dataset x method x seed, each in its OWN subprocess,
so a hang, an OOM kill or a segfault costs one unit and never the batch.

Beyond the house pattern (resume, per-unit hard timeout, atomic writes, manifest,
--no-resume for the final pass) this runner adds one thing experience made
necessary: while each unit runs, a sampler watches host MemAvailable and records
the LOW-WATER mark. On a 32 GB box that number, not arithmetic, decides whether
the full grid is safe.

Launch (always inside tmux):
    tmux new -s wgaudit
    bash runner/pipeline.sh microheavy
    # detach: Ctrl-b d
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Small-n / small-batch rule: parallelism belongs at the UNIT level, never inside
# one fit. Unpinned BLAS threads cost far more in sync than they buy in arithmetic.
WORKER_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}


def load_cfg():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "grid.yaml").read_text())
    mcfg = yaml.safe_load((ROOT / "configs" / "methods.yaml").read_text())
    return cfg, mcfg


def config_sha256():
    h = hashlib.sha256()
    for name in ("grid.yaml", "methods.yaml"):
        h.update((ROOT / "configs" / name).read_bytes())
    return h.hexdigest()[:16]


def unit_id(dataset, method, seed):
    return f"{dataset}__{method}__seed{seed}"


def mem_available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 2**20, 2)
    except Exception:
        pass
    return None


class HostMemWatch(threading.Thread):
    """Samples host MemAvailable while a unit runs. The low-water mark is the only
    honest evidence about whether the next stage fits in 32 GB."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period, self.low, self._stop = period, mem_available_gb(), threading.Event()

    def run(self):
        while not self._stop.wait(self.period):
            v = mem_available_gb()
            if v is not None and (self.low is None or v < self.low):
                self.low = v

    def stop(self):
        self._stop.set()
        return self.low


def build_units(cfg, mcfg, want_ds, want_methods, max_seeds, only):
    if only:
        d, m, s = only.split(",")
        return [(d, m, int(s))]
    methods = want_methods or list(mcfg["methods"])
    # DFR consumes the ERM run of the same (dataset, seed); ordering erm first keeps
    # every unit independent of any EARLIER run, including on a fresh --no-resume dir.
    methods = sorted(methods, key=lambda m: mcfg["methods"][m].get("needs_base") is not None)
    units = []
    for ds in (want_ds or list(cfg["datasets"])):
        for s in range(min(cfg["seeds"], max_seeds)):
            for m in methods:
                units.append((ds, m, s))
    return units


def run_orchestrator(args):
    cfg, mcfg = load_cfg()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True)

    units = build_units(cfg, mcfg,
                        args.datasets.split(",") if args.datasets else None,
                        args.methods.split(",") if args.methods else None,
                        args.max_seeds, args.only)
    jobs = max(1, int(args.jobs))
    run_started = datetime.now(timezone.utc).isoformat()
    (outdir / "run_meta.json").write_text(json.dumps({
        "run_started": run_started, "no_resume": args.no_resume,
        "n_units": len(units), "timeout_s": args.timeout_s, "jobs": jobs,
        "config_sha256": config_sha256(),
        "datasets": sorted({d for d, _, _ in units}),
        "methods": sorted({m for _, m, _ in units}),
        "seeds": sorted({s for _, _, s in units}),
        "hpo_trials": mcfg["hpo_trials"],
        "host_mem_available_gb_at_start": mem_available_gb(),
    }, indent=2))

    print(f"[runner] {len(units)} units | outdir={outdir} | jobs={jobs} "
          f"| no_resume={args.no_resume} | timeout={args.timeout_s}s "
          f"| mem_avail={mem_available_gb()} GB", flush=True)

    pending = []
    n_skip = 0
    for ds, m, s in units:
        out_path = outdir / f"{unit_id(ds, m, s)}.json"
        if out_path.exists() and not args.no_resume:
            n_skip += 1
            print(f"[skip] {unit_id(ds, m, s)}", flush=True)
            continue
        if out_path.exists():
            out_path.unlink()
        pending.append((ds, m, s))

    watch = HostMemWatch()
    watch.start()
    manifest_lock = threading.Lock()
    active, counts = [], {"ok": 0, "fail": 0, "timeout": 0}

    def finish(job, status):
        ds, m, s, proc, lf, t0 = job
        lf.close()
        uid = unit_id(ds, m, s)
        out_path = outdir / f"{uid}.json"
        rec = {"unit": uid, "dataset": ds, "method": m, "seed": s, "status": status,
               "started": run_started,
               "finished": datetime.now(timezone.utc).isoformat(),
               "wall_s": round(time.time() - t0, 1), "no_resume": args.no_resume,
               "jobs": jobs, "host_mem_avail_min_gb": watch.low}
        if out_path.exists():
            try:
                r = json.loads(out_path.read_text())
                rec.update({"peak_rss_gb": r.get("peak_rss_gb"),
                            "peak_vram_gb": r.get("peak_vram_gb"),
                            "eval_sha256": r.get("eval_sha256")})
            except Exception:
                pass
        with manifest_lock, (outdir / "manifest.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if status == "ok":
            counts["ok"] += 1
            print(f"[ok] {uid} ({rec['wall_s']}s, rss={rec.get('peak_rss_gb')} GB, "
                  f"vram={rec.get('peak_vram_gb')} GB, host_free_min={watch.low} GB)",
                  flush=True)
        elif status == "timeout":
            counts["timeout"] += 1
            print(f"[TIMEOUT] {uid} > {args.timeout_s}s -- batch continues", flush=True)
        else:
            counts["fail"] += 1
            hint = " (rc=-9 is the host OOM killer, not our code)" if status.endswith("-9)") else ""
            print(f"[FAIL] {uid} {status}{hint} -- batch continues "
                  f"| log: {logdir / (uid + '.log')}", flush=True)

    queue = list(pending)
    while queue or active:
        while queue and len(active) < jobs:
            ds, m, s = queue.pop(0)
            uid = unit_id(ds, m, s)
            lf = open(logdir / f"{uid}.log", "w")
            proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--worker",
                 "--dataset", ds, "--method", m, "--seed", str(s),
                 "--outdir", str(outdir)],
                stdout=lf, stderr=subprocess.STDOUT,
                env=dict(os.environ, **WORKER_ENV), start_new_session=True)
            active.append((ds, m, s, proc, lf, time.time()))

        time.sleep(1.0)
        for job in list(active):
            ds, m, s, proc, lf, t0 = job
            rc = proc.poll()
            if rc is None:
                if time.time() - t0 > args.timeout_s:
                    # Kill the whole process group: a hung dataloader worker would
                    # otherwise survive its parent and keep holding RAM.
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
                    proc.wait()
                    active.remove(job)
                    finish(job, "timeout")
                continue
            active.remove(job)
            finish(job, "ok" if rc == 0 else f"fail(rc={rc})")

    watch.stop()
    print(f"[runner] done | ok={counts['ok']} skip={n_skip} fail={counts['fail']} "
          f"timeout={counts['timeout']} | host_mem_low_water={watch.low} GB", flush=True)
    if counts["fail"] or counts["timeout"]:
        print("[runner] incomplete units present -- do NOT freeze numbers from this run.",
              flush=True)


def run_worker(args):
    import yaml

    from wgaudit.train import run_unit

    cfg, mcfg = load_cfg()
    tuned_path = ROOT / "configs" / "tuned.yaml"
    tuned = yaml.safe_load(tuned_path.read_text()) if tuned_path.exists() else None
    run_unit(args.dataset, args.method, args.seed, args.outdir, cfg, mcfg, tuned)


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--only", default="", help="dataset,method,seed -- single unit")
    ap.add_argument("--max-seeds", dest="max_seeds", type=int, default=999)
    ap.add_argument("--outdir", default="runs/dev")
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=7200)
    ap.add_argument("--jobs", type=int, default=1,
                    help="concurrent units on THIS host; set from microheavy's measured RSS")
    ap.add_argument("--no-resume", dest="no_resume", action="store_true")
    ap.add_argument("--dataset"); ap.add_argument("--method")
    ap.add_argument("--seed", type=int)
    return ap


if __name__ == "__main__":
    a = build_argparser().parse_args()
    run_worker(a) if a.worker else run_orchestrator(a)
