#!/usr/bin/env python3
"""Is adversarial debiasing seed-fragile, or only the configuration we froze?

Two reviewers made the same point: the frozen configuration was selected on one seed,
so its spread across seeds cannot by itself be attributed to the method. This
experiment trains the four next-best configurations from the same tuning log on five
seeds each, at the full training budget, and reads the result under a rule written
down BEFORE any unit runs.

    python scripts/adv_config_stability.py prepare            # writes the declaration
    bash runner/pipeline.sh advcfg                             # runs the 20 units
    python scripts/adv_config_stability.py report --final runs/final_<date>

`prepare` refuses to overwrite an existing declaration, and `report` refuses to read
results produced before the declaration was written. A reading rule chosen after the
numbers are known is no rule at all.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "advcfg"
GEN = ROOT / "latex" / "generated"
SEEDS = [0, 1, 2, 3, 4]
DATASET, METHOD = "celeba", "adv"

# The frozen configuration and the four next-best trials of the seed-0 tuning run
# on CelebA, at the precision the tuning log records (three significant figures,
# which is also the precision of the frozen configuration itself).
FROZEN = {"lr": 2.36e-3, "weight_decay": 8.11e-4, "lambda_adv": 0.113}
CONFIGS = {
    "t3": {"lr": 2.06e-2, "weight_decay": 1.08e-3, "lambda_adv": 0.313, "tune_score": 0.3361},
    "t6": {"lr": 3.12e-4, "weight_decay": 1.10e-3, "lambda_adv": 0.299, "tune_score": 0.3187},
    "t4": {"lr": 9.97e-4, "weight_decay": 8.15e-4, "lambda_adv": 0.0115, "tune_score": 0.3187},
    "t2": {"lr": 2.98e-4, "weight_decay": 6.12e-4, "lambda_adv": 0.0221, "tune_score": 0.3077},
}
THRESHOLD = 0.10
HIGH_LAMBDA = ["t3", "t6"]           # lambda >= 0.1, other than the frozen one
# The sentence the paper will print is also fixed in advance, one per outcome.
RESULT_TEXT = {
    "method": (r"Both did (standard deviations \AdvCfgSDThree{} and \AdvCfgSDSix), while "
               r"the two configurations with near-zero adversarial weight stayed near the "
               r"baseline's spread (\AdvCfgSDTwo{} and \AdvCfgSDFour), so the fragility "
               r"belongs to the adversarial regime of the method and not only to the "
               r"configuration we froze."),
    "configuration": (r"Neither did (standard deviations \AdvCfgSDThree{} and "
                      r"\AdvCfgSDSix), so the spread reported above belongs to the "
                      r"configuration selected at seed 0, and it was produced by the "
                      r"standard procedure of tuning once and training once."),
    "mixed": (r"Exactly one did (standard deviations \AdvCfgSDThree{} and "
              r"\AdvCfgSDSix), so the experiment does not separate the two readings, and "
              r"we report the attribution as unresolved."),
}
SELECTION_TEXT = (r"Tuning on each of the five seeds in turn would have selected the frozen "
                  r"configuration on \AdvCfgFrozenSelected{} of them.")
MACRO = {"t3": "Three", "t6": "Six", "frozen": "Frozen", "t2": "Two", "t4": "Four"}


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def prepare(force=False):
    decl_path = OUT / "DECLARATION.json"
    if decl_path.exists() and not force:
        raise SystemExit(f"[advcfg] {decl_path} exists. The declaration is written once, "
                         f"before any result exists; it is not rewritten.")
    tuned = yaml.safe_load((ROOT / "configs" / "tuned.yaml").read_text())
    got = tuned[DATASET][METHOD]
    for k, v in FROZEN.items():
        if abs(float(got[k]) - v) > 1e-12:
            raise SystemExit(f"[advcfg] configs/tuned.yaml holds {k}={got[k]} for "
                             f"{DATASET}/{METHOD}, expected {v}; this is not the frozen "
                             f"configuration the paper reports")
    for tag, c in CONFIGS.items():
        d = OUT / tag
        if any(d.glob("*seed*.json")) and not force:
            raise SystemExit(f"[advcfg] {d} already holds results; start from empty dirs")
        d.mkdir(parents=True, exist_ok=True)
        t = copy.deepcopy(tuned)
        t[DATASET][METHOD] = {k: v for k, v in c.items() if k != "tune_score"}
        (d / "tuned.yaml").write_text(yaml.safe_dump(t, sort_keys=True))

    body = {
        "question": "does the seed spread of adversarial debiasing on CelebA belong to the "
                    "method or to the configuration selected at seed 0?",
        "dataset": DATASET, "method": METHOD, "seeds": SEEDS,
        "budget": "full training budget, identical to the reported pass",
        "frozen": FROZEN, "configs": CONFIGS,
        "statistic": "standard deviation over the five seeds of test worst-group accuracy",
        "threshold": THRESHOLD,
        "rule": {
            "method": "both configurations with lambda >= 0.1 other than the frozen one "
                      f"({', '.join(HIGH_LAMBDA)}) exceed the threshold",
            "configuration": "neither of them exceeds the threshold",
            "mixed": "exactly one of them exceeds the threshold",
        },
        "threshold_rationale": "0.10 lies at the geometric midpoint between the five-seed "
                               "standard deviation of the baseline (0.055) and of the "
                               "frozen configuration (0.190) on seeds 0-4 of the reported "
                               "pass",
        "operating_characteristics": {
            "P(SD5 > 0.10 | sigma = 0.055)": 0.010,
            "P(SD5 > 0.10 | sigma = 0.190)": 0.893,
        },
        "result_sentences": RESULT_TEXT,
        "selection_sentence": SELECTION_TEXT,
        "caveat": "t3 pairs a high adversarial weight with a high learning rate (2.06e-2), "
                  "so its outcome alone cannot separate the two; t6 pairs a high weight "
                  "with a low learning rate and is the cleaner test. t2 and t4 have "
                  "near-zero weight and act as a within-method control.",
    }
    decl = {"written_utc": datetime.now(timezone.utc).isoformat(), **body,
            "sha256_of_body": _sha(body)}
    OUT.mkdir(parents=True, exist_ok=True)
    decl_path.write_text(json.dumps(decl, indent=2))
    print(f"[advcfg] declaration written {decl['written_utc']}  sha256 "
          f"{decl['sha256_of_body'][:16]}")
    print(f"[advcfg] configuration files: {', '.join(str(OUT / t / 'tuned.yaml') for t in CONFIGS)}")
    return 0


def _units(d):
    return {json.loads(p.read_text())["seed"]: json.loads(p.read_text())
            for p in sorted(Path(d).glob(f"{DATASET}__{METHOD}__seed*.json"))}


def sci(x):
    """3.12e-04 -> $3.12\\times10^{-4}$, the notation the text uses."""
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def write_outputs(out, decl):
    """Macros and table from a finished report. Used by `report` and by `render`,
    which re-typesets an existing report without touching the runs."""
    rows, verdict = out["rows"], out["verdict"]
    GEN.mkdir(parents=True, exist_ok=True)
    L = ["% GENERATED by scripts/adv_config_stability.py -- do not edit",
         f"\\newcommand{{\\AdvCfgVerdict}}{{{verdict}}}",
         f"\\newcommand{{\\AdvCfgThreshold}}{{{out['threshold']:.2f}}}",
         f"\\newcommand{{\\AdvCfgFrozenSelected}}{{{out['frozen_selected']}}}",
         # descriptive facts from the declared table, not part of the reading rule
         f"\\newcommand{{\\AdvCfgMaxWins}}{{{max(r['wins'] for r in rows.values())}}}",
         f"\\newcommand{{\\AdvCfgERMMean}}{{{sum(out['erm_test']) / len(out['erm_test']):.3f}}}",
         f"\\newcommand{{\\AdvCfgMaxMean}}{{{max(r['mean'] for r in rows.values()):.3f}}}",
         "\\newcommand{\\AdvCfgResult}{" + decl["result_sentences"][verdict] + " "
         + decl["selection_sentence"] + "}"]
    for tag, r in rows.items():
        n = MACRO[tag]
        L += [f"\\newcommand{{\\AdvCfgSD{n}}}{{{r['sd']:.3f}}}",
              f"\\newcommand{{\\AdvCfgMean{n}}}{{{r['mean']:.3f}}}",
              f"\\newcommand{{\\AdvCfgWins{n}}}{{{r['wins']}}}",
              f"\\newcommand{{\\AdvCfgLambda{n}}}{{{r['lambda']:.3g}}}"]
    (GEN / "advcfg_macros.tex").write_text("\n".join(L) + "\n")
    T = ["% GENERATED by scripts/adv_config_stability.py -- do not edit",
         r"\begin{table}[t]", r"\centering", r"\setlength{\tabcolsep}{4pt}\small",
         r"\caption{Adversarial debiasing on CelebA: the frozen configuration and the four "
         r"next-best configurations of the same tuning run, five seeds each at the full "
         r"budget. LR is the learning rate and SD the standard deviation over the five seeds; wins count the seeds on which the "
         r"configuration beats ERM trained on the same seed. The reading rule and its "
         r"threshold of " + f"{out['threshold']:.2f}" + r" were fixed before the runs.}",
         r"\label{tab:advcfg}",
         r"\begin{tabular}{lcccc}", r"\hline",
         r"$\lambda$ & LR & Worst-group acc. & SD & Wins \\", r"\hline"]
    for tag in ("t3", "t6", "frozen", "t2", "t4"):
        r = rows[tag]
        mark = r"$^{\mathrm{a}}$" if tag == "frozen" else ""
        T.append(f"{r['lambda']:.3g}{mark} & {sci(r['lr'])} & {r['mean']:.3f} & "
                 f"{r['sd']:.3f} & {r['wins']}/5 \\\\")
    T += [r"\hline", r"\multicolumn{5}{l}{\footnotesize $^{\mathrm{a}}$Frozen configuration.}\\",
          r"\end{tabular}", r"\end{table}", ""]
    (GEN / "tab_advcfg.tex").write_text("\n".join(T))


def render():
    out = json.loads((OUT / "report.json").read_text())
    decl = json.loads((OUT / "DECLARATION.json").read_text())
    if out.get("declaration_sha256") != decl["sha256_of_body"]:
        raise SystemExit("[advcfg] report.json was produced under a different declaration")
    write_outputs(out, decl)
    print(f"[advcfg] rendered from report.json (verdict {out['verdict']}) -> {GEN}")
    return 0


def report(final):
    final = Path(final)
    decl = json.loads((OUT / "DECLARATION.json").read_text())
    body = {k: v for k, v in decl.items() if k not in ("written_utc", "sha256_of_body")}
    if _sha(body) != decl["sha256_of_body"]:
        raise SystemExit("[advcfg] the declaration was edited after it was written")
    t_decl = datetime.fromisoformat(decl["written_utc"])

    ref = {}
    for s in SEEDS:
        for m in ("adv", "erm"):
            ref[(m, s)] = json.loads((final / f"{DATASET}__{m}__seed{s}.json").read_text())
    eval_ref, train_ref = ref[("adv", 0)]["eval_sha256"], ref[("adv", 0)]["train_sha256"]

    rows, problems = {}, []
    rows["frozen"] = {"lambda": FROZEN["lambda_adv"], "lr": FROZEN["lr"],
                      "test": [ref[("adv", s)]["metrics"]["acc_worst_group"] for s in SEEDS],
                      "val": [ref[("adv", s)]["val_worst_group"] for s in SEEDS]}
    for tag, c in CONFIGS.items():
        d = OUT / tag
        meta = json.loads((d / "run_meta.json").read_text())
        if datetime.fromisoformat(meta["run_started"]) < t_decl:
            problems.append(f"{tag}: run started before the declaration was written")
        if not meta.get("no_resume"):
            problems.append(f"{tag}: run did not have resume disabled")
        u = _units(d)
        if sorted(u) != SEEDS:
            problems.append(f"{tag}: seeds {sorted(u)} != {SEEDS}")
            continue
        for s, x in u.items():
            hp = x.get("hyperparams", {})
            for k in ("lr", "weight_decay", "lambda_adv"):
                if abs(float(hp.get(k, np.nan)) - c[k]) > 1e-12:
                    problems.append(f"{tag} seed {s}: {k}={hp.get(k)} != declared {c[k]}")
            if x["eval_sha256"] != eval_ref:
                problems.append(f"{tag} seed {s}: evaluated on different inputs")
            if x["train_sha256"] != train_ref:
                problems.append(f"{tag} seed {s}: trained on a different training set")
        rows[tag] = {"lambda": c["lambda_adv"], "lr": c["lr"],
                     "test": [u[s]["metrics"]["acc_worst_group"] for s in SEEDS],
                     "val": [u[s]["val_worst_group"] for s in SEEDS]}
    if problems:
        print("[advcfg] REFUSING to report:")
        for p in problems:
            print("   -", p)
        return 1

    erm = np.array([ref[("erm", s)]["metrics"]["acc_worst_group"] for s in SEEDS])
    for tag, r in rows.items():
        t = np.array(r["test"])
        r.update(mean=float(t.mean()), sd=float(t.std(ddof=1)), lo=float(t.min()),
                 hi=float(t.max()), wins=int((t > erm).sum()))
    # which configuration a practitioner would have selected, tuning on each seed
    tags = list(rows)
    chosen = [tags[int(np.argmax([rows[t]["val"][i] for t in tags]))]
              for i in range(len(SEEDS))]
    above = {t: rows[t]["sd"] > THRESHOLD for t in rows}
    n_high = sum(above[t] for t in HIGH_LAMBDA)
    verdict = "method" if n_high == 2 else ("configuration" if n_high == 0 else "mixed")

    out = {"verdict": verdict, "threshold": THRESHOLD, "rows": rows,
           "selected_per_seed": dict(zip(map(str, SEEDS), chosen)),
           "frozen_selected": chosen.count("frozen"),
           "erm_test": erm.tolist(), "declaration_sha256": decl["sha256_of_body"]}
    (OUT / "report.json").write_text(json.dumps(out, indent=2))

    print(f"{'config':8s}{'lambda':>8s}{'lr':>10s}{'mean':>8s}{'SD5':>8s}"
          f"{'range':>16s}{'wins/5':>8s}  above {THRESHOLD}")
    for tag in ("t3", "t6", "frozen", "t2", "t4"):
        r = rows[tag]
        print(f"{tag:8s}{r['lambda']:8.4g}{r['lr']:10.3g}{r['mean']:8.3f}{r['sd']:8.3f}"
              f"{f'[{r[chr(108)+chr(111)]:.3f},{r[chr(104)+chr(105)]:.3f}]':>16s}"
              f"{r['wins']:8d}  {above[tag]}")
    print(f"\nselected when tuning on seeds {SEEDS}: {chosen}")
    print(f"VERDICT under the declared rule: {verdict}")

    write_outputs(out, decl)
    print(f"[advcfg] macros and table -> {GEN}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "report", "render"])
    ap.add_argument("--final", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.mode == "prepare":
        sys.exit(prepare(a.force))
    if a.mode == "render":
        sys.exit(render())
    if not a.final:
        ap.error("--final runs/final_<date> is required for report")
    sys.exit(report(a.final))
