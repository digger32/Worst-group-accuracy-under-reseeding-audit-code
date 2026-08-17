#!/usr/bin/env python3
"""Decide what is noise BEFORE silencing anything.

Reads the per-unit logs of a completed stage, normalises each line (numbers, paths,
hex ids and timestamps replaced by placeholders), counts the resulting shapes, and
splits them in two: shapes that are safe to hide from the console, and shapes that
carry a risk keyword and must stay visible however often they repeat.

Output is a REPORT plus configs/log_filter.candidate.txt. Nothing is applied
automatically: promoting the candidate to configs/log_filter.txt is a human act,
and even then the raw stage log keeps every line.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = re.compile(r"error|traceback|exception|fail|nan|inf\b|oom|out of memory|"
                  r"killed|corrupt|truncat|skip|fallback|mismatch|nondeterministic|"
                  r"cuda|overflow|underflow|warn", re.I)
NORM = [(re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"), "<TS>"),
        (re.compile(r"0x[0-9a-f]+", re.I), "<HEX>"),
        (re.compile(r"/[\w./\-]+"), "<PATH>"),
        (re.compile(r"\b\d+\.\d+\b"), "<F>"),
        (re.compile(r"\b\d+\b"), "<N>")]


def shape(line):
    s = line.rstrip()
    for pat, rep in NORM:
        s = pat.sub(rep, s)
    return s.strip()


def main(stage_dir):
    logs = sorted((Path(stage_dir) / "logs").glob("*.log"))
    if not logs:
        print(f"[triage] no logs under {stage_dir}/logs -- run a stage first")
        return 1
    counts, examples = Counter(), {}
    for f in logs:
        for line in f.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            sh = shape(line)
            counts[sh] += 1
            examples.setdefault(sh, line.strip())

    total = sum(counts.values())
    risky = [(c, s) for s, c in counts.items() if RISK.search(s)]
    plain = [(c, s) for s, c in counts.items() if not RISK.search(s)]
    risky.sort(reverse=True); plain.sort(reverse=True)

    print(f"[triage] {len(logs)} logs, {total} lines, {len(counts)} distinct shapes\n")
    print("== KEEP VISIBLE (risk keyword present, whatever the count) ==")
    for c, s in risky[:25]:
        print(f"{c:8d}  {examples[s][:150]}")
    print("\n== CANDIDATES TO HIDE FROM THE CONSOLE (no risk keyword) ==")
    for c, s in plain[:25]:
        print(f"{c:8d}  {examples[s][:150]}")

    cand = ROOT / "configs" / "log_filter.candidate.txt"
    lines = ["# Candidate console filter produced by log_triage.py.",
             "# Review, then copy the lines you accept into configs/log_filter.txt.",
             "# Fixed strings, one per line; the RAW stage log is never filtered."]
    for c, s in plain:
        if c >= max(5, total // 500):
            lines.append(re.escape(examples[s][:60]).replace("\\ ", " "))
    cand.write_text("\n".join(lines) + "\n")
    print(f"\n[triage] candidate filter -> {cand} ({len(lines) - 3} shapes). "
          f"Nothing has been silenced yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot"))
