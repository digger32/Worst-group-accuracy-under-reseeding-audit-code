#!/usr/bin/env python3
"""Probe the CelebA mirrors BEFORE downloading gigabytes.

`pipeline.sh check` answers "is it reachable". This answers the question that
actually decides the Methods section: does the mirror let us reconstruct the
OFFICIAL partition, or are we forced into a disclosed substitute?

Four outcomes, in descending order of preference:
  A  the repo already ships the official splits (train/validation/test with
     162,770 / 19,867 / 19,962 rows)     -> use them directly, nothing to reconstruct
  B  a filename column exists            -> official split by sorted filename
  C  one 202,599-row split, and an index cut reproduces the official training group
     histogram                           -> official split, row order VERIFIED
  D  none of the above                   -> identity-disjoint split, disclosed

Only metadata and a couple of streamed chunks cross the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIRRORS = ["flwrlabs/celeba", "eurecom-ds/celeba", "tpremoli/CelebA-attrs"]
FILENAME_COLS = ("img_filename", "image_id", "file_name", "filename", "image_file")
IDENTITY_COLS = ("celeb_id", "identity")
CELEBA_TRAIN_HIST = [71629, 66874, 22880, 1387]
OFFICIAL_SPLITS = {"train": 162770, "validation": 19867, "test": 19962}


def probe(repo):
    from datasets import load_dataset

    from datasets import load_dataset_builder

    print(f"\n=== {repo} ===")
    try:
        builder = load_dataset_builder(repo)
        splits = {k: v.num_examples for k, v in (builder.info.splits or {}).items()}
    except Exception as e:
        print(f"  metadata unavailable: {str(e)[:160]}")
        splits = {}
    print(f"  splits: {splits or 'unknown'}")

    aliases = {"valid": "validation", "val": "validation", "dev": "validation"}
    norm = {aliases.get(k, k): v for k, v in splits.items()}
    official = all(norm.get(k) == v for k, v in OFFICIAL_SPLITS.items())
    print(f"  matches the official partition "
          f"({OFFICIAL_SPLITS['train']}/{OFFICIAL_SPLITS['validation']}/"
          f"{OFFICIAL_SPLITS['test']}): {official}")

    try:
        ds = load_dataset(repo, split="train", streaming=True)
    except Exception as e:
        print(f"  unavailable: {str(e)[:160]}")
        return None

    cols = list(ds.features)
    print(f"  columns ({len(cols)}): {cols[:14]}{' ...' if len(cols) > 14 else ''}")
    fname = next((c for c in FILENAME_COLS if c in cols), None)
    ident = next((c for c in IDENTITY_COLS if c in cols), None)
    has_attrs = all(c in cols for c in ("Blond_Hair", "Male"))
    print(f"  Blond_Hair + Male as columns: {has_attrs}")
    print(f"  filename column: {fname}   identity column: {ident}")

    n = norm.get("train")

    try:
        row = next(iter(ds.take(1)))
        img = row[next(c for c in cols if c.lower().startswith("image"))]
        print(f"  first row image field: {type(img).__name__}")
    except Exception as e:
        print(f"  could not read a row: {str(e)[:120]}")

    if official:
        verdict = ("A -- the repo ships the official partition; prep will use its own "
                   "splits and verify the training group histogram")
    elif fname:
        verdict = "B -- official split by sorted filename"
    elif n == 202599:
        verdict = ("C? -- one full-corpus split; prep will accept an index cut ONLY if "
                   "the official group histogram matches exactly")
    elif ident:
        verdict = "D -- identity-disjoint split, must be disclosed in Methods"
    else:
        verdict = "UNUSABLE -- no leakage-safe split can be built"
    print(f"  verdict: {verdict}")
    if not has_attrs:
        print("  NOTE: attributes are not plain columns here; prepare_data expects "
              "Blond_Hair and Male as columns")
    return verdict


if __name__ == "__main__":
    repos = sys.argv[1:] or MIRRORS
    for r in repos:
        probe(r)
    print(f"\nExpected official training histogram (g = y*2 + a, y=Blond_Hair, a=Male): "
          f"{CELEBA_TRAIN_HIST}")
    print("Prefer the first mirror whose verdict is A, then B, then C, then D.")
