#!/usr/bin/env python3
"""Acquire the corpora, and -- more importantly -- PROVE they are acquirable first.

`--check` probes every mirror and downloads nothing. Run it on day one, before any
code is written against a dataset that turns out to be gated, region-blocked or
moved. Google Drive is deliberately not a channel: it rate-limits and has broken
torchvision's own CelebA downloader for years.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# CelebA mirrors, in priority order. All carry the 40 binary attributes, so both
# Blond_Hair (target) and Male (spurious attribute) are present.
# Probed on 25 Jul 2026. Ordered by how safely the official partition can be used.
CELEBA_HF = [
    ("flwrlabs/celeba", "40 attributes as columns + celeb_id; official splits "
                        "(train/valid/test = 162770/19867/19962)"),
    ("eurecom-ds/celeba", "attributes as a length-40 vector + identity; official splits"),
    # tpremoli/CelebA-attrs is deliberately excluded: it reports validation=19962 and
    # test=19867, i.e. the two official splits swapped, and carries no identity
    # column, so neither the official partition nor a leakage-safe substitute can be
    # trusted from it.
]
TORCHVISION_WEIGHTS = "https://download.pytorch.org/models/resnet50-0676ba61.pth"


def _head(url, timeout=15):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, f"HTTP {r.status}"
    except Exception as e:
        return False, str(e)[:90]


def check():
    rows = []

    try:
        import wilds  # noqa: F401
        rows.append(("waterbirds", "wilds package", True,
                     "importable; get_dataset(download=True) fetches it"))
    except Exception as e:
        rows.append(("waterbirds", "wilds package", False, str(e)[:90]))

    try:
        from huggingface_hub import HfApi

        from datasets import load_dataset_builder

        api = HfApi()
        for repo, note in CELEBA_HF:
            try:
                api.dataset_info(repo)
                try:
                    b = load_dataset_builder(repo)
                    sp = {k: v.num_examples for k, v in (b.info.splits or {}).items()}
                except Exception:
                    sp = {}
                rows.append(("celeba", f"hf:{repo}", True, f"{note}; splits={sp or '?'}"))
            except Exception as e:
                rows.append(("celeba", f"hf:{repo}", False, str(e)[:90]))
    except Exception as e:
        rows.append(("celeba", "huggingface_hub", False, str(e)[:90]))

    ok, msg = _head(TORCHVISION_WEIGHTS)
    rows.append(("backbone", "torchvision resnet50", ok, msg))

    print(f"{'dataset':12s} {'mirror':28s} {'ok':>5s}  note")
    for d, m, ok, note in rows:
        print(f"{d:12s} {m:28s} {str(ok):>5s}  {note}")
    usable = {d for d, _, ok, _ in rows if ok}
    missing = {"waterbirds", "celeba", "backbone"} - usable
    if missing:
        print(f"\n[check] NOT reachable: {sorted(missing)} -- resolve before writing "
              f"any code against them.")
        return 1
    print("\n[check] every corpus and checkpoint is reachable.")
    print("[check] reachability is not enough for CelebA -- run `pipeline.sh probe` to "
          "see which split strategy each mirror actually permits.")
    return 0


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models"))

    print("[data] waterbirds via wilds ...")
    from wilds import get_dataset

    get_dataset(dataset="waterbirds", download=True, root_dir=str(RAW))

    print("[data] celeba via the first reachable HF mirror ...")
    from huggingface_hub import HfApi, snapshot_download

    last = None
    for repo, _ in CELEBA_HF:
        try:
            # These repos ship SEVERAL configs, each a complete copy of the corpus
            # under a different schema. Pulling all of them multiplies the download
            # for no benefit, and prep can only ever use one. Pick the leanest config
            # that still carries the attributes.
            files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                     if f.endswith(".parquet")]
            cfgs = sorted({f.rsplit("/", 1)[0] for f in files if "/" in f})
            allow = None
            if cfgs:
                pref = [c for c in cfgs if "attr" in c.lower()] or cfgs
                chosen = min(pref, key=lambda c: (len(c), c))
                allow = [f"{chosen}/*"]
                print(f"[data] {repo}: configs {cfgs} -> downloading only '{chosen}'")
            path = snapshot_download(repo_id=repo, repo_type="dataset",
                                     local_dir=str(RAW / "celeba"),
                                     allow_patterns=allow)
            (RAW / "celeba" / "SOURCE.txt").write_text(
                f"{repo}\nconfig_pattern={allow}\n")
            print(f"[data] celeba from {repo} -> {path}")
            break
        except Exception as e:
            last = e
            print(f"[data] {repo} failed: {str(e)[:160]}")
    else:
        raise SystemExit(f"[data] no CelebA mirror worked; last error: {last}")

    print("[data] backbone weights ...")
    import torchvision

    torchvision.models.resnet50(weights="IMAGENET1K_V1")
    print("[data] done. Licence note: CelebA is released for non-commercial research "
          "only; the paper's ethics paragraph must say so.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(check() if a.check else download())
