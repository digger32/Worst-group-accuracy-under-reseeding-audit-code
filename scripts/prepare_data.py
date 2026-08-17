#!/usr/bin/env python3
"""Decode every image ONCE, here, and never again during a run.

Four failure modes are designed out, each of which cost a real run.

DECODING (V1/A2). The datasets library decoded images inside its own generator and
PIL stalled on one truncated file upstream of the safety wrapper; worse, a model
could be scored on inputs it never received. Here nothing decodes images but
a5.data.safe_decode, and a constant image counts as a failure because it is
indistinguishable from no image at all.

HOST MEMORY (25 Jul 2026). The previous version built a Python list holding the raw
bytes of every image before decoding any of them: on a PNG mirror that is 12-29 GB,
and the host OOM killer took the process -- twice, silently, with tmux as collateral.
Images are now streamed in small Arrow row batches, so resident memory is flat and
independent of corpus size. Only the labels (a few hundred kilobytes) live in RAM.

SCHEMA MIXING. An HF repo ships several configs, each a full copy of the corpus
under a different schema. Pooling their shards makes the first one fix the schema
and the rest fail to cast. Configs are grouped and exactly one is used.

SPLIT INTEGRITY. The official CelebA partition is identity disjoint. This script
never guesses: it uses the repository's own splits when they are present, then a
filename column, then an index cut validated against the published training group
histogram, and finally an identity-disjoint split -- refusing outright if none of
these is possible.

Reading goes through pyarrow directly rather than the datasets library: it needs no
cast machinery, writes no second copy of the corpus into a cache, and gives batch
level control over memory. The prepared corpus is a frozen artifact, written once
and rewritten only under --force.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a5 import data as D  # noqa: E402

RAW = ROOT / "data" / "raw"
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SPLIT_ALIAS = {"train": "train", "test": "test", "validation": "validation",
               "valid": "validation", "val": "validation", "dev": "validation"}
# Official CelebA partition (202,599 images in filename order) and the published
# training group histogram, g = y*2 + a with y=Blond_Hair, a=Male. The histogram is
# used as a CHECKSUM on row order: a mirror without filenames whose rows are in
# filename order must reproduce it exactly.
CELEBA_BOUNDS = (162770, 182637)
OFFICIAL_SPLITS = {"train": 162770, "validation": 19867, "test": 19962}
CELEBA_TRAIN_HIST = [71629, 66874, 22880, 1387]
FILENAME_COLS = ("img_filename", "image_id", "file_name", "filename", "image_file")
IDENTITY_COLS = ("celeb_id", "identity")
VECTOR_COLS = ("attributes", "attr", "attributes_vector")
# Fixed list_attr_celeba.txt order; only needed for mirrors storing one vector.
ATTR_INDEX = {"Blond_Hair": 9, "Male": 20}
BATCH = 64


def rss_gb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 2**20, 2)
    except Exception:
        pass
    return None


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


class Source:
    """A corpus as labels in memory plus images streamed on demand.

    `y`, `a` and `split` are small arrays covering the whole corpus. Images are
    never all held at once: `iter_images` walks the backing store sequentially and
    yields only the requested indices, one at a time.
    """

    def __init__(self, y, a, split, note, paths=None, files=None, img_col=None):
        self.y = np.asarray(y, dtype=np.int8)
        self.a = np.asarray(a, dtype=np.int8)
        self.split = np.asarray(split, dtype=np.int8)
        self.note = note
        self.paths = paths          # waterbirds: one path per image
        self.files = files          # celeba: parquet shards, in corpus order
        self.img_col = img_col

    def __len__(self):
        return len(self.y)

    def iter_images(self, indices):
        wanted = np.zeros(len(self), dtype=bool)
        wanted[np.asarray(indices, dtype=np.int64)] = True
        if self.paths is not None:
            for i in np.flatnonzero(wanted):
                yield int(i), self.paths[i]
            return

        import pyarrow.parquet as pq

        pos = 0
        for f in self.files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=BATCH, columns=[self.img_col]):
                n = batch.num_rows
                if wanted[pos:pos + n].any():
                    col = batch.column(0).to_pylist()
                    for k in np.flatnonzero(wanted[pos:pos + n]):
                        yield pos + int(k), col[int(k)]
                    del col
                pos += n


# --------------------------------------------------------------------------- #
def collect_waterbirds():
    """Verified against wilds/datasets/waterbirds_dataset.py: metadata.csv carries
    y, place, img_filename and split, with split codes 0/1/2 = train/val/test."""
    root = next(RAW.glob("waterbirds_v1.0*"), None)
    if root is None or not (root / "metadata.csv").exists():
        raise SystemExit("[prep] waterbirds metadata.csv not found -- run `pipeline.sh data`")
    paths, y, a, sp = [], [], [], []
    with open(root / "metadata.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            paths.append(root / row["img_filename"])
            y.append(int(row["y"])); a.append(int(row["place"])); sp.append(int(row["split"]))
    return Source(y, a, sp, {"split_source": "wilds official metadata.csv"}, paths=paths)


def _celeba_shards(root):
    """Map {config directory -> {split -> [parquet files]}}.

    Repos usually offer several configs, each in its own subdirectory and each a
    complete copy of the corpus under a different schema. Collecting `train-*.parquet`
    across all of them and handing the list to one loader makes the first shard set
    the schema and every shard from another config fail to cast -- exactly how prep
    died earlier. Configs must never be mixed. Both `<config>/<split>-000xx.parquet`
    and `<config>/<split>/000xx.parquet` layouts are handled.
    """
    configs = {}
    for f in sorted(root.rglob("*.parquet")):
        rel = f.relative_to(root)
        tokens = []
        for part in rel.parts:
            tokens.append(part.lower())
            tokens.extend(part.lower().replace("_", "-").split("-"))
        split = next((SPLIT_ALIAS[t] for t in tokens if t in SPLIT_ALIAS), f.stem.lower())
        parts = list(rel.parts[:-1])
        if parts and SPLIT_ALIAS.get(parts[-1].lower()) == split:
            parts = parts[:-1]
        configs.setdefault("/".join(parts) or ".", {}).setdefault(split, []).append(str(f))
    return configs


def _columns_of(path):
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def _read_labels(files, tgt, spu, cols):
    """Attributes only, never images: a few hundred kilobytes for the whole corpus."""
    import pyarrow.parquet as pq

    vec = next((c for c in VECTOR_COLS if c in cols), None)
    if tgt in cols and spu in cols:
        tbl = pq.read_table(files, columns=[tgt, spu])
        y = np.asarray(tbl.column(0).to_numpy(zero_copy_only=False)).astype(np.int8)
        a = np.asarray(tbl.column(1).to_numpy(zero_copy_only=False)).astype(np.int8)
        return y, a
    if vec is not None:
        tbl = pq.read_table(files, columns=[vec])
        flat = tbl.column(0).combine_chunks().flatten().to_numpy(zero_copy_only=False)
        m = flat.reshape(-1, 40)
        # `> 0` covers mirrors encoding the attributes as -1/1 rather than 0/1.
        return ((m[:, ATTR_INDEX[tgt]] > 0).astype(np.int8),
                (m[:, ATTR_INDEX[spu]] > 0).astype(np.int8))
    raise SystemExit(f"[prep] neither {tgt}/{spu} columns nor an attribute vector; "
                     f"columns={cols}")


def _read_column(files, name):
    import pyarrow.parquet as pq

    return pq.read_table(files, columns=[name]).column(0).to_pylist()


def collect_celeba(cfg):
    root = RAW / "celeba"
    configs = _celeba_shards(root)
    if not configs:
        raise SystemExit("[prep] no CelebA parquet files -- run `pipeline.sh data`")

    tgt, spu = cfg["target_attr"], cfg["spurious_attr"]
    print(f"[prep] celeba: {len(configs)} config(s) in the snapshot", flush=True)
    usable = []
    for name, splits in sorted(configs.items()):
        cols = _columns_of(next(iter(splits.values()))[0])
        has_cols = tgt in cols and spu in cols
        has_vec = any(c in cols for c in VECTOR_COLS)
        print(f"[prep]   {name:42s} splits={sorted(splits)} "
              f"attrs={'columns' if has_cols else ('vector' if has_vec else 'NONE')}",
              flush=True)
        if set(OFFICIAL_SPLITS) <= set(splits) and (has_cols or has_vec):
            usable.append((0 if has_cols else 1, name, splits, cols))
    if not usable:
        raise SystemExit(f"[prep] no single config carries the three official splits "
                         f"together with the attributes; "
                         f"configs={ {k: sorted(v) for k, v in configs.items()} }")
    _, chosen, groups, cols = sorted(usable, key=lambda t: (t[0], t[1]))[0]
    print(f"[prep] celeba: using config '{chosen}' (configs are never mixed)", flush=True)

    # ---- A: the repository already carries the official partition -------------- #
    files, y_parts, a_parts, sp_parts, sizes = [], [], [], [], {}
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        shards = sorted(groups[name])
        yy, aa = _read_labels(shards, tgt, spu, cols)
        files.extend(shards)
        y_parts.append(yy); a_parts.append(aa)
        sp_parts.append(np.full(len(yy), code, dtype=np.int8))
        sizes[name] = int(len(yy))
    y = np.concatenate(y_parts); a = np.concatenate(a_parts); sp = np.concatenate(sp_parts)

    got = np.bincount(D.group_of(y, a)[sp == 0], minlength=4).tolist()
    exact = all(sizes[k] == v for k, v in OFFICIAL_SPLITS.items())
    note = {"split_source": "the repository's own train/validation/test splits",
            "config": chosen, "n_configs_in_snapshot": len(configs),
            "split_sizes": sizes, "sizes_match_official": bool(exact),
            "train_group_hist": got,
            "train_group_hist_matches_official": got == CELEBA_TRAIN_HIST}
    if not exact or got != CELEBA_TRAIN_HIST:
        note["warning"] = (f"split sizes {sizes} or training group histogram {got} "
                           f"differ from the official partition ({OFFICIAL_SPLITS}, "
                           f"{CELEBA_TRAIN_HIST}); Methods must state that this is not "
                           f"the published split")
    img_col = next((c for c in cols if c.lower().startswith(("image", "img"))), None)
    if img_col is None:
        raise SystemExit(f"[prep] no image column; columns={cols}")
    return Source(y, a, sp, note, files=files, img_col=img_col)


def collect_celeba_single(cfg, chosen, shards, cols):
    """Fallback for a mirror that exposes one undivided split."""
    tgt, spu = cfg["target_attr"], cfg["spurious_attr"]
    y, a = _read_labels(shards, tgt, spu, cols)
    n = len(y)
    img_col = next((c for c in cols if c.lower().startswith(("image", "img"))), None)
    fname_col = next((c for c in FILENAME_COLS if c in cols), None)
    ident_col = next((c for c in IDENTITY_COLS if c in cols), None)

    if fname_col is not None or n == 202599:
        lo, hi = CELEBA_BOUNDS
        if fname_col is not None:
            names = np.asarray([str(v) for v in _read_column(shards, fname_col)])
            rank = np.empty(n, dtype=np.int64)
            rank[np.argsort(names, kind="stable")] = np.arange(n)
        else:
            rank = np.arange(n)
        sp = np.where(rank < lo, 0, np.where(rank < hi, 1, 2)).astype(np.int8)
        got = np.bincount(D.group_of(y, a)[sp == 0], minlength=4).tolist()
        if got == CELEBA_TRAIN_HIST:
            src = (f"official index split over sorted {fname_col}" if fname_col else
                   "official index split; row order VERIFIED by the published "
                   "training group histogram")
            return Source(y, a, sp, {"split_source": src, "config": chosen,
                                     "train_group_hist": got},
                          files=shards, img_col=img_col)
        print(f"[prep] index split rejected: training group histogram {got} != official "
              f"{CELEBA_TRAIN_HIST}", flush=True)

    if ident_col is None:
        raise SystemExit(f"[prep] this mirror carries neither the official splits, nor "
                         f"a filename column, nor an identity column, so no "
                         f"leakage-safe split can be built. columns={cols}")
    ids = np.asarray(_read_column(shards, ident_col))
    uniq = np.unique(ids)
    rng = np.random.default_rng(20260723)
    rng.shuffle(uniq)
    n_tr, n_va = int(.8 * len(uniq)), int(.9 * len(uniq))
    code = {**{u: 0 for u in uniq[:n_tr]}, **{u: 1 for u in uniq[n_tr:n_va]},
            **{u: 2 for u in uniq[n_va:]}}
    sp = np.array([code[i] for i in ids], dtype=np.int8)
    note = {"split_source": f"identity-disjoint 80/10/10 over {ident_col}; NOT the "
                            f"official partition -- Methods must state this",
            "config": chosen, "n_identities": int(len(uniq))}
    return Source(y, a, sp, note, files=shards, img_col=img_col)


# --------------------------------------------------------------------------- #
def select(src, rule):
    """Apply the group cap BEFORE decoding, to the training split only."""
    g = D.group_of(src.y, src.a)
    keep = np.flatnonzero(src.split != 0)               # val and test kept whole
    tr = np.flatnonzero(src.split == 0)
    tr_keep = D.subsample_train(tr, g, rule) if rule else tr
    return (np.sort(np.concatenate([keep, tr_keep])),
            {"n_train_before_cap": int(len(tr)), "n_train_after_cap": int(len(tr_keep))})


def prepare(dataset, cfg, size, force):
    out = D.prepared_dir(dataset)
    if (out / "prep_manifest.json").exists() and not force:
        print(f"[prep] {dataset}: already prepared -- use --force to rebuild")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    src = collect_waterbirds() if dataset == "waterbirds" else collect_celeba(cfg)
    idx, cap_note = select(src, cfg.get("subsample"))
    crop = D.CROP[dataset]
    n_sel = len(idx)
    print(f"[prep] {dataset}: {len(src)} images, {n_sel} selected after the cap; "
          f"streaming decode to {size}x{size} (centre_crop={crop}), RSS={rss_gb()} GB",
          flush=True)

    tmp = out / "images.npy.part"
    mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(n_sel, size, size, 3))
    ys, as_, sps, quarantine, kept, seen = [], [], [], [], 0, 0
    for gidx, img_src in src.iter_images(idx):
        arr = D.safe_decode(img_src, size, crop)
        seen += 1
        if arr is None:
            quarantine.append({"index": int(gidx), "reason": "undecodable, wrong shape, "
                                                             "or constant"})
        else:
            mm[kept] = arr
            ys.append(int(src.y[gidx])); as_.append(int(src.a[gidx]))
            sps.append(int(src.split[gidx]))
            kept += 1
        if seen % 20000 == 0:
            print(f"[prep] {dataset}: {seen}/{n_sel} decoded, {len(quarantine)} "
                  f"quarantined, RSS={rss_gb()} GB", flush=True)
    mm.flush()
    del mm

    if kept == n_sel:
        tmp.rename(out / "images.npy")                  # no copy in the normal case
    else:
        final = np.memmap(out / "images.npy", dtype=np.uint8, mode="w+",
                          shape=(kept, size, size, 3))
        srcmm = np.memmap(tmp, dtype=np.uint8, mode="r", shape=(n_sel, size, size, 3))
        for i in range(0, kept, 2000):
            final[i:i + 2000] = srcmm[i:i + 2000]
        final.flush()
        del final, srcmm
        tmp.unlink()

    y = np.asarray(ys, dtype=np.int8)
    a = np.asarray(as_, dtype=np.int8)
    s = np.asarray(sps, dtype=np.int8)
    np.savez(out / "meta.npz", y=y, a=a, split=s)
    if quarantine:
        with open(out / "quarantine.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["index", "reason"])
            w.writeheader()
            w.writerows(quarantine)

    g = D.group_of(y, a)
    manifest = {
        "dataset": dataset, "image_size": size, "centre_crop": crop,
        "n_source": int(len(src)), "n_selected": int(n_sel), "n_images": int(kept),
        "n_quarantined": len(quarantine),
        "group_names": D.GROUP_NAMES[dataset],
        "group_hist_by_split": {name: np.bincount(g[s == code], minlength=4).tolist()
                                for name, code in SPLIT_CODE.items()},
        "training_set_fixed_across_seeds": True,
        "peak_rss_gb_at_end": rss_gb(),
        "images_sha256": sha256_file(out / "images.npy"),
        **src.note, **cap_note,
    }
    (out / "prep_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if quarantine:
        print(f"[prep] {len(quarantine)} image(s) quarantined -> {out/'quarantine.csv'}. "
              f"A handful means the guard is working; thousands means the mirror is "
              f"damaged and must be re-fetched.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    grid = yaml.safe_load((ROOT / "configs" / "grid.yaml").read_text())
    size = args.size or grid["image_size"]
    names = args.datasets.split(",") if args.datasets else list(grid["datasets"])
    rc = 0
    for name in names:
        rc |= prepare(name, grid["datasets"][name], size, args.force)
    sys.exit(rc)
