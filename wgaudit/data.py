"""Data layer.

Two lessons are designed into this file rather than patched onto it later.

(1) IMAGE DECODING. There, HF `datasets` auto-decoded images
    inside its own generator and PIL hung on one truncated file, upstream of our
    safety wrapper; and worse, models could silently receive nothing. Here every
    image is decoded EXACTLY ONCE, in prepare_data.py, under a guarded decoder,
    and written to a uint8 memmap. Training never touches PIL, so the hang class
    cannot occur during a run. The silent variant is caught by fingerprints below.

(2) HOST RAM. The memmap is read through the page cache,
    so resident memory stays bounded no matter how large the corpus is, and the
    RAM multiplier is the DataLoader worker count alone -- which is capped in
    grid.yaml and measured at micro-heavy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared"

# ImageNet statistics; both benchmarks use ImageNet-pretrained backbones.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Group ids are (y, a) in row-major order, so for both datasets:
#   0 = (y=0, a=0)   1 = (y=0, a=1)   2 = (y=1, a=0)   3 = (y=1, a=1)
# CelebA: y=blond, a=male  ->  group 3 = (blond, male), the minority group.
GROUP_NAMES = {
    "waterbirds": ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"],
    "celeba": ["non-blond/female", "non-blond/male", "blond/female", "blond/male"],
}


# --------------------------------------------------------------------------- #
# guarded decode (used ONLY by prepare_data.py)
# --------------------------------------------------------------------------- #
# Per-dataset geometry, matched to the published protocols so the numbers are
# comparable with the literature this paper audits.
#   waterbirds -- already distributed cropped and centred at 224x224; resize only.
#   celeba     -- native 178x218; the standard protocol centre-crops to 178x178
#                 first, then resizes. Resizing the raw frame instead distorts the
#                 aspect ratio and quietly changes every published number.
CROP = {"waterbirds": None, "celeba": 178}


def safe_decode(src, size, centre_crop=None):
    """Decode one image (path or raw bytes) to a uint8 HWC array, or return None.

    Truncated files are tolerated rather than fatal, and the decode is forced
    eagerly (`img.load()`) inside the guard so a corrupt file raises here, in
    the prepare stage, instead of stalling a training unit hours later.
    """
    import io

    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if isinstance(src, dict):
        # HF stores an undecoded image as {"bytes": ..., "path": ...}; either field
        # may be the live one depending on how the mirror was built.
        src = src.get("bytes") or src.get("path")
        if src is None:
            return None
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.load()
            if centre_crop:
                w, h = img.size
                c = min(centre_crop, w, h)
                left, top = (w - c) // 2, (h - c) // 2
                img = img.crop((left, top, left + c, top + c))
            img = img.resize((size, size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
    except Exception:
        return None
    if arr.shape != (size, size, 3) or arr.std() == 0:
        return None  # a constant image is indistinguishable from "no image at all"
    return arr


# --------------------------------------------------------------------------- #
# prepared corpus
# --------------------------------------------------------------------------- #
def prepared_dir(dataset):
    return PREPARED / dataset


def load_prepared(dataset):
    """Return (memmap images NxSxSx3 uint8, meta dict of arrays, manifest)."""
    d = prepared_dir(dataset)
    manifest = json.loads((d / "prep_manifest.json").read_text())
    n, s = manifest["n_images"], manifest["image_size"]
    images = np.memmap(d / "images.npy", dtype=np.uint8, mode="r", shape=(n, s, s, 3))
    meta = np.load(d / "meta.npz")
    return images, {k: meta[k] for k in meta.files}, manifest


def group_of(y, a):
    return (np.asarray(y).astype(np.int64) * 2 + np.asarray(a).astype(np.int64))


def split_indices(meta, split):
    return np.flatnonzero(meta["split"] == {"train": 0, "val": 1, "test": 2}[split])


def subsample_train(idx, groups, rule, seed=0):
    """Group-aware cap applied ONCE, at prepare time, with a fixed seed.

    The training set is deliberately IDENTICAL across the 20 seeds. The variance
    this paper measures is training stochasticity -- initialisation, batch order,
    augmentation -- which is what single-seed papers hide. Resampling the training
    data per seed would fold a second, different source of variance into the same
    error bar and make the claim unattributable. It also matches the protocol of
    the work being audited, where the training set is fixed.
    """
    if not rule:
        return idx
    rng = np.random.default_rng(20260723 + seed)
    keep_whole = set(rule.get("keep_whole") or [])
    cap = int(rule["cap_per_group"])
    out = []
    for g in np.unique(groups[idx]):
        gi = idx[groups[idx] == g]
        if int(g) in keep_whole or len(gi) <= cap:
            out.append(gi)
        else:
            out.append(rng.choice(gi, size=cap, replace=False))
    return np.sort(np.concatenate(out))


# --------------------------------------------------------------------------- #
# deterministic augmentation
# --------------------------------------------------------------------------- #
def _augment(img_u8, rng):
    """Random resized crop + horizontal flip, driven by an explicit RNG.

    The RNG is derived from (seed, epoch, sample index), so the augmentation a
    sample receives does not depend on worker count, prefetch order or dataloader
    internals. That is what makes a 20-seed audit reproducible.
    """
    s = img_u8.shape[0]
    scale = rng.uniform(0.7, 1.0)
    side = max(8, int(round(s * np.sqrt(scale))))
    top = rng.integers(0, s - side + 1)
    left = rng.integers(0, s - side + 1)
    crop = img_u8[top:top + side, left:left + side]
    if side != s:
        yi = np.linspace(0, side - 1, s).astype(np.int64)
        crop = crop[yi][:, yi]
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    return np.ascontiguousarray(crop)


def normalise(img_u8):
    x = img_u8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# --------------------------------------------------------------------------- #
# torch dataset
# --------------------------------------------------------------------------- #
class A5Dataset:
    """Indexable (x, y, g) view over the prepared memmap. torch is imported lazily
    so that prepare/aggregate/gate scripts can use this module without CUDA."""

    def __init__(self, dataset, indices, meta, images, train, seed=0):
        self.images, self.indices, self.train, self.seed = images, np.asarray(indices), train, seed
        self.y = meta["y"][self.indices].astype(np.int64)
        self.g = group_of(meta["y"], meta["a"])[self.indices].astype(np.int64)
        self.epoch = 0

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, i):
        import torch

        raw = np.asarray(self.images[self.indices[i]])
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, int(self.indices[i])))
            raw = _augment(raw, rng)
        x = torch.from_numpy(normalise(raw))
        return x, int(self.y[i]), int(self.g[i])


def make_loader(ds, batch_size, shuffle, seed, num_workers, prefetch_factor):
    import torch
    from torch.utils.data import DataLoader

    gen = torch.Generator()
    gen.manual_seed(seed)
    kw = {}
    if num_workers > 0:
        kw = {"prefetch_factor": prefetch_factor, "persistent_workers": False}
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      num_workers=num_workers, drop_last=False, pin_memory=False, **kw)


# --------------------------------------------------------------------------- #
# input fingerprints -- the invariant that catches a silent input bug
# --------------------------------------------------------------------------- #
def eval_fingerprint(images, indices, n_probe=64):
    """sha256 over the normalised tensors of a fixed probe slice of the eval split.

    Depends on the dataset ONLY, so it must be byte-identical across every unit of
    that dataset. If one method's units disagree, that method was fed different
    pixels -- the silent failure in which a model scores a benchmark it
    never actually received.
    """
    h = hashlib.sha256()
    probe = np.asarray(indices)[:n_probe]
    stds = []
    for i in probe:
        x = normalise(np.asarray(images[i]))
        h.update(np.round(x, 4).astype(np.float32).tobytes())
        stds.append(float(x.std()))
    return {"eval_sha256": h.hexdigest(), "n_probe": int(len(probe)),
            "min_probe_std": round(min(stds), 6) if stds else 0.0}


def train_fingerprint(indices, groups):
    """sha256 over the seeded train index order and its group histogram: catches a
    subsample or shuffle that silently differs between methods on the same seed."""
    h = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes())
    hist = np.bincount(np.asarray(groups)[np.asarray(indices)], minlength=4).tolist()
    return {"train_sha256": h.hexdigest(), "train_group_hist": hist,
            "n_train": int(len(indices))}
