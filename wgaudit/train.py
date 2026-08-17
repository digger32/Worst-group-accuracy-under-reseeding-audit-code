"""One unit = dataset x method x seed, trained and evaluated in its own process.

Every method shares the backbone, epoch count, batch size, optimiser and the model
selection rule (worst-group accuracy on validation). That parity is the point of
the paper: a debiasing gain measured against an under-tuned ERM is not a gain.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

from wgaudit import data as D
from wgaudit.metrics import score_all


# --------------------------------------------------------------------------- #
def peak_rss_gb():
    """Self-reported high-water RSS. The orchestrator cannot read this after the
    child exits, so the child records it -- this is the number that tells us
    whether the full grid fits in 32 GB."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 2**20, 3)
    except Exception:
        pass
    return None


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_backbone(name, n_classes, device):
    import torch
    import torchvision

    fn = {"resnet18": torchvision.models.resnet18,
          "resnet50": torchvision.models.resnet50}[name]
    model = fn(weights="IMAGENET1K_V1")  # cached by scripts/download_data.py
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


def grad_reverse(x, lam):
    import torch

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            return inp.view_as(inp)

        @staticmethod
        def backward(ctx, grad):
            return -lam * grad

    return _F.apply(x)


# --------------------------------------------------------------------------- #
def _features(model, x):
    import torch

    m = model
    z = m.conv1(x); z = m.bn1(z); z = m.relu(z); z = m.maxpool(z)
    z = m.layer1(z); z = m.layer2(z); z = m.layer3(z); z = m.layer4(z)
    z = m.avgpool(z)
    return torch.flatten(z, 1)


def _epoch_pass(model, loader, opt, cfg, mcfg, state, device, amp_dtype, adv_head):
    import torch
    import torch.nn.functional as F

    model.train()
    kind = mcfg["kind"]
    n_backoff = 0
    for x, y, g in loader:
        x, y, g = x.to(device, non_blocking=True), y.to(device), g.to(device)
        try:
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                z = _features(model, x)
                logits = model.fc(z)
                per_sample = F.cross_entropy(logits, y, reduction="none")

                if kind == "erm":
                    loss = per_sample.mean()
                elif kind == "reweight":
                    w = state["group_weights"][g]
                    loss = (per_sample * w).sum() / w.sum().clamp_min(1e-8)
                elif kind == "gdro":
                    q = state["q"]
                    losses_g = torch.zeros_like(q)
                    for gi in range(len(q)):
                        m = g == gi
                        if m.any():
                            losses_g[gi] = per_sample[m].mean()
                    with torch.no_grad():
                        q = q * torch.exp(mcfg["eta"] * losses_g.detach())
                        q = q / q.sum()
                        state["q"] = q
                    loss = (q * losses_g).sum()
                elif kind == "adversarial":
                    a = (g % 2).long()
                    lam = mcfg["lambda_adv"] * state["adv_ramp"]
                    loss = per_sample.mean() + F.cross_entropy(
                        adv_head(grad_reverse(z, lam)), a)
                else:  # dfr trains an ERM base, then replaces the last layer
                    loss = per_sample.mean()

            # Backward and step live OUTSIDE the autocast region, as the AMP
            # documentation requires; only the forward pass is autocast.
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        except getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError):
            # Deliberately fatal. Silently skipping a batch would change the
            # training set mid-run and destroy the compute-matching the whole
            # paper rests on. The unit dies, the orchestrator records it, and the
            # batch size is lowered in grid.yaml for the WHOLE grid, not for one
            # unlucky unit.
            torch.cuda.empty_cache()
            n_backoff += 1
            raise
    return n_backoff


def evaluate(model, loader, device, amp_dtype):
    import torch

    model.eval()
    ys, ps, gs = [], [], []
    with torch.no_grad():
        for x, y, g in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model.fc(_features(model, x))
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(np.asarray(y)); gs.append(np.asarray(g))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)


def _embed(model, loader, device):
    import torch

    model.eval()
    zs, ys, gs = [], [], []
    with torch.no_grad():
        for x, y, g in loader:
            zs.append(_features(model, x.to(device)).float().cpu().numpy())
            ys.append(np.asarray(y)); gs.append(np.asarray(g))
    return np.concatenate(zs), np.concatenate(ys), np.concatenate(gs)


# --------------------------------------------------------------------------- #
def run_unit(dataset, method, seed, outdir, cfg, methods_cfg, tuned=None):
    import time

    import torch

    t0 = time.time()
    dcfg = cfg["datasets"][dataset]
    mcfg = dict(methods_cfg["methods"][method])
    if tuned:
        over = dict(tuned.get(dataset, {}).get(method, {}) or {})
        # C, eta and lambda_adv belong to the METHOD, not the dataset. Merging them
        # into dcfg would leave them silently unread by the method that needs them.
        mcfg.update({k: over.pop(k) for k in ("C", "eta", "lambda_adv")
                     if k in over})
        dcfg = {**dcfg, **over}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg.get("amp") or "off"
    if amp == "fp16":
        # fp16 training needs a GradScaler; without one the gradients underflow
        # silently and the audit would measure the scaler, not the method.
        raise SystemExit("amp: fp16 is not supported -- use bf16 (Ampere and later) or off")
    amp_dtype = {"bf16": torch.bfloat16, "off": None}[amp]
    seed_everything(seed)

    images, meta, manifest = D.load_prepared(dataset)
    groups = D.group_of(meta["y"], meta["a"])
    # The group cap was applied once at prepare time, so every seed and every
    # method trains on byte-identical data; train_sha256 below proves it.
    tr = D.split_indices(meta, "train")
    # Validation is split once, deterministically, STRATIFIED BY GROUP and identically
    # for every method and seed:
    #   va      -- epoch selection for ALL methods, and the score the tuner reads
    #   va_dfr  -- the held-out rows DFR is allowed to fit its head on
    # The split exists because DFR must be SCORED on data it did not fit on. Without
    # it, DFR's reported validation number is its ERM base's, so the tuner spends
    # DFR's whole budget optimising a proxy -- which is exactly what the 25 Jul tune
    # log showed (DFR and ERM agreeing to four decimals on Waterbirds).
    # Stratifying keeps both halves at the same group proportions, so the worst-group
    # metric has the same granularity in each.
    va_all = D.split_indices(meta, "val")
    g_all = D.group_of(meta["y"], meta["a"])
    rs = np.random.default_rng(20260723)
    sel, fit = [], []
    for grp in np.unique(g_all[va_all]):
        gi = va_all[g_all[va_all] == grp]
        perm = rs.permutation(len(gi))
        half = len(gi) // 2
        sel.append(gi[perm[:half]]); fit.append(gi[perm[half:]])
    va = np.sort(np.concatenate(sel))
    va_dfr = np.sort(np.concatenate(fit))
    te = D.split_indices(meta, "test")

    fp = {**D.eval_fingerprint(images, te), **D.train_fingerprint(tr, groups)}
    if fp["min_probe_std"] <= 0.0:
        raise RuntimeError("eval probe contains a constant image -- inputs are not "
                           "reaching the model; refusing to produce a number")

    mk = lambda idx, train: D.make_loader(
        D.A5Dataset(dataset, idx, meta, images, train, seed), cfg["batch_size"],
        train, seed, cfg["num_workers"], cfg["prefetch_factor"])
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    model = build_backbone(dcfg["backbone"], 2, device)
    adv_head = torch.nn.Linear(model.fc.in_features, 2).to(device) \
        if mcfg["kind"] == "adversarial" else None
    params = list(model.parameters()) + (list(adv_head.parameters()) if adv_head else [])
    opt = torch.optim.SGD(params, lr=dcfg["lr"], momentum=0.9,
                          weight_decay=dcfg["weight_decay"])

    counts = np.bincount(groups[tr], minlength=dcfg["n_groups"]).astype(np.float32)
    state = {"q": torch.ones(dcfg["n_groups"], device=device) / dcfg["n_groups"],
             "group_weights": torch.tensor(
                 (counts.sum() / np.maximum(counts, 1)), device=device,
                 dtype=torch.float32),
             "adv_ramp": 0.0}

    best = {"worst": -1.0, "state": None}
    backoffs = 0
    for ep in range(int(dcfg["epochs"])):
        dl_tr.dataset.set_epoch(ep)
        state["adv_ramp"] = min(1.0, (ep + 1) / max(
            1, mcfg.get("grl_warmup_frac", 0.2) * dcfg["epochs"]))
        backoffs += _epoch_pass(model, dl_tr, opt, cfg, mcfg, state, device,
                                amp_dtype, adv_head)
        yv, pv, gv = evaluate(model, dl_va, device, amp_dtype)
        worst = score_all(yv, pv, gv, dcfg["n_groups"])["acc_worst_group"]
        if worst > best["worst"]:
            best = {"worst": worst,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])

    extra = {"val_worst_group_base": best["worst"]}
    val_score = best["worst"]
    if mcfg["kind"] == "dfr":
        model, dfr_info = _dfr_last_layer(model, mk, va_dfr, meta, images, seed, device,
                                          mcfg)
        extra.update(dfr_info)
        # Re-score on va, which DFR did NOT fit on. This -- not the base's number --
        # is what the tuner and the paper must read for DFR.
        yv, pv, gv = evaluate(model, dl_va, device, amp_dtype)
        val_score = score_all(yv, pv, gv, dcfg["n_groups"])["acc_worst_group"]

    yt, pt, gt = evaluate(model, dl_te, device, amp_dtype)
    result = {
        "dataset": dataset, "method": method, "seed": seed, "device": device,
        "backbone": dcfg["backbone"], "epochs": int(dcfg["epochs"]),
        "batch_size": cfg["batch_size"], "lr": dcfg["lr"],
        # the complete frozen configuration, so a run can be audited against
        # configs/tuned.yaml rather than trusted
        "hyperparams": {"lr": dcfg["lr"], "weight_decay": dcfg["weight_decay"],
                        **{k: mcfg[k] for k in ("C", "eta", "lambda_adv")
                           if k in mcfg}},
        "hpo_trials": methods_cfg["hpo_trials"],
        "selection_rule": "val_worst_group_accuracy",
        "metrics": score_all(yt, pt, gt, dcfg["n_groups"]),
        "val_worst_group": val_score,
        "oom_backoffs": backoffs,
        "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                         if device == "cuda" else None),
        "peak_rss_gb": peak_rss_gb(),
        "wall_s": round(time.time() - t0, 1),
        "n_val_selection": int(len(va)), "n_val_dfr": int(len(va_dfr)),
        **extra, **fp,
    }
    out = Path(outdir) / f"{dataset}__{method}__seed{seed}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.rename(out)  # atomic
    return result


def _dfr_last_layer(model, mk, idx_dfr, meta, images, seed, device, mcfg):
    """Retrain ONLY the final linear layer on group-balanced held-out data.

    Two things here are easy to get wrong and were got wrong first time round.

    (a) A binary LogisticRegression has coef_ of shape (1, d), but the head has two
        output rows. Copying the same row twice makes the logit difference constant,
        i.e. a classifier that always predicts one class -- which reads as "DFR does
        not work" rather than as a bug. The correct split of a binary model over two
        logits is (-w/2, +w/2) with bias (-b/2, +b/2); softmax is shift-invariant, so
        this reproduces the sklearn decision function exactly.

    (b) DFR standardises features before fitting. Rather than carry a scaler around,
        it is folded into the layer analytically: w.z_scaled + b = (w/s).z + (b - w.mu/s).
    """
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    groups = D.group_of(meta["y"], meta["a"])
    rng = np.random.default_rng(20260723 + seed)
    n = min(int((groups[idx_dfr] == g).sum()) for g in np.unique(groups[idx_dfr]))
    balanced = [rng.choice(idx_dfr[groups[idx_dfr] == g], size=n, replace=False)
                for g in np.unique(groups[idx_dfr])]
    idx = np.sort(np.concatenate(balanced))

    z, y, _ = _embed(model, mk(idx, False), device)
    scaler = StandardScaler().fit(z)
    clf = LogisticRegression(C=mcfg.get("C", 1.0), max_iter=2000).fit(scaler.transform(z), y)

    w = clf.coef_[0] / scaler.scale_
    b = float(clf.intercept_[0]) - float(np.dot(clf.coef_[0], scaler.mean_ / scaler.scale_))
    W = np.vstack([-w / 2.0, w / 2.0])
    B = np.array([-b / 2.0, b / 2.0])
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor(W, dtype=torch.float32, device=device))
        model.fc.bias.copy_(torch.tensor(B, dtype=torch.float32, device=device))

    check = (z @ W.T + B).argmax(1)
    ref = clf.predict(scaler.transform(z))
    if not np.array_equal(check, ref):
        raise RuntimeError("DFR head does not reproduce the fitted classifier")
    return model, {"dfr_n_per_group": int(n), "dfr_train_acc": float((ref == y).mean())}
