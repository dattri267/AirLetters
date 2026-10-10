"""Phase 4-5: train one trajectory classifier on the GPU and write a JSON result + checkpoint + predictions.

  python src/train.py --packed data/packed.npz --name cnn_xy_T64 --model cnn1d --dims 2 --T 64 --seed 0
  add --eval_test ONLY for the final frozen configuration.

All data lives on the GPU as tensors (trajectories are tiny), augmentation runs on the GPU, mixed precision is on
by default -> an epoch over ~100k samples takes seconds on a 6 GB laptop card.
"""
import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from features import DEFAULT_FEAT, JOINT_SETS, get_split_features, load_packed  # noqa: E402
from models import build, n_params  # noqa: E402


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def augment(geo, d, rot_deg=15.0, scale=0.15, shift=0.05, noise=0.01):
    """geo (B,T,n_geo) -> same, random rotation (in the xy-plane), scale, shift, jitter. Per-sample parameters."""
    B, T, G = geo.shape
    J = G // d
    g = geo.view(B, T, J, d)
    th = (torch.rand(B, device=geo.device) * 2 - 1) * math.radians(rot_deg)
    c, s = torch.cos(th), torch.sin(th)
    x, y = g[..., 0], g[..., 1]
    nx = c[:, None, None] * x - s[:, None, None] * y
    ny = s[:, None, None] * x + c[:, None, None] * y
    g = torch.cat([nx[..., None], ny[..., None], g[..., 2:]], -1)
    sc = 1 + (torch.rand(B, 1, 1, 1, device=geo.device) * 2 - 1) * scale
    sh = (torch.rand(B, 1, 1, d, device=geo.device) * 2 - 1) * shift
    g = g * sc + sh + noise * torch.randn_like(g)
    return g.reshape(B, T, G)


def assemble(xb, n_geo, d, vel, mask_ch, aug):
    geo = xb[..., :n_geo]
    if aug:
        geo = augment(geo, d)
    parts = [geo]
    if vel:
        parts.append(F.pad(geo[:, 1:] - geo[:, :-1], (0, 0, 1, 0)))
    if mask_ch:
        parts.append(xb[..., n_geo:])
    return torch.cat(parts, -1)


@torch.no_grad()
def predict(model, X, n_geo, d, vel, mask_ch, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = assemble(X[i:i + bs], n_geo, d, vel, mask_ch, False)
        out.append(model(xb).float())
    return torch.cat(out)


@torch.no_grad()
def latency_ms(model, T, C, dev, n=200, bs=1):
    model.eval()
    x = torch.randn(bs, T, C, device=dev)
    for _ in range(30):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def run(cfg):
    t_start = time.time()
    set_seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() and not cfg["cpu"] else "cpu")
    if dev.type == "cpu":
        print("WARNING: training on CPU. Run  python src/check_env.py  to fix your CUDA install.")
    torch.backends.cudnn.benchmark = True
    feat_cfg = {k: cfg[k] for k in DEFAULT_FEAT}
    pk = load_packed(cfg["packed"])
    classes = [str(c) for c in pk["classes"]]
    n_cls = len(classes)
    cache = str(Path(cfg["packed"]).parent / "cache")
    sets = {sp: get_split_features(pk, sp, feat_cfg, cache) for sp in ("train", "val", "test")}
    if sets["train"] is None or sets["val"] is None:
        raise SystemExit("The packed file has no train or val videos. Extraction has not produced landmark files yet "
                         "(or --lm_dir in 'features.py pack' points to the wrong folder). Check that the folder contains .npz files, "
                         "then run 'features.py pack ... --drop_missing' again.")
    Xtr, ytr, qtr, itr = sets["train"]
    if cfg["train_frac"] < 1.0:  # stratified subset (fixed seed so every run uses the same subset)
        rng = np.random.RandomState(1234)
        keep = np.concatenate([rng.permutation(np.flatnonzero(ytr == c))[: max(1, int(cfg["train_frac"] * (ytr == c).sum()))]
                               for c in range(n_cls)])
        Xtr, ytr, qtr, itr = Xtr[keep], ytr[keep], qtr[keep], itr[keep]
    keep = ytr >= 0
    Xtr, ytr = Xtr[keep], ytr[keep]
    d = cfg["dims"]
    n_geo = len(JOINT_SETS[cfg["joints"]]) * d
    C = n_geo * (2 if cfg["vel"] else 1) + int(cfg["mask_ch"])
    T = cfg["T"]
    tt = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    Xtr_t, ytr_t = tt(Xtr), tt(ytr, torch.long)
    Xva, yva, qva, iva = sets["val"]
    Xva_t = tt(Xva)

    model = build(cfg["model"], T, C, n_cls, cfg["width"], cfg["dropout"]).to(dev)
    npar = n_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    spe = math.ceil(len(Xtr_t) / cfg["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=spe)
    use_amp = dev.type == "cuda" and cfg["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"[{cfg['name']} s{cfg['seed']}] dev={dev} model={cfg['model']} params={npar:,} train={len(Xtr_t)} val={len(Xva_t)} C={C} T={T}")

    best, best_state, hist = -1.0, None, []
    t_train = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        tot = 0.0
        for i in range(0, len(perm), cfg["bs"]):
            b = perm[i:i + cfg["bs"]]
            if len(b) < 2:
                continue
            xb = assemble(Xtr_t[b], n_geo, d, cfg["vel"], cfg["mask_ch"], cfg["aug"])
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = crit(model(xb), ytr_t[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(b)
        pv = predict(model, Xva_t, n_geo, d, cfg["vel"], cfg["mask_ch"]).argmax(1).cpu().numpy()
        va = float((pv == yva).mean())
        hist.append((ep, tot / len(Xtr_t), va))
        if va > best:
            best, best_state = va, copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == cfg["epochs"] - 1:
            print(f"  ep {ep:3d} loss {tot/len(Xtr_t):.4f} val_acc {va:.4f} best {best:.4f}")
    train_time = time.time() - t_train
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0

    model.load_state_dict(best_state)
    res = dict(cfg=cfg, params=npar, n_train=int(len(Xtr_t)), train_time_s=round(train_time, 1), peak_gpu_mem_mb=round(peak_mb, 1),
               device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", torch=torch.__version__,
               python=platform.python_version(), epochs_hist=hist)
    logits_v = predict(model, Xva_t, n_geo, d, cfg["vel"], cfg["mask_ch"])
    pv = logits_v.argmax(1).cpu().numpy()
    ok = yva >= 0
    res["val_acc"] = float((pv[ok] == yva[ok]).mean())
    res["val_macro_f1"] = float(f1_score(yva[ok], pv[ok], average="macro"))
    preds = dict(val_true=yva, val_pred=pv, val_q=qva, val_idx=iva, val_prob=torch.softmax(logits_v, 1).cpu().numpy().astype(np.float16))
    if cfg["eval_test"] and sets["test"] is not None:
        Xte, yte, qte, ite = sets["test"]
        if (yte >= 0).all():
            pt = predict(model, tt(Xte), n_geo, d, cfg["vel"], cfg["mask_ch"]).argmax(1).cpu().numpy()
            res["test_acc"] = float((pt == yte).mean())
            res["test_macro_f1"] = float(f1_score(yte, pt, average="macro"))
            preds.update(test_true=yte, test_pred=pt, test_q=qte, test_idx=ite)
        else:
            res["test_acc"] = None
            res["test_note"] = "test labels unavailable in manifest (withheld?)"
    # efficiency (classifier forward pass only; feature extraction is timed separately in extract_landmarks.py)
    res["latency_ms_bs1_gpu"] = latency_ms(model, T, C, dev) if dev.type == "cuda" else None
    res["latency_ms_bs1_cpu"] = latency_ms(copy.deepcopy(model).cpu(), T, C, torch.device("cpu"))
    out = Path(cfg["out"])
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt" / f"{cfg['name']}_s{cfg['seed']}.pt"
    torch.save(dict(state=model.state_dict(), cfg=cfg, classes=classes, C=C, T=T), ck)
    res["ckpt_bytes"] = os.path.getsize(ck)
    res["wall_s"] = round(time.time() - t_start, 1)
    (out / f"{cfg['name']}_s{cfg['seed']}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / f"{cfg['name']}_s{cfg['seed']}_preds.npz", **preds)
    print(f"  => val_acc {res['val_acc']:.4f} macroF1 {res['val_macro_f1']:.4f} params {npar:,} "
          f"train {train_time:.0f}s peakmem {peak_mb:.0f}MB lat(gpu/cpu) {res['latency_ms_bs1_gpu']}/{res['latency_ms_bs1_cpu']:.2f} ms")
    return res


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="run")
    ap.add_argument("--model", default="cnn1d", choices=["mlp", "cnn1d", "bigru"])
    ap.add_argument("--T", type=int, default=DEFAULT_FEAT["T"])
    ap.add_argument("--dims", type=int, default=DEFAULT_FEAT["dims"], choices=[2, 3])
    ap.add_argument("--joints", default=DEFAULT_FEAT["joints"], choices=list(JOINT_SETS))
    ap.add_argument("--norm", default=DEFAULT_FEAT["norm"], choices=["raw", "center", "norm"])
    ap.add_argument("--no_mask_ch", dest="mask_ch", action="store_false")
    ap.add_argument("--vel", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_aug", dest="aug", action="store_false")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--eval_test", action="store_true")
    return vars(ap.parse_args(argv))


if __name__ == "__main__":
    run(parse())
