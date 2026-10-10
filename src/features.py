"""Phase 3: landmarks -> packed split file -> fixed-length trajectory features.

  python src/features.py pack --manifest data/manifest.csv --lm_dir data/lm --out data/packed.npz

`video_to_feature` is shared by training and the live webcam demo, so the
preprocessing is identical in both (no train/serve mismatch).

Layout of a feature tensor:  (T, n_geo [+1 mask channel])
  n_geo = n_joints * dims          (tip first, always)
Velocity channels are added later in train.py (after augmentation).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TIP, WRIST = 8, 0
JOINT_SETS = {
    "tip": [TIP],
    "tip_wrist": [TIP, WRIST],
    "tip_thumb_wrist": [TIP, 4, WRIST],
    "hand": [TIP] + [j for j in range(21) if j != TIP],
}

DEFAULT_FEAT = dict(T=64, dims=2, joints="tip", norm="norm", mask_ch=True)


# --------------------------------------------------------------------------- pack
def pack(manifest, lm_dir, out, drop_missing=False):
    df = pd.read_csv(manifest, dtype={"vid": str, "actor_id": str})
    if "vid" not in df.columns:
        df["vid"] = [f"{i:06d}" for i in range(len(df))]
    df["vid"] = df["vid"].astype(str)
    if "actor_id" not in df.columns:
        df["actor_id"] = -1
    classes = sorted(df["label"].dropna().astype(str).unique())  # from the FULL manifest -> stable class ids
    cid = {c: i for i, c in enumerate(classes)}
    if drop_missing:  # extraction stopped early: use only the videos that were actually processed
        n0 = len(df)
        df = df[[(Path(lm_dir) / f"{v}.npz").exists() for v in df.vid]].reset_index(drop=True)
        print(f"drop_missing: using {len(df)} of {n0} manifest videos that have landmark files")
    lms, masks, offs, aspect, ntot, tex = [], [], [0], [], [], []
    missing = 0
    for r in df.itertuples():
        f = Path(lm_dir) / f"{r.vid}.npz"
        if f.exists():
            z = np.load(f)
            lms.append(z["lm"])
            masks.append(z["mask"])
            aspect.append(float(z["aspect"]))
            ntot.append(int(z["n_total"]))
            tex.append(float(z["t_extract"]))
            offs.append(offs[-1] + len(z["mask"]))
        else:  # keep the sample (counts as a failure) -> honest accuracy
            missing += 1
            aspect.append(1.0)
            ntot.append(0)
            tex.append(np.nan)
            offs.append(offs[-1])
    lm_all = np.concatenate(lms).astype(np.float16) if lms else np.zeros((0, 21, 3), np.float16)
    mask_all = np.concatenate(masks) if masks else np.zeros(0, bool)
    y = np.array([cid.get(str(l), -1) if pd.notna(l) else -1 for l in df["label"]], np.int64)
    np.savez_compressed(
        out, lm=lm_all, mask=mask_all, offsets=np.array(offs, np.int64), aspect=np.array(aspect, np.float32),
        n_total=np.array(ntot, np.int32), t_extract=np.array(tex, np.float32), y=y,
        split=np.array(df["split"].astype(str).tolist(), dtype="U16"), actor=np.array(df["actor_id"].astype(str).tolist(), dtype="U32"),
        vid=np.array(df["vid"].astype(str).tolist(), dtype="U32"), classes=np.array(classes, dtype="U64"),
    )
    print(f"packed {len(df)} videos ({missing} without landmark file) -> {out}")
    for sp, g in df.groupby("split"):
        print(f"  {sp}: {len(g)} videos, {g['label'].nunique()} classes, {g['actor_id'].nunique()} actors")
    tr = set(df.loc[df.split == "train", "actor_id"])
    for sp in ("val", "test"):
        ov = tr & set(df.loc[df.split == sp, "actor_id"])
        if sp in set(df.split):
            print(f"  actor overlap train/{sp}: {len(ov)}  (must be 0 for a person-independent claim)")


def load_packed(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- features
def video_to_feature(lm, mask, aspect, cfg):
    """lm (n,21,3) float, mask (n,) bool -> (T, n_geo[+1]) float32, quality in [0,1]."""
    T, d = cfg["T"], cfg["dims"]
    joints = JOINT_SETS[cfg["joints"]]
    J = len(joints)
    n_geo = J * d
    out = np.zeros((T, n_geo + (1 if cfg["mask_ch"] else 0)), np.float32)
    n = len(mask)
    if n == 0:
        return out, 0.0
    q = float(mask.mean())
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return out, q
    pts = lm[:, joints, :d].astype(np.float32).copy()
    pts[..., 0] *= aspect  # un-distort x so that x and y share one scale
    tnew = np.linspace(idx[0], idx[-1], T)  # trim idle frames with no hand at both ends
    geo = np.empty((T, J, d), np.float32)
    for j in range(J):
        for k in range(d):
            geo[:, j, k] = np.interp(tnew, idx, pts[idx, j, k])  # linear fill of detection gaps
    mode = cfg["norm"]
    if mode in ("center", "norm"):
        tip = geo[:, 0, :]
        geo = geo - tip.mean(0)
        if mode == "norm":
            rng = tip[:, :2].max(0) - tip[:, :2].min(0)
            geo = geo / max(float(rng.max()), 1e-3)
    out[:, :n_geo] = geo.reshape(T, n_geo)
    if cfg["mask_ch"]:
        out[:, -1] = np.interp(tnew, np.arange(n), mask.astype(np.float32))
    return out, q


def _key(cfg):
    s = json.dumps({k: cfg[k] for k in sorted(DEFAULT_FEAT)}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def get_split_features(pk, split, cfg, cache_dir=None):
    """Returns X (N,T,C), y (N,), quality (N,), idx (N,) for a split; cached on disk."""
    sel = np.flatnonzero(pk["split"] == split)
    if len(sel) == 0:
        return None
    cf = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cf = Path(cache_dir) / f"feat_{split}_{_key(cfg)}_{len(pk['y'])}.npz"
        if cf.exists():
            z = np.load(cf)
            return z["X"], z["y"], z["q"], z["idx"]
    offs = pk["offsets"]
    X = np.zeros((len(sel), cfg["T"], len(JOINT_SETS[cfg["joints"]]) * cfg["dims"] + int(cfg["mask_ch"])), np.float32)
    q = np.zeros(len(sel), np.float32)
    for n, i in enumerate(sel):
        a, b = offs[i], offs[i + 1]
        X[n], q[n] = video_to_feature(pk["lm"][a:b].astype(np.float32), pk["mask"][a:b], pk["aspect"][i], cfg)
    y = pk["y"][sel]
    if cf is not None:
        np.savez(cf, X=X, y=y, q=q, idx=sel)
    return X, y, q, sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--manifest", required=True)
    p.add_argument("--lm_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--drop_missing", action="store_true",
                   help="use only videos that have a landmark file (use when extraction was stopped early; state the count in the paper)")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.manifest, a.lm_dir, a.out, a.drop_missing)
