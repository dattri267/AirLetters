"""Runs the whole experiment grid (docx Stage 10) on the GPU, several seeds each, then writes results/summary.csv.

  python src/run_grid.py --packed data/packed.npz --seeds 3 --epochs 40            # dev: validation only
  python src/run_grid.py --packed data/packed.npz --only final --eval_test         # frozen final protocol

Resumable: a run whose JSON already exists is skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train import parse, run  # noqa: E402

# name -> overrides.  Groups: base (model ladder), dims, T, norm, joints, extra
GRID = {
    "base": {
        "mlp_xy": dict(model="mlp"),
        "cnn_xy": dict(model="cnn1d"),
        "gru_xy": dict(model="bigru"),
    },
    "dims": {"cnn_xyz": dict(model="cnn1d", dims=3)},
    "T": {"cnn_xy_T32": dict(model="cnn1d", T=32), "cnn_xy_T128": dict(model="cnn1d", T=128)},
    "norm": {"cnn_xy_raw": dict(model="cnn1d", norm="raw"), "cnn_xy_center": dict(model="cnn1d", norm="center")},
    "joints": {"cnn_xy_tipwrist": dict(model="cnn1d", joints="tip_wrist"), "cnn_xy_hand": dict(model="cnn1d", joints="hand")},
    "extra": {"cnn_xy_vel": dict(model="cnn1d", vel=True), "cnn_xy_noaug": dict(model="cnn1d", aug=False),
              "cnn_xy_nomask": dict(model="cnn1d", mask_ch=False)},
    # combinations of the best single factors (run after the single-factor grid, validation only)
    "combo": {
        "cnn_tip_raw_xyz": dict(model="cnn1d", norm="raw", dims=3),
        "cnn_tipwrist_raw": dict(model="cnn1d", norm="raw", joints="tip_wrist"),
        "cnn_hand_raw": dict(model="cnn1d", norm="raw", joints="hand"),
        "cnn_hand_raw_xyz": dict(model="cnn1d", norm="raw", joints="hand", dims=3),
        "gru_hand_raw": dict(model="bigru", norm="raw", joints="hand"),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--only", default="", help="comma list of groups (base,dims,T,norm,joints,extra) or 'final'")
    ap.add_argument("--eval_test", action="store_true")
    ap.add_argument("--train_frac", type=float, default=1.0)
    a = ap.parse_args()
    groups = [g for g in a.only.split(",") if g] or list(GRID)
    exps = {}
    for g in groups:
        exps.update(GRID.get(g, {}))
    if "final" in groups:
        exps = {"final_cnn_xy": dict(model="cnn1d")}
    Path(a.out).mkdir(exist_ok=True, parents=True)
    for name, ov in exps.items():
        for s in range(a.seeds):
            if (Path(a.out) / f"{name}_s{s}.json").exists():
                continue
            cfg = parse(["--packed", a.packed, "--out", a.out, "--name", name, "--seed", str(s), "--epochs", str(a.epochs), "--bs", str(a.bs),
                         "--train_frac", str(a.train_frac)] + (["--eval_test"] if a.eval_test else []))
            cfg.update(ov)
            run(cfg)
    rows = []
    for f in sorted(Path(a.out).glob("*_s*.json")):
        r = json.loads(f.read_text())
        rows.append(dict(name=r["cfg"]["name"], seed=r["cfg"]["seed"], val_acc=r["val_acc"], val_f1=r["val_macro_f1"],
                         test_acc=r.get("test_acc"), params=r["params"], train_s=r["train_time_s"], cpu_ms=r["latency_ms_bs1_cpu"],
                         gpu_ms=r["latency_ms_bs1_gpu"], kb=r["ckpt_bytes"] / 1024))
    if rows:
        df = pd.DataFrame(rows)
        agg = df.groupby("name").agg(val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"), val_f1=("val_f1", "mean"),
                                     test_acc=("test_acc", "mean"), params=("params", "first"), kb=("kb", "first"),
                                     cpu_ms=("cpu_ms", "median"), gpu_ms=("gpu_ms", "median"), seeds=("seed", "count")).round(4)
        agg.to_csv(Path(a.out) / "summary.csv")
        print(agg.to_string())


if __name__ == "__main__":
    main()