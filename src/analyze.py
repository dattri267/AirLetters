"""Phase 6: failure analysis from a *_preds.npz written by train.py.

  python src/analyze.py --preds results/cnn1d_xy_s0_preds.npz --packed data/packed.npz --split val --out figures/cnn_xy

Produces: confusion matrix PNG, per-class accuracy CSV, top confusions CSV, accuracy-by-tracking-quality CSV/PNG.
The tracking-quality table answers the paper's key question: are errors caused by the tracker or by the representation/model?
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = np.load(a.preds)
    classes = [str(c) for c in np.load(a.packed)["classes"]]
    yt, yp, q = p[f"{a.split}_true"], p[f"{a.split}_pred"], p[f"{a.split}_q"]
    ok = yt >= 0
    yt, yp, q = yt[ok], yp[ok], q[ok]
    n = len(classes)
    cm = confusion_matrix(yt, yp, labels=range(n))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=6); ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=200); plt.close(fig)

    pc = pd.DataFrame({"class": classes, "n": cm.sum(1), "acc": np.diag(cmn)}).sort_values("acc")
    pc.to_csv(out / "per_class_acc.csv", index=False)
    off = [(classes[i], classes[j], int(cm[i, j]), float(cmn[i, j])) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    top = pd.DataFrame(off, columns=["true", "pred", "count", "rate"]).sort_values("count", ascending=False).head(25)
    top.to_csv(out / "top_confusions.csv", index=False)

    bins = [0, 0.5, 0.7, 0.85, 0.95, 1.01]
    lab = ["<50%", "50-70%", "70-85%", "85-95%", ">=95%"]
    qb = pd.cut(q, bins, labels=lab, right=False)
    rows = [(l, int((qb == l).sum()), float((yt[qb == l] == yp[qb == l]).mean()) if (qb == l).any() else np.nan) for l in lab]
    qt = pd.DataFrame(rows, columns=["frames_with_hand", "n_videos", "accuracy"])
    qt.to_csv(out / "acc_by_tracking_quality.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(qt.frames_with_hand, qt.accuracy, color="#3b6fb6")
    ax.set_ylabel("accuracy"); ax.set_xlabel("fraction of frames with hand detected"); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out / "acc_by_quality.png", dpi=200); plt.close(fig)

    print(f"overall acc {np.mean(yt == yp):.4f} on {len(yt)} videos")
    print("worst 8 classes:\n", pc.head(8).to_string(index=False))
    print("top confusions:\n", top.head(10).to_string(index=False))
    print("accuracy by tracking quality:\n", qt.to_string(index=False))


if __name__ == "__main__":
    main()
