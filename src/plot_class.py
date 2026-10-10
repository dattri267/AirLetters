"""Draw fingertip paths for ONE class from the packed file, so you can see how that character is written in AirLetters
(start = dark, end = bright) and copy that style in the live demo.

  python src/plot_class.py --packed C:\\airletters_work\\packed.npz --label G --n 12 --out figures\\class_G.png
  add --split test (default train) / --min_det 0.9 to show only well-tracked examples
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--split", default="train")
    ap.add_argument("--min_det", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/class.png")
    a = ap.parse_args()
    pk = np.load(a.packed, allow_pickle=False)
    classes = [str(c) for c in pk["classes"]]
    if a.label not in classes:
        raise SystemExit(f"unknown label {a.label!r}; classes: {classes}")
    c = classes.index(a.label)
    offs = pk["offsets"]
    idx = [i for i in np.flatnonzero((pk["split"] == a.split) & (pk["y"] == c)) if offs[i + 1] > offs[i] and pk["mask"][offs[i]:offs[i + 1]].mean() >= a.min_det]
    rng = np.random.RandomState(a.seed)
    idx = list(rng.permutation(idx)[: a.n])
    if not idx:
        raise SystemExit("no videos match (lower --min_det)")
    cols = 4
    nrow = int(np.ceil(len(idx) / cols))
    fig, axs = plt.subplots(nrow, cols, figsize=(3.0 * cols, 3.0 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, i in zip(axs, idx):
        lm = pk["lm"][offs[i]:offs[i + 1]].astype(np.float32)
        m = pk["mask"][offs[i]:offs[i + 1]].astype(bool)
        x, y = lm[:, 8, 0] * float(pk["aspect"][i]), lm[:, 8, 1]
        ax.plot(x[m], y[m], "-", color="#999999", lw=0.8)
        ax.scatter(x[m], y[m], c=np.flatnonzero(m), cmap="viridis", s=10)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{a.label}  |  hand {m.mean():.0%}", fontsize=9)
    for ax in axs[len(idx):]:
        ax.axis("off")
    fig.suptitle(f"How '{a.label}' is written in AirLetters ({a.split}); image coordinates, as the camera saw it", fontsize=10)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out} ({len(idx)} examples)")


if __name__ == "__main__":
    main()