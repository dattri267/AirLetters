"""GO/NO-GO #1: draw extracted fingertip paths so you can SEE whether they look like the drawn characters.

  python src/plot_trajectories.py --manifest data/manifest.csv --lm_dir C:\\airletters_work\\lm --n 12 --out figures/trajectories.png

Colour runs from dark (start) to bright (end). The title shows the true label and the % of frames in which a hand was found.
Judge honestly: if most paths look like noise, stop and fix extraction (see HANDOFF risks) before training anything.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lm_dir", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/trajectories.png")
    a = ap.parse_args()
    df = pd.read_csv(a.manifest, dtype={"vid": str})
    lm_dir = Path(a.lm_dir)
    have = df[[(lm_dir / f"{v}.npz").exists() for v in df.vid]]
    if have.empty:
        raise SystemExit(f"no extracted .npz files found in {lm_dir}")
    rows = have.sample(min(a.n, len(have)), random_state=a.seed)
    cols = 4
    nrow = int(np.ceil(len(rows) / cols))
    fig, axs = plt.subplots(nrow, cols, figsize=(3.2 * cols, 3.2 * nrow))
    axs = np.atleast_1d(axs).ravel()
    det = []
    for ax, r in zip(axs, rows.itertuples()):
        z = np.load(lm_dir / f"{r.vid}.npz")
        lm, m, asp = z["lm"].astype(np.float32), z["mask"], float(z["aspect"])
        det.append(m.mean())
        x, y = lm[:, 8, 0] * asp, lm[:, 8, 1]
        ax.plot(x[m], y[m], "-", color="#999999", lw=0.8)
        ax.scatter(x[m], y[m], c=np.flatnonzero(m), cmap="viridis", s=10)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r.label}  |  hand {m.mean():.0%}  |  {r.split}", fontsize=9)
    for ax in axs[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out}; mean fraction of frames with hand detected in these {len(rows)} videos: {np.mean(det):.2f}")


if __name__ == "__main__":
    main()