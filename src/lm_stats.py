"""Compare extraction settings on the SAME videos without re-running anything.

  python src/lm_stats.py C:\\airletters_work\\lm C:\\airletters_work\\lm_reuse [more folders]

Prints, per folder: number of videos, mean fraction of frames with a hand, mean frames, mean extraction seconds.
With 2+ folders it also prints the paired difference in detection rate on the videos they have in common (first folder = reference).
"""
import sys
from pathlib import Path

import numpy as np


def load(d):
    out = {}
    for f in Path(d).glob("*.npz"):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        out[f.stem] = (float(z["mask"].mean()), len(z["mask"]), float(z["t_extract"]))
    return out


def main():
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit("usage: python src/lm_stats.py <folder1> [<folder2> ...]")
    data = {d: load(d) for d in dirs}
    for d, v in data.items():
        if not v:
            print(f"{d}: no files")
            continue
        a = np.array(list(v.values()))
        print(f"{d}\n   videos {len(v)} | detection {a[:,0].mean():.3f} | frames {a[:,1].mean():.1f} | extraction {a[:,2].mean():.2f}s")
    ref = dirs[0]
    for d in dirs[1:]:
        common = sorted(set(data[ref]) & set(data[d]))
        if common:
            diff = np.array([data[d][k][0] - data[ref][k][0] for k in common])
            print(f"paired on {len(common)} common videos: {d} minus {ref} = {diff.mean():+.3f} detection "
                  f"(better on {(diff > 0.01).sum()}, worse on {(diff < -0.01).sum()})")


if __name__ == "__main__":
    main()