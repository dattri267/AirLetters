"""Synthetic stand-in for AirLetters so the WHOLE pipeline (pack -> train -> analyze) can be smoke-tested
before the real data is downloaded. Classes = distinct random smooth curves; actors differ in scale/offset/noise;
~15% of frames lose the hand. NOT real data: never report these numbers.

  python src/make_synthetic.py --out data_synth --n 3600
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--n", type=int, default=3600)
    ap.add_argument("--classes", type=int, default=36)
    ap.add_argument("--actors", type=int, default=60)
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    out = Path(a.out)
    (out / "lm").mkdir(parents=True, exist_ok=True)
    coef = rng.randn(a.classes, 2, 4, 2)  # per class: 4 harmonics (cos,sin) per axis
    rows = []
    for i in range(a.n):
        c = i % a.classes
        actor = rng.randint(a.actors)
        n = rng.randint(40, 90)
        t = np.linspace(0, 1, n)
        sp = 1 + 0.1 * rng.randn()
        x = sum(coef[c, 0, h, 0] * np.cos(2 * np.pi * (h + 1) * t * sp) + coef[c, 0, h, 1] * np.sin(2 * np.pi * (h + 1) * t * sp) for h in range(4))
        y = sum(coef[c, 1, h, 0] * np.cos(2 * np.pi * (h + 1) * t * sp) + coef[c, 1, h, 1] * np.sin(2 * np.pi * (h + 1) * t * sp) for h in range(4))
        sc = 0.05 + 0.02 * ((actor * 7919) % 10) / 10 + 0.01 * rng.rand()
        off = 0.3 + 0.4 * rng.rand(2)
        tip = np.stack([off[0] + sc * x + 0.004 * rng.randn(n), off[1] + sc * y + 0.004 * rng.randn(n), 0.02 * rng.randn(n)], 1)
        lm = np.repeat(tip[:, None, :], 21, 1) + 0.03 * rng.randn(1, 21, 3)
        lm[:, 8] = tip
        mask = rng.rand(n) > 0.15
        mask[:2] = True
        mask[-2:] = True
        lm[~mask] = np.nan
        vid = f"{i:06d}"
        np.savez_compressed(out / "lm" / f"{vid}.npz", lm=lm.astype(np.float16), mask=mask, score=np.ones(n, np.float16),
                            aspect=np.float32(16 / 9), n_total=np.int32(n), fps=np.float32(30), t_extract=np.float32(0.1))
        rows.append((f"synthetic/{vid}.mp4", f"cls{c:02d}", actor))
    df = pd.DataFrame(rows, columns=["video_path", "label", "actor_id"])
    df["vid"] = [f"{i:06d}" for i in range(len(df))]
    # actor-disjoint splits
    ua = np.arange(a.actors)
    rng.shuffle(ua)
    sp = {act: ("train" if k < 0.7 * a.actors else "val" if k < 0.85 * a.actors else "test") for k, act in enumerate(ua)}
    df["split"] = df.actor_id.map(sp)
    df.to_csv(out / "manifest.csv", index=False)
    print(df.split.value_counts().to_dict(), "->", out / "manifest.csv")


if __name__ == "__main__":
    main()
