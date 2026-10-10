"""Class-balanced subset of the manifest so extraction fits your time budget.
  python src/make_subset.py --manifest data/manifest.csv --out data/manifest_sub.csv --train 600 --val 100 --test 0
Numbers are per class. Selection is by actor-aware random sampling within each split (splits stay disjoint). 'test 0' = skip test."""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--train", type=int, default=600)
ap.add_argument("--val", type=int, default=100)
ap.add_argument("--test", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
df = pd.read_csv(a.manifest, dtype={"vid": str})
if "vid" not in df.columns:
    df["vid"] = [f"{i:06d}" for i in range(len(df))]  # ids refer to ORIGINAL row order -> stable across subsets
parts = []
for sp, n in (("train", a.train), ("val", a.val), ("test", a.test)):
    if n <= 0:
        continue
    d = df[df.split == sp]
    parts.append(d.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), n), random_state=a.seed)))
sub = pd.concat(parts).sort_index()
sub.to_csv(a.out, index=False)
print(sub.split.value_counts().to_dict(), "videos ->", a.out)
