"""Build data/manifest.csv from the official AirLetters train.csv / val.csv / test.csv.

  python src/build_manifest.py --root "C:\\Users\\you\\Downloads\\airletters" --out data/manifest.csv

Official csv columns:  id, filename, label, worker_id, video_duration   (note the spaces after the commas)
Output columns:        vid,video_path,label,split,actor_id,label_text,duration
  vid        = filename stem (e.g. 00004333) -> unique, stable
  video_path = videos/<filename>   (relative to --root, which you pass to extract_landmarks.py as --video_root)
  label      = short class: A..Z, 0..9, or the full text for the non-character (contrast) classes
  actor_id   = worker_id
Rows are shuffled with a fixed seed so any prefix of the file is a random sample of all splits.
Nothing is invented: every row comes from the official csv files; the script also checks the files exist.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

PAT = re.compile(r"Drawing the (letter|digit)\s+(\w)\s+in the air", re.I)


def short_label(text):
    m = PAT.search(text)
    if m:
        return m.group(2).upper() if m.group(1).lower() == "letter" else m.group(2)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder that contains videos/ and train.csv val.csv test.csv")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_file_check", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    parts = []
    for sp in ("train", "val", "test"):
        f = root / f"{sp}.csv"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        d = pd.read_csv(f, skipinitialspace=True, dtype={"filename": str})
        d.columns = [c.strip() for c in d.columns]
        for c in ("filename", "label"):
            d[c] = d[c].astype(str).str.strip()
        d["split"] = sp
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["vid"] = df["filename"].str.replace(".mp4", "", regex=False)
    df["video_path"] = "videos/" + df["filename"]
    df["label_text"] = df["label"]
    df["label"] = df["label_text"].map(short_label)
    df["actor_id"] = df["worker_id"].astype(str)
    df["duration"] = df["video_duration"]
    out = df[["vid", "video_path", "label", "split", "actor_id", "label_text", "duration"]]

    # ---- sanity checks (printed, so you can put them in the paper's dataset section) ----
    print(f"rows: {len(out)}   unique vid: {out.vid.nunique()}")
    if out.vid.duplicated().any():
        raise SystemExit("duplicate video ids across the csv files")
    print(out.split.value_counts().to_string())
    print(f"\nclasses: {out.label.nunique()}")
    mapped = out.drop_duplicates("label_text")[["label_text", "label"]]
    other = mapped[mapped.label_text == mapped.label]
    print("non-character (contrast) classes kept as-is:", other.label_text.tolist())
    cc = out.groupby("label").size()
    print(f"videos per class: min {cc.min()}  median {int(cc.median())}  max {cc.max()}")
    print("\nworkers per split:", out.groupby("split").actor_id.nunique().to_dict())
    tr = set(out.loc[out.split == "train", "actor_id"])
    for sp in ("val", "test"):
        ov = tr & set(out.loc[out.split == sp, "actor_id"])
        print(f"worker overlap train/{sp}: {len(ov)}   (0 = person-independent split)")
    ov = set(out.loc[out.split == "val", "actor_id"]) & set(out.loc[out.split == "test", "actor_id"])
    print(f"worker overlap val/test: {len(ov)}")
    if not a.skip_file_check:
        miss = [p for p in out.video_path if not (root / p).exists()]
        print(f"\nvideo files missing on disk: {len(miss)}", miss[:5])
    out = out.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()