"""Step 1 after downloading: show how the dataset folder is organised, so the manifest can be built correctly.

  python src/inspect_dataset.py "D:\\path\\to\\airletters"            (the folder you downloaded / unzipped)

Prints a report and also saves it to  data/dataset_report.txt  -> paste that file's contents back to the assistant.
Read-only: it never changes or deletes anything.
"""
import collections
import os
import sys
from pathlib import Path

VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
META = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".xlsx", ".parquet", ".pkl", ".yaml", ".yml"}


def head(path, n=6, width=220):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [next(f).rstrip("\n")[:width] for _ in range(n)]
    except StopIteration:
        return []
    except Exception as e:  # noqa: BLE001
        return [f"<could not read: {e}>"]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python src/inspect_dataset.py <dataset_folder>")
    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"not a folder: {root}")
    out = []
    P = out.append
    ext_count = collections.Counter()
    ext_bytes = collections.Counter()
    samples = collections.defaultdict(list)
    meta_files, dir_counts = [], collections.Counter()
    for dp, dn, fn in os.walk(root):
        rel = Path(dp).relative_to(root)
        depth = len(rel.parts)
        for f in fn:
            p = Path(dp) / f
            e = p.suffix.lower() or "<none>"
            ext_count[e] += 1
            try:
                ext_bytes[e] += p.stat().st_size
            except OSError:
                pass
            if len(samples[e]) < 5:
                samples[e].append(str(p.relative_to(root)))
            if e in META and len(meta_files) < 40:
                meta_files.append(p)
            if depth <= 2:
                dir_counts[str(rel)] += 1
    P(f"ROOT: {root}")
    P("\n== file types ==")
    for e, c in ext_count.most_common(15):
        P(f"{e:10s} {c:8d} files  {ext_bytes[e]/2**30:8.2f} GB   e.g. {samples[e][:2]}")
    P("\n== folders (depth<=2) and number of files directly inside ==")
    for d, c in sorted(dir_counts.items())[:60]:
        P(f"{d or '.'}: {c}")
    P("\n== metadata-like files (first lines) ==")
    for m in meta_files:
        P(f"--- {m.relative_to(root)}  ({m.stat().st_size/1024:.1f} KB)")
        if m.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".txt", ".yaml", ".yml"}:
            for line in head(m):
                P("   " + line)
    vids = [s for e in VIDEO for s in samples.get(e, [])]
    P("\n== sample videos (opened with OpenCV) ==")
    try:
        import cv2
        for s in vids[:5]:
            cap = cv2.VideoCapture(str(root / s))
            P(f"{s}: opened={cap.isOpened()} frames={int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} fps={cap.get(cv2.CAP_PROP_FPS):.1f} "
              f"size={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            cap.release()
    except ImportError:
        P("opencv not installed: pip install opencv-python")
    text = "\n".join(out)
    print(text)
    Path("data").mkdir(exist_ok=True)
    Path("data/dataset_report.txt").write_text(text, encoding="utf-8")
    print("\nSaved to data/dataset_report.txt")


if __name__ == "__main__":
    main()