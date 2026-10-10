"""Phase 2: video -> hand landmarks (MediaPipe Tasks HandLandmarker).

Input : manifest CSV with columns  video_path,label,split,actor_id   (+ optional vid)
Output: one <vid>.npz per video in --out_dir containing
          lm      (n,21,3) float16   normalized x,y (image coords) and relative z; NaN where no hand
          mask    (n,)     bool      hand detected in that processed frame
          score   (n,)     float16   handedness/detection score
          aspect  float    width/height (needed to un-distort x)
          n_total int      frames in the original video
          fps     float
          t_extract float  seconds spent on this video  (use for the efficiency table)

Extraction is CPU work (MediaPipe's Python API has no GPU delegate on Windows).
Use --workers = (CPU cores - 1).  The GPU is used later for training.

Examples
  python src/extract_landmarks.py --manifest data/manifest.csv --video_root data/videos \
      --out_dir data/lm --model models/hand_landmarker.task --limit 50          # speed test
  python src/extract_landmarks.py ... --workers 6                                # real run (resumable)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

G = {}  # per-process globals


def _init(model_path, stride, min_conf, reuse=False):
    import cv2
    cv2.setNumThreads(1)
    G.update(model=model_path, stride=stride, conf=min_conf, reuse=reuse)


def _make_landmarker():
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=G["model"]),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=G["conf"],
        min_hand_presence_confidence=G["conf"],
        min_tracking_confidence=G["conf"],
    )
    return vision.HandLandmarker.create_from_options(opts)


def process(job):
    vid, vpath, out_path = job
    import cv2
    import mediapipe as mp
    t0 = time.time()
    if os.path.exists(out_path):
        return vid, "skip", 0.0, 0.0, 0.0
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return vid, "open_fail", 0.0, 0.0, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    lm_list, mask, score = [], [], []
    stride = G["stride"]
    # default: new landmarker per video -> tracking state never leaks between videos.
    # --reuse: one landmarker per worker (faster start) with ever-increasing timestamps; tracking state may carry over.
    tc0 = time.time()
    if G.get("reuse"):
        if "det" not in G:
            G["det"], G["ts"] = _make_landmarker(), 0
        det, ts0 = G["det"], G["ts"]
    else:
        det, ts0 = _make_landmarker(), 0
    t_create = time.time() - tc0
    i = -1
    n_total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            n_total += 1
            if i % stride:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = det.detect_for_video(img, ts0 + int(i * 1000.0 / fps))
            if res.hand_landmarks:
                pts = res.hand_landmarks[0]
                lm_list.append([[p.x, p.y, p.z] for p in pts])
                mask.append(True)
                score.append(res.handedness[0][0].score if res.handedness else 0.0)
            else:
                lm_list.append(np.full((21, 3), np.nan).tolist())
                mask.append(False)
                score.append(0.0)
    finally:
        cap.release()
        if G.get("reuse"):
            G["ts"] = ts0 + int(n_total * 1000.0 / fps) + 1000
        else:
            det.close()
    if not lm_list:
        return vid, "no_frames", 0.0, 0.0, t_create
    dt = time.time() - t0
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        lm=np.asarray(lm_list, np.float16),
        mask=np.asarray(mask, bool),
        score=np.asarray(score, np.float16),
        aspect=np.float32(w / h if h else 1.0),
        n_total=np.int32(n_total),
        fps=np.float32(fps),
        t_extract=np.float32(dt),
    )
    os.replace(tmp, out_path)
    return vid, "ok", dt, float(np.mean(mask)), t_create


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--video_root", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", required=True, help="path to hand_landmarker.task")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--stride", type=int, default=1, help="process every k-th frame")
    ap.add_argument("--conf", type=float, default=0.3, help="min detection/presence/tracking confidence")
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (speed test)")
    ap.add_argument("--reuse", action="store_true", help="reuse one landmarker per worker (faster; tracking state may carry over between videos)")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"vid": str})
    if "vid" not in df.columns:
        df["vid"] = [f"{i:06d}" for i in range(len(df))]
    df["vid"] = df["vid"].astype(str)
    if args.limit:
        df = df.head(args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r.vid, str(Path(args.video_root) / r.video_path), str(out_dir / f"{r.vid}.npz"))
            for r in df.itertuples()]

    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "open_fail": 0, "no_frames": 0}
    times, rates, creates = [], [], []
    failed = []
    with Pool(args.workers, initializer=_init, initargs=(args.model, args.stride, args.conf, args.reuse)) as pool:
        for k, (vid, st, dt, rt, tcr) in enumerate(pool.imap_unordered(process, jobs, chunksize=4), 1):
            stats[st] += 1
            if st == "ok":
                times.append(dt)
                rates.append(rt)
                creates.append(tcr)
            elif st != "skip":
                failed.append((vid, st))
            if k % 200 == 0 or k == len(jobs):
                el = time.time() - t0
                rate = k / el
                eta = (len(jobs) - k) / rate / 3600
                print(f"[{k}/{len(jobs)}] {rate:.2f} videos/s  ETA {eta:.2f} h  {stats}", flush=True)
    el = time.time() - t0
    print(f"done in {el/60:.1f} min  {stats}")
    if times:
        print(f"mean per-video CPU time {np.mean(times):.2f}s (median {np.median(times):.2f}s) with {args.workers} workers")
        print(f"mean time to create the landmarker per video: {np.mean(creates):.2f}s   (reuse={args.reuse})")
        print(f"mean fraction of frames with a hand detected: {np.mean(rates):.3f}  (conf={args.conf}, stride={args.stride})")
    if failed:
        pd.DataFrame(failed, columns=["vid", "status"]).to_csv(out_dir / "_failed.csv", index=False)
        print(f"{len(failed)} failures listed in {out_dir/'_failed.csv'}")


if __name__ == "__main__":
    main()