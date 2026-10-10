"""Offline demo video built from the OFFICIAL TEST SPLIT (no webcam needed, fully reproducible, no cherry-picking).

For each randomly chosen test video (seed fixed, one video per randomly chosen class) it plays the original clip, draws
the extracted hand landmarks and the fingertip trail, and at the end shows the model's top-3 prediction next to the true
label. Wrong predictions are shown exactly like right ones. The title card states how the sample was drawn and the end card
reports the accuracy on this sample next to the accuracy on the whole test split.

  python demo/make_demo_video.py --packed C:\\airletters_work\\packed.npz --manifest data\\manifest.csv ^
      --video_root C:\\path\\to\\airletters --ckpt results_final\\ckpt\\final_hand_s0.pt --n 12 --seed 0 --out demo_video.mp4 --overall_acc 75.43
If a clip's video file cannot be found, only the trajectory is drawn (the prediction is unaffected).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import HAND_EDGES, Classifier  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, INK, MUTE = (250, 250, 250), (35, 35, 35), (120, 120, 120)
GREEN, RED, BLUE, ORANGE = (60, 150, 60), (50, 50, 200), (170, 110, 40), (30, 140, 230)
H, VW, PW, FPS = 480, 640, 400, 30   # canvas height, video area width, side panel width


def text(img, s, org, scale=0.7, color=INK, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def card(lines, size=(VW + PW, H), sub=None):
    img = np.full((size[1], size[0], 3), BG, np.uint8)
    y = size[1] // 2 - 30 * len(lines) // 2
    for i, s in enumerate(lines):
        (w, _), _ = cv2.getTextSize(s, FONT, 0.85 if i == 0 else 0.62, 2 if i == 0 else 1)
        text(img, s, ((size[0] - w) // 2, y + 36 * i), 0.85 if i == 0 else 0.62, INK if i == 0 else MUTE, 2 if i == 0 else 1)
    return img


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr


def fit(frame, w_max=VW, h=H):
    s = min(w_max / frame.shape[1], h / frame.shape[0])
    f = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
    canvas = np.full((h, w_max, 3), 235, np.uint8)
    y0, x0 = (h - f.shape[0]) // 2, (w_max - f.shape[1]) // 2
    canvas[y0:y0 + f.shape[0], x0:x0 + f.shape[1]] = f
    return canvas, (x0, y0, f.shape[1], f.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--video_root", required=True, help="dataset root that contains the videos/ folder")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo_video.mp4")
    ap.add_argument("--overall_acc", default="", help="e.g. 75.43 (full test accuracy, printed on the end card)")
    ap.add_argument("--slow", type=float, default=1.0, help="playback slowdown factor")
    a = ap.parse_args()

    pk = np.load(a.packed, allow_pickle=False)
    man = pd.read_csv(a.manifest, dtype={"vid": str, "actor_id": str}).set_index("vid")
    clf = Classifier(a.ckpt)
    classes = [str(c) for c in pk["classes"]]
    assert classes == [str(c) for c in clf.classes], "checkpoint classes differ from packed file"
    sel = np.flatnonzero(pk["split"] == a.split)
    rng = np.random.RandomState(a.seed)
    cls_pick = rng.permutation(len(classes))[: a.n]
    picks = []
    for c in cls_pick:
        cand = sel[pk["y"][sel] == c]
        if len(cand):
            picks.append(int(rng.choice(cand)))
    offs = pk["offsets"]

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VW + PW, H))
    title = card(["Air-written characters from hand landmarks",
                  f"Random sample of {len(picks)} videos from the official {a.split} split (seed {a.seed}),",
                  "one per randomly chosen class. Wrong predictions are shown too.",
                  f"Model: {clf.name}, {sum(p.numel() for p in clf.model.parameters())/1e3:.0f}k parameters"])
    for _ in range(int(3.5 * FPS)):
        vw.write(title)

    n_ok = 0
    for k, i in enumerate(picks, 1):
        lo, hi = offs[i], offs[i + 1]
        lm = pk["lm"][lo:hi].astype(np.float32)
        mask = pk["mask"][lo:hi].astype(bool)
        true = classes[int(pk["y"][i])]
        pr, q, lat = clf.predict(lm, mask, pk["aspect"][i])
        top = clf.top(pr, 3)
        pred = top[0][0]
        ok = pred == true
        n_ok += ok
        vid = str(pk["vid"][i])
        vp = Path(a.video_root) / man.loc[vid, "video_path"] if vid in man.index else None
        frames = read_frames(vp) if vp is not None and vp.exists() else []
        n_f = max(len(frames), len(mask), 1)
        trail = []
        reps = max(1, int(round(a.slow)))
        for f_i in range(n_f):
            li = min(int(f_i * len(mask) / n_f), len(mask) - 1) if len(mask) else 0
            base = frames[min(f_i, len(frames) - 1)] if frames else np.full((300, 400, 3), 235, np.uint8)
            canvas, (x0, y0, fw, fh) = fit(base)
            if len(mask) and mask[li]:
                p = lm[li]
                pts = [(int(x0 + p[j, 0] * fw), int(y0 + p[j, 1] * fh)) for j in range(21)]
                for a_, b_ in HAND_EDGES:
                    cv2.line(canvas, pts[a_], pts[b_], (200, 200, 200), 1, cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(canvas, pt, 2, (150, 150, 150), -1)
                trail.append(pts[8])
            for p1, p2 in zip(trail[:-1], trail[1:]):
                cv2.line(canvas, p1, p2, ORANGE, 3, cv2.LINE_AA)
            if trail:
                cv2.circle(canvas, trail[-1], 6, ORANGE, -1, cv2.LINE_AA)
            panel = np.full((H, PW, 3), BG, np.uint8)
            text(panel, f"Sample {k} of {len(picks)}", (20, 34), 0.62, MUTE)
            text(panel, "Writing...", (20, 80), 0.8, INK, 2)
            text(panel, f"hand detected in {q*100:.0f}% of frames", (20, H - 24), 0.5, MUTE)
            out = np.hstack([canvas, panel])
            for _ in range(reps):
                vw.write(out)
        # result hold
        for t in range(int(2.2 * FPS)):
            panel = np.full((H, PW, 3), BG, np.uint8)
            text(panel, f"Sample {k} of {len(picks)}", (20, 34), 0.62, MUTE)
            text(panel, "True label", (20, 90), 0.55, MUTE)
            text(panel, true, (20, 130), 1.0, INK, 2)
            text(panel, "Prediction", (20, 185), 0.55, MUTE)
            text(panel, pred, (20, 235), 1.4, GREEN if ok else RED, 3)
            text(panel, "CORRECT" if ok else "WRONG", (190, 235), 0.9, GREEN if ok else RED, 2)
            for r, (c, p) in enumerate(top):
                y = 290 + r * 34
                text(panel, c[:18], (20, y), 0.55, INK)
                cv2.rectangle(panel, (190, y - 14), (190 + int(180 * p), y + 4), BLUE if r == 0 else (200, 190, 180), -1)
                text(panel, f"{p*100:.0f}%", (190 + int(180 * p) + 6, y), 0.5, MUTE)
            text(panel, f"hand detected in {q*100:.0f}% of frames", (20, H - 24), 0.5, MUTE)
            text(panel, f"classifier time {lat:.1f} ms", (20, H - 48), 0.5, MUTE)
            vw.write(np.hstack([canvas, panel]))
        print(f"[{k}/{len(picks)}] true={true} pred={pred} {'ok' if ok else 'WRONG'}  detection={q:.2f}  video={'found' if frames else 'missing (trajectory only)'}")

    end = card(["Summary", f"This sample: {n_ok} of {len(picks)} correct ({100*n_ok/len(picks):.0f}%), a small sample, not an accuracy estimate.",
                *( [f"Accuracy on the whole {a.split} split: {a.overall_acc}%"] if a.overall_acc else []),
                "Main limitation: the hand is often not detected (see the detection rate per sample)."])
    for _ in range(int(5 * FPS)):
        vw.write(end)
    vw.release()
    print(f"wrote {a.out}  ({n_ok}/{len(picks)} correct on this sample)")


if __name__ == "__main__":
    main()