"""Live air-writing demo (qualitative only; the accuracy numbers in the paper come from the test split, not from this demo).

  python demo/webcam_demo.py --ckpt results_final\\ckpt\\final_hand_s0.pt --model_task models\\hand_landmarker.task

Keys:  SPACE start writing / SPACE again to stop and classify   c clear   r start/stop recording to demo_live.mp4   q quit
Write ONE character per take, large, with your index finger, facing the camera, in good light.
Uses the same preprocessing and model code as training (demo/infer.py), the same detector and confidences as extraction (0.1).

If the camera view is mirrored for the model, predictions for asymmetric characters (try Z, S, 2, 5) will be wrong:
run once with --flip_x and keep whichever setting gets Z right.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import HAND_EDGES, Classifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--model_task", required=True)
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--conf", type=float, default=0.1)
ap.add_argument("--flip_x", action="store_true", help="mirror x of the landmarks before the model (use if Z comes out wrong)")
ap.add_argument("--out", default="demo_live.mp4")
ap.add_argument("--log", default="", help="CSV file: after each prediction you type the character you actually wrote, so you get an honest tally")
a = ap.parse_args()

clf = Classifier(a.ckpt)
det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=a.model_task), running_mode=vision.RunningMode.VIDEO, num_hands=1,
    min_hand_detection_confidence=a.conf, min_hand_presence_confidence=a.conf, min_tracking_confidence=a.conf))
cap = cv2.VideoCapture(a.cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
FONT = cv2.FONT_HERSHEY_SIMPLEX
writer, rec, buf, msk, path, result = None, False, [], [], [], None
awaiting = False  # --log: waiting for you to type the true character
logged = []
t0, last_ts = time.time(), -1
if a.log and not Path(a.log).exists():
    Path(a.log).write_text("true,pred,confidence,hand_seen\n", encoding="utf-8")
print("ready. SPACE = start / stop and classify.")
while True:
    ok, cam = cap.read()
    if not ok:
        break
    h, w = cam.shape[:2]
    ts = max(int((time.time() - t0) * 1000), last_ts + 1)  # strictly increasing for MediaPipe VIDEO mode
    last_ts = ts
    res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)), ts)
    view = cv2.flip(cam, 1)  # mirrored view is only for the person looking at the screen
    pts_xy = None
    if res.hand_landmarks:
        pts = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], np.float32)
        pts_xy = [(int((1 - p[0]) * w), int(p[1] * h)) for p in pts]
        for e1, e2 in HAND_EDGES:
            cv2.line(view, pts_xy[e1], pts_xy[e2], (220, 220, 220), 1, cv2.LINE_AA)
        cv2.circle(view, pts_xy[8], 8, (0, 255, 0), -1)
        if rec:
            m = pts.copy()
            if a.flip_x:
                m[:, 0] = 1 - m[:, 0]
            buf.append(m)
            msk.append(True)
            path.append(pts_xy[8])
    elif rec:
        buf.append(np.full((21, 3), np.nan, np.float32))
        msk.append(False)
    for p1, p2 in zip(path[:-1], path[1:]):
        cv2.line(view, p1, p2, (0, 200, 255), 4, cv2.LINE_AA)
    cv2.rectangle(view, (0, 0), (w, 44), (30, 30, 30), -1)
    cv2.putText(view, "WRITING - press SPACE when done" if rec else "idle - press SPACE, then write one character", (12, 30), FONT, 0.7,
                (0, 0, 255) if rec else (255, 255, 255), 2, cv2.LINE_AA)
    if result:
        top, lat, q = result
        cv2.rectangle(view, (0, h - 150), (w, h), (30, 30, 30), -1)
        cv2.putText(view, top[0][0], (20, h - 40), FONT, 3.0, (80, 220, 80), 5, cv2.LINE_AA)
        for r, (c, p) in enumerate(top):
            y = h - 105 + r * 34
            cv2.putText(view, f"{c[:16]}  {p*100:.0f}%", (330, y), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(view, (600, y - 16), (600 + int(240 * p), y + 4), (230, 160, 60), -1)
        cv2.putText(view, f"hand seen in {q*100:.0f}% of frames | classifier {lat:.1f} ms", (20, h - 8), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if awaiting:
        cv2.rectangle(view, (0, 44), (w, 90), (0, 0, 160), -1)
        cv2.putText(view, "Type the character you WROTE (0-9, a-z). ESC = skip", (12, 78), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("air-writing demo", view)
    if writer is not None:
        writer.write(view)
    k = cv2.waitKey(1) & 0xFF
    if awaiting and k != 255:
        if k != 27 and (chr(k).isalnum()):
            true_c = chr(k).upper()
            logged.append((true_c, result[0][0][0], result[0][0][1], result[2]))
            ok_n = sum(t == p for t, p, _, _ in logged)
            print(f"logged: wrote {true_c}, predicted {result[0][0][0]} -> {'OK' if true_c == result[0][0][0] else 'WRONG'}   running {ok_n}/{len(logged)}")
            with open(a.log, "a", encoding="utf-8") as f:
                f.write(f"{true_c},{result[0][0][0]},{result[0][0][1]:.3f},{result[2]:.2f}\n")
        awaiting = False
        continue
    if k == ord("q"):
        break
    if k == ord("c"):
        path, result = [], None
    if k == ord("r"):
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            print("recording ->", a.out)
        else:
            writer.release()
            writer = None
            print("saved", a.out)
    if k == 32:
        if not rec:
            rec, buf, msk, path, result = True, [], [], [], None
        else:
            rec = False
            if sum(msk) >= 8:
                pr, q, lat = clf.predict(np.stack(buf), np.array(msk), w / h)
                result = (clf.top(pr, 3), lat, q)
                print("prediction:", result[0], f"(hand seen {q*100:.0f}%)")
                awaiting = bool(a.log)
            else:
                result = ([("too few hand frames", 0.0)] * 3, 0.0, 0.0)
if writer is not None:
    writer.release()
cap.release()
cv2.destroyAllWindows()