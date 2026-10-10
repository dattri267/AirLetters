"""Shared inference helper for the demos. Uses the SAME preprocessing (features.video_to_feature) and the SAME input
assembly (train.assemble) as training and evaluation, so a demo prediction is exactly what the evaluated model computes."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import DEFAULT_FEAT, JOINT_SETS, video_to_feature  # noqa: E402
from models import build  # noqa: E402
from train import assemble  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Classifier:
    def __init__(self, ckpt, device=None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg, self.classes = ck["cfg"], list(ck["classes"])
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build(self.cfg["model"], ck["T"], ck["C"], len(self.classes), self.cfg["width"] or None, self.cfg["dropout"]).to(self.dev)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.fcfg = {k: self.cfg[k] for k in DEFAULT_FEAT}
        self.n_geo = len(JOINT_SETS[self.fcfg["joints"]]) * self.fcfg["dims"]
        self.name = f"{self.cfg['model']} / {self.fcfg['joints']} / {self.fcfg['norm']}"

    @torch.no_grad()
    def predict(self, lm, mask, aspect):
        """lm (n,21,3) float (NaN where no hand), mask (n,) bool -> (probs (n_classes,), quality, latency_ms)."""
        t0 = time.perf_counter()
        x, q = video_to_feature(lm.astype(np.float32), mask.astype(bool), float(aspect), self.fcfg)
        xt = torch.from_numpy(x)[None].to(self.dev)
        xb = assemble(xt, self.n_geo, self.fcfg["dims"], self.cfg["vel"], self.fcfg["mask_ch"], False)
        pr = torch.softmax(self.model(xb).float(), 1)[0].cpu().numpy()
        return pr, q, (time.perf_counter() - t0) * 1000

    def top(self, pr, k=3):
        i = np.argsort(-pr)[:k]
        return [(self.classes[j], float(pr[j])) for j in i]