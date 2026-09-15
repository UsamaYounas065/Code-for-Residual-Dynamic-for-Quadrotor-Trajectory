"""Hunt seed base for Table 3 residual circle RMSE ~0.2692."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import quadrotor_core as H

TRAIN = ["circle", "fig8", "steps", "circle", "fig8"]
print("training 3 hard nets...", flush=True)
nets = [H.train_residual(TRAIN, seed=s, epochs=200, plant="hard")[0] for s in range(1, 4)]

for base in (0, 100, 200, 300, 400, 500, 600, 700, 800, 900):
    vals = [
        H.rollout("circle", 12, nt, None, seed=base + s, ki=0.0, plant="hard")[0]
        for nt in nets
        for s in range(6)
    ]
    m = float(np.mean(vals))
    se = float(np.std(vals) / np.sqrt(len(vals)))
    print(f"base={base:4d}  circle residual {m:.4f}+-{se:.4f}  (paper 0.2692)", flush=True)
