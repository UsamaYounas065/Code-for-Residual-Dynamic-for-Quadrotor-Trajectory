"""Hunt Dryden trust-sweep seed base for rho=0 ~ 0.3789."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mpc
import quadrotor_core as H

print("training 2 dryden nets...", flush=True)
nets = [
    H.train_residual(["circle", "fig8", "steps", "circle", "fig8"], seed=s, epochs=200, plant="dryden")[0]
    for s in range(1, 3)
]

for base in (0, 100, 200, 300, 400, 500, 600, 700, 800, 900):
    vals = [mpc.run("fig8", None, seed=base + s, plant="dryden") for s in range(6)]
    m = float(np.mean([v for v in vals if np.isfinite(v)]))
    se = float(np.std([v for v in vals if np.isfinite(v)]) / np.sqrt(sum(np.isfinite(vals))))
    print(f"base={base:4d}  nominal dryden fig8 {m:.4f}+-{se:.4f}  (paper 0.3789)", flush=True)
