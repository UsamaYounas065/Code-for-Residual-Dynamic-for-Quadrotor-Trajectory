"""Smoke-test quadrotor_core against paper ballpark numbers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import quadrotor_core as Q  # noqa: E402


def main() -> None:
    rmse_pd, effort = Q.rollout("fig8", Tend=12.0, seed=0, ki=0.0, plant="smooth")
    rmse_i, _ = Q.rollout("fig8", Tend=12.0, seed=0, ki=2.0, plant="smooth")
    print(f"PD only  fig8 RMSE={rmse_pd:.4f} (paper ~0.2107)  effort={effort:.2f}")
    print(f"PD+I     fig8 RMSE={rmse_i:.4f} (paper ~0.0973)")
    if abs(rmse_pd - 0.2107) > 0.02:
        raise SystemExit("PD-only RMSE far from paper; check core.")
    print("smoke_test_core: OK")


if __name__ == "__main__":
    main()
