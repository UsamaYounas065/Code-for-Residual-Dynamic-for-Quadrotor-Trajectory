"""
NeuroBEM processed CSV loader (open-loop prediction validation).

Expects CSVs under data/neurobem/processed_data/ after unzipping
processed_data.zip from https://download.ifi.uzh.ch/rpg/NeuroBEM/

Column layout follows the NeuroBEM Readme (1-indexed in docs; 0-indexed here).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEURO = ROOT / "data" / "neurobem"
PROCESSED = NEURO / "processed_data"

# NeuroBEM vehicle (readme)
MASS = 0.772
INERTIA_DIAG = np.array([0.0025, 0.0021, 0.0043])
G = 9.81

# CSV column names as in processed_data headers
# Actual processed CSV headers use "quat x" (not qx) — verified on disk.
COLS = {
    "t": "t",
    "ang_acc": ["ang acc x", "ang acc y", "ang acc z"],
    "ang_vel": ["ang vel x", "ang vel y", "ang vel z"],
    "quat": ["quat x", "quat y", "quat z", "quat w"],
    "acc": ["acc x", "acc y", "acc z"],
    "vel": ["vel x", "vel y", "vel z"],
    "pos": ["pos x", "pos y", "pos z"],
    "mot": ["mot 1", "mot 2", "mot 3", "mot 4"],
    "vbat": "vbat",
}


def list_segments(processed: Path = PROCESSED) -> list[Path]:
    if not processed.is_dir():
        return []
    return sorted(processed.glob("*.csv"))


def load_testset(path: Path = NEURO / "testset.txt") -> list[str]:
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_segment(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize possible slight header differences
    df.columns = [c.strip() for c in df.columns]
    return df


def body_speed(df: pd.DataFrame) -> np.ndarray:
    v = df[COLS["vel"]].to_numpy(dtype=float)
    return np.linalg.norm(v, axis=1)


def summarize_segment(csv_path: Path) -> dict:
    df = load_segment(csv_path)
    t = df[COLS["t"]].to_numpy(dtype=float)
    spd = body_speed(df)
    return {
        "file": csv_path.name,
        "n": len(df),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "dt_median": float(np.median(np.diff(t))) if len(t) > 1 else float("nan"),
        "speed_mean": float(spd.mean()),
        "speed_max": float(spd.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="List available CSV segments")
    ap.add_argument("--summary", action="store_true", help="Print per-segment stats")
    ap.add_argument("--limit", type=int, default=5, help="Max segments for --summary")
    args = ap.parse_args()

    segs = list_segments()
    print(f"processed_data dir: {PROCESSED}")
    print(f"segments found: {len(segs)}")
    print(f"testset entries: {len(load_testset())}")

    if not segs:
        print(
            "No CSVs yet. Unzip processed_data.zip into data/neurobem/processed_data/"
        )
        return

    if args.list:
        for p in segs[:50]:
            print(p.name)
        if len(segs) > 50:
            print(f"... ({len(segs) - 50} more)")

    if args.summary:
        for p in segs[: args.limit]:
            s = summarize_segment(p)
            print(
                f"{s['file']}: n={s['n']}  T={s['duration_s']:.1f}s  "
                f"dt~{s['dt_median']*1e3:.2f}ms  "
                f"|v| mean/max={s['speed_mean']:.2f}/{s['speed_max']:.2f}"
            )


if __name__ == "__main__":
    main()
