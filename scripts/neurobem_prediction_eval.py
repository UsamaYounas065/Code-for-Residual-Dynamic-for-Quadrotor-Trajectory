"""
NeuroBEM open-loop multi-step prediction evaluation (refined).

Arms:
  (i)   nominal — thrust-only rigid body (fitted c_T)
  (ii)  residual — MLP force/torque residual on top of nominal
  (iii) accel-DOB — LPF body-force disturbance, held over horizon
  (iv)  neurobem-ref — uses published bem+nn residual forces from the
        NeuroBEM release as an external reference (not our method)

Horizons: 10, 50, 100, 200, 350, 500 ms (simulation Table 4).

Data:
  processed_data/merged_*.csv
  bem+nn/bem+nn_*.csv  (41 cols, no header; cols 30–41 = pred + residual wrench)

Example:
  python scripts/neurobem_prediction_eval.py --quick --plot
  python scripts/neurobem_prediction_eval.py --epochs 60 --plot
  python scripts/neurobem_prediction_eval.py --high-speed-only --speed-min 5 --plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quadrotor_core import MLP  # noqa: E402

# Prefer repo data; fall back to sibling ../data/neurobem
NEURO = ROOT / "data" / "neurobem"
if not (NEURO / "processed_data").is_dir():
    NEURO = ROOT.parent / "data" / "neurobem"

PROCESSED = NEURO / "processed_data"
BEMNN = NEURO / "bem+nn"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MASS = 0.772
J_DIAG = np.array([0.0025, 0.0021, 0.0043])
G_VEC = np.array([0.0, 0.0, -9.81])  # world, z-up
HORIZONS_MS = (10, 50, 100, 200, 350, 500)

# Features: vel_b(3) + omega(3) + motors(4)  — no position (paper §3.5)
# Outputs: body force residual(3) [N] + body torque residual(3) [Nm]
NIN, NOUT, NH = 10, 6, 64


# --------------------------------------------------------------------------- geometry / math
def quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def segment_paths() -> list[Path]:
    return sorted(PROCESSED.glob("merged_*.csv"))


def load_testset_ids() -> set[str]:
    path = NEURO / "testset.txt"
    if not path.is_file():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def segment_key(path: Path) -> str:
    return path.stem.replace("merged_", "", 1)


def is_test_segment(path: Path, test_ids: set[str]) -> bool:
    return segment_key(path) in test_ids


def bemnn_path_for(processed: Path) -> Path | None:
    p = BEMNN / f"bem+nn_{segment_key(processed)}.csv"
    return p if p.is_file() else None


def load_processed(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return dict(
        t=df["t"].to_numpy(float),
        quat=df[["quat x", "quat y", "quat z", "quat w"]].to_numpy(float),
        vel_b=df[["vel x", "vel y", "vel z"]].to_numpy(float),
        acc_b=df[["acc x", "acc y", "acc z"]].to_numpy(float),
        omega=df[["ang vel x", "ang vel y", "ang vel z"]].to_numpy(float),
        ang_acc=df[["ang acc x", "ang acc y", "ang acc z"]].to_numpy(float),
        pos=df[["pos x", "pos y", "pos z"]].to_numpy(float),
        mot=df[["mot 1", "mot 2", "mot 3", "mot 4"]].to_numpy(float),
    )


def load_bemnn_wrench(path: Path) -> dict[str, np.ndarray]:
    """41-col no-header file: cols 29:35 pred wrench, 35:41 residual wrench."""
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 41:
        raise ValueError(f"{path.name}: expected >=41 cols, got {arr.shape[1]}")
    return dict(
        F_pred=arr[:, 29:32],
        tau_pred=arr[:, 32:35],
        F_res=arr[:, 35:38],
        tau_res=arr[:, 38:41],
    )


def fit_thrust_coeff(segments: list[Path], speed_max: float = 1.5) -> float:
    num = den = 0.0
    for path in segments:
        a = load_processed(path)
        mask = np.linalg.norm(a["vel_b"], axis=1) < speed_max
        if not np.any(mask):
            continue
        w2 = np.sum(a["mot"][mask] ** 2, axis=1)
        sz = a["acc_b"][mask, 2]  # specific force ≈ T/m
        num += float(np.sum(sz * w2))
        den += float(np.sum((w2 / MASS) * w2))
    if den < 1e-12:
        return MASS * 9.81 / (4 * (1200.0**2))
    return num / den


def nominal_thrust_force(mot: np.ndarray, c_t: float) -> np.ndarray:
    return np.array([0.0, 0.0, c_t * np.sum(mot**2)])


def nominal_torque(mot: np.ndarray, c_t: float) -> np.ndarray:
    """Crude X-quad allocation (geometry approx); mainly regularises rate channel."""
    w2 = mot**2
    l, c_q = 0.125, c_t * 0.05
    # mot: 1 BR, 2 FR, 3 BL, 4 FL
    tau_x = l * c_t * ((w2[1] + w2[3]) - (w2[0] + w2[2]))
    tau_y = l * c_t * ((w2[0] + w2[1]) - (w2[2] + w2[3]))
    tau_z = c_q * (w2[0] - w2[1] + w2[2] - w2[3])
    return np.array([tau_x, tau_y, tau_z])


def features_row(vel_b: np.ndarray, omega: np.ndarray, mot: np.ndarray) -> np.ndarray:
    return np.concatenate([vel_b, omega, mot])


def measured_force(acc_b: np.ndarray) -> np.ndarray:
    """Body specific force → force [N] (NeuroBEM acc convention)."""
    return MASS * acc_b


def measured_torque(ang_acc: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """tau ≈ J alpha + omega × (J omega)."""
    J = np.diag(J_DIAG)
    return J @ ang_acc + np.cross(omega, J @ omega)


# --------------------------------------------------------------------------- training
def build_training_arrays(
    train_paths: list[Path],
    c_t: float,
    max_samples: int | None,
    seed: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for path in train_paths:
        a = load_processed(path)
        n = len(a["t"])
        for k in range(0, n, stride):
            F_meas = measured_force(a["acc_b"][k])
            tau_meas = measured_torque(a["ang_acc"][k], a["omega"][k])
            F_nom = nominal_thrust_force(a["mot"][k], c_t)
            tau_nom = nominal_torque(a["mot"][k], c_t)
            y = np.concatenate([F_meas - F_nom, tau_meas - tau_nom])
            x = features_row(a["vel_b"][k], a["omega"][k], a["mot"][k])
            Xs.append(x)
            Ys.append(y)
    X, Y = np.asarray(Xs, float), np.asarray(Ys, float)
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, Y = X[idx], Y[idx]
    return X, Y


class ForceMLP(MLP):
    """Same optimiser as paper MLP; NeuroBEM force/torque I/O sizes."""

    def __init__(self, seed: int = 0):
        r = np.random.default_rng(seed)
        dims = [NIN, NH, NH, NOUT]
        self.W = [r.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self.xmu = np.zeros(NIN)
        self.xsd = np.ones(NIN)
        self.ymu = np.zeros(NOUT)
        self.ysd = np.ones(NOUT)


def train_mlp(X: np.ndarray, Y: np.ndarray, epochs: int, seed: int) -> ForceMLP:
    net = ForceMLP(seed)
    rng = np.random.default_rng(seed + 7)
    n = len(X)
    ntr = int(0.85 * n)
    perm = rng.permutation(n)
    X, Y = X[perm], Y[perm]
    net.xmu, net.xsd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-8
    net.ymu, net.ysd = Y[:ntr].mean(0), Y[:ntr].std(0) + 1e-8
    Ztr = (X[:ntr] - net.xmu) / net.xsd
    Wtr = (Y[:ntr] - net.ymu) / net.ysd
    Zva = (X[ntr:] - net.xmu) / net.xsd
    Wva = (Y[ntr:] - net.ymu) / net.ysd
    best, bw = np.inf, None
    for _ in range(epochs):
        p = rng.permutation(len(Ztr))
        for j in range(0, len(p), 512):
            net.step(Ztr[p[j : j + 512]], Wtr[p[j : j + 512]], lr=1.5e-3)
        vl = float(np.mean((net._fwd(Zva)[2] - Wva) ** 2))
        if vl < best:
            best = vl
            bw = ([w.copy() for w in net.W], [b.copy() for b in net.b])
    net.W, net.b = bw
    # one-step force R^2 on validation (de-norm)
    pred = net.predict(X[ntr:])
    ss_res = np.sum((Y[ntr:, :3] - pred[:, :3]) ** 2)
    ss_tot = np.sum((Y[ntr:, :3] - Y[ntr:, :3].mean(0)) ** 2) + 1e-12
    print(f"MLP: n={n}  best val MSE(norm)={best:.5f}  force R^2={1 - ss_res / ss_tot:.3f}")
    return net


# --------------------------------------------------------------------------- open-loop rollout
def predict_horizon_errors(
    path: Path,
    c_t: float,
    net: ForceMLP | None,
    mode: str,
    horizons_ms: tuple[int, ...],
    dob_tau: float,
    stride: int,
    speed_min: float | None,
) -> dict[int, list[float]]:
    """
    mode: nominal | residual | dob | neurobem-ref
    Logged motors + measured attitude; integrate position/velocity in world frame.
    """
    a = load_processed(path)
    bem = None
    if mode == "neurobem-ref":
        bp = bemnn_path_for(path)
        if bp is None:
            return {h: [] for h in horizons_ms}
        bem = load_bemnn_wrench(bp)
        if len(bem["F_res"]) != len(a["t"]):
            return {h: [] for h in horizons_ms}

    t = a["t"]
    dt_med = float(np.median(np.diff(t)))
    steps_h = {h: max(1, int(round((h / 1000.0) / dt_med))) for h in horizons_ms}
    out = {h: [] for h in horizons_ms}
    J = np.diag(J_DIAG)

    # DOB: body-force disturbance estimate along the log
    dob_F = np.zeros_like(a["acc_b"])
    if mode == "dob":
        d = np.zeros(3)
        for k in range(len(t)):
            F_nom = nominal_thrust_force(a["mot"][k], c_t)
            F_meas = measured_force(a["acc_b"][k])
            innov = F_meas - F_nom
            if k == 0:
                d = innov
            else:
                dt = max(t[k] - t[k - 1], 1e-4)
                d = d + (dt / max(dob_tau, dt)) * (innov - d)
            dob_F[k] = d

    max_h = max(steps_h.values())
    for k0 in range(0, len(t) - max_h - 1, stride):
        if speed_min is not None:
            if float(np.linalg.norm(a["vel_b"][k0])) < speed_min:
                continue

        pos = a["pos"][k0].copy()
        quat = a["quat"][k0].copy()
        v_w = quat_to_R(quat) @ a["vel_b"][k0]
        omega = a["omega"][k0].copy()
        pos_at: dict[int, np.ndarray] = {}

        for s in range(max_h):
            k = k0 + s
            dt = t[k + 1] - t[k]
            if dt <= 0:
                break

            F_nom = nominal_thrust_force(a["mot"][k], c_t)
            tau_nom = nominal_torque(a["mot"][k], c_t)
            F = F_nom.copy()
            tau = tau_nom.copy()

            if mode == "dob":
                F = F + dob_F[k0]  # hold start-of-horizon estimate
            elif mode == "residual" and net is not None:
                feat = features_row(a["vel_b"][k], omega, a["mot"][k])
                # use measured vel_b at k for features when using logged attitude path;
                # state vel from integration for consistency on long horizons:
                vel_b_pred = quat_to_R(quat).T @ v_w
                feat = features_row(vel_b_pred, omega, a["mot"][k])
                wr = net.predict(feat)[0]
                F = F + wr[:3]
                tau = tau + wr[3:]
            elif mode == "neurobem-ref" and bem is not None:
                # published residual on top of *their* predicted wrench ≈ measured
                # For fair open-loop force: use F_pred + F_res (= measured wrench proxy)
                F = bem["F_pred"][k] + bem["F_res"][k]
                tau = bem["tau_pred"][k] + bem["tau_res"][k]

            a_w = quat_to_R(quat) @ (F / MASS) + G_VEC
            v_w = v_w + a_w * dt
            pos = pos + v_w * dt
            wdot = np.linalg.solve(J, tau - np.cross(omega, J @ omega))
            omega = omega + wdot * dt
            # measured attitude (logged-input open-loop; isolates translational residual)
            quat = a["quat"][k + 1]
            pos_at[s + 1] = pos.copy()

        for h, nstep in steps_h.items():
            if nstep in pos_at:
                err = float(np.linalg.norm(pos_at[nstep] - a["pos"][k0 + nstep]))
                if np.isfinite(err):
                    out[h].append(err)
    return out


def aggregate(err_lists: dict[int, list[float]]) -> dict[int, dict[str, float]]:
    res = {}
    for h, vals in err_lists.items():
        if not vals:
            res[h] = dict(mean=float("nan"), se=float("nan"), n=0)
            continue
        v = np.asarray(vals, float)
        res[h] = dict(
            mean=float(v.mean()),
            se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            n=len(v),
        )
    return res


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-segments", type=int, default=120)
    p.add_argument("--max-train-samples", type=int, default=200_000)
    p.add_argument("--train-stride", type=int, default=2, help="subsample rate when building X,Y")
    p.add_argument("--max-test-segments", type=int, default=13)
    p.add_argument("--stride", type=int, default=20, help="eval start stride on test logs")
    p.add_argument("--dob-tau", type=float, default=0.05)
    p.add_argument("--high-speed-only", action="store_true")
    p.add_argument("--speed-min", type=float, default=5.0, help="m/s body speed gate")
    p.add_argument("--no-neurobem-ref", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 8
        args.max_train_segments = 12
        args.max_train_samples = 12_000
        args.max_test_segments = 3
        args.stride = 60
        args.train_stride = 4

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if not PROCESSED.is_dir():
        raise SystemExit(f"Missing processed_data at {PROCESSED}")
    if not BEMNN.is_dir():
        print(f"WARNING: bem+nn not found at {BEMNN}; neurobem-ref arm disabled")
        args.no_neurobem_ref = True

    all_segs = segment_paths()
    test_ids = load_testset_ids()
    test_segs = [p for p in all_segs if is_test_segment(p, test_ids)]
    train_segs = [p for p in all_segs if p not in test_segs]
    if not test_segs:
        n_te = max(1, len(all_segs) // 10)
        test_segs, train_segs = all_segs[-n_te:], all_segs[:-n_te]

    train_segs = train_segs[: args.max_train_segments]
    test_segs = test_segs[: args.max_test_segments]
    speed_min = args.speed_min if args.high_speed_only else None
    print(f"train={len(train_segs)}  test={len(test_segs)}  bem+nn={BEMNN.is_dir()}")
    if speed_min is not None:
        print(f"high-speed gate: |v|_body >= {speed_min} m/s")

    c_t = fit_thrust_coeff(train_segs[: min(30, len(train_segs))])
    print(f"c_T={c_t:.6e}")

    X, Y = build_training_arrays(
        train_segs, c_t, args.max_train_samples, args.seed, stride=args.train_stride
    )
    print(f"train samples={len(X)}")
    net = train_mlp(X, Y, args.epochs, args.seed)

    modes = ["nominal", "residual", "dob"]
    if not args.no_neurobem_ref:
        modes.append("neurobem-ref")

    pooled = {m: {h: [] for h in HORIZONS_MS} for m in modes}
    for path in test_segs:
        print(f"eval {path.name}")
        for mode in modes:
            errs = predict_horizon_errors(
                path,
                c_t,
                net if mode == "residual" else None,
                mode=mode,
                horizons_ms=HORIZONS_MS,
                dob_tau=args.dob_tau,
                stride=args.stride,
                speed_min=speed_min,
            )
            for h, vals in errs.items():
                pooled[mode][h].extend(vals)

    table = {m: aggregate(pooled[m]) for m in modes}

    hdr = "".join(f"{m:>16}" for m in modes)
    print(f"\nOpen-loop position error [m] (mean ± SE)")
    print(f"{'Horizon':>8}{hdr}")
    rows = []
    for h in HORIZONS_MS:
        cells = []
        row = {"horizon_ms": h}
        for m in modes:
            st = table[m][h]
            cells.append(f"{st['mean']:.4f}±{st['se']:.4f}")
            row[f"{m}_mean"] = st["mean"]
            row[f"{m}_se"] = st["se"]
            row[f"{m}_n"] = st["n"]
        print(f"{h:>6} ms " + " ".join(f"{c:>16}" for c in cells))
        rows.append(row)

    nom = table["nominal"][500]["mean"]
    for m in modes:
        if m == "nominal":
            continue
        mu = table[m][500]["mean"]
        pct = 100.0 * (nom - mu) / nom if nom and nom == nom else float("nan")
        print(f"reduction @500 ms ({m} vs nominal): {pct:.1f}%")

    tag = "hispeed" if args.high_speed_only else "all"
    out_csv = RESULTS / f"neurobem_prediction_table_{tag}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    meta = dict(
        c_t=c_t,
        modes=modes,
        n_train_segments=len(train_segs),
        n_test_segments=len(test_segs),
        n_train_samples=int(len(X)),
        epochs=args.epochs,
        dob_tau=args.dob_tau,
        speed_min=speed_min,
        bemnn_dir=str(BEMNN),
        residual="MLP force/torque residual; features=vel_b+omega+motors",
        note=(
            "Attitude from measured quaternion during rollout (logged-input). "
            "neurobem-ref uses published bem+nn wrench (pred+res), not our learner."
        ),
    )
    (RESULTS / f"neurobem_prediction_meta_{tag}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_csv}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        styles = {"nominal": "o-", "residual": "s-", "dob": "^-", "neurobem-ref": "d--"}
        for m in modes:
            means = [table[m][h]["mean"] for h in HORIZONS_MS]
            ses = [table[m][h]["se"] for h in HORIZONS_MS]
            ax.errorbar(HORIZONS_MS, means, yerr=ses, fmt=styles.get(m, "o-"), ms=4, label=m)
        ax.set_xlabel("prediction horizon [ms]")
        ax.set_ylabel("position error [m]")
        title = "NeuroBEM open-loop prediction"
        if speed_min is not None:
            title += f" (|v|≥{speed_min} m/s)"
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig_path = FIGURES / f"neurobem_prediction_horizons_{tag}.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
