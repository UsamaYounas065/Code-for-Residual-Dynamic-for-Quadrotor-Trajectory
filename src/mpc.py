"""MPPI-based MPC for the residual quadrotor study (Usama Colab export)."""
from __future__ import annotations

import numpy as np

import quadrotor_core as H

K, HOR, DTM = 256, 15, 0.05
SIG, LAM = 2.5, 1.0
QP, QV, RU = 1.0, 0.10, 0.004
RSCALE = 1.0  # residual trust factor inside the horizon (paper Fig. 6c / Sec. 5.9)


def feats_from_acc(v, a, psi):
    F = H.P.m * (a + H.g * H.e3)
    T = np.linalg.norm(F, axis=-1)
    z = F / np.maximum(T[..., None], 1e-6)
    sp, cp = np.sin(psi), np.cos(psi)
    phi = np.arcsin(np.clip(z[..., 0] * sp - z[..., 1] * cp, -1, 1))
    th = np.arctan2(z[..., 0] * cp + z[..., 1] * sp, np.maximum(z[..., 2], 1e-6))
    eta = np.stack([phi, th, np.full_like(phi, psi)], -1)
    w = np.zeros_like(eta)
    u = np.stack([T, np.zeros_like(T), np.zeros_like(T), np.zeros_like(T)], -1)
    return np.concatenate([v, eta, w, u], -1)


def mppi(p, v, refs, psi, net, a_prev, rng, dhat=None, rscale=None):
    rs = RSCALE if rscale is None else rscale
    A = a_prev[None] + SIG * rng.normal(0, 1, (K, HOR, 3))
    A[:, :, 2] = np.clip(A[:, :, 2], -8, 8)
    P = np.tile(p, (K, 1))
    V = np.tile(v, (K, 1))
    cost = np.zeros(K)
    for j in range(HOR):
        a = A[:, j]
        if net is not None:
            d = rs * net.predict(feats_from_acc(V, a, psi))[:, :3] / H.DT
            a_eff = a + d
        elif dhat is not None:
            a_eff = a + dhat
        else:
            a_eff = a
        V = V + a_eff * DTM
        P = P + V * DTM
        cost += (
            QP * np.sum((P - refs[j, :3]) ** 2, 1)
            + QV * np.sum((V - refs[j, 3:]) ** 2, 1)
            + RU * np.sum(a ** 2, 1)
        )
    wgt = np.exp(-(cost - cost.min()) / LAM)
    wgt /= wgt.sum()
    return np.einsum("k,kij->ij", wgt, A)


def run(
    kind,
    net=None,
    seed=0,
    Tend=12.0,
    plant="smooth",
    gust=None,
    dob_tau=None,
    rscale=None,
):
    """Closed-loop MPC RMSE (skip first 2 s). dob_tau enables offset-free DOB."""
    r = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 999)
    kd_scale = 1.0 + (
        r.uniform(-H.HARD.kd_spread, H.HARD.kd_spread) if plant in ("hard", "dryden") else 0.0
    )
    w_turb = np.zeros(3)
    x = np.zeros(12)
    x[:3] = H.traj(0, kind)[0] + r.normal(0, 0.05, 3)
    x[2] = max(x[2], 0.2)
    u_act = np.array([H.P.m * H.g, 0, 0, 0])
    A = np.zeros((HOR, 3))
    err = []
    dhat = np.zeros(3)
    v_prev = None
    steps_per_mpc = int(round(DTM / H.DT))
    for k in range(int(Tend / H.DT)):
        t = k * H.DT
        p_r, v_r, a_r, psi_r = H.traj(t, kind)
        R = H.R_of(x[6:9])
        if dob_tau:
            if v_prev is not None:
                a_meas = (x[3:6] - v_prev) / H.DT
                a_model = R @ H.e3 * (u_act[0] / H.P.m) - H.g * H.e3
                dhat += (H.DT / dob_tau) * ((a_meas - a_model) - dhat)
            v_prev = x[3:6].copy()
        if k % steps_per_mpc == 0:
            refs = np.array(
                [np.concatenate(H.traj(t + (j + 1) * DTM, kind)[:2]) for j in range(HOR)]
            )
            A = mppi(
                x[:3],
                x[3:6],
                refs,
                psi_r,
                net,
                np.vstack([A[1:], A[-1:]]),
                rng,
                dhat if dob_tau else None,
                rscale=rscale,
            )
        a_des = A[0] + H.Kp @ (p_r - x[:3]) + H.Kv @ (v_r - x[3:6])
        F = H.P.m * (a_des + H.g * H.e3)
        T = float(F @ (R @ H.e3))
        z = F / max(np.linalg.norm(F), 1e-6)
        yc = np.array([-np.sin(psi_r), np.cos(psi_r), 0.0])
        xd = np.cross(yc, z)
        xd /= max(np.linalg.norm(xd), 1e-6)
        Rd = np.column_stack([xd, np.cross(z, xd), z])
        eR = 0.5 * H.vee(Rd.T @ R - R.T @ Rd)
        tau = -H.KR @ eR - H.Kw @ x[9:12] + np.cross(x[9:12], H.P.J @ x[9:12])
        u_act += (H.saturate(np.concatenate([[T], tau])) - u_act) * (H.DT / H.T.tau_m)
        if plant in ("hard", "dryden"):
            up = u_act.copy()
            up[0] *= 1 - H.HARD.sag * t / Tend
            ks = H.T.kd
            H.T.kd = ks * kd_scale
            xn = H.rk4(H.f_true, x, up)
            H.T.kd = ks
            if plant == "dryden":
                V = max(np.linalg.norm(x[3:6]), 0.3)
                tw = H.HARD.L_turb / V
                sg = H.HARD.sigma_w * np.sqrt(V / H.HARD.V_ref)
                w_turb += (-w_turb / tw) * H.DT + sg * np.sqrt(H.DT) * r.normal(0, 1, 3)
            else:
                w_turb += (
                    (-w_turb / H.HARD.tau_w) * H.DT
                    + H.HARD.sigma_w * np.sqrt(H.DT) * r.normal(0, 1, 3)
                )
            xn[3:6] += w_turb * H.DT - H.HARD.hyst * np.tanh(5 * x[3:6]) * H.DT
        else:
            xn = H.rk4(H.f_true, x, u_act)
        if gust is not None and 4.0 <= t < 5.0:
            xn[3:6] += gust * H.DT
        xn[3:6] += r.normal(0, 2e-3, 3)
        err.append(np.linalg.norm(x[:3] - p_r))
        x = xn
        if not np.isfinite(x).all() or abs(x[6]) > 1.4 or abs(x[7]) > 1.4:
            return float("nan")
    return float(np.sqrt(np.mean(np.square(err[200:]))))
