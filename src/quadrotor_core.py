"""Hybrid quadrotor: physics-based nominal model + learned residual."""
import numpy as np

rng = np.random.default_rng(0)
g = 9.81
DT = 0.01                     # control / integration step

# ------------------------------------------------------------- parameters
class P:                      # nominal (what the controller believes)
    m = 1.00
    J = np.diag([0.010, 0.010, 0.018])
    l, cT, cQ = 0.20, 1.0e-5, 1.6e-7

class HARD:                   # stochastic / non-stationary plant
    tau_w = 0.6               # turbulence correlation time [s]
    sigma_w = 2.6             # turbulence intensity [m/s^2 / sqrt(s)]
    sag = 0.07                # fractional thrust loss over the episode
    kd_spread = 0.30          # per-episode drag variation (+/-)
    hyst = 0.25               # rate-dependent (hysteretic) drag
    L_turb = 2.0              # Dryden scale length [m]
    V_ref = 1.5               # reference airspeed [m/s]

class T(P):                   # true plant
    m = 1.12                                   # +12% mass mismatch
    J = np.diag([0.0125, 0.0115, 0.0205])      # inertia mismatch
    kd = np.diag([0.32, 0.32, 0.45])           # body-frame rotor drag
    kw = np.diag([0.006, 0.006, 0.008])        # aerodynamic damping
    tau_m = 0.04                               # motor lag [s]

# --------------------------------------------------------- small SO(3) ops
def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def R_of(eta):                                  # ZYX Euler -> R, Eq. (4)
    ph, th, ps = eta
    cp, sp, ct, st, cs, ss = np.cos(ph), np.sin(ph), np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)
    return np.array([
        [cs * ct, cs * st * sp - ss * cp, cs * st * cp + ss * sp],
        [ss * ct, ss * st * sp + cs * cp, ss * st * cp - cs * sp],
        [-st,     ct * sp,                ct * cp]])

def W_of(eta):                                  # Euler-rate map, Eq. (3)
    ph, th, _ = eta
    return np.array([[1, np.sin(ph) * np.tan(th), np.cos(ph) * np.tan(th)],
                     [0, np.cos(ph),             -np.sin(ph)],
                     [0, np.sin(ph) / np.cos(th), np.cos(ph) / np.cos(th)]])

e3 = np.array([0.0, 0.0, 1.0])

# ---------------------------------------------------- nominal / true dynamics
def f_nom(x, u):
    """xdot = f_nom(x,u), Eq. (8).  x = [p(3) v(3) eta(3) w(3)], u = [T, tau]."""
    v, eta, w = x[3:6], x[6:9], x[9:12]
    Tt, tau = u[0], u[1:4]
    a = R_of(eta) @ e3 * (Tt / P.m) - g * e3
    wd = np.linalg.solve(P.J, tau - np.cross(w, P.J @ w))
    return np.concatenate([v, a, W_of(eta) @ w, wd])

def f_true(x, u):
    """Perturbed plant: mass/inertia mismatch + rotor drag + rotational damping."""
    v, eta, w = x[3:6], x[6:9], x[9:12]
    Tt, tau = u[0], u[1:4]
    R = R_of(eta)
    d_v = -R @ T.kd @ R.T @ v
    a = R @ e3 * (Tt / T.m) - g * e3 + d_v / T.m
    d_w = -T.kw @ (w * np.abs(w))
    wd = np.linalg.solve(T.J, tau - np.cross(w, T.J @ w) + d_w)
    return np.concatenate([v, a, W_of(eta) @ w, wd])

def rk4(f, x, u, dt=DT):
    k1 = f(x, u); k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u); k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# ------------------------------------------------- allocation + saturation
ALLOC = np.array([[P.cT, P.cT, P.cT, P.cT],
                  [0, P.l * P.cT, 0, -P.l * P.cT],
                  [-P.l * P.cT, 0, P.l * P.cT, 0],
                  [P.cQ, -P.cQ, P.cQ, -P.cQ]])
ALLOC_INV = np.linalg.inv(ALLOC)
W2MAX = 900.0 ** 2

def saturate(u):
    """Round-trip u -> rotor speeds^2 -> u, clipped to feasible range, Eq. (7)."""
    w2 = np.clip(ALLOC_INV @ u, 0.0, W2MAX)
    return ALLOC @ w2

# ------------------------------------------------- geometric SE(3) control
Kp, Kv = np.diag([6.0] * 3), np.diag([4.0] * 3)
KR, Kw = np.diag([0.9, 0.9, 0.5]), np.diag([0.16, 0.16, 0.10])
EI_MAX = 1.5                                   # anti-windup clamp

def controller(x, ref, comp=None, ki=0.0, ei=None):
    """u = [T, tau]. comp = learned feed-forward; ki = integral gain (0 -> pure PD)."""
    p, v, eta, w = x[:3], x[3:6], x[6:9], x[9:12]
    p_r, v_r, a_r, psi_r = ref
    da = np.zeros(3) if comp is None else comp[0]
    dal = np.zeros(3) if comp is None else comp[1]
    ei = np.zeros(3) if ei is None else np.clip(ei + (p_r - p) * DT, -EI_MAX, EI_MAX)

    a_des = a_r + Kp @ (p_r - p) + Kv @ (v_r - v) + ki * ei - da   # Eq. (10) + I
    F_des = P.m * (a_des + g * e3)                                 # Eq. (11)
    R = R_of(eta)
    Tt = float(F_des @ (R @ e3))                                   # Eq. (12)

    zd = F_des / max(np.linalg.norm(F_des), 1e-6)
    yc = np.array([-np.sin(psi_r), np.cos(psi_r), 0.0])
    xd = np.cross(yc, zd); xd /= max(np.linalg.norm(xd), 1e-6)
    Rd = np.column_stack([xd, np.cross(zd, xd), zd])

    eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)                            # Eq. (13)
    tau = -KR @ eR - Kw @ w + np.cross(w, P.J @ w) - P.J @ dal     # Eq. (15)
    return saturate(np.concatenate([[Tt], tau])), ei

# --------------------------------------------------------- reference orbits
def traj(t, kind):
    if kind == "circle":
        A, wf = 1.2, 0.8
        p = np.array([A * np.cos(wf * t), A * np.sin(wf * t), 1.5])
        v = np.array([-A * wf * np.sin(wf * t), A * wf * np.cos(wf * t), 0])
        a = np.array([-A * wf**2 * np.cos(wf * t), -A * wf**2 * np.sin(wf * t), 0])
    elif kind == "fig8":
        A, wf = 1.5, 0.9
        p = np.array([A * np.sin(wf * t), A * np.sin(wf * t) * np.cos(wf * t),
                      1.2 + 0.3 * np.sin(0.5 * wf * t)])
        v = np.array([A * wf * np.cos(wf * t), A * wf * np.cos(2 * wf * t),
                      0.15 * wf * np.cos(0.5 * wf * t)])
        a = np.array([-A * wf**2 * np.sin(wf * t), -2 * A * wf**2 * np.sin(2 * wf * t),
                      -0.075 * wf**2 * np.sin(0.5 * wf * t)])
    else:                                        # aggressive steps
        s = np.floor(t / 2.0)
        p = np.array([1.0 * ((-1) ** s), 0.8 * np.sin(s), 1.0 + 0.4 * (s % 2)])
        v = np.zeros(3); a = np.zeros(3)
    return p, v, a, 0.3 * np.sin(0.4 * t)

# ------------------------------------------------------------ residual net
IDX = np.r_[3:12]        # features: v, eta, w (position excluded)
NIN, NOUT, NH = 9 + 4, 6, 64

class MLP:
    """2-hidden-layer tanh MLP + Adam, predicting the velocity/rate residual."""
    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        dims = [NIN, NH, NH, NOUT]
        self.W = [r.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]
        self.mW = [np.zeros_like(w) for w in self.W]; self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]; self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self.xmu = np.zeros(NIN); self.xsd = np.ones(NIN)
        self.ymu = np.zeros(NOUT); self.ysd = np.ones(NOUT)

    def _fwd(self, Z):
        h1 = np.tanh(Z @ self.W[0] + self.b[0])
        h2 = np.tanh(h1 @ self.W[1] + self.b[1])
        return h1, h2, h2 @ self.W[2] + self.b[2]

    def predict(self, X):                       # Eq. (17), de-normalised
        Z = (np.atleast_2d(X) - self.xmu) / self.xsd
        return self._fwd(Z)[2] * self.ysd + self.ymu

    def step(self, Xb, Yb, lr=2e-3):
        n = len(Xb)
        h1, h2, out = self._fwd(Xb)
        d = 2 * (out - Yb) / n                  # dL/dout, Eq. (19)
        gW2, gb2 = h2.T @ d, d.sum(0)
        d2 = (d @ self.W[2].T) * (1 - h2**2)
        gW1, gb1 = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ self.W[1].T) * (1 - h1**2)
        gW0, gb0 = Xb.T @ d1, d1.sum(0)
        self.t += 1
        for i, (gw, gb) in enumerate([(gW0, gb0), (gW1, gb1), (gW2, gb2)]):
            for p, m, v, gr in ((self.W[i], self.mW, self.vW, gw),
                                (self.b[i], self.mb, self.vb, gb)):
                m[i][:] = 0.9 * m[i] + 0.1 * gr
                v[i][:] = 0.999 * v[i] + 0.001 * gr**2
                p -= lr * (m[i] / (1 - 0.9**self.t)) / (np.sqrt(v[i] / (1 - 0.999**self.t)) + 1e-8)
        return float(np.mean((out - Yb) ** 2))

def features(x, u):
    return np.concatenate([x[IDX], u])

# ----------------------------------------------------------- rollout engine
def rollout(kind, Tend=12.0, net=None, gust=None, seed=0, log=False, ki=0.0,
            plant="smooth"):
    """Simulate the true plant under the (optionally residual-compensated) controller."""
    r = np.random.default_rng(seed)
    kd_scale = 1.0 + (r.uniform(-HARD.kd_spread, HARD.kd_spread)
                      if plant in ("hard", "dryden") else 0.0)
    w_turb = np.zeros(3)
    n = int(Tend / DT)
    x = np.zeros(12); x[:3] = traj(0, kind)[0] + r.normal(0, 0.05, 3); x[2] = max(x[2], 0.2)
    u = np.array([P.m * g, 0, 0, 0]); u_act = u.copy()
    X, Y, err, P_log, C_log, S_log = [], [], [], [], [], []
    comp, ei = None, np.zeros(3)
    dob = np.zeros(3); v_prev = None
    for k in range(n):
        t = k * DT
        ref = traj(t, kind)
        if net is not None:                              # learned correction
            d = net.predict(features(x, u_act))[0]
            comp = (d[:3] / DT, d[3:] / DT)
        if isinstance(ki, str) and ki.startswith("dob"):  # INDI-style observer
            tau_f = float(ki.split(":")[1])
            if v_prev is not None:
                a_meas = (x[3:6] - v_prev) / DT
                a_model = R_of(x[6:9]) @ e3 * (u_act[0] / P.m) - g * e3
                dob += (DT / tau_f) * ((a_meas - a_model) - dob)
            v_prev = x[3:6].copy()
            comp_dob = (dob, np.zeros(3))
            if comp is not None:
                comp_dob = (dob + comp[0], comp[1])
            u, ei = controller(x, ref, comp_dob, 0.0, ei)
        elif isinstance(ki, tuple):                       # speed-scheduled gain
            lo, hi, vmx = ki
            ki_eff = lo + (hi - lo) * min(np.linalg.norm(x[3:6]) / vmx, 1.0)
            u, ei = controller(x, ref, comp, ki_eff, ei)
        else:
            u, ei = controller(x, ref, comp, ki, ei)
        u_act += (u - u_act) * (DT / T.tau_m)             # motor lag
        x_nom = rk4(f_nom, x, u_act)
        if plant in ("hard", "dryden"):
            u_p = u_act.copy()
            u_p[0] *= (1.0 - HARD.sag * t / Tend)         # battery sag
            kd_save = T.kd; T.kd = kd_save * kd_scale
            x_new = rk4(f_true, x, u_p); T.kd = kd_save
            if plant == "dryden":
                V = max(np.linalg.norm(x[3:6]), 0.3)
                tw = HARD.L_turb / V
                sg = HARD.sigma_w * np.sqrt(V / HARD.V_ref)
                w_turb += (-w_turb / tw) * DT + sg * np.sqrt(DT) * r.normal(0, 1, 3)
            else:
                w_turb += (-w_turb / HARD.tau_w) * DT + HARD.sigma_w * np.sqrt(DT) * r.normal(0, 1, 3)
            x_new[3:6] += w_turb * DT
            x_new[3:6] -= HARD.hyst * np.tanh(5 * x[3:6]) * DT
        else:
            x_new = rk4(f_true, x, u_act)
        if gust is not None and 4.0 <= t < 5.0:
            x_new[3:6] += gust * DT
        x_new[3:6] += r.normal(0, 2e-3, 3)                # process noise
        if log:
            X.append(features(x, u_act))
            Y.append(np.concatenate([x_new[3:6] - x_nom[3:6],       # Eq. (16)
                                     x_new[9:12] - x_nom[9:12]]))
        err.append(np.linalg.norm(x[:3] - ref[0]))
        P_log.append(np.concatenate([[t], x[:3], ref[0], u_act]))
        _kilog = ki * ei if isinstance(ki, (int, float)) else np.zeros(3)
        C_log.append(np.concatenate([comp[0] if comp is not None else np.zeros(3), _kilog]))
        S_log.append(np.concatenate([x, u_act]))
        x = x_new
        if not np.isfinite(x).all() or abs(x[6]) > 1.4 or abs(x[7]) > 1.4:
            break
    rmse = float(np.sqrt(np.mean(np.square(err[200:]))))   # skip transient
    Pa = np.array(P_log)
    u_hov = np.array([P.m * g, 0, 0, 0])
    effort = float(DT * np.sum(np.linalg.norm(Pa[:, 7:11] - u_hov, axis=1)))
    if log == "traj":
        return dict(P=Pa, C=np.array(C_log), S=np.array(S_log), rmse=rmse,
                    effort=effort, err=np.array(err))
    return (np.array(X), np.array(Y), rmse) if log else (rmse, effort)

# --------------------------------------------------------- training routine
def train_residual(train_kinds, seed=1, epochs=300, verbose=False, plant="smooth"):
    """Collect data with the PD controller, then fit the residual network."""
    r = np.random.default_rng(100 + seed)
    Xs, Ys = [], []
    for s, kind in enumerate(train_kinds):
        Xi, Yi, _ = rollout(kind, 12.0, seed=s, log=True, plant=plant)
        Xs.append(Xi); Ys.append(Yi)
    X, Y = np.vstack(Xs), np.vstack(Ys)
    i = r.permutation(len(X)); X, Y = X[i], Y[i]; ntr = int(0.8 * len(X))
    net = MLP(seed)
    net.xmu, net.xsd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-8
    net.ymu, net.ysd = Y[:ntr].mean(0), Y[:ntr].std(0) + 1e-8
    Ztr, Wtr = (X[:ntr] - net.xmu) / net.xsd, (Y[:ntr] - net.ymu) / net.ysd
    Zva, Wva = (X[ntr:] - net.xmu) / net.xsd, (Y[ntr:] - net.ymu) / net.ysd
    best, bw, hist = np.inf, None, []
    for ep in range(epochs):
        p = r.permutation(len(Ztr))
        for j in range(0, len(p), 256):
            net.step(Ztr[p[j:j + 256]], Wtr[p[j:j + 256]])
        vl = float(np.mean((net._fwd(Zva)[2] - Wva) ** 2))
        hist.append((float(np.mean((net._fwd(Ztr)[2] - Wtr) ** 2)), vl))
        if vl < best:
            best, bw = vl, ([w.copy() for w in net.W], [b.copy() for b in net.b])
    net.W, net.b = bw                       # best-validation checkpoint
    if verbose:
        print(f"seed {seed}: {len(X)} samples, best val MSE = {best:.4f}")
    return net, best, np.array(hist)
