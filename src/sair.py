"""Phase 5 - SAIR simulation, fitting and validation (docs/SAIR_paper.md §5).

State = fractions of N: S, I, A, R. Every model is one explicit Euler scheme (events hit exactly)
    dS = -b_I*I*S - b_A*A*S - (x_I + x_A)*S
    dI =  b_I*I*S + x_I*S - g*I
    dA =  b_A*A*S + x_A*S - e*A
    dR =  g*I + e*A
with the rates of each model:
    (a)  const          constant-rate SAIR (Eq. 22): b_I = beta, b_A = alpha, g = gamma, e = eps, x = 0
    (a+) const_plus     (a) + Bass external influence x_I = eta_I, x_A = eta_A (D57)
    (b)  uniform        Eq. 26, Mult_eff = kappa for everyone: b_I = d1, b_A = d2, g = 1 - d1, e = 1 - d2
    (c)  eq26           Eq. 26 (corrected), data-driven d1 (D45)
    (d)  sairh          SAIR-H: b_I = beta*d1, b_A = alpha*d2, g = gamma, e = eps (D50)
    (d+) sairh_plus     (d) + Bass external influence (D57)
         sairh_uniform  as (d) with Mult_eff = kappa for everyone
    (e)  meanfield      as (d) with every user's P1 = the model's global I/(I + A)
Push rates d1(t), d2 = 1 - d1 (D45), recomputed every data bin (1 h for the events, 15 min for the story):
    rho_I = C_I_model / C_I_data (1 if data is 0; clipped to [0, 5]), rho_A the same
    P1_i  = rho_I*n_I,i / (rho_I*n_I,i + rho_A*n_A,i), n = friends who entered I / A in the data by t;
            no sided friends -> I/(I + A) of the model state (0.5 if both 0)
    Drif1_i = sigmoid((2*P1_i - 1) * Mult_eff_i(t)),  Mult_eff_i(t) = kappa * pct_rank(eta_i(t)), Pop at t
    d1 = mean Drif1 over users who have not entered in the data by t
Source injection (D46): at each supporting / denying source tweet's time, theta * w_s / sum(w) moves S -> I / A.
Euler dt = bin / 50, halved until (largest rate) * dt <= 0.025 (D53, D58).
Fit (D51, D58): least squares on C_I/C_I,final and C_A/C_A,final at the data bins, bounded Nelder-Mead in
log space for the rates and the half-life, 20 random starts (seed 42). AIC / BIC from the SSE.
The main model for Phases 6-8 is whichever of (d) / (d+) has the lower AIC on Charlie Hebdo (D61).
Stage 2 validation (D59, D60, D63) for (a), (a+) and the main model: forecast after max(50% point of C_I,
10th bin), parameter recovery (main model), moving-block residual bootstrap with a cold-start check.
The Euler steps between events run in a numba-compiled kernel, identical to the Python loop (D62).
Every finished optimiser run is checkpointed in data/processed/checkpoints/ and skipped on a restart.

Usage:
    python src/sair.py --check               stage 1 (fits) then stage 2 (validation), each followed by its check
    python src/sair.py --stage 2 --check     one stage only
    python src/sair.py --fresh ...           delete the stages' checkpoints first (needed after any change to the setup)
    python src/sair.py --check-only          checks on the saved results
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

import data_prep as dp
import drive as drv
import features as ft

SETS = ft.SETS
STEPS_PER_BIN = 50                           # Euler dt = bin / 50 (D47, D53) ...
RATE_STEP = 0.025                            # ... halved until the largest rate * dt <= 0.025 (D58)
RHO_MAX = 5.0
N_STARTS = 20
N_WORKERS = 12
MIN_CONVERGED = 15                           # of N_STARTS, for every main fit
N_RECOVERY_STARTS, RECOVERY_SEEDS, RECOVERY_SIGMA = 10, [1, 2, 3, 4, 5], 0.01
N_BOOT, BLOCK_LEN, N_COLD, N_COLD_STARTS = 50, 5, 5, 5
MAX_CROSS_ITER = 3                           # up/down cross-model seeding iterations in stage 1 (D64)
KAPPA_VERSION = "k50"                        # checkpoints of the kappa-models refit with kappa in [0, 50] (D65)
N_FRESH = 5                                  # fresh random starts over kappa in [0, 50] per main candidate (D65)
FC_MIN_BIN, FC_MIN_TEST = 9, 5               # forecast cut >= the 10th bin; N/A if < 5 bins remain (D59)
XATOL, FATOL = 1e-4, 1e-9                    # Nelder-Mead tolerances (fitting space, MSE)
INV_TOL = 1e-9
BOUND_TOL = 0.01                             # flag a parameter within 1% of the range from a bound
POP_GRID = np.linspace(0, 1, 201)            # Mult_eff rank table resolution in Pop (D45)
CKPT_DIR = os.path.join(dp.PROCESSED, "checkpoints")   # per-job results for restarts (D62)

MODELS = ["const", "const_plus", "uniform", "eq26", "sairh", "sairh_plus", "sairh_uniform", "sairh_plus_uniform",
          "meanfield"]
MAIN_CANDIDATES = ["sairh", "sairh_plus", "sairh_uniform", "sairh_plus_uniform"]      # D66
RATES = ["beta", "alpha", "gamma", "eps"]
BASS = ["eta_I", "eta_A"]
NAMES = {"const": RATES + ["theta"],
         "const_plus": RATES + ["theta"] + BASS,
         "uniform": ["kappa", "theta"],
         "eq26": ["half_life_h", "kappa", "theta"],
         "sairh": RATES + ["half_life_h", "kappa", "theta"],
         "sairh_plus": RATES + ["half_life_h", "kappa", "theta"] + BASS,
         "sairh_uniform": RATES + ["kappa", "theta"],
         "sairh_plus_uniform": RATES + ["kappa", "theta"] + BASS,
         "meanfield": RATES + ["half_life_h", "kappa", "theta"]}
ALL_PARAMS = RATES + ["half_life_h", "kappa", "theta"] + BASS
RANGE = {**{n: (1e-3, 50.0) for n in RATES}, "half_life_h": (0.5, 168.0), "kappa": (0.0, 50.0), "theta": (0.0, 1.0),
         "eta_I": (0.0, 1.0), "eta_A": (0.0, 1.0)}
LOG_PARAMS = set(RATES) | {"half_life_h"}                                # fitted on a log scale
MAX_FEV = {"const": 3000, "const_plus": 4000, "uniform": 1000, "eq26": 1500, "sairh": 6000, "sairh_plus": 8000,
           "sairh_uniform": 5000, "sairh_plus_uniform": 7000, "meanfield": 6000}
# how each model gets d1: P1 from the friend network or the global share; Mult_eff ranked or uniform
P1_MODE = {"uniform": "network", "eq26": "network", "sairh": "network", "sairh_plus": "network",
           "sairh_plus_uniform": "network",
           "sairh_uniform": "network", "meanfield": "global"}
ME_MODE = {"uniform": "uniform", "eq26": "rank", "sairh": "rank", "sairh_plus": "rank", "sairh_uniform": "uniform",
           "sairh_plus_uniform": "uniform",
           "meanfield": "rank"}
EQ26_KINDS = ("uniform", "eq26")
MODEL_LABEL = {"const": "(a) constant-rate SAIR", "const_plus": "(a+) constant-rate + external",
               "uniform": "(b) Eq. 26, uniform Mult_eff", "eq26": "(c) Eq. 26", "sairh": "(d) SAIR-H",
               "sairh_plus": "(d+) SAIR-H + external", "sairh_uniform": "(d) SAIR-H, uniform Mult_eff",
               "sairh_plus_uniform": "(d+) SAIR-H + external, uniform Mult_eff",
               "meanfield": "(e) SAIR-H mean-field"}
MODEL_COLORS = {"const": "#2a78d6", "const_plus": "#0b4f9c", "uniform": "#9b9a94", "eq26": "#eda100",
                "sairh": "#1baf7a", "sairh_plus": "#0d6b49", "sairh_uniform": "#7a5af5", "sairh_plus_uniform": "#d6336c", "meanfield": "#eb6834"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def euler_step(S, I, A, R, b_i, b_a, g, e, dt, x_i=0.0, x_a=0.0):
    """One explicit Euler step of the general SAIR scheme (x = Bass external influence);
    returns the new state and the step's inflows into I and A."""
    new_i = b_i * I * S + x_i * S
    new_a = b_a * A * S + x_a * S
    return (S - (new_i + new_a) * dt, I + (new_i - g * I) * dt, A + (new_a - e * A) * dt,
            R + (g * I + e * A) * dt, new_i * dt, new_a * dt)


def to_fit(names, x):
    """Model parameters -> fitting coordinates (log for rates and the half-life)."""
    return np.array([np.log(v) if n in LOG_PARAMS else v for n, v in zip(names, x)], dtype=float)


def from_fit(names, u):
    return np.array([np.exp(v) if n in LOG_PARAMS else v for n, v in zip(names, u)], dtype=float)


def fit_bounds(names):
    return [(np.log(RANGE[n][0]), np.log(RANGE[n][1])) if n in LOG_PARAMS else RANGE[n] for n in names]


def bound_flags(names, x):
    """Parameters within 1% of the fitting-space range from a bound, e.g. 'kappa@low'."""
    flags = []
    for n, u, (lo, hi) in zip(names, to_fit(names, x), fit_bounds(names)):
        if u - lo <= BOUND_TOL * (hi - lo):
            flags.append(f"{n}@low")
        elif hi - u <= BOUND_TOL * (hi - lo):
            flags.append(f"{n}@high")
    return ";".join(flags)


def free_names(kind, fixed=None):
    return [n for n in NAMES[kind] if n not in (fixed or {})]


def full_params(kind, free, x_free, fixed=None):
    """Assemble the full parameter vector (NAMES order) from free values and fixed ones."""
    p = {**(fixed or {}), **dict(zip(free, x_free))}
    return np.array([p[n] for n in NAMES[kind]], dtype=float)


def step_dt(sc, kind, params):
    """bin / 50, halved until the largest rate * dt <= RATE_STEP (piecewise constant in the parameters)."""
    p = dict(zip(NAMES[kind], params))
    rate = max([1.0] + [p[n] for n in RATES if n in p] + [p.get("eta_I", 0.0) + p.get("eta_A", 0.0)])
    dt = sc["bin_h"] / STEPS_PER_BIN
    while rate * dt > RATE_STEP:
        dt /= 2
    return dt


# ---------------------------------------------------------------- set context

def rank_table_of(model, user_trait, static):
    """Each user's percentile rank of eta_i(Pop) = base_i + slope_i * Pop on POP_GRID (D42, D45);
    the regression coefficients, scaler and reference eta sample are the fitted Phase 4 ones."""
    coef, mu, sd = np.array(model["coef"]), np.array(model["scaler_mean"]), np.array(model["scaler_scale"])
    base = model["intercept"] + ((static - mu[1:]) / sd[1:]) @ coef[1:] - coef[0] * mu[0] / sd[0]
    slope = coef[0] * user_trait / sd[0]
    eta_sorted = np.array(model["eta_sorted"])
    eta_u = np.unique(eta_sorted)
    rank_u = drv.mult_eff(eta_u, eta_sorted, 1.0)
    return np.array([np.interp(base + slope * q, eta_u, rank_u, left=0.0, right=1.0) for q in POP_GRID])


def set_context(key, variant="profile"):
    """Data and precomputed arrays for one set (independent of the fitted parameters)."""
    ctx = drv.make_context(key, variant)
    curves = pd.read_csv(os.path.join(dp.PROCESSED, f"real_curves_{key}.csv"))
    tweets = pd.read_csv(os.path.join(dp.PROCESSED, f"tweets_{key}.csv"), dtype={"user_id": str, "tweet_id": str})
    users = pd.read_csv(os.path.join(dp.PROCESSED, f"users_{key}.csv"), dtype={"user_id": str})
    N = len(ctx["users"])
    grid = curves["t_hours"].to_numpy()
    bin_h = float(grid[1] - grid[0])
    upd = np.arange(0, grid[-1] + 1e-9, bin_h)                 # delta update times = data bins

    side, t_enter, M = ctx["side"], ctx["t_enter"], ctx["M"]
    n_i = np.empty((len(upd), N), dtype=np.float32)
    n_a = np.empty((len(upd), N), dtype=np.float32)
    not_entered = np.empty((len(upd), N), dtype=bool)
    c_i_data = np.empty(len(upd))
    c_a_data = np.empty(len(upd))
    for k, t in enumerate(upd):
        entered = t_enter <= t
        n_i[k] = M @ (entered & (side == 1)).astype(float)
        n_a[k] = M @ (entered & (side == 2)).astype(float)
        not_entered[k] = ~entered
        c_i_data[k] = (entered & (side == 1)).sum()
        c_a_data[k] = (entered & (side == 2)).sum()

    t0 = pd.to_datetime(users["first_t"], utc=True).min()
    src = tweets[(tweets["kind"] == "source") & tweets["stance"].isin(["support", "deny"])].copy()
    src["t_h"] = (pd.to_datetime(src["created_at"], utc=True) - t0) / pd.Timedelta(hours=1)
    src["w"] = np.log1p(src["followers_count"].astype(float))
    src = src.sort_values("t_h")

    c_i_curve, c_a_curve = curves["C_I"].to_numpy() / N, curves["C_A"].to_numpy() / N
    return {"key": key, "variant": variant, "N": N, "users": ctx["users"], "grid": grid, "bin_h": bin_h, "upd": upd,
            "end_h": float(grid[-1]), "n_i": n_i, "n_a": n_a, "not_entered": not_entered,
            "c_i_data": c_i_data, "c_a_data": c_a_data, "drive_model": ctx["model"],
            "user_trait": ctx["user_trait"], "static": ctx["static"],
            "rank_table": rank_table_of(ctx["model"], ctx["user_trait"], ctx["static"]), "stories": ctx["stories"],
            "src_t": src["t_h"].to_numpy(), "src_to_i": (src["stance"] == "support").to_numpy(),
            "src_share": (src["w"] / src["w"].sum()).to_numpy(),
            "I0": c_i_curve[0], "A0": c_a_curve[0], "C_I": c_i_curve, "C_A": c_a_curve,
            "fin_I": c_i_curve[-1], "fin_A": c_a_curve[-1],
            "I_display": curves["I"].to_numpy() / N, "A_display": curves["A"].to_numpy() / N,
            "pop_one": False, "p1_mode": None}


_CONTEXTS = {}


def get_context(key):
    """Cached context (one copy per process; workers build their own on first use).
    key = set name, or (set name, knock-out name) for a Phase 6 knock-out context (src/ablation.py)."""
    if key not in _CONTEXTS:
        if isinstance(key, tuple):
            import ablation                                    # Phase 6 knock-outs build on the set context
            _CONTEXTS[key] = ablation.ko_context(*key)
        else:
            _CONTEXTS[key] = set_context(key)
    return _CONTEXTS[key]


def drif1_vector(sc, k, c_i, c_a, I, A, kappa, pop_k, kind, freeze_k=None):
    """Drif1 of every user at update k, and the mask of users not yet entered in the data (D45).
    freeze_k: data inputs (friend counts, C_data, not-entered mask) are taken at min(k, freeze_k) (forecast, D52)."""
    kd = k if freeze_k is None else min(k, freeze_k)
    mask = sc["not_entered"][kd]
    if (sc["p1_mode"] or P1_MODE[kind]) == "global":
        p1 = I / (I + A) if I + A > 0 else 0.5
    else:
        N = sc["N"]
        rho_i = np.clip(c_i * N / sc["c_i_data"][kd], 0, RHO_MAX) if sc["c_i_data"][kd] > 0 else 1.0
        rho_a = np.clip(c_a * N / sc["c_a_data"][kd], 0, RHO_MAX) if sc["c_a_data"][kd] > 0 else 1.0
        num = rho_i * sc["n_i"][kd]
        den = num + rho_a * sc["n_a"][kd]
        fallback = I / (I + A) if I + A > 0 else 0.5
        p1 = np.where(den > 0, num / np.where(den > 0, den, 1.0), fallback)
    if ME_MODE[kind] == "uniform":
        me = kappa
    else:
        # percentile rank of eta_i(t) = base_i + slope_i * Pop(t), linear in Pop between tabulated grid points
        x = pop_k * (len(POP_GRID) - 1)
        g = min(int(x), len(POP_GRID) - 2)
        f = x - g
        me = kappa * ((1 - f) * sc["rank_table"][g] + f * sc["rank_table"][g + 1])
    return np.broadcast_to(sigmoid((2 * p1 - 1) * me), mask.shape), mask


def delta_at(sc, k, c_i, c_a, I, A, kappa, pop_k, kind, freeze_k=None):
    """d1 at update k: mean Drif1 over users not yet entered in the data (D45)."""
    d1, mask = drif1_vector(sc, k, c_i, c_a, I, A, kappa, pop_k, kind, freeze_k)
    return float(d1[mask].mean()) if mask.any() else float(d1.mean())   # plain float for the step loop


def pops_for(sc, p):
    """Pop at the update times (all ones when popularity is knocked out, Phase 6)."""
    if sc["pop_one"]:
        return np.ones(len(sc["upd"]))
    return ft.pop_at(sc["stories"], sc["upd"], sc["grid"], p["half_life_h"]) if "half_life_h" in p else None


# ---------------------------------------------------------------- simulation

@njit(cache=True)
def _advance(S, I, A, R, c_i, c_a, t, nxt, dt, b_i, b_a, g, e, x_i, x_a, buf, record):
    """Euler steps from event time t to the next event time nxt with constant rates (compiled with numba).
    Same arithmetic, step rule and order as euler_step in a Python loop, so results are identical.
    If record, the state after every intermediate step (not the arrival at nxt) is written to buf rows
    (t, S, I, A, R, C_I, C_A). Returns the state at nxt and the number of rows written."""
    n = 0
    while True:
        h = min(dt, nxt - t)
        new_i = b_i * I * S + x_i * S
        new_a = b_a * A * S + x_a * S
        S, I, A, R = S - (new_i + new_a) * h, I + (new_i - g * I) * h, A + (new_a - e * A) * h, R + (g * I + e * A) * h
        c_i, c_a = c_i + new_i * h, c_a + new_a * h
        if nxt - (t + h) < 1e-9:
            return S, I, A, R, c_i, c_a, n
        t = t + h
        if record:
            buf[n, 0], buf[n, 1], buf[n, 2], buf[n, 3], buf[n, 4], buf[n, 5], buf[n, 6] = t, S, I, A, R, c_i, c_a
            n += 1


def simulate(sc, kind, params, dt=None, record=False, end_h=None, freeze_k=None):
    """Run one model (kind in MODELS; params as in NAMES[kind]) up to end_h (default: the set's end).
    Source injections and delta updates happen in Python at event times; the Euler steps between events
    run in the compiled _advance. Returns arrays t, C_I, C_A at the event times (every data bin is one), or,
    if record, a DataFrame with S, I, A, R, C_I, C_A, d1, d2 and t at every Euler step."""
    p = dict(zip(NAMES[kind], (float(v) for v in params)))
    theta, kappa = p["theta"], p.get("kappa", 0.0)
    x_i, x_a = p.get("eta_I", 0.0), p.get("eta_A", 0.0)
    S, I, A, R = 1.0 - sc["I0"] - sc["A0"], sc["I0"], sc["A0"], 0.0
    S, I, A, R = float(S), float(I), float(A), float(R)
    c_i, c_a = I, A
    uses_delta = kind not in ("const", "const_plus")
    pops = pops_for(sc, p)
    src_t, src_to_i = sc["src_t"].tolist(), sc["src_to_i"].tolist()
    amounts = (theta * sc["src_share"]).tolist()
    dt = step_dt(sc, kind, params) if dt is None else dt
    upd, end, dt = sc["upd"].tolist(), float(sc["end_h"] if end_h is None else end_h), float(dt)
    if not uses_delta:
        b_i, b_a, g, e = p["beta"], p["alpha"], p["gamma"], p["eps"]
    j, k_next, d1, t = 0, 0, np.nan, 0.0
    rows, blocks = [], []
    empty = np.empty((0, 7))
    while True:
        # events at time t: source injections (D46), then the delta update for this bin (D45)
        while j < len(src_t) and src_t[j] <= t + 1e-9:
            move = min(amounts[j], S)
            S -= move
            if src_to_i[j]:
                I += move
                c_i += move
            else:
                A += move
                c_a += move
            j += 1
        if k_next < len(upd) and upd[k_next] <= t + 1e-9:
            if uses_delta:
                d1 = delta_at(sc, k_next, c_i, c_a, I, A, kappa, pops[k_next] if pops is not None else 0.0,
                              kind, freeze_k)
                if kind in EQ26_KINDS:
                    b_i, b_a, g, e = d1, 1 - d1, 1 - d1, d1
                else:
                    b_i, b_a, g, e = p["beta"] * d1, p["alpha"] * (1 - d1), p["gamma"], p["eps"]
            k_next += 1
        rows.append((t, S, I, A, R, c_i, c_a, d1))
        if record:
            blocks.append(np.array([rows[-1]]))
        if t >= end - 1e-9:
            break
        # Euler steps of at most dt up to exactly the next event (for every model, so that all models use
        # the same step grid and the nesting tests are exact)
        nxt = end
        if j < len(src_t):
            nxt = min(nxt, src_t[j])
        if k_next < len(upd):
            nxt = min(nxt, upd[k_next])
        buf = np.empty((int((nxt - t) / dt) + 3, 7)) if record else empty
        S, I, A, R, c_i, c_a, n = _advance(S, I, A, R, c_i, c_a, t, nxt, dt, b_i, b_a, g, e, x_i, x_a, buf, record)
        if record and n:
            blocks.append(np.column_stack([buf[:n], np.full(n, d1)]))
        t = nxt
    if record:
        out = np.vstack(blocks)
        return pd.DataFrame(out[:, 1:7], columns=["S", "I", "A", "R", "C_I", "C_A"]).assign(
            d1=out[:, 7], d2=1 - out[:, 7], t=out[:, 0])
    out = np.array(rows)
    return out[:, 0], out[:, 5], out[:, 6]


def simulate_python(sc, kind, params, dt=None, record=False, end_h=None, freeze_k=None):
    """Reference pure-Python Euler loop (kept for the numba equivalence check, D62)."""
    p = dict(zip(NAMES[kind], (float(v) for v in params)))
    theta, kappa = p["theta"], p.get("kappa", 0.0)
    x_i, x_a = p.get("eta_I", 0.0), p.get("eta_A", 0.0)
    S, I, A, R = 1.0 - sc["I0"] - sc["A0"], sc["I0"], sc["A0"], 0.0
    S, I, A, R = float(S), float(I), float(A), float(R)
    c_i, c_a = I, A
    uses_delta = kind not in ("const", "const_plus")
    pops = pops_for(sc, p)
    # plain Python floats in the step loop (numpy scalars would make every update ~10x slower)
    src_t, src_to_i = sc["src_t"].tolist(), sc["src_to_i"].tolist()
    amounts = (theta * sc["src_share"]).tolist()
    dt = step_dt(sc, kind, params) if dt is None else dt
    upd, end, dt = sc["upd"].tolist(), float(sc["end_h"] if end_h is None else end_h), float(dt)
    if not uses_delta:
        b_i, b_a, g, e = p["beta"], p["alpha"], p["gamma"], p["eps"]
    j, k_next, d1, t = 0, 0, np.nan, 0.0
    out = []
    while True:
        # events at time t: source injections (D46), then the delta update for this bin (D45)
        while j < len(src_t) and src_t[j] <= t + 1e-9:
            move = min(amounts[j], S)
            S -= move
            if src_to_i[j]:
                I += move
                c_i += move
            else:
                A += move
                c_a += move
            j += 1
        if k_next < len(upd) and upd[k_next] <= t + 1e-9:
            if uses_delta:
                d1 = delta_at(sc, k_next, c_i, c_a, I, A, kappa, pops[k_next] if pops is not None else 0.0,
                              kind, freeze_k)
                if kind in EQ26_KINDS:
                    b_i, b_a, g, e = d1, 1 - d1, 1 - d1, d1
                else:
                    b_i, b_a, g, e = p["beta"] * d1, p["alpha"] * (1 - d1), p["gamma"], p["eps"]
            k_next += 1
        if record:
            out.append((t, S, I, A, R, c_i, c_a, d1, 1 - d1))
        else:
            out.append((t, c_i, c_a))
        if t >= end - 1e-9:
            break
        # Euler step of at most dt that stops exactly at the next event (for every model, so that
        # all models use the same step grid and the nesting tests are exact)
        nxt = end
        if j < len(src_t):
            nxt = min(nxt, src_t[j])
        if k_next < len(upd):
            nxt = min(nxt, upd[k_next])
        h = min(dt, nxt - t)
        S, I, A, R, gi, ga = euler_step(S, I, A, R, b_i, b_a, g, e, h, x_i, x_a)
        c_i, c_a = c_i + gi, c_a + ga
        t = nxt if nxt - (t + h) < 1e-9 else t + h
    out = np.array(out)
    if record:
        return pd.DataFrame(out[:, 1:], columns=["S", "I", "A", "R", "C_I", "C_A", "d1", "d2"]).assign(t=out[:, 0])
    return out[:, 0], out[:, 1], out[:, 2]


def model_on_grid(sc, t_arr, c_i, c_a, nb=None):
    """Model C_I, C_A at the first nb data bins, normalised by the data's final values."""
    g = sc["grid"][:nb]
    return np.interp(g, t_arr, c_i) / sc["fin_I"], np.interp(g, t_arr, c_a) / sc["fin_A"]


def data_targets(sc):
    return sc["C_I"] / sc["fin_I"], sc["C_A"] / sc["fin_A"]


def residuals(sc, kind, x, target=None, lo=0, hi=None, freeze_k=None):
    """Model minus target at data bins [lo, hi) for C_I and C_A (normalised curves)."""
    ti, ta = data_targets(sc) if target is None else target
    hi = len(sc["grid"]) if hi is None else hi
    t_arr, c_i, c_a = simulate(sc, kind, x, end_h=sc["grid"][hi - 1], freeze_k=freeze_k)
    mi, ma = model_on_grid(sc, t_arr, c_i, c_a, hi)
    return mi[lo:] - ti[lo:hi], ma[lo:] - ta[lo:hi]


def metrics(ri, ra, k):
    """RMSE per curve, their mean, SSE and the Gaussian AIC / BIC over n = all C_I and C_A points."""
    n = len(ri) + len(ra)
    sse = float((ri ** 2).sum() + (ra ** 2).sum())
    rmse_i, rmse_a = float(np.sqrt(np.mean(ri ** 2))), float(np.sqrt(np.mean(ra ** 2)))
    return {"rmse_I": rmse_i, "rmse_A": rmse_a, "rmse_mean": (rmse_i + rmse_a) / 2,
            "rmse_pooled": float(np.sqrt(sse / n)), "n": n, "k": k, "SSE": sse,
            "AIC": n * np.log(sse / n) + 2 * k, "BIC": n * np.log(sse / n) + k * np.log(n)}


# ---------------------------------------------------------------- fitting

def _objective(u, key, kind, free, fixed, target, hi, freeze_k):
    sc = get_context(key)
    bounds = fit_bounds(free)
    u = np.clip(u, [b[0] for b in bounds], [b[1] for b in bounds])
    ri, ra = residuals(sc, kind, full_params(kind, free, from_fit(free, u), fixed), target, 0, hi, freeze_k)
    return float((ri ** 2).sum() + (ra ** 2).sum()) / (len(ri) + len(ra))


def run_start(job):
    """One bounded Nelder-Mead run in fitting coordinates (module-level so it runs in a worker process).
    job: key, kind, u0 (free parameters, fitting coordinates), tag, and optionally fixed {name: value},
    target (normalised C_I, C_A), hi (bins used), freeze_k, maxfev."""
    kind, fixed = job["kind"], job.get("fixed")
    free = free_names(kind, fixed)
    res = minimize(_objective, job["u0"], args=(job["key"], kind, free, fixed, job.get("target"), job.get("hi"),
                                                job.get("freeze_k")),
                   method="Nelder-Mead", bounds=fit_bounds(free),
                   options={"maxfev": job.get("maxfev", MAX_FEV[kind]), "xatol": XATOL, "fatol": FATOL})
    lo, hi = zip(*fit_bounds(free))
    return {"tag": job["tag"], "x": full_params(kind, free, from_fit(free, np.clip(res.x, lo, hi)), fixed),
            "mse": float(res.fun), "converged": bool(res.success), "nfev": int(res.nfev)}


def random_starts(names, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = (np.array(b) for b in zip(*fit_bounds(names)))
    return [lo + rng.random(len(lo)) * (hi - lo) for _ in range(n)]


def start_jobs(key, kind, tag, n=N_STARTS, seed=dp.SEED, fixed=None, **extra):
    return [{"key": key, "kind": kind, "u0": u0, "tag": tag, "fixed": fixed, **extra}
            for u0 in random_starts(free_names(kind, fixed), n, seed)]


def _as_tuple(x):
    return tuple(_as_tuple(v) for v in x) if isinstance(x, list) else x


def load_checkpoint(ckpt):
    """id -> result of every job in data/processed/checkpoints/<ckpt>.jsonl (empty if none)."""
    done = {}
    path = os.path.join(CKPT_DIR, f"{ckpt}.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r["tag"], r["x"] = _as_tuple(r["tag"]), np.array(r["x"])
                done[r["id"]] = r
    return done


def fingerprint(x):
    """Short hash of a parameter vector: jobs built from fitted values carry it in their checkpoint id, so a
    checkpointed result is reused only if the fit it was built from is unchanged."""
    return hashlib.md5(np.round(np.asarray(x, dtype=float), 12).tobytes()).hexdigest()[:10]


def run_jobs(pool, jobs, label="runs", ckpt=None):
    """Run all jobs in the pool; group the results by tag. With ckpt, every finished job is appended to
    data/processed/checkpoints/<ckpt>.jsonl as it completes, and jobs already there are skipped (restart).
    Logs every job: done / total, elapsed and ETA."""
    count = {}
    for j in jobs:                                              # deterministic id: tag + index within the tag
        n = count.get(repr(j["tag"]), 0)
        count[repr(j["tag"])] = n + 1
        j["id"] = f"{j['tag']!r}#{n}" + (f"@{j['basis']}" if j.get("basis") else "")
    path = os.path.join(CKPT_DIR, f"{ckpt}.jsonl") if ckpt else None
    done = load_checkpoint(ckpt) if ckpt else {}
    out = {}
    for j in jobs:
        if j["id"] in done:
            out.setdefault(done[j["id"]]["tag"], []).append(done[j["id"]])
    todo = [j for j in jobs if j["id"] not in done]
    print(f"[{label}] {len(jobs)} jobs: {len(jobs) - len(todo)} from checkpoint, {len(todo)} to run", flush=True)
    if path:
        os.makedirs(CKPT_DIR, exist_ok=True)
    t0 = time.time()
    futures = {pool.submit(run_start, j): j for j in todo}
    for n, f in enumerate(as_completed(futures), 1):
        r = f.result()
        r["id"] = futures[f]["id"]
        if path:
            with open(path, "a") as fh:
                fh.write(json.dumps({**r, "x": [float(v) for v in r["x"]]}) + "\n")
        out.setdefault(r["tag"], []).append(r)
        el = time.time() - t0
        print(f"[{label}] {n}/{len(todo)} done  {r['tag']}  mse {r['mse']:.3e}  conv {r['converged']}  "
              f"elapsed {el:.0f} s  ETA {el / n * (len(todo) - n):.0f} s", flush=True)
    return out


def best_of(results):
    best = min(results, key=lambda r: r["mse"])
    return best, sum(r["converged"] for r in results)


def params_of(row, kind):
    return np.array([row[n] for n in NAMES[kind]], dtype=float)


def fit_row(key, kind, results, fixed=None, nested=None):
    """Best start's parameters and metrics; k = free parameters. nested = results of the starts embedded from
    submodel optima (D64): they compete for the best fit; the converged count is over the random starts."""
    sc = get_context(key)
    best, n_conv = best_of(results)
    best_nested = best_of(nested)[0] if nested else None
    best_from = "random"
    if best_nested is not None and best_nested["mse"] < best["mse"]:
        best, best_from = best_nested, f"seeded from {best_nested['tag'][3]}"
    x = best["x"]
    free = free_names(kind, fixed)
    m = metrics(*residuals(sc, kind, x), len(free))
    return {"set": key if isinstance(key, str) else key[0], "model": kind,
            **{n: x[NAMES[kind].index(n)] if n in NAMES[kind] else np.nan for n in ALL_PARAMS},
            **m, "bound_flags": bound_flags(free, [x[NAMES[kind].index(n)] for n in free]),
            "best_converged": best["converged"], "starts_converged": n_conv, "starts": len(results),
            "best_from": best_from, "random_best_SSE": best_of(results)[0]["mse"] * m["n"],
            "nfev_total": sum(r["nfev"] for r in results + (nested or []))}


# submodel -> model embeddings (D64): the submodel's optimum is an exact point of the bigger model
NESTED_FROM = {"const_plus": ["const"], "sairh": ["const"], "sairh_uniform": ["const"], "meanfield": ["const"],
               "sairh_plus": ["sairh", "const_plus"], "sairh_plus_uniform": ["const_plus", "sairh_uniform"]}
FLATTEN_FROM = {"sairh_uniform": ["sairh"], "sairh_plus_uniform": ["sairh_plus"]}   # approximate seeds (D66)
NESTED_PAIRS = [(sup, sub) for sup, subs in NESTED_FROM.items() for sub in subs]


def embed(sub, sup, x_sub):
    """Parameters of model sup that reproduce submodel sub exactly: eta = 0 for the Bass terms; for a
    constant-rate submodel inside SAIR-H, kappa = 0 (d1 = d2 = 1/2) and doubled beta, alpha (clipped to the bounds)."""
    p = dict(zip(NAMES[sub], x_sub))
    p.setdefault("eta_I", 0.0)
    p.setdefault("eta_A", 0.0)
    if sub in ("const", "const_plus") and sup not in ("const", "const_plus"):
        p["beta"], p["alpha"], p["kappa"] = 2 * p["beta"], 2 * p["alpha"], 0.0
        p.setdefault("half_life_h", ft.POP_HALF_LIFE_H)
    return np.array([np.clip(p[n], *RANGE[n]) for n in NAMES[sup]])


def project(sup, sub, x_sup):
    """Starting point for submodel sub from model sup's optimum (the inverse map of embed; exact only when
    sup's kappa = 0 and eta = 0): SAIR-H's beta, alpha halved; kappa, w and eta dropped."""
    p = dict(zip(NAMES[sup], x_sup))
    if sup not in ("const", "const_plus") and sub in ("const", "const_plus"):
        p["beta"], p["alpha"] = p["beta"] / 2, p["alpha"] / 2
    p.setdefault("eta_I", 0.0)
    p.setdefault("eta_A", 0.0)
    return np.array([np.clip(p[n], *RANGE[n]) for n in NAMES[sub]])


def flatten(src, tgt, x_src):
    """Seed for a uniform-Mult_eff model from a ranked one: Mult_eff_i = kappa * rank_i flattened to its mean
    kappa / 2 (ranks are uniform in [0, 1]); the half-life drops out. Approximate, so a start only."""
    p = dict(zip(NAMES[src], x_src))
    p["kappa"] = p["kappa"] / 2
    return np.array([np.clip(p[n], *RANGE[n]) for n in NAMES[tgt]])


def main_model(table):
    """Main model for Phases 6-8: the SAIR-H-family model ((d), (d+), (d)-uniform, (d+)-uniform) with the best
    TOTAL AIC summed over the three sets, since Phases 7-8 use one model for every event (D69, supersedes D66)."""
    t = table[table["model"].isin(MAIN_CANDIDATES)]
    return t.groupby("model")["AIC"].sum().idxmin()


def main_rule_table(table):
    """Per-set and total AIC / BIC of the main candidates (D69)."""
    t = table[table["model"].isin(MAIN_CANDIDATES)]
    out = t.pivot(index="model", columns="set", values="AIC")[SETS].add_prefix("AIC_")
    out["AIC_total"] = out.sum(axis=1)
    bic = t.pivot(index="model", columns="set", values="BIC")[SETS].add_prefix("BIC_")
    out = out.join(bic)
    out["BIC_total"] = bic.sum(axis=1)
    return out.sort_values("AIC_total")


def load_main():
    """(main model name, fitted parameter table) from the saved Phase 5 results."""
    table = pd.read_csv(os.path.join(dp.RESULTS, "phase5_params.csv"))
    return main_model(table), table


# ---------------------------------------------------------------- validation pieces

def forecast_cut(sc):
    """Last training bin: the later of the first bin where C_I reaches 50% of its final value and the 10th bin."""
    return max(int(np.argmax(sc["C_I"] >= 0.5 * sc["fin_I"])), FC_MIN_BIN)


def forecast_ok(sc):
    return len(sc["grid"]) - forecast_cut(sc) - 1 >= FC_MIN_TEST


def block_resample(r, rng):
    """Moving-block bootstrap of one residual series (blocks of BLOCK_LEN consecutive bins)."""
    n = len(r)
    idx = []
    while len(idx) < n:
        s = int(rng.integers(0, n - BLOCK_LEN + 1))
        idx.extend(range(s, s + BLOCK_LEN))
    return r[np.array(idx[:n])]


def bootstrap_jobs(sc, kind, x_best, seed):
    """Moving-block residual bootstrap: fitted curve + block-resampled residuals (data - fit) per curve.
    Every resample is refit warm from the best fit; the first N_COLD also from N_COLD_STARTS cold random starts."""
    ri, ra = residuals(sc, kind, x_best)
    ti, ta = data_targets(sc)
    fi, fa = ti + ri, ta + ra                                  # fitted (model) curves
    rng = np.random.default_rng(seed)
    jobs = []
    for b in range(N_BOOT):
        target = (fi - block_resample(ri, rng), fa - block_resample(ra, rng))
        jobs.append({"key": sc["key"], "kind": kind, "u0": to_fit(NAMES[kind], x_best),
                     "tag": ("boot", sc["key"], kind, b), "target": target, "basis": fingerprint(x_best)})
        if b < N_COLD:
            jobs += start_jobs(sc["key"], kind, ("cold", sc["key"], kind, b), n=N_COLD_STARTS, seed=dp.SEED + 100 + b,
                               target=target, basis=fingerprint(x_best))
    return jobs


def recovery_target(sc, kind, x_true, seed):
    """Model curves at the data bins + Gaussian noise (sigma on the normalised curves)."""
    t_arr, c_i, c_a = simulate(sc, kind, x_true)
    mi, ma = model_on_grid(sc, t_arr, c_i, c_a)
    rng = np.random.default_rng(seed)
    return mi + rng.normal(0, RECOVERY_SIGMA, len(mi)), ma + rng.normal(0, RECOVERY_SIGMA, len(ma))


# ---------------------------------------------------------------- plots

def set_title(key):
    return dp.EVENT_NAMES.get(key, "Ottawa story")


def plot_states(runs, main):
    fig, axes = plt.subplots(3, len(SETS), figsize=(5 * len(SETS), 10), squeeze=False)
    for c, key in enumerate(SETS):
        r = runs[key][main]
        ax = axes[0, c]
        for s, label in (("S", "S susceptible"), ("R", "R removed")):
            ax.plot(r["t"], r[s], color=dp.COLORS[s], lw=2, label=label)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8)
        dp.style(ax, f"{set_title(key)}: S and R", "Fraction of N", xlabel="")
        for row, s, label in ((1, "I", "I - rumour spreaders"), (2, "A", "A - correction spreaders")):
            ax = axes[row, c]
            ax.plot(r["t"], r[s], color=dp.COLORS[s], lw=2)
            ax.set_ylim(0, max(r[s].max() * 1.15, 1e-6))
            dp.style(ax, label, "Fraction of N", xlabel="Hours since first post" if row == 2 else "")
    fig.suptitle(f"Plot 1 - fitted main model {MODEL_LABEL[main]}: S/A/I/R state fractions", fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot1_states.png")


def plot_cumulative(runs, table):
    shown = ["const", "const_plus", "eq26", "sairh", "sairh_plus", "sairh_uniform", "sairh_plus_uniform"]
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 3.8 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc = get_context(key)
        for c, (col, s) in enumerate((("C_I", "I"), ("C_A", "A"))):
            ax = axes[r_i, c]
            final = sc[f"fin_{s}"]
            ax.plot(sc["grid"], sc[col] / final, color="#0b0b0b", lw=0, marker="o", ms=3, label="real data")
            for kind in shown:
                r = runs[key][kind]
                row = table[(table["set"] == key) & (table["model"] == kind)].iloc[0]
                ax.plot(r["t"], r[col] / final, color=MODEL_COLORS[kind], lw=1.8,
                        ls="--" if kind in ("eq26", "const_plus", "sairh_plus") else "-",
                        label=f"{MODEL_LABEL[kind]}  RMSE {row[f'rmse_{s}']:.3f}")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
            ax.set_ylim(0, max(1.15, ax.get_ylim()[1]))
            dp.style(ax, f"{set_title(key)}: {col} / final", "Share of final value",
                     xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle("Plot 2 - predicted vs real cumulative entries (fitting target)", fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot2_cumulative_fit.png")


def plot_prevalence(runs, main):
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 3.6 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc, r = get_context(key), runs[key][main]
        for c, s in enumerate("IA"):
            ax = axes[r_i, c]
            ax.plot(sc["grid"], sc[f"{s}_display"], color="#0b0b0b", lw=1.5, ls="--",
                    label="real (display: exp. lifespan 12 h)")
            ax.plot(r["t"], r[s], color=dp.COLORS[s], lw=2, label=MODEL_LABEL[main])
            ax.legend(frameon=False, fontsize=8)
            dp.style(ax, f"{set_title(key)}: {s} prevalence", "Fraction of N",
                     xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle(f"Plot 2b - {MODEL_LABEL[main]} I and A prevalence vs the display curves (visual only, not fitted)",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot2b_prevalence.png")


def plot_forecast(fc_runs, fc_table):
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 3.8 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc = get_context(key)
        cut_t = sc["grid"][forecast_cut(sc)]
        for c, (col, s) in enumerate((("C_I", "I"), ("C_A", "A"))):
            ax = axes[r_i, c]
            final = sc[f"fin_{s}"]
            ax.axvspan(sc["grid"][0], cut_t, color="#e6e5e1", alpha=0.6, lw=0, label=f"training window (to {cut_t:g} h)")
            ax.plot(sc["grid"], sc[col] / final, color="#0b0b0b", lw=0, marker="o", ms=3, label="real data")
            if key in fc_runs:
                for kind in fc_runs[key]:
                    r = fc_runs[key][kind]
                    row = fc_table[(fc_table["set"] == key) & (fc_table["model"] == kind)].iloc[0]
                    ax.plot(r["t"], r[col] / final, color=MODEL_COLORS[kind], lw=1.8, ls="--" if "plus" in kind else "-",
                            label=f"{MODEL_LABEL[kind]}  held-out RMSE {row[f'fc_rmse_{s}']:.3f}")
            else:
                ax.text(0.5, 0.5, f"forecast N/A (< {FC_MIN_TEST} bins after the cut)", transform=ax.transAxes, ha="center")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
            ax.set_ylim(0, max(1.15, ax.get_ylim()[1]))
            dp.style(ax, f"{set_title(key)}: {col} / final", "Share of final value",
                     xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle("Plot 2c - forecast: fit up to max(50% point of C_I, 10th bin) (shaded), predict the rest",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot2c_forecast.png")


# ---------------------------------------------------------------- stages

SECONDARY = ["sairh_uniform"]                 # main model under D66, kept as a secondary row (D69)


def validated(main):
    """Models that get the forecast and the bootstrap: (a), (a+), the main model (D63) and the secondary rows."""
    return ["const", "const_plus", main] + [k for k in SECONDARY if k != main]


def recovered_models(main):
    return [main] + [k for k in SECONDARY if k != main]


def ci_rows(key, kind, est, results):
    """95% percentile CIs of every parameter and of the mean RMSE (vs the real data) over bootstrap refits."""
    sc = get_context(key)
    xs = np.array([r["x"] for r in results])
    rm = np.array([metrics(*residuals(sc, kind, r["x"]), 0)["rmse_mean"] for r in results])
    rows = []
    for j, n in enumerate(NAMES[kind] + ["rmse_mean_vs_data"]):
        v = xs[:, j] if j < len(NAMES[kind]) else rm
        rows.append({"set": key, "model": kind, "parameter": n, "estimate": est[n] if n in NAMES[kind] else est["rmse_mean"],
                     "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)),
                     "boot_median": float(np.median(v)), "resamples": len(results),
                     "resamples_converged": sum(r["converged"] for r in results)})
    return rows


def comparison_table(table, fc_table=None):
    comp = table[["set", "model", "k", "n", "SSE", "rmse_I", "rmse_A", "rmse_mean", "rmse_pooled", "AIC", "BIC",
                  "dAIC_vs_best", "bound_flags", "starts_converged", "starts", "best_converged", "best_from",
                  "random_best_SSE"]].copy()
    if fc_table is not None:
        comp = comp.merge(fc_table[[c for c in ["set", "model", "cut_h", "train_points", "test_points", "fc_rmse_I",
                                                "fc_rmse_A", "fc_rmse_pooled", "fc_starts_converged", "fc_bound_flags",
                                                "fc_note"] if c in fc_table]], on=["set", "model"], how="left")
    comp.insert(2, "label", comp["model"].map(MODEL_LABEL))
    comp["main_model"] = comp["model"] == main_model(table)
    return comp


def stage1(pool):
    """Stage 1: every model on every set, N_STARTS random starts; parameters, comparison table, plots 1, 2, 2b."""
    no_kappa = [k for k in MODELS if "kappa" not in NAMES[k]]
    with_kappa = [k for k in MODELS if "kappa" in NAMES[k]]
    jobs = [j for key in SETS for kind in no_kappa for j in start_jobs(key, kind, ("fit", key, kind))]
    res = run_jobs(pool, jobs, "stage 1 fits, models without kappa", "stage1_fits")
    # models with kappa: kappa widened to [0, 50] (D65). Each restarts from its N_STARTS endpoints checkpointed
    # under kappa <= 10 (random starts for a model that has none, i.e. (d+)-uniform)
    old = load_checkpoint("stage1_fits")
    jobs = []
    for key in SETS:
        for kind in with_kappa:
            prev = sorted((r for r in old.values() if r["tag"] == ("fit", key, kind)), key=lambda r: r["id"])
            starts = [to_fit(NAMES[kind], r["x"]) for r in prev] or random_starts(NAMES[kind], N_STARTS, dp.SEED)
            jobs += [{"key": key, "kind": kind, "u0": u0, "tag": ("fit", key, kind), "fixed": None} for u0 in starts]
    res.update(run_jobs(pool, jobs, "stage 1 fits, models with kappa", f"stage1_fits_{KAPPA_VERSION}"))
    # plus N_FRESH new random starts over the full kappa range for the main candidates (so nothing above the old
    # bound of 10 is missed); they count as random starts of the model
    jobs = [j for key in SETS for kind in MAIN_CANDIDATES
            for j in start_jobs(key, kind, ("fresh", key, kind), n=N_FRESH, seed=dp.SEED + 7)]
    fresh = run_jobs(pool, jobs, "stage 1 fresh starts over kappa in [0, 50]", f"stage1_fresh_{KAPPA_VERSION}")
    for (_, key, kind), rr in fresh.items():
        res[("fit", key, kind)] = res[("fit", key, kind)] + rr
    # cross-model starts (D64, D66): one extra start per related pair, seeded from the other model's current best.
    # Up: a submodel's optimum embedded exactly in the bigger model (round a: from (a), (a+); round b: from the
    # kappa-models, plus the flattened-ranking seeds). Down: a bigger model's optimum projected into the submodel.
    # Repeated until no best fit improves, at most MAX_CROSS_ITER times.
    best = {(key, kind): best_of(res[("fit", key, kind)])[0] for key in SETS for kind in MODELS}

    def cross_round(pairs, name, it):
        jobs = [{"key": key, "kind": tgt, "u0": to_fit(NAMES[tgt], seed(src, tgt, best[(key, src)]["x"])),
                 "tag": ("nest", key, tgt, src) + ((it,) if it > 1 else ()), "fixed": None}
                for key in SETS for tgt, src, seed in pairs]
        out = run_jobs(pool, jobs, f"stage 1 {name}, iteration {it}", f"stage1_{KAPPA_VERSION}_{name.replace(' ', '_')}_{it}")
        res.update(out)
        improved = False
        for rr in out.values():
            r0 = rr[0]
            if r0["mse"] < best[(r0["tag"][1], r0["tag"][2])]["mse"] * (1 - 1e-9):
                best[(r0["tag"][1], r0["tag"][2])] = r0
                improved = True
        return improved

    up = [(sup, sub, embed) for sup, sub in NESTED_PAIRS] + \
         [(tgt, src, flatten) for tgt, srcs in FLATTEN_FROM.items() for src in srcs]
    for it in range(1, MAX_CROSS_ITER + 1):
        changed = cross_round([q for q in up if q[1] in ("const", "const_plus")], "up starts a", it)
        changed |= cross_round([q for q in up if q[1] not in ("const", "const_plus")], "up starts b", it)
        changed |= cross_round([(sub, sup, project) for sup, sub in NESTED_PAIRS], "down starts", it)
        if not changed:
            break
    nested = {(key, kind): [x for tag, rr in res.items() if tag[0] == "nest" and tag[1] == key and tag[2] == kind
                            for x in rr] for key in SETS for kind in MODELS}
    table = pd.DataFrame([fit_row(key, kind, res[("fit", key, kind)], nested=nested[(key, kind)])
                          for key in SETS for kind in MODELS])
    table["dAIC_vs_best"] = table["AIC"] - table.groupby("set")["AIC"].transform("min")
    table.to_csv(os.path.join(dp.RESULTS, "phase5_params.csv"), index=False)
    main = main_model(table)
    comp = comparison_table(table)
    comp.to_csv(os.path.join(dp.RESULTS, "phase5_model_comparison.csv"), index=False)
    runs = {}
    for key in SETS:
        sc = get_context(key)
        runs[key] = {kind: simulate(sc, kind, params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind),
                                    record=True) for kind in MODELS}
        runs[key][main].to_csv(os.path.join(dp.PROCESSED, f"model_curves_{key}.csv"), index=False)
    plot_states(runs, main)
    plot_cumulative(runs, table)
    plot_prevalence(runs, main)
    rule = main_rule_table(table)
    rule.to_csv(os.path.join(dp.RESULTS, "phase5_main_model_rule.csv"))
    print(f"\nMain model for Phases 6-8 (best total AIC over the three sets among {', '.join(MAIN_CANDIDATES)}): {main}")
    print(rule.round(1).to_string())
    print("\n=== Fitted parameters ===")
    print(table[["set", "model"] + ALL_PARAMS + ["bound_flags", "starts_converged", "best_converged"]].round(4).to_string(index=False))
    print("\n=== Model comparison ===")
    print(comp.drop(columns=["label"]).round(4).to_string(index=False))


def stage2(pool):
    """Stage 2: forecast ((a), (a+), main), parameter recovery (main, CH), moving-block bootstrap ((a), (a+), main)."""
    main, table = load_main()
    kinds = validated(main)
    sc_ch = get_context("charliehebdo")
    x_true = {k: params_of(table[(table["set"] == "charliehebdo") & (table["model"] == k)].iloc[0], k)
              for k in recovered_models(main)}
    jobs = []
    for key in SETS:
        sc = get_context(key)
        if forecast_ok(sc):
            k_cut = forecast_cut(sc)
            for kind in kinds:
                jobs += start_jobs(key, kind, ("fc", key, kind), hi=k_cut + 1, freeze_k=k_cut,
                                   basis=KAPPA_VERSION if "kappa" in NAMES[kind] else None)
    for k in recovered_models(main):
        for seed in RECOVERY_SEEDS:
            jobs += start_jobs("charliehebdo", k, ("rec", k, seed), n=N_RECOVERY_STARTS, seed=dp.SEED + seed,
                               target=recovery_target(sc_ch, k, x_true[k], seed), basis=fingerprint(x_true[k]))
    for key in SETS:
        for kind in kinds:
            jobs += bootstrap_jobs(get_context(key), kind, params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind),
                                   dp.SEED)
    res = run_jobs(pool, jobs, "stage 2 validation", "stage2_validation")

    fc_rows, fc_runs = [], {}
    for key in SETS:
        sc = get_context(key)
        k_cut = forecast_cut(sc)
        for kind in kinds:
            row = {"set": key, "model": kind, "cut_h": sc["grid"][k_cut], "train_points": 2 * (k_cut + 1),
                   "test_points": 2 * (len(sc["grid"]) - k_cut - 1), "k": len(NAMES[kind])}
            if forecast_ok(sc):
                best, n_conv = best_of(res[("fc", key, kind)])
                ri, ra = residuals(sc, kind, best["x"], lo=k_cut + 1, freeze_k=k_cut)
                row.update({"fc_rmse_I": float(np.sqrt(np.mean(ri ** 2))), "fc_rmse_A": float(np.sqrt(np.mean(ra ** 2))),
                            "fc_rmse_pooled": float(np.sqrt(np.mean(np.r_[ri, ra] ** 2))),
                            "fc_starts_converged": n_conv, "fc_bound_flags": bound_flags(NAMES[kind], best["x"]),
                            **{f"fc_{n}": v for n, v in zip(NAMES[kind], best["x"])}})
                fc_runs.setdefault(key, {})[kind] = simulate(sc, kind, best["x"], record=True, freeze_k=k_cut)
            else:
                row["fc_note"] = f"N/A: fewer than {FC_MIN_TEST} bins after the cut"
            fc_rows.append(row)
    fc_table = pd.DataFrame(fc_rows)
    fc_table.to_csv(os.path.join(dp.RESULTS, "phase5_forecast.csv"), index=False)

    rec_rows = []
    for k in recovered_models(main):
        for seed in RECOVERY_SEEDS:
            best, n_conv = best_of(res[("rec", k, seed)])
            for n, tv, rv in zip(NAMES[k], x_true[k], best["x"]):
                rec_rows.append({"model": k, "role": "main" if k == main else "secondary", "noise_seed": seed,
                                 "parameter": n, "true": tv, "recovered": rv, "abs_error": rv - tv,
                                 "pct_error": 100 * (rv - tv) / tv if tv != 0 else np.nan,
                                 "starts_converged": n_conv, "starts": N_RECOVERY_STARTS, "fit_mse": best["mse"]})
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(os.path.join(dp.RESULTS, "phase5_recovery.csv"), index=False)

    boot_rows, cold_rows = [], []
    for key in SETS:
        for kind in kinds:
            est = table[(table["set"] == key) & (table["model"] == kind)].iloc[0]
            warm = [res[("boot", key, kind, b)][0] for b in range(N_BOOT)]
            boot_rows += ci_rows(key, kind, est, warm)
            for b in range(N_COLD):
                cold, n_conv = best_of(res[("cold", key, kind, b)])
                rel = np.abs(cold["x"] - warm[b]["x"]) / np.maximum(np.abs(warm[b]["x"]), 1e-12)
                cold_rows.append({"set": key, "model": kind, "resample": b, "warm_mse": warm[b]["mse"],
                                  "cold_mse": cold["mse"], "cold_minus_warm_mse": cold["mse"] - warm[b]["mse"],
                                  "cold_starts_converged": n_conv, "max_rel_param_diff": float(rel.max()),
                                  "param_most_different": NAMES[kind][int(rel.argmax())],
                                  **{f"warm_{n}": v for n, v in zip(NAMES[kind], warm[b]["x"])},
                                  **{f"cold_{n}": v for n, v in zip(NAMES[kind], cold["x"])}})
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(os.path.join(dp.RESULTS, "phase5_bootstrap_ci.csv"), index=False)
    cold = pd.DataFrame(cold_rows)
    cold.to_csv(os.path.join(dp.RESULTS, "phase5_bootstrap_coldcheck.csv"), index=False)
    comparison_table(table, fc_table).to_csv(os.path.join(dp.RESULTS, "phase5_model_comparison.csv"), index=False)
    plot_forecast(fc_runs, fc_table)

    print("\n=== Forecast ===")
    print(fc_table.round(4).to_string(index=False))
    print(f"\n=== Parameter recovery (CH, {', '.join(recovered_models(main))}, sigma = {RECOVERY_SIGMA}) ===")
    print(rec.round(4).to_string(index=False))
    print("\n=== Bootstrap 95% CIs (moving blocks of 5 bins, warm starts) ===")
    print(boot.round(4).to_string(index=False))
    print("\n=== Bootstrap cold-start check ===")
    print(cold[["set", "model", "resample", "warm_mse", "cold_mse", "cold_minus_warm_mse", "max_rel_param_diff",
                "param_most_different"]].round(6).to_string(index=False))


# ---------------------------------------------------------------- check

NESTING_TESTS = [("sairh", "const"), ("sairh_plus", "const_plus"), ("sairh_uniform", "const"),
                 ("sairh_plus_uniform", "const_plus")]


def nesting_diff(sc, kind_d, kind_a, x):
    """(d)/(d+) with kappa = 0 has d1 = d2 = 1/2, so it must equal (a)/(a+) with beta/2, alpha/2."""
    p = dict(zip(NAMES[kind_d], x))
    xd = {**p, "kappa": 0.0}
    xa = {**p, "beta": p["beta"] / 2, "alpha": p["alpha"] / 2}
    dt = step_dt(sc, kind_d, x)                                # the same step grid for both
    rd = simulate(sc, kind_d, [xd[n] for n in NAMES[kind_d]], dt=dt, record=True)
    ra = simulate(sc, kind_a, [xa[n] for n in NAMES[kind_a]], dt=dt, record=True)
    assert np.allclose(rd["d1"], 0.5, atol=1e-12) and len(rd) == len(ra)
    cols = ["S", "I", "A", "R", "C_I", "C_A"]
    return float(np.abs(rd[cols].to_numpy() - ra[cols].to_numpy()).max())


def check1():
    print("\n--- Phase 5 check, stage 1 ---")
    S, I, A, R, *_ = euler_step(0.90, 0.05, 0.05, 0.0, 0.6, 0.4, 1 - 0.6, 1 - 0.4, 1.0)
    assert np.allclose([S, I, A, R], [0.855, 0.057, 0.038, 0.050], atol=5e-4), (S, I, A, R)
    print(f"unit test (docs worked step, Eq. 26): S={S:.3f} I={I:.3f} A={A:.3f} R={R:.3f}  OK")
    main, table = load_main()
    comp = pd.read_csv(os.path.join(dp.RESULTS, "phase5_model_comparison.csv"))
    print(f"main model for Phases 6-8: {main} ({MODEL_LABEL[main]})")
    for key in SETS:
        sc = get_context(key)
        nest = {}
        for kd, ka in NESTING_TESTS:
            x = params_of(table[(table["set"] == key) & (table["model"] == kd)].iloc[0], kd)
            nest[kd] = nesting_diff(sc, kd, ka, x)
            assert nest[kd] <= 1e-9, f"{key}: nesting test {kd} vs {ka} failed ({nest[kd]:.2e})"
            ref = simulate_python(sc, kd, x, record=True)       # numba kernel == Python loop on the fitted parameters
            new = simulate(sc, kd, x, record=True)
            cols = ["t", "S", "I", "A", "R", "C_I", "C_A", "d1"]
            nb = float(np.abs(ref[cols].to_numpy() - new[cols].to_numpy()).max())
            assert len(ref) == len(new) and nb <= 1e-12, f"{key}/{kd}: numba differs from the Python loop ({nb:.1e})"
        worst_sum, worst_dt = 0.0, {}
        hours = np.arange(0, sc["end_h"] + 1e-9, 1.0)
        for kind in MODELS:
            x = params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind)
            r = simulate(sc, kind, x, record=True)
            states = r[["S", "I", "A", "R"]].to_numpy()
            worst_sum = max(worst_sum, float(np.abs(states.sum(axis=1) - 1).max()))
            assert worst_sum <= INV_TOL, f"{key}/{kind}: states do not sum to 1 ({worst_sum:.2e})"
            assert states.min() >= -INV_TOL and states.max() <= 1 + INV_TOL, f"{key}/{kind}: a state outside [0, 1]"
            if kind not in ("const", "const_plus"):
                assert r["d1"].between(0, 1).all() and r["d2"].between(0, 1).all(), f"{key}/{kind}: delta outside [0, 1]"
            b = simulate(sc, kind, x, dt=step_dt(sc, kind, x) / 2, record=True)   # dt-convergence, every model
            d = max(float(np.abs(np.interp(hours, r["t"], r[s]) - np.interp(hours, b["t"], b[s])).max())
                    for s in ["S", "I", "A", "R", "C_I", "C_A"])
            assert d < 1e-3, f"{key}/{kind}: dt-convergence failed, max |diff| = {d:.2e}"
            worst_dt[kind] = d
        c = comp[comp["set"] == key].set_index("model")
        for kd, ka in NESTING_TESTS:                                          # kappa ~ 0: (d) is (a) (D64)
            x = params_of(table[(table["set"] == key) & (table["model"] == kd)].iloc[0], kd)
            if dict(zip(NAMES[kd], x))["kappa"] < 1e-3:
                assert c.loc[ka, "SSE"] <= c.loc[kd, "SSE"] * (1 + 1e-3), \
                    f"{key}: {kd} has kappa ~ 0 but its submodel {ka} fits worse ({c.loc[ka, 'SSE']:.6f} > {c.loc[kd, 'SSE']:.6f})"
        for sup, sub in NESTED_PAIRS:                          # a model never fits worse than its submodel (D64)
            assert c.loc[sup, "SSE"] <= c.loc[sub, "SSE"] * (1 + 1e-6), \
                f"{key}: {sup} SSE {c.loc[sup, 'SSE']:.6f} > submodel {sub} SSE {c.loc[sub, 'SSE']:.6f}"
        assert (c["starts_converged"] >= MIN_CONVERGED).all(), \
            f"{key}: fewer than {MIN_CONVERGED}/{N_STARTS} converged starts: {c['starts_converged'].to_dict()}"
        if key == "charliehebdo":
            assert c.loc["const", "best_converged"], "Charlie Hebdo (a): the best start did not converge"
        for m in c.index[~c["best_converged"].astype(bool)]:
            print(f"{key:22s} WARNING  {m}: best start hit maxfev (flags: {c.loc[m, 'bound_flags']})")
        print(f"{key:22s} OK  nesting (kappa = 0) max diff {max(nest.values()):.1e} over {len(nest)} pairs  numba == Python  "
              f"nested SSE order OK  max|sum-1|={worst_sum:.1e}  dt vs dt/2: (d) {worst_dt['sairh']:.1e}, (d+) {worst_dt['sairh_plus']:.1e}, "
              f"(e) {worst_dt['meanfield']:.1e}, all <= {max(worst_dt.values()):.1e}")
        print(f"{'':22s}     converged starts: " + ", ".join(f"{m} {int(c.loc[m, 'starts_converged'])}/{int(c.loc[m, 'starts'])}" for m in MODELS))
        bounded = c[c["bound_flags"].fillna("") != ""]["bound_flags"]
        print(f"{'':22s}     parameters at a bound: " + ("; ".join(f"{m}: {f}" for m, f in bounded.items()) or "none"))
    print("Stage 1 checks passed.")


def check2():
    print("\n--- Phase 5 check, stage 2 ---")
    main, _ = load_main()
    fc = pd.read_csv(os.path.join(dp.RESULTS, "phase5_forecast.csv"))
    for key in SETS:
        sc = get_context(key)
        f = fc[fc["set"] == key].set_index("model")
        if forecast_ok(sc):
            print(f"{key:22s} forecast cut {f['cut_h'].iloc[0]:g} h; converged starts "
                  + ", ".join(f"{m} {int(f.loc[m, 'fc_starts_converged'])}/{N_STARTS}" for m in validated(main)))
        else:
            print(f"{key:22s} forecast N/A (fewer than {FC_MIN_TEST} bins after the cut)")
    rec = pd.read_csv(os.path.join(dp.RESULTS, "phase5_recovery.csv"))
    assert set(rec["noise_seed"]) == set(RECOVERY_SEEDS)
    summary = rec.groupby(["model", "parameter"], sort=False).agg(
        true=("true", "first"), median_recovered=("recovered", "median"),
        median_abs_pct_error=("pct_error", lambda v: np.median(np.abs(v))))
    print(f"parameter recovery (CH, {len(RECOVERY_SEEDS)} noise seeds): median |% error| per parameter")
    print(summary.round(4).to_string())
    boot = pd.read_csv(os.path.join(dp.RESULTS, "phase5_bootstrap_ci.csv"))
    assert (boot["resamples"] == N_BOOT).all()
    cold = pd.read_csv(os.path.join(dp.RESULTS, "phase5_bootstrap_coldcheck.csv"))
    print(f"bootstrap: {N_BOOT} moving-block resamples per set and model; converged "
          + ", ".join(f"{s[:5]}/{m} {int(n)}" for (s, m), n in boot.groupby(["set", "model"], sort=False)["resamples_converged"].first().items()))
    print(f"cold-start check ({N_COLD} resamples x {N_COLD_STARTS} starts): cold better than warm (MSE lower by > 1e-7) in "
          f"{int((cold['cold_minus_warm_mse'] < -1e-7).sum())}/{len(cold)} cases")
    print("Stage 2 checks passed.")


if __name__ == "__main__":
    stages = [int(sys.argv[sys.argv.index("--stage") + 1])] if "--stage" in sys.argv else [1, 2]
    if "--check-only" not in sys.argv:
        if "--fresh" in sys.argv:
            for name in ("stage1_fits", "stage2_validation"):
                if int(name[5]) in stages and os.path.exists(os.path.join(CKPT_DIR, f"{name}.jsonl")):
                    os.remove(os.path.join(CKPT_DIR, f"{name}.jsonl"))
        for key in SETS:
            get_context(key)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for s in stages:
                stage1(pool) if s == 1 else stage2(pool)
                if "--check" in sys.argv:
                    check1() if s == 1 else check2()
    else:
        for s in stages:
            check1() if s == 1 else check2()
