"""Phase 5 - SAIR simulation, fitting and validation (docs/SAIR_paper.md §5).

State = fractions of N: S, I, A, R. Every model is one explicit Euler scheme (dt = 0.02 h, events hit exactly)
    dS = -b_I*I*S - b_A*A*S
    dI =  b_I*I*S - g*I
    dA =  b_A*A*S - e*A
    dR =  g*I + e*A
with the rates of each model:
    (a) const          constant-rate SAIR (Eq. 22): b_I = beta, b_A = alpha, g = gamma, e = eps
    (b) uniform        Eq. 26, Mult_eff = kappa for everyone: b_I = d1, b_A = d2, g = 1 - d1, e = 1 - d2
    (c) eq26           Eq. 26 (corrected), data-driven d1 (D45)
    (d) sairh          SAIR-H, main model: b_I = beta*d1, b_A = alpha*d2, g = gamma, e = eps (D50)
        sairh_uniform  as (d) with Mult_eff = kappa for everyone
    (e) meanfield      as (d) with every user's P1 = the model's global I/(I + A)
Push rates d1(t), d2 = 1 - d1 (D45), recomputed every data bin (1 h for the events, 15 min for the story):
    rho_I = C_I_model / C_I_data (1 if data is 0; clipped to [0, 5]), rho_A the same
    P1_i  = rho_I*n_I,i / (rho_I*n_I,i + rho_A*n_A,i), n = friends who entered I / A in the data by t;
            no sided friends -> I/(I + A) of the model state (0.5 if both 0)
    Drif1_i = sigmoid((2*P1_i - 1) * Mult_eff_i(t)),  Mult_eff_i(t) = kappa * pct_rank(eta_i(t)), Pop at t
    d1 = mean Drif1 over users who have not entered in the data by t
Source injection (D46): at each supporting / denying source tweet's time, theta * w_s / sum(w) moves S -> I / A.
Fit (D51): least squares on C_I/C_I,final and C_A/C_A,final at the data bins, bounded Nelder-Mead in
log space for the rates and the half-life, 20 random starts (seed 42). AIC / BIC from the SSE.
Validation (D52): forecast from the 50% point of C_I, parameter recovery, residual bootstrap.

Usage:
    python src/sair.py           fit all models, validate, save tables and plots
    python src/sair.py --check   the same, then run the Phase 5 checks
    python src/sair.py --check-only   checks on the saved results (no refit)
    python src/sair.py --check --sets story_three_shootings   refit only the listed sets (other rows kept)
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import data_prep as dp
import drive as drv
import features as ft

SETS = ft.SETS
STEPS_PER_BIN = 50                           # Euler dt = bin / 50: 0.02 h for 1-h bins, 0.005 h for 15-min bins (D47, D53)
RHO_MAX = 5.0
N_STARTS = 20
N_WORKERS = 12
MIN_CONVERGED = 15                           # of N_STARTS, for every main fit
N_RECOVERY_STARTS, RECOVERY_SEEDS, RECOVERY_SIGMA = 10, [1, 2, 3, 4, 5], 0.01
N_BOOT = 50
XATOL, FATOL = 1e-4, 1e-9                    # Nelder-Mead tolerances (fitting space, MSE)
INV_TOL = 1e-9
BOUND_TOL = 0.01                             # flag a parameter within 1% of the range from a bound
POP_GRID = np.linspace(0, 1, 201)            # Mult_eff rank table resolution in Pop (D45)

MODELS = ["const", "uniform", "eq26", "sairh", "sairh_uniform", "meanfield"]
MAIN = "sairh"
NAMES = {"const": ["beta", "alpha", "gamma", "eps", "theta"],
         "uniform": ["kappa", "theta"],
         "eq26": ["half_life_h", "kappa", "theta"],
         "sairh": ["beta", "alpha", "gamma", "eps", "half_life_h", "kappa", "theta"],
         "sairh_uniform": ["beta", "alpha", "gamma", "eps", "kappa", "theta"],
         "meanfield": ["beta", "alpha", "gamma", "eps", "half_life_h", "kappa", "theta"]}
ALL_PARAMS = ["beta", "alpha", "gamma", "eps", "half_life_h", "kappa", "theta"]
RANGE = {"beta": (1e-3, 5.0), "alpha": (1e-3, 5.0), "gamma": (1e-3, 5.0), "eps": (1e-3, 5.0),
         "half_life_h": (0.5, 168.0), "kappa": (0.0, 10.0), "theta": (0.0, 1.0)}
LOG_PARAMS = {"beta", "alpha", "gamma", "eps", "half_life_h"}           # fitted on a log scale
MAX_FEV = {"const": 3000, "uniform": 1000, "eq26": 1500, "sairh": 6000, "sairh_uniform": 5000, "meanfield": 6000}
# how each model gets d1: P1 from the friend network or the global share; Mult_eff ranked or uniform
P1_MODE = {"uniform": "network", "eq26": "network", "sairh": "network", "sairh_uniform": "network", "meanfield": "global"}
ME_MODE = {"uniform": "uniform", "eq26": "rank", "sairh": "rank", "sairh_uniform": "uniform", "meanfield": "rank"}
MODEL_LABEL = {"const": "(a) constant-rate SAIR", "uniform": "(b) Eq. 26, uniform Mult_eff", "eq26": "(c) Eq. 26",
               "sairh": "(d) SAIR-H", "sairh_uniform": "(d) SAIR-H, uniform Mult_eff", "meanfield": "(e) SAIR-H mean-field"}
MODEL_COLORS = {"const": "#2a78d6", "uniform": "#9b9a94", "eq26": "#eda100", "sairh": "#1baf7a",
                "sairh_uniform": "#7a5af5", "meanfield": "#eb6834"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def euler_step(S, I, A, R, b_i, b_a, g, e, dt):
    """One explicit Euler step of the general SAIR scheme; returns the new state and the step's inflows."""
    new_i = b_i * I * S
    new_a = b_a * A * S
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


# ---------------------------------------------------------------- set context

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

    m = ctx["model"]
    coef, mu, sd = np.array(m["coef"]), np.array(m["scaler_mean"]), np.array(m["scaler_scale"])
    base = m["intercept"] + ((ctx["static"] - mu[1:]) / sd[1:]) @ coef[1:] - coef[0] * mu[0] / sd[0]
    slope = coef[0] * ctx["user_trait"] / sd[0]                 # eta_i(t) = base_i + slope_i * Pop(t)

    t0 = pd.to_datetime(users["first_t"], utc=True).min()
    src = tweets[(tweets["kind"] == "source") & tweets["stance"].isin(["support", "deny"])].copy()
    src["t_h"] = (pd.to_datetime(src["created_at"], utc=True) - t0) / pd.Timedelta(hours=1)
    src["w"] = np.log1p(src["followers_count"].astype(float))
    src = src.sort_values("t_h")

    # percentile-rank lookup table: unique sample eta -> mid-rank share (same values as drv.mult_eff / kappa)
    eta_sorted = np.array(m["eta_sorted"])
    eta_u = np.unique(eta_sorted)
    rank_u = drv.mult_eff(eta_u, eta_sorted, 1.0)
    # eta depends on time only through Pop(t) in [0, 1]: tabulate each user's rank on a fine Pop grid
    rank_table = np.array([np.interp(base + slope * q, eta_u, rank_u, left=0.0, right=1.0) for q in POP_GRID])

    c_i_curve, c_a_curve = curves["C_I"].to_numpy() / N, curves["C_A"].to_numpy() / N
    return {"key": key, "variant": variant, "N": N, "grid": grid, "bin_h": bin_h, "upd": upd, "end_h": float(grid[-1]),
            "n_i": n_i, "n_a": n_a, "not_entered": not_entered, "c_i_data": c_i_data, "c_a_data": c_a_data,
            "rank_table": rank_table, "stories": ctx["stories"],
            "src_t": src["t_h"].to_numpy(), "src_to_i": (src["stance"] == "support").to_numpy(),
            "src_share": (src["w"] / src["w"].sum()).to_numpy(),
            "I0": c_i_curve[0], "A0": c_a_curve[0], "C_I": c_i_curve, "C_A": c_a_curve,
            "fin_I": c_i_curve[-1], "fin_A": c_a_curve[-1],
            "I_display": curves["I"].to_numpy() / N, "A_display": curves["A"].to_numpy() / N}


_CONTEXTS = {}


def get_context(key):
    """Cached set context (one copy per process; workers build their own on first use)."""
    if key not in _CONTEXTS:
        _CONTEXTS[key] = set_context(key)
    return _CONTEXTS[key]


def delta_at(sc, k, c_i, c_a, I, A, kappa, pop_k, kind, freeze_k=None):
    """d1 at update k: mean Drif1 over users not yet entered in the data (D45).
    freeze_k: data inputs (friend counts, C_data, not-entered mask) are taken at min(k, freeze_k) (forecast, D52)."""
    kd = k if freeze_k is None else min(k, freeze_k)
    mask = sc["not_entered"][kd]
    if P1_MODE[kind] == "global":
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
    d1 = np.broadcast_to(sigmoid((2 * p1 - 1) * me), mask.shape)
    return float(d1[mask].mean()) if mask.any() else float(d1.mean())   # plain float for the step loop


# ---------------------------------------------------------------- simulation

def simulate(sc, kind, params, dt=None, record=False, end_h=None, freeze_k=None):
    """Run one model (kind in MODELS; params as in NAMES[kind]) up to end_h (default: the set's end).
    Returns arrays t, C_I, C_A (or a DataFrame with S, I, A, R, C_I, C_A, d1, d2 and t if record)."""
    p = dict(zip(NAMES[kind], (float(v) for v in params)))
    theta, kappa = p["theta"], p.get("kappa", 0.0)
    S, I, A, R = 1.0 - sc["I0"] - sc["A0"], sc["I0"], sc["A0"], 0.0
    S, I, A, R = float(S), float(I), float(A), float(R)
    c_i, c_a = I, A
    uses_delta = kind != "const"
    pops = ft.pop_at(sc["stories"], sc["upd"], sc["grid"], p["half_life_h"]) if "half_life_h" in p else None
    # plain Python floats in the step loop (numpy scalars would make every update ~10x slower)
    src_t, src_to_i = sc["src_t"].tolist(), sc["src_to_i"].tolist()
    amounts = (theta * sc["src_share"]).tolist()
    dt = sc["bin_h"] / STEPS_PER_BIN if dt is None else dt
    upd, end, dt = sc["upd"].tolist(), float(sc["end_h"] if end_h is None else end_h), float(dt)
    if kind == "const":
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
                if kind in ("uniform", "eq26"):
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
        # all models use the same step grid and the nesting test is exact)
        nxt = end
        if j < len(src_t):
            nxt = min(nxt, src_t[j])
        if k_next < len(upd):
            nxt = min(nxt, upd[k_next])
        h = min(dt, nxt - t)
        S, I, A, R, gi, ga = euler_step(S, I, A, R, b_i, b_a, g, e, h)
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

def _objective(u, key, kind, target, hi, freeze_k):
    sc = get_context(key)
    names = NAMES[kind]
    bounds = fit_bounds(names)
    u = np.clip(u, [b[0] for b in bounds], [b[1] for b in bounds])
    ri, ra = residuals(sc, kind, from_fit(names, u), target, 0, hi, freeze_k)
    return float((ri ** 2).sum() + (ra ** 2).sum()) / (len(ri) + len(ra))


def run_start(job):
    """One bounded Nelder-Mead run in fitting coordinates (module-level so it runs in a worker process).
    job: key, kind, u0, tag, and optionally target (normalised C_I, C_A), hi (bins used), freeze_k, maxfev."""
    kind = job["kind"]
    names = NAMES[kind]
    res = minimize(_objective, job["u0"], args=(job["key"], kind, job.get("target"), job.get("hi"), job.get("freeze_k")),
                   method="Nelder-Mead", bounds=fit_bounds(names),
                   options={"maxfev": job.get("maxfev", MAX_FEV[kind]), "xatol": XATOL, "fatol": FATOL})
    lo, hi = zip(*fit_bounds(names))
    return {"tag": job["tag"], "x": from_fit(names, np.clip(res.x, lo, hi)), "mse": float(res.fun),
            "converged": bool(res.success), "nfev": int(res.nfev)}


def random_starts(kind, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = (np.array(b) for b in zip(*fit_bounds(NAMES[kind])))
    return [lo + rng.random(len(lo)) * (hi - lo) for _ in range(n)]


def start_jobs(key, kind, tag, n=N_STARTS, seed=dp.SEED, **extra):
    return [{"key": key, "kind": kind, "u0": u0, "tag": tag, **extra} for u0 in random_starts(kind, n, seed)]


def run_jobs(pool, jobs):
    """Run all jobs in the pool; group the results by tag."""
    out = {}
    for r in pool.map(run_start, jobs):
        out.setdefault(r["tag"], []).append(r)
    return out


def best_of(results):
    best = min(results, key=lambda r: r["mse"])
    return best, sum(r["converged"] for r in results)


def params_of(row, kind):
    return np.array([row[n] for n in NAMES[kind]], dtype=float)


# ---------------------------------------------------------------- validation pieces

def forecast_cut(sc):
    """Last training bin: the first bin where C_I reaches 50% of its final value (inclusive)."""
    return int(np.argmax(sc["C_I"] >= 0.5 * sc["fin_I"]))


def bootstrap_jobs(sc, kind, x_best, n, seed):
    """Residual bootstrap: fitted curve + residuals resampled with replacement within each curve."""
    ri, ra = residuals(sc, kind, x_best)
    fi, fa = ri + data_targets(sc)[0], ra + data_targets(sc)[1]          # fitted (model) curves
    rng = np.random.default_rng(seed)
    jobs = []
    for b in range(n):
        target = (fi - rng.choice(ri, len(ri), replace=True), fa - rng.choice(ra, len(ra), replace=True))
        jobs.append({"key": sc["key"], "kind": kind, "u0": to_fit(NAMES[kind], x_best), "tag": ("boot", sc["key"], kind, b),
                     "target": target})
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


def plot_states(runs):
    fig, axes = plt.subplots(3, len(SETS), figsize=(5 * len(SETS), 10), squeeze=False)
    for c, key in enumerate(SETS):
        r = runs[key][MAIN]
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
    fig.suptitle("Plot 1 - fitted SAIR-H (model d): S/A/I/R state fractions", fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot1_states.png")


def plot_cumulative(runs, table):
    shown = ["const", "eq26", "sairh", "meanfield"]
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 3.6 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc = get_context(key)
        for c, (col, s) in enumerate((("C_I", "I"), ("C_A", "A"))):
            ax = axes[r_i, c]
            final = sc[f"fin_{s}"]
            ax.plot(sc["grid"], sc[col] / final, color="#0b0b0b", lw=0, marker="o", ms=3, label="real data")
            for kind in shown:
                r = runs[key][kind]
                row = table[(table["set"] == key) & (table["model"] == kind)].iloc[0]
                ax.plot(r["t"], r[col] / final, color=MODEL_COLORS[kind], lw=2, ls="--" if kind == "eq26" else "-",
                        label=f"{MODEL_LABEL[kind]}  RMSE {row[f'rmse_{s}']:.3f}")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            ax.set_ylim(0, max(1.15, ax.get_ylim()[1]))
            dp.style(ax, f"{set_title(key)}: {col} / final", "Share of final value",
                     xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle("Plot 2 - predicted vs real cumulative entries (fitting target)", fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot2_cumulative_fit.png")


def plot_prevalence(runs):
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 3.6 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc, r = get_context(key), runs[key][MAIN]
        for c, s in enumerate("IA"):
            ax = axes[r_i, c]
            ax.plot(sc["grid"], sc[f"{s}_display"], color="#0b0b0b", lw=1.5, ls="--",
                    label="real (display: exp. lifespan 12 h)")
            ax.plot(r["t"], r[s], color=dp.COLORS[s], lw=2, label="SAIR-H (d)")
            ax.legend(frameon=False, fontsize=8)
            dp.style(ax, f"{set_title(key)}: {s} prevalence", "Fraction of N",
                     xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle("Plot 2b - SAIR-H I and A prevalence vs the display curves (visual only, not fitted)",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot2b_prevalence.png")


def plot_forecast(fc_runs, fc_table):
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 3.6 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc = get_context(key)
        cut_t = sc["grid"][forecast_cut(sc)]
        for c, (col, s) in enumerate((("C_I", "I"), ("C_A", "A"))):
            ax = axes[r_i, c]
            final = sc[f"fin_{s}"]
            ax.axvspan(sc["grid"][0], cut_t, color="#e6e5e1", alpha=0.6, lw=0, label=f"training window (to {cut_t:g} h)")
            ax.plot(sc["grid"], sc[col] / final, color="#0b0b0b", lw=0, marker="o", ms=3, label="real data")
            for kind in ("const", MAIN):
                r = fc_runs[key][kind]
                row = fc_table[(fc_table["set"] == key) & (fc_table["model"] == kind)].iloc[0]
                ax.plot(r["t"], r[col] / final, color=MODEL_COLORS[kind], lw=2,
                        label=f"{MODEL_LABEL[kind]}  held-out RMSE {row[f'fc_rmse_{s}']:.3f}")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            ax.set_ylim(0, max(1.15, ax.get_ylim()[1]))
            dp.style(ax, f"{set_title(key)}: {col} / final", "Share of final value",
                     xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle("Plot 2c - forecast: fit up to the 50% point of C_I (shaded), predict the rest",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot2c_forecast.png")


# ---------------------------------------------------------------- main

def fit_row(key, kind, results):
    sc = get_context(key)
    best, n_conv = best_of(results)
    x = best["x"]
    m = metrics(*residuals(sc, kind, x), len(NAMES[kind]))
    return {"set": key, "model": kind, **{n: x[NAMES[kind].index(n)] if n in NAMES[kind] else np.nan for n in ALL_PARAMS},
            **m, "bound_flags": bound_flags(NAMES[kind], x), "best_converged": best["converged"],
            "starts_converged": n_conv, "starts": len(results), "nfev_total": sum(r["nfev"] for r in results)}


def merge_saved(new, name, fit_sets):
    """Replace the rows of the refitted sets in a saved results table (keeps the other sets' rows)."""
    path = os.path.join(dp.RESULTS, name)
    if set(fit_sets) != set(SETS) and os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[~old["set"].isin(fit_sets)], new], ignore_index=True)
        new = new.sort_values("set", key=lambda c: c.map(SETS.index), kind="stable").reset_index(drop=True)
    new.to_csv(path, index=False)
    return new


def main(fit_sets=SETS):
    """Fit and validate the sets in fit_sets; rows of the other sets are kept from the saved tables."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for key in SETS:
        get_context(key)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        # wave 1: main fits (all models) and forecast fits ((a) and (d))
        t0 = time.time()
        jobs = []
        for key in fit_sets:
            sc = get_context(key)
            k_cut = forecast_cut(sc)
            for kind in MODELS:
                jobs += start_jobs(key, kind, ("fit", key, kind))
            for kind in ("const", MAIN):
                jobs += start_jobs(key, kind, ("fc", key, kind), hi=k_cut + 1, freeze_k=k_cut)
        res = run_jobs(pool, jobs)
        print(f"wave 1 (main + forecast fits, {len(jobs)} runs): {time.time() - t0:.0f} s", flush=True)

        table = pd.DataFrame([fit_row(key, kind, res[("fit", key, kind)]) for key in fit_sets for kind in MODELS])
        table["dAIC_vs_best"] = table["AIC"] - table.groupby("set")["AIC"].transform("min")
        table = merge_saved(table, "phase5_params.csv", fit_sets)

        fc_rows = []
        for key in fit_sets:
            sc = get_context(key)
            k_cut = forecast_cut(sc)
            for kind in ("const", MAIN):
                best, n_conv = best_of(res[("fc", key, kind)])
                ri, ra = residuals(sc, kind, best["x"], lo=k_cut + 1, freeze_k=k_cut)
                fc_rows.append({"set": key, "model": kind, "cut_h": sc["grid"][k_cut], "train_points": 2 * (k_cut + 1),
                                "test_points": len(ri) + len(ra), "k": len(NAMES[kind]),
                                "fc_rmse_I": float(np.sqrt(np.mean(ri ** 2))), "fc_rmse_A": float(np.sqrt(np.mean(ra ** 2))),
                                "fc_rmse_pooled": float(np.sqrt(np.mean(np.r_[ri, ra] ** 2))),
                                "fc_starts_converged": n_conv, "fc_bound_flags": bound_flags(NAMES[kind], best["x"]),
                                **{f"fc_{n}": v for n, v in zip(NAMES[kind], best["x"])}})
        fc_table = merge_saved(pd.DataFrame(fc_rows), "phase5_forecast.csv", fit_sets)

        # wave 2: parameter recovery (CH, model d) and residual bootstrap ((a), (d), every set)
        t0 = time.time()
        jobs = []
        x_true = params_of(table[(table["set"] == "charliehebdo") & (table["model"] == MAIN)].iloc[0], MAIN)
        sc_ch = get_context("charliehebdo")
        if "charliehebdo" in fit_sets:
            for seed in RECOVERY_SEEDS:
                jobs += start_jobs("charliehebdo", MAIN, ("rec", seed), n=N_RECOVERY_STARTS, seed=dp.SEED + seed,
                                   target=recovery_target(sc_ch, MAIN, x_true, seed))
        for key in fit_sets:
            for kind in ("const", MAIN):
                x_best = params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind)
                jobs += bootstrap_jobs(get_context(key), kind, x_best, N_BOOT, dp.SEED)
        res = run_jobs(pool, jobs)
        print(f"wave 2 (recovery + bootstrap, {len(jobs)} runs): {time.time() - t0:.0f} s", flush=True)

    if "charliehebdo" in fit_sets:
        rec_rows = []
        for seed in RECOVERY_SEEDS:
            best, n_conv = best_of(res[("rec", seed)])
            for n, tv, rv in zip(NAMES[MAIN], x_true, best["x"]):
                rec_rows.append({"noise_seed": seed, "parameter": n, "true": tv, "recovered": rv, "abs_error": rv - tv,
                                 "pct_error": 100 * (rv - tv) / tv if tv != 0 else np.nan,
                                 "starts_converged": n_conv, "starts": N_RECOVERY_STARTS, "fit_mse": best["mse"]})
        rec = pd.DataFrame(rec_rows)
        rec.to_csv(os.path.join(dp.RESULTS, "phase5_recovery.csv"), index=False)
    else:
        rec = pd.read_csv(os.path.join(dp.RESULTS, "phase5_recovery.csv"))

    boot_rows = []
    for key in fit_sets:
        for kind in ("const", MAIN):
            sc = get_context(key)
            rs = [r for (tag, rr) in res.items() if tag[0] == "boot" and tag[1] == key and tag[2] == kind for r in rr]
            xs = np.array([r["x"] for r in rs])
            rm = np.array([metrics(*residuals(sc, kind, r["x"]), 0)["rmse_mean"] for r in rs])
            est = table[(table["set"] == key) & (table["model"] == kind)].iloc[0]
            n_conv = sum(r["converged"] for r in rs)
            for j, n in enumerate(NAMES[kind] + ["rmse_mean_vs_data"]):
                v = xs[:, j] if j < len(NAMES[kind]) else rm
                boot_rows.append({"set": key, "model": kind, "parameter": n,
                                  "estimate": est[n] if n in est else est["rmse_mean"],
                                  "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)),
                                  "boot_median": float(np.median(v)), "resamples": len(rs), "resamples_converged": n_conv})
    boot = merge_saved(pd.DataFrame(boot_rows), "phase5_bootstrap_ci.csv", fit_sets)

    comp = table[["set", "model", "k", "n", "SSE", "rmse_I", "rmse_A", "rmse_mean", "rmse_pooled", "AIC", "BIC",
                  "dAIC_vs_best", "bound_flags", "starts_converged", "starts", "best_converged"]].merge(
        fc_table[["set", "model", "cut_h", "train_points", "fc_rmse_I", "fc_rmse_A", "fc_rmse_pooled",
                  "fc_starts_converged", "fc_bound_flags"]], on=["set", "model"], how="left")
    comp.insert(2, "label", comp["model"].map(MODEL_LABEL))
    comp.to_csv(os.path.join(dp.RESULTS, "phase5_model_comparison.csv"), index=False)

    runs, fc_runs = {}, {}
    for key in SETS:
        sc = get_context(key)
        fc_runs[key] = {}
        for kind in ("const", MAIN):
            row = fc_table[(fc_table["set"] == key) & (fc_table["model"] == kind)].iloc[0]
            fc_runs[key][kind] = simulate(sc, kind, [row[f"fc_{n}"] for n in NAMES[kind]], record=True,
                                          freeze_k=forecast_cut(sc))
        runs[key] = {kind: simulate(sc, kind, params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind),
                                    record=True) for kind in MODELS}
        runs[key][MAIN].to_csv(os.path.join(dp.PROCESSED, f"model_curves_{key}.csv"), index=False)
    plot_states(runs)
    plot_cumulative(runs, table)
    plot_prevalence(runs)
    plot_forecast(fc_runs, fc_table)

    print("\n=== Fitted parameters ===")
    print(table[["set", "model"] + ALL_PARAMS + ["bound_flags", "starts_converged", "best_converged"]].round(4).to_string(index=False))
    print("\n=== Model comparison (fit on all bins; forecast = fit to the 50% point of C_I, RMSE on the rest) ===")
    print(comp.drop(columns=["label"]).round(4).to_string(index=False))
    print("\n=== Forecast fits ===")
    print(fc_table.round(4).to_string(index=False))
    print("\n=== Parameter recovery (CH, model d, sigma = 0.01) ===")
    print(rec.round(4).to_string(index=False))
    print("\n=== Bootstrap 95% CIs ===")
    print(boot.round(4).to_string(index=False))


# ---------------------------------------------------------------- check

def check():
    print("\n--- Phase 5 check ---")
    S, I, A, R, *_ = euler_step(0.90, 0.05, 0.05, 0.0, 0.6, 0.4, 1 - 0.6, 1 - 0.4, 1.0)
    assert np.allclose([S, I, A, R], [0.855, 0.057, 0.038, 0.050], atol=5e-4), (S, I, A, R)
    print(f"unit test (docs worked step, Eq. 26): S={S:.3f} I={I:.3f} A={A:.3f} R={R:.3f}  OK")

    table = pd.read_csv(os.path.join(dp.RESULTS, "phase5_params.csv"))
    comp = pd.read_csv(os.path.join(dp.RESULTS, "phase5_model_comparison.csv"))
    for key in SETS:
        sc = get_context(key)
        # nesting: (d) with kappa = 0 has d1 = d2 = 1/2, so (d)(beta, alpha) == (a)(beta/2, alpha/2)
        x = params_of(table[(table["set"] == key) & (table["model"] == MAIN)].iloc[0], MAIN)
        p = dict(zip(NAMES[MAIN], x))
        xd = [p["beta"], p["alpha"], p["gamma"], p["eps"], p["half_life_h"], 0.0, p["theta"]]
        xa = [p["beta"] / 2, p["alpha"] / 2, p["gamma"], p["eps"], p["theta"]]
        rd, ra_ = simulate(sc, MAIN, xd, record=True), simulate(sc, "const", xa, record=True)
        nest = float(np.abs(rd[["S", "I", "A", "R", "C_I", "C_A"]].to_numpy() - ra_[["S", "I", "A", "R", "C_I", "C_A"]].to_numpy()).max())
        assert np.allclose(rd["d1"], 0.5, atol=1e-12) and nest <= 1e-9, f"{key}: nesting test failed ({nest:.2e})"

        worst_sum, worst_dt = 0.0, {}
        hours = np.arange(0, sc["end_h"] + 1e-9, 1.0)
        dt_main = sc["bin_h"] / STEPS_PER_BIN
        for kind in MODELS:
            x = params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind)
            r = simulate(sc, kind, x, record=True)
            states = r[["S", "I", "A", "R"]].to_numpy()
            worst_sum = max(worst_sum, float(np.abs(states.sum(axis=1) - 1).max()))
            assert worst_sum <= INV_TOL, f"{key}/{kind}: states do not sum to 1 ({worst_sum:.2e})"
            assert states.min() >= -INV_TOL and states.max() <= 1 + INV_TOL, f"{key}/{kind}: a state outside [0, 1]"
            if kind != "const":
                assert r["d1"].between(0, 1).all() and r["d2"].between(0, 1).all(), f"{key}/{kind}: delta outside [0, 1]"
            b = simulate(sc, kind, x, dt=dt_main / 2, record=True)         # dt-convergence for every model (D47)
            d = max(float(np.abs(np.interp(hours, r["t"], r[s]) - np.interp(hours, b["t"], b[s])).max())
                    for s in ["S", "I", "A", "R", "C_I", "C_A"])
            assert d < 1e-3, f"{key}/{kind}: dt-convergence failed, max |diff| = {d:.2e}"
            worst_dt[kind] = d
        c = comp[comp["set"] == key].set_index("model")
        assert (c["starts_converged"] >= MIN_CONVERGED).all(), \
            f"{key}: fewer than {MIN_CONVERGED}/{N_STARTS} converged starts: {c['starts_converged'].to_dict()}"
        if key == "charliehebdo":
            assert c.loc["const", "best_converged"], "Charlie Hebdo (a): the best start did not converge"
        for m in c.index[~c["best_converged"].astype(bool)]:
            print(f"{key:22s} WARNING  {m}: best start hit maxfev (flags: {c.loc[m, 'bound_flags']})")
        print(f"{key:22s} OK  nesting max|diff|={nest:.1e}  max|sum-1|={worst_sum:.1e}  "
              f"dt {dt_main:g} vs {dt_main / 2:g}: (d) {worst_dt[MAIN]:.1e}, (e) {worst_dt['meanfield']:.1e}, all <= {max(worst_dt.values()):.1e}")
        print(f"{'':22s}     converged starts: " + ", ".join(f"{m} {int(c.loc[m, 'starts_converged'])}/{N_STARTS}" for m in MODELS)
              + "; forecast: " + ", ".join(f"{m} {int(c.loc[m, 'fc_starts_converged'])}/{N_STARTS}" for m in ("const", MAIN)))
    rec = pd.read_csv(os.path.join(dp.RESULTS, "phase5_recovery.csv"))
    summary = rec.groupby("parameter", sort=False).agg(true=("true", "first"), median_recovered=("recovered", "median"),
                                                       median_abs_pct_error=("pct_error", lambda v: np.median(np.abs(v))))
    print("parameter recovery (CH, model d, 5 noise seeds): median |% error| per parameter")
    print(summary.round(4).to_string())
    boot = pd.read_csv(os.path.join(dp.RESULTS, "phase5_bootstrap_ci.csv"))
    assert (boot["resamples"] == N_BOOT).all()
    print(f"bootstrap: {N_BOOT} resamples per set and model; converged "
          + ", ".join(f"{s[:5]}/{m} {int(n)}" for (s, m), n in boot.groupby(["set", "model"], sort=False)["resamples_converged"].first().items()))
    print("All Phase 5 checks passed.")


if __name__ == "__main__":
    if "--check-only" not in sys.argv:
        main([s for s in SETS if s in sys.argv[sys.argv.index("--sets") + 1:]] if "--sets" in sys.argv else SETS)
    if "--check" in sys.argv or "--check-only" in sys.argv:
        check()
