"""Phase 5 - SAIR simulation, fitting and validation (docs/SAIR_paper.md §5).

State = fractions of N: S, I, A, R. Corrected SAIR (CLAUDE.md), explicit Euler, dt = 0.02 h (events hit exactly):
    dS = -d1*I*S - d2*A*S
    dI =  d1*I*S - (1 - d1)*I
    dA =  d2*A*S - (1 - d2)*A
    dR = (1 - d1)*I + (1 - d2)*A
Push rates d1(t), d2(t) (DECISIONS D45), recomputed every data bin (1 h for the events):
    rho_I = C_I_model / C_I_data (1 if data is 0; clipped to [0, 5]), rho_A the same
    P1_i  = rho_I*n_I,i / (rho_I*n_I,i + rho_A*n_A,i), n = friends who entered I / A in the data by t;
            no sided friends -> I/(I + A) of the model state (0.5 if both 0)
    Drif1_i = sigmoid((2*P1_i - 1) * Mult_eff_i(t)),  Mult_eff_i(t) = kappa * pct_rank(eta_i(t)), Pop at t
    d1 = mean Drif1 over users who have not entered in the data by t;  d2 = 1 - d1
Source injection (D46): at each supporting / denying source tweet's time, theta * w_s / sum(w) moves S -> I / A.
Fit: half-life w, kappa, theta by Nelder-Mead (10 random starts), loss = mean of the RMSEs of the
cumulative curves C_I and C_A, each divided by its final data value.
Baselines: (a) constant-rate SAIR (Eq. 22) and (b) uniform Mult_eff = kappa, same loss and injection.

Usage:
    python src/sair.py           fit all models on all sets, save parameters and plots
    python src/sair.py --check   fit, then run the Phase 5 checks
    python src/sair.py --check-only   checks on the saved parameters (no refit)
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
DT = 0.02                                    # Euler step, hours (0.1 failed the convergence check, D47)
DT_CHECK = 0.01
RHO_MAX = 5.0
N_STARTS = 10
MAX_FEV = {"full": 300, "uniform": 200, "const": 600}
MODELS = ["const", "uniform", "full"]        # (a), (b), (c)
MODEL_LABEL = {"const": "(a) constant-rate SAIR", "uniform": "(b) uniform Mult_eff", "full": "(c) full model",
               "full_followgroups": "(c) full, follow-Louvain groups"}
BOUNDS = {"full": [(0.5, 48.0), (0.1, 10.0), (0.0, 0.5)],               # w, kappa, theta
          "uniform": [(0.1, 10.0), (0.0, 0.5)],                          # kappa, theta
          "const": [(0.0, 1.0)] * 4 + [(0.0, 0.5)]}                      # alpha, beta, eps, gamma, theta
NAMES = {"full": ["half_life_h", "kappa", "theta"], "uniform": ["kappa", "theta"],
         "const": ["alpha", "beta", "eps", "gamma", "theta"]}
MODEL_COLORS = {"const": "#2a78d6", "uniform": "#eb6834", "full": "#1baf7a"}
INV_TOL = 1e-9
POP_GRID = np.linspace(0, 1, 201)            # Mult_eff rank table resolution in Pop (D45)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def euler_step(S, I, A, R, d1, d2, dt):
    """One explicit Euler step of the corrected SAIR equations (CLAUDE.md)."""
    new_i = d1 * I * S
    new_a = d2 * A * S
    out_i = (1 - d1) * I
    out_a = (1 - d2) * A
    return (S - (new_i + new_a) * dt, I + (new_i - out_i) * dt, A + (new_a - out_a) * dt,
            R + (out_i + out_a) * dt, new_i * dt, new_a * dt)


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

    return {"key": key, "variant": variant, "N": N, "grid": grid, "bin_h": bin_h, "upd": upd, "end_h": float(grid[-1]),
            "n_i": n_i, "n_a": n_a, "not_entered": not_entered, "c_i_data": c_i_data, "c_a_data": c_a_data,
            "rank_table": rank_table, "stories": ctx["stories"],
            "src_t": src["t_h"].to_numpy(), "src_to_i": (src["stance"] == "support").to_numpy(),
            "src_share": (src["w"] / src["w"].sum()).to_numpy(),
            "I0": curves["C_I"].iloc[0] / N, "A0": curves["C_A"].iloc[0] / N,
            "C_I": curves["C_I"].to_numpy() / N, "C_A": curves["C_A"].to_numpy() / N,
            "I_display": curves["I"].to_numpy() / N, "A_display": curves["A"].to_numpy() / N}


def delta_at(sc, k, c_i, c_a, I, A, kappa, pop_k, uniform):
    """d1 at update k: mean Drif1 over users not yet entered in the data (D45)."""
    N = sc["N"]
    rho_i = np.clip(c_i * N / sc["c_i_data"][k], 0, RHO_MAX) if sc["c_i_data"][k] > 0 else 1.0
    rho_a = np.clip(c_a * N / sc["c_a_data"][k], 0, RHO_MAX) if sc["c_a_data"][k] > 0 else 1.0
    num = rho_i * sc["n_i"][k]
    den = num + rho_a * sc["n_a"][k]
    fallback = I / (I + A) if I + A > 0 else 0.5
    p1 = np.where(den > 0, num / np.where(den > 0, den, 1.0), fallback)
    if uniform:
        me = kappa
    else:
        # percentile rank of eta_i(t) = base_i + slope_i * Pop(t), linear in Pop between tabulated grid points
        x = pop_k * (len(POP_GRID) - 1)
        g = min(int(x), len(POP_GRID) - 2)
        f = x - g
        me = kappa * ((1 - f) * sc["rank_table"][g] + f * sc["rank_table"][g + 1])
    d1 = sigmoid((2 * p1 - 1) * me)
    mask = sc["not_entered"][k]
    return float(d1[mask].mean()) if mask.any() else float(d1.mean())   # plain float for the step loop


# ---------------------------------------------------------------- simulation

def simulate(sc, kind, params, dt=DT, record=False):
    """Run one model. kind in {'full', 'uniform', 'const'}; params as in NAMES[kind].
    Returns arrays of t, C_I, C_A (and S, I, A, R, d1, d2 if record)."""
    p = dict(zip(NAMES[kind], params))
    theta = p["theta"]
    S, I, A, R = 1.0 - sc["I0"] - sc["A0"], sc["I0"], sc["A0"], 0.0
    c_i, c_a = sc["I0"], sc["A0"]
    if kind == "full":
        pops = ft.pop_at(sc["stories"], sc["upd"], sc["grid"], p["half_life_h"])
    # plain Python floats in the step loop (numpy scalars would make every update ~10x slower)
    src_t, src_to_i = sc["src_t"].tolist(), sc["src_to_i"].tolist()
    amounts = (theta * sc["src_share"]).tolist()
    upd, end, dt = sc["upd"].tolist(), float(sc["end_h"]), float(dt)
    S, I, A, R, c_i, c_a = float(S), float(I), float(A), float(R), float(c_i), float(c_a)
    uses_delta = kind != "const"
    j, k_next, d1, t = 0, 0, 0.5, 0.0
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
        if uses_delta and k_next < len(upd) and upd[k_next] <= t + 1e-9:
            d1 = delta_at(sc, k_next, c_i, c_a, I, A, p["kappa"], pops[k_next] if kind == "full" else 0.0,
                          kind == "uniform")
            k_next += 1
        if record:
            out.append((t, S, I, A, R, c_i, c_a, d1 if uses_delta else np.nan, 1 - d1 if uses_delta else np.nan))
        else:
            out.append((t, c_i, c_a))
        if t >= end - 1e-9:
            break
        # Euler step of at most dt that stops exactly at the next event, so event timing does not depend on dt
        nxt = end
        if j < len(src_t):
            nxt = min(nxt, src_t[j])
        if uses_delta and k_next < len(upd):
            nxt = min(nxt, upd[k_next])
        h = min(dt, nxt - t)
        if kind == "const":
            new_i = p["beta"] * I * S
            new_a = p["alpha"] * A * S
            S, I, A, R = (S - (new_i + new_a) * h, I + (new_i - p["gamma"] * I) * h,
                          A + (new_a - p["eps"] * A) * h, R + (p["gamma"] * I + p["eps"] * A) * h)
            c_i, c_a = c_i + new_i * h, c_a + new_a * h
        else:
            S, I, A, R, gi, ga = euler_step(S, I, A, R, d1, 1 - d1, h)
            c_i, c_a = c_i + gi, c_a + ga
        t = nxt if nxt - (t + h) < 1e-9 else t + h
    out = np.array(out)
    if record:
        return pd.DataFrame(out[:, 1:], columns=["S", "I", "A", "R", "C_I", "C_A", "d1", "d2"]).assign(t=out[:, 0])
    return out[:, 0], out[:, 1], out[:, 2]


def losses(sc, t_arr, c_i, c_a):
    """RMSE of C/C_final (model vs data) at the data bins, for I and A, and their mean."""
    fi, fa = sc["C_I"][-1], sc["C_A"][-1]
    mi = np.interp(sc["grid"], t_arr, c_i) / fi
    ma = np.interp(sc["grid"], t_arr, c_a) / fa
    rmse_i = float(np.sqrt(np.mean((mi - sc["C_I"] / fi) ** 2)))
    rmse_a = float(np.sqrt(np.mean((ma - sc["C_A"] / fa) ** 2)))
    return rmse_i, rmse_a, (rmse_i + rmse_a) / 2


def _objective(x, sc, kind, lo, hi):
    t_arr, c_i, c_a = simulate(sc, kind, np.clip(x, lo, hi))
    return losses(sc, t_arr, c_i, c_a)[2]


def _run_start(args):
    """One Nelder-Mead run from start x0 (module-level so it can run in a worker process)."""
    sc, kind, x0 = args
    bounds = BOUNDS[kind]
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    res = minimize(_objective, x0, args=(sc, kind, lo, hi), method="Nelder-Mead", bounds=bounds,
                   options={"maxfev": MAX_FEV[kind], "xatol": 1e-4, "fatol": 1e-6})
    return res.x, float(res.fun), bool(res.success), int(res.nfev)


def fit(sc, kind, pool=None):
    """Nelder-Mead with N_STARTS random starts (seed 42) inside the bounds; keep the best.
    Starts are independent, so they run in parallel when a process pool is given."""
    bounds = BOUNDS[kind]
    rng = np.random.default_rng(dp.SEED)
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    starts = [lo + rng.random(len(bounds)) * (hi - lo) for _ in range(N_STARTS)]
    jobs = [(sc, kind, x0) for x0 in starts]
    results = list(pool.map(_run_start, jobs)) if pool else [_run_start(j) for j in jobs]
    best = min(results, key=lambda r: r[1])
    n_success = sum(r[2] for r in results)
    nfev = sum(r[3] for r in results)
    best_success = best[2]
    x = np.clip(best[0], lo, hi)
    t_arr, c_i, c_a = simulate(sc, kind, x)
    rmse_i, rmse_a, rmse = losses(sc, t_arr, c_i, c_a)
    return {"set": sc["key"], "model": kind if sc["variant"] == "profile" else f"{kind}_followgroups",
            **dict(zip(NAMES[kind], x)), "rmse_I": rmse_i, "rmse_A": rmse_a, "rmse_mean": rmse,
            "best_converged": best_success, "starts_converged": n_success, "starts": N_STARTS, "nfev_total": nfev}


def params_of(row, kind):
    return np.array([row[n] for n in NAMES[kind]], dtype=float)


# ---------------------------------------------------------------- plots

def plot_states(runs):
    fig, axes = plt.subplots(3, len(SETS), figsize=(5 * len(SETS), 10), squeeze=False)
    for c, key in enumerate(SETS):
        r = runs[key]["full"]
        ax = axes[0, c]
        for s, label in (("S", "S susceptible"), ("R", "R removed")):
            ax.plot(r["t"], r[s], color=dp.COLORS[s], lw=2, label=label)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8)
        dp.style(ax, f"{dp.EVENT_NAMES.get(key, 'Ottawa story')}: S and R", "Fraction of N", xlabel="")
        for row, s, label in ((1, "I", "I - rumour spreaders"), (2, "A", "A - correction spreaders")):
            ax = axes[row, c]
            ax.plot(r["t"], r[s], color=dp.COLORS[s], lw=2)
            ax.set_ylim(0, max(r[s].max() * 1.15, 1e-6))
            dp.style(ax, label, "Fraction of N", xlabel="Hours since first post" if row == 2 else "")
    fig.suptitle("Plot 1 - fitted full SAIR model: S/A/I/R state fractions", fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot1_states.png")


def plot_cumulative(runs, scs, table):
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 3.6 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc = scs[key]
        for c, (col, s) in enumerate((("C_I", "I"), ("C_A", "A"))):
            ax = axes[r_i, c]
            final = sc[col][-1]
            ax.plot(sc["grid"], sc[col] / final, color="#0b0b0b", lw=0, marker="o", ms=3, label="real data")
            for kind in MODELS:
                r = runs[key][kind]
                row = table[(table["set"] == key) & (table["model"] == kind)].iloc[0]
                ax.plot(r["t"], r[col] / final, color=MODEL_COLORS[kind], lw=2,
                        label=f"{MODEL_LABEL[kind]}  RMSE {row[f'rmse_{s}']:.3f}")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            ax.set_ylim(0, max(1.15, ax.get_ylim()[1]))
            dp.style(ax, f"{dp.EVENT_NAMES.get(key, 'Ottawa story')}: {col} / final", "Share of final value",
                     xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle("Plot 2 - predicted vs real cumulative entries (fitting target), three models",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot2_cumulative_fit.png")


def plot_prevalence(runs, scs):
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 3.6 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc, r = scs[key], runs[key]["full"]
        for c, s in enumerate("IA"):
            ax = axes[r_i, c]
            ax.plot(sc["grid"], sc[f"{s}_display"], color="#0b0b0b", lw=1.5, ls="--",
                    label="real (display: exp. lifespan 12 h)")
            ax.plot(r["t"], r[s], color=dp.COLORS[s], lw=2, label="full model")
            ax.legend(frameon=False, fontsize=8)
            dp.style(ax, f"{dp.EVENT_NAMES.get(key, 'Ottawa story')}: {s} prevalence", "Fraction of N",
                     xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle("Plot 2b - model I and A prevalence vs the display curves (visual only, not fitted)",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase5_plot2b_prevalence.png")


# ---------------------------------------------------------------- main + check

def main():
    pd.set_option("display.width", 230)
    rows, scs, runs = [], {}, {}
    with ProcessPoolExecutor(max_workers=N_STARTS) as pool:
        for key in SETS:
            sc = set_context(key)
            scs[key] = sc
            for kind in MODELS:
                t0 = time.time()
                rows.append(fit(sc, kind, pool))
                print(f"{key:22s} {kind:8s} mean RMSE {rows[-1]['rmse_mean']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
            sc_f = set_context(key, "follow")
            t0 = time.time()
            rows.append(fit(sc_f, "full", pool))
            print(f"{key:22s} full/follow-groups mean RMSE {rows[-1]['rmse_mean']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(dp.RESULTS, "phase5_params.csv"), index=False)
    for key in SETS:
        runs[key] = {kind: simulate(scs[key], kind, params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind),
                                    record=True) for kind in MODELS}
        runs[key]["full"].to_csv(os.path.join(dp.PROCESSED, f"model_curves_{key}.csv"), index=False)
    plot_states(runs)
    plot_cumulative(runs, scs, table)
    plot_prevalence(runs, scs)
    print("\n=== Fitted parameters and RMSE (loss = mean of RMSE(C_I/final), RMSE(C_A/final)) ===")
    print(table.round(4).to_string(index=False))


def check():
    print("\n--- Phase 5 check ---")
    S, I, A, R, *_ = euler_step(0.90, 0.05, 0.05, 0.0, 0.6, 0.4, 1.0)
    assert np.allclose([S, I, A, R], [0.855, 0.057, 0.038, 0.050], atol=5e-4), (S, I, A, R)
    print(f"unit test (docs worked step): S={S:.3f} I={I:.3f} A={A:.3f} R={R:.3f}  OK")
    table = pd.read_csv(os.path.join(dp.RESULTS, "phase5_params.csv"))
    for key in SETS:
        scs = {"profile": set_context(key), "follow": set_context(key, "follow")}
        worst_sum, worst_range = 0.0, 0.0
        for _, row in table[table["set"] == key].iterrows():
            kind = row["model"].replace("_followgroups", "")
            sc = scs["follow" if row["model"].endswith("_followgroups") else "profile"]
            r = simulate(sc, kind, params_of(row, kind), record=True)
            states = r[["S", "I", "A", "R"]].to_numpy()
            worst_sum = max(worst_sum, float(np.abs(states.sum(axis=1) - 1).max()))
            assert worst_sum <= INV_TOL, f"{key}/{row['model']}: states do not sum to 1 ({worst_sum:.2e})"
            low, high = states.min(), states.max()
            worst_range = max(worst_range, -low, high - 1)
            assert low >= -INV_TOL and high <= 1 + INV_TOL, f"{key}/{row['model']}: a state outside [0, 1]"
            if kind != "const":
                assert r["d1"].between(0, 1).all() and r["d2"].between(0, 1).all(), f"{key}/{row['model']}: delta outside [0, 1]"
        sc = scs["profile"]
        hours = np.arange(0, sc["end_h"] + 1e-9, 1.0)
        diff = 0.0
        for kind in MODELS:                              # dt-convergence for every model (D47)
            x = params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind)
            a = simulate(sc, kind, x, dt=DT, record=True)
            b = simulate(sc, kind, x, dt=DT_CHECK, record=True)
            d = max(float(np.abs(np.interp(hours, a["t"], a[s]) - np.interp(hours, b["t"], b[s])).max())
                    for s in ["S", "I", "A", "R", "C_I", "C_A"])
            assert d < 1e-3, f"{key}/{kind}: dt-convergence failed, max |diff| = {d:.2e}"
            diff = max(diff, d)
        conv = table[(table["set"] == key) & (table["model"] == "full")]["starts_converged"].iloc[0]
        assert conv >= 1, f"{key}: no converged Nelder-Mead start for the full model"
        rm = table[table["set"] == key].set_index("model")["rmse_mean"]
        verdict = ", ".join(f"{'beats' if rm['full'] < rm[k] else 'does NOT beat'} {k} ({rm[k]:.4f})" for k in ("const", "uniform"))
        print(f"{key:22s} OK  max|sum-1|={worst_sum:.1e}  dt {DT} vs {DT_CHECK} max|diff| (all models)={diff:.1e}  "
              f"converged starts (full)={conv}/{N_STARTS}  full RMSE {rm['full']:.4f}: {verdict}")
    print("All Phase 5 checks passed.")


if __name__ == "__main__":
    if "--check-only" not in sys.argv:
        main()
    if "--check" in sys.argv or "--check-only" in sys.argv:
        check()
