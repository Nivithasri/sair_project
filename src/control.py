"""Phase 7 - SAIR-C: blocking and correction-campaign control on the fitted main model (docs/extension.md §1, D73).

State = fractions of N: S, I, A, R, B. The main model (d+)-uniform (Phase 5) plus two constant knobs from t = 0:
    blocking            dI -= u*I,  dB = u*I
    correction campaign dS -= v*S,  dA += v*S      (campaign recruits count as A entries, i.e. in C_A)
Network feedback: blocked users stop posting, so P1 uses rho_I = (C_I,model - B) / C_I,data.
Knobs are event-relative (D73): u = lam_u * gamma (gamma = the set's fitted removal rate of I),
v = lam_v / T90 (T90 = hour at which the data's C_I reaches 90% of its final value).
Outcomes: D = integral of I dt over the set's window (rumour exposure), D / D0 (D0 = no control), rumour peak,
final C_I, C_A, B, effort E_u = integral u*I dt (= final B), E_v = integral v*S dt, E = E_u + E_v (fractions of N).

Usage:
    python src/control.py --check             stage 1 (runs) then stage 2 (tables, plots), then the check
    python src/control.py --stage 1|2 ...     one stage only
    python src/control.py --fresh ...         delete the stage-1 checkpoint first
    python src/control.py --check-only        check on the saved results
    python src/control.py --plot4-10pct       Plot 4b (E = 10% of N, with C_A(t)) from the saved phase7_strategies.csv
"""
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
from matplotlib.colors import LogNorm

import data_prep as dp
import sair

SETS = sair.SETS
LAM_U = [0, 0.25, 0.5, 1, 2, 3, 5]                  # blocking rate in units of gamma
LAM_V = [0, 0.1, 0.25, 0.5, 1, 1.5, 2]              # campaign rate in units of 1 / T90
BUDGETS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]       # total effort E_u + E_v, fraction of N (2/5/10% asked)
MAIN_BUDGETS = [0.02, 0.05, 0.10]
RATIOS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]            # hybrid rays: lam_v / lam_u
LAM_MAX, R_MIN, N_SCAN = 50.0, 0.01, 60              # budget search: scale r of each ray from R_MIN, log scan
GRID_CAP = (max(LAM_U), max(LAM_V))                 # primary search stays inside the specified knob grid; "wide" = up to LAM_MAX (D73)
W_COST = [0.01, 0.05, 0.2]
SECOND = ["const_plus", "sairh_uniform"]            # robustness models (a+) and (d)-uniform
STRAT = ["block", "refute", "hybrid"]
STRAT_COLOR = {"none": "#6b6b6b", "block": "#0072B2", "refute": "#E69F00", "hybrid": "#009E73"}
STRAT_LABEL = {"none": "no control", "block": "block only", "refute": "refute only", "hybrid": "hybrid"}
DT_DIV = {"sydneysiege": 2}                         # finer Euler step where dt vs dt/2 failed at the largest knob (D76)
CKPT = os.path.join(sair.CKPT_DIR, "phase7_stage1.jsonl")
OUT = {n: os.path.join(dp.RESULTS, f"phase7_{n}.csv") for n in ("grid", "strategies", "feedback", "cost")}


# ---------------------------------------------------------------- one run

_SETUP = {}


_VARIANT = {}                                      # tag -> (context key, parameter vector): refits / model variants (D76)


def setup(key, kind, tag=None):
    """Context, fitted parameters, gamma and T90 of one set and model (tag = a registered variant, e.g. a kappa-fixed refit)."""
    if tag is not None:
        if (key, kind, tag) not in _SETUP:
            ctx_key, x = _VARIANT[tag]
            sc = sair.get_context(ctx_key)
            _SETUP[(key, kind, tag)] = (sc, np.asarray(x, dtype=float), dict(zip(sair.NAMES[kind], x))["gamma"],
                                        float(np.interp(0.9, sc["C_I"] / sc["fin_I"], sc["grid"])))
        return _SETUP[(key, kind, tag)]
    if (key, kind) not in _SETUP:
        table = pd.read_csv(os.path.join(dp.RESULTS, "phase5_params.csv"))
        x = sair.params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind)
        sc = sair.get_context(key)
        t90 = float(np.interp(0.9, sc["C_I"] / sc["fin_I"], sc["grid"]))
        _SETUP[(key, kind)] = (sc, x, dict(zip(sair.NAMES[kind], x))["gamma"], t90)
    return _SETUP[(key, kind)]


def knobs(key, kind, lu, lv, tag=None):
    _, _, gamma, t90 = setup(key, kind, tag)
    return lu * gamma, lv / t90


def run_point(key, kind, lu, lv, mode="fb", d1_path=None, record=False, halve=1, div=None, tag=None):
    """One SAIR-C run. mode: 'fb' = network feedback as fitted, 'rho1' = rho frozen at 1, 'dfix' = d1(t) frozen to d1_path.
    dt = the Phase 5 step (u, v in the largest-rate rule) divided by DT_DIV[set] (D76: Sydney runs at dt/2 because its
    lam_u = 5 point is a sharp tipping point); div = 1 gives exactly the Phase 5 step; halve = extra halvings for the
    dt-convergence check. Returns the outcome dict (and the DataFrame if record)."""
    sc, x, _, _ = setup(key, kind, tag)
    u, v = knobs(key, kind, lu, lv, tag)
    dt = sair.step_dt(sc, kind, x, u, v) / (DT_DIV.get(key, 1) if div is None else div) / halve
    ex = {}
    res = sair.simulate(sc, kind, x, record=record, dt=dt, u=u, v=v, extras=ex, rho_one=(mode == "rho1"),
                        d1_path=d1_path if mode == "dfix" else None)
    out = {"lam_u": float(lu), "lam_v": float(lv), "u": u, "v": v, "D": float(ex["D"]), "peak_I": float(ex["peak_I"]),
           "C_I_final": float(ex["C_I"]), "C_A_final": float(ex["C_A"]), "B_final": float(ex["B"]),
           "E_u": float(ex["B"]), "E_v": float(ex["E_v"]), "E_total": float(ex["B"] + ex["E_v"]),
           "S_final": float(ex["S"]), "d1_upd": ex["d1_upd"]}
    return (out, res) if record else out


def strategy_of(lu, lv):
    return "none" if lu == 0 and lv == 0 else "block" if lv == 0 else "refute" if lu == 0 else "hybrid"


def clean(o):
    return {k: v for k, v in o.items() if k != "d1_upd"}


# ---------------------------------------------------------------- stage 1 jobs

def grid_job(args):
    """The full (lam_u, lam_v) grid of one set / model / mode."""
    key, kind, mode = args
    d1_path = run_point(key, kind, 0, 0)["d1_upd"] if mode == "dfix" else None
    return [{**clean(run_point(key, kind, lu, lv, mode, d1_path)), "set": key, "model": kind, "mode": mode}
            for lu in LAM_U for lv in LAM_V]


def ray_job(args):
    """Equal-effort search along one ray (lam_u, lam_v) = r * (a, b): scan r on a log grid, then bisect every crossing of
    each effort budget (the effort is not monotone in r for blocking, so there can be two roots; D is evaluated at all)."""
    key, kind, name, a, b, cap_u, cap_v = args[:7]
    tag = None
    if len(args) > 7:                                   # variant: (tag, context key, parameters)
        tag = args[7]
        _VARIANT[tag] = (args[8], args[9])
    rs = np.geomspace(R_MIN, min(cap_u / a if a else np.inf, cap_v / b if b else np.inf), N_SCAN)
    scan = [run_point(key, kind, r * a, r * b, tag=tag) for r in rs]
    e = np.array([s["E_total"] for s in scan])
    rows = [{"set": key, "model": kind, "ray": name, "kind": "scan", "budget": np.nan, "root": -1, "r": float(r),
             **clean(s)} for r, s in zip(rs, scan)]
    for bud in BUDGETS:
        root = 0
        for i in range(len(rs) - 1):
            f0, f1 = e[i] - bud, e[i + 1] - bud
            if f0 * f1 > 0:
                continue
            lo, hi = np.log(rs[i]), np.log(rs[i + 1])
            best = scan[i] if abs(f0) <= abs(f1) else scan[i + 1]
            r_best = rs[i] if abs(f0) <= abs(f1) else rs[i + 1]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                s = run_point(key, kind, np.exp(mid) * a, np.exp(mid) * b, tag=tag)
                if abs(s["E_total"] - bud) < abs(best["E_total"] - bud):
                    best, r_best = s, float(np.exp(mid))
                if abs(s["E_total"] - bud) < 1e-6:
                    break
                if (s["E_total"] - bud) * f0 > 0:
                    lo = mid
                else:
                    hi = mid
            rows.append({"set": key, "model": kind, "ray": name, "kind": "root", "budget": bud, "root": root,
                         "r": r_best, **clean(best)})
            root += 1
    return rows


def run_jobs(pool, jobs, fn_name, label, ckpt=None):
    """Run (id, args) jobs in the pool; every finished job is appended to the checkpoint (restart-safe) with ETA logging."""
    os.makedirs(sair.CKPT_DIR, exist_ok=True)
    done = {}
    ckpt = ckpt or CKPT
    if os.path.exists(ckpt):
        for line in open(ckpt):
            r = json.loads(line)
            done[r["id"]] = r["rows"]
    todo = [(i, a) for i, a in jobs if i not in done]
    print(f"[{label}] {len(jobs)} jobs: {len(jobs) - len(todo)} from checkpoint, {len(todo)} to run", flush=True)
    t0 = time.time()
    futures = {pool.submit(fn_name, a): i for i, a in todo}
    for n, f in enumerate(as_completed(futures), 1):
        rows = f.result()
        done[futures[f]] = rows
        with open(ckpt, "a") as fh:
            fh.write(json.dumps({"id": futures[f], "rows": rows}) + "\n")
        el = time.time() - t0
        print(f"[{label}] {n}/{len(todo)} done  {futures[f]}  elapsed {el:.0f} s  ETA {el / n * (len(todo) - n):.0f} s",
              flush=True)
    return done


def dispatch(job):
    kind, args = job
    return grid_job(args) if kind == "grid" else ray_job(args)


def stage1(pool, main):
    jobs = []
    for key in SETS:
        for mode in ("fb", "rho1", "dfix"):
            jobs.append((f"grid|{key}|{main}|{mode}", ("grid", (key, main, mode))))
        for kind in SECOND:
            jobs.append((f"grid|{key}|{kind}|fb", ("grid", (key, kind, "fb"))))
        for kind in [main] + SECOND:
            for name, a, b in ray_dirs():
                jobs.append((f"ray|{key}|{kind}|{name}", ("ray", (key, kind, name, a, b) + GRID_CAP)))
                if kind == main:                                 # sensitivity: knobs beyond the grid, main model only
                    jobs.append((f"rayw|{key}|{kind}|{name}", ("ray", (key, kind, name, a, b, LAM_MAX, LAM_MAX))))
    done = run_jobs(pool, [(i, j) for i, j in jobs], dispatch, "phase 7 stage 1")
    return {i: done[i] for i, _ in jobs}


def ray_dirs():
    return [("block", 1.0, 0.0), ("refute", 0.0, 1.0)] + [(f"hybrid_q{q:g}", 1.0, q) for q in RATIOS]


# ---------------------------------------------------------------- stage 2: tables

def grid_tables(done, main):
    """phase7_grid.csv: the feedback grid of every model with D/D0, cost J and the Pareto flag."""
    frames = []
    for i, rows in done.items():
        if i.startswith("grid|") and i.endswith("|fb"):
            frames.append(pd.DataFrame(rows))
    g = pd.concat(frames, ignore_index=True)
    g["strategy"] = [strategy_of(a, b) for a, b in zip(g["lam_u"], g["lam_v"])]
    g["D0"] = g.groupby(["set", "model"])["D"].transform("first")
    g["D_over_D0"] = g["D"] / g["D0"]
    g["peak_over_peak0"] = g["peak_I"] / g.groupby(["set", "model"])["peak_I"].transform("first")
    for w in W_COST:
        g[f"J_w{w:g}"] = g["D_over_D0"] + w * (g["lam_u"] ** 2 + g["lam_v"] ** 2)
    g["pareto"] = False
    for _, sub in g.groupby(["set", "model"]):
        best = np.inf
        for idx in sub.sort_values(["E_total", "D_over_D0"]).index:
            if g.loc[idx, "D_over_D0"] < best - 1e-12:
                g.loc[idx, "pareto"] = True
                best = g.loc[idx, "D_over_D0"]
    cols = ["model", "set", "lam_u", "lam_v", "strategy", "u", "v", "D", "D0", "D_over_D0", "peak_I", "peak_over_peak0",
            "C_I_final", "C_A_final", "B_final", "E_u", "E_v", "E_total", "S_final"] + [f"J_w{w:g}" for w in W_COST] + ["pareto"]
    return g[cols]


def cost_table(g):
    rows = []
    for (model, key), sub in g.groupby(["model", "set"], sort=False):
        for w in W_COST:
            best = sub.loc[sub[f"J_w{w:g}"].idxmin()]
            rows.append({"model": model, "set": key, "w": w, "lam_u": best["lam_u"], "lam_v": best["lam_v"],
                         "strategy": best["strategy"], "J": best[f"J_w{w:g}"], "D_over_D0": best["D_over_D0"],
                         "E_total": best["E_total"], "hybrid_is_cheapest": best["strategy"] == "hybrid",
                         **{f"best_{s}_J": sub[sub["strategy"] == s][f"J_w{w:g}"].min() for s in ["none"] + STRAT}})
    return pd.DataFrame(rows)


def strategy_table(done):
    """Best point of each strategy at equal total effort (E = budget): the minimum D over all roots of all rays.
    range 'grid' = knobs inside the specified grid (primary); 'wide' = knobs up to LAM_MAX (main model, sensitivity)."""
    rows = []
    for model in sorted({i.split("|")[2] for i in done if i.startswith("ray|")}):
        for key in SETS:
            base = run_point(key, model, 0, 0)
            for rng, prefix in (("grid", "ray"), ("wide", "rayw")):
                ids = [i for i in done if i.startswith(f"{prefix}|{key}|{model}|")]
                if not ids:
                    continue
                ray_rows = pd.DataFrame([r for i in ids for r in done[i]])
                for bud in BUDGETS:
                    for strat in STRAT:
                        pick = ray_rows["ray"].str.startswith("hybrid") if strat == "hybrid" else (ray_rows["ray"] == strat)
                        sel = ray_rows[(ray_rows["kind"] == "root") & (ray_rows["budget"] == bud) & pick]
                        if sel.empty:
                            rows.append({"model": model, "set": key, "range": rng, "budget": bud, "strategy": strat,
                                         "reachable": False, "max_E_total_reached": ray_rows[(ray_rows["kind"] == "scan") & pick]["E_total"].max()})
                            continue
                        best = sel.loc[sel["D"].idxmin()]
                        rows.append({"model": model, "set": key, "range": rng, "budget": bud, "strategy": strat, "reachable": True,
                                     "ray": best["ray"], "lam_u": best["lam_u"], "lam_v": best["lam_v"], "u": best["u"],
                                     "v": best["v"], "E_total": best["E_total"], "E_u": best["E_u"], "E_v": best["E_v"],
                                     "D": best["D"], "D0": base["D"], "D_over_D0": best["D"] / base["D"],
                                     "peak_I": best["peak_I"], "peak_over_peak0": best["peak_I"] / base["peak_I"],
                                     "C_I_final": best["C_I_final"], "C_A_final": best["C_A_final"],
                                     "B_final": best["B_final"], "n_roots": len(sel)})
    return pd.DataFrame(rows)


def feedback_tables(done, main, strat):
    """phase7_feedback.csv: (i) the whole grid with rho frozen at 1 and with d1(t) frozen to the no-control path;
    (ii) the best equal-effort points of the main model re-run with the feedback frozen at the SAME knobs."""
    rows = []
    fb = pd.DataFrame(done_rows(done, "grid", main, "fb"))
    for mode in ("rho1", "dfix"):
        fz = pd.DataFrame(done_rows(done, "grid", main, mode))
        for key in SETS:
            a, b = fb[fb["set"] == key].reset_index(drop=True), fz[fz["set"] == key].reset_index(drop=True)
            d0a, d0b = a["D"].iloc[0], b["D"].iloc[0]
            for k in range(len(a)):
                ea, eb = 1 - a["D"][k] / d0a, 1 - b["D"][k] / d0b
                rows.append({"row_type": "grid", "range": "grid", "set": key, "frozen": mode, "budget": np.nan, "strategy": strategy_of(a["lam_u"][k], a["lam_v"][k]),
                             "lam_u": a["lam_u"][k], "lam_v": a["lam_v"][k], "D_over_D0_feedback": a["D"][k] / d0a,
                             "D_over_D0_frozen": b["D"][k] / d0b, "effect_feedback": ea, "effect_frozen": eb,
                             "feedback_share": 1 - eb / ea if ea > 1e-9 else np.nan,
                             "C_A_final_feedback": a["C_A_final"][k], "C_A_final_frozen": b["C_A_final"][k]})
    for key in SETS:
        base = run_point(key, main, 0, 0)
        d0 = {"rho1": run_point(key, main, 0, 0, "rho1")["D"], "dfix": base["D"]}
        for _, r in strat[(strat["model"] == main) & (strat["set"] == key) & strat["reachable"]
                          & strat["budget"].isin(MAIN_BUDGETS)].iterrows():
            if r["strategy"] == "none":
                continue
            ea = 1 - r["D"] / base["D"]
            for mode in ("rho1", "dfix"):
                z = run_point(key, main, r["lam_u"], r["lam_v"], mode, base["d1_upd"])
                eb = 1 - z["D"] / d0[mode]
                rows.append({"row_type": "best_point", "range": r["range"], "set": key, "frozen": mode, "budget": r["budget"], "strategy": r["strategy"],
                             "lam_u": r["lam_u"], "lam_v": r["lam_v"], "D_over_D0_feedback": r["D"] / base["D"],
                             "D_over_D0_frozen": z["D"] / d0[mode], "effect_feedback": ea, "effect_frozen": eb,
                             "feedback_share": 1 - eb / ea if ea > 1e-9 else np.nan,
                             "C_A_final_feedback": r["C_A_final"], "C_A_final_frozen": z["C_A_final"]})
    return pd.DataFrame(rows)


def done_rows(done, kind, model, mode):
    return [r for i, rs in done.items() if i.split("|")[0] == kind and i.endswith(f"|{model}|{mode}") for r in rs]


# ---------------------------------------------------------------- plots

def plot4(strat, main, budget=0.05):
    """I(t) of the best equal-effort point of each strategy at E = budget (grid range, re-run with record).
    E = 10% (Plot 4b) adds a C_A(t) row so the CH tipping (corrections taking over) is visible, and checks that each
    re-run's D/D0 equals the phase7_strategies.csv value shown in the legend."""
    ca_row = budget != 0.05
    fig, axes = plt.subplots(2 if ca_row else 1, 3, figsize=(15, 8.4 if ca_row else 4.2), squeeze=False)
    worst = 0.0
    for col, key in enumerate(SETS):
        ax = axes[0, col]
        s = strat[(strat["model"] == main) & (strat["set"] == key) & (strat["budget"] == budget) & (strat["range"] == "grid")].set_index("strategy")
        d0 = None
        for name in ["none"] + STRAT:
            lu, lv = (0, 0) if name == "none" else (s.loc[name, "lam_u"], s.loc[name, "lam_v"]) if s.loc[name, "reachable"] else (None, None)
            if lu is None:
                continue
            o, res = run_point(key, main, lu, lv, record=True)
            if name == "none":
                d0 = o["D"]
            elif ca_row:
                diff = abs(o["D"] / d0 - s.loc[name, "D_over_D0"])
                worst = max(worst, diff)
                print(f"{key:22s} {name:7s} lam_u={lu:.4g} lam_v={lv:.4g}  D/D0 re-run {o['D'] / d0:.6f}  csv {s.loc[name, 'D_over_D0']:.6f}  "
                      f"diff {diff:.1e}  C_A final {o['C_A_final']:.3f}")
            lab = STRAT_LABEL[name] + ("" if name == "none" else
                                       f"  (λu={lu:.2g}, λv={lv:.2g}; D/D0={s.loc[name, 'D_over_D0']:.{3 if ca_row else 2}f})")
            ax.plot(res["t"], 100 * res["I"], color=STRAT_COLOR[name], lw=2, label=lab, ls="--" if name == "none" else "-")
            if ca_row:
                axes[1, col].plot(res["t"], 100 * res["C_A"], color=STRAT_COLOR[name], lw=2, label=STRAT_LABEL[name],
                                  ls="--" if name == "none" else "-")
        ax.set_title(sair.set_title(key))
        ax.set_xlabel("time (h)")
        ax.set_ylabel("I (% of N)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        if ca_row:
            a2 = axes[1, col]
            a2.axhline(50, color="black", lw=0.8, ls=":", label="C_A = 50% of N (tipping threshold)")
            a2.set_title(sair.set_title(key) + " - cumulative corrections C_A(t)")
            a2.set_xlabel("time (h)")
            a2.set_ylabel("C_A (% of N)")
            a2.legend(fontsize=7)
            a2.grid(alpha=0.25)
    if ca_row:
        assert worst < 1e-9, f"Plot 4b: re-run D/D0 differs from phase7_strategies.csv ({worst:.1e})"
        print(f"Plot 4b: legend D/D0 == phase7_strategies.csv, re-run max diff {worst:.1e}")
        fig.suptitle(f"Plot 4b - rumour prevalence I(t) under the four strategies at equal total effort E = {100 * budget:g}% of N")
        name = f"phase7_plot4_strategies_{100 * budget:g}pct.png"
    else:
        fig.suptitle(f"Plot 4 - rumour prevalence I(t) under the four strategies at equal total effort E = 5% of N ({sair.MODEL_LABEL[main]})")
        name = "phase7_plot4_strategies.png"
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, name), dpi=130)
    plt.close(fig)


def plot5(g, cost, main):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, key in zip(axes, SETS):
        sub = g[(g["model"] == main) & (g["set"] == key)]
        m = sub.pivot(index="lam_u", columns="lam_v", values="D_over_D0").loc[LAM_U, LAM_V]
        im = ax.imshow(m.to_numpy(), origin="lower", cmap="YlGnBu_r", norm=LogNorm(vmin=max(m.to_numpy().min(), 1e-3), vmax=1.0))
        for i in range(len(LAM_U)):
            for j in range(len(LAM_V)):
                ax.text(j, i, f"{m.to_numpy()[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m.to_numpy()[i, j] < 0.1 else "black")
        c = cost[(cost["model"] == main) & (cost["set"] == key) & (cost["w"] == 0.05)].iloc[0]
        ax.plot(LAM_V.index(c["lam_v"]), LAM_U.index(c["lam_u"]), marker="*", ms=18, mfc="none", mec="#d6336c", mew=2,
                label=f"cheapest, w=0.05: λu={c['lam_u']:g}, λv={c['lam_v']:g}")
        ax.set_xticks(range(len(LAM_V)), [f"{v:g}" for v in LAM_V])
        ax.set_yticks(range(len(LAM_U)), [f"{u:g}" for u in LAM_U])
        ax.set_xlabel("λv  (campaign rate × T90)")
        ax.set_ylabel("λu  (blocking rate / γ)")
        ax.set_title(sair.set_title(key))
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        fig.colorbar(im, ax=ax, label="D / D0 (log scale)")
    fig.suptitle(f"Plot 5 - rumour exposure relative to no control over the knob grid ({sair.MODEL_LABEL[main]})")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase7_plot5_heatmap.png"), dpi=130)
    plt.close(fig)


def plot5b(g, strat, main):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, key in zip(axes, SETS):
        sub = g[(g["model"] == main) & (g["set"] == key)]
        ax.scatter(100 * sub["E_total"], sub["D_over_D0"], s=14, color="#b0b0b0", label="grid points", zorder=1)
        front = sub[sub["pareto"]].sort_values("E_total")
        ax.step(100 * front["E_total"], front["D_over_D0"], where="post", color="black", lw=1.2, label="grid Pareto front")
        for name in STRAT:
            s = strat[(strat["model"] == main) & (strat["set"] == key) & (strat["strategy"] == name) & (strat["range"] == "grid") & strat["reachable"]].sort_values("budget")
            ax.plot(100 * s["budget"], np.minimum.accumulate(s["D_over_D0"].to_numpy()), "-o", ms=4, lw=2, color=STRAT_COLOR[name],
                    label=f"{STRAT_LABEL[name]} (best at effort ≤ E)")
        ax.set_yscale("log")
        ax.set_xlim(0, 22)
        ax.set_xlabel("total effort E = E_u + E_v  (% of N)")
        ax.set_ylabel("D / D0  (rumour exposure vs no control)")
        ax.set_title(sair.set_title(key))
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
    fig.suptitle(f"Plot 5b - Pareto front: exposure vs effort ({sair.MODEL_LABEL[main]}); lines = running best over the equal-effort points")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase7_plot5b_pareto.png"), dpi=130)
    plt.close(fig)


def stage2(done, main):
    g = grid_tables(done, main)
    cost = cost_table(g)
    strat = strategy_table(done)
    fbk = feedback_tables(done, main, strat)
    g.to_csv(OUT["grid"], index=False)
    strat.to_csv(OUT["strategies"], index=False)
    fbk.to_csv(OUT["feedback"], index=False)
    cost.to_csv(OUT["cost"], index=False)
    plot4(strat, main)
    plot4(strat, main, 0.10)
    plot5(g, cost, main)
    plot5b(g, strat, main)
    report(g, cost, strat, fbk, main)


def report(g, cost, strat, fbk, main):
    pd.set_option("display.width", 220)
    for model, rng in [(main, "grid"), (main, "wide")] + [(m, "grid") for m in SECOND]:
        print(f"\n=== {sair.MODEL_LABEL[model]} [{rng} knob range]: best point of each strategy at equal total effort (D/D0; lam_u, lam_v) ===")
        s = strat[(strat["model"] == model) & (strat["range"] == rng) & strat["budget"].isin(MAIN_BUDGETS)]
        for key in SETS:
            for bud in MAIN_BUDGETS:
                t = s[(s["set"] == key) & (s["budget"] == bud)].set_index("strategy")
                cells = [f"{n}: " + (f"{t.loc[n, 'D_over_D0']:.3f} ({t.loc[n, 'lam_u']:.3g}, {t.loc[n, 'lam_v']:.3g})" if t.loc[n, "reachable"]
                                     else f"unreachable (max {t.loc[n, 'max_E_total_reached']:.3f})") for n in STRAT]
                print(f"{key:22s} E={100 * bud:4.1f}%  " + "   ".join(cells))
        if rng == "wide":
            continue
        print(f"--- cheapest weighted cost J = D/D0 + w(lam_u^2 + lam_v^2) ({sair.MODEL_LABEL[model]})")
        print(cost[cost["model"] == model][["set", "w", "lam_u", "lam_v", "strategy", "J", "D_over_D0", "E_total"]].round(4).to_string(index=False))
    f = fbk[fbk["row_type"] == "best_point"]
    print("\n=== feedback share of each strategy's effect (main model, best equal-effort points; share = 1 - effect_frozen / effect_feedback) ===")
    print(f.pivot_table(index=["range", "set", "budget", "strategy"], columns="frozen", values=["effect_feedback", "effect_frozen", "feedback_share"]).round(3).to_string())


# ---------------------------------------------------------------- tipping robustness (D76)

KAPPAS = [5, 12.5, 25, 50]                           # kappa fixed, the other parameters refit
TIP_KEY, TIP_BUDGETS, N_REFIT = "charliehebdo", [0.05, 0.10], 10
TIP_CKPT = "phase7_tipping_fits"
TIP_RAY_CKPT = os.path.join(sair.CKPT_DIR, "phase7_tipping_rays.jsonl")


def tipping_fits(pool, main):
    """Refit the main model on CH with kappa fixed at each KAPPAS value, and with the fallback P1 = 0.5 (kappa free):
    one warm start from the main fit plus N_REFIT random starts each (checkpointed). Returns tag -> (context key, x, k)."""
    table = pd.read_csv(os.path.join(dp.RESULTS, "phase5_params.csv"))
    x_main = sair.params_of(table[(table["set"] == TIP_KEY) & (table["model"] == main)].iloc[0], main)
    warm = dict(zip(sair.NAMES[main], x_main))
    variants = [(f"kappa{k:g}", TIP_KEY, {"kappa": float(k)}) for k in KAPPAS] + [("fb_half", (TIP_KEY, "fallback_half"), None)]
    jobs = []
    for tag, ctx_key, fixed in variants:
        free = sair.free_names(main, fixed)
        starts = [sair.to_fit(free, np.clip([warm[n] for n in free], *zip(*[sair.RANGE[n] for n in free])))]
        starts += sair.random_starts(free, N_REFIT, dp.SEED)
        jobs += [{"key": ctx_key, "kind": main, "u0": u0, "tag": tag, "fixed": fixed, "maxfev": 5000} for u0 in starts]
    res = sair.run_jobs(pool, jobs, "phase 7 tipping refits", ckpt=TIP_CKPT)
    out = {}
    for tag, ctx_key, fixed in variants:
        best, n_conv = sair.best_of(res[tag])
        out[tag] = (ctx_key, np.array(best["x"]), len(sair.free_names(main, fixed)), n_conv, len(res[tag]))
    return out, x_main


def tipping_rays(pool, main, variants):
    jobs = []
    for tag, (ctx_key, x, *_ ) in variants.items():
        for name, a, b in ray_dirs():
            jobs.append((f"tray|{tag}|{name}", ("ray", (TIP_KEY, main, name, a, b) + GRID_CAP + (tag, ctx_key, [float(v) for v in x]))))
    return run_jobs(pool, jobs, dispatch, "phase 7 tipping rays", ckpt=TIP_RAY_CKPT)


def tipping_tables(main, variants, x_main, done):
    """phase7_kappa_fits.csv (fit quality per variant) and phase7_tipping.csv (CH strategies at E = 5 / 10% per variant)."""
    names = sair.NAMES[main]
    main_ctx = sair.get_context(TIP_KEY)
    fits, rows = [], []
    allv = {"main": (TIP_KEY, x_main, len(names), np.nan, np.nan), **variants}
    for tag, (ctx_key, x, k, n_conv, n_st) in allv.items():
        sc = sair.get_context(ctx_key)
        m = sair.metrics(*sair.residuals(sc, main, x), k)
        fits.append({"variant": tag, "k_free": k, **dict(zip(names, x)), "SSE": m["SSE"], "AIC": m["AIC"], "BIC": m["BIC"],
                     "starts_converged": n_conv, "starts": n_st, "bound_flags": sair.bound_flags(
                         [n for n in names if not (tag.startswith("kappa") and n == "kappa")],
                         [v for n, v in zip(names, x) if not (tag.startswith("kappa") and n == "kappa")])})
    fits = pd.DataFrame(fits)
    fits["dAIC_vs_main"] = fits["AIC"] - fits.loc[fits["variant"] == "main", "AIC"].iloc[0]
    strat = pd.read_csv(OUT["strategies"])
    strat = strat[(strat["model"] == main) & (strat["set"] == TIP_KEY) & (strat["range"] == "grid")]
    for tag, (ctx_key, x, *_) in allv.items():
        if tag != "main":
            _VARIANT[tag] = (ctx_key, x)
        t = None if tag == "main" else tag
        base = run_point(TIP_KEY, main, 0, 0, tag=t)
        for bud in TIP_BUDGETS:
            for name in STRAT:
                if tag == "main":
                    r = strat[(strat["budget"] == bud) & (strat["strategy"] == name)]
                    rows.append({"variant": tag, "budget": bud, "strategy": name, **({"reachable": False} if not r["reachable"].iloc[0] else {
                        "reachable": True, "lam_u": r["lam_u"].iloc[0], "lam_v": r["lam_v"].iloc[0], "D_over_D0": r["D_over_D0"].iloc[0],
                        "C_A_final": r["C_A_final"].iloc[0], "C_I_final": r["C_I_final"].iloc[0]}), "C_A_final_no_control": base["C_A_final"]})
                    continue
                ray_rows = pd.DataFrame([x for i, rs in done.items() if i.startswith(f"tray|{tag}|") for x in rs])
                pick = ray_rows["ray"].str.startswith("hybrid") if name == "hybrid" else (ray_rows["ray"] == name)
                sel = ray_rows[(ray_rows["kind"] == "root") & (ray_rows["budget"] == bud) & pick]
                if sel.empty:
                    rows.append({"variant": tag, "budget": bud, "strategy": name, "reachable": False, "C_A_final_no_control": base["C_A_final"]})
                    continue
                b = sel.loc[sel["D"].idxmin()]
                rows.append({"variant": tag, "budget": bud, "strategy": name, "reachable": True, "lam_u": b["lam_u"], "lam_v": b["lam_v"],
                             "D_over_D0": b["D"] / base["D"], "C_A_final": b["C_A_final"], "C_I_final": b["C_I_final"],
                             "C_A_final_no_control": base["C_A_final"]})
    tip = pd.DataFrame(rows)
    tip["tipped"] = (tip["D_over_D0"] < 0.5) & (tip["C_A_final"] > 0.5)           # exposure halved AND the A side takes over
    return fits, tip


def tipping_report(fits, tip):
    pd.set_option("display.width", 220)
    print("=== CH refits for the tipping robustness (D76) ===")
    print(fits[["variant", "k_free", "beta", "alpha", "gamma", "eps", "kappa", "theta", "eta_I", "eta_A", "SSE", "AIC",
                "dAIC_vs_main", "starts_converged", "bound_flags"]].round(4).to_string(index=False))
    print("=== CH strategies at equal effort per variant: D/D0 (lam_u, lam_v), final C_A; tipped = D/D0 < 0.5 and C_A > 0.5 ===")
    for bud in TIP_BUDGETS:
        print(f"--- E = {100 * bud:g}% of N")
        for v in tip["variant"].unique():
            t = tip[(tip["variant"] == v) & (tip["budget"] == bud)].set_index("strategy")
            cells = [f"{n}: " + (f"{t.loc[n, 'D_over_D0']:.3f} ({t.loc[n, 'lam_u']:.3g}, {t.loc[n, 'lam_v']:.3g}) C_A {t.loc[n, 'C_A_final']:.2f}"
                                 + (" TIPPED" if t.loc[n, "tipped"] else "") if t.loc[n, "reachable"] else "unreachable") for n in STRAT]
            print(f"{v:14s} (no-control C_A {t['C_A_final_no_control'].iloc[0]:.3f})  " + "   ".join(cells))


# ---------------------------------------------------------------- check

def ref_hours(key):
    return np.arange(0, sair.get_context(key)["end_h"] + 1e-9, 1.0)


def check(main):
    print("\n--- Phase 7 check ---")
    # worked step of docs/extension.md §1 (Eq. 26 form, hand-checkable)
    S, I, A, R, B, *_ = sair.euler_step_c(0.90, 0.05, 0.05, 0.0, 0.0, 0.6, 0.4, 1 - 0.6, 1 - 0.4, 1.0, 0.0, 0.0, 0.1, 0.02)
    assert np.allclose([S, I, A, B, R], [0.837, 0.052, 0.056, 0.005, 0.050], atol=5e-4), (S, I, A, B, R)
    print(f"unit test (docs worked step): S={S:.3f} I={I:.3f} A={A:.3f} B={B:.3f} R={R:.3f} sum={S + I + A + B + R:.6f}  OK")
    g = pd.read_csv(OUT["grid"])
    strat = pd.read_csv(OUT["strategies"])
    viol = []
    for key in SETS:
        sc, x, gamma, t90 = setup(key, main)
        # 1. lam = 0 reproduces the Phase 5 main-model run exactly (saved curves written by Phase 5, and the Phase 5 step)
        _, r0 = run_point(key, main, 0, 0, record=True, div=1)          # the Phase 5 step: exact reproduction
        _, r0f = run_point(key, main, 0, 0, record=True)                # the step used in Phase 7 (finer on Sydney)
        dfine = float(np.abs(np.interp(ref_t := ref_hours(key), r0["t"], r0["I"]) - np.interp(ref_t, r0f["t"], r0f["I"])).max())
        ref = pd.read_csv(os.path.join(dp.PROCESSED, f"model_curves_{key}.csv"))
        cols = ["S", "I", "A", "R", "C_I", "C_A", "d1", "d2", "t"]
        d_ref = float(np.abs(r0[cols].to_numpy() - ref[cols].to_numpy()).max())
        assert len(r0) == len(ref) and d_ref <= 1e-12, f"{key}: lam = 0 differs from the Phase 5 run ({d_ref:.1e})"
        assert float(np.abs(r0["B"]).max()) == 0.0
        # 2. invariants on every grid point (record mode) and on the largest knobs
        worst_sum, lo, hi = 0.0, 1.0, 0.0
        for lu in LAM_U:
            for lv in LAM_V:
                _, r = run_point(key, main, lu, lv, record=True)
                st = r[["S", "I", "A", "R", "B"]].to_numpy()
                worst_sum = max(worst_sum, float(np.abs(st.sum(axis=1) - 1).max()))
                lo, hi = min(lo, st.min()), max(hi, st.max())
        assert worst_sum <= sair.INV_TOL and lo >= -sair.INV_TOL and hi <= 1 + sair.INV_TOL, (key, worst_sum, lo, hi)
        # 3. dt-convergence at the largest knob values
        hours = np.arange(0, sc["end_h"] + 1e-9, 1.0)
        worst_dt, worst_D, over = 0.0, 0.0, []
        for lu, lv in [(5, 0), (0, 2), (5, 2), (2, 1)]:
            oa, ra = run_point(key, main, lu, lv, record=True)
            ob, rb = run_point(key, main, lu, lv, record=True, halve=2)
            d = max(float(np.abs(np.interp(hours, ra["t"], ra[c]) - np.interp(hours, rb["t"], rb[c])).max())
                    for c in ["S", "I", "A", "R", "B", "C_I", "C_A"])
            worst_dt = max(worst_dt, d)
            worst_D = max(worst_D, abs(oa["D"] - ob["D"]) / ob["D"], abs(oa["peak_I"] - ob["peak_I"]) / ob["peak_I"])
            if d >= 1e-3:
                over.append((lu, lv, d))
        # states: Phase 5 criterion 1e-3, exceedances are reported (D73); outcomes D and peak: relative 1e-3, asserted
        assert worst_dt < 5e-3 and worst_D < 1e-3, f"{key}: dt-convergence failed (states {worst_dt:.2e}, outcomes {worst_D:.2e})"
        for lu, lv, d in over:
            print(f"{key:22s} WARNING  dt vs dt/2 at (lam_u, lam_v) = ({lu}, {lv}): max state difference {d:.2e} >= 1e-3 (outcomes within {worst_D:.1e})")
        # 4. numba kernel == Python reference with the knobs on
        worst_nb = 0.0
        for lu, lv in [(1, 0), (0, 1), (2, 1), (5, 2)]:
            u, v = knobs(key, main, lu, lv)
            ea, eb = {}, {}
            a = sair.simulate(sc, main, x, record=True, u=u, v=v, extras=ea)
            b = sair.simulate_python(sc, main, x, record=True, u=u, v=v, extras=eb)
            c = ["t", "S", "I", "A", "R", "C_I", "C_A", "B", "d1"]
            worst_nb = max(worst_nb, float(np.abs(a[c].to_numpy() - b[c].to_numpy()).max()),
                           max(abs(ea[k] - eb[k]) for k in ("D", "E_v", "peak_I", "B")))
        assert worst_nb <= 1e-12, f"{key}: numba differs from the Python loop with knobs ({worst_nb:.1e})"
        # 5. effort identity, and D non-increasing along each grid line (reported, not forced)
        sub = g[(g["model"] == main) & (g["set"] == key)]
        assert np.allclose(sub["E_u"], sub["B_final"], atol=1e-12) and abs(sub[(sub.lam_u == 0) & (sub.lam_v == 0)]["D_over_D0"].iloc[0] - 1) < 1e-12
        for model in [main] + SECOND:
            gm = g[(g["model"] == model) & (g["set"] == key)]
            for fixed, along in (("lam_v", "lam_u"), ("lam_u", "lam_v")):
                for val, line in gm.groupby(fixed):
                    d = line.sort_values(along)["D_over_D0"].to_numpy()
                    bad = np.where(np.diff(d) > 1e-9)[0]
                    for k in bad:
                        viol.append((model, key, f"{fixed}={val:g}", f"{along}: {sorted(line[along])[k]:g}->{sorted(line[along])[k + 1]:g}", float(d[k + 1] - d[k])))
        print(f"{key:22s} OK  lam=0 == Phase 5 run (max diff {d_ref:.1e}; Phase 7 step vs Phase 5 step {dfine:.1e})  max|sum-1|={worst_sum:.1e}  states in [{lo:.1e}, {hi:.6f}]  "
              f"dt vs dt/2 <= {worst_dt:.1e} (D, peak rel. {worst_D:.1e})  numba == Python ({worst_nb:.1e})")
    r = strat[strat["reachable"] == True]
    assert (np.abs(r["E_total"] - r["budget"]) < 1e-5).all(), "equal-effort points are not at the budget"
    rg = r[r["range"] == "grid"]
    assert (rg["lam_u"] <= max(LAM_U) + 1e-9).all() and (rg["lam_v"] <= max(LAM_V) + 1e-9).all(), "grid-range search left the grid"
    worst_w = 0.0                                                       # dt-convergence of the wide-range (sensitivity) points
    for _, w in r[(r["range"] == "wide") & r["budget"].isin(MAIN_BUDGETS)].iterrows():
        a, b = run_point(w["set"], w["model"], w["lam_u"], w["lam_v"]), run_point(w["set"], w["model"], w["lam_u"], w["lam_v"], halve=2)
        worst_w = max(worst_w, abs(a["D"] - b["D"]) / b["D"], abs(a["peak_I"] - b["peak_I"]) / b["peak_I"])
    print(f"wide-range sensitivity points: D and peak, dt vs dt/2 relative difference <= {worst_w:.1e}")
    print(f"equal-effort points: {len(r)} reachable, max |E - budget| = {np.abs(r['E_total'] - r['budget']).max():.1e}; "
          f"{int((strat['reachable'] == False).sum())} (model, set, budget, strategy) cells unreachable")
    print(f"D non-increasing along grid lines: {len(viol)} violation(s)" + (" (listed: model, set, line, step, increase in D/D0)" if viol else ""))
    for v in viol:
        print("   ", *v)
    print("Phase 7 checks passed.")


if __name__ == "__main__":
    main, _ = sair.load_main()
    if "--tipping" in sys.argv:
        get = sair.get_context(TIP_KEY)
        with ProcessPoolExecutor(max_workers=sair.N_WORKERS) as pool:
            variants, x_main = tipping_fits(pool, main)
            for tag, (ctx_key, x, *_) in variants.items():
                _VARIANT[tag] = (ctx_key, x)
            variants["fb_half_fixed"] = ((TIP_KEY, "fallback_half"), x_main, len(sair.NAMES[main]), np.nan, np.nan)   # fallback only, no refit
            done = tipping_rays(pool, main, variants)
        fits, tip = tipping_tables(main, variants, x_main, done)
        fits.to_csv(os.path.join(dp.RESULTS, "phase7_kappa_fits.csv"), index=False)
        tip.to_csv(os.path.join(dp.RESULTS, "phase7_tipping.csv"), index=False)
        tipping_report(fits, tip)
        sys.exit(0)
    if "--plot4-10pct" in sys.argv:                     # Plot 4b only, from the saved equal-effort points (no re-search)
        plot4(pd.read_csv(OUT["strategies"]), main, 0.10)
        sys.exit(0)
    if "--check-only" not in sys.argv:
        stages = [int(sys.argv[sys.argv.index("--stage") + 1])] if "--stage" in sys.argv else [1, 2]
        if "--fresh" in sys.argv and 1 in stages and os.path.exists(CKPT):
            os.remove(CKPT)
        for key in SETS:
            sair.get_context(key)
        with ProcessPoolExecutor(max_workers=sair.N_WORKERS) as pool:
            done = stage1(pool, main)
        if 2 in stages:
            stage2(done, main)
    if "--check" in sys.argv or "--check-only" in sys.argv:
        check(main)
