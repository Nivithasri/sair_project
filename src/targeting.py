"""Phase 8 - centrality-targeted intervention (docs/extension.md §2, D77-D82).

Part A (PHEME, main model (d+)-uniform with the fitted parameters of each set): pick k accounts by degree, PageRank,
betweenness (k = 500 sampling) or at random (15 draws), all on the FOLLOW graph, and
  block   the chosen accounts (pool = accounts that enter I in the data). Channel (i): they leave the n_I counts of their
          followers from the moment they would enter I; channel (ii): a supporting source tweet they authored loses its injection.
  recruit the chosen accounts (pool = accounts that never enter I and authored no source tweet) as correction broadcasters
          from t = 0: they count in n_A for their followers and inject s * theta * w_i / sum(w) into A (s = recruit scale, D81).
  hybrid  ceil(k/2) blocked + floor(k/2) recruited (top of each list).
Each channel is also run alone. Effort = % of N moved (blocked accounts + injected A mass), compared at equal effort (D81).
Sensitivity: the other parameters refit with alpha fixed at the ends of its bootstrap CI (D81).
Part B (synthetic Barabasi-Albert networks): agent-based SAIR-H+ with CH's fitted rates, 10 graphs x 20 stochastic runs, in two
variants: seeds only (D79-D80, fails validation) and source injection on hubs with calibrated beta / alpha multipliers (D82).

Usage:
    python src/targeting.py --part A|B|AB [--check] [--fresh]     run (checkpointed, restartable) and tabulate
    python src/targeting.py --check-only                          check on the saved results
"""
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

import data_prep as dp
import sair

SETS = sair.SETS
KS = [10, 25, 50, 100]
METHODS = ["degree", "pagerank", "betweenness", "random"]
ACTIONS = ["block", "recruit", "hybrid"]
CHANNELS = ["both", "friend", "inject"]
SCALES = [1.0, 0.1, 0.01]                                    # recruit injection scale s (D81); block ignores it
EFFORTS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]           # equal-effort levels, fraction of N moved
ALPHA_ENDS = [4.4, 21.5]                                     # bootstrap CI ends of alpha on CH (D68)
N_RANDOM = 15
BETWEENNESS_K = 500
CKPT_A = os.path.join(sair.CKPT_DIR, "phase8_pheme_v2.jsonl")
CKPT_B = os.path.join(sair.CKPT_DIR, "phase8_ba.jsonl")             # seeds-only variant (D79-D80)
CKPT_BI = os.path.join(sair.CKPT_DIR, "phase8_ba_inject.jsonl")     # source-injection variant (D82)
CKPT_CAL = os.path.join(sair.CKPT_DIR, "phase8_ba_calibration.jsonl")
OUT = {n: os.path.join(dp.RESULTS, f"phase8_{n}.csv") for n in
       ("pheme", "channels", "effort", "alpha_fits", "ba", "ba_validation", "ba_calibration")}
COLORS = {"degree": "#0072B2", "pagerank": "#E69F00", "betweenness": "#009E73", "random": "#6b6b6b"}
ACOLOR = {"block": "#0072B2", "recruit": "#E69F00", "hybrid": "#009E73"}

# Part B
BA_N, BA_M, BA_SEEDS, BA_RUNS = 8251, 2, list(range(42, 52)), 20
BA_KPCT = [0.001, 0.005, 0.01, 0.02]
BA_I_SEED, BA_A_SEED, BA_A_TIME = 0.005, 0.001, 25.0       # seeds-only variant: fractions of N; CH's first denial time (h)
BA_P_MAX, BA_DT_CAP = 0.05, 0.5                             # per-step probability <= 0.05; step cap (h)
BA_T_END = 135.0                                            # CH window
BA_RECRUIT_S = 0.1                                          # recruit injection scale in the injection variant
CAL_BETA = [1, 2, 4, 8, 16]                                 # multipliers of beta / alpha, coarse grid (D82)
CAL_ALPHA = [1, 0.3, 0.1, 0.03, 0.01]
CAL_GRAPHS, CAL_RUNS = [42, 43, 44], 5
VAL_TOL = 0.1                                               # final C_I / C_A shares within 0.1 of the data


def main_fit():
    main, table = sair.load_main()
    return main, {key: sair.params_of(table[(table["set"] == key) & (table["model"] == main)].iloc[0], main) for key in SETS}


def tag_params(tag, key):
    """Parameter vector of the main model for a set (tag 'main') or of an alpha-fixed CH refit (D81)."""
    main, fits = main_fit()
    if tag == "main":
        return fits[key]
    t = pd.read_csv(OUT["alpha_fits"]).set_index("variant")
    return np.array([t.loc[tag, n] for n in sair.NAMES[main]], dtype=float)


# ================================================================ Part A: PHEME

_CACHE = {}


def centralities(key):
    """Degree (in-dataset followers = in-degree of the follow graph), PageRank (alpha .85) and betweenness (k = 500, seed 42) on
    the follow-edge graph (u -> v = u follows v). Interaction edges are excluded as in D32 (they are created by the outcome)."""
    if key not in _CACHE:
        sc = sair.get_context(key)
        with open(os.path.join(dp.PROCESSED, f"graph_{key}.gpickle"), "rb") as f:
            G = pickle.load(f)
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes)
        H.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "follow")
        idx = {u: i for i, u in enumerate(sc["users"])}
        out = {}
        for name, d in (("degree", dict(H.in_degree())), ("pagerank", nx.pagerank(H, alpha=0.85)),
                        ("betweenness", nx.betweenness_centrality(H, k=BETWEENNESS_K, seed=dp.SEED))):
            arr = np.zeros(len(idx))
            for u, v in d.items():
                arr[idx[u]] = v
            out[name] = arr
        _CACHE[key] = out
    return _CACHE[key]


def pools(sc):
    """block pool = users who enter I in the data; recruit pool = users who never enter I and authored no source tweet (D77)."""
    is_src = np.isin(np.asarray(sc["users"]), np.unique(sc["src_user"]))
    return np.where(sc["side"] == 1)[0], np.where((sc["side"] != 1) & ~is_src)[0]


def top(score, pool, k):
    return pool[np.lexsort((pool, -score[pool]))][:k]          # highest score first, ties by user index


def choose(sc, cen, method, action, k, rng):
    bp, rp = pools(sc)
    kb = {"block": k, "recruit": 0, "hybrid": (k + 1) // 2}[action]
    kr = {"block": 0, "recruit": k, "hybrid": k // 2}[action]
    if method == "random":
        return (np.sort(rng.choice(bp, kb, replace=False)) if kb else np.array([], int),
                np.sort(rng.choice(rp, kr, replace=False)) if kr else np.array([], int))
    return (top(cen[method], bp, kb) if kb else np.array([], int)), (top(cen[method], rp, kr) if kr else np.array([], int))


def friend_arrays(sc, blocked, recruited):
    """Channel (i): data friend counts after the intervention. Blocked accounts leave n_I from the bin where they would enter I;
    recruited accounts leave their data entry (A side) and count in n_A for all bins from t = 0; they are no longer susceptible."""
    upd, M = sc["upd"], sc["M"]
    n_i, n_a, ne = sc["n_i"].copy(), sc["n_a"].copy(), sc["not_entered"].copy()
    K = len(upd)
    if len(blocked):
        cols = M[:, blocked].toarray().T                               # k x N
        delta = np.zeros((K + 1, n_i.shape[1]))
        for c, j in zip(cols, blocked):
            delta[min(int(np.searchsorted(upd, sc["t_enter"][j])), K)] += c
        n_i -= np.cumsum(delta, axis=0)[:K].astype(np.float32)
    if len(recruited):
        cols = M[:, recruited].toarray().T
        delta = np.zeros((K + 1, n_a.shape[1]))
        for c, j in zip(cols, recruited):
            if sc["side"][j] == 2:                                     # the data entry as A is replaced by "A from t = 0"
                delta[min(int(np.searchsorted(upd, sc["t_enter"][j])), K)] += c
        n_a += (cols.sum(axis=0)[None, :] - np.cumsum(delta, axis=0)[:K]).astype(np.float32)
        ne[:, recruited] = False
    return n_i, n_a, ne


def inject_arrays(sc, blocked, recruited, s=1.0):
    """Channel (ii): blocked authors' supporting-source injections are removed (sum of weights unchanged); recruited accounts
    inject s * theta * w_i / sum(w) into A at t = 0 with w_i = log(1 + followers), the source-injection rule (D46) scaled by s (D81).
    Returns the arrays, the number of removed injections, the injected A mass and the removed I mass (both in units of theta)."""
    users = np.asarray(sc["users"])
    share = sc["src_share"].copy()
    gone = np.isin(sc["src_user"], users[blocked]) & sc["src_to_i"]
    removed = float(share[gone].sum())
    share[gone] = 0.0
    extra = s * np.log1p(sc["followers"][recruited]) / sc["src_wsum"]
    return (np.concatenate([np.zeros(len(recruited)), sc["src_t"]]),
            np.concatenate([np.zeros(len(recruited), bool), sc["src_to_i"]]),
            np.concatenate([extra, share]), int(gone.sum()), float(extra.sum()), removed)


def target_ctx(sc, fa, ia, channel):
    ctx = dict(sc)
    if channel in ("both", "friend"):
        ctx["n_i"], ctx["n_a"], ctx["not_entered"] = fa
    if channel in ("both", "inject"):
        ctx["src_t"], ctx["src_to_i"], ctx["src_share"] = ia[:3]
    return ctx


def outcome(sc, main, x):
    ex = {}
    sair.simulate(sc, main, x, extras=ex)
    return {"D": float(ex["D"]), "peak_I": float(ex["peak_I"]), "C_I": float(ex["C_I"]), "C_A": float(ex["C_A"]),
            "S": float(ex["S"]), "I": float(ex["I"]), "A": float(ex["A"]), "R": float(ex["R"])}


def pheme_job(args):
    """All methods, draws and channels for one (tag, set, action, k, s)."""
    tag, key, action, k, s = args
    main, _ = main_fit()
    x, sc, cen = tag_params(tag, key), sair.get_context(key), centralities(key)
    theta = dict(zip(sair.NAMES[main], x))["theta"]
    rows = []
    for method in METHODS:
        for draw in range(N_RANDOM if method == "random" else 1):
            blocked, recruited = choose(sc, cen, method, action, k, np.random.default_rng(dp.SEED + 1000 * k + draw))
            fa, ia = friend_arrays(sc, blocked, recruited), inject_arrays(sc, blocked, recruited, s)
            inj_a, inj_i = theta * ia[4], theta * ia[5]                  # mass (fraction of N) moved into A / removed from I
            for ch in CHANNELS:
                rows.append({"tag": tag, "set": key, "action": action, "k": k, "s": s, "method": method, "draw": draw, "channel": ch,
                             "n_blocked": len(blocked), "n_recruited": len(recruited), "n_injections_removed": ia[3],
                             "inj_A_mass": inj_a, "inj_I_removed_mass": inj_i, "effort": len(blocked) / sc["N"] + inj_a,
                             **outcome(target_ctx(sc, fa, ia, ch), main, x)})
    return rows


def run_cached(pool, fn, jobs, ckpt, label):
    """jobs: (id, args); every finished job is appended to the checkpoint file (restart-safe), with ETA logging."""
    os.makedirs(sair.CKPT_DIR, exist_ok=True)
    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            r = json.loads(line)
            done[r["id"]] = r["rows"]
    todo = [(i, a) for i, a in jobs if i not in done]
    print(f"[{label}] {len(jobs)} jobs: {len(jobs) - len(todo)} from checkpoint, {len(todo)} to run", flush=True)
    t0 = time.time()
    futures = {pool.submit(fn, a): i for i, a in todo}
    for n, f in enumerate(as_completed(futures), 1):
        done[futures[f]] = f.result()
        with open(ckpt, "a") as fh:
            fh.write(json.dumps({"id": futures[f], "rows": done[futures[f]]}) + "\n")
        el = time.time() - t0
        print(f"[{label}] {n}/{len(todo)} done  {futures[f]}  elapsed {el:.0f} s  ETA {el / n * (len(todo) - n):.0f} s", flush=True)
    return {i: done[i] for i, _ in jobs}


def ci95(v):
    v = np.asarray(v, dtype=float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


def alpha_fits(pool, main):
    """Refit the main model on CH with alpha fixed at each end of its bootstrap CI (1 warm start from the main fit + 10 random starts
    each, checkpointed); saves results/phase8_alpha_fits.csv. Tags 'alpha4.4', 'alpha21.5'."""
    _, fits = main_fit()
    names = sair.NAMES[main]
    warm = dict(zip(names, fits["charliehebdo"]))
    jobs = []
    for a in ALPHA_ENDS:
        fixed = {"alpha": a}
        free = sair.free_names(main, fixed)
        starts = [sair.to_fit(free, np.clip([warm[n] for n in free], *zip(*[sair.RANGE[n] for n in free])))]
        starts += sair.random_starts(free, 10, dp.SEED)
        jobs += [{"key": "charliehebdo", "kind": main, "u0": u0, "tag": f"alpha{a:g}", "fixed": fixed, "maxfev": 5000} for u0 in starts]
    res = sair.run_jobs(pool, jobs, "phase 8 alpha refits", ckpt="phase8_alpha_fits")
    sc = sair.get_context("charliehebdo")
    rows = []
    for tag, x in [("main", fits["charliehebdo"])] + [(f"alpha{a:g}", sair.best_of(res[f"alpha{a:g}"])[0]["x"]) for a in ALPHA_ENDS]:
        k = len(names) - (0 if tag == "main" else 1)
        m = sair.metrics(*sair.residuals(sc, main, np.asarray(x)), k)
        rows.append({"variant": tag, **dict(zip(names, x)), "SSE": m["SSE"], "AIC": m["AIC"], "k_free": k})
    t = pd.DataFrame(rows)
    t["dAIC_vs_main"] = t["AIC"] - t.loc[t["variant"] == "main", "AIC"].iloc[0]
    t.to_csv(OUT["alpha_fits"], index=False)
    print(t.round(4).to_string(index=False))


def pheme_tables(done):
    raw = pd.DataFrame([r for rows in done.values() for r in rows])
    main, _ = main_fit()
    base = {(tag, key): outcome(sair.get_context(key), main, tag_params(tag, key)) for tag in raw["tag"].unique()
            for key in raw[raw["tag"] == tag]["set"].unique()}
    for col in ("D", "peak_I"):
        raw[f"{col}_ratio"] = [r[col] / base[(r["tag"], r["set"])][col] for _, r in raw.iterrows()]
    raw["red_D"] = 1 - raw["D_ratio"]
    raw["red_peak"] = 1 - raw["peak_I_ratio"]
    raw["red_C_I"] = [1 - r["C_I"] / base[(r["tag"], r["set"])]["C_I"] for _, r in raw.iterrows()]
    grp = ["tag", "set", "action", "k", "s", "method", "channel"]
    agg = raw.groupby(grp).agg(n_draws=("draw", "count"), red_D=("red_D", "mean"), red_D_sd=("red_D", "std"),
                               red_D_ci95=("red_D", ci95), red_peak=("red_peak", "mean"), red_C_I=("red_C_I", "mean"),
                               C_I_final=("C_I", "mean"), C_A_final=("C_A", "mean"), D_ratio=("D_ratio", "mean"),
                               peak_ratio=("peak_I_ratio", "mean"), effort=("effort", "mean"), inj_A_mass=("inj_A_mass", "mean"),
                               inj_I_removed_mass=("inj_I_removed_mass", "mean"), n_blocked=("n_blocked", "mean"),
                               n_recruited=("n_recruited", "mean"), n_injections_removed=("n_injections_removed", "mean")).reset_index()
    full = agg[agg["channel"] == "both"].copy()
    key5 = ["tag", "set", "action", "k", "s"]
    rnd = full[full["method"] == "random"].set_index(key5)
    idx = list(zip(*[full[c] for c in key5]))
    full["random_mean"] = [rnd.loc[i, "red_D"] for i in idx]
    full["random_ci95"] = [rnd.loc[i, "red_D_ci95"] for i in idx]
    full["beats_random_ci"] = (full["red_D"] > full["random_mean"] + full["random_ci95"]) & (full["method"] != "random")
    for c, f in (("baseline_D", "D"), ("baseline_peak", "peak_I"), ("baseline_C_I", "C_I"), ("baseline_C_A", "C_A")):
        full[c] = [base[(t, s_)][f] for t, s_ in zip(full["tag"], full["set"])]
    ch = agg.pivot_table(index=["tag", "set", "action", "k", "s", "method"], columns="channel", values="red_D").reset_index()
    ch = ch.rename(columns={"both": "red_both", "friend": "red_friend_only", "inject": "red_inject_only"})
    ch["red_sum_alone"] = ch["red_friend_only"] + ch["red_inject_only"]
    ch["share_inject"] = np.where(ch["red_sum_alone"].abs() > 1e-9, ch["red_inject_only"] / ch["red_sum_alone"], np.nan)
    ch["interaction"] = ch["red_both"] - ch["red_sum_alone"]
    return full.drop(columns="channel"), ch, effort_table(full)


def effort_table(full):
    """Best reduction of each action with effort <= E (E = fraction of N moved: blocked accounts + injected A mass), over all (k, s)
    points of one (tag, set, method): the equal-effort comparison of block, recruit and hybrid (D81)."""
    rows = []
    for (tag, key, method), t in full.groupby(["tag", "set", "method"]):
        for action in ACTIONS:
            pts = t[t["action"] == action]
            for e in EFFORTS:
                ok = pts[pts["effort"] <= e + 1e-12]
                if ok.empty:
                    rows.append({"tag": tag, "set": key, "method": method, "action": action, "effort_level": e, "reachable": False})
                    continue
                b = ok.loc[ok["red_D"].idxmax()]
                rows.append({"tag": tag, "set": key, "method": method, "action": action, "effort_level": e, "reachable": True,
                             "red_D": b["red_D"], "effort": b["effort"], "k": b["k"], "s": b["s"],
                             "random_mean": b["random_mean"], "random_ci95": b["random_ci95"]})
    return pd.DataFrame(rows)


def pheme_report(full, ch, eff):
    pd.set_option("display.width", 220)
    f1 = full[(full["tag"] == "main")]
    print("\n=== PHEME main model, recruit scale s = 1 (as in D78): reduction 1 - D/D0 at k = 50 ===")
    for key in SETS:
        print(f"--- {key}\n" + f1[(f1["set"] == key) & (f1["k"] == 50) & (f1["s"] == 1.0)].pivot(index="action", columns="method", values="red_D")[METHODS].round(4).to_string())
    print("\n=== recruit / hybrid with the injection scaled (k = 50): reduction and effort (fraction of N moved), degree and random ===")
    for key in SETS:
        t = f1[(f1["set"] == key) & (f1["k"] == 50) & f1["action"].isin(["recruit", "hybrid"]) & f1["method"].isin(["degree", "random"])]
        print(f"--- {key}\n" + t.pivot_table(index=["action", "s"], columns="method", values=["red_D", "effort"]).round(4).to_string())
    for tag in ["main"] + [f"alpha{a:g}" for a in ALPHA_ENDS]:
        print(f"\n=== equal effort: best reduction with effort <= E (% of N moved), tag {tag}; degree [random] ===")
        for key in (SETS if tag == "main" else ["charliehebdo"]):
            t = eff[(eff["tag"] == tag) & (eff["set"] == key) & eff["method"].isin(["degree", "random"])]
            lines = []
            for e in EFFORTS:
                row = []
                for a in ACTIONS:
                    v = []
                    for m in ("degree", "random"):
                        r = t[(t["method"] == m) & (t["action"] == a) & (t["effort_level"] == e)].iloc[0]
                        v.append(f"{r['red_D']:.3f}" if r["reachable"] else "  -  ")
                    row.append(f"{a[:3]} {v[0]} [{v[1]}]")
                lines.append(f"E<={100 * e:4.2f}%: " + "  ".join(row))
            print(f"--- {key}\n" + "\n".join(lines))
    print("\n=== channels (main, s = 1, k = 50): injection share of the summed single-channel effects ===")
    c = ch[(ch["tag"] == "main") & (ch["k"] == 50) & (ch["s"] == 1.0) & (ch["method"].isin(["degree", "random"]))]
    print(c.round(4).to_string(index=False))


def plot6(full, s=1.0, name="phase8_plot6_pheme_reduction.png", tag="main"):
    f = full[full["tag"] == tag]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for i, key in enumerate(SETS):
        for j, action in enumerate(ACTIONS):
            ax = axes[i, j]
            for m in METHODS:
                t = f[(f["set"] == key) & (f["action"] == action) & (f["method"] == m) & (f["s"] == (1.0 if action == "block" else s))].sort_values("k")
                ax.plot(t["k"], 100 * t["red_D"], "-o", ms=4, lw=2, color=COLORS[m], label=m)
                if m == "random":
                    ax.fill_between(t["k"], 100 * (t["red_D"] - t["red_D_ci95"]), 100 * (t["red_D"] + t["red_D_ci95"]), color=COLORS[m], alpha=0.25)
            ax.set_title(f"{sair.set_title(key)} - {action}" + ("" if action == "block" else f" (s = {s:g})"), fontsize=10)
            ax.set_xscale("log")
            ax.set_xticks(KS, [str(k) for k in KS])
            ax.grid(alpha=0.25)
            if i == 2:
                ax.set_xlabel("k (accounts acted on)")
            if j == 0:
                ax.set_ylabel("rumour exposure reduction 1 - D/D0 (%)")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Plot 6{'' if s == 1.0 else 'b'} - PHEME: rumour reduction vs k per selection method, recruit injection scale s = {s:g} "
                 f"(random: mean of 15 draws, band = 95% CI of the mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, name), dpi=130)
    plt.close(fig)


def plot7(full, k=50):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    bars = [("block", 1.0)] + [("recruit", s) for s in SCALES] + [("hybrid", s) for s in SCALES]
    cols = {"block": "#0072B2", "recruit": "#E69F00", "hybrid": "#009E73"}
    shade = {1.0: 1.0, 0.1: 0.65, 0.01: 0.4}
    w = 0.11
    for ax, key in zip(axes, SETS):
        for b, (action, s) in enumerate(bars):
            t = full[(full["tag"] == "main") & (full["set"] == key) & (full["action"] == action) & (full["k"] == k) & (full["s"] == s)].set_index("method").loc[METHODS]
            ax.bar(np.arange(4) + (b - 3) * w, 100 * t["red_D"], w, color=cols[action], alpha=shade[s],
                   label=action if action == "block" else f"{action} s = {s:g}",
                   yerr=[100 * (c if m == "random" else 0) for m, c in zip(METHODS, t["red_D_ci95"])], capsize=2)
        ax.set_xticks(range(4), METHODS)
        ax.set_ylabel("rumour exposure reduction 1 - D/D0 (%)")
        ax.set_title(sair.set_title(key))
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.suptitle(f"Plot 7 - PHEME: reduction at k = {k} accounts for block, recruit and hybrid at injection scales s = 1, 0.1, 0.01 (error bar = 95% CI of the random draws)")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase8_plot7_pheme_bars.png"), dpi=130)
    plt.close(fig)


def plot9(full):
    """Reduction vs effort (% of N moved) for block, recruit (each s) and hybrid: the equal-effort comparison (D81)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, key in zip(axes, SETS):
        for action in ACTIONS:
            for s in ([1.0] if action == "block" else SCALES):
                for m, ls in (("degree", "-"), ("random", "--")):
                    t = full[(full["tag"] == "main") & (full["set"] == key) & (full["action"] == action) & (full["s"] == s) & (full["method"] == m)].sort_values("effort")
                    lab = (f"{action}" + ("" if action == "block" else f" s={s:g}") + f", {m}") if (m == "degree" or action == "block") else None
                    ax.plot(100 * t["effort"], 100 * t["red_D"], ls=ls, marker="o", ms=3.5, lw=1.6, color=ACOLOR[action],
                            alpha={1.0: 1.0, 0.1: 0.7, 0.01: 0.45}[s], label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("effort: % of N moved (blocked accounts + injected A mass)")
        ax.set_title(sair.set_title(key))
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("rumour exposure reduction 1 - D/D0 (%)")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Plot 9 - PHEME: reduction at equal effort (solid = degree, dashed = random; points are k = 10 / 25 / 50 / 100)")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase8_plot9_effort.png"), dpi=130)
    plt.close(fig)


# ================================================================ Part B: agent-based SAIR-H+ on BA graphs

@njit(cache=True)
def _abm(indptr, indices, deg, state, beta, alpha, gamma, eps, kappa, eta_i, eta_a, n_aseed, a_time, t_end, p_max, dt_cap, seed,
         ev_t, ev_to, ev_need, ev_hub, ev_ptr, ev_nodes):
    """Synchronous stochastic SAIR-H+ on a graph. States 0 S, 1 I, 2 A, 3 R, 4 B (blocked). Hazards of an S node (counts of
    I / A neighbours nI, nA, degree d, P1 = nI/(nI + nA) over sided neighbours; with none, the hazards reduce to eta):
        to I: beta * Drif1 * nI / d + eta_I      to A: alpha * (1 - Drif1) * nA / d + eta_A,   Drif1 = sigmoid((2 P1 - 1) kappa)
    I -> R at gamma, A -> R at eps. Step dt = min(dt_cap, p_max / largest hazard), clipped to the next whole hour (records), to
    a_time (A seeds) and to the next injection event; a node moves with probability 1 - exp(-hazard * dt) <= p_max.
    Injection events (D82): at time ev_t[e], if the hub ev_hub[e] is not blocked, the first ev_need[e] susceptible nodes of the list
    ev_nodes[ev_ptr[e]:ev_ptr[e+1]] (the hub, then random neighbours of the hub) move to state ev_to[e].
    Returns hourly (I, A, C_I, C_A) counts, the peak I count, final C_I, C_A and the numbers moved by events into I and A."""
    np.random.seed(seed)
    N = len(state)
    n_ev = len(ev_t)
    nI = np.zeros(N, np.int32)
    nA = np.zeros(N, np.int32)
    cntI, cntA, cI, cA = 0, 0, 0, 0
    for i in range(N):
        if state[i] == 1 or state[i] == 2:
            if state[i] == 1:
                cntI += 1
                cI += 1
            else:
                cntA += 1
                cA += 1
            for q in range(indptr[i], indptr[i + 1]):
                if state[i] == 1:
                    nI[indices[q]] += 1
                else:
                    nA[indices[q]] += 1
    hI = np.zeros(N)
    hA = np.zeros(N)
    chg_node = np.empty(N, np.int64)
    chg_to = np.empty(N, np.int8)
    n_rec = int(t_end) + 1
    rec = np.zeros((n_rec, 4))
    t, next_rec, seeded, peak, ne = 0.0, 0, n_aseed == 0, cntI, 0
    app_i, app_a = 0, 0
    while True:
        while ne < n_ev and ev_t[ne] <= t + 1e-9:
            if state[ev_hub[ne]] != 4:
                applied = 0
                for z in range(ev_ptr[ne], ev_ptr[ne + 1]):
                    if applied >= ev_need[ne]:
                        break
                    i = ev_nodes[z]
                    if state[i] == 0:
                        state[i] = ev_to[ne]
                        if ev_to[ne] == 1:
                            cntI += 1
                            cI += 1
                            app_i += 1
                        else:
                            cntA += 1
                            cA += 1
                            app_a += 1
                        for q in range(indptr[i], indptr[i + 1]):
                            if ev_to[ne] == 1:
                                nI[indices[q]] += 1
                            else:
                                nA[indices[q]] += 1
                        applied += 1
            ne += 1
        if cntI > peak:
            peak = cntI
        if (not seeded) and t >= a_time - 1e-9:
            s_nodes = np.where(state == 0)[0]
            np.random.shuffle(s_nodes)
            for z in range(min(n_aseed, len(s_nodes))):
                i = s_nodes[z]
                state[i] = 2
                cntA += 1
                cA += 1
                for q in range(indptr[i], indptr[i + 1]):
                    nA[indices[q]] += 1
            seeded = True
        while next_rec < n_rec and next_rec <= t + 1e-9:
            rec[next_rec, 0], rec[next_rec, 1], rec[next_rec, 2], rec[next_rec, 3] = cntI, cntA, cI, cA
            next_rec += 1
        if t >= t_end - 1e-9:
            break
        hmax = max(gamma, eps)
        for i in range(N):
            if state[i] == 0:
                ni, na = nI[i], nA[i]
                if ni + na > 0:
                    d1 = 1.0 / (1.0 + np.exp(-(2.0 * ni / (ni + na) - 1.0) * kappa))
                    hI[i] = beta * d1 * ni / deg[i] + eta_i
                    hA[i] = alpha * (1.0 - d1) * na / deg[i] + eta_a
                else:
                    hI[i] = eta_i
                    hA[i] = eta_a
                if hI[i] + hA[i] > hmax:
                    hmax = hI[i] + hA[i]
        dt = min(dt_cap, p_max / hmax)
        dt = min(dt, np.floor(t + 1e-9) + 1.0 - t)
        if not seeded:
            dt = min(dt, a_time - t)
        if ne < n_ev:
            dt = min(dt, ev_t[ne] - t)
        p_g = 1.0 - np.exp(-gamma * dt)
        p_e = 1.0 - np.exp(-eps * dt)
        nchg = 0
        for i in range(N):
            s = state[i]
            if s == 0:
                tot = hI[i] + hA[i]
                if tot > 0.0 and np.random.random() < 1.0 - np.exp(-tot * dt):
                    chg_node[nchg] = i
                    chg_to[nchg] = 1 if np.random.random() * tot < hI[i] else 2
                    nchg += 1
            elif s == 1:
                if np.random.random() < p_g:
                    chg_node[nchg] = i
                    chg_to[nchg] = 3
                    nchg += 1
            elif s == 2:
                if np.random.random() < p_e:
                    chg_node[nchg] = i
                    chg_to[nchg] = 3
                    nchg += 1
        for z in range(nchg):
            i, new, old = chg_node[z], chg_to[z], state[chg_node[z]]
            state[i] = new
            if old == 1:
                cntI -= 1
            elif old == 2:
                cntA -= 1
            if new == 1:
                cntI += 1
                cI += 1
            elif new == 2:
                cntA += 1
                cA += 1
            if old == 1 or new == 1:
                for q in range(indptr[i], indptr[i + 1]):
                    nI[indices[q]] += 1 if new == 1 else -1
            if old == 2 or new == 2:
                for q in range(indptr[i], indptr[i + 1]):
                    nA[indices[q]] += 1 if new == 2 else -1
        t += dt
        if abs(t - round(t)) < 1e-9:
            t = float(round(t))
        if cntI > peak:
            peak = cntI
    return rec, peak, cI, cA, app_i, app_a


def ba_graph(gseed):
    """BA graph (N, m, seed), CSR adjacency, degrees and the three centralities (betweenness k = 500 sampling, seed gseed);
    built once and cached in data/processed/ba_graph_<seed>.pkl (git-ignored)."""
    path = os.path.join(dp.PROCESSED, f"ba_graph_{gseed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    G = nx.barabasi_albert_graph(BA_N, BA_M, seed=gseed)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.int8)
    deg = np.diff(A.indptr).astype(np.float64)
    cen = {"degree": deg.copy(), "pagerank": np.array([v for _, v in sorted(nx.pagerank(G, alpha=0.85).items())]),
           "betweenness": np.array([v for _, v in sorted(nx.betweenness_centrality(G, k=BETWEENNESS_K, seed=gseed).items())])}
    out = A.indptr.astype(np.int64), A.indices.astype(np.int64), deg, cen
    with open(path + f".{os.getpid()}", "wb") as f:
        pickle.dump(out, f)
    os.replace(path + f".{os.getpid()}", path)
    return out


_BA_CACHE = {}
_NO_EVENTS = (np.empty(0), np.empty(0, np.int8), np.empty(0, np.int64), np.empty(0, np.int64), np.zeros(1, np.int64), np.empty(0, np.int64))


def stoch_round(x, rng):
    f = int(np.floor(x))
    return f + int(rng.random() < x - f)


def ba_events(gseed, run, graph, theta, recruits, s):
    """Injection events of one run (D82): the j-th source (CH's source times, chronological) is the j-th highest-degree node (the hubs
    are not reused); theta * w_s / sum(w) of N nodes move into I (supporting) or A (denying): the hub, then random neighbours of the hub.
    Recruited nodes inject s * theta * log(1 + degree) / sum(w) of N into A at t = 0 on random neighbours of the node.
    Sum(w) and the shares are CH's (sair context); amounts are rounded stochastically."""
    indptr, indices, deg, _ = graph
    sc = sair.get_context("charliehebdo")
    rng = np.random.default_rng(2_000_003 * gseed + run)
    rank = np.lexsort((np.arange(BA_N), -deg))
    ev = []
    for j, (ts, to_i, share) in enumerate(zip(sc["src_t"], sc["src_to_i"], sc["src_share"])):
        hub = int(rank[j])
        ev.append((float(ts), 1 if to_i else 2, stoch_round(theta * share * BA_N, rng), hub,
                   np.concatenate([[hub], rng.permutation(indices[indptr[hub]:indptr[hub + 1]])])))
    for r in recruits:
        need = stoch_round(theta * s * np.log1p(deg[r]) / sc["src_wsum"] * BA_N, rng)
        if need > 0:
            ev.append((0.0, 2, need, int(r), np.concatenate([[r], rng.permutation(indices[indptr[r]:indptr[r + 1]])])))
    ev.sort(key=lambda e: e[0])                                    # stable: sources keep their order at equal times
    ptr = np.concatenate([[0], np.cumsum([len(e[4]) for e in ev])]).astype(np.int64)
    return (np.array([e[0] for e in ev]), np.array([e[1] for e in ev], np.int8), np.array([e[2] for e in ev], np.int64),
            np.array([e[3] for e in ev], np.int64), ptr, np.concatenate([e[4] for e in ev]).astype(np.int64))


def source_hubs(deg):
    return np.lexsort((np.arange(BA_N), -deg))[:len(sair.get_context("charliehebdo")["src_t"])]


def ba_run(gseed, run, rates, kappa, method, action, k, inject=False, theta=0.0, mult=(1.0, 1.0), s=BA_RECRUIT_S):
    """One stochastic run. Targets are chosen first (block: state B at t = 0; recruit: state A at t = 0), then, in the seeds-only
    variant, the seeds among the rest (the same random permutation per (graph, run) for every configuration, so configurations are
    paired). inject=True: source injection on hubs instead of seeds, beta / alpha multiplied by mult, recruits additionally inject
    (scale s) and are never the source hubs (D82)."""
    if gseed not in _BA_CACHE:
        _BA_CACHE[gseed] = ba_graph(gseed)
    graph = _BA_CACHE[gseed]
    indptr, indices, deg, cen = graph
    rng = np.random.default_rng(1_000_003 * gseed + run)
    perm = rng.permutation(BA_N)
    state = np.zeros(BA_N, np.int8)
    recruits = np.array([], int)
    if k > 0:
        kb = {"block": k, "recruit": 0, "hybrid": (k + 1) // 2}[action]
        kr = {"block": 0, "recruit": k, "hybrid": k // 2}[action]
        hubs = set(source_hubs(deg).tolist()) if inject else set()
        den = set(source_hubs(deg)[~sair.get_context("charliehebdo")["src_to_i"]].tolist()) if inject else set()   # correction sources
        if method == "random":
            if inject:                                                 # blocking targets rumour spreaders: never the denying source's hub (D82)
                blocked = rng.choice(np.setdiff1d(np.arange(BA_N), list(den)), kb, replace=False) if kb else np.array([], int)
                rest = np.setdiff1d(np.arange(BA_N), np.concatenate([blocked, list(hubs)]).astype(int))
                recruits = rng.choice(rest, kr, replace=False) if kr else np.array([], int)
            else:
                targets = rng.choice(BA_N, k, replace=False)           # random draw per run; hybrid: first half blocked
                blocked, recruits = targets[:kb], targets[kb:kb + kr]
        else:
            order = np.lexsort((np.arange(BA_N), -cen[method]))          # ranks 1..kb blocked, then the next kr recruited
            order_b = np.array([i for i in order if i not in den], int)
            blocked = order_b[:kb]
            rest = [i for i in order if i not in hubs and i not in set(blocked.tolist())]
            recruits = np.array(rest[:kr], int)
        state[blocked] = 4
        state[recruits] = 2
    beta, alpha, gamma, eps, eta_i, eta_a = rates
    seed = int(1_000_003 * gseed + run) % (2 ** 31)
    if inject:
        ev = ba_events(gseed, run, graph, theta, recruits, s)
        return _abm(indptr, indices, deg, state, beta * mult[0], alpha * mult[1], gamma, eps, kappa, eta_i, eta_a, 0, BA_A_TIME, BA_T_END,
                    BA_P_MAX, BA_DT_CAP, seed, *ev)
    free = perm[state[perm] == 0]
    state[free[:int(round(BA_I_SEED * BA_N))]] = 1
    return _abm(indptr, indices, deg, state, beta, alpha, gamma, eps, kappa, eta_i, eta_a, int(round(BA_A_SEED * BA_N)), BA_A_TIME,
                BA_T_END, BA_P_MAX, BA_DT_CAP, seed, *_NO_EVENTS)


def rates_of(p):
    return (p["beta"], p["alpha"], p["gamma"], p["eps"], p["eta_I"], p["eta_A"])


def ba_job(args):
    """One graph x kappa mode x (method, action) with every k (or the no-targeting baseline): BA_RUNS runs each.
    variant 'plain' = seeds only (D79); 'inject' = source injection with calibrated multipliers (D82)."""
    gseed, kmode, method, action, kpcts, variant, mult = args
    main, fits = main_fit()
    p = dict(zip(sair.NAMES[main], fits["charliehebdo"]))
    kappa = p["kappa"] if kmode == "fitted" else 0.0
    inject = variant == "inject"
    rows = []
    for kp in kpcts:
        k = int(round(kp * BA_N))
        res = [ba_run(gseed, r, rates_of(p), kappa, method, action, k, inject, p["theta"], tuple(mult)) for r in range(BA_RUNS)]
        curves = np.mean([r[0] for r in res], axis=0) / BA_N
        rows.append({"variant": variant, "gseed": gseed, "kappa_mode": kmode, "method": method, "action": action, "kpct": kp, "k": k,
                     "C_I_final": float(np.mean([r[2] for r in res]) / BA_N), "C_A_final": float(np.mean([r[3] for r in res]) / BA_N),
                     "peak_I": float(np.mean([r[1] for r in res]) / BA_N), "C_I_runs": [float(r[2] / BA_N) for r in res],
                     "moved_I": float(np.mean([r[4] for r in res]) / BA_N), "moved_A": float(np.mean([r[5] for r in res]) / BA_N),
                     "curve_I": curves[:, 0].tolist(), "curve_A": curves[:, 1].tolist(),
                     "curve_CI": curves[:, 2].tolist(), "curve_CA": curves[:, 3].tolist()})
    return rows


def ba_jobs(variant, mult=(1.0, 1.0)):
    pre = "ba" if variant == "plain" else "bai"
    jobs = []
    for g in BA_SEEDS:
        for km in ("fitted", "zero"):
            jobs.append((f"{pre}|{g}|{km}|none", (g, km, "none", "block", [0.0], variant, mult)))
            for m in METHODS:
                for a in ACTIONS:
                    jobs.append((f"{pre}|{g}|{km}|{m}|{a}", (g, km, m, a, BA_KPCT, variant, mult)))
    return jobs


# ---- calibration of the beta / alpha multipliers (D82)

def cal_job(args):
    """Mean agent curves of one (beta multiplier, alpha multiplier) over CAL_GRAPHS x CAL_RUNS runs (no targeting, source injection)."""
    mb, ma = args
    main, fits = main_fit()
    p = dict(zip(sair.NAMES[main], fits["charliehebdo"]))
    res = [ba_run(g, r, rates_of(p), p["kappa"], "none", "block", 0, True, p["theta"], (mb, ma)) for g in CAL_GRAPHS for r in range(CAL_RUNS)]
    cur = np.mean([r[0] for r in res], axis=0) / BA_N
    return [{"mult_beta": mb, "mult_alpha": ma, "curve_I": cur[:, 0].tolist(), "curve_A": cur[:, 1].tolist(), "curve_CI": cur[:, 2].tolist(),
             "curve_CA": cur[:, 3].tolist(), "moved_I": float(np.mean([r[4] for r in res]) / BA_N), "moved_A": float(np.mean([r[5] for r in res]) / BA_N)}]


def validation_metrics(curve_I, curve_A, curve_CI, curve_CA):
    """Shape RMSE against CH's normalised cumulative curves (pooled C_I and C_A, each divided by its own final value, at the hourly
    data bins), final shares against the data, and the order of the peaks of the mean I(t), A(t) (D82)."""
    sc = sair.get_context("charliehebdo")
    grid = sc["grid"]
    hours = np.arange(len(curve_CI), dtype=float)
    ci, ca = np.interp(grid, hours, curve_CI), np.interp(grid, hours, curve_CA)
    ri = ci / ci[-1] - sc["C_I"] / sc["fin_I"] if ci[-1] > 0 else np.full(len(grid), np.nan)
    ra = ca / ca[-1] - sc["C_A"] / sc["fin_A"] if ca[-1] > 0 else np.full(len(grid), np.nan)
    rmse = float(np.sqrt(np.mean(np.concatenate([ri, ra]) ** 2)))
    tp_i, tp_a = float(hours[int(np.argmax(curve_I))]), float(hours[int(np.argmax(curve_A))])
    d_i, d_a = float(curve_CI[-1] - sc["fin_I"]), float(curve_CA[-1] - sc["fin_A"])
    return {"rmse_shape": rmse, "final_C_I": float(curve_CI[-1]), "final_C_A": float(curve_CA[-1]), "data_final_C_I": float(sc["fin_I"]),
            "data_final_C_A": float(sc["fin_A"]), "diff_C_I": d_i, "diff_C_A": d_a, "peak_I_h": tp_i, "peak_A_h": tp_a,
            "I_peaks_before_A": bool(tp_i < tp_a), "passes": bool(abs(d_i) <= VAL_TOL and abs(d_a) <= VAL_TOL and tp_i < tp_a)}


def calibrate(done):
    rows = []
    for rs in done.values():
        for r in rs:
            rows.append({"mult_beta": r["mult_beta"], "mult_alpha": r["mult_alpha"], "moved_I": r["moved_I"], "moved_A": r["moved_A"],
                         **validation_metrics(r["curve_I"], r["curve_A"], r["curve_CI"], r["curve_CA"])})
    t = pd.DataFrame(rows).sort_values(["mult_beta", "mult_alpha"], ascending=[True, False]).reset_index(drop=True)
    ok = t[t["passes"]]
    pick = (ok if len(ok) else t).sort_values("rmse_shape").iloc[0]
    t["selected"] = (t["mult_beta"] == pick["mult_beta"]) & (t["mult_alpha"] == pick["mult_alpha"])
    t["selection_rule"] = "min shape RMSE among grid points that pass validation" if len(ok) else "min shape RMSE overall (no grid point passes)"
    return t, (float(pick["mult_beta"]), float(pick["mult_alpha"]))


# ---- tables

def ba_tables(raw, label):
    """Per variant: reduction tables (paired over graphs, 95% t-CI over the 10 graph means), conformity share."""
    base = raw[raw["method"] == "none"].set_index(["variant", "gseed", "kappa_mode"])
    rows = []
    for (var, km, m, a, kp), t in raw[raw["method"] != "none"].groupby(["variant", "kappa_mode", "method", "action", "kpct"]):
        t = t.set_index("gseed").sort_index()
        b = base.xs(var, level="variant").xs(km, level="kappa_mode").sort_index()
        red_f = 1 - t["C_I_final"] / b["C_I_final"]
        red_p = 1 - t["peak_I"] / b["peak_I"]
        rows.append({"variant": var, "validation_status": label[var], "kappa_mode": km, "method": m, "action": a, "kpct": kp,
                     "k": int(t["k"].iloc[0]), "n_graphs": len(t), "n_runs": len(t) * BA_RUNS,
                     "C_I_final": t["C_I_final"].mean(), "C_I_final_ci95": ci95(t["C_I_final"]), "peak_I": t["peak_I"].mean(),
                     "peak_I_ci95": ci95(t["peak_I"]), "C_A_final": t["C_A_final"].mean(),
                     "baseline_C_I_final": b["C_I_final"].mean(), "baseline_peak_I": b["peak_I"].mean(),
                     "red_final": red_f.mean(), "red_final_ci95": ci95(red_f), "red_peak": red_p.mean(), "red_peak_ci95": ci95(red_p)})
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["variant", "method", "action", "kpct"], columns="kappa_mode", values="red_final")
    piv["conformity_share"] = np.where(piv["fitted"].abs() > 1e-9, (piv["fitted"] - piv["zero"]) / piv["fitted"], np.nan)
    return out.merge(piv[["conformity_share"]].reset_index(), on=["variant", "method", "action", "kpct"])


def ba_validation(raw, label):
    """Agent model (no targeting, fitted kappa) vs the mean-field main model on CH, per variant."""
    main, fits = main_fit()
    sc = sair.get_context("charliehebdo")
    mf = sair.simulate(sc, main, fits["charliehebdo"], record=True)
    val, curves = [], {}
    for var in raw["variant"].unique():
        b = raw[(raw["variant"] == var) & (raw["method"] == "none") & (raw["kappa_mode"] == "fitted")]
        cur = {c: np.mean(list(b[f"curve_{c}"]), axis=0) for c in ("I", "A", "CI", "CA")}
        hours = np.arange(len(cur["I"]), dtype=float)
        mfc = {c: np.interp(hours, mf["t"], mf[col]) for c, col in (("I", "I"), ("A", "A"), ("CI", "C_I"), ("CA", "C_A"))}
        curves[var] = (hours, cur, mfc)
        vm = validation_metrics(cur["I"], cur["A"], cur["CI"], cur["CA"])
        for q, ag, mfv in [("peak I (fraction of N)", cur["I"].max(), mfc["I"].max()),
                           ("time of peak I (h)", float(hours[cur["I"].argmax()]), float(hours[mfc["I"].argmax()])),
                           ("time of peak A (h)", float(hours[cur["A"].argmax()]), float(hours[mfc["A"].argmax()])),
                           ("final C_I", cur["CI"][-1], mfc["CI"][-1]), ("final C_A", cur["CA"][-1], mfc["CA"][-1]),
                           ("time C_I reaches 50% of its final value (h)", float(np.interp(0.5, cur["CI"] / cur["CI"][-1], hours)), float(np.interp(0.5, mfc["CI"] / mfc["CI"][-1], hours))),
                           ("corr(I(t))", float(np.corrcoef(cur["I"], mfc["I"])[0, 1]), 1.0), ("corr(C_I(t))", float(np.corrcoef(cur["CI"], mfc["CI"])[0, 1]), 1.0),
                           ("corr(C_A(t))", float(np.corrcoef(cur["CA"], mfc["CA"])[0, 1]), 1.0)]:
            val.append({"variant": var, "validation_status": label[var], "quantity": q, "agent": ag, "mean_field": mfv})
        val.append({"variant": var, "validation_status": label[var], "quantity": "shape RMSE vs data (normalised C_I, C_A)", "agent": vm["rmse_shape"], "mean_field": np.nan})
        val.append({"variant": var, "validation_status": label[var], "quantity": "validation passes (final shares within 0.1, I peaks before A)", "agent": float(vm["passes"]), "mean_field": np.nan})
    return pd.DataFrame(val), curves


def ba_report(out, val, cal=None):
    pd.set_option("display.width", 220)
    if cal is not None:
        print("\n=== BA calibration grid (beta multiplier x alpha multiplier; 3 graphs x 5 runs) ===")
        print(cal.drop(columns=["selection_rule"]).round(4).to_string(index=False))
        print(cal["selection_rule"].iloc[0])
    print("\n=== BA: agent-model validation against the mean-field main model on CH (no targeting) ===")
    print(val.round(4).to_string(index=False))
    for var in out["variant"].unique():
        lab = out[out["variant"] == var]["validation_status"].iloc[0]
        for km in ("fitted", "zero"):
            print(f"\n=== BA [{var}; {lab}] ({km} kappa): reduction of the final rumour size 1 - C_I/C_I0 (mean over 10 graphs ± 95% CI) ===")
            t = out[(out["variant"] == var) & (out["kappa_mode"] == km)]
            for a in ACTIONS:
                p = t[t["action"] == a].pivot(index="kpct", columns="method", values="red_final")[METHODS]
                c = t[t["action"] == a].pivot(index="kpct", columns="method", values="red_final_ci95")[METHODS]
                print(f"--- {a}\n" + pd.DataFrame({m: [f"{100 * p.loc[i, m]:6.1f} ± {100 * c.loc[i, m]:4.1f}" for i in p.index] for m in METHODS},
                                                  index=[f"{100 * i:g}%" for i in p.index]).to_string())
        print(f"\n=== BA [{var}; {lab}]: share of the benefit attributable to conformity (fitted vs kappa = 0), final size ===")
        print(out[(out["variant"] == var) & (out["kappa_mode"] == "fitted")].pivot_table(index=["action", "kpct"], columns="method", values="conformity_share")[METHODS].round(3).to_string())


def plot8(out, var, name, label):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharey=True)
    t0 = out[out["variant"] == var]
    for ax, a in zip(axes, ACTIONS):
        for m in METHODS:
            for km, ls in (("fitted", "-"), ("zero", "--")):
                t = t0[(t0["action"] == a) & (t0["method"] == m) & (t0["kappa_mode"] == km)].sort_values("kpct")
                ax.errorbar(100 * t["kpct"], 100 * t["red_final"], yerr=100 * t["red_final_ci95"], color=COLORS[m], ls=ls, marker="o", ms=4, lw=1.8,
                            capsize=2, label=f"{m}, {'κ fitted' if km == 'fitted' else 'κ = 0'}")
        ax.set_xscale("log")
        ax.set_xticks([100 * k for k in BA_KPCT], [f"{100 * k:g}" for k in BA_KPCT])
        ax.set_xlabel("k (% of N acted on)")
        ax.set_title(a + (f" (recruit injection scale s = {BA_RECRUIT_S:g})" if var == "inject" and a != "block" else ""))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("reduction of the final rumour size (%)")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle(f"Plot 8 - BA network, {label}: final-size reduction vs k, fitted κ (solid) vs κ = 0 (dashed); 10 graphs x 20 runs, bars = 95% CI over graphs")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, name), dpi=130)
    plt.close(fig)


def plot8b(curves, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cols = {"plain": "#7a7a7a", "inject": "#0072B2"}
    for var, (hours, cur, mfc) in curves.items():
        axes[0].plot(hours, 100 * cur["I"], color=cols[var], lw=2, label=f"agent, {var}: {label[var]}")
        axes[1].plot(hours, 100 * cur["CI"], color=cols[var], lw=2, label=f"agent C_I, {var}")
        axes[1].plot(hours, 100 * cur["CA"], color=cols[var], lw=2, ls="--", label=f"agent C_A, {var}")
    hours, cur, mfc = list(curves.values())[0]
    axes[0].plot(hours, 100 * mfc["I"], color="#d6336c", lw=2, label="mean-field main model, CH fit")
    axes[1].plot(hours, 100 * mfc["CI"], color="#d6336c", lw=2, label="mean-field C_I")
    axes[1].plot(hours, 100 * mfc["CA"], color="#d6336c", lw=2, ls="--", label="mean-field C_A")
    axes[0].set_ylabel("I (% of N)")
    axes[1].set_ylabel("cumulative entries (% of N)")
    for ax in axes:
        ax.set_xlabel("time (h)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5)
    fig.suptitle("Plot 8b - validation: agent model (no targeting) vs mean-field main model, Charlie Hebdo")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase8_plot8b_ba_validation.png"), dpi=130)
    plt.close(fig)


def plot8c(cal):
    sc = sair.get_context("charliehebdo")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, ttl in zip(axes, ["rmse_shape", "final_C_A"], ["shape RMSE (normalised C_I, C_A vs data)", "final C_A share (data: %.3f)" % sc["fin_A"]]):
        m = cal.pivot(index="mult_alpha", columns="mult_beta", values=col).sort_index(ascending=False)
        im = ax.imshow(m.to_numpy(), cmap="YlGnBu_r" if col == "rmse_shape" else "viridis", origin="upper")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.to_numpy()[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white")
        sel = cal[cal["selected"]].iloc[0]
        ax.plot(list(m.columns).index(sel["mult_beta"]), list(m.index).index(sel["mult_alpha"]), marker="s", ms=22, mfc="none", mec="#d6336c", mew=2)
        ax.set_xticks(range(m.shape[1]), [f"{v:g}" for v in m.columns])
        ax.set_yticks(range(m.shape[0]), [f"{v:g}" for v in m.index])
        ax.set_xlabel("β multiplier")
        ax.set_ylabel("α multiplier")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Plot 8c - BA calibration grid with source injection (red square = selected)")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase8_plot8c_ba_calibration.png"), dpi=130)
    plt.close(fig)


# ================================================================ checks

def check_a():
    print("\n--- Phase 8 check, part A (PHEME) ---")
    main, fits = main_fit()
    full = pd.read_csv(OUT["pheme"])
    ch = pd.read_csv(OUT["channels"])
    for key in SETS:
        sc, x, cen = sair.get_context(key), fits[key], centralities(key)
        base = outcome(sc, main, x)
        ref = pd.read_csv(os.path.join(dp.PROCESSED, f"model_curves_{key}.csv"))
        r = sair.simulate(sc, main, x, record=True)
        assert float(np.abs(r[["S", "I", "A", "R", "C_I", "C_A"]].to_numpy() - ref[["S", "I", "A", "R", "C_I", "C_A"]].to_numpy()).max()) <= 1e-12
        e = np.array([], int)
        z = outcome(target_ctx(sc, friend_arrays(sc, e, e), inject_arrays(sc, e, e), "both"), main, x)
        assert z == base, f"{key}: empty intervention differs from the baseline"
        bp, rp = pools(sc)
        n_inj = {}
        for action in ACTIONS:
            for k in KS:
                sizes = set()
                for method in METHODS:
                    b, rc = choose(sc, cen, method, action, k, np.random.default_rng(1))
                    sizes.add(len(b) + len(rc))
                    assert np.isin(b, bp).all() and np.isin(rc, rp).all(), "targets outside their pool"
                    assert len(set(b) & set(rc)) == 0 and len(set(b)) == len(b) and len(set(rc)) == len(rc)
                    if method != "random":
                        n_inj[(action, k, method)] = inject_arrays(sc, b, rc)[3]
                assert sizes == {k}, f"{key}/{action}/k={k}: methods use different numbers of accounts {sizes}"
                if action == "hybrid":
                    assert len(choose(sc, cen, "degree", action, k, np.random.default_rng(1))[0]) == (k + 1) // 2
        b, rc = choose(sc, cen, "pagerank", "hybrid", 100, np.random.default_rng(1))
        fa, ia = friend_arrays(sc, b, rc), inject_arrays(sc, b, rc)
        assert (fa[0] >= -1e-6).all() and (fa[1] >= -1e-6).all()
        res = sair.simulate(target_ctx(sc, fa, ia, "both"), main, x, record=True)
        worst = float(np.abs(res[["S", "I", "A", "R"]].sum(axis=1) - 1).max())
        assert worst <= sair.INV_TOL and res[["S", "I", "A", "R"]].to_numpy().min() >= -sair.INV_TOL
        # recruit injection scales linearly with s; effort bookkeeping (blocked accounts + injected A mass)
        b, rc = choose(sc, cen, "degree", "hybrid", 50, np.random.default_rng(1))
        m1, m01 = inject_arrays(sc, b, rc, 1.0)[4], inject_arrays(sc, b, rc, 0.1)[4]
        assert abs(m01 - 0.1 * m1) < 1e-12 and m1 > 0
        t = full[(full["tag"] == "main") & (full["set"] == key)]
        assert (t["n_injections_removed"] >= 0).all() and (t["effort"] >= 0).all()
        d = t[(t["action"] == "hybrid") & (t["method"] == "degree") & (t["k"] == 50)]
        theta = fits[key][sair.NAMES[main].index("theta")]
        for s_, g in d.groupby("s"):
            assert abs(g["effort"].iloc[0] - (25 / sc["N"] + theta * inject_arrays(sc, b, rc, s_)[4])) < 1e-9
        print(f"{key:22s} OK  empty intervention == baseline == Phase 5 curves; same k for every method; targets in pools; max|sum-1|={worst:.1e}; "
              f"effort = blocked/N + injected A mass (s scales it exactly); source injections removed by block (k=100): degree {n_inj[('block', 100, 'degree')]}, "
              f"PageRank {n_inj[('block', 100, 'pagerank')]}, betweenness {n_inj[('block', 100, 'betweenness')]}")
    assert set(full["k"]) == set(KS) and set(full["method"]) == set(METHODS) and set(full["s"]) == set(SCALES)
    assert set(full["tag"]) == {"main"} | {f"alpha{a:g}" for a in ALPHA_ENDS}
    assert (full[full["method"] == "random"]["n_draws"] == N_RANDOM).all()
    af = pd.read_csv(OUT["alpha_fits"]).set_index("variant")
    assert abs(af.loc["alpha4.4", "alpha"] - 4.4) < 1e-9 and abs(af.loc["alpha21.5", "alpha"] - 21.5) < 1e-9
    print(f"channel table rows {len(ch)}; alpha refits: " + "; ".join(f"{v}: dAIC {af.loc[v, 'dAIC_vs_main']:+.1f}" for v in af.index) + "; Part A checks passed.")


def check_b():
    print("\n--- Phase 8 check, part B (BA agent model) ---")
    main, fits = main_fit()
    p = dict(zip(sair.NAMES[main], fits["charliehebdo"]))
    rates = rates_of(p)
    rec, peak, cI, cA, *_ = ba_run(42, 0, rates, p["kappa"], "none", "block", 0)
    assert rec[0, 0] == round(BA_I_SEED * BA_N) and rec[0, 1] == 0, "initial seeds wrong"
    assert (np.diff(rec[:, 2]) >= 0).all() and (np.diff(rec[:, 3]) >= 0).all(), "cumulative counts decrease"
    assert rec[int(BA_A_TIME) - 1, 1] == 0 and rec[int(BA_A_TIME), 1] > 0, "A seeds must appear at the first denial time"
    for action in ("block", "recruit"):
        rec2, *_ = ba_run(42, 0, rates, p["kappa"], "degree", action, 83)
        assert (rec2[0, 1] == 83) if action == "recruit" else (rec2[0, 0] == round(BA_I_SEED * BA_N))
    assert np.array_equal(ba_run(42, 3, rates, p["kappa"], "pagerank", "hybrid", 41)[0], ba_run(42, 3, rates, p["kappa"], "pagerank", "hybrid", 41)[0])
    # injection variant: events move at most the specified mass; a blocked hub removes its event; recruit injection moves more into A
    cal = pd.read_csv(OUT["ba_calibration"])
    sel = cal[cal["selected"]].iloc[0]
    mult = (float(sel["mult_beta"]), float(sel["mult_alpha"]))
    sc = sair.get_context("charliehebdo")
    exp_i = p["theta"] * sc["src_share"][sc["src_to_i"]].sum()
    exp_a = p["theta"] * sc["src_share"][~sc["src_to_i"]].sum()
    mi, ma = [], []
    for r in range(10):
        out = ba_run(42, r, rates, p["kappa"], "none", "block", 0, True, p["theta"], mult)
        mi.append(out[4] / BA_N)
        ma.append(out[5] / BA_N)
    assert np.mean(mi) <= exp_i + 0.005 and np.mean(ma) <= exp_a + 0.005 and np.mean(mi) > 0.2 * exp_i, (np.mean(mi), exp_i, np.mean(ma), exp_a)
    graph = ba_graph(42)
    hubs = source_hubs(graph[2])
    out_b = ba_run(42, 0, rates, p["kappa"], "degree", "block", 41, True, p["theta"], mult)          # top-41 degree blocked: includes the source hubs
    out_0 = ba_run(42, 0, rates, p["kappa"], "none", "block", 0, True, p["theta"], mult)
    assert out_b[4] < out_0[4], "blocking the source hubs must remove injections"
    assert out_b[5] > 0, "the denying source's hub must never be blocked (correction injection stays)"
    out_r = ba_run(42, 0, rates, p["kappa"], "random", "recruit", 41, True, p["theta"], mult, 1.0)
    out_r0 = ba_run(42, 0, rates, p["kappa"], "random", "recruit", 41, True, p["theta"], mult, 0.0)
    assert out_r[5] > out_r0[5], "recruit injection (s > 0) must move more nodes into A than s = 0"
    ev = ba_events(42, 0, graph, p["theta"], np.array([], int), 0.0)
    assert len(ev[0]) == len(sc["src_t"]) and (np.diff(ev[0]) >= 0).all() and len(set(ev[3].tolist())) == len(ev[3]), "source events: one distinct hub per source"
    assert set(ev[3].tolist()) == set(hubs.tolist())
    out = pd.read_csv(OUT["ba"])
    assert set(out["n_graphs"]) == {len(BA_SEEDS)} and set(out["n_runs"]) == {len(BA_SEEDS) * BA_RUNS}
    assert set(out["kpct"]) == set(BA_KPCT) and set(out["variant"]) == {"plain", "inject"}
    print(f"seeds-only variant: initialisation, monotone cumulative counts, A seeds at 25 h, reproducibility OK; injection variant (multipliers {mult}): "
          f"mean moved into I {np.mean(mi):.3f} (specified {exp_i:.3f}), into A {np.mean(ma):.4f} (specified {exp_a:.4f}); blocked source hubs remove injections; "
          f"recruit injection moves mass into A")
    print("Part B checks passed.")


# ================================================================ main

if __name__ == "__main__":
    part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "AB"
    if "--check-only" not in sys.argv:
        if "--fresh" in sys.argv:
            for c in ((CKPT_A,) if "A" in part else ()) + ((CKPT_BI, CKPT_CAL) if "B" in part else ()):
                if os.path.exists(c):
                    os.remove(c)
        for key in SETS:
            sair.get_context(key)
        main, _ = main_fit()
        with ProcessPoolExecutor(max_workers=sair.N_WORKERS) as pool:
            if "A" in part:
                alpha_fits(pool, main)
                jobs = []
                for tag in ["main"] + [f"alpha{a:g}" for a in ALPHA_ENDS]:
                    for key in (SETS if tag == "main" else ["charliehebdo"]):
                        for k in KS:
                            jobs.append((f"A|{tag}|{key}|block|{k}|1", (tag, key, "block", k, 1.0)))
                            for a in ("recruit", "hybrid"):
                                for s in SCALES:
                                    jobs.append((f"A|{tag}|{key}|{a}|{k}|{s:g}", (tag, key, a, k, s)))
                done = run_cached(pool, pheme_job, jobs, CKPT_A, "phase 8 part A")
                full, ch, eff = pheme_tables(done)
                full.to_csv(OUT["pheme"], index=False)
                ch.to_csv(OUT["channels"], index=False)
                eff.to_csv(OUT["effort"], index=False)
                pheme_report(full, ch, eff)
                plot6(full, 1.0, "phase8_plot6_pheme_reduction.png")
                plot6(full, 0.1, "phase8_plot6b_pheme_reduction_s0.1.png")
                plot7(full)
                plot9(full)
            if "B" in part:
                list(pool.map(ba_graph, BA_SEEDS))                  # build and cache the 10 graphs and their centralities
                cal_done = run_cached(pool, cal_job, [(f"cal|{mb:g}|{ma:g}", (mb, ma)) for mb in CAL_BETA for ma in CAL_ALPHA], CKPT_CAL, "phase 8 BA calibration")
                cal, mult = calibrate(cal_done)
                cal.to_csv(OUT["ba_calibration"], index=False)
                plot8c(cal)
                passed = bool(cal[cal["selected"]]["passes"].iloc[0])
                label = {"plain": "seeds only (no injection): validation failed, QUALITATIVE ONLY",
                         "inject": "source injection: validation passed" if passed else "source injection: validation FAILED, QUALITATIVE ONLY"}
                status = {"plain": "qualitative only (validation failed)", "inject": "validated" if passed else "qualitative only (validation failed)"}
                done = run_cached(pool, ba_job, ba_jobs("plain"), CKPT_B, "phase 8 part B (seeds only)")
                done_i = run_cached(pool, ba_job, ba_jobs("inject", mult), CKPT_BI, "phase 8 part B (source injection)")
                raw = pd.DataFrame([r for rows in list(done.values()) + list(done_i.values()) for r in rows])
                if "variant" not in raw:
                    raw["variant"] = "plain"
                raw["variant"] = raw["variant"].fillna("plain")
                out = ba_tables(raw, status)
                val, curves = ba_validation(raw, status)
                out.to_csv(OUT["ba"], index=False)
                val.to_csv(OUT["ba_validation"], index=False)
                ba_report(out, val, cal)
                plot8(out, "inject", "phase8_plot8_ba_reduction.png", label["inject"])
                plot8(out, "plain", "phase8_plot8_ba_reduction_noinjection.png", label["plain"])
                plot8b(curves, label)
    if "--check" in sys.argv or "--check-only" in sys.argv:
        if "A" in part:
            check_a()
        if "B" in part:
            check_b()
