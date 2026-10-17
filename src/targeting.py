"""Phase 8 - centrality-targeted intervention (docs/extension.md §2, D77-D80).

Part A (PHEME, main model (d+)-uniform with the fitted parameters of each set): pick k accounts by degree, PageRank,
betweenness (k = 500 sampling) or at random (15 draws), all on the FOLLOW graph, and
  block   the chosen accounts (pool = accounts that enter I in the data). Channel (i): they leave the n_I counts of their
          followers from the moment they would enter I; channel (ii): a supporting source tweet they authored loses its injection.
  recruit the chosen accounts (pool = accounts that never enter I and authored no source tweet) as correction broadcasters
          from t = 0: they count in n_A for their followers (from t = 0) and inject theta * w_i / sum(w) into A.
  hybrid  ceil(k/2) blocked + floor(k/2) recruited (top of each list).
Each channel is also run alone.
Part B (synthetic Barabasi-Albert networks): agent-based SAIR-H+ with CH's fitted rates, 10 graphs x 20 stochastic runs.

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
from scipy.sparse import csr_matrix

import data_prep as dp
import sair

SETS = sair.SETS
KS = [10, 25, 50, 100]
METHODS = ["degree", "pagerank", "betweenness", "random"]
ACTIONS = ["block", "recruit", "hybrid"]
CHANNELS = ["both", "friend", "inject"]
N_RANDOM = 15
BETWEENNESS_K = 500
CKPT_A = os.path.join(sair.CKPT_DIR, "phase8_pheme.jsonl")
CKPT_B = os.path.join(sair.CKPT_DIR, "phase8_ba.jsonl")
OUT = {n: os.path.join(dp.RESULTS, f"phase8_{n}.csv") for n in ("pheme", "channels", "ba", "ba_validation")}
COLORS = {"degree": "#0072B2", "pagerank": "#E69F00", "betweenness": "#009E73", "random": "#6b6b6b"}

# Part B
BA_N, BA_M, BA_SEEDS, BA_RUNS = 8251, 2, list(range(42, 52)), 20
BA_KPCT = [0.001, 0.005, 0.01, 0.02]
BA_I_SEED, BA_A_SEED, BA_A_TIME = 0.005, 0.001, 25.0       # fractions of N; CH's first denial time (h)
BA_P_MAX, BA_DT_CAP = 0.05, 0.5                             # per-step probability <= 0.05; step cap (h)
BA_T_END = 135.0                                            # CH window


def main_fit():
    main, table = sair.load_main()
    return main, {key: sair.params_of(table[(table["set"] == key) & (table["model"] == main)].iloc[0], main) for key in SETS}


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


def inject_arrays(sc, blocked, recruited):
    """Channel (ii): blocked authors' supporting-source injections are removed (sum of weights unchanged); recruited accounts
    inject theta * w_i / sum(w) into A at t = 0 with w_i = log(1 + followers), the source-injection rule (D46)."""
    users = np.asarray(sc["users"])
    share = sc["src_share"].copy()
    gone = np.isin(sc["src_user"], users[blocked]) & sc["src_to_i"]
    share[gone] = 0.0
    extra = np.log1p(sc["followers"][recruited]) / sc["src_wsum"]
    return (np.concatenate([np.zeros(len(recruited)), sc["src_t"]]),
            np.concatenate([np.zeros(len(recruited), bool), sc["src_to_i"]]),
            np.concatenate([extra, share]), int(gone.sum()))


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
    """All methods, draws and channels for one (set, action, k)."""
    key, action, k = args
    main, fits = main_fit()
    x, sc, cen = fits[key], sair.get_context(key), centralities(key)
    rows = []
    for method in METHODS:
        for draw in range(N_RANDOM if method == "random" else 1):
            blocked, recruited = choose(sc, cen, method, action, k, np.random.default_rng(dp.SEED + 1000 * k + draw))
            fa, ia = friend_arrays(sc, blocked, recruited), inject_arrays(sc, blocked, recruited)
            for ch in CHANNELS:
                rows.append({"set": key, "action": action, "k": k, "method": method, "draw": draw, "channel": ch,
                             "n_blocked": len(blocked), "n_recruited": len(recruited), "n_injections_removed": ia[3],
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


def pheme_tables(done):
    raw = pd.DataFrame([r for rows in done.values() for r in rows])
    base = {key: outcome(sair.get_context(key), main_fit()[0], main_fit()[1][key]) for key in SETS}
    for col, b in (("D", "D"), ("peak_I", "peak_I")):
        raw[f"{col}_ratio"] = [r[col] / base[r["set"]][b] for _, r in raw.iterrows()]
    raw["red_D"] = 1 - raw["D_ratio"]
    raw["red_peak"] = 1 - raw["peak_I_ratio"]
    raw["red_C_I"] = [1 - r["C_I"] / base[r["set"]]["C_I"] for _, r in raw.iterrows()]
    grp = ["set", "action", "k", "method", "channel"]
    agg = raw.groupby(grp).agg(n_draws=("draw", "count"), red_D=("red_D", "mean"), red_D_sd=("red_D", "std"),
                               red_D_ci95=("red_D", ci95), red_peak=("red_peak", "mean"), red_C_I=("red_C_I", "mean"),
                               C_I_final=("C_I", "mean"), C_A_final=("C_A", "mean"), D_ratio=("D_ratio", "mean"),
                               peak_ratio=("peak_I_ratio", "mean"), n_blocked=("n_blocked", "mean"),
                               n_recruited=("n_recruited", "mean"), n_injections_removed=("n_injections_removed", "mean")).reset_index()
    full = agg[agg["channel"] == "both"].copy()
    rnd = full[full["method"] == "random"].set_index(["set", "action", "k"])
    for col in ("red_D",):
        m = [rnd.loc[(r["set"], r["action"], r["k"]), "red_D"] for _, r in full.iterrows()]
        h = [rnd.loc[(r["set"], r["action"], r["k"]), "red_D_ci95"] for _, r in full.iterrows()]
        sd = [rnd.loc[(r["set"], r["action"], r["k"]), "red_D_sd"] for _, r in full.iterrows()]
        full["random_mean"], full["random_ci95"] = m, h
        full["beats_random_ci"] = (full["red_D"] > full["random_mean"] + full["random_ci95"]) & (full["method"] != "random")
        full["beats_random_by_sd"] = (full["red_D"] > full["random_mean"] + np.array(sd)) & (full["method"] != "random")
    full["baseline_D"] = [base[s]["D"] for s in full["set"]]
    full["baseline_peak"] = [base[s]["peak_I"] for s in full["set"]]
    full["baseline_C_I"] = [base[s]["C_I"] for s in full["set"]]
    full["baseline_C_A"] = [base[s]["C_A"] for s in full["set"]]
    ch = agg.pivot_table(index=["set", "action", "k", "method"], columns="channel", values="red_D").reset_index()
    ch = ch.rename(columns={"both": "red_both", "friend": "red_friend_only", "inject": "red_inject_only"})
    ch["red_sum_alone"] = ch["red_friend_only"] + ch["red_inject_only"]
    ch["share_inject"] = np.where(ch["red_sum_alone"].abs() > 1e-9, ch["red_inject_only"] / ch["red_sum_alone"], np.nan)
    ch["interaction"] = ch["red_both"] - ch["red_sum_alone"]
    return full.drop(columns="channel"), ch


def pheme_report(full, ch):
    pd.set_option("display.width", 220)
    print("\n=== PHEME: rumour-exposure reduction 1 - D/D0 (mean over random draws; deterministic methods single value) ===")
    for key in SETS:
        for action in ACTIONS:
            t = full[(full["set"] == key) & (full["action"] == action)].pivot(index="k", columns="method", values="red_D")[METHODS]
            print(f"--- {key} / {action}\n{t.round(4).to_string()}")
    print("\n=== winner per set / action / k, and degree / PageRank vs the random CI ===")
    rows = []
    for (key, action, k), t in full.groupby(["set", "action", "k"]):
        best = t.loc[t["red_D"].idxmax()]
        r = t[t["method"] == "random"].iloc[0]
        rows.append({"set": key, "action": action, "k": k, "winner": best["method"], "red_D": round(best["red_D"], 4),
                     "random": f"{r['red_D']:.4f}±{r['red_D_ci95']:.4f}",
                     "degree>rand+CI": bool(t[t["method"] == "degree"]["beats_random_ci"].iloc[0]),
                     "pagerank>rand+CI": bool(t[t["method"] == "pagerank"]["beats_random_ci"].iloc[0])})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== channels (k = 50): reduction from friend counts alone, injection alone, both; injection share of the sum ===")
    print(ch[ch["k"] == 50].round(4).to_string(index=False))


def plot6(full):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for i, key in enumerate(SETS):
        for j, action in enumerate(ACTIONS):
            ax = axes[i, j]
            for m in METHODS:
                t = full[(full["set"] == key) & (full["action"] == action) & (full["method"] == m)].sort_values("k")
                ax.plot(t["k"], 100 * t["red_D"], "-o", ms=4, lw=2, color=COLORS[m], label=m)
                if m == "random":
                    ax.fill_between(t["k"], 100 * (t["red_D"] - t["red_D_ci95"]), 100 * (t["red_D"] + t["red_D_ci95"]), color=COLORS[m], alpha=0.25)
            ax.set_title(f"{sair.set_title(key)} - {action}", fontsize=10)
            ax.set_xscale("log")
            ax.set_xticks(KS, [str(k) for k in KS])
            ax.grid(alpha=0.25)
            if i == 2:
                ax.set_xlabel("k (accounts acted on)")
            if j == 0:
                ax.set_ylabel("rumour exposure reduction 1 - D/D0 (%)")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Plot 6 - PHEME: rumour reduction vs k per selection method (random: mean of 15 draws, band = 95% CI of the mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase8_plot6_pheme_reduction.png"), dpi=130)
    plt.close(fig)


def plot7(full, k=50):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    w = 0.25
    cols = {"block": "#0072B2", "recruit": "#E69F00", "hybrid": "#009E73"}
    for ax, key in zip(axes, SETS):
        for a, action in enumerate(ACTIONS):
            t = full[(full["set"] == key) & (full["action"] == action) & (full["k"] == k)].set_index("method").loc[METHODS]
            ax.bar(np.arange(4) + (a - 1) * w, 100 * t["red_D"], w, color=cols[action], label=action,
                   yerr=[100 * (c if m == "random" else 0) for m, c in zip(METHODS, t["red_D_ci95"])], capsize=3)
        ax.set_xticks(range(4), METHODS)
        ax.set_ylabel("rumour exposure reduction 1 - D/D0 (%)")
        ax.set_title(sair.set_title(key))
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.suptitle(f"Plot 7 - PHEME: reduction at k = {k} accounts (error bar = 95% CI of the random draws)")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase8_plot7_pheme_bars.png"), dpi=130)
    plt.close(fig)


# ================================================================ Part B: agent-based SAIR-H+ on BA graphs

@njit(cache=True)
def _abm(indptr, indices, deg, state, beta, alpha, gamma, eps, kappa, eta_i, eta_a, n_aseed, a_time, t_end, p_max, dt_cap, seed):
    """Synchronous stochastic SAIR-H+ on a graph. States 0 S, 1 I, 2 A, 3 R, 4 B (blocked). Hazards of an S node (counts of
    I / A neighbours nI, nA, degree d, P1 = nI/(nI + nA) over sided neighbours; with none, the hazards reduce to eta):
        to I: beta * Drif1 * nI / d + eta_I      to A: alpha * (1 - Drif1) * nA / d + eta_A,   Drif1 = sigmoid((2 P1 - 1) kappa)
    I -> R at gamma, A -> R at eps. Step dt = min(dt_cap, p_max / largest hazard), clipped to the next whole hour (records) and
    to a_time (A seeds); a node moves with probability 1 - exp(-hazard * dt) <= p_max. Returns hourly (I, A, C_I, C_A) counts,
    the peak I count and the final C_I, C_A."""
    np.random.seed(seed)
    N = len(state)
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
    t, next_rec, seeded, peak = 0.0, 0, n_aseed == 0, cntI
    while True:
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
    return rec, peak, cI, cA


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


def ba_run(gseed, run, rates, kappa, method, action, k):
    """One stochastic run. Targets (block: state B; recruit: state A from t = 0) are chosen first, then the seeds among the rest
    (the same random permutation per (graph, run) for every configuration, so configurations are paired)."""
    if gseed not in _BA_CACHE:
        _BA_CACHE[gseed] = ba_graph(gseed)
    indptr, indices, deg, cen = _BA_CACHE[gseed]
    rng = np.random.default_rng(1_000_003 * gseed + run)
    perm = rng.permutation(BA_N)
    state = np.zeros(BA_N, np.int8)
    if k > 0:
        kb = {"block": k, "recruit": 0, "hybrid": (k + 1) // 2}[action]
        kr = {"block": 0, "recruit": k, "hybrid": k // 2}[action]
        if method == "random":
            targets = rng.choice(BA_N, k, replace=False)               # random draw per run; hybrid: first half blocked
        else:
            targets = np.lexsort((np.arange(BA_N), -cen[method]))[:k]    # the top-k, ranks 1..kb blocked, the next kr recruited
        state[targets[:kb]] = 4
        state[targets[kb:kb + kr]] = 2
    free = perm[state[perm] == 0]
    state[free[:int(round(BA_I_SEED * BA_N))]] = 1
    beta, alpha, gamma, eps, eta_i, eta_a = rates
    return _abm(indptr, indices, deg, state, beta, alpha, gamma, eps, kappa, eta_i, eta_a, int(round(BA_A_SEED * BA_N)),
                BA_A_TIME, BA_T_END, BA_P_MAX, BA_DT_CAP, int(1_000_003 * gseed + run) % (2 ** 31))


def ba_job(args):
    """One graph x kappa mode x (method, action) with every k (or the no-targeting baseline): BA_RUNS runs each."""
    gseed, kmode, method, action, kpcts = args
    main, fits = main_fit()
    p = dict(zip(sair.NAMES[main], fits["charliehebdo"]))
    kappa = p["kappa"] if kmode == "fitted" else 0.0
    rates = (p["beta"], p["alpha"], p["gamma"], p["eps"], p["eta_I"], p["eta_A"])
    rows = []
    for kp in kpcts:
        k = int(round(kp * BA_N))
        res = [ba_run(gseed, r, rates, kappa, method, action, k) for r in range(BA_RUNS)]
        curves = np.mean([r[0] for r in res], axis=0) / BA_N
        rows.append({"gseed": gseed, "kappa_mode": kmode, "method": method, "action": action, "kpct": kp, "k": k,
                     "C_I_final": float(np.mean([r[2] for r in res]) / BA_N), "C_A_final": float(np.mean([r[3] for r in res]) / BA_N),
                     "peak_I": float(np.mean([r[1] for r in res]) / BA_N),
                     "C_I_runs": [float(r[2] / BA_N) for r in res],
                     "curve_I": curves[:, 0].tolist(), "curve_A": curves[:, 1].tolist(),
                     "curve_CI": curves[:, 2].tolist(), "curve_CA": curves[:, 3].tolist()})
    return rows


def ba_jobs():
    jobs = []
    for g in BA_SEEDS:
        for km in ("fitted", "zero"):
            jobs.append((f"ba|{g}|{km}|none", (g, km, "none", "block", [0.0])))
            for m in METHODS:
                for a in ACTIONS:
                    jobs.append((f"ba|{g}|{km}|{m}|{a}", (g, km, m, a, BA_KPCT)))
    return jobs


def ba_tables(done):
    raw = pd.DataFrame([r for rows in done.values() for r in rows])
    base = raw[raw["method"] == "none"].set_index(["gseed", "kappa_mode"])
    rows = []
    for (km, m, a, kp), t in raw[raw["method"] != "none"].groupby(["kappa_mode", "method", "action", "kpct"]):
        t = t.set_index("gseed").sort_index()
        b = base.xs(km, level="kappa_mode").sort_index()
        red_f = 1 - t["C_I_final"] / b["C_I_final"]
        red_p = 1 - t["peak_I"] / b["peak_I"]
        rows.append({"kappa_mode": km, "method": m, "action": a, "kpct": kp, "k": int(t["k"].iloc[0]), "n_graphs": len(t), "n_runs": len(t) * BA_RUNS,
                     "C_I_final": t["C_I_final"].mean(), "C_I_final_ci95": ci95(t["C_I_final"]), "peak_I": t["peak_I"].mean(),
                     "peak_I_ci95": ci95(t["peak_I"]), "C_A_final": t["C_A_final"].mean(),
                     "baseline_C_I_final": b["C_I_final"].mean(), "baseline_peak_I": b["peak_I"].mean(),
                     "red_final": red_f.mean(), "red_final_ci95": ci95(red_f), "red_peak": red_p.mean(), "red_peak_ci95": ci95(red_p)})
    out = pd.DataFrame(rows)
    # conformity feedback: benefit with fitted kappa vs kappa = 0 (paired over graphs)
    piv = out.pivot_table(index=["method", "action", "kpct"], columns="kappa_mode", values="red_final")
    piv["conformity_share"] = np.where(piv["fitted"].abs() > 1e-9, (piv["fitted"] - piv["zero"]) / piv["fitted"], np.nan)
    out = out.merge(piv[["conformity_share"]].reset_index(), on=["method", "action", "kpct"])
    # validation of the agent model against the mean-field main model on CH
    val = []
    base_f = raw[(raw["method"] == "none") & (raw["kappa_mode"] == "fitted")]
    abm_I = np.mean([r for r in base_f["curve_I"]], axis=0)
    abm_CI = np.mean([r for r in base_f["curve_CI"]], axis=0)
    abm_CA = np.mean([r for r in base_f["curve_CA"]], axis=0)
    main, fits = main_fit()
    sc = sair.get_context("charliehebdo")
    mf = sair.simulate(sc, main, fits["charliehebdo"], record=True)
    hours = np.arange(len(abm_I), dtype=float)
    mf_I, mf_CI, mf_CA = (np.interp(hours, mf["t"], mf[c]) for c in ("I", "C_I", "C_A"))
    val.append({"quantity": "peak I (fraction of N)", "agent": abm_I.max(), "mean_field": mf_I.max()})
    val.append({"quantity": "time of peak I (h)", "agent": float(hours[abm_I.argmax()]), "mean_field": float(hours[mf_I.argmax()])})
    val.append({"quantity": "final C_I", "agent": abm_CI[-1], "mean_field": mf_CI[-1]})
    val.append({"quantity": "final C_A", "agent": abm_CA[-1], "mean_field": mf_CA[-1]})
    val.append({"quantity": "time C_I reaches 50% of its final value (h)", "agent": float(np.interp(0.5, abm_CI / abm_CI[-1], hours)),
                "mean_field": float(np.interp(0.5, mf_CI / mf_CI[-1], hours))})
    val.append({"quantity": "corr(I(t)) agent vs mean-field", "agent": float(np.corrcoef(abm_I, mf_I)[0, 1]), "mean_field": 1.0})
    val.append({"quantity": "corr(C_I(t)) agent vs mean-field", "agent": float(np.corrcoef(abm_CI, mf_CI)[0, 1]), "mean_field": 1.0})
    return out, pd.DataFrame(val), (hours, abm_I, abm_CI, abm_CA, mf_I, mf_CI, mf_CA, raw)


def ba_report(out, val):
    pd.set_option("display.width", 220)
    print("\n=== BA: agent-model validation against the mean-field main model on CH (no targeting) ===")
    print(val.round(4).to_string(index=False))
    for km in ("fitted", "zero"):
        print(f"\n=== BA ({km} kappa): reduction of the final rumour size 1 - C_I/C_I0 (mean over 10 graphs ± 95% CI) ===")
        t = out[out["kappa_mode"] == km]
        for a in ACTIONS:
            p = t[t["action"] == a].pivot(index="kpct", columns="method", values="red_final")[METHODS]
            c = t[t["action"] == a].pivot(index="kpct", columns="method", values="red_final_ci95")[METHODS]
            print(f"--- {a}\n" + pd.DataFrame({m: [f"{100 * p.loc[i, m]:6.1f} ± {100 * c.loc[i, m]:4.1f}" for i in p.index] for m in METHODS}, index=[f"{100 * i:g}%" for i in p.index]).to_string())
    print(f"\n=== BA: share of the benefit attributable to conformity (fitted vs kappa = 0), final size ===")
    print(out[out["kappa_mode"] == "fitted"].pivot_table(index=["action", "kpct"], columns="method", values="conformity_share")[METHODS].round(3).to_string())


def plot8(out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, a in zip(axes, ACTIONS):
        for m in METHODS:
            for km, ls in (("fitted", "-"), ("zero", "--")):
                t = out[(out["action"] == a) & (out["method"] == m) & (out["kappa_mode"] == km)].sort_values("kpct")
                ax.errorbar(100 * t["kpct"], 100 * t["red_final"], yerr=100 * t["red_final_ci95"], color=COLORS[m], ls=ls, marker="o", ms=4, lw=1.8,
                            capsize=2, label=f"{m}, {'κ fitted' if km == 'fitted' else 'κ = 0'}")
        ax.set_xscale("log")
        ax.set_xticks([100 * k for k in BA_KPCT], [f"{100 * k:g}" for k in BA_KPCT])
        ax.set_xlabel("k (% of N acted on)")
        ax.set_title(a)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("reduction of the final rumour size (%)")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Plot 8 - BA network (N = 8251, m = 2; 10 graphs x 20 runs): final-size reduction vs k per method, fitted κ (solid) vs κ = 0 (dashed); bars = 95% CI over graphs")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase8_plot8_ba_reduction.png"), dpi=130)
    plt.close(fig)


def plot8b(curves):
    hours, abm_I, abm_CI, abm_CA, mf_I, mf_CI, mf_CA, raw = curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hours, 100 * abm_I, color="#0072B2", lw=2, label="agent model on BA (mean of 200 runs)")
    axes[0].plot(hours, 100 * mf_I, color="#d6336c", lw=2, label="mean-field main model, CH fit")
    axes[0].set_ylabel("I (% of N)")
    axes[1].plot(hours, 100 * abm_CI, color="#0072B2", lw=2, label="agent: C_I")
    axes[1].plot(hours, 100 * mf_CI, color="#d6336c", lw=2, label="mean-field: C_I")
    axes[1].plot(hours, 100 * abm_CA, color="#0072B2", lw=2, ls="--", label="agent: C_A")
    axes[1].plot(hours, 100 * mf_CA, color="#d6336c", lw=2, ls="--", label="mean-field: C_A")
    axes[1].set_ylabel("cumulative entries (% of N)")
    for ax in axes:
        ax.set_xlabel("time (h)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Plot 8b - validation: agent model (no targeting) vs mean-field main model, Charlie Hebdo rates")
    fig.tight_layout()
    fig.savefig(os.path.join(dp.RESULTS, "phase8_plot8b_ba_validation.png"), dpi=130)
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
        # k = 0 / empty interventions reproduce the baseline exactly
        e = np.array([], int)
        fa, ia = friend_arrays(sc, e, e), inject_arrays(sc, e, e)
        z = outcome(target_ctx(sc, fa, ia, "both"), main, x)
        assert z == base, f"{key}: empty intervention differs from the baseline"
        bp, rp = pools(sc)
        users = np.asarray(sc["users"])
        worst = 0.0
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
        # states sum to one for a heavy intervention; friend counts never negative
        b, rc = choose(sc, cen, "pagerank", "hybrid", 100, np.random.default_rng(1))
        fa, ia = friend_arrays(sc, b, rc), inject_arrays(sc, b, rc)
        assert (fa[0] >= -1e-6).all() and (fa[1] >= -1e-6).all()
        res = sair.simulate(target_ctx(sc, fa, ia, "both"), main, x, record=True)
        worst = float(np.abs(res[["S", "I", "A", "R"]].sum(axis=1) - 1).max())
        assert worst <= sair.INV_TOL and res[["S", "I", "A", "R"]].to_numpy().min() >= -sair.INV_TOL
        # the injection channel only acts when a blocked account authored a supporting source tweet
        none_inj = [kk for kk, v in n_inj.items() if v > 0]
        t = full[(full["set"] == key)]
        assert (t["n_injections_removed"] >= 0).all()
        d = t[(t["action"] == "block") & (t["method"].isin(["degree", "pagerank", "betweenness"]))]
        print(f"{key:22s} OK  empty intervention == baseline == Phase 5 curves; same k for every method; targets in pools; max|sum-1|={worst:.1e}; "
              f"supporting-source injections removed by block (k=100): degree {n_inj[('block', 100, 'degree')]}, PageRank {n_inj[('block', 100, 'pagerank')]}, "
              f"betweenness {n_inj[('block', 100, 'betweenness')]}")
    assert set(full["k"]) == set(KS) and set(full["method"]) == set(METHODS)
    assert (full[full["method"] == "random"]["n_draws"] == N_RANDOM).all()
    both = ch["red_both"]
    print(f"channel table rows {len(ch)}; Part A checks passed.")


def check_b():
    print("\n--- Phase 8 check, part B (BA agent model) ---")
    # step probability bound and exact seeds / counts on one run
    main, fits = main_fit()
    p = dict(zip(sair.NAMES[main], fits["charliehebdo"]))
    rates = (p["beta"], p["alpha"], p["gamma"], p["eps"], p["eta_I"], p["eta_A"])
    rec, peak, cI, cA = ba_run(42, 0, rates, p["kappa"], "none", "block", 0)
    assert rec[0, 0] == round(BA_I_SEED * BA_N) and rec[0, 1] == 0, "initial seeds wrong"
    assert (np.diff(rec[:, 2]) >= 0).all() and (np.diff(rec[:, 3]) >= 0).all(), "cumulative counts decrease"
    assert (rec[:, 0] + rec[:, 1] <= BA_N).all() and peak >= rec[:, 0].max()
    assert rec[int(BA_A_TIME) - 1, 1] == 0 and rec[int(BA_A_TIME), 1] > 0, "A seeds must appear at the first denial time"
    # recruit / block put the targets in the right state
    for action in ("block", "recruit"):
        rec2, *_ = ba_run(42, 0, rates, p["kappa"], "degree", action, 83)
        if action == "recruit":
            assert rec2[0, 1] == 83, "recruited nodes must start in A"
        else:
            assert rec2[0, 0] == round(BA_I_SEED * BA_N), "blocked nodes must not become I seeds"
    # reproducible
    assert np.array_equal(ba_run(42, 3, rates, p["kappa"], "pagerank", "hybrid", 41)[0], ba_run(42, 3, rates, p["kappa"], "pagerank", "hybrid", 41)[0])
    out = pd.read_csv(OUT["ba"])
    assert set(out["n_graphs"]) == {len(BA_SEEDS)} and set(out["n_runs"]) == {len(BA_SEEDS) * BA_RUNS}
    assert set(out["kpct"]) == set(BA_KPCT)
    print("seeds, state initialisation, monotone cumulative counts, A seeds at 25 h, reproducibility: OK")
    print("Part B checks passed.")


# ================================================================ main

if __name__ == "__main__":
    part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "AB"
    if "--check-only" not in sys.argv:
        if "--fresh" in sys.argv:
            for c in ((CKPT_A,) if "A" in part else ()) + ((CKPT_B,) if "B" in part else ()):
                if os.path.exists(c):
                    os.remove(c)
        for key in SETS:
            sair.get_context(key)
        with ProcessPoolExecutor(max_workers=sair.N_WORKERS) as pool:
            if "A" in part:
                done = run_cached(pool, pheme_job, [(f"A|{key}|{a}|{k}", (key, a, k)) for key in SETS for a in ACTIONS for k in KS], CKPT_A, "phase 8 part A")
                full, ch = pheme_tables(done)
                full.to_csv(OUT["pheme"], index=False)
                ch.to_csv(OUT["channels"], index=False)
                pheme_report(full, ch)
                plot6(full)
                plot7(full)
            if "B" in part:
                list(pool.map(ba_graph, BA_SEEDS))                  # build and cache the 10 graphs and their centralities
                done = run_cached(pool, ba_job, ba_jobs(), CKPT_B, "phase 8 part B")
                out, val, curves = ba_tables(done)
                out.to_csv(OUT["ba"], index=False)
                val.to_csv(OUT["ba_validation"], index=False)
                ba_report(out, val)
                plot8(out)
                plot8b(curves)
    if "--check" in sys.argv or "--check-only" in sys.argv:
        if "A" in part:
            check_a()
        if "B" in part:
            check_b()
