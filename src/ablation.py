"""Phase 6 - ablation of the main SAIR model (docs/SAIR_paper.md §6, paper Fig. 6-8).

Main model = Phase 5's (d) or (d+), whichever has the lower AIC on Charlie Hebdo (sair.main_model, D61).
Knock-outs (fitted parameters held fixed, as in the paper's Fig. 7), each changing exactly one input:
    no_emotion       S_emo = 1           (User_att = S_emo * Pop -> Pop)
    no_popularity    Pop = 1             (in User_att; the half-life w drops out)
    no_content_fit   Info_match = 1      (enters through r = 1 - mpr_norm, mpr = 0.5 pr_norm + 0.5 Info_match)
    no_connectivity  co_ij = 1           (in hnode / lnode)
    no_friend_power  fp_j = 1            (in hnode / lnode)
    no_network       P1 = the model's global I/(I + A) (as model e)
    no_injection     theta = 0
    no_external      eta_I = eta_A = 0   (only if the main model is d+)
hnode / lnode are rebuilt from per-friend-edge components (co_tag, co_follow, fp, r_j), which reproduce the
Phase 4 values exactly. The Phase 4 regression (coefficients, scaler, reference eta) stays fixed.
Refit importance: each knock-out model is refit (warm start + 9 random starts); dAIC = AIC_knockout - AIC_full
(positive = the factor is needed).
Paper Fig. 8 test: mean Drif1 of strong- vs weak-layer users over time; Drif1 vs group Gact and Gpre.
Paper Fig. 6 test (Charlie Hebdo): the Phase 4 regression refit per 10-hour entry window.
If the main model uses a uniform Mult_eff, the rank-based knock-outs have no effect by construction, so the full
knock-out set is ALSO run on (d+) (ranked Mult_eff) as the paper-faithful ablation (D67).

Usage:
    python src/ablation.py           run, save tables and plots
    python src/ablation.py --check   run, then the Phase 6 checks
    python src/ablation.py --check-only
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data_prep as dp
import drive as drv
import features as ft
import network as nw
import sair

SETS = sair.SETS
KNOCKOUTS = ["no_emotion", "no_popularity", "no_content_fit", "no_connectivity", "no_friend_power",
             "no_network", "no_injection", "no_external"]
KO_LABEL = {"no_emotion": "no emotion (S_emo = 1)", "no_popularity": "no popularity (Pop = 1)",
            "no_content_fit": "no content fit (Info_match = 1)", "no_connectivity": "no connectivity (co = 1)",
            "no_friend_power": "no friend power (fp = 1)", "no_network": "no network (global P1)",
            "no_injection": "no source injection (theta = 0)", "no_external": "no external influence (eta = 0)"}
KO_COLORS = dict(zip(KNOCKOUTS, ["#eb6834", "#eda100", "#7a5af5", "#2a78d6", "#0b4f9c", "#d6336c", "#9b9a94", "#1baf7a"]))
# what each knock-out may change (checked): context fields and fixed parameters
KO_CHANGES = {"no_emotion": {"user_trait", "rank_table"}, "no_popularity": {"pop_one"},
              "no_content_fit": {"lnode_inf", "hnode_inf", "rank_table"},
              "no_connectivity": {"lnode_inf", "hnode_inf", "rank_table"},
              "no_friend_power": {"lnode_inf", "hnode_inf", "rank_table"},
              "no_network": {"p1_mode"}, "no_injection": set(), "no_external": set()}
N_REFIT_RANDOM = 9                           # + 1 warm start from the full fit
RANK_ONLY = {"no_emotion", "no_popularity", "no_content_fit", "no_connectivity", "no_friend_power"}  # act via Mult_eff ranks
FIG6_WINDOW_H, FIG6_MIN_A, FIG6_MIN_SIDED = 10, 30, 200
KAPPA_DIAG = 2.0                             # Phase 4 default kappa, a diagnostic for the Fig. 8 test


# ---------------------------------------------------------------- knock-out inputs

def edge_table(key):
    """Per friend edge (i, j): co_tag, co_follow, fp_j (log1p of replies + retweets j received before t_i,
    excluding i's own) and whether j's layer >= i's layer. Built once and cached in data/processed/."""
    path = os.path.join(dp.PROCESSED, f"ablation_edges_{key}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"user_id": str, "friend_id": str})
    tweets, users, curves, G, feats, net, stories = drv.load(key)
    t_ref, _ = ft.reference_times(users, curves)
    _, tags, _ = ft.text_features(tweets, users, t_ref, ft.load_lexicon())
    friends, follow, _ = ft.friend_sets(G, tweets, users, t_ref)
    by_target, by_pair = ft.received_events(tweets)
    layer = net["layer_code"].to_dict()
    rows = []
    for i in users.index:                                  # same loop and order as drive.node_blocks
        t = t_ref[i].timestamp()
        for j in sorted(friends[i]):
            n = int(np.searchsorted(by_target.get(j, np.empty(0)), t, side="left"))
            n_self = int(np.searchsorted(by_pair.get((j, i), np.empty(0)), t, side="left"))
            rows.append((i, j, ft.jaccard(tags[i], tags[j]), ft.jaccard(follow[i], follow[j]), np.log1p(n - n_self),
                         int(layer[j] >= layer[i])))
    e = pd.DataFrame(rows, columns=["user_id", "friend_id", "co_tag", "co_follow", "fp", "high"])
    e.to_csv(path, index=False)
    return e


def r_values(key, content_fit=True):
    """r per user (Eq. 17, D33); content_fit=False sets Info_match = 1 for everyone."""
    net = pd.read_csv(os.path.join(dp.PROCESSED, f"network_{key}.csv"), dtype={"user_id": str}).set_index("user_id")
    if content_fit:
        return net["r"]
    return nw.efficiency(net["pr_norm"], pd.Series(1.0, index=net.index))[1]


def node_blocks_from_edges(users, e, r, co_one=False, fp_one=False):
    """hnode_inf / lnode_inf = mean of co * fp * (1 - r_j) over same-or-higher / lower-layer friends (0 if none)."""
    co = 1.0 if co_one else e["co_tag"].to_numpy() + e["co_follow"].to_numpy()
    fp = 1.0 if fp_one else e["fp"].to_numpy()
    val = co * fp * (1 - e["friend_id"].map(r).to_numpy())
    pos = pd.Index(users).get_indexer(e["user_id"])
    out = {}
    for name, sel in (("hnode_inf", e["high"].to_numpy() == 1), ("lnode_inf", e["high"].to_numpy() == 0)):
        s = np.bincount(pos[sel], weights=val[sel], minlength=len(users))
        n = np.bincount(pos[sel], minlength=len(users))
        out[name] = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return out


def ko_context(key, ko):
    """Set context with one knock-out applied (ko = 'full' gives the unchanged context)."""
    base = sair.get_context(key)
    sc = dict(base)
    sc["ko"] = ko
    user_trait, static = base["user_trait"].copy(), base["static"].copy()      # static = lnode, hnode, Group_att
    if ko == "no_emotion":
        user_trait = np.ones_like(user_trait)
    elif ko == "no_popularity":
        sc["pop_one"] = True
    elif ko in ("no_content_fit", "no_connectivity", "no_friend_power"):
        blocks = node_blocks_from_edges(base["users"], edge_table(key), r_values(key, ko != "no_content_fit"),
                                        co_one=ko == "no_connectivity", fp_one=ko == "no_friend_power")
        static[:, 0], static[:, 1] = blocks["lnode_inf"], blocks["hnode_inf"]
    elif ko == "no_network":
        sc["p1_mode"] = "global"
    if ko in ("no_emotion", "no_content_fit", "no_connectivity", "no_friend_power"):
        sc["user_trait"], sc["static"] = user_trait, static
        sc["rank_table"] = sair.rank_table_of(base["drive_model"], user_trait, static)
    return sc


def ko_fixed(ko, kind, x_full):
    """Parameters a knock-out fixes (removed from the refit)."""
    p = dict(zip(sair.NAMES[kind], x_full))
    if ko == "no_injection":
        return {"theta": 0.0}
    if ko == "no_external":
        return {"eta_I": 0.0, "eta_A": 0.0}
    if ko == "no_popularity" and "half_life_h" in p:
        return {"half_life_h": p["half_life_h"]}            # Pop = 1: the half-life has no effect
    return {}


def ko_params(ko, kind, x_full):
    """Fitted parameters with the knock-out's fixed values applied (for the held-fixed runs)."""
    p = {**dict(zip(sair.NAMES[kind], x_full)), **ko_fixed(ko, kind, x_full)}
    return np.array([p[n] for n in sair.NAMES[kind]])


def knockouts_for(kind):
    return [k for k in KNOCKOUTS if k != "no_external" or "eta_I" in sair.NAMES[kind]]


def no_effect(ko, kind):
    """With uniform Mult_eff = kappa, the rank-based knock-outs cannot change delta (no effect by construction)."""
    return ko in RANK_ONLY and sair.ME_MODE[kind] == "uniform"


def changed_inputs(base, sc, x_full, x_ko, kind):
    """Names of the inputs a knock-out context / parameter vector differs in (for the check)."""
    changed = set()
    for name in ("user_trait", "rank_table"):
        if not np.array_equal(base[name], sc[name]):
            changed.add(name)
    for c, name in enumerate(["lnode_inf", "hnode_inf", "Group_att"]):
        if not np.array_equal(base["static"][:, c], sc["static"][:, c]):
            changed.add(name)
    for name in ("pop_one", "p1_mode"):
        if base[name] != sc[name]:
            changed.add(name)
    for n, a, b in zip(sair.NAMES[kind], x_full, x_ko):
        if a != b:
            changed.add(n)
    return changed


# ---------------------------------------------------------------- runs and metrics

def run_metrics(sc, r):
    i_peak = int(r["I"].to_numpy().argmax())
    return {"peak_I": float(r["I"].iloc[i_peak]), "t_peak_h": float(r["t"].iloc[i_peak]),
            "final_C_I": float(r["C_I"].iloc[-1]), "final_C_A": float(r["C_A"].iloc[-1]),
            "final_C_I_share": float(r["C_I"].iloc[-1] / sc["fin_I"]), "final_C_A_share": float(r["C_A"].iloc[-1] / sc["fin_A"])}


def layer_drif(sc, kind, x, r, kappa=None):
    """Mean Drif1 over not-yet-entered users per layer at every update time, from a recorded run."""
    p = dict(zip(sair.NAMES[kind], x))
    kappa = p["kappa"] if kappa is None else kappa
    pops = sair.pops_for(sc, p)
    net = pd.read_csv(os.path.join(dp.PROCESSED, f"network_{sc['key']}.csv"), dtype={"user_id": str}).set_index("user_id")
    layer = net["layer"].reindex(sc["users"]).to_numpy()
    t = r["t"].to_numpy()
    rows, per_user = [], []
    for k, tu in enumerate(sc["upd"]):
        q = int(np.searchsorted(t, tu - 1e-9))               # the recorded state at the update time
        s = r.iloc[q]
        d1, mask = sair.drif1_vector(sc, k, s["C_I"], s["C_A"], s["I"], s["A"], kappa,
                                     pops[k] if pops is not None else 0.0, kind)
        row = {"t": float(tu)}
        for lay in ("core", "strong", "weak"):
            m = mask & (layer == lay)
            row[f"drif1_{lay}"] = float(d1[m].mean()) if m.any() else np.nan
            row[f"n_{lay}"] = int(m.sum())
        rows.append(row)
        per_user.append(np.where(mask, d1, np.nan))
    with warnings.catch_warnings():                          # users entered at t = 0 have no value: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        user_mean = np.nanmean(np.array(per_user), axis=0)   # per-user mean while not entered
    return pd.DataFrame(rows), user_mean


def group_test(sc, user_mean_drif):
    """Spearman correlation over groups of the group-mean Drif1 with Gact_norm and Gpre (paper Fig. 8)."""
    net = pd.read_csv(os.path.join(dp.PROCESSED, f"network_{sc['key']}.csv"), dtype={"user_id": str}).set_index("user_id")
    net = net.reindex(sc["users"])
    g = pd.DataFrame({"group": net["group"].to_numpy(), "drif1": user_mean_drif, "Gact_norm": net["Gact_norm"].to_numpy(),
                      "Gpre": net["Gpre"].to_numpy()}).dropna(subset=["drif1"])
    gm = g.groupby("group").agg(drif1=("drif1", "mean"), Gact_norm=("Gact_norm", "first"), Gpre=("Gpre", "first"))
    out = {}
    for c in ("Gact_norm", "Gpre"):
        if gm[c].nunique() > 1 and gm["drif1"].nunique() > 1:
            rho, pval = spearmanr(gm[c], gm["drif1"])
        else:
            rho, pval = np.nan, np.nan
        out[f"rho_{c}"], out[f"p_{c}"] = float(rho), float(pval)
    out["groups"] = len(gm)
    out["Gpre_distinct_values"] = int(gm["Gpre"].nunique())
    return out


def fig6_test(key="charliehebdo"):
    """Phase 4 regression refit on users entering in each 10-hour window (>= 30 A users or >= 200 sided users)."""
    b = pd.read_csv(os.path.join(dp.PROCESSED, f"drive_{key}.csv"), dtype={"user_id": str})
    s = b[b["side"].notna()].copy()
    s["window"] = (s["t_ref_h"] // FIG6_WINDOW_H).astype(int)
    rows = []
    for w, d in s.groupby("window"):
        n_a, n = int((d["side"] == "A").sum()), len(d)
        row = {"window_h": f"{w * FIG6_WINDOW_H}-{(w + 1) * FIG6_WINDOW_H}", "n_sided": n, "n_A": n_a,
               "qualifies": n_a >= FIG6_MIN_A or n >= FIG6_MIN_SIDED, "few_A_lt_10": n_a < 10}
        if row["qualifies"] and 0 < n_a < n:
            _, model = drv.fit_model(d[drv.BLOCKS].to_numpy(dtype=float), (d["side"] == "I").astype(int).to_numpy())
            c = dict(zip(drv.BLOCKS, model.coef_[0]))
            own = abs(c["User_att"])
            ext = abs(c["hnode_inf"]) + abs(c["lnode_inf"]) + abs(c["Group_att"])
            row.update({f"gamma_{k}": v for k, v in c.items()})
            row.update({"abs_own_User_att": own, "abs_external_sum": ext, "own_share": own / (own + ext) if own + ext else np.nan,
                        "fitted": True})
        else:
            row["fitted"] = False
            row["note"] = ("one class only" if row["qualifies"] else "below the size rule")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots

def plot_knockouts(runs, kind, name, tag):
    fig, axes = plt.subplots(len(SETS), 2, figsize=(14, 3.9 * len(SETS)), squeeze=False)
    for r_i, key in enumerate(SETS):
        sc = sair.get_context(key)
        for c, (col, ylabel) in enumerate((("I", "I - rumour spreaders (fraction of N)"), ("C_I", "C_I / final"))):
            ax = axes[r_i, c]
            scale = sc["fin_I"] if col == "C_I" else 1.0
            for ko, r in runs[key].items():
                ax.plot(r["t"], r[col] / scale, color="#0b0b0b" if ko == "full" else KO_COLORS[ko],
                        lw=2.4 if ko == "full" else 1.4, ls="-" if ko == "full" else "--",
                        label="full model" if ko == "full" else KO_LABEL[ko])
            if col == "C_I":
                ax.plot(sc["grid"], sc["C_I"] / scale, color="#0b0b0b", lw=0, marker="o", ms=2, label="real data")
            ax.legend(frameon=False, fontsize=7)
            dp.style(ax, f"{sair.set_title(key)}: {col}", ylabel, xlabel="Hours since first post" if r_i == len(SETS) - 1 else "")
    fig.suptitle(f"Plot 3{tag} - knock-outs of {sair.MODEL_LABEL[kind]} (fitted parameters held fixed)", fontsize=13, x=0.01, ha="left")
    return dp.save(fig, name)


def plot_importance(abl, imp, kind, name, tag):
    fig, axes = plt.subplots(2, len(SETS), figsize=(5.4 * len(SETS), 8), squeeze=False)
    for c, key in enumerate(SETS):
        a = abl[abl["set"] == key].set_index("knockout")
        i = imp[imp["set"] == key].set_index("knockout")
        kos = list(a.index)
        y = np.arange(len(kos))
        for row, (vals, xlabel, title) in enumerate(((a["d_peak_I_pct"], "Change in rumour peak vs full model (%)", "held-fixed knock-out"),
                                                    (i.reindex(kos)["dAIC"], "dAIC vs full model (positive = factor needed)", "refit importance"))):
            ax = axes[row, c]
            ax.barh(y, vals.to_numpy(), color=[KO_COLORS[k] for k in kos])
            ax.axvline(0, color="#0b0b0b", lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([KO_LABEL[k] + (" *" if no_effect(k, kind) else "") for k in kos], fontsize=8)
            ax.invert_yaxis()
            dp.style(ax, f"{sair.set_title(key)}: {title}", "", xlabel=xlabel)
    note = "  (* no effect by construction: uniform Mult_eff)" if sair.ME_MODE[kind] == "uniform" else ""
    fig.suptitle(f"Plot 3b{tag} - factor importance in {sair.MODEL_LABEL[kind]}: change in rumour peak (held fixed) "
                 f"and dAIC (refit){note}", fontsize=12, x=0.01, ha="left")
    return dp.save(fig, name)


def plot_layers(layers):
    models = list(dict.fromkeys(layers["model"]))
    fig, axes = plt.subplots(len(models), len(SETS), figsize=(5.4 * len(SETS), 4.2 * len(models)), squeeze=False)
    for r_i, kind in enumerate(models):
        for c, key in enumerate(SETS):
            ax = axes[r_i, c]
            for kap, ls in (("fitted", "-"), ("kappa=2", "--")):
                d = layers[(layers["model"] == kind) & (layers["set"] == key) & (layers["kappa_used"] == kap)]
                for lay, col in (("strong", "#d6336c"), ("weak", "#2a78d6")):
                    ax.plot(d["t"], d[f"drif1_{lay}"], color=col, ls=ls, lw=1.8, label=f"{lay} layer ({kap})")
            ax.legend(frameon=False, fontsize=7)
            dp.style(ax, f"{sair.set_title(key)}, {sair.MODEL_LABEL[kind]}", "Mean Drif1 (not-yet-entered users)",
                     xlabel="Hours since first post" if r_i == len(models) - 1 else "")
    fig.suptitle("Plot 3c - strong vs weak layer push to spread the rumour (paper Fig. 8 test)", fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase6_plot3c_layers.png")


# ---------------------------------------------------------------- main + check

PAPER_MODEL = "sairh_plus"                   # ranked Mult_eff: the paper-faithful ablation (D67)


def ablation_models():
    """(model, role) pairs: the main model, plus (d+) as the paper-faithful ablation if the main model is not (d+)."""
    kind, table = sair.load_main()
    pairs = [(kind, "main")] + ([(PAPER_MODEL, "paper-faithful (d+)")] if kind != PAPER_MODEL else [])
    pairs += [(k, "secondary (main under D66)") for k in sair.SECONDARY if k not in (kind, PAPER_MODEL)]
    return pairs, table


def held_fixed(kind, full):
    """Knock-out runs with the fitted parameters held fixed: metrics and the recorded runs."""
    rows, runs = [], {}
    for key in SETS:
        x = sair.params_of(full[key], kind)
        runs[key] = {}
        for ko in ["full"] + knockouts_for(kind):
            sc = sair.get_context((key, ko))
            r = sair.simulate(sc, kind, x if ko == "full" else ko_params(ko, kind, x), record=True)
            runs[key][ko] = r
            rows.append({"model": kind, "set": key, "knockout": ko, "no_effect_by_construction": no_effect(ko, kind),
                         **run_metrics(sc, r)})
    abl = pd.DataFrame(rows)
    for key in SETS:
        f = abl[(abl["set"] == key) & (abl["knockout"] == "full")].iloc[0]
        m = abl["set"] == key
        for c in ("peak_I", "t_peak_h", "final_C_I", "final_C_A"):
            abl.loc[m, f"d_{c}"] = abl.loc[m, c] - f[c]
        abl.loc[m, "d_peak_I_pct"] = 100 * (abl.loc[m, "peak_I"] / f["peak_I"] - 1)
    return abl, runs


def refit_jobs(kind, full):
    """Warm start from the full fit + N_REFIT_RANDOM random starts per knock-out (skipped if no effect)."""
    jobs = []
    for key in SETS:
        x = sair.params_of(full[key], kind)
        for ko in knockouts_for(kind):
            if no_effect(ko, kind):
                continue
            fixed = ko_fixed(ko, kind, x)
            free = sair.free_names(kind, fixed)
            warm = sair.to_fit(free, [dict(zip(sair.NAMES[kind], x))[n] for n in free])
            basis = sair.fingerprint(x)
            jobs.append({"key": (key, ko), "kind": kind, "u0": warm, "tag": (key, ko, kind), "fixed": fixed, "basis": basis})
            jobs += sair.start_jobs((key, ko), kind, (key, ko, kind), n=N_REFIT_RANDOM, fixed=fixed, basis=basis)
    return jobs


def importance(kind, full, res):
    rows = []
    for key in SETS:
        for ko in knockouts_for(kind):
            if no_effect(ko, kind):
                rows.append({"model": kind, "set": key, "knockout": ko, "k": int(full[key]["k"]), "k_full": int(full[key]["k"]),
                             "dAIC": 0.0, "dBIC": 0.0, "note": "no effect by construction (uniform Mult_eff); not refit"})
                continue
            row = sair.fit_row((key, ko), kind, res[(key, ko, kind)],
                               fixed=ko_fixed(ko, kind, sair.params_of(full[key], kind)))
            rows.append({"model": kind, "set": key, "knockout": ko, "k": row["k"], "k_full": int(full[key]["k"]),
                         "SSE": row["SSE"], "AIC": row["AIC"], "AIC_full": full[key]["AIC"], "dAIC": row["AIC"] - full[key]["AIC"],
                         "dBIC": row["BIC"] - full[key]["BIC"], "rmse_mean": row["rmse_mean"],
                         "rmse_mean_full": full[key]["rmse_mean"], "bound_flags": row["bound_flags"],
                         "starts_converged": row["starts_converged"], "starts": row["starts"],
                         **{f"refit_{n}": row[n] for n in sair.NAMES[kind]}})
    return pd.DataFrame(rows)


def layer_tests(kind, full, runs):
    lay_rows, grp_rows = [], []
    for key in SETS:
        sc = sair.get_context(key)
        x = sair.params_of(full[key], kind)
        for kap_label, kap in (("fitted", None), ("kappa=2", KAPPA_DIAG)):
            if kap is None:
                r = runs[key]["full"]
            else:
                xp = dict(zip(sair.NAMES[kind], x))
                xp["kappa"] = kap
                r = sair.simulate(sc, kind, [xp[n] for n in sair.NAMES[kind]], record=True)
            lay, per_user = layer_drif(sc, kind, x, r, kap)
            lay.insert(0, "kappa_used", kap_label)
            lay.insert(0, "set", key)
            lay.insert(0, "model", kind)
            lay_rows.append(lay)
            diff = lay["drif1_strong"] - lay["drif1_weak"]
            grp_rows.append({"model": kind, "set": key, "kappa_used": kap_label,
                             "kappa": float(dict(zip(sair.NAMES[kind], x))["kappa"] if kap is None else kap),
                             "mean_strong_minus_weak": float(diff.mean()), "share_bins_strong_higher": float((diff > 0).mean()),
                             "max_abs_diff": float(diff.abs().max()), **group_test(sc, per_user)})
    return pd.concat(lay_rows, ignore_index=True), pd.DataFrame(grp_rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    models, table = ablation_models()
    print("ablation models: " + ", ".join(f"{k} ({role})" for k, role in models))
    for key in SETS:
        edge_table(key)                                       # build the cache before the workers need it
    full = {kind: {key: table[(table["set"] == key) & (table["model"] == kind)].iloc[0] for key in SETS} for kind, _ in models}

    held = {kind: held_fixed(kind, full[kind]) for kind, _ in models}
    jobs = [j for kind, _ in models for j in refit_jobs(kind, full[kind])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=sair.N_WORKERS) as pool:
        res = sair.run_jobs(pool, jobs, "stage 3 importance refits", "stage3_importance")
    print(f"importance refits ({len(jobs)} runs): {time.time() - t0:.0f} s", flush=True)

    abl = pd.concat([held[kind][0].assign(role=role) for kind, role in models], ignore_index=True)
    imp = pd.concat([importance(kind, full[kind], res).assign(role=role) for kind, role in models], ignore_index=True)
    abl.to_csv(os.path.join(dp.RESULTS, "phase6_ablation.csv"), index=False)
    imp.to_csv(os.path.join(dp.RESULTS, "phase6_importance.csv"), index=False)

    tests = [layer_tests(kind, full[kind], held[kind][1]) for kind, _ in models]
    layers = pd.concat([t[0] for t in tests], ignore_index=True)
    fig8 = pd.concat([t[1] for t in tests], ignore_index=True)
    layers.to_csv(os.path.join(dp.RESULTS, "phase6_layers.csv"), index=False)
    fig8.to_csv(os.path.join(dp.RESULTS, "phase6_fig8_test.csv"), index=False)
    fig6 = fig6_test()
    fig6.to_csv(os.path.join(dp.RESULTS, "phase6_fig6_test.csv"), index=False)

    for kind, role in models:
        if role.startswith("secondary"):
            continue                                          # tables only for the secondary model
        sfx, tag = ("", "") if role == "main" else ("_paper", " (paper-faithful)")
        plot_knockouts(held[kind][1], kind, f"phase6_plot3_knockouts{sfx}.png", tag)
        plot_importance(abl[(abl["model"] == kind) & (abl["knockout"] != "full")], imp[imp["model"] == kind], kind,
                        f"phase6_plot3b_importance{sfx}.png", tag)
    plot_layers(layers)

    print("\n=== Knock-outs (fitted parameters held fixed) ===")
    print(abl.round(4).to_string(index=False))
    print("\n=== Refit importance (dAIC > 0: the factor is needed) ===")
    print(imp.drop(columns=[c for c in imp if c.startswith("refit_")]).round(4).to_string(index=False))
    print("\n=== Paper Fig. 8 test: strong vs weak layer Drif1, group correlations ===")
    print(fig8.round(4).to_string(index=False))
    print("\n=== Paper Fig. 6 test (Charlie Hebdo, 10-h entry windows) ===")
    print(fig6.round(4).to_string(index=False))


def check():
    print("\n--- Phase 6 check ---")
    models, table = ablation_models()
    abl = pd.read_csv(os.path.join(dp.RESULTS, "phase6_ablation.csv"))
    for key in SETS:
        base = sair.get_context(key)
        # rebuilt hnode / lnode reproduce Phase 4 (to rounding)
        blocks = node_blocks_from_edges(base["users"], edge_table(key), r_values(key))
        assert np.allclose(blocks["lnode_inf"], base["static"][:, 0], rtol=0, atol=1e-12)
        assert np.allclose(blocks["hnode_inf"], base["static"][:, 1], rtol=0, atol=1e-12)
        full_sc = sair.get_context((key, "full"))
        for kind, role in models:
            x = sair.params_of(table[(table["set"] == key) & (table["model"] == kind)].iloc[0], kind)
            # the full model through the knock-out path reproduces the Phase 5 fit exactly
            assert not changed_inputs(base, full_sc, x, x, kind)
            a = sair.simulate(base, kind, x, record=True)
            b = sair.simulate(full_sc, kind, x, record=True)
            assert a.equals(b), f"{key}/{kind}: the full model does not reproduce the Phase 5 run"
            sse = sair.metrics(*sair.residuals(full_sc, kind, x), 0)["SSE"]
            sse_saved = table[(table["set"] == key) & (table["model"] == kind)].iloc[0]["SSE"]
            assert abs(sse - sse_saved) <= 1e-9 * max(1.0, sse_saved), f"{key}/{kind}: SSE {sse} vs saved {sse_saved}"
            worst, none = 0.0, []
            for ko in knockouts_for(kind):
                sc = sair.get_context((key, ko))
                xk = ko_params(ko, kind, x)
                changed = changed_inputs(base, sc, x, xk, kind)
                allowed = KO_CHANGES[ko] | set(ko_fixed(ko, kind, x))
                assert changed <= allowed, f"{key}/{kind}/{ko}: changed {changed}, allowed {allowed}"
                if not changed:
                    none.append(ko)
                r = sair.simulate(sc, kind, xk, record=True)
                if no_effect(ko, kind):
                    assert r.equals(a), f"{key}/{kind}/{ko}: should have no effect with uniform Mult_eff"
                s = r[["S", "I", "A", "R"]].to_numpy()
                worst = max(worst, float(np.abs(s.sum(axis=1) - 1).max()))
                assert worst <= sair.INV_TOL and s.min() >= -sair.INV_TOL, f"{key}/{kind}/{ko}: invariant broken"
                assert r["d1"].between(0, 1).all()
            assert len(abl[(abl["set"] == key) & (abl["model"] == kind)]) == 1 + len(knockouts_for(kind))
            print(f"{key:22s} OK  {kind} ({role}): full model == Phase 5 run (SSE {sse:.6f}); "
                  f"{len(knockouts_for(kind))} knock-outs change only their input; max|sum-1|={worst:.1e}"
                  + (f"; unchanged input (already at the knocked-out value): {none}" if none else ""))
    print("All Phase 6 checks passed.")


if __name__ == "__main__":
    if "--check-only" not in sys.argv:
        main()
    if "--check" in sys.argv or "--check-only" in sys.argv:
        check()
