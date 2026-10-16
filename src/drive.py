"""Phase 4 - influence blocks, regression and push to spread (docs/SAIR_paper.md §4, Eq. 12-21).

Per user, from information available before their entry time t_ref:
  User_att  = S_emo * Info_match * Pop(t_ref)       (Eq. 13; degeneracy fallback S_emo * Pop, D40)
  hnode_inf = mean over friends j in the same or a higher layer of co_ij * fp_j * (1 - r_j)   (Eq. 14)
  lnode_inf = the same over friends in lower layers                                          (Eq. 15)
  Group_att = Gact_norm * Gpre                        (Eq. 16; fallback Gact_norm * (Gpre + 0.01), D40)
Regression (Eq. 12): logistic, y = 1 if the first side is I, on the standardised blocks of sided users.
Mult_eff = kappa * min-max(eta) in [0, kappa];  P1 = share of friends in I among friends who took a side;
Drif1 = sigmoid((2 P1 - 1) * Mult_eff), Drif2 = 1 - Drif1  (Eq. 18-21).

Phase 5 uses make_context(key) + drif(ctx, t, side) to get P1, Mult_eff and Drif1 at any time t.

Usage:
    python src/drive.py           fit and save
    python src/drive.py --check   fit, save, then run the Phase 4 checks
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import data_prep as dp
import features as ft

SETS = ft.SETS
BLOCKS = ["User_att", "lnode_inf", "hnode_inf", "Group_att"]   # gamma_1..gamma_4 in Eq. 12
KAPPA = 2.0                                  # Mult_eff scale; fitted in Phase 5 (DECISIONS D42)
NEAR_ZERO = 1e-6
DEGENERATE_SHARE = 0.9                       # fallback if > 90% of users are near zero (D40)
GPRE_EPS = 0.01
CV_FOLDS = 5
MAX_CV_GAP = 0.10
SIDE_CODE = {"I": 1, "A": 2}
VARIANTS = ["profile", "follow"]             # group variant: primary profile-KMeans, robustness follow-Louvain
GROUP_SUFFIX = {"profile": "", "follow": "_follow"}
VARIANT_FILE = {"profile": "", "follow": "_followgroups"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def drif1_of(p1, mult_eff):
    """Eq. 18-21: Drif1 = sigmoid((2*P1 - 1) * Mult_eff)."""
    return sigmoid((2 * np.asarray(p1) - 1) * np.asarray(mult_eff))


# ---------------------------------------------------------------- loading

def load(key):
    tweets, users, curves, G = ft.load_set(key)
    feats = pd.read_csv(os.path.join(dp.PROCESSED, f"features_{key}.csv"), dtype={"user_id": str}).set_index("user_id")
    net = pd.read_csv(os.path.join(dp.PROCESSED, f"network_{key}.csv"), dtype={"user_id": str,
                      "first_thread": str}).set_index("user_id")
    stories = pd.read_csv(os.path.join(dp.RESULTS, f"pop_stories_{key}.csv"))
    return tweets, users, curves, G, feats.reindex(users.index), net.reindex(users.index), stories


def near_zero_share(x):
    return float((np.abs(np.asarray(x, dtype=float)) < NEAR_ZERO).mean())


# ---------------------------------------------------------------- blocks

def node_blocks(G, tweets, users, net, t_ref, tags):
    """hnode_inf / lnode_inf: mean of co_ij * fp_j * (1 - r_j) over friends j in the same-or-higher /
    lower layer. co_ij = tag Jaccard + friend-list Jaccard; fp_j = log1p(replies + retweets received by
    j before t_i, excluding user i's own). Empty subset -> 0 and a flag."""
    friends, follow, _ = ft.friend_sets(G, tweets, users, t_ref)
    by_target, by_pair = ft.received_events(tweets)
    layer = net["layer_code"].to_dict()
    r = net["r"].to_dict()
    rows = []
    for i in users.index:
        t = t_ref[i].timestamp()
        high, low = [], []
        for j in sorted(friends[i]):                 # fixed order: float sums reproducible across runs
            n = int(np.searchsorted(by_target.get(j, np.empty(0)), t, side="left"))
            n_self = int(np.searchsorted(by_pair.get((j, i), np.empty(0)), t, side="left"))
            co = ft.jaccard(tags[i], tags[j]) + ft.jaccard(follow[i], follow[j])
            val = co * np.log1p(n - n_self) * (1 - r[j])
            (high if layer[j] >= layer[i] else low).append(val)
        rows.append({"user_id": i, "hnode_inf": float(np.mean(high)) if high else 0.0, "has_high_friends": int(bool(high)),
                     "lnode_inf": float(np.mean(low)) if low else 0.0, "has_low_friends": int(bool(low))})
    return pd.DataFrame(rows).set_index("user_id"), friends


def build_blocks(key, lex, variant="profile"):
    """variant 'profile' = primary profile-KMeans groups; 'follow' = follow-Louvain groups (robustness)."""
    tweets, users, curves, G, feats, net, stories = load(key)
    sfx = GROUP_SUFFIX[variant]
    t_ref, t0 = ft.reference_times(users, curves)
    t_ref_h = ((t_ref - t0) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    pop = ft.pop_at(stories, t_ref_h, curves["t_hours"])

    b = pd.DataFrame(index=users.index)
    b["side"] = users["side"]
    b["first_thread"] = net["first_thread"]
    b["t_ref_h"] = t_ref_h
    b["pop_at_ref"] = pop

    # User_att (Eq. 13); the 90% degeneracy rule is judged only on users with real text AND emotion
    # words, since imputed values would hide the degeneracy (D40)
    paper_user = feats["S_emo"] * feats["Info_match"] * pop
    judged = (feats["has_text"] == 1) & (feats["has_emotion"] == 1)
    user_fallback = near_zero_share(paper_user[judged]) > DEGENERATE_SHARE
    b["user_trait"] = feats["S_emo"] if user_fallback else feats["S_emo"] * feats["Info_match"]
    b["User_att"] = b["user_trait"] * pop

    # Group_att (Eq. 16) with the same rule
    paper_group = net["Gact_norm" + sfx] * net["Gpre" + sfx]
    group_fallback = near_zero_share(paper_group) > DEGENERATE_SHARE
    b["Group_att"] = net["Gact_norm" + sfx] * (net["Gpre" + sfx] + GPRE_EPS) if group_fallback else paper_group

    # hnode / lnode (Eq. 14-15)
    _, tags, _ = ft.text_features(tweets, users, t_ref, lex)
    nodes, friends = node_blocks(G, tweets, users, net, t_ref, tags)
    b = b.join(nodes)

    degeneracy = {"set": key, "variant": variant, "User_att_judged_users": int(judged.sum()),
                  "User_att_paper_near_zero_pct_all": 100 * near_zero_share(paper_user),
                  "User_att_paper_near_zero_pct_judged": 100 * near_zero_share(paper_user[judged]),
                  "User_att_fallback": user_fallback, "User_att_used_near_zero_pct": 100 * near_zero_share(b["User_att"]),
                  "Group_att_paper_near_zero_pct": 100 * near_zero_share(paper_group),
                  "Group_att_fallback": group_fallback, "Group_att_used_near_zero_pct": 100 * near_zero_share(b["Group_att"]),
                  "hnode_inf_near_zero_pct": 100 * near_zero_share(b["hnode_inf"]),
                  "no_high_friends_pct": 100 * (1 - b["has_high_friends"].mean()),
                  "lnode_inf_near_zero_pct": 100 * near_zero_share(b["lnode_inf"]),
                  "no_low_friends_pct": 100 * (1 - b["has_low_friends"].mean())}
    edges = pd.DataFrame([(i, j) for i in users.index for j in sorted(friends[i])], columns=["user_id", "friend_id"])
    return b, degeneracy, edges


# ---------------------------------------------------------------- regression

def fit_model(X, y):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=dp.SEED)
    model.fit(scaler.transform(X), y)
    return scaler, model


def oof_auc(X, y, splits):
    """Pooled out-of-fold AUC. A fold whose training part has one class predicts 0.5 (counted)."""
    pred = np.full(len(y), np.nan)
    single = 0
    for tr, te in splits:
        if np.unique(y[tr]).size < 2:
            pred[te] = 0.5
            single += 1
            continue
        scaler, model = fit_model(X[tr], y[tr])
        pred[te] = model.predict_proba(scaler.transform(X[te]))[:, 1]
    return roc_auc_score(y, pred), single


def regression(key, b):
    sample = b[b["side"].notna()]
    X = sample[BLOCKS].to_numpy(dtype=float)
    y = (sample["side"] == "I").astype(int).to_numpy()
    scaler, model = fit_model(X, y)
    coef = {"gamma0_intercept": float(model.intercept_[0]),
            **{f"gamma{k + 1}_{blk}": float(c) for k, (blk, c) in enumerate(zip(BLOCKS, model.coef_[0]))}}

    strat = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=dp.SEED).split(X, y)
    auc_strat, _ = oof_auc(X, y, strat)
    threads = sample["first_thread"].to_numpy()
    n_groups = min(CV_FOLDS, len(np.unique(threads)))
    auc_thread, single = oof_auc(X, y, GroupKFold(n_groups).split(X, y, threads))
    auc_train = roc_auc_score(y, model.predict_proba(scaler.transform(X))[:, 1])
    cv = {"set": key, "n_sample": len(y), "n_I": int(y.sum()), "n_A": int((y == 0).sum()),
          "threads": int(len(np.unique(threads))), "thread_folds": n_groups, "thread_folds_single_class": single,
          "auc_train": auc_train, "auc_cv_stratified": auc_strat, "auc_cv_thread_heldout": auc_thread,
          "gap": auc_strat - auc_thread, "gap_flag_over_0.10": (auc_strat - auc_thread) > MAX_CV_GAP}
    return scaler, model, coef, cv


def mult_eff(eta, eta_sorted, kappa=KAPPA):
    """Mult_eff = kappa * percentile rank of eta among the regression sample's eta (mid-rank for ties),
    in [0, kappa] (D42). eta_sorted = sorted sample eta."""
    eta_sorted = np.asarray(eta_sorted)
    lo = np.searchsorted(eta_sorted, eta, side="left")
    hi = np.searchsorted(eta_sorted, eta, side="right")
    return kappa * (lo + hi) / (2 * len(eta_sorted))


# ---------------------------------------------------------------- P1 and Drif

def friend_matrix(users, edges):
    idx = {u: k for k, u in enumerate(users.index)}
    rows = edges["user_id"].map(idx).to_numpy()
    cols = edges["friend_id"].map(idx).to_numpy()
    return sparse.csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(len(users), len(users)))


def p1_from_sides(M, side_code):
    """P1_i = friends in I / friends in I or A; users without sided friends get the global I share
    among sided users (0.5 if nobody has taken a side yet) (D43)."""
    n_i = M @ (side_code == 1).astype(float)
    n_a = M @ (side_code == 2).astype(float)
    tot_i, tot_a = (side_code == 1).sum(), (side_code == 2).sum()
    global_share = tot_i / (tot_i + tot_a) if tot_i + tot_a > 0 else 0.5
    has = (n_i + n_a) > 0
    p1 = np.where(has, n_i / np.where(has, n_i + n_a, 1), global_share)
    return p1, ~has


def entry_snapshot(b, M, eta, eta_sorted):
    """P1, Mult_eff and Drif at each user's own t_ref, using friends who took a side strictly before."""
    side = b["side"].map(SIDE_CODE).fillna(0).to_numpy()
    t_enter = np.where(b["side"].notna(), b["t_ref_h"], np.inf)
    order = np.argsort(b["t_ref_h"].to_numpy(), kind="stable")
    p1 = np.empty(len(b))
    fallback = np.zeros(len(b), dtype=bool)
    t_ref = b["t_ref_h"].to_numpy()
    for k in order:
        before = np.where(t_enter < t_ref[k], side, 0)
        row = M.getrow(k)
        n_i = row @ (before == 1).astype(float)
        n_a = row @ (before == 2).astype(float)
        if n_i[0] + n_a[0] > 0:
            p1[k] = n_i[0] / (n_i[0] + n_a[0])
        else:
            tot_i, tot_a = (before == 1).sum(), (before == 2).sum()
            p1[k] = tot_i / (tot_i + tot_a) if tot_i + tot_a > 0 else 0.5
            fallback[k] = True
    m = mult_eff(eta, eta_sorted)
    d1 = drif1_of(p1, m)
    return pd.DataFrame({"P1_entry": p1, "P1_fallback": fallback.astype(int), "Mult_eff": m,
                         "Drif1": d1, "Drif2": 1 - d1}, index=b.index)


def make_context(key, variant="profile"):
    """Everything Phase 5 needs to evaluate drif(t) for this set (and group variant)."""
    tag = key + VARIANT_FILE[variant]
    with open(os.path.join(dp.PROCESSED, f"drive_model_{tag}.json")) as f:
        model = json.load(f)
    b = pd.read_csv(os.path.join(dp.PROCESSED, f"drive_{tag}.csv"), dtype={"user_id": str}).set_index("user_id")
    edges = pd.read_csv(os.path.join(dp.PROCESSED, f"friends_{key}.csv"), dtype={"user_id": str, "friend_id": str})
    curves = pd.read_csv(os.path.join(dp.PROCESSED, f"real_curves_{key}.csv"))
    stories = pd.read_csv(os.path.join(dp.RESULTS, f"pop_stories_{key}.csv"))
    return {"users": b.index, "M": friend_matrix(b, edges), "user_trait": b["user_trait"].to_numpy(),
            "static": b[BLOCKS[1:]].to_numpy(dtype=float), "model": model, "stories": stories,
            "grid": curves["t_hours"].to_numpy(), "t_enter": np.where(b["side"].notna(), b["t_ref_h"], np.inf),
            "side": b["side"].map(SIDE_CODE).fillna(0).to_numpy()}


def drif(ctx, t, side_code=None, kappa=None, half_life_h=ft.POP_HALF_LIFE_H):
    """P1, Mult_eff and Drif1 for every user at time t (hours since the first post).
    side_code: per-user 0 (no side) / 1 (I) / 2 (A) at time t; default = real data (side taken by t).
    User_att uses Pop(t) at the current time; the other blocks are fixed user traits."""
    m = ctx["model"]
    if side_code is None:
        side_code = np.where(ctx["t_enter"] <= t, ctx["side"], 0)
    pop_t = ft.pop_at(ctx["stories"], [t], ctx["grid"], half_life_h)[0]
    X = np.column_stack([ctx["user_trait"] * pop_t, ctx["static"]])
    z = (X - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])
    eta = m["intercept"] + z @ np.array(m["coef"])
    me = mult_eff(eta, m["eta_sorted"], m["kappa"] if kappa is None else kappa)
    p1, _ = p1_from_sides(ctx["M"], side_code)
    return p1, me, drif1_of(p1, me)


# ---------------------------------------------------------------- per set

def within_group_auc(y, score, groups):
    """Pooled AUC over pairs (positive, negative) taken only WITHIN the same group (ties count 1/2).
    Returns AUC, number of groups with both classes, and number of users in them."""
    y, score, groups = np.asarray(y), np.asarray(score, dtype=float), np.asarray(groups)
    good = bad = 0.0
    n_groups = n_users = 0
    for g in np.unique(groups):
        m = groups == g
        pos, neg = score[m & (y == 1)], np.sort(score[m & (y == 0)])
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_groups += 1
        n_users += int(m.sum())
        below = np.searchsorted(neg, pos, side="left")
        ties = np.searchsorted(neg, pos, side="right") - below
        good += below.sum() + 0.5 * ties.sum()
        bad += len(pos) * len(neg)
    return (good / bad if bad else np.nan), n_groups, n_users


def p1_thread_diagnostic(key, b):
    """How much of P1's separation is thread identity? AUC of P1 at entry overall vs within threads
    that contain both I and A users (D44)."""
    s = b[b["side"].notna()]
    y = (s["side"] == "I").astype(int).to_numpy()
    auc_w, n_threads, n_users = within_group_auc(y, s["P1_entry"], s["first_thread"])
    return {"set": key, "auc_P1_overall": roc_auc_score(y, s["P1_entry"]), "threads_total": s["first_thread"].nunique(),
            "threads_with_I_and_A": n_threads, "users_in_those_threads": n_users,
            "auc_P1_within_thread_pooled": auc_w}


def run_set(key, lex, variant="profile"):
    tag = key + VARIANT_FILE[variant]
    b, degeneracy, edges = build_blocks(key, lex, variant)
    scaler, model, coef, cv = regression(key, b)
    eta = model.intercept_[0] + scaler.transform(b[BLOCKS].to_numpy(dtype=float)) @ model.coef_[0]
    sided = b["side"].notna().to_numpy()
    eta_sorted = np.sort(eta[sided])
    b["eta"] = eta
    M = friend_matrix(b, edges)
    b = b.join(entry_snapshot(b, M, eta, eta_sorted))

    model_json = {"set": key, "variant": variant, "blocks": BLOCKS, "coef": model.coef_[0].tolist(),
                  "intercept": float(model.intercept_[0]),
                  "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "kappa": KAPPA,
                  "user_att_fallback": bool(degeneracy["User_att_fallback"]),
                  "group_att_fallback": bool(degeneracy["Group_att_fallback"]), "gpre_eps": GPRE_EPS,
                  "eta_sorted": [round(float(x), 8) for x in eta_sorted]}
    with open(os.path.join(dp.PROCESSED, f"drive_model_{tag}.json"), "w") as f:
        json.dump(model_json, f)
    b.to_csv(os.path.join(dp.PROCESSED, f"drive_{tag}.csv"))
    edges.to_csv(os.path.join(dp.PROCESSED, f"friends_{key}.csv"), index=False)
    snapshot = {"set": key, "variant": variant, "P1_fallback_pct": 100 * b["P1_fallback"].mean(),
                "Mult_eff_mean": b["Mult_eff"].mean(), "Drif1_mean_I": b.loc[b["side"] == "I", "Drif1"].mean(),
                "Drif1_mean_A": b.loc[b["side"] == "A", "Drif1"].mean(),
                "Drif1_mean_S": b.loc[b["side"].isna(), "Drif1"].mean()}
    return b, degeneracy, {"set": key, "variant": variant, **coef}, {"variant": variant, **cv}, snapshot


# ---------------------------------------------------------------- plots

def plot_drif_entry(all_b):
    fig, axes = plt.subplots(1, len(SETS), figsize=(5 * len(SETS), 4.3), squeeze=False)
    bins = np.linspace(0, 1, 26)
    for ax, key in zip(axes[0], SETS):
        b = all_b[key]
        for side, label in (("I", "I users"), ("A", "A users")):
            x = b.loc[b["side"] == side, "Drif1"]
            ax.hist(x, bins=bins, density=True, histtype="step", lw=2, color=dp.COLORS[side],
                    label=f"{label} (n = {len(x):,}, mean {x.mean():.2f})")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        dp.style(ax, dp.EVENT_NAMES.get(key, "Ottawa story"), "Density", xlabel="Drif1 at entry")
    fig.suptitle("Push to spread the rumour (Drif1) at each user's entry, by the side they took",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase4_drif_entry.png")


def plot_drif_vs_p1(all_b):
    fig, axes = plt.subplots(1, len(SETS), figsize=(5 * len(SETS), 4.3), squeeze=False)
    p = np.linspace(0, 1, 101)
    for ax, key in zip(axes[0], SETS):
        b = all_b[key]
        for m, ls in ((0.5, ":"), (1.0, "--"), (KAPPA, "-")):
            ax.plot(p, drif1_of(p, m), color="#b5b4ae", lw=1, ls=ls, label=f"Mult_eff = {m:g}")
        for side in ("I", "A"):
            d = b[b["side"] == side]
            ax.scatter(d["P1_entry"], d["Drif1"], s=10, color=dp.COLORS[side], alpha=0.5, edgecolor="none",
                       label=f"{side} users")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        dp.style(ax, dp.EVENT_NAMES.get(key, "Ottawa story"), "Drif1 at entry", xlabel="P1 at entry (share of sided friends in I)")
    fig.suptitle("Drif1 = sigmoid((2 P1 - 1) Mult_eff): the friends' majority sets the direction, Mult_eff the strength",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase4_drif_vs_p1.png")


# ---------------------------------------------------------------- main + check

def main():
    pd.set_option("display.width", 230)
    lex = ft.load_lexicon()
    all_b, degs, coefs, cvs, snaps, diags = {}, [], [], [], [], []
    for key in SETS:
        for variant in VARIANTS:
            b, deg, coef, cv, snap = run_set(key, lex, variant)
            degs.append(deg)
            coefs.append(coef)
            cvs.append(cv)
            snaps.append(snap)
            if variant == "profile":
                all_b[key] = b
                diags.append(p1_thread_diagnostic(key, b))
    degs, coefs, cvs, snaps, diags = map(pd.DataFrame, (degs, coefs, cvs, snaps, diags))
    diags.to_csv(os.path.join(dp.RESULTS, "phase4_p1_within_thread.csv"), index=False)
    degs.to_csv(os.path.join(dp.RESULTS, "phase4_blocks.csv"), index=False)
    coefs.to_csv(os.path.join(dp.RESULTS, "phase4_coefficients.csv"), index=False)
    cvs.to_csv(os.path.join(dp.RESULTS, "phase4_cv.csv"), index=False)
    snaps.to_csv(os.path.join(dp.RESULTS, "phase4_snapshot.csv"), index=False)
    plot_drif_entry(all_b)
    plot_drif_vs_p1(all_b)
    print("=== Block degeneracy (share of users with |block| < 1e-6) and fallbacks ===")
    print(degs.round(1).T.to_string(header=False))
    print("\n=== Regression coefficients (Eq. 12, on standardised blocks; y = 1 if first side I) ===")
    print(coefs.round(4).to_string(index=False))
    print("\n=== Cross-validated AUC ===")
    print(cvs.round(4).to_string(index=False))
    print("\n=== Snapshot at entry ===")
    print(snaps.round(4).to_string(index=False))
    print("\n=== P1 at entry: overall AUC vs pooled AUC within threads that contain both I and A ===")
    print(diags.round(4).to_string(index=False))


def check():
    print("\n--- Phase 4 check ---")
    d = drif1_of(0.8, 2.0)
    assert abs(d - 0.7685) < 5e-4, f"worked example gives {d:.4f}"
    print(f"worked example: Mult_eff = 2, P1 = 0.8 -> Drif1 = {d:.4f} (expected ~0.769)")
    cvs = pd.read_csv(os.path.join(dp.RESULTS, "phase4_cv.csv"))
    cvs = cvs[cvs["variant"] == "profile"].set_index("set")
    for key in SETS:
        b = pd.read_csv(os.path.join(dp.PROCESSED, f"drive_{key}.csv"), dtype={"user_id": str})
        users = pd.read_csv(os.path.join(dp.PROCESSED, f"users_{key}.csv"), dtype={"user_id": str})
        assert set(b["user_id"]) == set(users["user_id"]), f"{key}: rows != users"
        cols = BLOCKS + ["P1_entry", "Mult_eff", "Drif1", "Drif2", "eta"]
        assert b[cols].notna().all().all(), f"{key}: NaN in Phase 4 outputs"
        dev = float(np.abs(b["Drif1"] + b["Drif2"] - 1).max())
        assert dev <= 1e-15, f"{key}: Drif1 + Drif2 != 1 (max deviation {dev})"
        assert b["Mult_eff"].between(0, KAPPA).all(), f"{key}: Mult_eff outside [0, {KAPPA}]"
        assert b[["Drif1", "Drif2", "P1_entry"]].apply(lambda c: c.between(0, 1).all()).all()
        for c in ("auc_cv_stratified", "auc_cv_thread_heldout"):
            assert cvs.at[key, c] <= ft.AUC_LIMIT, f"{key}: {c} = {cvs.at[key, c]:.3f} > {ft.AUC_LIMIT} (leakage?)"
        ctx = make_context(key)                          # drif(t) reproduces the entry snapshot for one user
        k = int(np.argmax(b["side"].notna().to_numpy()))
        p1, me, d1 = drif(ctx, float(b["t_ref_h"].iloc[k]) - 1e-9)
        assert np.isfinite(d1).all() and ((d1 >= 0) & (d1 <= 1)).all()
        gap = cvs.at[key, "gap"]
        print(f"{key:22s} OK  N={len(b):,}  max|Drif1+Drif2-1|={dev:.0e}  Mult_eff in [{b['Mult_eff'].min():.2f}, "
              f"{b['Mult_eff'].max():.2f}]  CV AUC strat={cvs.at[key, 'auc_cv_stratified']:.3f} "
              f"thread={cvs.at[key, 'auc_cv_thread_heldout']:.3f} gap={gap:+.3f}{'  <-- GAP > 0.10' if gap > MAX_CV_GAP else ''}")
    print("All Phase 4 checks passed.")


if __name__ == "__main__":
    main()
    if "--check" in sys.argv:
        check()
