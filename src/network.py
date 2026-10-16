"""Phase 3 - layered network and hidden groups (docs/SAIR_paper.md §3).

For each set (Charlie Hebdo, Sydney Siege, Ottawa robustness story):
  PageRank on the follow graph (edge u -> v = u follows v, so followed accounts score high)
  core = top 1% by PageRank; strong layer = follows a core node; weak = everyone else
  mpr = tau * (k1 * pr_norm + k2 * Info_match)  (Eq. 17);  loss factor r = 1 - mpr_norm, clipped
  groups = Louvain communities of the undirected follow graph (no retweet-based Prb: that is the outcome)
  Gact = mean log1p(statuses_count) of members; Gpre = Jaccard(group TF-IDF keywords, rumour tags)
  Group_att = Gact_norm * Gpre

Usage:
    python src/network.py           build layers and groups
    python src/network.py --check   build, then run the Phase 3 checks
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

import data_prep as dp
import features as ft

SETS = ft.SETS
CORE_SHARE = 0.01                            # core = top 1% by PageRank (DECISIONS D32)
CORE_SENSITIVITY = [0.005, 0.01, 0.02]
PAGERANK_ALPHA = 0.85
TAU, K1, K2 = 1.0, 0.5, 0.5                  # Eq. 17 weights (DECISIONS D33)
R_CLIP = (0.05, 0.95)
TOP_K_GROUP = 30
LAYERS = ["core", "strong", "weak"]
LAYER_CODE = {"weak": 0, "strong": 1, "core": 2}
AUC_COLUMNS = ["pr_norm", "layer_code", "r", "Gact", "Gpre", "Group_att"]


# ---------------------------------------------------------------- layers

def follow_graph(G):
    """Directed graph with follow edges only (u -> v = u follows v); every user kept as a node."""
    F = nx.DiGraph()
    F.add_nodes_from(G.nodes)
    F.add_edges_from((u, v) for u, v, k in G.edges(data="kind") if k == "follow")
    return F


def layers_for(F, pr, share):
    """core = top `share` of users by PageRank (ties broken by user id); strong = non-core users who
    follow a core node; weak = the rest."""
    k = max(1, math.ceil(share * F.number_of_nodes()))
    order = sorted(F.nodes, key=lambda u: (-pr[u], u))
    core = set(order[:k])
    layer = {}
    for u in F.nodes:
        if u in core:
            layer[u] = "core"
        elif any(v in core for v in F.successors(u)):
            layer[u] = "strong"
        else:
            layer[u] = "weak"
    return pd.Series(layer, name="layer"), core


def efficiency(pr_norm, info_match):
    """mpr = tau*(k1*pr_norm + k2*Info_match); r = 1 - min-max(mpr), clipped to R_CLIP (Eq. 17, D33)."""
    mpr = TAU * (K1 * pr_norm + K2 * info_match)
    mpr_norm = (mpr - mpr.min()) / (mpr.max() - mpr.min())
    return mpr, (1 - mpr_norm).clip(*R_CLIP)


def source_authors_in_core(tweets, core):
    src = tweets[tweets["kind"] == "source"]
    out = {}
    for stance, label in (("support", "rumour"), ("deny", "correction")):
        authors = set(src.loc[src["stance"] == stance, "user_id"])
        out[f"{label}_source_authors"] = len(authors)
        out[f"{label}_authors_in_core"] = len(authors & core)
        out[f"{label}_authors_in_core_pct"] = 100 * len(authors & core) / len(authors) if authors else np.nan
    return out


# ---------------------------------------------------------------- groups

def louvain_groups(F):
    """Louvain communities of the undirected follow graph (seed 42); users without follow edges
    become singleton groups."""
    U = F.to_undirected()
    comms = nx.community.louvain_communities(U, seed=dp.SEED)
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    group = {u: g for g, c in enumerate(comms) for u in c}
    return pd.Series(group, name="group"), nx.community.modularity(U, comms)


def group_scores(group, users, texts, rumour_tags):
    """Per group: size, Gact (mean log1p statuses_count), Gpre (Jaccard of top-30 TF-IDF keywords of
    members' own pre-entry text + profiles with the rumour tags), Group_att = Gact_norm * Gpre."""
    df = pd.DataFrame({"group": group, "activity": np.log1p(users["statuses_count"].astype(float))})
    df["text"] = [texts[u] for u in df.index]
    g = df.groupby("group")
    scores = pd.DataFrame({"size": g.size(), "Gact": g["activity"].mean(),
                           "doc": g["text"].apply(lambda t: " ".join(x for x in t if x))})
    scores["Gact_norm"] = (scores["Gact"] - scores["Gact"].min()) / (scores["Gact"].max() - scores["Gact"].min())
    has_doc = scores["doc"].str.strip() != ""
    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-z][a-z']+\b")
    matrix = vectorizer.fit_transform(scores.loc[has_doc, "doc"])
    vocab = vectorizer.get_feature_names_out()
    keywords = {gid: ft.top_keywords(matrix[i], vocab, TOP_K_GROUP) for i, gid in enumerate(scores.index[has_doc])}
    scores["keywords"] = [" ".join(sorted(keywords.get(gid, set()))) for gid in scores.index]
    scores["Gpre"] = [ft.jaccard(keywords.get(gid, set()), rumour_tags) for gid in scores.index]
    scores["Group_att"] = scores["Gact_norm"] * scores["Gpre"]
    return scores.drop(columns="doc")


# ---------------------------------------------------------------- per set

def build_set(key, lex, rumour_tags):
    tweets, users, curves, G = ft.load_set(key)
    feats = pd.read_csv(os.path.join(dp.PROCESSED, f"features_{key}.csv"), dtype={"user_id": str}).set_index("user_id")
    F = follow_graph(G)
    pr = nx.pagerank(F, alpha=PAGERANK_ALPHA)

    net = pd.DataFrame({"pagerank": pd.Series(pr)}).reindex(users.index)
    net["pr_norm"] = net["pagerank"].rank(pct=True, method="average")
    net["layer"], core = layers_for(F, pr, CORE_SHARE)
    net["layer_code"] = net["layer"].map(LAYER_CODE)
    net["mpr"], net["r"] = efficiency(net["pr_norm"], feats["Info_match"].reindex(users.index))

    layer_rows = []
    for share in CORE_SENSITIVITY:
        lay, c = layers_for(F, pr, share)
        counts = lay.value_counts().reindex(LAYERS, fill_value=0)
        layer_rows.append({"set": key, "core_share_pct": 100 * share, "N": len(lay),
                           **{f"{l}_users": int(counts[l]) for l in LAYERS},
                           **{f"{l}_pct": 100 * counts[l] / len(lay) for l in LAYERS},
                           **source_authors_in_core(tweets, c)})

    t_ref, _ = ft.reference_times(users, curves)
    texts, _, _, _ = ft.user_texts(tweets, users, t_ref, lex)
    group, modularity = louvain_groups(F)
    scores = group_scores(group, users, texts, rumour_tags)
    net["group"] = group.reindex(users.index)
    net = net.join(scores[["size", "Gact", "Gact_norm", "Gpre", "Group_att"]].rename(columns={"size": "group_size"}),
                   on="group")
    net["y"] = feats["y"]

    sizes = scores["size"]
    group_row = {"set": key, "N": len(users), "groups": len(scores), "singletons": int((sizes == 1).sum()),
                 "groups_size_ge_2": int((sizes >= 2).sum()), "groups_size_ge_10": int((sizes >= 10).sum()),
                 "largest": int(sizes.max()), "median_size": float(sizes.median()),
                 "users_in_groups_ge_10_pct": 100 * sizes[sizes >= 10].sum() / len(users),
                 "modularity": modularity, "Gpre_nonzero_groups": int((scores["Gpre"] > 0).sum()),
                 "Gpre_nonzero_users_pct": 100 * (net["Gpre"] > 0).mean()}

    net.to_csv(os.path.join(dp.PROCESSED, f"network_{key}.csv"))
    scores.to_csv(os.path.join(dp.PROCESSED, f"groups_{key}.csv"))
    return net, scores, layer_rows, group_row


def auc_table(net, key):
    sided = net[net["y"].notna()]
    rows = []
    for c in AUC_COLUMNS:
        x = sided[c].to_numpy(dtype=float)
        auc = roc_auc_score(sided["y"], x) if np.unique(x).size > 1 else 0.5
        rows.append({"set": key, "feature": c, "auc": auc, "auc_any_direction": max(auc, 1 - auc),
                     "flag_over_0.95": max(auc, 1 - auc) > ft.AUC_LIMIT})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots

def plot_layers(layers):
    """Layer sizes per set, grouped by core share (sensitivity)."""
    colors = dict(zip(CORE_SENSITIVITY, ["#2a78d6", "#eb6834", "#1baf7a"]))   # categorical slots 1-3
    fig, axes = plt.subplots(1, len(SETS), figsize=(5 * len(SETS), 4.3), squeeze=False)
    x = np.arange(len(LAYERS))
    width = 0.26
    for ax, key in zip(axes[0], SETS):
        d = layers[layers["set"] == key]
        for i, share in enumerate(CORE_SENSITIVITY):
            row = d[np.isclose(d["core_share_pct"], 100 * share)].iloc[0]
            vals = [row[f"{l}_pct"] for l in LAYERS]
            bars = ax.bar(x + (i - 1) * width, vals, width * 0.92, color=colors[share],
                          label=f"core = top {100 * share:g}%" + (" (used)" if share == CORE_SHARE else ""))
            if share == CORE_SHARE:
                for b, v in zip(bars, vals):
                    ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=8, color="#52514e")
        ax.set_xticks(x, [l.capitalize() for l in LAYERS])
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8)
        dp.style(ax, f"{dp.EVENT_NAMES.get(key, 'Ottawa story')} (N = {int(d['N'].iloc[0]):,})", "Share of users (%)",
                 xlabel="Layer")
    fig.suptitle("Layer sizes: core by PageRank on the follow graph, strong = follows a core node",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase3_layer_sizes.png")


def plot_group_sizes(all_scores, groups):
    fig, axes = plt.subplots(1, len(SETS), figsize=(5 * len(SETS), 4.3), squeeze=False)
    for ax, key in zip(axes[0], SETS):
        sizes = all_scores[key]["size"]
        bins = np.logspace(0, np.log10(sizes.max() + 1), 20)
        ax.hist(sizes, bins=bins, color="#2a78d6", edgecolor="white", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        g = groups[groups["set"] == key].iloc[0]
        dp.style(ax, f"{dp.EVENT_NAMES.get(key, 'Ottawa story')}: {int(g['groups']):,} groups, "
                     f"Q = {g['modularity']:.2f}", "Number of groups (log)", xlabel="Group size (users, log)")
    fig.suptitle("Louvain groups on the undirected follow graph (singletons = users without follow edges)",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase3_group_sizes.png")


# ---------------------------------------------------------------- main + check

def main():
    pd.set_option("display.width", 230, "display.max_colwidth", 90)
    lex = ft.load_lexicon()
    coverage = pd.read_csv(os.path.join(dp.RESULTS, "phase2_coverage.csv")).set_index("set")
    layers, groups, aucs, all_scores = [], [], [], {}
    for key in SETS:
        rumour_tags = set(coverage.at[key, "rumour_tags"].split())
        net, scores, layer_rows, group_row = build_set(key, lex, rumour_tags)
        layers += layer_rows
        groups.append(group_row)
        aucs.append(auc_table(net, key))
        all_scores[key] = scores
        top = scores[scores["size"] >= 10].sort_values("Group_att", ascending=False).head(3)
        print(f"\n{key}: top groups (size >= 10) by Group_att")
        print(top[["size", "Gact", "Gpre", "Group_att", "keywords"]].round(3).to_string())
    layers, groups, auc = pd.DataFrame(layers), pd.DataFrame(groups), pd.concat(aucs, ignore_index=True)
    layers.to_csv(os.path.join(dp.RESULTS, "phase3_layers.csv"), index=False)
    groups.to_csv(os.path.join(dp.RESULTS, "phase3_groups.csv"), index=False)
    auc.to_csv(os.path.join(dp.RESULTS, "phase3_auc.csv"), index=False)
    plot_layers(layers)
    plot_group_sizes(all_scores, groups)
    print("\n=== Layers (core share sensitivity) and source authors in the core ===")
    print(layers.round(1).to_string(index=False))
    print("\n=== Groups ===")
    print(groups.round(3).to_string(index=False))
    print("\n=== Leakage check: single-feature ROC-AUC (y = first side I) ===")
    print(auc.round(4).to_string(index=False))


def check():
    print("\n--- Phase 3 check ---")
    layers = pd.read_csv(os.path.join(dp.RESULTS, "phase3_layers.csv"))
    auc = pd.read_csv(os.path.join(dp.RESULTS, "phase3_auc.csv"))
    for key in SETS:
        net = pd.read_csv(os.path.join(dp.PROCESSED, f"network_{key}.csv"), dtype={"user_id": str})
        users = pd.read_csv(os.path.join(dp.PROCESSED, f"users_{key}.csv"), dtype={"user_id": str})
        N = len(users)
        assert set(net["user_id"]) == set(users["user_id"]) and net["user_id"].is_unique, f"{key}: rows != users"
        assert net["layer"].isin(LAYERS).all()
        n_core = int((net["layer"] == "core").sum())
        assert n_core == math.ceil(CORE_SHARE * N) and n_core <= 0.02 * N + 1, f"{key}: core not small"
        for share in CORE_SENSITIVITY:
            row = layers[(layers["set"] == key) & np.isclose(layers["core_share_pct"], 100 * share)].iloc[0]
            assert sum(row[f"{l}_users"] for l in LAYERS) == N, f"{key}: layer sizes do not sum to N"
        assert net["group"].notna().all() and (net.groupby("user_id")["group"].nunique() == 1).all(), \
            f"{key}: a user without exactly one group"
        assert net["r"].between(*R_CLIP).all(), f"{key}: r outside {R_CLIP}"
        assert net["pr_norm"].between(0, 1).all() and net["mpr"].between(0, TAU).all()
        for c in ("Gact_norm", "Gpre", "Group_att"):
            assert net[c].notna().all() and net[c].between(0, 1).all(), f"{key}: {c} outside [0, 1]"
        bad = auc[(auc["set"] == key) & (auc["auc_any_direction"] > ft.AUC_LIMIT)]
        assert bad.empty, f"{key}: possible leakage, AUC > {ft.AUC_LIMIT}: {bad[['feature', 'auc']].to_dict('records')}"
        worst = auc[auc["set"] == key].sort_values("auc_any_direction").iloc[-1]
        sizes = net.groupby("group").size()
        print(f"{key:22s} OK  N={N:,}  core={n_core} ({100 * n_core / N:.1f}%)  groups={len(sizes):,}  "
              f"r in [{net['r'].min():.2f}, {net['r'].max():.2f}]  max AUC={worst['auc_any_direction']:.3f} ({worst['feature']})")
    print("All Phase 3 checks passed.")


if __name__ == "__main__":
    main()
    if "--check" in sys.argv:
        check()
