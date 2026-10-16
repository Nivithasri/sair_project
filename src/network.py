"""Phase 3 - layered network and hidden groups (docs/SAIR_paper.md §3).

For each set (Charlie Hebdo, Sydney Siege, Ottawa robustness story):
  PageRank on the follow graph (edge u -> v = u follows v, so followed accounts score high)
  core = top 1% by PageRank; strong layer = follows a core node; weak = everyone else
  mpr = tau * (k1 * pr_norm + k2 * Info_match)  (Eq. 17);  loss factor r = 1 - mpr_norm, clipped
  groups = KMeans on profile-only features (primary; no event behaviour, no follow graph), with
           Louvain on the undirected follow graph kept as a robustness variant (DECISIONS D37)
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
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import data_prep as dp
import features as ft

SETS = ft.SETS
CORE_SHARE = 0.01                            # core = top 1% by PageRank (DECISIONS D32)
CORE_SENSITIVITY = [0.005, 0.01, 0.02]
PAGERANK_ALPHA = 0.85
TAU, K1, K2 = 1.0, 0.5, 0.5                  # Eq. 17 weights (DECISIONS D33)
R_CLIP = (0.05, 0.95)
TOP_K_GROUP = 30
K_OPTIONS = [20, 50, 100]                    # KMeans k candidates, chosen by silhouette (D37)
PROFILE_SVD_DIMS = 50
SILHOUETTE_SAMPLE = 5000
MAX_THREAD_NMI = 0.3
LAYERS = ["core", "strong", "weak"]
LAYER_CODE = {"weak": 0, "strong": 1, "core": 2}
AUC_COLUMNS = ["pr_norm", "layer_code", "r", "Gact", "Gpre", "Group_att",
               "Gact_follow", "Gpre_follow", "Group_att_follow"]


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


def profile_matrix(users, t0):
    """Profile-only features (no event behaviour): TF-IDF of profile descriptions reduced by SVD, plus
    standardised log1p of followers, friends, statuses and account age in days at the event start.
    Both blocks are scaled to the same total variance (DECISIONS D37)."""
    docs = [" ".join(ft.tokens(d)) for d in users["description"]]
    tfidf = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-z][a-z']+\b", min_df=2).fit_transform(docs)
    n_comp = min(PROFILE_SVD_DIMS, tfidf.shape[1] - 1)
    text = StandardScaler().fit_transform(TruncatedSVD(n_comp, random_state=dp.SEED).fit_transform(tfidf))
    created = pd.to_datetime(users["user_created_at"], format="%a %b %d %H:%M:%S +0000 %Y", utc=True)
    age_days = ((t0 - created) / pd.Timedelta(days=1)).clip(lower=0)
    numeric = np.log1p(np.column_stack([users["followers_count"], users["friends_count"],
                                        users["statuses_count"], age_days]).astype(float))
    numeric = StandardScaler().fit_transform(numeric)
    text *= np.sqrt(numeric.shape[1] / n_comp)
    return np.hstack([text, numeric])


def profile_groups(users, t0):
    """KMeans (seed 42) on profile-only features; k from K_OPTIONS by silhouette score."""
    X = profile_matrix(users, t0)
    rows, labels = [], {}
    for k in K_OPTIONS:
        if k >= len(users):
            continue
        labels[k] = KMeans(k, n_init=10, random_state=dp.SEED).fit_predict(X)
        sil = silhouette_score(X, labels[k], sample_size=min(SILHOUETTE_SAMPLE, len(users)), random_state=dp.SEED)
        rows.append({"k": k, "silhouette": sil})
    sil = pd.DataFrame(rows)
    best = int(sil.loc[sil["silhouette"].idxmax(), "k"])
    return pd.Series(labels[best], index=users.index, name="group"), best, sil


def first_thread(tweets):
    return tweets.sort_values("created_at").groupby("user_id")["thread_id"].first()


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

    t_ref, t0 = ft.reference_times(users, curves)
    texts, _, _, _ = ft.user_texts(tweets, users, t_ref, lex)
    thread = first_thread(tweets).reindex(users.index)

    # primary: KMeans on profile-only features; robustness variant: Louvain on the follow graph (D37)
    group, k, sil = profile_groups(users, t0)
    group_f, modularity_f = louvain_groups(F)
    group_f = group_f.reindex(users.index)
    scores = group_scores(group, users, texts, rumour_tags)
    scores_f = group_scores(group_f, users, texts, rumour_tags)
    net["group"] = group
    net = net.join(scores[["size", "Gact", "Gact_norm", "Gpre", "Group_att"]].rename(columns={"size": "group_size"}),
                   on="group")
    net["group_follow"] = group_f
    net = net.join(scores_f[["size", "Gact", "Gact_norm", "Gpre", "Group_att"]].add_suffix("_follow")
                   .rename(columns={"size_follow": "group_size_follow"}), on="group_follow")
    net["first_thread"] = thread
    net["y"] = feats["y"]

    group_rows = []
    for variant, g, sc in (("profile_kmeans", group, scores), ("follow_louvain", group_f, scores_f)):
        sizes = sc["size"]
        group_rows.append({
            "set": key, "variant": variant, "primary": variant == "profile_kmeans", "N": len(users),
            "groups": len(sc), "k_chosen": k if variant == "profile_kmeans" else np.nan,
            "silhouette": float(sil["silhouette"].max()) if variant == "profile_kmeans" else np.nan,
            "modularity": modularity_f if variant == "follow_louvain" else np.nan,
            "singletons": int((sizes == 1).sum()), "smallest": int(sizes.min()), "median_size": float(sizes.median()),
            "largest": int(sizes.max()),
            "nmi_first_thread": normalized_mutual_info_score(thread, g),
            "Gpre_nonzero_users_pct": 100 * (g.map(sc["Gpre"]) > 0).mean()})
    sil.insert(0, "set", key)

    net.to_csv(os.path.join(dp.PROCESSED, f"network_{key}.csv"))
    scores.to_csv(os.path.join(dp.PROCESSED, f"groups_{key}.csv"))
    scores_f.to_csv(os.path.join(dp.PROCESSED, f"groups_follow_{key}.csv"))
    return net, scores, layer_rows, group_rows, sil


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
    """Histogram of profile-KMeans group sizes per set (primary grouping)."""
    fig, axes = plt.subplots(1, len(SETS), figsize=(5 * len(SETS), 4.3), squeeze=False)
    for ax, key in zip(axes[0], SETS):
        sizes = all_scores[key]["size"]
        ax.hist(sizes, bins=20, color="#2a78d6", edgecolor="white", linewidth=1.5)
        g = groups[(groups["set"] == key) & groups["primary"]].iloc[0]
        dp.style(ax, f"{dp.EVENT_NAMES.get(key, 'Ottawa story')}: k = {int(g['k_chosen'])}, "
                     f"silhouette {g['silhouette']:.2f}, NMI(thread) {g['nmi_first_thread']:.2f}",
                 "Number of groups", xlabel="Group size (users)")
    fig.suptitle("Profile-only KMeans groups (primary): sizes per set",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, "phase3_group_sizes.png")


# ---------------------------------------------------------------- main + check

def main():
    pd.set_option("display.width", 230, "display.max_colwidth", 90)
    lex = ft.load_lexicon()
    coverage = pd.read_csv(os.path.join(dp.RESULTS, "phase2_coverage.csv")).set_index("set")
    layers, groups, aucs, all_scores, sils = [], [], [], {}, []
    for key in SETS:
        rumour_tags = set(coverage.at[key, "rumour_tags"].split())
        net, scores, layer_rows, group_rows, sil = build_set(key, lex, rumour_tags)
        layers += layer_rows
        groups += group_rows
        sils.append(sil)
        aucs.append(auc_table(net, key))
        all_scores[key] = scores
        top = scores[scores["size"] >= 10].sort_values("Group_att", ascending=False).head(3)
        print(f"\n{key}: top groups (size >= 10) by Group_att")
        print(top[["size", "Gact", "Gpre", "Group_att", "keywords"]].round(3).to_string())
    layers, groups, auc = pd.DataFrame(layers), pd.DataFrame(groups), pd.concat(aucs, ignore_index=True)
    layers.to_csv(os.path.join(dp.RESULTS, "phase3_layers.csv"), index=False)
    groups.to_csv(os.path.join(dp.RESULTS, "phase3_groups.csv"), index=False)
    sils = pd.concat(sils, ignore_index=True)
    sils.to_csv(os.path.join(dp.RESULTS, "phase3_kmeans_silhouette.csv"), index=False)
    auc.to_csv(os.path.join(dp.RESULTS, "phase3_auc.csv"), index=False)
    plot_layers(layers)
    plot_group_sizes(all_scores, groups)
    print("\n=== Layers (core share sensitivity) and source authors in the core ===")
    print(layers.round(1).to_string(index=False))
    print("\n=== KMeans silhouette by k ===")
    print(sils.round(4).to_string(index=False))
    print("\n=== Groups (primary = profile KMeans; follow Louvain = robustness variant) ===")
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
        for col in ("group", "group_follow"):
            assert net[col].notna().all() and (net.groupby("user_id")[col].nunique() == 1).all(), \
                f"{key}: a user without exactly one {col}"
        nmi = normalized_mutual_info_score(net["first_thread"], net["group"])
        assert nmi < MAX_THREAD_NMI, f"{key}: profile groups still track threads (NMI {nmi:.3f})"
        nmi_f = normalized_mutual_info_score(net["first_thread"], net["group_follow"])
        assert net["r"].between(*R_CLIP).all(), f"{key}: r outside {R_CLIP}"
        assert net["pr_norm"].between(0, 1).all() and net["mpr"].between(0, TAU).all()
        for c in ("Gact_norm", "Gpre", "Group_att", "Gact_norm_follow", "Gpre_follow", "Group_att_follow"):
            assert net[c].notna().all() and net[c].between(0, 1).all(), f"{key}: {c} outside [0, 1]"
        bad = auc[(auc["set"] == key) & (auc["auc_any_direction"] > ft.AUC_LIMIT)]
        assert bad.empty, f"{key}: possible leakage, AUC > {ft.AUC_LIMIT}: {bad[['feature', 'auc']].to_dict('records')}"
        worst = auc[auc["set"] == key].sort_values("auc_any_direction").iloc[-1]
        sizes = net.groupby("group").size()
        print(f"{key:22s} OK  N={N:,}  core={n_core} ({100 * n_core / N:.1f}%)  groups={len(sizes):,}  "
              f"NMI(group, thread) profile={nmi:.3f} vs follow={nmi_f:.3f}  "
              f"r in [{net['r'].min():.2f}, {net['r'].max():.2f}]  max AUC={worst['auc_any_direction']:.3f} ({worst['feature']})")
    print("All Phase 3 checks passed.")


if __name__ == "__main__":
    main()
    if "--check" in sys.argv:
        check()
