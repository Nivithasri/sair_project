"""Phase 2 - per-user features (docs/SAIR_paper.md §2).

For every user of each set (Charlie Hebdo, Sydney Siege, Ottawa robustness story):
  S_emo        cosine of the user's NRC emotion vector and the rumour's emotion vector
  Info_match   Jaccard of the user's tags and the rumour's tags
  connectivity mean over friends of [tag Jaccard + friend-list Jaccard]
  friend_power log1p(mean over friends of replies + retweets + mu * favourites per own post)
and the event-level popularity curve Pop(t) (one decaying term per story, DECISIONS D26).

No leakage (DECISIONS D27): user text = the user's OWN posts (sources, replies) written strictly
before t_ref, plus the profile description. Retweeted text is never used. t_ref = entry time into
I/A, or the end of the event window for users who never take a side.

Usage:
    python src/features.py           build features
    python src/features.py --check   build features, then run the Phase 2 checks
"""
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

import data_prep as dp

SETS = dp.EVENTS + [dp.STORY["key"]]
LEXICON = os.path.join(dp.ROOT, "data", "raw", "nrc_emotion_lexicon", "NRC-Emotion-Lexicon",
                       "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
INTENSIFIERS = {"very", "so", "really", "extremely", "totally", "absolutely", "too", "super"}
NEGATIONS = {"not", "no", "never", "nothing", "nobody", "neither", "nor", "dont", "don't", "isn't",
             "wasn't", "can't", "cannot", "won't", "didn't", "doesn't", "aren't"}
INTENSIFY = 1.5                              # modifier coefficient theta for intensifiers (DECISIONS D28)
LAMBDA = np.log(2) / 12                      # recency rate per hour: half-life 12 h (DECISIONS D28)
TOP_K_USER, TOP_K_RUMOUR = 10, 30            # TF-IDF keywords per user / for the rumour
MU = 0.5                                     # weight of favourites in friend power (DECISIONS D29)
POP_HALF_LIFE_H = 12                         # default half-life w; free parameter in Phase 5 (D26)
AUC_LIMIT = 0.95
FEATURES = ["S_emo", "Info_match", "connectivity", "friend_power"]
RANGES = {"S_emo": (0, 1), "Info_match": (0, 1), "connectivity": (0, 2), "friend_power": (0, np.inf)}

STOP = {"rt", "amp", "http", "https", "via"}


# ---------------------------------------------------------------- text

def load_lexicon():
    """word -> 8-dim 0/1 vector of NRC emotions."""
    lex = pd.read_csv(LEXICON, sep="\t", names=["word", "emotion", "flag"], keep_default_na=False)
    lex = lex[lex["emotion"].isin(EMOTIONS)].pivot(index="word", columns="emotion", values="flag")[EMOTIONS]
    lex = lex[lex.sum(axis=1) > 0]
    return {w: v.astype(float) for w, v in zip(lex.index, lex.to_numpy())}


def tokens(text):
    """Lowercase word tokens; URLs, @mentions and the '#' sign removed."""
    text = re.sub(r"http\S+|@\w+", " ", str(text).lower()).replace("#", " ")
    return [t for t in re.findall(r"[a-z][a-z']+", text) if t not in STOP]


def emotion_vector(text, lex):
    """NRC emotion intensity per token. An intensifier just before a word multiplies it by INTENSIFY;
    a negation in the two preceding tokens cancels it (DECISIONS D28)."""
    toks = tokens(text)
    v = np.zeros(len(EMOTIONS))
    for i, t in enumerate(toks):
        if t not in lex:
            continue
        if NEGATIONS & set(toks[max(0, i - 2):i]):
            continue
        v += lex[t] * (INTENSIFY if i > 0 and toks[i - 1] in INTENSIFIERS else 1.0)
    return v / max(len(toks), 1)


def hashtags_of(text):
    return {h.lower() for h in re.findall(r"#(\w+)", str(text))}


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


# ---------------------------------------------------------------- loading

def load_set(key):
    tweets = pd.read_csv(os.path.join(dp.PROCESSED, f"tweets_{key}.csv"),
                         dtype={"tweet_id": str, "user_id": str, "thread_id": str, "in_reply_to_status_id": str,
                                "in_reply_to_user_id": str, "story_veracity": str},
                         keep_default_na=False, na_values={"followers_count": [""]})
    tweets["created_at"] = pd.to_datetime(tweets["created_at"], utc=True)
    users = pd.read_csv(os.path.join(dp.PROCESSED, f"users_{key}.csv"), dtype={"user_id": str},
                        keep_default_na=False, na_values={"side": [""], "t_enter": [""]}).set_index("user_id")
    users["first_t"] = pd.to_datetime(users["first_t"], utc=True)
    users["t_enter"] = pd.to_datetime(users["t_enter"], utc=True)
    curves = pd.read_csv(os.path.join(dp.PROCESSED, f"real_curves_{key}.csv"))
    with open(os.path.join(dp.PROCESSED, f"graph_{key}.gpickle"), "rb") as f:
        G = pickle.load(f)
    return tweets, users, curves, G


def reference_times(users, curves):
    """t_ref = entry time into I/A; for users who never take a side, the end of the event window."""
    t0 = users["first_t"].min()
    t_end = t0 + pd.Timedelta(hours=float(curves["t_hours"].iloc[-1]))
    return users["t_enter"].fillna(t_end), t0


# ---------------------------------------------------------------- features

def user_documents(tweets, users, t_ref):
    """Own posts (source/reply) strictly before t_ref, per user, as (text, created_at, hashtags) lists."""
    own = tweets[tweets["kind"].isin(["source", "reply"])].copy()
    own["t_ref"] = own["user_id"].map(t_ref)
    own = own[own["created_at"] < own["t_ref"]]
    return {u: d for u, d in own.groupby("user_id")}


def rumour_profile(tweets, lex):
    """Rumour emotion vector and tags from the supporting source tweets of the kept stories."""
    src = tweets[(tweets["kind"] == "source") & (tweets["stance"] == "support")]
    vec = np.mean([emotion_vector(t, lex) for t in src["text"]], axis=0)
    doc = " ".join(" ".join(tokens(t)) for t in src["text"])
    tags = set().union(*[set(h.split()) for h in src["hashtags"] if h]) if len(src) else set()
    return vec, doc, tags, len(src)


def top_keywords(tfidf_row, vocab, k):
    row = tfidf_row.tocoo()
    order = np.argsort(-row.data)[:k]
    return {vocab[row.col[i]] for i in order}


def text_features(tweets, users, t_ref, lex):
    """has_text, S_emo, Info_match, and each user's tag set."""
    docs = user_documents(tweets, users, t_ref)
    r_vec, r_doc, r_hashtags, n_src = rumour_profile(tweets, lex)
    t_ref_h = {u: t for u, t in t_ref.items()}

    texts, emo, n_own, hashtag_sets = {}, {}, {}, {}
    for u, desc in users["description"].items():
        d = docs.get(u)
        parts, vecs, weights, tags = [], [], [], hashtags_of(desc)
        if desc and tokens(desc):
            parts.append(desc)
            vecs.append(emotion_vector(desc, lex))
            weights.append(1.0)                              # profile counts as current at t_ref
        if d is not None:
            for text, t, h in zip(d["text"], d["created_at"], d["hashtags"]):
                parts.append(text)
                vecs.append(emotion_vector(text, lex))
                weights.append(np.exp(-LAMBDA * (t_ref_h[u] - t) / pd.Timedelta(hours=1)))
                tags |= set(h.split()) if h else set()
        n_own[u] = 0 if d is None else len(d)
        texts[u] = " ".join(" ".join(tokens(p)) for p in parts)
        emo[u] = np.average(vecs, axis=0, weights=weights) if vecs else None
        hashtag_sets[u] = tags

    uids = list(users.index)
    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-z][a-z']+\b")
    matrix = vectorizer.fit_transform([texts[u] for u in uids] + [r_doc])
    vocab = vectorizer.get_feature_names_out()
    rumour_tags = top_keywords(matrix[len(uids)], vocab, TOP_K_RUMOUR) | r_hashtags

    rows, tags = [], {}
    for i, u in enumerate(uids):
        has_text = bool(texts[u].strip())
        tags[u] = (top_keywords(matrix[i], vocab, TOP_K_USER) | hashtag_sets[u]) if has_text else set()
        rows.append({"user_id": u, "has_text": int(has_text), "n_own_posts_before_ref": n_own[u],
                     "S_emo": cosine(emo[u], r_vec) if has_text else np.nan,
                     "emotion_words": int(has_text and np.any(emo[u] > 0)),
                     "Info_match": jaccard(tags[u], rumour_tags) if has_text else np.nan})
    info = {"rumour_sources": n_src, "rumour_tags": " ".join(sorted(rumour_tags)),
            "rumour_emotion": " ".join(f"{e}={x:.3f}" for e, x in zip(EMOTIONS, r_vec))}
    return pd.DataFrame(rows).set_index("user_id"), tags, info


def friend_sets(G, tweets, users, t_ref):
    """Friends = follow successors; if a user has none, interaction partners from the user's own
    replies/retweets made strictly before t_ref (an interaction created by the entry post itself
    would leak the outcome, DECISIONS D27)."""
    follow = {u: {v for v in G.successors(u) if G[u][v]["kind"] == "follow"} for u in G}
    source_author = tweets[tweets["kind"] == "source"].set_index("thread_id")["user_id"]
    early = tweets[tweets["kind"].isin(["reply", "retweet"])].copy()
    early = early[early["created_at"] < early["user_id"].map(t_ref)]
    partner = np.where(early["kind"] == "reply", early["in_reply_to_user_id"],
                       early["thread_id"].map(source_author))
    inter = {}
    for u, v in zip(early["user_id"], partner):
        if v in G and v != u:
            inter.setdefault(u, set()).add(v)
    friends = {u: (follow[u] if follow[u] else inter.get(u, set())) for u in users.index}
    kind = {u: ("follow" if follow[u] else "interaction" if friends[u] else "none") for u in users.index}
    return friends, follow, kind


def engagement_lookup(tweets):
    """Per author: post times and cumulative engagement (replies in data + retweets + MU * favourites)."""
    own = tweets[tweets["kind"].isin(["source", "reply"])].copy()
    replies_in_data = tweets[tweets["kind"] == "reply"]["in_reply_to_status_id"].value_counts()
    own["engagement"] = (own["tweet_id"].map(replies_in_data).fillna(0) + own["retweet_count"]
                         + MU * own["favorite_count"])
    look = {}
    for u, d in own.sort_values("created_at").groupby("user_id"):
        look[u] = (d["created_at"].map(pd.Timestamp.timestamp).to_numpy(), np.cumsum(d["engagement"].to_numpy()))
    return look


def friend_power_of(u_friends, t_ref_u, look):
    """Mean over friends of their average engagement per own post written before t_ref_u."""
    vals = []
    for f in u_friends:
        if f not in look:
            vals.append(0.0)
            continue
        times, cum = look[f]
        n = int(np.searchsorted(times, t_ref_u.timestamp(), side="left"))   # posts strictly before t_ref
        vals.append(cum[n - 1] / n if n > 0 else 0.0)
    return float(np.log1p(np.mean(vals)))


def network_features(G, tweets, users, t_ref, tags):
    friends, follow, kind = friend_sets(G, tweets, users, t_ref)
    look = engagement_lookup(tweets)
    rows = []
    for u in users.index:
        fs = friends[u]
        if fs:
            conn = np.mean([jaccard(tags[u], tags[f]) + jaccard(follow[u], follow[f]) for f in fs])
            power = friend_power_of(fs, t_ref[u], look)
        else:
            conn = power = np.nan
        rows.append({"user_id": u, "has_friends": int(bool(fs)), "friend_kind": kind[u], "n_friends": len(fs),
                     "connectivity": conn, "friend_power": power})
    return pd.DataFrame(rows).set_index("user_id")


def popularity(tweets, users, t_hours, half_life_h=POP_HALF_LIFE_H):
    """Pop(t) = sum_s w_s * 0.5^((t - t0_s)/w) for t >= t0_s, normalised to max 1 (DECISIONS D26).
    t0_s = first source tweet of story s; w_s = follower count of that tweet's author (known at t0_s)."""
    t0 = users["first_t"].min()
    first_src = tweets[tweets["kind"] == "source"].sort_values("created_at").groupby("story").head(1)
    start = ((first_src["created_at"] - t0) / pd.Timedelta(hours=1)).to_numpy()
    weight = first_src["followers_count"].astype(float).to_numpy()
    t = np.asarray(t_hours, dtype=float)[:, None]
    terms = np.where(t >= start, weight * 0.5 ** ((t - start) / half_life_h), 0.0)
    pop = terms.sum(axis=1)
    stories = pd.DataFrame({"story": first_src["story"].to_numpy(), "t0_h": start, "weight_followers": weight})
    return pop / pop.max(), stories


# ---------------------------------------------------------------- per set

def build_set(key, lex):
    tweets, users, curves, G = load_set(key)
    t_ref, t0 = reference_times(users, curves)
    text, tags, rumour_info = text_features(tweets, users, t_ref, lex)
    net = network_features(G, tweets, users, t_ref, tags)
    feats = users[["side"]].join(text).join(net)
    feats["t_ref"] = t_ref
    feats["y"] = (feats["side"] == "I").astype(float).where(feats["side"].notna())   # NaN for S users

    coverage = {"set": key, "N": len(feats),
                "has_text_pct": 100 * feats["has_text"].mean(),
                "own_posts_before_ref_pct": 100 * (feats["n_own_posts_before_ref"] > 0).mean(),
                "description_pct": 100 * (users["description"].str.strip() != "").mean(),
                "has_friends_pct": 100 * feats["has_friends"].mean(),
                "friends_follow_pct": 100 * (feats["friend_kind"] == "follow").mean(),
                "friends_interaction_pct": 100 * (feats["friend_kind"] == "interaction").mean(),
                "text_but_no_emotion_words": int(((feats["has_text"] == 1) & (feats["emotion_words"] == 0)).sum()),
                **rumour_info}
    for f, flag in (("S_emo", "has_text"), ("Info_match", "has_text"),
                    ("connectivity", "has_friends"), ("friend_power", "has_friends")):
        feats[f] = feats[f].fillna(feats.loc[feats[flag] == 1, f].mean())   # missing -> event mean

    pop, stories = popularity(tweets, users, curves["t_hours"])
    pop_df = pd.DataFrame({"t_hours": curves["t_hours"], "pop": pop, "new_I": curves["C_I"].diff().fillna(curves["C_I"]),
                           "new_A": curves["C_A"].diff().fillna(curves["C_A"])})
    feats.to_csv(os.path.join(dp.PROCESSED, f"features_{key}.csv"))
    pop_df.to_csv(os.path.join(dp.PROCESSED, f"pop_{key}.csv"), index=False)
    stories.to_csv(os.path.join(dp.RESULTS, f"pop_stories_{key}.csv"), index=False)
    return feats, coverage, pop_df, stories


def auc_table(feats, key):
    """Single-feature ROC-AUC for y = first side is I (sided users only)."""
    sided = feats[feats["y"].notna()]
    rows = []
    for f in FEATURES + ["has_text", "has_friends"]:
        x = sided[f].to_numpy(dtype=float)
        auc = roc_auc_score(sided["y"], x) if np.unique(x).size > 1 else 0.5
        rows.append({"set": key, "feature": f, "auc": auc, "auc_any_direction": max(auc, 1 - auc),
                     "flag_over_0.95": max(auc, 1 - auc) > AUC_LIMIT})
    return pd.DataFrame(rows), {"set": key, "sided": len(sided), "y=1 (I)": int(sided["y"].sum()),
                                "y=0 (A)": int((sided["y"] == 0).sum()), "share_I": sided["y"].mean()}


def summary_table(feats, key):
    rows = []
    for f in FEATURES:
        x = feats[f]
        rows.append({"set": key, "feature": f, "mean": x.mean(), "std": x.std(), "min": x.min(),
                     "median": x.median(), "max": x.max(), "mean_I": x[feats["side"] == "I"].mean(),
                     "mean_A": x[feats["side"] == "A"].mean(), "mean_S": x[feats["side"].isna()].mean()})
    return pd.DataFrame(rows)


def plot_pop(pop_df, stories, key):
    """Pop(t) (top) above new I entries per bin (bottom), shared time axis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax1.plot(pop_df["t_hours"], pop_df["pop"], color="#6250d6", lw=2)
    for t in stories["t0_h"]:
        ax1.axvline(t, color="#b5b4ae", lw=0.8, ls=":")
    ax1.set_ylim(0, 1.08)
    dp.style(ax1, f"Pop(t): one decaying term per story (half-life {POP_HALF_LIFE_H} h, "
                  f"weights = source-author followers); dotted = story starts", "Pop (max = 1)", xlabel="")
    width = float(pop_df["t_hours"].diff().iloc[1])
    ax2.bar(pop_df["t_hours"], pop_df["new_I"], width=width * 0.9, color=dp.COLORS["I"])
    dp.style(ax2, "New users entering I per bin (real data)", "Users")
    fig.suptitle(f"{dp.EVENT_NAMES.get(key, key)}: do the popularity waves line up with rumour entries?",
                 fontsize=13, x=0.01, ha="left")
    return dp.save(fig, f"pop_vs_entries_{key}.png")


# ---------------------------------------------------------------- main + check

def main():
    pd.set_option("display.width", 220, "display.max_colwidth", 120)
    lex = load_lexicon()
    summaries, coverages, aucs, balances = [], [], [], []
    for key in SETS:
        feats, coverage, pop_df, stories = build_set(key, lex)
        summaries.append(summary_table(feats, key))
        coverages.append(coverage)
        a, b = auc_table(feats, key)
        aucs.append(a)
        balances.append(b)
        if key == dp.EVENTS[0]:
            plot_pop(pop_df, stories, key)
    summary = pd.concat(summaries, ignore_index=True)
    coverage = pd.DataFrame(coverages)
    auc = pd.concat(aucs, ignore_index=True)
    balance = pd.DataFrame(balances)
    summary.to_csv(os.path.join(dp.RESULTS, "phase2_feature_summary.csv"), index=False)
    coverage.to_csv(os.path.join(dp.RESULTS, "phase2_coverage.csv"), index=False)
    auc.to_csv(os.path.join(dp.RESULTS, "phase2_auc.csv"), index=False)
    print(f"NRC lexicon: {len(lex):,} emotion words\n")
    print("=== Feature summary (after filling missing values with the event mean) ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== Coverage ===")
    print(coverage.drop(columns=["rumour_tags", "rumour_emotion"]).round(1).to_string(index=False))
    for _, r in coverage.iterrows():
        print(f"  {r['set']}: rumour tags = {r['rumour_tags']}\n  {r['set']}: rumour emotion = {r['rumour_emotion']}")
    print("\n=== Class balance of y (1 = first side I) ===")
    print(balance.round(4).to_string(index=False))
    print("\n=== Leakage check: single-feature ROC-AUC ===")
    print(auc.round(4).to_string(index=False))


def check():
    print("\n--- Phase 2 check ---")
    auc = pd.read_csv(os.path.join(dp.RESULTS, "phase2_auc.csv"))
    for key in SETS:
        feats = pd.read_csv(os.path.join(dp.PROCESSED, f"features_{key}.csv"), dtype={"user_id": str})
        users = pd.read_csv(os.path.join(dp.PROCESSED, f"users_{key}.csv"), dtype={"user_id": str})
        pop = pd.read_csv(os.path.join(dp.PROCESSED, f"pop_{key}.csv"))
        assert set(feats["user_id"]) == set(users["user_id"]), f"{key}: feature rows != users"
        for f, (lo, hi) in RANGES.items():
            assert feats[f].notna().all(), f"{key}: NaN in {f}"
            assert feats[f].between(lo, hi).all(), f"{key}: {f} outside [{lo}, {hi}]"
        assert feats[["has_text", "has_friends"]].isin([0, 1]).all().all()
        assert pop["pop"].notna().all() and pop["pop"].between(0, 1).all(), f"{key}: Pop outside [0, 1]"
        assert np.isclose(pop["pop"].max(), 1.0), f"{key}: Pop not normalised"
        bad = auc[(auc["set"] == key) & (auc["auc_any_direction"] > AUC_LIMIT)]
        assert bad.empty, f"{key}: possible leakage, AUC > {AUC_LIMIT}: {bad[['feature', 'auc']].to_dict('records')}"
        worst = auc[auc["set"] == key].sort_values("auc_any_direction").iloc[-1]
        print(f"{key:22s} OK  N={len(feats):,}  features in range, no NaN, Pop max=1  "
              f"max AUC={worst['auc_any_direction']:.3f} ({worst['feature']})")
    print("All Phase 2 checks passed.")


if __name__ == "__main__":
    main()
    if "--check" in sys.argv:
        check()
