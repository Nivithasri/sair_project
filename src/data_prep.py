"""Phase 1 - data preparation.

Loads PHEME rumour-scheme threads (source tweets, replies, retweets), keeps only FALSE and
UNVERIFIED rumour stories, assigns each tweet a stance toward its rumour, builds the user graph
and the user table, and produces
  - cumulative entry curves C_I(t), C_A(t)       -> primary fitting target for Phase 5
  - expected S/A/I/R prevalence curves            -> display only (exponential lifespan, mean 12 h)
plus event comparison, story, stance and graph diagnostics, and a single-story robustness set.

Usage:
    python src/data_prep.py --compare   event comparison table only (all candidate events)
    python src/data_prep.py             build everything for the chosen events
    python src/data_prep.py --check     build everything, then run the Phase 1 checks
"""
import json
import os
import pickle
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEME = os.path.join(ROOT, "data", "raw", "pheme_rumour_scheme", "pheme-rumour-scheme-dataset")
PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")

CANDIDATES = ["sydneysiege", "ottawashooting", "charliehebdo"]
EVENTS = ["charliehebdo", "sydneysiege"]     # primary first; set from the event comparison (DECISIONS D21)
EVENT_NAMES = {"sydneysiege": "Sydney Siege", "ottawashooting": "Ottawa shooting", "charliehebdo": "Charlie Hebdo"}
KEEP_VERACITY = {"false", "unverified"}      # TRUE stories are excluded (DECISIONS D20)
STORY = {"key": "story_three_shootings", "event": "ottawashooting",
         "story": "There were three separate shooting incidents", "bin_minutes": 15}

STANCES = {"support", "deny", "query", "comment"}
LIFESPAN_H = 12                              # mean spreading lifespan of display curves (DECISIONS D19)
BIN_OPTIONS = [60, 30]                       # minutes; 30 only if 60 gives too few informative bins
MIN_A_USERS = 100
MIN_INFORMATIVE_BINS = 15                    # bins with C between 5% and 95% of its final value
TRIM_LEVEL, TRIM_EXTRA_H = 0.99, 12          # cut at 99% of both C_I and C_A, plus 12 h (DECISIONS D22)

# reply stance toward the rumour = source stance composed with reply-vs-source (DECISIONS D3)
COMPOSE = {("supporting", "agreed"): "support", ("supporting", "disagreed"): "deny",
           ("denying", "agreed"): "deny", ("denying", "disagreed"): "support"}
SOURCE_STANCE = {"supporting": "support", "denying": "deny"}
SIDE = {"support": "I", "deny": "A"}

# categorical palette slots 1-4 (dataviz reference palette, light mode)
COLORS = {"S": "#2a78d6", "I": "#eb6834", "A": "#1baf7a", "R": "#eda100"}


# ---------------------------------------------------------------- loading

def read_json_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def load_annotations():
    """tweet_id -> annotation dict (English rumour-scheme annotations)."""
    rows = read_json_lines(os.path.join(SCHEME, "annotations", "en-scheme-annotations.json"))
    return {a["tweetid"]: a for a in rows}


def tweet_row(t, thread_id, kind):
    u = t["user"]
    return {
        "tweet_id": t["id_str"],
        "thread_id": thread_id,
        "kind": kind,
        "user_id": u["id_str"],
        "screen_name": u["screen_name"],
        "created_at": pd.to_datetime(t["created_at"], format="%a %b %d %H:%M:%S +0000 %Y", utc=True),
        "text": t["text"],
        "hashtags": " ".join(sorted({h["text"].lower() for h in t["entities"]["hashtags"]})),
        "retweet_count": t["retweet_count"],
        "favorite_count": t["favorite_count"],
        "in_reply_to_status_id": t.get("in_reply_to_status_id_str"),
        "in_reply_to_user_id": t.get("in_reply_to_user_id_str"),
        "followers_count": u["followers_count"],
        "friends_count": u["friends_count"],
        "statuses_count": u["statuses_count"],
        "verified": u["verified"],
        "user_created_at": u["created_at"],
    }


def thread_dirs(event):
    edir = os.path.join(SCHEME, "threads", "en", event)
    return [(th, os.path.join(edir, th)) for th in sorted(os.listdir(edir))]


def veracity(a):
    """Thread-level PHEME veracity: false (misinformation=1), true (true=1), otherwise unverified."""
    if str(a.get("misinformation")) == "1":
        return "false"
    if str(a.get("true")) == "1":
        return "true"
    return "unverified"


def load_tweets(event, annotations):
    """All posts of an event: source tweets, replies and retweets of the source, one row each.
    Adds the thread's rumour story (annotation.json 'category'), its veracity and the source stance."""
    rows = []
    for th, d in thread_dirs(event):
        with open(os.path.join(d, "annotation.json"), encoding="utf-8") as f:
            ann = json.load(f)
        info = {"story": ann.get("category") or th, "story_veracity": veracity(ann),
                "source_support": annotations.get(th, {}).get("support", "unlabelled")}
        for sub, kind in (("source-tweets", "source"), ("reactions", "reply")):
            for fn in sorted(os.listdir(os.path.join(d, sub))):
                if fn.endswith(".json") and not fn.startswith("."):
                    with open(os.path.join(d, sub, fn), encoding="utf-8") as f:
                        rows.append({**tweet_row(json.load(f), th, kind), **info})
        rt_path = os.path.join(d, "retweets.json")
        if os.path.exists(rt_path):
            rows += [{**tweet_row(t, th, "retweet"), **info} for t in read_json_lines(rt_path)]
    tweets = pd.DataFrame(rows)
    n_before = len(tweets)
    tweets = tweets.drop_duplicates("tweet_id").sort_values("created_at").reset_index(drop=True)
    tweets.attrs["n_duplicates_dropped"] = n_before - len(tweets)
    tweets["stance"] = load_stance(tweets, annotations)
    return tweets


def load_stance(tweets, annotations):
    """Stance of each tweet toward the rumour: support / deny / query / comment (DECISIONS D3, D4).
    Sources and their retweets take the source's own stance; only supporting/denying sources give I/A."""
    stance = []
    for tid, src, kind in zip(tweets["tweet_id"], tweets["source_support"], tweets["kind"]):
        if kind in ("source", "retweet"):
            s = SOURCE_STANCE.get(src, "comment")
        else:
            r = annotations.get(tid, {}).get("responsetype-vs-source")
            s = COMPOSE.get((src, r), "query" if r == "appeal-for-more-information" else "comment")
        stance.append(s)
    return pd.Series(stance, index=tweets.index, name="stance")


def veracity_filter(tweets):
    """Keep only FALSE and UNVERIFIED stories (DECISIONS D20). Returns kept tweets and what was dropped."""
    keep = tweets["story_veracity"].isin(KEEP_VERACITY)
    kept = tweets[keep].reset_index(drop=True)
    dropped = tweets[~keep]
    info = {
        "threads_all": tweets["thread_id"].nunique(),
        "threads_dropped_true": dropped["thread_id"].nunique(),
        "stories_dropped_true": dropped["story"].nunique(),
        "posts_dropped": len(dropped),
        "users_all": tweets["user_id"].nunique(),
        "users_dropped": tweets["user_id"].nunique() - kept["user_id"].nunique(),
    }
    kept.attrs.update(tweets.attrs)
    return kept, info


# ---------------------------------------------------------------- users and graph

def build_users(tweets):
    """One row per user. The first support/deny post fixes the side (I or A) and the entry time."""
    t = tweets.sort_values("created_at")
    g = t.groupby("user_id", sort=False)
    last = g.tail(1).set_index("user_id")
    users = pd.DataFrame({
        "screen_name": last["screen_name"],
        "n_posts": g.size(),
        "n_source": g["kind"].apply(lambda k: (k == "source").sum()),
        "n_reply": g["kind"].apply(lambda k: (k == "reply").sum()),
        "n_retweet": g["kind"].apply(lambda k: (k == "retweet").sum()),
        "n_support": g["stance"].apply(lambda s: (s == "support").sum()),
        "n_deny": g["stance"].apply(lambda s: (s == "deny").sum()),
        "first_t": g["created_at"].min(),
        "last_t": g["created_at"].max(),
        "hashtags": g["hashtags"].apply(lambda h: " ".join(sorted(set(" ".join(h).split())))),
        "followers_count": last["followers_count"],
        "friends_count": last["friends_count"],
        "statuses_count": last["statuses_count"],
        "verified": last["verified"],
        "user_created_at": last["user_created_at"],
    })
    users["reposted_rumour"] = (t[(t["kind"] == "retweet") & (t["stance"] == "support")]
                                .groupby("user_id").size().reindex(users.index).notna().astype(int))
    sided = t[t["stance"].isin(SIDE)].groupby("user_id").head(1).set_index("user_id")
    users["side"] = sided["stance"].map(SIDE).reindex(users.index)        # NaN -> stays S
    users["t_enter"] = sided["created_at"].reindex(users.index)
    users["enter_kind"] = sided["kind"].reindex(users.index)              # source / reply / retweet
    users["enter_story"] = sided["story"].reindex(users.index)
    users["both_stances"] = (users["n_support"] > 0) & (users["n_deny"] > 0)
    users.index.name = "user_id"
    return users


def build_graph(event, tweets, users):
    """Directed graph, edge u -> v = u receives information from v.
    kind='follow' from who-follows-whom (u follows v); kind='interaction' when u replied to or
    retweeted v and no follow edge exists (DECISIONS D2). Only threads kept by the filter are used."""
    G = nx.DiGraph(event=event)
    G.add_nodes_from(users.index)
    kept_threads = set(tweets["thread_id"])
    for th, d in thread_dirs(event):
        path = os.path.join(d, "who-follows-whom.dat")
        if th not in kept_threads or not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                a = line.split()
                if len(a) == 2 and a[0] != a[1] and a[0] in G and a[1] in G:
                    G.add_edge(a[0], a[1], kind="follow")
    source_author = tweets[tweets["kind"] == "source"].set_index("thread_id")["user_id"]
    for u, kind, th, rep in zip(tweets["user_id"], tweets["kind"], tweets["thread_id"],
                                tweets["in_reply_to_user_id"]):
        if kind == "reply":
            v = rep
        elif kind == "retweet":
            v = source_author.get(th)
        else:
            continue
        if v in G and v != u and not G.has_edge(u, v):
            G.add_edge(u, v, kind="interaction")
    return G


def graph_stats(G):
    kinds = list(nx.get_edge_attributes(G, "kind").values())
    n, m = G.number_of_nodes(), G.number_of_edges()
    wcc = max(nx.weakly_connected_components(G), key=len)
    scc = max(nx.strongly_connected_components(G), key=len)
    no_friends = sum(1 for u in G if G.out_degree(u) == 0)
    return {
        "nodes": n,
        "edges": m,
        "edges_follow": kinds.count("follow"),
        "edges_interaction": kinds.count("interaction"),
        "share_follow": kinds.count("follow") / m,
        "isolates": nx.number_of_isolates(G),
        "no_friends_outdeg0": no_friends,
        "share_no_friends": no_friends / n,
        "largest_wcc": len(wcc),
        "share_largest_wcc": len(wcc) / n,
        "largest_scc": len(scc),
    }


# ---------------------------------------------------------------- curves

def entry_hours(users):
    """Entry time into I/A in hours since the first post of the population (NaN = never sided)."""
    t0 = users["first_t"].min()
    return ((pd.to_datetime(users["t_enter"], utc=True) - t0) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def trim_hours(users):
    """End of the time axis: when both C_I and C_A have reached 99% of their final value, plus 12 h."""
    enter, side = entry_hours(users), users["side"].to_numpy()
    t99 = max(np.quantile(enter[side == s], TRIM_LEVEL, method="higher") for s in "IA")
    return t99 + TRIM_EXTRA_H


def real_curves(users, bin_minutes, end_h=None, lifespan_h=LIFESPAN_H):
    """Per bin t = k*bin hours after the first post:
    C_I, C_A = users who have entered I / A by t (cumulative, the fitting target);
    S, A, I, R = expected prevalence if every spreader stays active for an Exp(mean lifespan_h)
    time after entering (display only, DECISIONS D19). S + A + I + R = N exactly."""
    side = users["side"].to_numpy()
    enter = entry_hours(users)
    if end_h is None:
        end_h = trim_hours(users)
    n_bins = int(np.ceil(end_h * 60 / bin_minutes))
    N = len(users)
    rows = []
    for k in range(n_bins + 1):
        t = k * bin_minutes / 60
        entered = enter <= t
        alive = np.where(entered, np.exp(-(t - np.where(entered, enter, t)) / lifespan_h), 0.0)
        C_I = int(np.sum(entered & (side == "I")))
        C_A = int(np.sum(entered & (side == "A")))
        I = float(alive[side == "I"].sum())
        A = float(alive[side == "A"].sum())
        rows.append({"bin": k, "t_hours": t, "S": float(N - C_I - C_A), "A": A, "I": I,
                     "R": float((C_I - I) + (C_A - A)), "C_I": C_I, "C_A": C_A})
    curves = pd.DataFrame(rows)
    for c in ["S", "A", "I", "R", "C_I", "C_A"]:
        curves[c + "_frac"] = curves[c] / N
    curves.attrs["N"] = N
    return curves


def informative_bins(c, lo=0.05, hi=0.95):
    """Number of bins where a cumulative curve is between lo and hi of its final value."""
    c = np.asarray(c, dtype=float)
    return int(np.sum((c >= lo * c[-1]) & (c <= hi * c[-1])))


def max_bin_drop(y, min_share=0.1):
    """Largest one-bin relative drop of a prevalence curve, over bins where it is >= min_share of its peak."""
    y = np.asarray(y, dtype=float)
    prev, cur = y[:-1], y[1:]
    mask = prev >= min_share * y.max()
    return float(np.max(np.where(mask, (prev - cur) / np.where(prev > 0, prev, 1), 0)))


def saturation_tail(c, tail=0.1):
    """Share of the final cumulative total added in the last `tail` fraction of bins."""
    c = np.asarray(c, dtype=float)
    k = int(np.floor(len(c) * (1 - tail)))
    return float((c[-1] - c[k - 1]) / c[-1])


def compare_event(event, users):
    """Usability of an event under the rule: >= 100 A users and >= 15 informative bins for C_I and C_A
    (1-h bins, or 30-min bins if 1 h gives fewer than 15)."""
    nI, nA = int((users["side"] == "I").sum()), int((users["side"] == "A").sum())
    row = {"event": event, "N": len(users), "I_users": nI, "A_users": nA, "A_over_I": nA / nI}
    for b in BIN_OPTIONS:
        c = real_curves(users, b)
        row[f"C_I_bins_{b}m"] = informative_bins(c["C_I"])
        row[f"C_A_bins_{b}m"] = informative_bins(c["C_A"])
    ok = {b: min(row[f"C_I_bins_{b}m"], row[f"C_A_bins_{b}m"]) >= MIN_INFORMATIVE_BINS for b in BIN_OPTIONS}
    row["bin_minutes"] = next((b for b in BIN_OPTIONS if ok[b]), None)
    row["usable"] = nA >= MIN_A_USERS and row["bin_minutes"] is not None
    return row


# ---------------------------------------------------------------- diagnostics

def story_table(tweets):
    """Per rumour story (annotation 'category'): veracity, threads, users, users supporting / denying it."""
    g = tweets.groupby("story")
    table = pd.DataFrame({
        "veracity": g["story_veracity"].first(),
        "threads": g["thread_id"].nunique(),
        "users": g["user_id"].nunique(),
        "support_users": tweets[tweets["stance"] == "support"].groupby("story")["user_id"].nunique(),
        "deny_users": tweets[tweets["stance"] == "deny"].groupby("story")["user_id"].nunique(),
        "first_post": g["created_at"].min(),
    })
    table[["support_users", "deny_users"]] = table[["support_users", "deny_users"]].fillna(0).astype(int)
    return table.sort_values(["deny_users", "users"], ascending=False)


def both_stance_split(tweets, users):
    """For users who post both stances: do they support and deny the SAME story, or different stories?"""
    both = users.index[users["both_stances"]]
    t = tweets[tweets["user_id"].isin(both) & tweets["stance"].isin(SIDE)]
    same = diff = 0
    for _, d in t.groupby("user_id"):
        sup = set(d.loc[d["stance"] == "support", "story"])
        den = set(d.loc[d["stance"] == "deny", "story"])
        if sup & den:
            same += 1
        else:
            diff += 1
    return {"both_stance_users": len(both), "same_story": same, "different_stories": diff}


def stance_breakdown(tweets, users):
    """Side (I/A) by the kind of post that put the user there, plus the retweet sanity numbers."""
    tab = users.dropna(subset=["side"]).groupby(["side", "enter_kind"]).size().unstack(fill_value=0)
    rt = tweets[tweets["kind"] == "retweet"]
    neutral_rt = rt[~rt["source_support"].isin(SOURCE_STANCE)]
    info = {
        "retweets_of_neutral_sources": len(neutral_rt),
        "retweets_of_neutral_sources_sided": int(neutral_rt["stance"].isin(SIDE).sum()),
    }
    return tab, info


# ---------------------------------------------------------------- plots

def style(ax, title, ylabel, xlabel="Hours since first post"):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prevalence(curves, key, title, bin_minutes):
    """Three panels: S and R fractions; expected I users; expected A users."""
    N = curves.attrs["N"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    t = curves["t_hours"]
    ax = axes[0]
    for s, label in (("S", "S susceptible"), ("R", "R removed")):
        ax.plot(t, curves[s + "_frac"], color=COLORS[s], lw=2, label=label)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    style(ax, "S and R", "Fraction of users")
    for ax, s, label in ((axes[1], "I", "I - rumour spreaders"), (axes[2], "A", "A - correction spreaders")):
        ax.plot(t, curves[s], color=COLORS[s], lw=2)
        k = curves[s].idxmax()
        ax.plot(t[k], curves[s][k], "o", color=COLORS[s], ms=8, mec="white", mew=2)
        ax.annotate(f"peak {curves[s][k]:,.0f} at {t[k]:.1f} h", (t[k], curves[s][k]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#52514e")
        ax.set_ylim(0, curves[s].max() * 1.18)
        style(ax, label, "Users (expected)")
    fig.suptitle(f"{title}: expected S/A/I/R prevalence, display only "
                 f"(N = {N:,}, exponential lifespan mean {LIFESPAN_H} h, {bin_minutes}-min bins)",
                 fontsize=13, x=0.01, ha="left")
    return save(fig, f"real_curves_{key}.png")


def plot_cumulative(curves, key, title, bin_minutes):
    """C_I and C_A with the times they reach 50% and 90% of their final value."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    t = curves["t_hours"]
    for ax, c, s, label in ((axes[0], "C_I", "I", "C_I - users who entered I (supported)"),
                            (axes[1], "C_A", "A", "C_A - users who entered A (denied)")):
        y = curves[c]
        ax.plot(t, y, color=COLORS[s], lw=2)
        for q in (0.5, 0.9):
            k = int(np.argmax(y >= q * y.iloc[-1]))
            ax.plot(t[k], y[k], "o", color=COLORS[s], ms=8, mec="white", mew=2)
            ax.annotate(f"{int(q * 100)}% by {t[k]:.1f} h", (t[k], y[k]), textcoords="offset points",
                        xytext=(8, -12), fontsize=9, color="#52514e")
        ax.set_ylim(0, y.iloc[-1] * 1.12)
        style(ax, f"{label}: {y.iloc[-1]:,} users", "Users (cumulative)")
    fig.suptitle(f"{title}: cumulative entries into I and A, Phase 5 fitting target "
                 f"({bin_minutes}-min bins)", fontsize=13, x=0.01, ha="left")
    return save(fig, f"cumulative_curves_{key}.png")


# ---------------------------------------------------------------- main + check

def load_event(event, annotations):
    tweets_all = load_tweets(event, annotations)
    tweets, drop_info = veracity_filter(tweets_all)
    return tweets_all, tweets, build_users(tweets), drop_info


def compare_events(annotations):
    rows = []
    for event in CANDIDATES:
        _, _, users, drop_info = load_event(event, annotations)
        rows.append({**compare_event(event, users), **drop_info})
    table = pd.DataFrame(rows)
    usable = table[table["usable"]].sort_values("A_users", ascending=False)["event"].tolist()
    table["role"] = table["event"].map({e: r for e, r in zip(usable, ["primary", "secondary"])}).fillna("-")
    return table


def save_set(key, title, tweets, users, bin_minutes):
    """Curves + plots for one population (an event or the robustness story)."""
    curves = real_curves(users, bin_minutes)
    curves.to_csv(os.path.join(PROCESSED, f"real_curves_{key}.csv"), index=False)
    plot_prevalence(curves, key, title, bin_minutes)
    plot_cumulative(curves, key, title, bin_minutes)
    return curves


def prepare_event(event, annotations, bin_minutes):
    tweets_all, tweets, users, drop_info = load_event(event, annotations)
    G = build_graph(event, tweets, users)
    tweets.to_csv(os.path.join(PROCESSED, f"tweets_{event}.csv"), index=False)
    users.to_csv(os.path.join(PROCESSED, f"users_{event}.csv"))
    with open(os.path.join(PROCESSED, f"graph_{event}.gpickle"), "wb") as f:
        pickle.dump(G, f)
    curves = save_set(event, EVENT_NAMES[event], tweets, users, bin_minutes)
    story_table(tweets_all).to_csv(os.path.join(RESULTS, f"stories_{event}.csv"))
    breakdown, rt_info = stance_breakdown(tweets, users)
    breakdown.insert(0, "event", event)
    summary = {
        "set": event,
        **drop_info,
        "threads_kept": tweets["thread_id"].nunique(),
        "stories_kept": tweets["story"].nunique(),
        "posts": len(tweets),
        "users_N": len(users),
        "users_I_side": int((users["side"] == "I").sum()),
        "users_A_side": int((users["side"] == "A").sum()),
        "users_neutral_S": int(users["side"].isna().sum()),
        **both_stance_split(tweets, users),
        **rt_info,
        **graph_stats(G),
        **curve_summary(curves, bin_minutes),
    }
    return summary, breakdown


def curve_summary(curves, bin_minutes):
    return {
        "bin_minutes": bin_minutes,
        "n_bins": len(curves),
        "end_h": float(curves["t_hours"].iloc[-1]),
        "C_I_informative_bins": informative_bins(curves["C_I"]),
        "C_A_informative_bins": informative_bins(curves["C_A"]),
        "peak_I": round(float(curves["I"].max()), 1),
        "peak_I_h": float(curves.loc[curves["I"].idxmax(), "t_hours"]),
        "peak_A": round(float(curves["A"].max()), 1),
        "peak_A_h": float(curves.loc[curves["A"].idxmax(), "t_hours"]),
        "final_C_I": int(curves["C_I"].iloc[-1]),
        "final_C_A": int(curves["C_A"].iloc[-1]),
        "max_I_drop_1bin": max_bin_drop(curves["I"]),
    }


def prepare_story(annotations):
    """Single-story robustness set (Phase 5): one FALSE story, 15-min bins (DECISIONS D23)."""
    tweets_all = load_tweets(STORY["event"], annotations)
    tweets = tweets_all[tweets_all["story"] == STORY["story"]].reset_index(drop=True)
    assert set(tweets["story_veracity"]) <= KEEP_VERACITY, "robustness story must be false/unverified"
    users = build_users(tweets)
    tweets.to_csv(os.path.join(PROCESSED, f"tweets_{STORY['key']}.csv"), index=False)
    users.to_csv(os.path.join(PROCESSED, f"users_{STORY['key']}.csv"))
    curves = save_set(STORY["key"], f"Ottawa story \"{STORY['story']}\"", tweets, users, STORY["bin_minutes"])
    return {"set": STORY["key"], "threads_kept": tweets["thread_id"].nunique(), "posts": len(tweets),
            "users_N": len(users), "users_I_side": int((users["side"] == "I").sum()),
            "users_A_side": int((users["side"] == "A").sum()),
            "users_neutral_S": int(users["side"].isna().sum()), **curve_summary(curves, STORY["bin_minutes"])}


def main():
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    annotations = load_annotations()
    comparison = compare_events(annotations)
    comparison.to_csv(os.path.join(RESULTS, "phase1_event_comparison.csv"), index=False)
    print("=== Event comparison (false + unverified stories only) ===")
    print(comparison.to_string(index=False))
    if "--compare" in sys.argv:
        return
    bins = comparison.set_index("event")["bin_minutes"]
    summaries, breakdowns = [], []
    for event in EVENTS:
        s, br = prepare_event(event, annotations, int(bins[event]))
        summaries.append(s)
        breakdowns.append(br)
    summaries.append(prepare_story(annotations))
    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(RESULTS, "phase1_summary.csv"), index=False)
    breakdown = pd.concat(breakdowns).reset_index()
    breakdown.to_csv(os.path.join(RESULTS, "phase1_stance_breakdown.csv"), index=False)
    print("\n=== Stance breakdown: users by side and the kind of post that put them there ===")
    print(breakdown.to_string(index=False))
    print("\n=== Summary ===")
    print(summary.set_index("set").T.to_string())


def check_curves(curves, N, label):
    """Invariants shared by every curve set."""
    prev = ["S_frac", "A_frac", "I_frac", "R_frac"]
    total = curves[list("SAIR")].sum(axis=1)
    assert np.allclose(total, N, atol=1e-9 * N, rtol=0), f"{label}: S+A+I+R != N"
    frac_sum = curves[prev].sum(axis=1)
    assert np.allclose(frac_sum, 1.0, atol=1e-9, rtol=0), f"{label}: fractions do not sum to 1"
    assert (curves[list("SAIR")] >= -1e-9).all().all(), f"{label}: negative values"
    assert curves[prev].apply(lambda c: c.between(-1e-12, 1 + 1e-12).all()).all(), f"{label}: fraction outside [0,1]"
    assert curves["R"].diff().dropna().ge(-1e-9).all(), f"{label}: R decreases"
    assert curves["S"].diff().dropna().le(1e-9).all(), f"{label}: S increases"
    drop = max_bin_drop(curves["I"])
    assert drop <= 0.5, f"{label}: I prevalence drops {drop:.0%} in one bin (cliff)"
    for c in ("C_I", "C_A"):
        assert curves[c].is_monotonic_increasing, f"{label}: {c} decreases"
        tail = saturation_tail(curves[c])
        assert tail < 0.05, f"{label}: {c} not saturated, last 10% of bins add {tail:.1%}"
    assert ((curves["C_I"] >= curves["I"] - 1e-9) & (curves["C_A"] >= curves["A"] - 1e-9)).all()
    return frac_sum, drop


def check():
    """Phase 1 checks on the saved outputs. Nothing is bypassed."""
    print("\n--- Phase 1 check ---")
    comparison = pd.read_csv(os.path.join(RESULTS, "phase1_event_comparison.csv")).set_index("event")
    for event in EVENTS:
        assert comparison.at[event, "usable"], f"{event} is not usable under the event rule"
        assert min(comparison.at[event, f"C_I_bins_{int(comparison.at[event, 'bin_minutes'])}m"],
                   comparison.at[event, f"C_A_bins_{int(comparison.at[event, 'bin_minutes'])}m"]) >= MIN_INFORMATIVE_BINS
    assert comparison.at[EVENTS[0], "role"] == "primary" and comparison.at[EVENTS[1], "role"] == "secondary"

    for key in EVENTS + [STORY["key"]]:
        tweets = pd.read_csv(os.path.join(PROCESSED, f"tweets_{key}.csv"),
                             dtype={"tweet_id": str, "user_id": str, "story_veracity": str})
        users = pd.read_csv(os.path.join(PROCESSED, f"users_{key}.csv"), dtype={"user_id": str}).set_index("user_id")
        curves = pd.read_csv(os.path.join(PROCESSED, f"real_curves_{key}.csv"))
        N = len(users)

        # data
        assert tweets["tweet_id"].is_unique, "duplicate tweet ids"
        assert set(tweets["stance"]) <= STANCES, f"bad stance values {set(tweets['stance']) - STANCES}"
        assert set(tweets["story_veracity"]) <= KEEP_VERACITY, "a TRUE story is in the data"
        assert set(users["side"].dropna()) <= {"I", "A"}
        assert set(users.index) == set(tweets["user_id"]), "users table != posting users"
        rt = tweets[tweets["kind"] == "retweet"]
        expected = rt["source_support"].map(SOURCE_STANCE).fillna("comment")
        assert (rt["stance"] == expected).all(), "retweet stance differs from its source stance"
        assert not rt.loc[~rt["source_support"].isin(SOURCE_STANCE), "stance"].isin(SIDE).any(), \
            "retweet of a neutral source assigned to I or A"

        # curves
        frac_sum, drop = check_curves(curves, N, key)
        cut = {}
        for c, s in (("C_I", "I"), ("C_A", "A")):     # trimming may leave out < 1% late entries (D22)
            total = int((users["side"] == s).sum())
            assert TRIM_LEVEL * total <= curves[c].iloc[-1] <= total, f"{key}: {c} lost more than 1% to trimming"
            cut[s] = total - int(curves[c].iloc[-1])

        # graph (events only)
        if key in EVENTS:
            with open(os.path.join(PROCESSED, f"graph_{key}.gpickle"), "rb") as f:
                G = pickle.load(f)
            assert set(G.nodes) == set(users.index), "graph nodes != users"
            assert set(nx.get_edge_attributes(G, "kind").values()) <= {"follow", "interaction"}
            assert G.number_of_edges() == len(nx.get_edge_attributes(G, "kind"))

        print(f"{key:22s} OK  N={N:,}  bins={len(curves)}  max|sum frac - 1|={np.abs(frac_sum - 1).max():.1e}  "
              f"max 1-bin I drop={drop:.1%}  tail C_I={saturation_tail(curves['C_I']):.2%}  "
              f"tail C_A={saturation_tail(curves['C_A']):.2%}  entries after trim I/A={cut['I']}/{cut['A']}")
    print("All Phase 1 checks passed.")


if __name__ == "__main__":
    main()
    if "--check" in sys.argv:
        check()
