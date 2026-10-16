"""Phase 1 - data preparation.

Loads PHEME rumour-scheme threads (source tweets, replies, retweets), assigns each tweet a
stance toward the rumour, builds the user graph and the user table, and turns user states
into real S/A/I/R curves.

Usage:
    python src/data_prep.py           build everything
    python src/data_prep.py --check   build everything, then run the Phase 1 checks
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

EVENTS = ["sydneysiege", "ottawashooting"]   # primary first (DECISIONS D11)
EVENT_NAMES = {"sydneysiege": "Sydney Siege", "ottawashooting": "Ottawa shooting"}
STANCES = {"support", "deny", "query", "comment"}
INACTIVE = pd.Timedelta(hours=24)            # R = no post for 24 h
BIN_CANDIDATES = [60, 30, 15]                # minutes, preferred first
MIN_POINTS = 20

# reply stance toward the rumour = source stance composed with reply-vs-source (DECISIONS D3)
COMPOSE = {("supporting", "agreed"): "support", ("supporting", "disagreed"): "deny",
           ("denying", "agreed"): "deny", ("denying", "disagreed"): "support"}
SOURCE_STANCE = {"supporting": "support", "denying": "deny"}
SIDE = {"support": "I", "deny": "A"}

# categorical palette slots 1-4 (dataviz reference palette, light mode)
COLORS = {"S": "#2a78d6", "I": "#eb6834", "A": "#1baf7a", "R": "#eda100"}
BIN_COLORS = {60: "#2a78d6", 30: "#eb6834", 15: "#1baf7a"}


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


def load_tweets(event):
    """All posts of an event: source tweets, replies and retweets of the source, one row each."""
    rows = []
    for th, d in thread_dirs(event):
        for sub, kind in (("source-tweets", "source"), ("reactions", "reply")):
            for fn in sorted(os.listdir(os.path.join(d, sub))):
                if fn.endswith(".json") and not fn.startswith("."):
                    with open(os.path.join(d, sub, fn), encoding="utf-8") as f:
                        rows.append(tweet_row(json.load(f), th, kind))
        rt_path = os.path.join(d, "retweets.json")
        if os.path.exists(rt_path):
            rows += [tweet_row(t, th, "retweet") for t in read_json_lines(rt_path)]
    tweets = pd.DataFrame(rows)
    n_before = len(tweets)
    tweets = tweets.drop_duplicates("tweet_id").sort_values("created_at").reset_index(drop=True)
    tweets.attrs["n_duplicates_dropped"] = n_before - len(tweets)
    return tweets


def load_stance(tweets, annotations):
    """Stance of each tweet toward the rumour: support / deny / query / comment (DECISIONS D3, D4)."""
    source_support = {th: annotations.get(th, {}).get("support") for th in tweets["thread_id"].unique()}
    stance = []
    for tid, th, kind in zip(tweets["tweet_id"], tweets["thread_id"], tweets["kind"]):
        src = source_support[th]
        if kind in ("source", "retweet"):          # a retweet inherits the source stance
            s = SOURCE_STANCE.get(src, "comment")
        else:
            r = annotations.get(tid, {}).get("responsetype-vs-source")
            s = COMPOSE.get((src, r), "query" if r == "appeal-for-more-information" else "comment")
        stance.append(s)
    return pd.Series(stance, index=tweets.index, name="stance")


# ---------------------------------------------------------------- users and graph

def removal_time(times):
    """Time a user becomes R: 24 h after the first post that is followed by a 24 h gap."""
    times = sorted(times)
    for a, b in zip(times, times[1:]):
        if b - a > INACTIVE:
            return a + INACTIVE
    return times[-1] + INACTIVE


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
    users["both_stances"] = (users["n_support"] > 0) & (users["n_deny"] > 0)
    t_remove = {}
    for uid, grp in t[t["user_id"].isin(sided.index)].groupby("user_id"):
        after = grp.loc[grp["created_at"] >= users.at[uid, "t_enter"], "created_at"]
        t_remove[uid] = removal_time(list(after))
    users["t_remove"] = pd.Series(t_remove).reindex(users.index)
    users.index.name = "user_id"
    return users


def build_graph(event, tweets, users):
    """Directed graph, edge u -> v = u receives information from v.
    kind='follow' from who-follows-whom (u follows v); kind='interaction' when u replied to or
    retweeted v and no follow edge exists (DECISIONS D2)."""
    G = nx.DiGraph(event=event)
    G.add_nodes_from(users.index)
    for _, d in thread_dirs(event):
        path = os.path.join(d, "who-follows-whom.dat")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                a = line.split()
                if len(a) == 2 and a[0] != a[1] and a[0] in G and a[1] in G:
                    G.add_edge(a[0], a[1], kind="follow")
    author = tweets.set_index("tweet_id")["user_id"]
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


# ---------------------------------------------------------------- curves

def real_curves(users, bin_minutes):
    """S, A, I, R counts and fractions at t0 + k*bin, until every I/A user has become R."""
    t0 = users["first_t"].min()
    dt = pd.Timedelta(minutes=bin_minutes)
    n_bins = int(np.ceil((users["t_remove"].max() - t0) / dt)) + 1
    side = users["side"].to_numpy()
    hours = pd.Timedelta(hours=1)
    enter = ((pd.to_datetime(users["t_enter"], utc=True) - t0) / hours).to_numpy(dtype=float)   # NaN = never sided
    remove = ((pd.to_datetime(users["t_remove"], utc=True) - t0) / hours).to_numpy(dtype=float)
    N = len(users)
    rows = []
    for k in range(n_bins + 1):
        t = k * bin_minutes / 60
        active = (enter <= t) & ~(remove <= t)
        I = int(np.sum(active & (side == "I")))
        A = int(np.sum(active & (side == "A")))
        R = int(np.sum(remove <= t))
        rows.append({"bin": k, "t_hours": k * bin_minutes / 60, "S": N - I - A - R, "A": A, "I": I, "R": R})
    curves = pd.DataFrame(rows)
    for c in "SAIR":
        curves[c + "_frac"] = curves[c] / N
    curves.attrs["N"] = N
    return curves


def curve_shape(y):
    """Points with the curve above zero, and whether it rises then falls (interior peak, ends below 10% of peak)."""
    y = np.asarray(y)
    nz = np.nonzero(y)[0]
    if len(nz) == 0:
        return 0, 0, False
    peak = int(np.argmax(y))
    n_points = int(nz[-1] - nz[0] + 1)
    n_rise = int(peak - nz[0])
    ok = 0 < peak < len(y) - 1 and y[-1] < 0.1 * y[peak] and n_rise >= 3
    return n_points, n_rise, ok


def choose_bin(users):
    """Largest bin (1 h preferred) whose I and A curves both rise and fall with >= MIN_POINTS points."""
    table, curves = [], {}
    for b in BIN_CANDIDATES:
        c = real_curves(users, b)
        curves[b] = c
        row = {"bin_minutes": b, "n_bins": len(c)}
        for s in "IA":
            n_points, n_rise, ok = curve_shape(c[s])
            row.update({f"{s}_points": n_points, f"{s}_rise_points": n_rise, f"{s}_rise_fall": ok,
                        f"{s}_peak": int(c[s].max()), f"{s}_peak_h": float(c.loc[c[s].idxmax(), "t_hours"])})
        row["ok"] = all(row[f"{s}_rise_fall"] and row[f"{s}_points"] >= MIN_POINTS for s in "IA")
        table.append(row)
    table = pd.DataFrame(table)
    passing = table[table["ok"]]
    chosen = int(passing["bin_minutes"].iloc[0]) if len(passing) else BIN_CANDIDATES[-1]
    return chosen, table, curves


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


def plot_curves(curves, event, bin_minutes):
    """Three panels: S and R fractions; I users; A users (each on its own scale, same x)."""
    name = EVENT_NAMES[event]
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
        ax.annotate(f"peak {curves[s][k]:,} at {t[k]:.1f} h", (t[k], curves[s][k]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#52514e")
        ax.set_ylim(0, curves[s].max() * 1.18)
        style(ax, label, "Users")
    fig.suptitle(f"{name}: real S/A/I/R from PHEME (N = {N:,} users, {bin_minutes}-min bins)",
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(RESULTS, f"real_curves_{event}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bin_comparison(all_curves, event):
    """I and A fractions for 15/30/60-min bins on the same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, s, label in ((axes[0], "I", "I - rumour spreaders"), (axes[1], "A", "A - correction spreaders")):
        for b in BIN_CANDIDATES:
            c = all_curves[b]
            ax.plot(c["t_hours"], c[s + "_frac"], color=BIN_COLORS[b], lw=2 if b == 60 else 1.5,
                    label=f"{b}-min bins ({len(c)} points)")
        ax.legend(frameon=False)
        style(ax, label, "Fraction of users")
    fig.suptitle(f"{EVENT_NAMES[event]}: effect of bin size on real I and A curves",
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(RESULTS, f"bin_comparison_{event}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- main + check

def prepare_event(event, annotations):
    tweets = load_tweets(event)
    tweets["stance"] = load_stance(tweets, annotations)
    users = build_users(tweets)
    G = build_graph(event, tweets, users)
    chosen, bin_table, all_curves = choose_bin(users)
    curves = all_curves[chosen]

    tweets.to_csv(os.path.join(PROCESSED, f"tweets_{event}.csv"), index=False)
    users.to_csv(os.path.join(PROCESSED, f"users_{event}.csv"))
    curves.to_csv(os.path.join(PROCESSED, f"real_curves_{event}.csv"), index=False)
    with open(os.path.join(PROCESSED, f"graph_{event}.gpickle"), "wb") as f:
        pickle.dump(G, f)
    bin_table.insert(0, "event", event)
    bin_table["chosen"] = bin_table["bin_minutes"] == chosen
    plot_curves(curves, event, chosen)
    plot_bin_comparison(all_curves, event)

    kinds = nx.get_edge_attributes(G, "kind")
    summary = {
        "event": event,
        "threads": tweets["thread_id"].nunique(),
        "posts": len(tweets),
        "sources": int((tweets["kind"] == "source").sum()),
        "replies": int((tweets["kind"] == "reply").sum()),
        "retweets": int((tweets["kind"] == "retweet").sum()),
        "duplicates_dropped": tweets.attrs["n_duplicates_dropped"],
        "users_N": len(users),
        "users_I_side": int((users["side"] == "I").sum()),
        "users_A_side": int((users["side"] == "A").sum()),
        "users_neutral_S": int(users["side"].isna().sum()),
        "users_both_stances": int(users["both_stances"].sum()),
        "edges_follow": sum(k == "follow" for k in kinds.values()),
        "edges_interaction": sum(k == "interaction" for k in kinds.values()),
        "isolated_nodes": nx.number_of_isolates(G),
        "bin_minutes": chosen,
        "n_bins": len(curves),
        "peak_I": int(curves["I"].max()),
        "peak_I_h": float(curves.loc[curves["I"].idxmax(), "t_hours"]),
        "peak_A": int(curves["A"].max()),
        "peak_A_h": float(curves.loc[curves["A"].idxmax(), "t_hours"]),
        "final_R": int(curves["R"].iloc[-1]),
        "final_S": int(curves["S"].iloc[-1]),
    }
    return summary, bin_table


def main():
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    annotations = load_annotations()
    summaries, bin_tables = [], []
    for event in EVENTS:
        s, b = prepare_event(event, annotations)
        summaries.append(s)
        bin_tables.append(b)
    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(RESULTS, "phase1_summary.csv"), index=False)
    bins = pd.concat(bin_tables, ignore_index=True)
    bins.to_csv(os.path.join(RESULTS, "phase1_bin_choice.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary.set_index("event").T.to_string())
    print()
    print(bins[["event", "bin_minutes", "n_bins", "I_points", "I_rise_points", "I_rise_fall", "A_points",
                "A_rise_points", "A_rise_fall", "ok", "chosen"]].to_string(index=False))


def check():
    """Phase 1 checks on the saved outputs."""
    print("\n--- Phase 1 check ---")
    for event in EVENTS:
        tweets = pd.read_csv(os.path.join(PROCESSED, f"tweets_{event}.csv"), dtype={"tweet_id": str, "user_id": str})
        users = pd.read_csv(os.path.join(PROCESSED, f"users_{event}.csv"), dtype={"user_id": str}).set_index("user_id")
        curves = pd.read_csv(os.path.join(PROCESSED, f"real_curves_{event}.csv"))
        with open(os.path.join(PROCESSED, f"graph_{event}.gpickle"), "rb") as f:
            G = pickle.load(f)
        N = len(users)

        assert tweets["tweet_id"].is_unique, "duplicate tweet ids"
        assert set(tweets["stance"]) <= STANCES, f"bad stance values {set(tweets['stance']) - STANCES}"
        assert set(users["side"].dropna()) <= {"I", "A"}
        assert (curves[list("SAIR")].sum(axis=1) == N).all(), "S+A+I+R != N"
        frac_sum = curves[["S_frac", "A_frac", "I_frac", "R_frac"]].sum(axis=1)
        assert np.allclose(frac_sum, 1.0, atol=1e-9, rtol=0), "fractions do not sum to 1"
        assert (curves[list("SAIR")] >= 0).all().all(), "negative counts"
        assert curves[["S_frac", "A_frac", "I_frac", "R_frac"]].apply(lambda c: c.between(0, 1).all()).all()
        assert curves["R"].is_monotonic_increasing, "R must never decrease"
        assert curves["S"].is_monotonic_decreasing, "S must never increase"
        for s in "IA":
            y = curves[s].to_numpy()
            peak = int(np.argmax(y))
            assert 0 < peak < len(y) - 1, f"{s} peak not interior"
            assert y[-1] < y[peak], f"{s} does not fall after its peak"
        assert set(G.nodes) <= set(users.index), "graph nodes not all users"
        assert set(nx.get_edge_attributes(G, "kind").values()) <= {"follow", "interaction"}
        assert G.number_of_edges() == len(nx.get_edge_attributes(G, "kind"))
        print(f"{event:15s} OK  N={N:,}  bins={len(curves)}  edges={G.number_of_edges():,}  "
              f"max|sum frac - 1|={np.abs(frac_sum - 1).max():.1e}")
    print("All Phase 1 checks passed.")


if __name__ == "__main__":
    main()
    if "--check" in sys.argv:
        check()
