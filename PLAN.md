# Implementation Phase Plan

Each phase: plan → code → run check → show results → stop for approval.
Equations and definitions live in `docs/SAIR_paper.md` and `docs/extension.md`.

## File structure
```
src/
  data_prep.py   Phase 1
  features.py    Phase 2
  network.py     Phase 3
  drive.py       Phase 4
  sair.py        Phase 5–6
  control.py     Phase 7
  targeting.py   Phase 8
notebooks/experiments.ipynb   runs everything end-to-end
results/                      plots + tables
```

---

## Phase 1 — Data preparation  (`src/data_prep.py`)
1. Load PHEME rumour-scheme events (sources, replies, retweets); keep only FALSE + UNVERIFIED stories.
2. Stance toward the rumour (reply stance composed with source stance; retweets inherit source stance).
   Users: first support/deny post fixes I or A; query/comment-only → S.
3. Pick events by rule (≥ 100 A users, ≥ 15 informative bins for C_I and C_A): Charlie Hebdo (primary), Sydney Siege (secondary), 1-h bins.
4. Build graph G: follow edges + reply/retweet interaction edges (edge attribute `kind`). User table.
5. Cumulative entry curves C_I(t), C_A(t) = fitting target, trimmed at 99% + 12 h. Expected S/A/I/R prevalence (exponential lifespan, mean 12 h) = display only.
6. Robustness set: Ottawa story "three separate shooting incidents", 15-min bins.
7. Diagnostics: event comparison, stories per event, stance breakdown by post kind, graph stats.

**Output:** `data/processed/graph_<event>.gpickle`, `users_<event>.csv`, `real_curves_<event>.csv` (prevalence + C_I, C_A); `results/phase1_*.csv`, plots.
**Check:** no TRUE stories; S+A+I+R = N every bin; C_I, C_A non-decreasing and saturated (last 10% of bins add < 5%); I prevalence drops ≤ 50% in one bin; retweets of neutral sources never I/A; chosen events pass the event rule.

## Phase 2 — User features  (`src/features.py`)  — docs/SAIR_paper.md §2
Only the user's OWN posts before t_ref + profile description (no retweeted text).
1. Emotion vectors (NRC, 8 dims, recency-weighted) → emotional similarity (cosine) with supporting sources.
2. Tags: TF-IDF top keywords + hashtags of own text; rumour = same from supporting source tweets.
3. Content fit (Jaccard). 4. Popularity Pop(t) = Σ_s w_s·0.5^((t−t0_s)/w), one term per story, w_s = log(1 + author followers).
5. Connectivity (tag Jaccard + friend-list Jaccard over follow friends). 6. Friend power (log1p mean of timestamped replies + retweets friends received before entry, excluding the user's own).

**Output:** `data/processed/features_<set>.csv`, `pop_<set>.csv`; `results/phase2_*.csv`, `pop_vs_entries_charliehebdo.png`.
**Check:** ranges valid, no NaN; coverage report; leakage check (single-feature AUC ≤ 0.95, class balance); Pop waves vs entries plot.

## Phase 3 — Network structure  (`src/network.py`)  — docs/SAIR_paper.md §3
1. PageRank on follow edges only (u→v = u follows v) → core = top 1% (0.5% / 2% sensitivity).
2. Layers: strong = follows a core node, weak = rest.
3. mpr = τ(k1·pr_norm + k2·Info_match) (Eq. 17); r = 1 − mpr_norm clipped to [0.05, 0.95].
4. Groups: Louvain on the undirected follow graph (no Prb from event retweets — outcome leakage).
5. Gact = mean log1p(statuses_count); Gpre = Jaccard(group top-30 TF-IDF keywords of own pre-entry text + profiles, rumour tags); Group_att = Gact_norm × Gpre.

**Output:** `data/processed/network_<set>.csv`, `groups_<set>.csv`; `results/phase3_*.csv`, layer and group-size plots.
**Check:** core small, layers sum to N; one group per user (count, sizes, modularity); r in [0.05, 0.95]; leakage AUC ≤ 0.95.

## Phase 4 — Influence & push  (`src/drive.py`)  — docs/SAIR_paper.md §4
1. Blocks: User_att, hnode_inf (low r), lnode_inf (high r), Group_att.
2. Regression (target = reposted rumour 1/0) → γ0..γ4 → Mult_eff.
3. P1 per user from friends' states. 4. Drif1, Drif2.

**Output:** mult_eff, drif1, drif2 per user. **Check:** coefficients printed; Drif1+Drif2 = 1.

## Phase 5 — SAIR simulation & validation  (`src/sair.py`)  — docs/SAIR_paper.md §5
1. δ1(t), δ2(t) = mean pushes over susceptible users, recomputed each hour.
2. Euler loop of the corrected SAIR equations.
3. Fit free params (half-life w, scale) to the real **cumulative** curves C_I, C_A (model: cumulative inflow ∫δ1·I·S, ∫δ2·A·S) with scipy.optimize.minimize.
4. RMSE (predicted vs real C_I and C_A); compare shape/timing, not absolute peak height.

**Output:** Plot 1 (S/A/I/R curves), Plot 2 (predicted vs real + RMSE). **Check:** sum = 1 each step.

## Phase 6 — Ablation  (`src/sair.py`)  — docs/SAIR_paper.md §6
Re-run with emotion removed (S_emo=1) and popularity removed (Pop=1); optional others.
**Output:** Plot 3. **Check:** no-popularity never peaks; no-emotion has a lower peak.

## Phase 7 — SAIR-C  (`src/control.py`)  — docs/extension.md §1
1. Add B, knobs u and v; accumulate cost J.
2. Four strategies: none / block / refute / hybrid.
3. Grid search u ∈ [0,0.5], v ∈ [0,0.1] → cost heatmap.

**Output:** Plot 4 (I(t) per strategy + cost bars), Plot 5 (heatmap, minimum marked).
**Check:** S+A+I+B+R = 1; report which strategy is cheapest.

## Phase 8 — Centrality targeting  (`src/targeting.py`)  — docs/extension.md §2
1. Degree, PageRank, betweenness (k=500 sampling), random (15 runs averaged).
2. For k ∈ {10,25,50,100}: block / recruit / hybrid (k/2 + k/2).
3. Recompute P1 → Drif1 → δ1 → run SAIR-C; record peak and total reach.

**Output:** Plot 6 (peak vs k per method), Plot 7 (bars at fixed k incl. hybrid), results CSV.
**Check:** same k for all methods; random averaged.

## Phase 9 — Packaging
1. Results table (RMSE, strategy costs, peak reduction per centrality).
2. README: problem, pipeline, contributions, corrections (from DECISIONS.md), limitations, future work.
3. `notebooks/experiments.ipynb` runs top to bottom.
