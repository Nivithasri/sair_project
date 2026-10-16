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
1. Load PHEME rumour-scheme events Sydney Siege (primary) + Ottawa shooting: sources, replies, retweets.
2. Stance toward the rumour (reply stance composed with source stance; retweets inherit source stance).
   Users: first support/deny post fixes I or A; query/comment-only → S.
3. Build graph G: follow edges + reply/retweet interaction edges (edge attribute `kind`).
4. User table: user_id, posts, timestamps, hashtags, stance, user fields.
5. Time bins (1 h default; 15/30 min compared, choice per event logged); real S, A, I, R counts (R = inactive 24 h).

**Output:** `data/processed/graph_<event>.gpickle`, `users_<event>.csv`, `real_curves_<event>.csv`
**Check:** curves plotted; S+A+I+R = total users every hour; I and A rise then fall.

## Phase 2 — User features  (`src/features.py`)  — docs/SAIR_paper.md §2
1. Emotion vectors (NRC, 8 dims, recency-weighted) → emotional similarity (cosine).
2. Tags: user = top keywords/hashtags of own tweets; rumour = keywords of rumour text.
3. Content fit (Jaccard). 4. Popularity Pop(t) = 0.5^((t−t0)/w).
5. Connectivity (tag sim + friend-list Jaccard; friends = in-dataset follows). 6. Friend power (avg engagement of friends).

**Output:** feature columns in users table. **Check:** ranges valid, no NaN.

## Phase 3 — Network structure  (`src/network.py`)  — docs/SAIR_paper.md §3
1. PageRank → top-k core. 2. Layers: strong = linked to core, weak = rest.
3. Efficiency loss r per layer (from mpr, Eq. 17).
4. Prb matrix (same action on same post). 5. Groups (KNN similarity graph → clusters).
6. Gact (activity) and Gpre (TextRank keywords of group posts inside the event window vs rumour tags, Jaccard).

**Output:** layer + group per user, group score table. **Check:** core small; group sizes sensible.

## Phase 4 — Influence & push  (`src/drive.py`)  — docs/SAIR_paper.md §4
1. Blocks: User_att, hnode_inf (low r), lnode_inf (high r), Group_att.
2. Regression (target = reposted rumour 1/0) → γ0..γ4 → Mult_eff.
3. P1 per user from friends' states. 4. Drif1, Drif2.

**Output:** mult_eff, drif1, drif2 per user. **Check:** coefficients printed; Drif1+Drif2 = 1.

## Phase 5 — SAIR simulation & validation  (`src/sair.py`)  — docs/SAIR_paper.md §5
1. δ1(t), δ2(t) = mean pushes over susceptible users, recomputed each hour.
2. Euler loop of the corrected SAIR equations.
3. Fit free params (half-life w, scale) to real curves with scipy.optimize.minimize.
4. RMSE (predicted vs real I and A).

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
