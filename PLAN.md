# Implementation Phase Plan

Each phase: plan → code → run check → show results → stop for approval.
**Status:** Phases 1–5 done (Phase 5 revised, D50–D56). Phases 6–9 not started. Phase 8 must include the BA network.
See `docs/HANDOFF.md` for results and findings.
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

## Phase 1 — Data preparation  (`src/data_prep.py`)  ✅ DONE
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

## Phase 2 — User features  (`src/features.py`)  — docs/SAIR_paper.md §2  ✅ DONE
Only the user's OWN posts before t_ref + profile description (no retweeted text).
1. Emotion vectors (NRC, 8 dims, recency-weighted) → emotional similarity (cosine) with supporting sources.
2. Tags: TF-IDF top keywords + hashtags of own text; rumour = same from supporting source tweets.
3. Content fit (Jaccard). 4. Popularity Pop(t) = Σ_s w_s·0.5^((t−t0_s)/w), one term per story, w_s = log(1 + author followers).
5. Connectivity (tag Jaccard + friend-list Jaccard over follow friends). 6. Friend power (log1p mean of timestamped replies + retweets friends received before entry, excluding the user's own).

**Output:** `data/processed/features_<set>.csv`, `pop_<set>.csv`; `results/phase2_*.csv`, `pop_vs_entries_charliehebdo.png`.
**Check:** ranges valid, no NaN; coverage report; leakage check (single-feature AUC ≤ 0.95, class balance); Pop waves vs entries plot.

## Phase 3 — Network structure  (`src/network.py`)  — docs/SAIR_paper.md §3  ✅ DONE
1. PageRank on follow edges only (u→v = u follows v) → core = top 1% (0.5% / 2% sensitivity).
2. Layers: strong = follows a core node, weak = rest.
3. mpr = τ(k1·pr_norm + k2·Info_match) (Eq. 17); r = 1 − mpr_norm clipped to [0.05, 0.95].
4. Groups: KMeans on profile-only features, k = 20, NMI(group, thread) < 0.3; follow-Louvain as robustness variant (no Prb from event retweets — outcome leakage).
5. Gact = mean log1p(statuses_count); Gpre = Jaccard(group top-30 TF-IDF keywords of own pre-entry text + profiles, rumour tags); Group_att = Gact_norm × Gpre.

**Output:** `data/processed/network_<set>.csv`, `groups_<set>.csv`; `results/phase3_*.csv`, layer and group-size plots.
**Check:** core small, layers sum to N; one group per user (count, sizes, silhouette / modularity); NMI(group, first thread) < 0.3; r in [0.05, 0.95]; leakage AUC ≤ 0.95.

## Phase 4 — Influence & push  (`src/drive.py`)  — docs/SAIR_paper.md §4  ✅ DONE
1. Blocks from pre-entry info: User_att = S_emo·Info_match·Pop(t_ref), hnode_inf / lnode_inf (friends in same-or-higher / lower layer: co·fp·(1−r)), Group_att = Gact_norm·Gpre; 90% degeneracy fallbacks.
2. Logistic regression on sided users (y = first side I), standardised blocks, class_weight balanced → γ0..γ4; CV AUC stratified + thread-held-out.
3. Mult_eff = κ·percentile_rank(η), κ = 2. P1 per user from friends who took a side (fallback global share). 4. Drif1, Drif2; `drif(ctx, t)` for Phase 5.

**Output:** `data/processed/drive_<set>.csv`, `drive_model_<set>.json`, `friends_<set>.csv`; `results/phase4_*.csv`, Drif plots.
**Check:** Drif1 + Drif2 = 1; Mult_eff in [0, κ]; no NaN; degeneracy table; coefficients; CV AUCs ≤ 0.95 (flag gap > 0.10); worked example 0.769.

## Phase 5 — SAIR simulation & validation  (`src/sair.py`)  — docs/SAIR_paper.md §5  ✅ DONE (revised)
First version `cee4567` (D45–D49); revision D50–D56.
Common setup for every model: fractions of N; Euler dt = bin/50 (0.02 h events, 0.005 h story), events hit exactly
(D47, D53); source injection θ·w_s/Σw (D46); I0, A0 from the first bin; target = cumulative C_I, C_A (normalised by the
finals); least squares; bounded Nelder-Mead (log space for rates and w), 20 random starts, seed 42 (D51).
1. Models: (a) constant-rate SAIR (Eq. 22) · (b) Eq. 26, uniform Mult_eff · (c) Eq. 26 with data-driven δ (D45) ·
   **(d) SAIR-H** (main model, D50) · (d)-uniform · (e) SAIR-H mean-field (global P1).
2. AIC / BIC per set and model; bound flags; converged counts.
3. Validation (D52): forecast from the 50% point of C_I (data inputs frozen at the cut); parameter recovery (CH, (d),
   σ = 0.01, 5 seeds × 10 starts); residual bootstrap ((a), (d), 50 resamples, warm start).
4. **Result (D54–D56): (d) does NOT beat (a) on AIC in any set** (CH +0.3, SS +4.0, story +85.6 with the specified
   bounds; with non-binding rate bounds the story flips to (d) −21.3, D55). Forecasts tie. (d)-uniform beats (d).
   κ, w, α are not identifiable (recovery bimodal).
5. **Decision rule fired → Phase 8 MUST also run on a synthetic Barabási–Albert network.**

**Output:** `results/phase5_params.csv` (all models; Phases 6–8 reuse `model == "sairh"`), `phase5_model_comparison.csv`,
`phase5_forecast.csv`, `phase5_recovery.csv`, `phase5_bootstrap_ci.csv`; Plot 1 (states, d), Plot 2 (cumulative fit
a/c/d/e), Plot 2b (d prevalence vs display), Plot 2c (forecast); `data/processed/model_curves_<set>.csv` (d).
**Check:** sum = 1 each step (±1e-9), states and δ in [0, 1]; dt-convergence (bin/50 vs bin/100) < 1e-3 for every model;
nesting unit test ((d) with κ = 0 = (a) with β/2, α/2, to 1e-9); docs worked-step unit test; ≥ 15/20 converged starts
for every fit and CH (a) best start converged; recovery error reported.

## Phase 6 — Ablation  (`src/sair.py`)  — docs/SAIR_paper.md §6
Uses Phase 5 **model (d)** and its fitted parameters. Re-run with emotion removed (S_emo=1) and popularity removed (Pop=1); optional others.
**Output:** Plot 3. **Check:** no-popularity never peaks; no-emotion has a lower peak.

## Phase 7 — SAIR-C  (`src/control.py`)  — docs/extension.md §1
Built on Phase 5 **model (d)**.
1. Add B, knobs u and v; accumulate cost J.
2. Four strategies: none / block / refute / hybrid.
3. Grid search u ∈ [0,0.5], v ∈ [0,0.1] → cost heatmap.

**Output:** Plot 4 (I(t) per strategy + cost bars), Plot 5 (heatmap, minimum marked).
**Check:** S+A+I+B+R = 1; report which strategy is cheapest.

## Phase 8 — Centrality targeting  (`src/targeting.py`)  — docs/extension.md §2
Built on Phase 5 **model (d)**. Run on PHEME **and** on a synthetic **Barabási–Albert network** (same k values,
methods and actions; BA size and m to be chosen and logged). **The BA run is REQUIRED: (d) did not beat (a)
on AIC in the primary event (Phase 5, D54).**
1. Degree, PageRank, betweenness (k=500 sampling), random (15 runs averaged).
2. For k ∈ {10,25,50,100}: block / recruit / hybrid (k/2 + k/2).
3. Recompute P1 → Drif1 → δ1 → run SAIR-C; record peak and total reach.

**Output:** Plot 6 (peak vs k per method), Plot 7 (bars at fixed k incl. hybrid), results CSV.
**Check:** same k for all methods; random averaged.

## Phase 9 — Packaging
1. Results table (RMSE, strategy costs, peak reduction per centrality).
2. README: problem, pipeline, contributions, corrections (from DECISIONS.md), limitations, future work.
3. `notebooks/experiments.ipynb` runs top to bottom.
