# Implementation Phase Plan

Each phase: plan → code → run check → show results → stop for approval.
**Status:** Phases 1–6 done (last commit `dc97afe`; Phase 5 refit D57–D71, Phase 6 D67). Phases 7–9 not started.
Main model for Phases 6–8: **(d+)-uniform** (D69). Plan for 7–9: D72.
See `docs/HANDOFF.md` for results and findings.
Equations and definitions live in `docs/SAIR_paper.md` and `docs/extension.md`.

## File structure
```
src/
  data_prep.py   Phase 1
  features.py    Phase 2
  network.py     Phase 3
  drive.py       Phase 4
  sair.py        Phase 5
  ablation.py    Phase 6
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

## Phase 5 — SAIR simulation & validation  (`src/sair.py`)  — docs/SAIR_paper.md §5  ✅ DONE (refit)
First version `cee4567` (D45–D49); revision `136a585` (D50–D56); refit D57–D71.
Setup: fractions of N; Euler dt = bin/50 halved until rate·dt ≤ 0.025, steps in a numba kernel identical to the Python
loop (D58, D62); source injection (D46); least squares on C/C_final; bounded Nelder-Mead (log space for rates and w);
rates ∈ [1e-3, 50], κ ∈ [0, 50], w ∈ [0.5, 168], θ, η ∈ [0, 1]; 20 random starts (seed 42) + cross-model starts (D64);
checkpointed, staged run (`--stage 1|2`, D62).
1. Models: (a) constant-rate, (a+) + Bass external influence (D57), (b) Eq. 26 uniform, (c) Eq. 26, (d) SAIR-H, (d+),
   (d)-uniform, (d+)-uniform, (e) mean-field.
2. Stage 1: AIC / BIC per set and model, bound flags, converged counts. **Main model = (d+)-uniform** (best total AIC over
   the three sets among the SAIR-H family, D69).
3. Stage 2 (validation of (a), (a+), main, and (d)-uniform as a secondary row): forecast after max(50% of C_I, 10th bin)
   with data inputs frozen at the cut (D59); recovery on CH (D60); moving-block bootstrap (blocks of 5 bins) with a
   cold-start check (D59).
4. Findings: Bass term decisive on SS / story; regression ranking adds nothing; network P1 adds on CH; κ, w (and α in
   ranked models) not identifiable — do not interpret (D60, D70); no model forecasts later waves (D71).

**Output:** `results/phase5_params.csv`, `phase5_model_comparison.csv`, `phase5_main_model_rule.csv`, `phase5_forecast.csv`,
`phase5_recovery.csv`, `phase5_bootstrap_ci.csv`, `phase5_bootstrap_coldcheck.csv`; plots 1, 2, 2b, 2c;
`data/processed/model_curves_<set>.csv` (main model).
**Check:** sum = 1 (±1e-9), states and δ in [0, 1]; dt vs dt/2 < 1e-3 for every model; nesting at κ = 0 exact;
numba == Python; no model worse than its submodels; docs worked step; ≥ 15 converged starts; recovery reported.

## Phase 6 — Ablation  (`src/ablation.py`)  — docs/SAIR_paper.md §6  ✅ DONE
Main model (d+)-uniform and, as the paper-faithful ablation, (d+) (ranked Mult_eff) (D67); (d)-uniform as a secondary row.
1. Knock-outs with fitted parameters held fixed: emotion, popularity, content fit, connectivity, friend power (rebuilt
   from per-edge components), network (global P1), source injection, Bass external influence.
2. Refit importance: each knock-out refit (warm + 9 random starts), ΔAIC vs the full model.
3. Paper Fig. 8 test (strong vs weak layer Drif1; Drif1 vs Gact, Gpre) and Fig. 6 test (regression per 10-h window, CH).

**Output:** `results/phase6_ablation.csv`, `phase6_importance.csv`, `phase6_layers.csv`, `phase6_fig8_test.csv`,
`phase6_fig6_test.csv`; Plot 3 / 3b (main model), Plot 3 / 3b `_paper` ((d+)), Plot 3c (layers).
**Check:** sums = 1 for every variant; the full model reproduces the Phase 5 fit exactly; each knock-out changes only its
intended input; rank knock-outs have no effect on uniform-Mult_eff models.

## Phase 7 — SAIR-C  (`src/control.py`)  — docs/extension.md §1
Built on the Phase 5 **main model (d+)-uniform** (`sair.load_main()`), per set with its fitted parameters.
**The exact spec is pasted by the user at the start of the session (D72).** Fixed so far:
1. Add the blocked state B, knob u (blocking, I → B) and knob v (correction campaign, S → A); S + I + A + B + R = 1.
2. **Event-relative knobs:** u and v scaled to each event's fitted rates / timescale (not one absolute grid).
3. **Pareto front** of damage (e.g. rumour reach / peak) vs control effort, per event; fixed strategies none / block /
   refute / hybrid placed on it. The DR-DB cost J (A1 = 1, c1 = c2 = 2) may be reported as one point of view.
4. Feedback through P1 (ρ-feedback, D45): with a uniform Mult_eff this is the only route by which interventions change δ.

**Output / check:** per the spec; at least S + I + A + B + R = 1, numba kernel == Python reference, plots + CSVs.

## Phase 8 — Centrality targeting  (`src/targeting.py`)  — docs/extension.md §2
Built on the Phase 5 **main model (d+)-uniform**. Runs on PHEME **and on a synthetic Barabási–Albert network** (D72;
same k values, methods and actions; size and m to be chosen and logged).
1. Degree, PageRank, betweenness (k=500 sampling), random (15 runs averaged).
2. For k ∈ {10,25,50,100}: block / recruit / hybrid (k/2 + k/2).
3. Recompute P1 → Drif1 → δ1 → run SAIR-C; record peak and total reach.
Note: with uniform Mult_eff, targeting acts on δ only through P1 (the friends' majority), which blocking and recruiting change.

**Output:** Plot 6 (peak vs k per method), Plot 7 (bars at fixed k incl. hybrid), results CSV.
**Check:** same k for all methods; random averaged.

## Phase 9 — Packaging
0. Optional (D72): extra recovery starts for the main model (more random starts or cross-model seeding per noise seed),
   to separate optimiser failure from non-identifiability.
1. Results table (RMSE, strategy costs, peak reduction per centrality).
2. README: problem, pipeline, contributions, corrections (from DECISIONS.md), limitations, future work.
3. `notebooks/experiments.ipynb` runs top to bottom.
