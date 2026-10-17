> **New session? Read `docs/HANDOFF.md` first.**

# Project: Whom to Silence, Whom to Amplify
Centrality-Targeted Hybrid Control of Rumour–Correction Competition on Social Networks

## What this project is
- **Base paper:** SAIR model, Li et al., IEEE Trans. Big Data 2026 — `papers/SAIR.pdf`, decoded in `docs/SAIR_paper.md`.
- **Our extension:** SAIR-C (blocking + correction-campaign knobs with a cost score) and centrality-targeted intervention — `docs/extension.md`.
- **Borrowed idea only (do NOT implement this paper):** DR-DB rumor control, Zhong et al., JSSC 2026 — `docs/DRDB_reference.md`.
- **Phase plan:** `PLAN.md`. **Assumptions log:** `DECISIONS.md`.

## Working principles
- **Strict honesty.** Report what the data and models actually show, including negative results and failed checks.
- **Never relax the leakage rules to match the paper** (own pre-entry text only, timestamped friend power,
  profile-only groups, follow-only PageRank, single-feature AUC ≤ 0.95; see HANDOFF §5).
- **Never tune the setup to force a model to win.** Same loss, data, injection and optimiser for every compared model.
- **Report negative results** prominently (e.g. D49: Eq. 26 loses to constant-rate SAIR; D67: the paper's per-user
  factors are not supported, Fig. 7 does not match).
- **A bound must never decide a verdict**; flag every parameter at a bound. **A model must never fit worse than its own
  submodel**: use the cross-model starts and the SSE-order check (D64).
- **Long runs are checkpointed and staged** (`--stage`, D62): log progress with an ETA, report after each stage, and
  resume from checkpoints after a crash.
- **Every deviation from the paper or the plan is logged in `DECISIONS.md`** with the reason and the numbers.
- **One phase at a time, then stop** for approval.
- **Detailed final report per phase** (steps, files, check output, tables, observations, problems and fixes,
  decisions, open questions).

## How to work
- Work **ONE phase at a time** from `PLAN.md`. Never start the next phase without my approval.
- For each phase: read the relevant `docs/` section → propose a short plan → code → **run the phase check** → show output/plots → give a **full, detailed final report**: everything done step by step, files created/changed/deleted, check output, numbers and tables, observations, problems and fixes, decisions logged, open questions.
- Keep only needed files in the repo: before committing, delete stale outputs (old plots/tables, dropped events), placeholder `.gitkeep` files in folders that now have content, and scratch scripts (scratch work goes outside the repo).
- Where the paper is vague, choose a sensible option, **log it in `DECISIONS.md`**, and tell me.
- If something in these rules conflicts with the real data, stop and ask.

## Fixed decisions (do not change without asking)
- Data: PHEME rumour scheme dataset (figshare 2068650), English threads, **only FALSE and UNVERIFIED stories** (TRUE stories excluded, DECISIONS D20). **Primary event: Charlie Hebdo; secondary: Sydney Siege** (D21). Robustness set: Ottawa story "There were three separate shooting incidents", 15-min bins (D23).
- State labels from stance: supports rumour → **I**, denies rumour → **A**, no post yet → **S**. Display prevalence uses an exponential lifespan (mean 12 h) for I/A → R (D19).
- **Fitting target = cumulative entry curves C_I(t), C_A(t)**, trimmed at 99% + 12 h (D14, D22). Compare shape and timing, not absolute peak height (N = posting users only). A user's I/A side is fixed by their **first** support/deny post (no I↔A transition). Query/comment-only users stay S.
- Graph: directed, **follow edges (who-follows-whom) + reply/retweet interaction edges**, each edge has attribute `kind` ∈ {follow, interaction}. Edge u→v = u receives information from v.
- Emotion: NRC emotion lexicon, **8 dimensions**; recency weight `exp(-lam*(t_ref - t_j))`, lam = ln2/12 per h.
- **No leakage:** user features use only the user's OWN posts written before t_ref (entry time, or event end for S users) + profile description; never retweeted text (DECISIONS D27). Check single-feature AUC ≤ 0.95.
- Popularity: Pop(t) = Σ_s w_s·0.5^((t−t0_s)/w), one term per kept story, w_s = log(1 + source-author followers), normalised max 1; half-life w free in Phase 5 (D26).
- Network (Phase 3): PageRank on FOLLOW edges only (u→v = u follows v); core = top 1%; strong = follows a core node; weak = rest. mpr = τ(k1·pr_norm + k2·Info_match), τ = 1, k1 = k2 = 0.5; r = 1 − mpr_norm clipped [0.05, 0.95].
- Groups: KMeans (seed 42) on PROFILE-ONLY features (description TF-IDF/SVD + log followers, friends, statuses, account age), k = 20 for every set (silhouette ~0.07 at all k); NMI(group, first thread) must be < 0.3 (D37). Follow-Louvain kept as a robustness variant. NOT KNN on Prb (event retweets = outcome). The follow graph is collected per thread — limitation (D38). Gact = mean log1p(statuses_count); Gpre = Jaccard(group top-30 TF-IDF keywords of own pre-entry text + profiles, rumour tags); Group_att = Gact_norm × Gpre.
- Friend power: log1p of the mean over friends of timestamped replies + retweets RECEIVED before the user's t_ref, excluding the user's own; no favourites (D29).
- S_emo: users with no emotion words → has_emotion = 0, filled with the event mean (cosine undefined).
- Content fit, group preference: Jaccard on tag/keyword sets.
- Connectivity: tag similarity + **Jaccard of friend lists** (normalised, not raw mutual-friend count). Friend lists = follows **inside the dataset** (full Twitter friend lists are not available).
- Group preference (Gpre): keywords from group members' OWN pre-entry text + profile descriptions (D36; no pre-event posts exist).
- Drop the undefined "Initiative" variable from Algorithm 1.
- Rename the paper's group-activity weight γ → `w_part` (γ clashes with a removal rate).
- Regression target for Mult_eff = **first side is I (1) vs A (0)**, on users who took a side (D39).
- `Drif1 = sigmoid((2*P1 - 1) * Mult_eff)`, `Drif2 = 1 - Drif1`.
- Phase 4: regression sample = sided users (I ∪ A), logistic, class_weight balanced, standardised blocks; Mult_eff = κ·percentile_rank(η) ∈ [0, κ], κ = 2 (fitted in Phase 5); P1 = friends who have taken side I / those who took any side, fallback = global share; 90% degeneracy rule for User_att (judged on has_text & has_emotion users) / Group_att (D39–D44); lnode_inf kept but structurally empty.
- `delta1(t) = mean Drif1 over users not yet entered in the data`, `delta2 = 1 - delta1` (Eq. 24's binomial reduces to this). P1 uses data friend counts re-weighted by ρ = C_model/C_data (clip [0, 5]) so interventions feed back; recomputed every data bin (1 h) (D45).
- Source injection: θ·w_s/Σw moves S → I/A at each supporting/denying source tweet (D46). Phase 5 (refit, D57–D71): least squares, bounded Nelder-Mead (log space for rates), 20 random starts (seed 42) + cross-model starts, rates ∈ [1e-3, 50], κ ∈ [0, 50], θ, η ∈ [0, 1]; Euler dt = bin/50 halved until rate·dt ≤ 0.025, steps in a numba kernel; checkpointed staged run; parameters in `results/phase5_params.csv`.
- **SAIR-H** (D50): dS = −β·δ1·I·S − α·δ2·A·S, dI = β·δ1·I·S − γ·I, dA = α·δ2·A·S − ε·A, dR = γ·I + ε·A; "+" variants add the Bass external influence η_I·S, η_A·S (D57). **Main model for Phases 6–8 = (d+)-uniform** (SAIR-H + Bass, network P1, uniform Mult_eff = κ), chosen by best total AIC over the three sets (D69). The corrected Eq. 26 below is kept as model (c); it cannot let I and A grow at the same time (D49). κ, w, α are not identifiable from cumulative curves: do not interpret their values (D60).
- **Corrected SAIR (paper's Eq. 26 swaps δ1/δ2):**
  - dS = −δ1·I·S − δ2·A·S
  - dI = δ1·I·S − (1−δ1)·I
  - dA = δ2·A·S − (1−δ2)·A
  - dR = (1−δ1)·I + (1−δ2)·A
- Phases 6–8 use the Phase 5 **main model (d+)-uniform** (`sair.load_main()`). Phase 7 = SAIR-C with event-relative knobs and a Pareto front (spec pasted by the user); Phase 8 also runs on a synthetic Barabási–Albert network (D72).
- SAIR-C adds B (blocked), knobs u (blocking) and v (correction campaign), scaled per event (event-relative, D72); DR-DB cost weights A1=1, c1=c2=2 for the cost J.

## Coding rules
- Python 3, networkx, numpy, pandas, scipy, scikit-learn, matplotlib, numba (Euler kernel, D62). Add only what is needed.
- One file per concern in `src/` (see `PLAN.md`). Plain functions, simple readable code, no class hierarchies or over-engineering.
- Every phase has a check (asserts or a small script). **Run it before saying a phase is done.**
- Invariants: population fractions sum to 1 at every step (±1e-9); all probabilities in [0, 1].
- Save plots to `results/` as PNG with titles and axis labels. Save tables as CSV.
- Never modify `data/raw/`. Write intermediate outputs to `data/processed/`.
- Fix random seeds (seed = 42) for reproducibility.

## Git rules
- Commit messages, PR descriptions and pushed content must contain **no trace of Claude**: no `Co-Authored-By: Claude`, no "Generated with Claude Code", no Claude/Anthropic mentions or tags. Author is the user only.
