# Project: Whom to Silence, Whom to Amplify
Centrality-Targeted Hybrid Control of Rumour–Correction Competition on Social Networks

## What this project is
- **Base paper:** SAIR model, Li et al., IEEE Trans. Big Data 2026 — `papers/SAIR.pdf`, decoded in `docs/SAIR_paper.md`.
- **Our extension:** SAIR-C (blocking + correction-campaign knobs with a cost score) and centrality-targeted intervention — `docs/extension.md`.
- **Borrowed idea only (do NOT implement this paper):** DR-DB rumor control, Zhong et al., JSSC 2026 — `docs/DRDB_reference.md`.
- **Phase plan:** `PLAN.md`. **Assumptions log:** `DECISIONS.md`.

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
- Groups: KMeans (seed 42) on PROFILE-ONLY features (description TF-IDF/SVD + log followers, friends, statuses, account age), k ∈ {20,50,100} by silhouette; NMI(group, first thread) must be < 0.3 (D37). Follow-Louvain kept as a robustness variant. NOT KNN on Prb (event retweets = outcome). The follow graph is collected per thread — limitation (D38). Gact = mean log1p(statuses_count); Gpre = Jaccard(group top-30 TF-IDF keywords of own pre-entry text + profiles, rumour tags); Group_att = Gact_norm × Gpre.
- Friend power: log1p of the mean over friends of timestamped replies + retweets RECEIVED before the user's t_ref, excluding the user's own; no favourites (D29).
- S_emo: users with no emotion words → has_emotion = 0, filled with the event mean (cosine undefined).
- Content fit, group preference: Jaccard on tag/keyword sets.
- Connectivity: tag similarity + **Jaccard of friend lists** (normalised, not raw mutual-friend count). Friend lists = follows **inside the dataset** (full Twitter friend lists are not available).
- Group preference (Gpre): keywords from the group's posts **inside the event window** (no pre-event posts exist).
- Drop the undefined "Initiative" variable from Algorithm 1.
- Rename the paper's group-activity weight γ → `w_part` (γ clashes with a removal rate).
- Regression target for Mult_eff = **user reposted the rumour (1/0)**.
- `Drif1 = sigmoid((2*P1 - 1) * Mult_eff)`, `Drif2 = 1 - Drif1`.
- `delta1(t) = mean Drif1 over susceptible users`, `delta2(t) = mean Drif2` (Eq. 24's binomial reduces to this). Recompute every step.
- **Corrected SAIR (paper's Eq. 26 swaps δ1/δ2):**
  - dS = −δ1·I·S − δ2·A·S
  - dI = δ1·I·S − (1−δ1)·I
  - dA = δ2·A·S − (1−δ2)·A
  - dR = (1−δ1)·I + (1−δ2)·A
- SAIR-C adds B (blocked), knobs u (blocking) and v (correction campaign); cost weights A1=1, c1=c2=2.

## Coding rules
- Python 3, networkx, numpy, pandas, scipy, scikit-learn, matplotlib. Add only what is needed.
- One file per concern in `src/` (see `PLAN.md`). Plain functions, simple readable code, no class hierarchies or over-engineering.
- Every phase has a check (asserts or a small script). **Run it before saying a phase is done.**
- Invariants: population fractions sum to 1 at every step (±1e-9); all probabilities in [0, 1].
- Save plots to `results/` as PNG with titles and axis labels. Save tables as CSV.
- Never modify `data/raw/`. Write intermediate outputs to `data/processed/`.
- Fix random seeds (seed = 42) for reproducibility.

## Git rules
- Commit messages, PR descriptions and pushed content must contain **no trace of Claude**: no `Co-Authored-By: Claude`, no "Generated with Claude Code", no Claude/Anthropic mentions or tags. Author is the user only.
