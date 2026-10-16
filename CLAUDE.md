# Project: Whom to Silence, Whom to Amplify
Centrality-Targeted Hybrid Control of Rumour–Correction Competition on Social Networks

## What this project is
- **Base paper:** SAIR model, Li et al., IEEE Trans. Big Data 2026 — `papers/SAIR.pdf`, decoded in `docs/SAIR_paper.md`.
- **Our extension:** SAIR-C (blocking + correction-campaign knobs with a cost score) and centrality-targeted intervention — `docs/extension.md`.
- **Borrowed idea only (do NOT implement this paper):** DR-DB rumor control, Zhong et al., JSSC 2026 — `docs/DRDB_reference.md`.
- **Phase plan:** `PLAN.md`. **Assumptions log:** `DECISIONS.md`.

## How to work
- Work **ONE phase at a time** from `PLAN.md`. Never start the next phase without my approval.
- For each phase: read the relevant `docs/` section → propose a short plan → code → **run the phase check** → show output/plots → summarise in ≤5 lines.
- Where the paper is vague, choose a sensible option, **log it in `DECISIONS.md`**, and tell me.
- If something in these rules conflicts with the real data, stop and ask.

## Fixed decisions (do not change without asking)
- Data: PHEME rumour scheme dataset (figshare 2068650), English threads. **Primary event: Sydney Siege; second event: Ottawa shooting** (replaced Germanwings, which has only 14 deniers — see DECISIONS D11).
- State labels from stance: supports rumour → **I**, denies rumour → **A**, no post yet → **S**, inactive 24 h → **R**. A user's I/A side is fixed by their **first** support/deny post (no I↔A transition). Query/comment-only users stay S.
- Graph: directed, **follow edges (who-follows-whom) + reply/retweet interaction edges**, each edge has attribute `kind` ∈ {follow, interaction}. Edge u→v = u receives information from v.
- Emotion: NRC emotion lexicon, **8 dimensions**; recency weight `exp(-lam*(t_now - t_j))`.
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
