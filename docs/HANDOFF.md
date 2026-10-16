# Session handoff — read this first

Last updated: 2026-10-16 (end of session 1). Current commit before this handoff: `cee4567`.
Companion files: `CLAUDE.md` (rules + fixed decisions), `PLAN.md` (phases), `DECISIONS.md` (D1–D49, every
choice and deviation with numbers), `data/raw/README.md` (data provenance and licences).

---

## 1. Project goal and framing

**Whom to Silence, Whom to Amplify.** We
1. reimplement the **SAIR** rumour–correction model (Li et al., IEEE Trans. Big Data 2026; notes in
   `docs/SAIR_paper.md`), with the corrected Eq. 26 (the paper swaps δ1/δ2);
2. extend it to **SAIR-C**: control knobs u (blocking, I → B) and v (correction campaign, S → A) with a
   quadratic cost (form borrowed from DR-DB; `docs/extension.md` §1);
3. test **centrality-targeted intervention** (degree / PageRank / betweenness / random; block, recruit,
   hybrid) on the real graph (`docs/extension.md` §2).

**Data** (`data/raw/`, git-ignored, re-downloadable; see `data/raw/README.md`):
PHEME rumour scheme dataset, journalism use case (figshare 2068650). It is the only source with human stance
labels **and** who-follows-whom **and** retweets. RumourEval-2019 was used only to cross-check labels.
PHEME-RNR was downloaded as a fallback and is unused. The NRC Emotion Lexicon v0.92 is non-commercial research use only.

**Sets used**
| Set | Role | Why |
|---|---|---|
| Charlie Hebdo (`charliehebdo`) | **primary** | Event rule (D21): ≥ 100 A users and ≥ 15 informative bins for C_I and C_A; most A users among usable events (157 A, 51/47 bins) |
| Sydney Siege (`sydneysiege`) | **secondary** | 148 A, 16/21 bins |
| Ottawa story "There were three separate shooting incidents" (`story_three_shootings`) | **robustness** | The only genuine one-rumour-vs-its-correction case (442 supporters vs 268 deniers of one FALSE story), 15-min bins |
| Germanwings | dropped | Only 14 deniers |
| Ottawa (event level) | not usable | Entries too compressed (6/4 informative bins at 1 h, 13/7 at 30 min) |

**Veracity filter (D20):** only **FALSE and UNVERIFIED** stories are kept; TRUE stories are excluded. SAIR is about
malicious vs anti-malicious information, and supporters of true news are not rumour spreaders. The filter drops
CH 39/74 threads (12,348/20,599 users) and SS 51/71 threads (8,140/11,245 users).

**Event-level framing (D25):** "crisis misinformation vs corrections". In both events, I and A come largely
from *different* stories (no kept story has ≥ 20 supporters and ≥ 20 deniers). In SAIR the sides compete through the
shared S pool and the friends' majority P1, not by one-to-one story pairing.

**Fitting target (D14, D22):** cumulative entry curves C_I(t), C_A(t) (users whose first support/deny
post is by t), trimmed at 99% of both + 12 h. They correspond to the model's cumulative inflows ∫δ1·I·S, ∫δ2·A·S.
S/A/I/R prevalence curves are **display only** (exponential lifespan, mean 12 h, D19). N = posting users only,
so compare shape and timing, not absolute peak height (D15).

**States (D3–D6):** reply stance toward the rumour = source stance ∘ reply-vs-source label; retweets inherit
the source stance; the **first** support/deny post fixes I or A (no switching); query/comment-only → S.

---

## 2. Phase status

| Phase | Module | Status | Commits |
|---|---|---|---|
| 0 Data | `data/raw/` | done | `ef4cf16`, `9edae2a` |
| 1 Data prep | `src/data_prep.py` | **done** | `c46e9d9` (+ small additions in `364c879`) |
| 2 Features | `src/features.py` | **done** | `364c879`, `7f808be`, `04e64ab` |
| 3 Network | `src/network.py` | **done** | `6f5ec07`, `b668911`, `8b15077` |
| 4 Drive | `src/drive.py` | **done** | `27e749c`, `04e64ab`, `8b15077` |
| 5 SAIR fit | `src/sair.py` | **first version done — being REVISED (next task)** | `cee4567` |
| 6 Ablation | — | not started | — |
| 7 SAIR-C | — | not started | — |
| 8 Targeting | — | not started | — |
| 9 Packaging | — | not started | — |

---

## 3. Key results per phase

### Phase 1 — data
| Set | N | I users | A users | Neutral (S) | Bin | Window end |
|---|---|---|---|---|---|---|
| Charlie Hebdo | 8,251 | 7,389 | 157 | 705 | 60 min | 135 h |
| Sydney | 3,105 | 2,834 | 148 | 123 | 60 min | 46 h |
| Ottawa story | 721 | 441 | 266 | 14 | 15 min | 31 h |

- Entry routes (I / A): CH I = 26 source, 56 reply, 7,307 retweet; A = 0 / 16 / 141. SS I = 13 / 37 / 2,784; A = 1 / 41 / 106.
- **A is dominated by one story with almost no supporters:** CH 148/157 deniers deny "ID card found in car" (0
  supporters); SS 112/148 deny "airspace closed" (1 supporter) (D24).
- CH arrives in 4 waves (story starts at 0, 11, 24 and 44 h).
- Graph (follow + interaction edges): CH 10,619 follow + 4,756 interaction, SS 3,593 + 1,927; largest weakly connected component 99–100%.

### Phase 2 — user features (only own pre-entry text + profile)
| Set | has_text | wrote a post before entry | has_emotion | has_friends |
|---|---|---|---|---|
| CH | 81.2% | **3.8%** | 51.0% | 83.0% |
| SS | 83.6% | **4.4%** | 54.0% | 69.4% |
| story | 82.4% | **2.1%** | 54.9% | 84.3% |

Class balance (sided users, y = 1 if first side I): CH 7,389 / 157 (97.9% I), SS 2,834 / 148 (95.0%), story 441 / 266 (62.4%).

Single-feature leakage AUC (max of AUC, 1 − AUC), final version:
| Feature | CH | SS | story |
|---|---|---|---|
| S_emo | 0.539 | 0.502 | 0.518 |
| Info_match | 0.507 | 0.521 | 0.502 |
| connectivity | 0.520 | 0.513 | 0.604 |
| friend_power | 0.641 | 0.533 | 0.763 |
| has_text / has_emotion / has_friends | 0.50 / 0.54 / 0.54 | 0.50 / 0.54 / 0.51 | 0.50 / 0.52 / 0.53 |

The story's friend_power of 0.763 (down from 0.836 with snapshot counts) is pre-entry social proof: 338/441 supporters
follow @cnnbrk, 120/266 deniers follow @CTVNews.
Pop(t) (log-follower weights): corr with new I entries 0.49 at lag 0. CH wave weight shares
0.30 / 0.08 / 0.12 / 0.50 vs entry shares 0.22 / 0.06 / 0.21 / 0.50.

### Phase 3 — layers and groups
| Set | Core (1%) | Strong | Weak | Rumour source authors in core | Groups (k = 20) NMI with first thread (profile / follow-Louvain) |
|---|---|---|---|---|---|
| CH | 83 | 56.5% | 42.5% | 19/26 (73%) | **0.038** / 0.707 |
| SS | 32 | 45.0% | 54.0% | 11/14 (79%) | **0.035** / 0.639 |
| story | 8 | 69.1% | 29.8% | 1/1 | **0.013** / 0.393 |

Silhouette is about 0.06–0.09 at every k (soft clusters). Group AUCs with profile groups: max 0.58. With follow-Louvain
they were up to 0.741 (SS Gpre), which was the leak that motivated the switch.

### Phase 4 — blocks, regression, Drif
- Degeneracy fallbacks fired everywhere: **User_att = S_emo × Pop** (paper version near zero for 95–98% of users
  with text + emotion words) and **Group_att = Gact_norm × (Gpre + 0.01)**. lnode_inf is structurally empty
  (≥ 99.8% zero), and hnode_inf is 93% zero.
- Coefficients (standardised; γ0, γ1 User_att, γ2 lnode, γ3 hnode, γ4 Group_att):
  CH 0.050, 0.000, 0.065, 0.433, −0.086 · SS 0.058, −0.006, −0.025, 0.201, 0.318 · story 0.004, 0.071, 0.000, −0.304, 0.089.
- Regression CV AUC (stratified / thread-held-out): **CH 0.512 / 0.314 (gap 0.198)**, SS 0.583 / 0.532,
  **story 0.559 / 0.086 (gap 0.473)**.
- **P1 at entry:** overall AUC CH 0.859, SS 0.864, story 0.978. **Within threads containing both I and A:
  CH 0.428 (12 threads, 2,284 users), SS 0.491 (14, 2,069), story 0.427 (2, 707).**
- Mult_eff = κ × percentile rank of η (mean ≈ 1.0 at κ = 2). Worked example: Mult_eff 2, P1 0.8 → Drif1 0.7685.

### Phase 5 — first version (to be revised)
Loss = mean of RMSE(C_I/C_I,final) and RMSE(C_A/C_A,final). Euler dt = 0.02 h (D47). Source injection θ (D46).
| Set | Model | RMSE_I | RMSE_A | Mean | Fitted parameters |
|---|---|---|---|---|---|
| CH | (a) constant-rate SAIR | 0.054 | 0.058 | **0.056** | α 0.068, β 0.101, ε 0.010, γ 0.047, θ 0.270 |
| CH | (b) uniform Mult_eff | 0.103 | 0.179 | 0.141 | κ 0.1 (bound), θ 0.389 |
| CH | (c) full | 0.094 | 0.174 | 0.134 | w 48 (bound), κ 0.1 (bound), θ 0.382 |
| CH | (c) follow groups | 0.094 | 0.174 | 0.134 | w 6.3, κ 0.1, θ 0.385 |
| SS | (a) | 0.028 | 0.117 | **0.073** | α 1.0 (bound), β 0.245, ε 0.300, γ 0.149, θ 0.5 |
| SS | (b) | 0.056 | 0.267 | 0.161 | κ 0.1, θ 0.5 |
| SS | (c) | 0.064 | 0.255 | 0.160 | w 48, κ 0.1, θ 0.5 |
| SS | (c) follow groups | 0.065 | 0.255 | 0.160 | w 32.6, κ 0.1, θ 0.5 |
| story | (a) | 0.046 | 0.034 | **0.040** | α 1.0, β 0.963, ε 0.267, γ 0.309, θ 0.5 |
| story | (b) | 0.071 | 0.282 | 0.177 | κ 0.996, θ 0.5 |
| story | (c) | 0.065 | 0.288 | 0.176 | w 0.5 (bound), κ 3.90, θ 0.5 |
| story | (c) follow groups | 0.067 | 0.285 | 0.176 | w 0.5, κ 3.25, θ 0.5 |

**The full model does not beat (a) on any set; it beats (b) only by 0.001–0.007.** Checks pass (sum = 1 to 5e-15,
dt-convergence ≤ 5.7e-4 for all models, 10/10 converged starts for (c)). The CH constant-rate best start hit maxfev (3/10 converged).

---

## 4. Key findings for the write-up

(a) **Text and emotion features are weak.** Only 2–4% of users wrote anything before entering (most enter by
    retweet), so about 97% of user text is profile descriptions. S_emo / Info_match AUC is 0.50–0.54 and Info_match
    is almost always 0. Expect the Phase 6 emotion ablation to differ from the paper (much smaller effect) (D31).

(b) **P1's separation is thread identity.** Overall P1 AUC is 0.86–0.98, but **within threads** that contain both
    sides it is **0.43–0.49** (at or below chance). Accepted as the intended contagion mechanism (D44), but reported.

(c) **The Phase 4 regression is close to noise.** CV AUC 0.51–0.58, inconsistent coefficient signs between sets,
    thread-held-out AUC below chance for CH (0.31) and the story (0.09).

(d) **Structural result (D49).** In Eq. 26 with δ2 = 1 − δ1: I grows only if δ1 > 1/(1 + S) ≥ 0.5; A grows only if
    δ1 < 1/(1 + S) ≤ 0.5. **So I and A can never both grow at once.** In the fitted full model δ1 ≈ 0.51; endogenous
    growth of I is possible 0.5–1.7% of the time and of A 0%. The model reduces to injection + decay, so RMSE_A is
    large. The constant-rate model has four independent rates and wins.

(e) **The follow graph was collected per thread (D38).** who-follows-whom only lists follows among each thread's
    participants. Louvain groups on it reconstructed threads (NMI 0.71); PageRank, friends and P1 still use it.
    There is no alternative graph in the data.

---

## 5. Leakage fixes made (keep all of them)

| Fix | Why | Decision |
|---|---|---|
| Exclude TRUE stories | Supporting true news is not rumour spreading | D20 |
| User text = OWN posts strictly before t_ref + profile; **never retweeted text** | Retweeted text *is* the rumour, so S_emo / Info_match would be ≈ 1 for spreaders | D27 |
| Interaction friends only from posts before t_ref | An edge created by the entry retweet encodes the outcome | D27 |
| has_emotion flag; zero emotion vector → missing (mean fill), not cosine 0 | Cosine is undefined | D28 |
| **Friend power from timestamped replies + retweets received before t_i, excluding user i's own; no favourites** | Snapshot retweet/favourite counts include the user's own later retweet | D29, D30 |
| Pop weights = log(1 + source-author followers), known at t0_s; never the story's eventual size | Eventual size leaks the outcome; raw followers let one account take 38% | D26 |
| **PageRank on follow edges only** (interaction weight 0) | Retweet edges point at source authors because of the spreading | D32 |
| **Groups = KMeans on profile-only features** (not KNN on Prb, not follow-Louvain); NMI(group, thread) < 0.3 asserted | Prb = shared retweets = outcome; follow-Louvain reconstructed threads | D34, D37 |
| Gact = account history (log1p statuses), not event originals/reposts | Event posts are the outcome | D35 |
| Gpre from members' own pre-entry text + profiles | Same as D27 | D36 |
| User_att degeneracy judged only on users with real text and emotion | Imputed values hid the degeneracy | D40 |
| Regression on sided users only; thread-held-out CV reported next to stratified | Measures thread learning | D39, D41 |
| Single-feature AUC ≤ 0.95 asserted in Phases 2–4 | Leakage tripwire | — |
| Deterministic friend iteration (sorted) | Hash-seed-dependent float sums | commit `04e64ab` |

---

## 6. NEXT TASK — Phase 5 revision

Revise `src/sair.py` (Phase 5) before Phase 6. Models, all with the same loss, cumulative target, injection (D46),
Euler dt = 0.02 h and multi-start Nelder-Mead:
- (a) constant-rate SAIR (Eq. 22) — exists
- (b) uniform Mult_eff — exists
- (c) full corrected SAIR with data-driven δ (D45) — exists
- **(d) SAIR-H** — new
- **(e) mean-field** — new

> **The exact equations of (d) SAIR-H and (e) mean-field were NOT specified in session 1.** Take them from the
> user's Phase 5 revision prompt. Do not invent them. If they are not given, ask before coding.

Required evaluation (user spec). The protocol details marked *(suggested)* are not yet agreed; confirm or replace them with the user's spec:
1. **AIC / BIC** per set and model. *(suggested)* Gaussian form on the fitted loss residuals:
   AIC = n·ln(SSE/n) + 2k, BIC = n·ln(SSE/n) + k·ln n, with n = number of fitted points (C_I and C_A bins) and k = free parameters incl. θ.
2. **Parameter recovery test.** *(suggested)* Simulate synthetic curves from the fitted parameters (± noise), refit
   from random starts, and report the recovered vs true parameters (bias and relative error); flag non-identifiable parameters.
3. **Bootstrap confidence intervals.** *(suggested)* Residual or block bootstrap of the curves (e.g. 200 replicates,
   seed 42), refit, 95% percentile CIs per parameter and for RMSE / AIC.
4. **Forecast test.** *(suggested)* Fit on an early window (e.g. up to the 50% point of C_I, or the first wave)
   and score out-of-sample RMSE on the rest, per model.

**Decision rule (fixed): if (d) does not beat (a) on AIC, Phase 8 must also run on a synthetic
Barabási–Albert network** (in addition to PHEME). Phases 6–8 use **model (d)** with its fitted parameters.

Honesty rules apply (CLAUDE.md "Working principles"): do not tune the setup to make (d) win; report the AIC table as it comes out.

---

## 7. How to run

From the repo root. Python 3.10; numpy, pandas, scipy, scikit-learn, networkx, matplotlib.
Set `LOKY_MAX_CPU_COUNT=4` to silence a harmless joblib warning on Windows (`[WinError 2]`, core count).

| Order | Command | Runtime | Main outputs |
|---|---|---|---|
| 1 | `python src/data_prep.py --check` | ~90 s | `data/processed/{tweets,users,real_curves,graph}_<set>.*`, `results/phase1_*`, `cumulative_curves_*`, `real_curves_*.png`, `stories_*.csv` |
| 2 | `python src/features.py --check` | ~15 s | `data/processed/{features,pop}_<set>.csv`, `results/phase2_*`, `pop_*`, `pop_vs_entries_charliehebdo.png` |
| 3 | `python src/network.py --check` | ~30 s | `data/processed/{network,groups,groups_follow}_<set>.csv`, `results/phase3_*` |
| 4 | `python src/drive.py --check` | ~30 s | `data/processed/{drive,drive_model,friends}_<set>[_followgroups].*`, `results/phase4_*` |
| 5 | `python src/sair.py --check` (refit) or `--check-only` (checks on saved params, ~7 s) | ~3 min (10 parallel starts) | `results/phase5_params.csv`, `phase5_plot1/2/2b_*.png`, `data/processed/model_curves_<set>.csv` |

- `<set>` ∈ {charliehebdo, sydneysiege, story_three_shootings}.
- The pipeline is deterministic (seed 42; two full runs gave byte-identical outputs).
- Raw data must be present in `data/raw/` (see its README for URLs and MD5s).
- Large files are git-ignored: graphs, tweets, users, features, network, groups, drive, friends, model_curves.
- Phase 5 reuse: `drive.make_context(key, variant)` + `drive.drif(ctx, t, side_code)`;
  `sair.set_context(key)`, `sair.simulate(sc, kind, params, dt, record)`, `sair.fit(sc, kind, pool)`.

Each module runs its own phase check with `--check`. Every phase check must pass before a commit.

---

## 8. Open decisions

None, other than what the Phase 5 revision decides: the definitions of (d) and (e) (from the user), the
AIC verdict of (d) vs (a), and therefore whether Phase 8 adds the BA network.
