# Session handoff — read this first

Last updated: 2026-10-16 (session 2: Phase 5 revision done). Previous handoff commit: `8128a85`.
Companion files: `CLAUDE.md` (rules + fixed decisions), `PLAN.md` (phases), `DECISIONS.md` (D1–D56, every
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
| 5 SAIR fit | `src/sair.py` | **done (revised: models a–e, AIC/BIC, forecast, recovery, bootstrap)** | `cee4567`, revision commit after `8128a85` |
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

### Phase 5 — revised (D50–D56)
Least squares on C_I/final and C_A/final; same bounds for all models (rates [1e-3, 5], w [0.5, 168], κ [0, 10],
θ [0, 1]); 20 starts; dt = bin/50. Converged starts 20/20 everywhere except story (e) 19/20 (flat w at κ = 0).
| Set | Model | k | Mean RMSE | AIC | ΔAIC vs best | Forecast RMSE (held-out) | Bound flags |
|---|---|---|---|---|---|---|---|
| CH | (a) constant-rate | 5 | 0.0555 | −1563.0 | 9.3 | 0.359 | — |
| CH | (c) Eq. 26 | 3 | 0.1341 | −1064.8 | 507.5 | — | w high |
| CH | **(d) SAIR-H** | 7 | 0.0551 | −1562.7 | 9.6 | 0.359 (fit → κ = 0) | κ high, w high |
| CH | (d)-uniform | 6 | 0.0543 | **−1572.3** | 0 | — | — |
| CH | (e) mean-field | 7 | 0.0555 | −1559.0 | 13.3 | — | κ low (= a) |
| SS | (a) | 5 | 0.0689 | **−460.2** | 0 | 0.495 | — |
| SS | (d) | 7 | 0.0690 | −456.2 | 4.0 | 0.495 | α high |
| SS | (d)-uniform / (e) | 6 / 7 | 0.0690 / 0.0689 | −458.3 / −456.2 | 1.9 / 4.0 | — | α high / κ low |
| story | (a) | 5 | 0.0210 | **−1856.2** | 0 | 0.608 | α high |
| story | (d) | 7 | 0.0259 | −1770.5 | 85.6 | 0.637 | α high, κ low |
(b) Eq. 26 uniform ≈ (c) everywhere. Full table: `results/phase5_model_comparison.csv`.
- **(i)** (d) does not beat (a) on AIC in any set; forecasts tie (CH, SS) or (a) wins (story).
- **(ii)** (d) vs (e): (d) is better only in CH, by 3.7, because (e) collapses to κ = 0. The friend network adds little.
- **(iii)** (d)-uniform beats (d): the regression ranking adds nothing.
- **Story sensitivity (D55):** the specified α ≤ 5 binds (a), and nesting needs (d)'s α = 10. With rate bounds of 50,
  **(d) beats (a) on the story by 21.3 AIC** (κ 3.0). Open question: widen the bounds?
- **Recovery (D56):** 2 of 5 seeds recover the CH (d) truth; 3 of 5 fall into the constant-rate mode (κ ≈ 0) with the same
  fit. κ, w and α are not identifiable; γ, ε and θ are (median error ≤ 8%).
- **Bootstrap (D52, D56):** CH (d) κ [3.9, 10], w [8.9, 168]; SS κ [0, 10]. These CIs are too narrow (i.i.d. residuals, warm start).
- All models miss SS's early A entries (0–12 h, 25% of C_A) because A = 0 is absorbing until the first denying source.

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
    δ1 < S/(1 + S) ≤ 0.5. **So I and A can never both grow at once.** In the fitted full model δ1 ≈ 0.51; endogenous
    growth of I is possible 0.5–1.7% of the time and of A 0%. The model reduces to injection + decay, so RMSE_A is
    large. The constant-rate model has four independent rates and wins.
    (Correction in session 2: the A bound is δ1 < S/(1 + S) ≤ 1/2. The paper's Fig. 9 discussion and our data both show early co-growth.)

(f) **SAIR-H (d) does not beat constant-rate SAIR (a)** on AIC or forecast in any set under the specified bounds (D54).
    The data-driven drives are not identifiable from cumulative curves (D56), and uniform Mult_eff beats the regression
    ranking. On the story, with non-binding bounds, (d) does win (D55). Phase 8 must add a Barabási–Albert network.

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

## 6. NEXT TASK — Phase 6 (ablation), after the user approves Phase 5

Phase 5 is done (D50–D56). Phases 6–8 use **model (d) SAIR-H**: `results/phase5_params.csv`, row `model == "sairh"`,
via `sair.simulate(sc, "sairh", params)`. Phase 6 = emotion off (S_emo = 1) and popularity off (Pop = 1) in Mult_eff
(PLAN.md). Expect small effects: in CH κ sits at its bound with w = 168 h (Pop nearly flat), and κ = 0 on the story,
where the ablation does nothing. Report this, do not tune. Phase 8 must include the BA network (decision rule fired).

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
| 5 | `python src/sair.py --check` (refit all) · `--check --sets story_three_shootings` (refit some sets) · `--check-only` (checks only, ~2 min) | ~2 h on 12 cores (wave 1 fits 41 min, wave 2 recovery + bootstrap 46 min, story refit at dt 0.005 another 21 min) | `results/phase5_{params,model_comparison,forecast,recovery,bootstrap_ci}.csv`, `phase5_plot1/2/2b/2c_*.png`, `data/processed/model_curves_<set>.csv` (model d) |

- `<set>` ∈ {charliehebdo, sydneysiege, story_three_shootings}.
- The pipeline is deterministic (seed 42; two full runs gave byte-identical outputs).
- Raw data must be present in `data/raw/` (see its README for URLs and MD5s).
- Large files are git-ignored: graphs, tweets, users, features, network, groups, drive, friends, model_curves.
- Phase 5 reuse: `drive.make_context(key, variant)` + `drive.drif(ctx, t, side_code)`;
  `sair.get_context(key)`, `sair.simulate(sc, kind, params, record=True)` with kind in `sair.MODELS`; fitting via `sair.start_jobs` + `sair.run_jobs`.

Each module runs its own phase check with `--check`. Every phase check must pass before a commit.

---

## 8. Open decisions

1. **Rate bounds:** the specified [1e-3, 5] binds (a) and (d) on the story (and (d)'s α on SS). Widening to 50 flips the story
   verdict to (d) (D55). Keep the bounds, or widen them and refit all sets?
2. **BA network for Phase 8:** size N and m to be chosen (required by the decision rule).
3. Phase 6 on the story is empty (κ = 0) under the current bounds.
