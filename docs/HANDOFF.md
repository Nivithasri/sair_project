# Session handoff — read this first

Last updated: 2026-10-18 (session 3: Phase 5 refit and Phase 6 done). Previous commits: `8128a85` (handoff), `136a585` (revision).
Companion files: `CLAUDE.md` (rules + fixed decisions), `PLAN.md` (phases), `DECISIONS.md` (D1–D71, every
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
| 5 SAIR fit | `src/sair.py` | **done (refit: 9 models, Bass term, rate/κ bounds 50, cross-model starts, numba, staged)** | `cee4567`, `136a585`, refit commit after it |
| 6 Ablation | `src/ablation.py` | **done** (main model + paper-faithful (d+)) | same commit as the Phase 5 refit |
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

### Phase 5 — refit (D57–D71)
Least squares on C_I/final and C_A/final; rates ∈ [1e-3, 50], κ ∈ [0, 50], w ∈ [0.5, 168], θ, η ∈ [0, 1]; 20 random starts
(25 for the main candidates) + cross-model starts (D64); dt = bin/50 halved until rate·dt ≤ 0.025 (numba kernel, D62).
| Model | k | CH AIC | SS AIC | story AIC | Total AIC |
|---|---|---|---|---|---|
| (a) constant-rate | 5 | −1563.0 | −460.2 | −1859.1 | |
| (a+) + Bass external | 7 | −1570.6 | −658.8 | **−1954.6** | |
| (c) Eq. 26 | 3 | −1064.8 | −384.8 | −1334.9 | |
| (d) SAIR-H | 7 | −1583.1 | −456.6 | −1880.8 | −3920.4 |
| (d+) | 9 | −1587.8 | −680.4 | −1950.8 | −4218.9 |
| (d)-uniform | 6 | **−1661.5** | −458.7 | −1877.0 | −3997.2 |
| **(d+)-uniform = MAIN** | 8 | −1657.7 | **−684.7** | −1952.8 | **−4295.2** |
| (e) mean-field | 7 | −1559.0 | −456.2 | −1876.9 | |
Full table: `results/phase5_model_comparison.csv`; rule table: `phase5_main_model_rule.csv` (D69).
- **Main model (D69): (d+)-uniform** (best total AIC over the three sets; Phases 7–8 need one model for every event).
  CH parameters: β 0.087, α 13.0, γ 0.044, ε 0.055, κ 12.5, θ 0.32, η_I 0.0005, η_A 0. **Do not interpret κ, w, α (D60).**
- (i) Drives: (d+)-uniform beats (a+) by 87.1 AIC on CH and 25.9 on SS; ties on the story (κ = 0). (ii) Network: (d) vs (e)
  −24.1 on CH, ~0 elsewhere. (iii) Regression ranking: adds nothing (uniform beats ranked by 70–78 AIC on CH).
- Bass term: decisive on SS (+198.6 AIC for (a+) over (a)) and the story (+95.6); not needed on CH (η ≈ 0).
- **Forecast** (held-out pooled RMSE; (a) / (a+) / main / (d)-uniform): CH 0.328 / 0.361 / 0.331 / 0.366; SS 0.360 / 0.366 /
  **0.328** / 2.468; story 0.109 / 0.036 / **0.028** / 0.109. No model forecasts later waves before their sources (D71).
- **Recovery (CH):** (d)-uniform: everything except κ within 7% (κ overestimated, 36–50 vs 12.5). Main model: only 1 of 5
  seeds reached the noise floor (and then recovered all parameters within ~10%); 4 stopped in local optima, so the
  8-parameter fit needs many starts (D68).
- **Bootstrap (moving blocks, D59):** main model CH κ [4.8, 50], α [4.4, 21.5], θ [0.26, 0.34]; SS κ [1.7, 4.6]. Warm vs
  cold starts: cold better in only 2 / 60 resamples.
- Ranked models sit at κ = 50 on CH: effectively a step "follow the sided-friends majority" (D70).

### Phase 6 — ablation (D67)
Main model (d+)-uniform: refit ΔAIC (CH / SS / story) network **+95.9** / −1.2 / 0.0; injection +305 / +184 / +71;
Bass −3.8 / **+226** / **+81**; the five per-user factors have no effect by construction (uniform Mult_eff).
Paper-faithful (d+): emotion +6.2 / +0.1 / 0; popularity **−38.6** / −0.8 / −2.0; content fit ~0; connectivity −24.6 / −4.9 / 0;
friend power −10.3 / +0.4 / 0 (negative = fits better without the factor); held-fixed peak changes ≤ 1%.
Fig. 8: strong layer pushes harder on CH only (+0.035); Gact correlation follows the regression sign; Gpre none.
Fig. 6 (CH): own-trait share 0.45 → 0.19 in the only two windows with ≥ 10 A users; external dominates.

---

## 4. Key findings for the write-up

(a) **Text and emotion features are weak.** Only 2–4% of users wrote anything before entering (most enter by
    retweet), so about 97% of user text is profile descriptions. S_emo / Info_match AUC is 0.50–0.54 and Info_match
    is almost always 0 (D31). Phase 6 confirms it: removing emotion changes the rumour peak by ≤ 0.4%.

(b) **P1's separation is thread identity.** Overall P1 AUC is 0.86–0.98, but **within threads** that contain both
    sides it is **0.43–0.49** (at or below chance). Accepted as the intended contagion mechanism (D44), but reported.

(c) **The Phase 4 regression is close to noise** (CV AUC 0.51–0.58), and in Phase 5 the regression *ranking* of Mult_eff
    is beaten by a uniform Mult_eff (D66, D68). The paper's per-user factors are not supported by these data (D67).

(d) **Structural result (D49).** In Eq. 26 with δ2 = 1 − δ1: I grows iff δ1 > 1/(1 + S); A grows iff δ1 < S/(1 + S).
    **So I and A can never both grow at once**, while the paper's Fig. 9 text and our data show early co-growth.

(e) **The follow graph was collected per thread (D38).** PageRank, friends and P1 use it; no alternative graph exists.

(f) **What the data support:** SAIR-H contact/removal rates, source injection, the Bass external-influence term (decisive
    on SS and the story), and network P1 as a majority rule (CH). Not supported: the per-user regression ranking and its
    factors (emotion, popularity, content fit, connectivity, friend power). κ, w, α are not identifiable (D60, D70).

(g) **Paper claims (Phase 6):** Fig. 7 (factor ablation) does not match; Fig. 8 (layers, groups) partly; Fig. 6 (own vs
    external over time) partly at most, with only two usable windows. Details: D67 and the Phase 6 report.

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

## 6. NEXT TASK — Phase 7 (SAIR-C), after the user approves Phases 5–6

Use the main model **(d+)-uniform** (`sair.load_main()` → kind and fitted table; `sair.simulate(sc, kind, params)`).
Add B, the knobs u (blocking) and v (correction campaign) and the cost J (`docs/extension.md` §1, SAIR-H rates). With a
uniform Mult_eff, interventions act on δ only through P1 (the friends' majority, via ρ-feedback, D45), plus the Bass
and injection terms. Decide the open questions in §8 first.

---

## 7. How to run

From the repo root. Python 3.10; numpy, pandas, scipy, scikit-learn, networkx, matplotlib, numba (0.63.1, Euler kernel).
Set `LOKY_MAX_CPU_COUNT=4` to silence a harmless joblib warning on Windows (`[WinError 2]`, core count).

| Order | Command | Runtime | Main outputs |
|---|---|---|---|
| 1 | `python src/data_prep.py --check` | ~90 s | `data/processed/{tweets,users,real_curves,graph}_<set>.*`, `results/phase1_*`, `cumulative_curves_*`, `real_curves_*.png`, `stories_*.csv` |
| 2 | `python src/features.py --check` | ~15 s | `data/processed/{features,pop}_<set>.csv`, `results/phase2_*`, `pop_*`, `pop_vs_entries_charliehebdo.png` |
| 3 | `python src/network.py --check` | ~30 s | `data/processed/{network,groups,groups_follow}_<set>.csv`, `results/phase3_*` |
| 4 | `python src/drive.py --check` | ~30 s | `data/processed/{drive,drive_model,friends}_<set>[_followgroups].*`, `results/phase4_*` |
| 5 | `python src/sair.py --check` (stage 1 then 2) · `--stage 1\|2 --check` · `--fresh` (drop the stages' checkpoints) · `--check-only` | from scratch ~6–8 h on 12 cores; from checkpoints minutes | `results/phase5_*.csv`, `phase5_plot1/2/2b/2c_*.png`, `data/processed/model_curves_<set>.csv` (main model) |
| 6 | `python src/ablation.py --check` (`--check-only`) | ~1.5–2 h from scratch (refits checkpointed) | `results/phase6_*.csv`, `phase6_plot3*_*.png`, `data/processed/ablation_edges_<set>.csv` |

- `<set>` ∈ {charliehebdo, sydneysiege, story_three_shootings}.
- The pipeline is deterministic (seed 42; two full runs gave byte-identical outputs).
- Raw data must be present in `data/raw/` (see its README for URLs and MD5s).
- Large files are git-ignored: graphs, tweets, users, features, network, groups, drive, friends, model_curves,
  ablation_edges, checkpoints.
- Phase 5 reuse: `sair.load_main()`, `sair.get_context(key)`, `sair.simulate(sc, kind, params, record=True)` with kind in
  `sair.MODELS`; fitting via `sair.start_jobs` + `sair.run_jobs(pool, jobs, label, ckpt)` (checkpointed, D62).
- Checkpoints (`data/processed/checkpoints/`): after any change to the setup use `--fresh` or delete the affected files.
  Jobs built from fitted values carry a fingerprint and rerun automatically if that fit changes.

Each module runs its own phase check with `--check`. Every phase check must pass before a commit.

---

## 8. Open decisions

1. **Barabási–Albert network in Phase 8:** the rule fired for (d) in the first revision (D54); the main model now beats
   (a) on CH by 94.7 AIC. Keep the BA run (robustness for targeting) or drop it?
2. **Recovery of the main model:** only 1 of 5 seeds reached the noise floor with 10 random starts. More starts (or
   cross-model seeding) would separate optimiser failure from non-identifiability, at several hours of compute.
3. Phase 7 knob ranges (u ≤ 0.5, v ≤ 0.1 per hour) were set for the old rates; with the fitted SAIR-H rates (e.g. CH
   α = 13 /h) they may need rescaling. Decide before Phase 7.
