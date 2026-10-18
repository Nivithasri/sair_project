# Session handoff — read this first

Last updated: 2026-10-18 (session 4: all phases done, tagged v1.0). Phase 7 commit: see `git log` (message "Phase 7: SAIR-C control").
Companion files: `CLAUDE.md` (rules + fixed decisions), `PLAN.md` (phases), `DECISIONS.md` (D1–D83, with a final index, every
choice and deviation with numbers), `data/raw/README.md` (data provenance and licences).

---

## 0. Summary (start here)

- **Status: project complete (v1.0).** Phases 1–9 done (Phase 7 = SAIR-C D73–D76, Phase 8 = targeting D77–D82, Phase 9 = packaging D83).
  Deliverables: `docs/REPORT.md` + `REPORT.pdf`, `presentation/slides.pptx`, `docs/VIVA_QA.md`, `README.md`, `notebooks/experiments.ipynb`.
- **Main model for Phases 7–8: (d+)-uniform** = SAIR-H + Bass external influence, network P1, uniform Mult_eff = κ.
  It was chosen as the SAIR-H-family model with the **best total AIC over the three sets (D69)**: total AIC −4295.2,
  BIC −4217.8. Load it with `kind, table = sair.load_main()`.
- **What the data support:** SAIR-H contact and removal rates, source injection, the Bass term (decisive on Sydney and
  the story) and network P1 as a "follow the sided-friends majority" rule (CH). **Not supported:** the paper's Eq. 26 (I
  and A can never grow together, D49), the per-user regression ranking, and its factors (emotion, popularity, content
  fit, connectivity, friend power).
- **Paper claims:** Fig. 7 does **not** match; Fig. 8 **partly**; Fig. 6 **partly at most** (§3, Phase 6).
- **Caveats:** κ, w and α are **not identifiable**, so do not interpret their values (D60, D70). Recovery of the main model
  is limited by the optimiser (only 1 of 5 seeds reached the noise floor). No model forecasts a later wave before its
  source tweets appear (D71).
- **Phase 7 result (main model, equal total effort E = 2 / 5 / 10% of N):** the three strategies tie at 2–5% (within
  ~2 pp of D/D0); at 10% hybrid wins clearly only on CH (D/D0 .016 vs refute .37, block .87), refute is best on SS (.263 vs
  hybrid .290) and block on the story (.526 vs hybrid .541). **"Hybrid is best" is therefore NOT a general result**; it holds
  on CH at 10% and in the weighted cost J (which favours hybrids by construction). Feedback: ρ frozen at 1 changes ≈ nothing;
  the feedback that matters runs through the global-share fallback (d1 frozen: up to 47–70% of the effect at 10% effort on
  SS / CH refute; 0 on the story where κ = 0). Details §3b, D75.
- **Phase 7 follow-ups (D76):** Sydney now runs at dt/2 (the dt warning is gone; adding u, v to `step_dt` is a no-op). **Tipping robustness (CH):**
  the 10%-effort hybrid tipping survives κ ∈ {12.5, 25, 50} (D/D0 .016–.037; κ = 5 is disfavoured by +28.7 AIC and tips even earlier) but **not**
  the fallback P1 = 0.5 (the refit goes to κ ≈ 0.09, AIC +100.9, no tipping): it depends on the global-share fallback the CH fit needs.
- **Phase 8 result (after the D81 / D82 fixes):** PHEME: **block** works through the removal of source authors' injection (88–99.8% of its effect; friend-count
  channel ≤ 0.9%), so degree / PageRank win it (CH k = 50: .53 / .16 / .03 / .00 for degree / PageRank / betweenness / random). **The earlier "recruit is saturated" was an
  artefact of the injection scale** (s = 1 moves 32% of N for 50 recruits): with s = 0.1 / 0.01 and effort counted as % of N moved, **recruit beats block at equal effort
  on CH and Sydney, hybrid is best from ~0.5-1% of N moved (recruit alone below that), and block is ahead of recruit on the story at E <= 2%** and **centrality matters** (CH s = 0.01, k = 50: recruit .815 degree vs .028 random; an interaction of a small A mass with the followers' P1, not an injection
  effect). Unchanged at both α CI ends (refits +39.5 / +11.5 AIC). **BA with source injection on hubs is calibrated (β×2, α×0.1) and passes the validation criteria** (final C_I .879 vs
  .895, C_A .025 vs .019, I peaks before A; shape RMSE .134, moderate): hubs beat random by a wide margin (block 90% vs ≤ 3.5%, hybrid 99% vs 63–79%), the three centralities are
  indistinguishable, conformity contributes ≤ 3.5% of block and 10–23% of recruit. The seeds-only BA variant stays labelled "qualitative only". Details §3b, D77–D82.
- **Extra recovery (D83):** from the true parameters every seed reaches the noise floor and β, α, γ, ε, θ are recovered within 8.3%, κ is not
  (moves to 50 at the same fit); random starts reach the floor in 2 of 150 runs. So earlier recovery failures were the optimiser, and κ is the
  non-identifiable parameter.
- **Next task:** none planned. Possible extensions are listed in REPORT Section 8.
- **Rules that matter most:** strict honesty, report negative results, never tune to make a model win, never relax the
  leakage rules (§5), log every deviation in `DECISIONS.md`, one phase at a time and then stop, no Claude attribution in git.

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
| 5 SAIR fit | `src/sair.py` | **done** (refit: 9 models, Bass term, rate/κ bounds 50, cross-model starts, numba, staged) | `cee4567`, `136a585`, `dc97afe` |
| 6 Ablation | `src/ablation.py` | **done** (main model + paper-faithful (d+)) | `dc97afe` |
| 7 SAIR-C | `src/control.py` (+ `_advance` / `simulate` extended in `sair.py`) | **done** (D73–D75) | see `git log` |
| 8 Targeting | `src/targeting.py` | **done** (D77–D80) | see `git log` |
| 9 Packaging | `src/report_check.py`, `export_report.py`, `slides.py` | **done** (D83) | tag `v1.0` |

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

### Main model: (d+)-uniform (D57, D66, D69)
```
dS = −β·δ1·I·S − α·δ2·A·S − (η_I + η_A)·S
dI =  β·δ1·I·S + η_I·S − γ·I
dA =  α·δ2·A·S + η_A·S − ε·A
dR =  γ·I + ε·A
δ1 = mean over not-yet-entered users of sigmoid(κ·(2·P1_i − 1)),  δ2 = 1 − δ1
P1_i = friends' I share (data friend counts re-weighted by ρ = C_model / C_data, D45); source injection θ (D46)
```
Uniform Mult_eff: every user's push uses the same κ, so the per-user factors do not enter δ. Fitted parameters (rates /h):

| Set | β | α | γ | ε | κ | θ | η_I | η_A | AIC | at a bound |
|---|---|---|---|---|---|---|---|---|---|---|
| Charlie Hebdo | 0.0874 | 13.01 | 0.0437 | 0.0554 | 12.48 | 0.318 | 0.00055 | 0 | −1657.7 | η_I, η_A low |
| Sydney Siege | 0.490 | 21.06 | 0.677 | 1.645 | 2.88 | 0.319 | 0.0449 | 0.0015 | −684.7 | η_A low |
| Ottawa story | 40.0 | 32.54 | 14.92 | 4.997 | ≈ 0 | 0.067 | 0.1034 | 0.0290 | −1952.8 | κ low |

On the story κ = 0, so the model reduces to (a+) with β/2 and α/2. Main-rule table: `results/phase5_main_model_rule.csv`.

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
Refit ΔAIC = AIC(knock-out refit) − AIC(full); positive = the factor is needed. Held fixed = the fitted parameters are
kept and only the input is knocked out. A negative ΔAIC is possible because the input knock-outs are not nested.

**Main model (d+)-uniform: which components matter**
| Knock-out | ΔAIC CH | ΔAIC SS | ΔAIC story | Held-fixed rumour peak (CH / SS / story) |
|---|---|---|---|---|
| Network P1 → global share | **+95.9** | −1.2 | 0.0 | +0.6% / +0.1% / 0 (CH final C_A → 39% of data) |
| Source injection θ = 0 | **+305.4** | **+184.3** | **+70.9** | −52% / −59% / −34% |
| Bass external η = 0 | −3.8 | **+226.0** | **+80.8** | −1% / −26% / −17% |
| Emotion, popularity, content fit, connectivity, friend power | 0 | 0 | 0 | no effect by construction (uniform Mult_eff) |

**Paper-faithful (d+) (ranked Mult_eff): the paper's factors**
| Knock-out | ΔAIC CH | ΔAIC SS | ΔAIC story | Held-fixed peak (CH / SS) |
|---|---|---|---|---|
| Emotion (S_emo = 1) | +6.2 | +0.1 | 0 | +0.1% / −0.4% |
| Popularity (Pop = 1) | **−38.6** | −0.8 | −2.0 | +0.9% / −0.6% |
| Content fit (Info_match = 1) | −0.1 | 0 | 0 | 0 / 0 |
| Connectivity (co = 1) | **−24.6** | −4.9 | 0 | +0.9% / +1.0% |
| Friend power (fp = 1) | **−10.3** | +0.4 | 0 | +0.4% / 0 |
| Network | +21.1 | −2.5 | 0 | +0.3% / +0.1% |
| Injection | +430.4 | +134.0 | +113.2 | −36% / −62% |
| Bass | +4.7 | +223.8 | +70.0 | −15% / −24% |
(On the story κ = 0, so the per-user factors have no effect.) (d)-uniform, the secondary row, is in the same CSVs.

**Paper claims vs our results**
| Paper claim | Our result | Verdict |
|---|---|---|
| Fig. 7: no popularity → the push never peaks | Peak ±0.9%; refit is *better* without popularity (CH −38.6) | **Does not match** (User_att weight ≈ 0 in Phase 4) |
| Fig. 7: no emotion → much lower peak | ≤ 0.4%; +6.2 AIC on CH only | **Does not match** (weak text features, D31) |
| Fig. 7: no content fit / connectivity / friend power → lower, later peak | ≤ 1%, partly the opposite direction; refit better without them | **Does not match** (Info_match ≈ 0; regression near noise) |
| Fig. 8: the strong layer pushes harder | (d+) CH +0.035 (strong higher in 100% of bins); SS −0.008; story only at κ = 2 | **Partly** |
| Fig. 8: push rises with group activity / preference | Gact ρ: CH −0.43, SS +0.81, story +0.83 (sign = the Phase 4 Group_att coefficient); Gpre: no relation | **Partly / does not match** (regression artefact) |
| Fig. 6: own traits early, external later | CH own-trait share 0.45 → 0.19 in the only two windows with ≥ 10 A users; external dominates throughout | **Partly at most** |

---

### Phase 7 — SAIR-C (D73–D75)
Knobs u = λu·γ, v = λv/T90 (T90: CH 49.8 h, SS 16.5 h, story 2.0 h); ρ_I = (C_I − B)/C_I,data; D = ∫I dt over the window.
**Best point at equal total effort, D/D0 (block / refute / hybrid), main model, knobs inside the grid:**
| Set | E = 2% | E = 5% | E = 10% |
|---|---|---|---|
| CH | .974 / .970 / .971 | .934 / .930 / **.929** | .865 / .370 / **.016** |
| SS | .977 / .970 / .974 | .942 / **.918** / .925 | .884 / **.263** / .290 |
| story | .925 / .923 / .924 | **.800** / .806 / .801 | **.526** / .623 / .541 |
- Hybrid best only on CH at 10% (blocking tips P1, the campaign recruits A, final C_A .94 vs .02). Ties at 2–5%. Refute-only cannot
  reach 15% effort on CH / SS and block-only not on the story inside the grid (effort is non-monotone in the knobs, D74).
- Weighted cost J (w .01 / .05 / .2): cheapest grid point is a hybrid in all sets (CH w .05: λu .5, λv .5, D/D0 .068). J penalises
  knob size, so this is partly by construction. Robustness: (a+) refute best on CH / SS; (d)-uniform as the main model.
- Grid Pareto front: 14 (CH), 16 (SS), 15 (story) non-dominated points. Wide search (knobs to 50) only as sensitivity (CH hybrid .002).
- Feedback share at the best points: ρ = 1 ≈ 0 (≤ 0.5% SS, ≤ 9% CH); d1(t) frozen: CH block 0.3%, refute / hybrid 5–9% (2–5%), 70% refute at 10%;
  SS 1% / 5–17% / 46–47%; story 0 (κ = 0). The effect runs through the P1 fallback I/(I + A) for users with no sided friends.
- Checks: λ = 0 == Phase 5 (2.8e-14); sums ≤ 1.3e-13; numba == Python; D monotone on all grid lines; one dt warning (SS (λu, λv) = (5, 0), 1.04e-3) was
  resolved in D76 by running Sydney at dt/2 (now ≤ 5.2e-4).

### Phase 7 follow-up and Phase 8 (D76–D80)
**Tipping robustness (CH, 10% effort, hybrid D/D0):** κ = 5 → .013 (tips already at 5%); 12.5 → .016; 25 → .036; 50 → .037; fallback P1 = 0.5 (refit,
κ → 0.09, ΔAIC +100.9) → .72, no tipping. Refits: κ = 12.5 / 25 / 50 tie with the main fit, κ = 5 is +28.7 AIC.

**PHEME targeting, reduction 1 − D/D0 at k = 50 (degree / PageRank / betweenness / random):**
| Set | block | recruit | hybrid |
|---|---|---|---|
| CH | .532 / .160 / .035 / .001 | .977 / .977 / .977 / .975 | .985 / .985 / .973 / .970 |
| SS | .078 / .078 / .004 / .000 | .975 / .974 / .974 / .973 | .996 / .985 / .982 / .968 |
| story | .374 / .374 / 0 / .125 (± .10) | .926 / .925 / .926 / .908 | .975 / .977 / .875 / .840 |
- Block acts through source injection (88–99.8%); friend-count channel ≤ 0.9%. Story: friend channel 0 (κ = 0). Recruit and hybrid are saturated **at s = 1 only (D81: an artefact of the injection scale)**.

**BA agent model, seeds-only variant (superseded by D82; 10 graphs × 20 runs):** validation fails (final C_I 4.5% vs 91%, C_A 95% vs 1.9%, peak 2.7% vs 36.8%). Block reduction (fitted κ) 2 → 8%
vs random 0.3 → 2%; recruit .88 for every method; hybrid with hubs falls to .84 (random .875) at 2%; κ = 0: block −0.3 … −6%, recruit −6 pp.

### Phase 8 fixes (D81, D82)
**Equal effort (CH, degree [random]; block / recruit / hybrid, effort = fraction of N moved):** E ≤ 0.5%: .352 [.003] / .815 [.086] / .921 [.007]; E ≤ 1%: .532 / .950 / .993.
Recruit s = 1 moves 32% of N per 50 recruits (s = 0.1: 3.2%, s = 0.01: 0.32%). Sydney: same order; story: block (.374) beats recruit (.308) at E <= 2%, loses at 5%. α = 4.4 / 21.5 refits (+39.5 / +11.5 AIC): the CH order is unchanged.
CH s = 0.01 recruit: injection alone .005, friend counts alone .105, both .815 (interaction).

**BA with source injection (calibrated β×2, α×0.1; validation passes, shape RMSE .134):** reduction of the final size at k = 0.1 / 0.5 / 1 / 2% (fitted κ; degree vs random): block 55 / 90 / 90 / 90 vs
0.1 / 0.9 / 1.9 / 3.5; recruit (s = 0.1) 70 / 83 / 86 / 89 vs 45 / 73 / 79 / 84; hybrid 88 / 99 / 99 / 99 vs 32 / 63 / 72 / 79. κ = 0: block 54 / 89 / 89 / 88, recruit 54 / 70 / 75 / 80, hybrid 70 / 97 / 98 / 98.

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

## 4b. Decisions taken in session 3 (2026-10-17/18; full text in DECISIONS.md)

| # | Decision |
|---|---|
| D57 | Bass external-influence term η_I·S, η_A·S (η ∈ [0, 1]); variants (a+), (d+), (d+)-uniform |
| D58 | Rate bounds [1e-3, 50] for every model; Euler dt = bin/50, halved until rate·dt ≤ 0.025 |
| D59 | Forecast cut = max(50% point of C_I, 10th bin), N/A if < 5 bins remain; moving-block bootstrap (5 bins, 50 resamples) + cold-start check |
| D60 | Recovery for the main model; **κ, w and α are not identifiable, so their values must not be interpreted** |
| D61 | (Superseded) main = (d) or (d+) by CH AIC |
| D62 | numba Euler kernel (bitwise identical to Python); per-job checkpoints; staged run with progress / ETA logging |
| D63 | Stage 2 validates (a), (a+) and the main model (+ (d)-uniform as a secondary row) |
| D64 | Cross-model starts (up: exact embeddings; down: projections; flattened-ranking seeds); new check that no model fits worse than its submodel |
| D65 | κ widened to [0, 50]; restarts from the κ ≤ 10 endpoints + 5 fresh starts per main candidate |
| D66 | (Superseded) (d+)-uniform variant added; main = CH-AIC-best among the four SAIR-H models → (d)-uniform |
| D67 | Phase 6 ablation: main model + paper-faithful (d+); held-fixed and refit ΔAIC; Fig. 6 / 8 tests |
| D68 | Phase 5 refit results (model contest, forecast, recovery, bootstrap) |
| D69 | **Final main-model rule: best total AIC over the three sets → (d+)-uniform** |
| D70 | κ at 50 in the ranked models = a step "follow the sided-friends majority"; κ not widened again |
| D71 | Forecast limitation: no model forecasts later waves; (d)-uniform unstable on SS; main model ≥ (a) |
| D72 | Plan for Phases 7–9: event-relative knobs + Pareto front; BA network in Phase 8; optional extra recovery starts in Phase 9 |
| D76 | Phase 7 fixes: Sydney dt/2 (u, v in step_dt is a no-op); CH tipping robustness over κ and the fallback P1 |
| D77 | Phase 8 A design: follow-graph centralities, pools, block / recruit channels, hybrid split, own checks |
| D78 | Phase 8 A results: injection channel dominates; recruiting saturated |
| D79 | Phase 8 B design: BA agent model, targeting, paired runs, CI over graphs |
| D80 | Phase 8 B results (seeds-only variant); validation against mean-field fails |
| D81 | Recruit fairness: injection scale s, effort as % of N moved, equal-effort comparison, α-fixed refits |
| D82 | BA with source injection on hubs: calibration, validation passes, targeting rerun (denying hub never blocked) |
| D73 | SAIR-C model and knobs; campaign counts in C_A; ρ_I uses C_I − B; kernel extended, Phase 5 bitwise unchanged |
| D74 | Equal-effort search along rays (all roots), grid range primary + wide sensitivity, Pareto, cost J |
| D75 | Phase 7 results, feedback test (ρ = 1 vs d1 frozen), checks and the SS dt warning |

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

## 6. Final state and how to rebuild the deliverables

- Report: `python src/report_check.py --write` (refresh the tables from the CSVs), `python src/report_check.py` (verify; 28 tables, 61 claims),
  `python src/export_report.py` (PDF). Prose numbers that are not in a table must be added to `claims()` in `src/report_check.py`.
- Slides: `python src/slides.py --render` (needs PowerPoint for the PNG render and layout check; building alone needs only python-pptx).
- Notebook: `notebooks/experiments.ipynb` reruns everything from the checkpoints (about 5 min).
- If any result changes, rerun the affected phase, then the three commands above; the check fails if a quoted number no longer matches.

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
| 9 | `python src/sair.py --recovery-extra`; `python src/report_check.py`; `python src/export_report.py`; `python src/slides.py --render` | ~1.5 h (recovery, checkpointed); seconds; ~10 s; ~1 min | `results/phase5_recovery_extra.csv`, `docs/REPORT.pdf`, `presentation/` |
| 8 | `python src/targeting.py --part A or B or AB --check` (`--fresh`, `--check-only`) | A ~25 min (22 α refits ~12 min + 140 jobs), B ~15 min (25 calibration + 260 jobs, checkpointed; the first BA run builds 10 graphs; set `PYTHONIOENCODING=utf-8` when redirecting output) | `results/phase8_{pheme,channels,effort,alpha_fits,ba,ba_validation,ba_calibration}.csv`, `phase8_plot6/6b/7/8/8b/8c/9_*.png` |
| 7 | `python src/control.py --check` (`--stage 1\|2`, `--fresh`, `--check-only`; `--tipping` = CH κ / fallback refits, ~25 min) | ~2.5 min stage 1 (147 jobs, 12 workers, checkpointed), seconds for stage 2 | `results/phase7_{grid,strategies,feedback,cost}.csv`, `phase7_plot4/5/5b_*.png` |

- `<set>` ∈ {charliehebdo, sydneysiege, story_three_shootings}.
- The pipeline is deterministic (seed 42; two full runs gave byte-identical outputs).
- Raw data must be present in `data/raw/` (see its README for URLs and MD5s).
- Large files are git-ignored: graphs, tweets, users, features, network, groups, drive, friends, model_curves,
  ablation_edges, checkpoints.
- Phase 5 reuse: `sair.load_main()`, `sair.get_context(key)`, `sair.simulate(sc, kind, params, record=True)` with kind in
  `sair.MODELS`; fitting via `sair.start_jobs` + `sair.run_jobs(pool, jobs, label, ckpt)` (checkpointed, D62).
- Checkpoints (`data/processed/checkpoints/`): after any change to the setup use `--fresh` or delete the affected files.
  Jobs built from fitted values carry a fingerprint and rerun automatically if that fit changes.

Phase 5 stages: **Stage 1** = all model fits (`--stage 1`), **Stage 2** = validation of (a), (a+), the main model and
(d)-uniform (`--stage 2`); without `--stage` both run in order. **Stage 3** = Phase 6 (`python src/ablation.py --check`).
A run that is interrupted resumes from its checkpoints (a crash and a system restart were both recovered this way).
Monitor progress in the log: every job prints done / total, elapsed and ETA.

Each module runs its own phase check with `--check`. Every phase check must pass before a commit.

---

## 8. Open decisions

None.
