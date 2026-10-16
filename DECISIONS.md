# Decisions & Corrections Log

Every choice made where the paper is vague or wrong. Claude Code appends new entries here.

## Corrections to the original SAIR paper
| # | Issue in paper | Our fix |
|---|---|---|
| C1 | Eq. 26 attaches δ1 (rumour push) to A and δ2 to I | δ1 drives I, δ2 drives A (see CLAUDE.md) |
| C2 | Eq. 24 binomial looks like a new quantity | Mean of Binomial(m,p)/m = p, so δ = Drif; use mean over susceptible users |
| C3 | "Initiative(v_i)" in Algorithm 1 is never defined | Dropped |
| C4 | Regression target for Mult_eff not specified | Target = user reposted the rumour (1/0) |
| C5 | Eq. 7 adds a 0–1 similarity to a raw mutual-friend count | Use Jaccard of friend lists |
| C6 | γ used for group weight (Eq. 10) and removal rate (Eq. 22) | Group weight renamed `w_part` |

## Modelling choices
| # | Choice | Reason |
|---|---|---|
| M1 | Strong layer = directly linked to a core node; weak = rest | Paper gives no exact rule |
| M2 | Groups = clusters of a KNN similarity graph on Prb | KNN is a classifier; paper's intent is grouping |
| M3 | r derived from mpr (higher mpr → lower r) | Paper never links mpr to r |
| M4 | NRC lexicon, 8 emotions | Paper's lexicon unspecified; tweets are English |
| M5 | Validation adds RMSE | Paper reports plots only |

## Added during implementation
<!-- Claude Code: append new entries below with date + phase -->
| # | Date / phase | Choice | Reason |
|---|---|---|---|
| D1 | 2026-10-16 / P0 | **Primary data = PHEME rumour scheme dataset** (figshare 2068650), English threads only (GW 25 threads, SS 71). RumourEval-2019 is used only to cross-check labels. PHEME-RNR is a fallback and is not used for the main results. | It is the only source with human stance labels **and** follow edges **and** retweets. Gold labels are the most defensible. See `data/raw/README.md`. |
| D2 | 2026-10-16 / P1 (revised) | Directed graph, edge u→v = u receives information from v. `kind="follow"` comes from `who-follows-whom.dat` (u follows v), merged over the event's threads. `kind="interaction"` is added when u replied to or retweeted v and there is no follow edge already. | Follow edges cover about 75% of users. P1 (Phase 4) needs neighbours. Result: SS 15,804 follow + 7,348 interaction edges, 2 isolated nodes. Ottawa 26,249 + 9,591, 5 isolated nodes. |
| D3 | 2026-10-16 / P0 | Reply stance **toward the rumour** = composition of source stance and `responsetype-vs-source`: agreed + supporting → support, disagreed + supporting → deny, agreed + denying → deny, disagreed + denying → support. Appeal-for-more-information → query, otherwise comment. A source tweet's `support` gives its own stance (underspecified → comment). | PHEME and RumourEval reply labels are relative to the *source tweet*, not the rumour. RumourEval SDQC equals the raw vs-source label on 279/281 (GW) and 1103/1107 (SS) shared tweets. |
| D4 | 2026-10-16 / P0 | A **retweet of a source tweet** inherits the source stance (retweet of a supporting source → I, of a denying source → A, of an underspecified source → neutral). Retweeters join the population. | They are reposts of the claim. This raises N from 226 to 4,308 (GW) and from 873 to 11,245 (SS), and gives the Phase 4 target "reposted rumour" directly. |
| D5 | 2026-10-16 / P0 | Users who only query or comment stay in **S**. | User's decision: exposed but no side taken. |
| D6 | 2026-10-16 / P1 (revised) | State per user: S until their **first** support/deny post. That post fixes I or A for good (no I↔A switch). From then on, **any** post (source, reply or retweet) counts as activity. The user moves to R 24 h after the first post that is followed by a 24 h gap (or 24 h after their last post), and R is absorbing. t0 = first post of the event. | SAIR has no I↔A transition (user decision). Implements "inactive 24 h → R". |
| D7 | 2026-10-16 / P0 | Friend lists = **in-dataset** follow lists only (who-follows-whom is restricted to thread users). | Full Twitter friend lists are not available. The Jaccard in C5 uses these. |
| D8 | 2026-10-16 / P0 | No posts from "the month before the rumour" exist. Group keywords (Gpre/TextRank) use group posts inside the event window. | Data limitation (affects Phase 3). |
| D9 | 2026-10-16 / P0 | Graphs are saved with `pickle` under the `.gpickle` name. | networkx 3 removed `write_gpickle`. |
| D10 | 2026-10-16 / P0 | The A state is thin, especially for GW (12 deniers by reply + 2 retweets of the one denying source; SS: 83 + 109). It is reported as-is and **not** augmented with a classifier. | Keeps gold-label fidelity. This is a known limitation. |
| D11 | 2026-10-16 / P1 | Events: **Sydney Siege = primary, Ottawa shooting = second**. Germanwings is dropped. | Users who ever deny, per English event (after D3 + D4): GW 14, SS 190, **Ottawa 823**, Charlie Hebdo 294, Ferguson 148, prince-toronto 32, putinmissing 55, ebola-essien 6. Ottawa has by far the richest A curve. Deviates from the SAIR paper's GW + SS pairing. |
| D12 | 2026-10-16 / P1 | Users who post **both** stances are counted and keep the side of their first post (D6). | SS: 26 users (of 10,727 sided). Ottawa: **205** (of 13,902 sided; about 25% of its deniers also support at some point). GW: 3. |
| D13 | 2026-10-16 / P1 | Bin size per event: compare 60/30/15 min and keep the largest whose I and A curves both rise and fall (interior peak, ≥ 3 rising points, end < 10% of peak) with ≥ 20 non-zero points. **SS: 60 min** (I 82 points, A 58). **Ottawa: 60 min** (I 85, A 49). | The 15/30-min curves overlay the 1-h ones almost exactly (`results/bin_comparison_*.png`), so 1 h loses nothing. Table: `results/phase1_bin_choice.csv`. |
| D14 | 2026-10-16 / P1 | **Observation, not fixed:** the real I curve plateaus and peaks at exactly 24 h, at about 93% of N. | Most users post once (a retweet) in the first hours, so under the 24 h rule they all stay in I until hour 24. N contains only users who posted (non-posting followers are not in the data), so S drains to 2–5% within hours. Relevant for fitting in Phase 5. |
