# Rumour vs. Correction: Network Propagation and Mitigation


When a false rumour spreads online during a crisis, a correction usually starts spreading too, and the two compete for the same
people. This project models that competition as a contagion: every user starts as susceptible (S), then either spreads the rumour (I),
spreads the correction (A), or loses interest and stops (R). How fast each side spreads depends on what a user's friends are sharing,
on the source accounts that first broadcast the claim, and on outside exposure such as news.

We build on the SAIR model of Li et al. (IEEE Trans. Big Data, 2026) and fit it to real Twitter data from the PHEME dataset
(Charlie Hebdo, Sydney Siege and one Ottawa shooting story; false and unverified rumours only). We show that the paper's final model
cannot let the rumour and the correction grow at the same time, fix this with **SAIR-H**, and then use the fitted model as a simulator
to compare ways of fighting a rumour: blocking rumour spreaders, running a correction campaign, or both.

## Data
 
We use PHEME, a public dataset of Twitter conversations from breaking news events, in which each tweet is labelled with a stance
(support, deny, query, comment) and each story with a veracity (true, false, unverified). Users who support or retweet a supporting
source become rumour spreaders (I), users who deny or retweet a denying source become correction spreaders (A), and an event is kept
only if at least 100 people posted corrections and it has at least 15 informative hours; we fit the cumulative curves C_I(t) and
C_A(t)

## Key findings

- **The paper's final model (Eq. 26) has a seesaw flaw.** The rumour grows only if δ1 > 1/(1 + S) and the correction only if
  δ1 < S/(1 + S). Since S/(1 + S) ≤ ½ ≤ 1/(1 + S), both can never grow together. Yet the real data show both growing together early
  in all three datasets (0.25–1.5 h in the Ottawa story, 1–5 h in Sydney Siege, 1–3 h in Charlie Hebdo).
- **Our fix, SAIR-H, keeps separate joining and stopping rates** (from the paper's Eq. 22) and multiplies only the joining term by the
  friends' push δ(t). With κ = 0 it becomes Eq. 22, so any gain over Eq. 22 comes from the network.
- **The main model, SAIR-H with outside exposure and source injection, fits best.** Its total AIC is −4295.2, the best of nine
  models, and 1,510.7 better than the paper's Eq. 26 (592.9 on Charlie Hebdo, 299.9 on Sydney Siege, 617.9 on the Ottawa story).
  It also wins on BIC.
- **The best control strategy depends on the event.** At an equal effort of 10% of users, the rumour damage D/D0 is:

  | Dataset | Block only | Refute only | Hybrid |
  |---|---|---|---|
  | Charlie Hebdo | 0.865 | 0.370 | **0.016** |
  | Sydney Siege | 0.884 | **0.263** | 0.290 |
  | Ottawa story | **0.526** | 0.623 | 0.541 |

  On Charlie Hebdo the hybrid pushes corrections past a tipping point: they become the majority within about 6 hours. At 2–5% effort
  all strategies give almost the same small gain, so hybrid control is not always best. Blocking alone is weak in most cases.
- **Whom to target:** ranking rumour spreaders by degree and blocking the top k worked best overall, compared with PageRank,
  betweenness and random picks (15 draws).


## Repository map

```
src/
  data_prep.py      Phase 1  PHEME -> users, states, cumulative curves, graph
  features.py       Phase 2  user features (emotion, tags, popularity, connectivity, friend power)
  network.py        Phase 3  PageRank layers, profile-based groups
  drive.py          Phase 4  influence blocks, regression, Mult_eff, P1, Drif
  sair.py           Phase 5  all models, Euler simulation, fitting (Nelder-Mead), AIC/BIC comparison
  ablation.py       Phase 6  checks of which model parts the data need
  control.py        Phase 7  SAIR-C knobs (blocking, correction campaign), equal-effort strategies
  targeting.py      Phase 8  centrality targeting (degree, PageRank, betweenness, random)
  report_check.py   Phase 9  checks the reported numbers against the result CSVs
  export_report.py  Phase 9  report export
results/            every table (CSV) and figure (PNG)
notebooks/          experiments.ipynb: reruns everything from checkpoints and redraws every figure
data/raw/           downloaded data (not in git; see data/raw/README.md); data/processed/ intermediate files and checkpoints
```

## How to run

```
python -m pip install -r requirements.txt          # Python 3.10
# download the raw data into data/raw/ (URLs and MD5 checksums in data/raw/README.md)
set LOKY_MAX_CPU_COUNT=4                            # silences a harmless joblib warning on Windows
python src/data_prep.py --check                     # Phase 1, ~90 s
python src/features.py --check                      # Phase 2, ~15 s
python src/network.py --check                       # Phase 3, ~30 s
python src/drive.py --check                         # Phase 4, ~30 s
python src/sair.py --check                          # Phase 5, 6–8 h from scratch on 12 cores, minutes from checkpoints
python src/ablation.py --check                      # Phase 6, ~1.5–2 h from scratch
python src/control.py --check                       # Phase 7, ~3 min
python src/targeting.py --part AB --check           # Phase 8, ~40 min from scratch
python src/report_check.py                          # verifies the reported numbers against the CSVs
```

Every long phase saves a checkpoint after each finished optimiser run or simulation job in `data/processed/checkpoints/`, so an
interrupted run resumes where it stopped. `--check-only` runs only the phase check on the saved results, and `--fresh` drops a
phase's checkpoints. The pipeline is deterministic (seed 42). Set `PYTHONIOENCODING=utf-8` when redirecting output on Windows.

## References

- Li, T., Liu, C., Liu, J., Wang, R., Zhu, H., Xiao, Y. (2026). A malicious and anti-malicious information propagation dynamics model
  based on higher-order diffusion networks. *IEEE Trans. Big Data* 12(3), 1020–1032.
- Zhong, X., Zeng, J., Xiang, W., Caraballo, T., Deng, F., Peng, Y. (2026). Stochastic hybrid rumor control: a data-driven ensemble
  learning control algorithm. *J. Syst. Sci. Complex.* 39(4), 1436–1462.
