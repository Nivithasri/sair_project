# Whom to Silence, Whom to Amplify

**Centrality-targeted hybrid control of rumour–correction competition on social networks.**

This project re-implements the SAIR rumour–correction model of Li et al. (IEEE Trans. Big Data, 2026) on the PHEME Twitter rumour
dataset (Charlie Hebdo, Sydney Siege and one Ottawa story; false and unverified rumours only), under strict information-leakage
controls, and tests it honestly against simpler models. The paper's model turns out to have a structural flaw (its rumour and
correction can never grow at the same time), and its per-user push factors are not supported by the data. We replace it with
SAIR-H+ (SAIR rates driven by the friends' majority, plus Bass external influence and source injection), which wins a nine-model
contest by total AIC, and extend it to **SAIR-C** (blocking and correction-campaign control) and to **centrality-targeted**
intervention on the real follow graph and on synthetic Barabási–Albert networks. The full write-up is
[`docs/REPORT.md`](docs/REPORT.md) ([PDF](docs/REPORT.pdf)); every decision is logged in [`DECISIONS.md`](DECISIONS.md).

## Key findings

- **The paper's Eq. 26 cannot produce co-growth.** With δ2 = 1 − δ1, the rumour grows only if δ1 > 1/(1 + S) and the correction only if
  δ1 < S/(1 + S); these never overlap. Eq. 26 is the worst of nine models on every set (report Section 4.2, Table 4).
- **SAIR-H+ is the best model family** (total AIC -4295.2 over the three sets). Source injection and Bass external influence are what
  the data need; the paper's emotion, popularity, content-fit, connectivity and friend-power factors change the rumour peak by about 1%
  or less (Tables 10–11).
- **"Hybrid is best" is not general.** At equal effort the strategies tie at 2–5% of N; at 10% a hybrid tips Charlie Hebdo into a
  correction-dominated state (D/D0 = 0.016), Sydney prefers refutation and the Ottawa story blocking. The tipping survives κ from 5 to 50
  but depends on how users without sided friends are modelled (Tables 12, 16).
- **Targeting hubs works, but for a specific reason.** Blocking the top accounts works because the hubs are the rumour's source
  authors (injection channel), not through conformity; recruiting central correctors works through an interaction of a small
  correction mass with the followers' friends' majority (Tables 17–21).
- **On a calibrated Barabási–Albert agent model, hubs beat random targeting by a wide margin** (blocking the top 2% by degree: 90.5% vs
  3.5% at random); degree, PageRank and betweenness are indistinguishable there (Tables 22–24).

## Repository map

```
src/
  data_prep.py      Phase 1  PHEME -> users, states, cumulative curves, graph
  features.py       Phase 2  leakage-safe user features (emotion, tags, popularity, friend power)
  network.py        Phase 3  PageRank layers, profile-based groups
  drive.py          Phase 4  influence blocks, regression, Mult_eff, P1, Drif
  sair.py           Phase 5  all models, numba Euler kernel, fitting, validation, extra recovery (--recovery-extra)
  ablation.py       Phase 6  knock-outs, refit importance, tests of the paper's Fig. 6 and Fig. 8
  control.py        Phase 7  SAIR-C knobs, equal-effort strategies, Pareto front, feedback test, tipping (--tipping)
  targeting.py      Phase 8  centrality targeting on PHEME (Part A) and the BA agent model (Part B)
  report_check.py   Phase 9  generates the report tables from the CSVs and checks every quoted number
  export_report.py  Phase 9  REPORT.md -> REPORT.pdf
  slides.py         Phase 9  presentation/slides.pptx (+ PNG render check)
docs/               REPORT.md / .pdf, VIVA_QA.md, HANDOFF.md, paper notes (SAIR_paper.md, extension.md, DRDB_reference.md)
results/            every table (CSV) and figure (PNG)
notebooks/          experiments.ipynb: reruns everything from checkpoints and redraws every figure
presentation/       slides.pptx and the slide renders
data/raw/           downloaded data (not in git; see data/raw/README.md); data/processed/ intermediate files and checkpoints
DECISIONS.md        corrections C1–C6 and decisions D1–D83     PLAN.md   phase plan
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
python src/control.py --check                       # Phase 7, ~3 min (+ --tipping, ~25 min)
python src/targeting.py --part AB --check           # Phase 8, ~40 min from scratch
python src/report_check.py                          # verifies the report's numbers against the CSVs
```

Every long phase checkpoints each finished optimiser run or simulation job in `data/processed/checkpoints/`, so an interrupted run
resumes where it stopped; `--check-only` runs only the phase check on the saved results, and `--fresh` drops a phase's checkpoints.
The pipeline is deterministic (seed 42). Set `PYTHONIOENCODING=utf-8` when redirecting output on Windows.

## Where each figure comes from

| Report figure | File | Produced by |
|---|---|---|
| 1 data curves | `results/cumulative_curves_charliehebdo.png` | `src/data_prep.py` |
| 2 model fits | `results/phase5_plot2_cumulative_fit.png` | `src/sair.py` (stage 1) |
| 3 forecasts | `results/phase5_plot2c_forecast.png` | `src/sair.py` (stage 2) |
| 4 ablation (paper-faithful) | `results/phase6_plot3b_importance_paper.png` | `src/ablation.py` |
| 5 Pareto front | `results/phase7_plot5b_pareto.png` | `src/control.py` |
| 6 knob heatmap | `results/phase7_plot5_heatmap.png` | `src/control.py` |
| 7 equal effort (PHEME) | `results/phase8_plot9_effort.png` | `src/targeting.py --part A` |
| 8 BA validation | `results/phase8_plot8b_ba_validation.png` | `src/targeting.py --part B` |
| 9 BA targeting | `results/phase8_plot8_ba_reduction.png` | `src/targeting.py --part B` |

All other PNGs in `results/` (per-phase diagnostics) are produced by the same modules; the notebook redraws all of them.

## Citation

Data: Zubiaga, A., Liakata, M., Procter, R., Wong Sak Hoi, G., Tolmie, P. (2016). Analysing how people orient to and spread rumours in
social media by looking at conversational threads. *PLoS ONE* 11(3), e0150989. PHEME rumour scheme dataset, figshare,
doi:10.6084/m9.figshare.2068650.v2 (CC BY 4.0). Emotion lexicon: Mohammad, S. M., Turney, P. D. (2013), *Computational Intelligence*
29(3), 436–465 (NRC Emotion Lexicon, non-commercial research use).

Papers: Li, T., Liu, C., Liu, J., Wang, R., Zhu, H., Xiao, Y. (2026). A malicious and anti-malicious information propagation dynamics
model based on higher-order diffusion networks. *IEEE Trans. Big Data* 12(3), 1020–1032. Zhong, X., Zeng, J., Xiang, W., Caraballo, T.,
Deng, F., Peng, Y. (2026). Stochastic hybrid rumor control: a data-driven ensemble learning control algorithm. *J. Syst. Sci. Complex.*
39(4), 1436–1462. Bass, F. M. (1969). A new product growth for model consumer durables. *Management Science* 15(5), 215–227.
