# DR-DB Paper — Reference Only (borrowed ideas, NOT implemented)

**Paper:** X. Zhong, J. Zeng, W. Xiang, T. Caraballo, F. Deng, Y. Peng, "Stochastic Hybrid Rumor Control:
A Data-Driven Ensemble Learning Control Algorithm," *J. Syst. Sci. Complex.*, 39(4): 1436–1462, 2026.
DOI 10.1007/s11424-025-4372-4. Original PDF: `papers/DRDB.pdf`.

## One-paragraph summary
A 7-group stochastic rumor model (S, I1 silent believers, I2 spreaders, B blocked, M refuters,
X debunking media, Y uninvolved media) with two control families: **Double Refutation** (people and
media spread the truth) and **Double Blocking** (continuous blocking u1, u2 plus impulse "mass-ban"
blocking at fixed times). Optimal controls come from the stochastic maximum principle; a random forest
is then trained to imitate them (only 4.1% costlier than the optimum, much faster).

## What we borrow
1. **The idea of combining refutation and blocking** — our knobs v (refute) and u (block).
2. **The quadratic cost form:** J = damage + ½·c·(control)²  with weights A = 1, c = 2 (their Table 3).
3. **Comparing fixed baseline strategies by cost** — like their "Case 1–6" comparison.
4. **A blocked state B** separate from voluntary removal.

## What we do NOT use
Stochastic differential equations, impulse control, adjoint equations / Theorem 3.3, random-forest controller.
These are listed as future work.
