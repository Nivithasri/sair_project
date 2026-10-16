# Our Extension — SAIR-C and Centrality Targeting

SAIR only *predicts* the rumour–correction contest. We turn it into a decision tool:
**how hard** to intervene (SAIR-C) and **whom** to target (centrality).

---

## §1 SAIR-C: control knobs + cost (Phase 7)

**Knobs (0 = off, 1 = max)**
| Knob | Model effect | Real-world analogue |
|---|---|---|
| u (blocking) | moves I → B (Blocked) at rate u | suspending accounts / deleting posts |
| v (correction campaign) | moves S → A directly at rate v | official fact-check pushed to everyone |

B is separate from R: R = stopped voluntarily, B = forced to stop (counts as control effort).

**Equations (new terms marked ←)**
```
dS/dt = −δ1·I·S − δ2·A·S − v·S              ←
dI/dt =  δ1·I·S − (1 − δ1)·I − u·I          ←
dA/dt =  δ2·A·S + v·S − (1 − δ2)·A          ←
dB/dt =  u·I                                 ←
dR/dt = (1 − δ1)·I + (1 − δ2)·A
```
Invariant: S + I + A + B + R = 1.

**Phase 7 uses the SAIR-H rates** (Phase 5 model (d), `docs/SAIR_paper.md` §5, DECISIONS D50): the contact terms become
β·δ1·I·S and α·δ2·A·S, and the removals become γ·I and ε·A. The u and v terms are unchanged. The worked step below
keeps the Eq. 26 form as a hand-checkable example.

**Cost (form borrowed from DR-DB paper)**
J = Σ_t [ A1·I(t) + ½·c1·u² + ½·c2·v² ] Δt,  with A1 = 1, c1 = c2 = 2.
Squared costs: heavy intervention gets disproportionately expensive → forces a balanced strategy.

**Experiments**
- Four fixed strategies: none (0,0), block only (0.1,0), refute only (0,0.02), hybrid (0.1,0.02).
- Grid search: u ∈ {0, 0.05, …, 0.5}, v ∈ {0, 0.01, …, 0.1} → cost heatmap, mark minimum.
- Expected: hybrid cheapest.

**Feedback effect worth reporting:** blocking lowers P1 for followers → lower Drif1 → lower δ1, because δ is recomputed each step. Fixed-rate models (like DR-DB) miss this.

Worked step: S=0.90, I=0.05, A=0.05, δ1=0.6, δ2=0.4, u=0.1, v=0.02, Δt=1
→ S=0.837, I=0.052, A=0.056, B=0.005, R=0.050 (sum 1.000); step cost = 0.0604.

---

## §2 Centrality-targeted intervention (Phase 8)

Budget: act on exactly **k users**; same k for every method (fair comparison, equal control cost → compare damage only).

**Selection methods**
| Method | Meaning | networkx |
|---|---|---|
| Degree | most connections / followers | `nx.degree_centrality` |
| PageRank | followed by important accounts | `nx.pagerank` |
| Betweenness | bridges between communities | `nx.betweenness_centrality(G, k=500)` |
| Random | baseline (average 15 runs) | `random.sample` |

**Actions**
- Block: top-k current rumour spreaders → B (no longer count as rumour friends).
- Recruit: top-k susceptible users → start in A (count as correction friends).
- Hybrid: k/2 block + k/2 recruit.

**Mechanism:** targeting changes P1 for every follower of a chosen user → Drif1 = sigmoid((2P1−1)·Mult_eff) drops → mean δ1 drops → SAIR-C run gives a smaller rumour curve.
Example: friends {H(rumour hub), a, b (rumour), c (correction)}, Mult_eff=2: P1=0.75 → Drif1=0.73; block H → P1=0.67 → Drif1=0.66. A hub shifts this for all its followers at once.

**Metrics:** rumour peak (max I), total reach (users who passed through I, incl. B).
**k values:** 10, 25, 50, 100.

**Expected:** degree & PageRank ≫ random (scale-free hubs — same logic as targeted attack vs random failure in robustness); betweenness strong when the graph has communities; hybrid likely best. Report actual results honestly.

---

## Contributions (for README / slides)
1. SAIR-C: control knobs u, v + blocked state B + cost function, cheapest-strategy search.
2. Centrality-targeted intervention on the real Twitter graph.
3. RMSE validation (paper had plots only).
4. Six documented corrections to the paper (DECISIONS.md).

## Limitations
- Mean-field: users averaged into rates; no individual spreading paths.
- Knobs constant over time (time-varying optimal control = future work).
- Two Twitter events, text only.

## Future work
Time-varying optimal control (DR-DB style); agent-based simulation on the graph; RL-learned knobs (SICR-TM style); multimodal emotion.
