# SAIR Paper — Implementation Notes

**Paper:** T. Li, C. Liu, J. Liu, R. Wang, H. Zhu, Y. Xiao, "A Malicious and Anti-Malicious Information
Propagation Dynamics Model Based on Higher-Order Diffusion Networks," *IEEE Transactions on Big Data*,
vol. 12, no. 3, pp. 1020–1032, May/June 2026. DOI 10.1109/TBDATA.2026.3668545. Original PDF: `papers/SAIR.pdf`.

These notes restate the model in our own words with every equation needed for code.
Known errors in the paper are marked **[FIX]** and logged in `DECISIONS.md`.

---

## §1 Problem and big picture
- Rumour (malicious info) and its correction (anti-malicious info) spread at the same time and compete.
- Goal: predict hour by hour how many users spread the rumour, the correction, or have stopped.
- Base: SIR epidemic model; new infections ∝ S × I (contact between susceptible and spreaders).

**States**
| State | Meaning |
|---|---|
| S | Not yet exposed to either message |
| A | Spreads the correction (anti-infected) |
| I | Spreads the rumour (infected) |
| R | Lost interest / stopped (removed) |

**Three mechanisms the paper adds**
1. Emotion influence — users share content whose emotion matches theirs.
2. "Higher-order" network — here this just means a **layered** network (core / strong / weak) plus hidden behaviour groups. (Not hypergraphs.)
3. Game between rumour and correction — users side with what their friends spread.

**Pipeline**
```
data (tweets, follows, emotions, tags)
 → per-user factors (§2) → layers + groups (§3)
 → regression → Mult_eff → game → Drif1, Drif2 (§4)
 → average pushes → δ1, δ2 → SAIR ODEs → S,A,I,R curves (§5)
 → compare with real data (§6)
```

**Inputs:** follow graph G=(V,E); behaviour log A={(action, user, time)}; rumour texts C={(id, text)}.
**Outputs:** Drif per user; S(t), A(t), I(t), R(t).

---

## §2 Per-user factors (paper §III-A, Eq. 1–8)

**Emotional similarity (Eq. 1–4)**
- Emotion set E = {e1..en} (we use NRC, n = 8).
- Per post: intensity on each emotion from lexicon score × modifier coefficients θ_m ("very" boosts, "not" flips).
- Recency weight of post j: ω_j = exp(−λ (t_current − t_j)).
- User vector V = weighted average of post vectors. Rumour vector V_current.
- S_emo(v_i) = (V · V_current) / (‖V‖ ‖V_current‖)  (cosine)

**Content fit (Eq. 5)**
Info_match(v_i) = |Tag(rumour) ∩ Tag(user)| / |Tag(rumour) ∪ Tag(user)|  (Jaccard)

**Information popularity (Eq. 6)** — half-life decay
Pop(t) = (1/2)^((t − t0)/w),  t0 = rumour appearance time, w = half-life.

**User connectivity (Eq. 7)**
co(v_i, v_j) = sim_tag(v_i, v_j) + sim_friends(v_i, v_j)
**[FIX]** paper's sim_friends is a raw count → use Jaccard of friend lists.

**Friend propagation power (Eq. 8)**
Fri_drive(v_i) = (1/N) Σ_j ( Ns[c(v_j)] + Ns[r(v_j)] + μ · Ns[rd(v_j)] )
average comments, retweets, and reads (weakened by μ < 1) per message over the user's N friends.

---

## §3 Layered network and hidden groups (paper §IV-B first half, Eq. 9–11, 17)

**Layers**
1. PageRank on G → top nodes = **core**.
2. Strong layer = linked to core; weak layer = others (paper gives no exact rule — see DECISIONS M1).
3. Messages lose strength crossing layers: efficiency factor r, used as (1 − r).
4. Node efficiency score (Eq. 17): mpr(v_i) = τ [k1 · pr(v_i) + k2 · Info_match(v_i)].  (Link mpr → r is our choice, DECISIONS M3.)

**Groups**
- Behaviour similarity (Eq. 9): Prb(v_i, v_j) = number of identical actions on the same post.
- Cluster users by Prb (paper says KNN; we use KNN graph → clusters, DECISIONS M2).
- Group activity (Eq. 10): Gact(g_i) = σ · (#original posts) + w_part · (#reposts) **[FIX: γ renamed w_part]**
- Group preference (Eq. 11): Gpre(g_i) = Jaccard(Gtag(g_i), Tag(rumour)), Gtag from TextRank keywords of the group's posts in the month before the rumour.

---

## §4 Influence score and push to spread (paper Eq. 12–16, 18–21)

**Blocks**
| Block | Formula |
|---|---|
| User_att (Eq. 13) | S_emo × Info_match × Pop(t) |
| hnode_inf (Eq. 14) | co × Fri_drive × (1 − r)  — message from a higher layer (small r) |
| lnode_inf (Eq. 15) | co × Fri_drive × (1 − r)  — message from a lower layer (large r) |
| Group_att (Eq. 16) | Gact × Gpre |

Products mean every part must be present (Pop → 0 kills User_att).

**Multiple linear regression (Eq. 12)**
Mult_eff(v_i) = γ0 + γ1·User_att + γ2·lnode_inf + γ3·hnode_inf + γ4·Group_att
**[FIX]** target not specified → reposted rumour (1/0). Algorithm 1's "Initiative" is undefined → dropped.

**Game (Eq. 18–21)**
- P1 = fraction of the user's friends spreading the rumour (among friends in I or A).
- earn1 = P1 · Mult_eff,  earn2 = (1 − P1) · Mult_eff
- Drif1 = e^(earn1−earn2) / (1 + e^(earn1−earn2)) = sigmoid((2P1 − 1) · Mult_eff)
- Drif2 = sigmoid(earn2 − earn1) = 1 − Drif1

Note: the paper calls this an evolutionary/dynamic game; mathematically it is a sigmoid choice on the friends' majority.

Worked check: Mult_eff = 2, P1 = 0.8 → Drif1 = sigmoid(1.2) ≈ 0.77.

---

## §5 SAIR equations (paper Eq. 22–26, Algorithm 1)

**Assumptions:** closed population S + A + I + R = 1; spread by contact; message lifespan ~12 h.

**Basic SAIR (Eq. 22)** — constants α (correction contact), β (rumour contact), ε, γ (removal), α+β<1:
dS = −αAS − βIS;  dA = αAS − εA;  dI = βIS − γI;  dR = εA + γI

**Rates from pushes (Eq. 23–25)**
δ1(t) = Σ_k (k/m) C(m,k) Drif1^k (1 − Drif1)^(m−k) = Drif1  (mean of binomial / m).
**[FIX]** δ is population-level, Drif is per-user → δ1(t) = mean Drif1 over susceptible users, recomputed every step (Pop(t) and P1 change).

**Final model (Eq. 26) — CORRECTED [FIX: paper swaps δ1 and δ2]**
```
dS/dt = −δ1·I·S − δ2·A·S
dI/dt =  δ1·I·S − (1 − δ1)·I
dA/dt =  δ2·A·S − (1 − δ2)·A
dR/dt = (1 − δ1)·I + (1 − δ2)·A
```
Rumour grows only when δ1·S > 1 − δ1.

Worked Euler step (Δt = 1): S=0.90, I=0.05, A=0.05, R=0, δ1=0.6, δ2=0.4 → S=0.855, I=0.057, A=0.038, R=0.050 (sum 1.000).

**Algorithm 1 (paper):** PageRank layers + KNN groups → init S,A,I,R → per user: features, Prb/Gact/Gpre, regression, Drif1/Drif2 → mean-field ODE → output δ1(t), δ2(t), S,A,I,R. Complexity O(N).

---

## §6 Experiments in the paper (paper §V)

**Data:** Weibo ("2012 end of the world", "Zhou Kehua was not killed"), Twitter ("Germanwings plane crash", "Sydney Siege"). 1-hour bins; user inactive 24 h → R.

1. **Validation (Fig. 3–4):** S falls; I peaks first; A rises later and pushes I down; R saturates. Curves roughly match real data (plots only, no error metric — we add RMSE).
2. **vs SIRH baseline (Fig. 5):** SIRH rumour peak ≈ 0.44; SAIR rumour peak ≈ 0.20 (A) / 0.32 (B); correction peaks ~30 h and lasts longer.
3. **Self vs external factors (Fig. 6):** own traits dominate early, friends/groups dominate later.
4. **Ablation (Fig. 7):** no popularity → push never peaks; no emotion → much lower peak; no content fit / connectivity / friend power → lower, later peak, same shape. (Fig. 7's "Drif2…Drif6" are ablation variants, not the correction push.)
5. **Layers & groups (Fig. 8):** strong layer pushes harder than weak; push rises with group activity and preference.
6. **Rumour vs correction (Fig. 9):** rumour bursts first, correction overtakes and lasts longer.

**Limitations stated:** text only; static network snapshot; parameters fitted per dataset.
