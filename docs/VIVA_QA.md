# Viva preparation: 25 likely questions with short answers

Numbers are the ones in `docs/REPORT.md` (checked against `results/*.csv` by `src/report_check.py`). "D" numbers refer to `DECISIONS.md`.

---

**1. Why not simply reproduce the paper's model and figures?**
We did implement it faithfully first (with six necessary corrections, C1–C6). Implemented without information leakage, it fits worst of all
nine candidate models, and its claimed factor effects (Fig. 7) do not appear. Copying the figures would have meant reporting results the
data do not support. The project therefore keeps the paper's core idea (users follow their friends' majority) and fixes the structure.

**2. What exactly is wrong with Eq. 26?**
Two things. The paper attaches the rumour push δ1 to the correction and δ2 to the rumour (we swapped them back, C1). More fundamentally, even
the corrected Eq. 26 lets the rumour grow only if δ1 > 1/(1 + S) and the correction only if δ1 < S/(1 + S). Since S/(1 + S) ≤ 1/2 ≤ 1/(1 + S),
both can never grow at the same moment (proof in report Section 4.2), but in every data set both grow together in the first hours.

**3. How do you know the structural result matters in practice and is not just a theoretical curiosity?**
Because the fit shows it: Eq. 26 is the worst model in every set; on Charlie Hebdo it loses by 593.0 AIC to the main model. In the fitted
Eq. 26 the dynamics collapse to source pulses followed by decay (D49).

**4. Why fit cumulative entry curves rather than the number of active spreaders?**
Active (prevalence) curves need an assumption about when a user stops spreading, which the data do not record; a fixed window created
artificial cliffs (D19). The cumulative curves C_I(t), C_A(t) are directly observed (time of each user's first support or deny post) and
correspond exactly to the model's cumulative inflows (D14).

**5. Why PHEME and not a larger dataset?**
It is the only public dataset we found with human stance labels, who-follows-whom edges and retweets in the same threads (D1). Without
stance labels we would have to classify stance automatically, adding noise; without follow edges there is no network for P1 or targeting.

**6. Why exclude TRUE stories?**
SAIR is about malicious versus anti-malicious information. Someone supporting a true story is not a rumour spreader, so mixing true stories
in would mislabel the I and A sides (D20).

**7. What is "leakage" here and how did you prevent it?**
Using information created by the outcome to predict the outcome. Examples: retweeted text *is* the rumour, so it would make emotion
similarity trivially high; retweet edges point to source authors because of the spreading. We use only each user's own posts written
before they took a side plus their profile (D27), timestamped friend activity before entry (D29), follow edges only for PageRank (D32),
profile-based groups (D37), and we assert that no single feature reaches an AUC above 0.95 (report Table 2).

**8. Then how can friend power still reach an AUC of 0.763 on the Ottawa story?**
It reflects pre-entry social proof: most supporters follow the account that posted the rumour first, and its post had already collected
many retweets from other users before each supporter acted (D30). It uses only information available before the user's entry.

**9. Why don't you interpret κ, α or the popularity half-life?**
Parameter recovery and the bootstrap show they are not identifiable from cumulative curves: different values fit equally well (D60, D68). For
example, the bootstrap interval of κ on Charlie Hebdo reaches the upper bound of 50. Only the model comparisons, and conclusions that we
checked to be robust over κ (the tipping analysis), are reported. An extra recovery run (30 random starts per seed plus one start at the
true values) showed that β, α, γ, ε and θ are recovered when the optimiser reaches the optimum, while κ is not: it moved as far as the
bound of 50 at the same fit. Random starts reached the optimum in only 2 of 150 runs, so the earlier recovery failures were the optimiser.

**10. Why choose the main model by total AIC over three sets?**
The control and targeting experiments need one model for every event. A model that wins on one set but fails on another would bias them;
(d)-uniform, for example, is slightly better on Charlie Hebdo but 226 AIC worse on Sydney because it lacks the Bass term (D69). (d+)-uniform
has the best total, total AIC -4295.2.

**11. Isn't AIC overconfident with autocorrelated cumulative residuals?**
Yes. The effective sample size is far smaller than the number of points, so we treat differences below about 10 as ties (D51) and back the
contest with a held-out forecast, parameter recovery and a block bootstrap that respects short-range autocorrelation (D59).

**12. What is the Bass term and why add it?**
Bass (1969) splits adoption into external influence (proportional to the susceptible pool) and imitation (proportional to contacts). Users
meet rumours and corrections outside the observed posters (news, timelines). Without it, the correction cannot start before the first
denying source, and every model missed Sydney's early deniers; with it, (a+) beats (a) by 198.6 AIC on Sydney.

**13. What is source injection, and is it not cheating?**
At each source tweet's time, a share θ·w_s/Σw of the population moves to I or A, with w_s = log(1 + author followers). Source times and
follower counts are exogenous and known when the tweet is posted, so this is not leakage; it represents the broadcast reach of a source
(D46). The ablation shows it is the most important component (ΔAIC 305.4 on Charlie Hebdo).

**14. If the paper's per-user factors do not matter, what does?**
Source injection, external influence, and on Charlie Hebdo the friends' majority as an almost step-like "follow the sided-friends
majority" rule. Each of the paper's factors moves the rumour peak by about 1% or less, and several refits are better without them.

**15. Why might the paper find strong factor effects that you do not?**
We cannot check the paper's code, but leakage is a likely reason: if text features include retweeted text or features include outcome-made
edges, they look predictive. With leak-free features, only 2–4% of users wrote anything before taking a side, so the text signals are
weak (D31).

**16. What is SAIR-C and why event-relative knobs?**
SAIR-C adds blocking (I → B at rate u) and a correction campaign (S → A at rate v). The fitted rates differ by about two orders of magnitude
between sets, so a fixed absolute grid would be meaningless; u is scaled by each event's removal rate γ and v by its time to 90% of the
rumour's entries (D73).

**17. Is "hybrid is best" true?**
Not in general. At equal effort the three strategies tie at 2–5% of N. At 10%, a hybrid tips Charlie Hebdo (D/D0 0.016), but Sydney
prefers refutation and the Ottawa story blocking. The weighted cost J always picks a hybrid, but partly by construction, because J penalises
knob size and splitting effort over two knobs lowers a quadratic penalty (D75).

**18. Is the tipping result robust?**
It survives every κ from 5 to 50 with the other parameters refitted, so it is not decided by the non-identifiable κ. It disappears if users
without sided friends are given P1 = 0.5 instead of the global share I/(I + A); that model fits Charlie Hebdo 100.9 AIC worse (D76).

**19. Why compare strategies at equal effort?**
Otherwise a strategy can look better simply because it spends more. We solve "effort = budget" exactly for each strategy, including both
roots when effort is not monotone in the knob (D74).

**20. Why does blocking hubs work on PHEME?**
Because the hubs of the follow graph are the authors of the rumour's source tweets. Blocking them removes their broadcast (injection):
on Charlie Hebdo the injection channel alone gives 0.502 of the 0.532 reduction, the friend-count channel 0.001 (D78).

**21. Then does network position matter at all?**
Yes, for recruiting correctors once the injected mass is realistic: at a small injection, recruiting 50 central accounts on Charlie Hebdo
reduces exposure by 0.815 against 0.028 for random accounts. That effect is an interaction of the small correction mass with the shift of
the followers' friends' majority (D81).

**22. Why did the first recruiting results saturate?**
Our first specification let every recruit inject like a news source, so 50 recruits moved about a third of the population and any
selection, even random, ended the rumour. Scaling the injection (s = 0.1, 0.01) and measuring effort as the share of the population moved
fixed this (D81).

**23. What does the Barabási–Albert test add?**
A clean network-science check, independent of PHEME's per-thread follow graph: an agent-based model on scale-free graphs, calibrated to
pass explicit validation criteria. It confirms that hubs beat random targeting by a wide margin (blocking the top 2%: 90.5% vs 3.5%), and
shows that degree, PageRank and betweenness pick the same hubs there (D82).

**24. Why was the first agent model rejected, and is the second one trustworthy?**
The seeds-only model had no source injection, and the fitted correction rate made corrections take over the whole graph (final C_A 95.4% of
N), so it failed validation and is labelled qualitative only (D80). The injection model passes the stated criteria (final shares within 0.1,
rumour peaks first), but its time profile matches only moderately (shape RMSE 0.134) and two rates were rescaled, so it is a calibrated
analogue, not the fitted model.

**25. What would you do with more data?**
Obtain complete follower graphs and exposure (views) so that silent readers are counted and κ becomes identifiable; add more events and
platforms; study time-varying optimal control; and simulate the agent model directly on the real follow graph.
