"""Phase 9 - the numbers of docs/REPORT.md come from results/*.csv.

Every table in REPORT.md sits between <!-- table:NAME --> and <!-- /table:NAME --> markers and is generated here from the CSVs;
every number quoted in the prose is listed in CLAIMS with the CSV value it must equal.

Usage:
    python src/report_check.py --write     regenerate the table blocks inside docs/REPORT.md (prose untouched)
    python src/report_check.py             check: table blocks identical to a fresh generation, every claim found with its CSV value,
                                           and a coverage count of the numeric tokens in the prose
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
REPORT = os.path.join(ROOT, "docs", "REPORT.md")
SETS = ["charliehebdo", "sydneysiege", "story_three_shootings"]
SET_LABEL = {"charliehebdo": "Charlie Hebdo", "sydneysiege": "Sydney Siege", "story_three_shootings": "Ottawa story"}
MAIN = "sairh_plus_uniform"
LABEL = {"const": "(a) constant-rate SAIR", "const_plus": "(a+) constant-rate + Bass", "uniform": "(b) Eq. 26, uniform drive",
         "eq26": "(c) Eq. 26 (corrected)", "sairh": "(d) SAIR-H", "sairh_plus": "(d+) SAIR-H + Bass",
         "sairh_uniform": "(d) SAIR-H, uniform drive", "sairh_plus_uniform": "**(d+) SAIR-H+, uniform drive (main)**",
         "meanfield": "(e) SAIR-H mean-field"}
ORDER = ["const", "const_plus", "uniform", "eq26", "sairh", "sairh_plus", "sairh_uniform", "sairh_plus_uniform", "meanfield"]


def csv(name):
    return pd.read_csv(os.path.join(RES, f"{name}.csv"))


def f(x, d=3):
    """Fixed-point format with an ASCII minus (the format used everywhere in REPORT.md)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{x:.{d}f}"
    return "0." + "0" * d if s in ("-0." + "0" * d,) else s


def pct(x, d=1):
    return f(100 * x, d)


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ================================================================ tables

def t_data():
    s = csv("phase1_summary").set_index("set")
    rows = []
    for k in SETS:
        r = s.loc[k]
        rows.append([SET_LABEL[k], int(r["users_N"]), int(r["users_I_side"]), int(r["users_A_side"]), int(r["users_neutral_S"]),
                     int(r["threads_kept"]), int(r["bin_minutes"]), f(r["end_h"], 0), int(r["C_I_informative_bins"]), int(r["C_A_informative_bins"]),
                     "n/a" if np.isnan(r["edges_follow"]) else int(r["edges_follow"]), "n/a" if np.isnan(r["edges_interaction"]) else int(r["edges_interaction"])])
    return md(["Set", "N (posters)", "I users", "A users", "neutral (S)", "threads", "bin (min)", "window (h)", "informative bins C_I",
               "informative bins C_A", "follow edges", "interaction edges"], rows)


def t_leak():
    a = csv("phase2_auc")
    feats = ["S_emo", "Info_match", "connectivity", "friend_power", "has_text", "has_emotion", "has_friends"]
    rows = [[ft] + [f(a[(a["set"] == k) & (a["feature"] == ft)]["auc_any_direction"].iloc[0]) for k in SETS] for ft in feats]
    return md(["Feature (single-feature AUC, either direction)"] + [SET_LABEL[k] for k in SETS], rows)


def t_p1():
    cv = csv("phase4_cv").query("variant == 'profile'").set_index("set")
    p1 = csv("phase4_p1_within_thread").set_index("set")
    rows = [[SET_LABEL[k], f(cv.loc[k, "auc_cv_stratified"]), f(cv.loc[k, "auc_cv_thread_heldout"]), f(p1.loc[k, "auc_P1_overall"]),
             f(p1.loc[k, "auc_P1_within_thread_pooled"]), int(p1.loc[k, "threads_with_I_and_A"])] for k in SETS]
    return md(["Set", "regression CV AUC (stratified)", "regression CV AUC (thread held out)", "P1 AUC overall", "P1 AUC within threads",
               "threads with both sides"], rows)


def t_contest():
    c = csv("phase5_model_comparison")
    rows = []
    for m in ORDER:
        t = c[c["model"] == m].set_index("set")
        aic = [t.loc[k, "AIC"] for k in SETS]
        rows.append([LABEL[m], int(t["k"].iloc[0])] + [f(v, 1) for v in aic] + [f(sum(aic), 1)])
    return md(["Model", "k", "AIC Charlie Hebdo", "AIC Sydney", "AIC Ottawa story", "AIC total"], rows)


def t_params():
    p = csv("phase5_params")
    p = p[p["model"] == MAIN].set_index("set")
    names = ["beta", "alpha", "gamma", "eps", "kappa", "theta", "eta_I", "eta_A"]
    rows = [[SET_LABEL[k]] + [f(p.loc[k, n], 4) for n in names] + [str(p.loc[k, "bound_flags"]).replace(";", ", ") if isinstance(p.loc[k, "bound_flags"], str) else "none"]
            for k in SETS]
    return md(["Set", "β", "α", "γ", "ε", "κ", "θ", "η_I", "η_A", "within 1% of a bound"], rows)


def t_forecast():
    fc = csv("phase5_forecast")
    models = ["const", "const_plus", MAIN, "sairh_uniform"]
    rows = []
    for k in SETS:
        t = fc[fc["set"] == k].set_index("model")
        rows.append([SET_LABEL[k], f(t["cut_h"].iloc[0], 2)] + [f(t.loc[m, "fc_rmse_pooled"]) for m in models])
    return md(["Set", "cut (h)", "(a)", "(a+)", "main (d+)-uniform", "(d)-uniform"], rows)


def t_recovery():
    r = csv("phase5_recovery")
    rows = []
    for m in [MAIN, "sairh_uniform"]:
        t = r[r["model"] == m]
        for n, g in t.groupby("parameter", sort=False):
            rows.append([LABEL[m].replace("*", ""), n, f(g["true"].iloc[0], 4), f(g["recovered"].median(), 4), f(np.median(np.abs(g["pct_error"])), 1)
                         if g["true"].iloc[0] != 0 else "n/a"])
    return md(["Model", "parameter", "true", "median recovered", "median abs % error"], rows)


def t_recovery_mse():
    r = csv("phase5_recovery")
    t = r[r["model"] == MAIN].groupby("noise_seed").first()
    rows = [[int(s), f(t.loc[s, "fit_mse"] * 1e4, 2), int(t.loc[s, "starts_converged"])] for s in t.index]
    return md(["noise seed", "best fit MSE (×1e-4; noise floor ≈ 1)", "converged starts of 10"], rows)


def t_recovery_extra():
    path = os.path.join(RES, "phase5_recovery_extra.csv")
    if not os.path.exists(path):
        return "_The optional extra recovery run (30 starts × 5 seeds) did not finish; see Section 5.2._"
    r = pd.read_csv(path)
    g = r.groupby("noise_seed").first()
    rows = [[int(s), f(g.loc[s, "noise_floor_mse"] * 1e4, 2), f(g.loc[s, "best_random_mse"] * 1e4, 2), f(g.loc[s, "oracle_mse"] * 1e4, 2),
             f"{int(g.loc[s, 'random_starts_at_floor'])} / {int(g.loc[s, 'random_starts'])}"] for s in g.index]
    return md(["noise seed", "noise floor MSE (×1e-4)", "best of 30 random starts (×1e-4)", "start at the truth (×1e-4)", "random starts at the floor"], rows)


def t_recovery_extra_err():
    path = os.path.join(RES, "phase5_recovery_extra.csv")
    if not os.path.exists(path):
        return "_n/a_"
    r = pd.read_csv(path)
    rows = []
    for n, g in r.groupby("parameter", sort=False):
        if g["true"].iloc[0] == 0:
            continue
        rows.append([n, f(g["true"].iloc[0], 4), f(np.median(np.abs(g["pct_error_best_random"])), 1), f(np.median(np.abs(g["pct_error_oracle"])), 1)])
    return md(["parameter", "true", "median abs % error, best random start", "median abs % error, start at the truth"], rows)


def t_boot():
    b = csv("phase5_bootstrap_ci")
    b = b[b["model"] == MAIN]
    names = ["beta", "alpha", "gamma", "eps", "kappa", "theta", "eta_I"]
    rows = []
    for k in SETS:
        t = b[b["set"] == k].set_index("parameter")
        rows.append([SET_LABEL[k]] + [f"{f(t.loc[n, 'estimate'], 3)} [{f(t.loc[n, 'ci_low'], 3)}, {f(t.loc[n, 'ci_high'], 3)}]" for n in names])
    return md(["Set"] + ["β", "α", "γ", "ε", "κ", "θ", "η_I"], rows)


KO = {"no_network": "network P1 → global share", "no_injection": "source injection θ = 0", "no_external": "Bass external η = 0",
      "no_emotion": "emotion (S_emo = 1)", "no_popularity": "popularity (Pop = 1)", "no_content_fit": "content fit (Info_match = 1)",
      "no_connectivity": "connectivity (co = 1)", "no_friend_power": "friend power (fp = 1)"}


def t_ablation(model):
    imp, abl = csv("phase6_importance"), csv("phase6_ablation")
    kos = ["no_network", "no_injection", "no_external"] if model == MAIN else list(KO)
    rows = []
    for ko in kos:
        r = [KO[ko]]
        for k in SETS:
            d = imp[(imp["model"] == model) & (imp["set"] == k) & (imp["knockout"] == ko)]["dAIC"].iloc[0]
            r.append(f(d, 1))
        for k in SETS:
            p = abl[(abl["model"] == model) & (abl["set"] == k) & (abl["knockout"] == ko)]["d_peak_I_pct"].iloc[0]
            r.append(f(p, 1))
        rows.append(r)
    return md(["Knock-out", "refit ΔAIC CH", "refit ΔAIC SS", "refit ΔAIC story", "held-fixed peak Δ% CH", "held-fixed peak Δ% SS",
               "held-fixed peak Δ% story"], rows)


def t_fig8():
    t = csv("phase6_fig8_test")
    rows = []
    for m in ["sairh_plus", MAIN]:
        for k in SETS:
            r = t[(t["model"] == m) & (t["set"] == k) & (t["kappa_used"] == "fitted")].iloc[0]
            rows.append([LABEL[m].replace("*", ""), SET_LABEL[k], f(r["mean_strong_minus_weak"]), pct(r["share_bins_strong_higher"], 0),
                         f(r["rho_Gact_norm"], 2), f(r["p_Gact_norm"], 3)])
    return md(["Model", "Set", "Drif1 strong − weak", "% bins strong higher", "Spearman ρ(Drif1, Gact)", "p"], rows)


def t_fig6():
    t = csv("phase6_fig6_test")
    t = t[t["fitted"]]
    rows = [[r["window_h"], int(r["n_sided"]), int(r["n_A"]), f(r["own_share"], 2)] for _, r in t.iterrows()]
    return md(["Entry window (h)", "sided users", "A users", "own-trait share"], rows)


def t_strategies(model=MAIN):
    s = csv("phase7_strategies")
    s = s[(s["model"] == model) & (s["range"] == "grid")]
    rows = []
    for k in SETS:
        for b in [0.02, 0.05, 0.10]:
            t = s[(s["set"] == k) & (s["budget"] == b)].set_index("strategy")
            cells = []
            for st in ["block", "refute", "hybrid"]:
                if t.loc[st, "reachable"] in (True, "True"):
                    cells.append(f"{f(t.loc[st, 'D_over_D0'])} ({f(t.loc[st, 'lam_u'], 2)}, {f(t.loc[st, 'lam_v'], 2)})")
                else:
                    cells.append("unreachable")
            rows.append([SET_LABEL[k], pct(b, 0) + "%"] + cells)
    return md(["Set", "effort E", "block: D/D0 (λu, λv)", "refute: D/D0 (λu, λv)", "hybrid: D/D0 (λu, λv)"], rows)


def t_strategies_robust():
    s = csv("phase7_strategies")
    rows = []
    for model in ["const_plus", "sairh_uniform"]:
        t0 = s[(s["model"] == model) & (s["range"] == "grid") & (s["budget"] == 0.10)]
        for k in SETS:
            t = t0[t0["set"] == k].set_index("strategy")
            rows.append([LABEL[model], SET_LABEL[k]] + [f(t.loc[st, "D_over_D0"]) if t.loc[st, "reachable"] in (True, "True") else "unreachable"
                                                        for st in ["block", "refute", "hybrid"]])
    return md(["Model", "Set", "block D/D0 at E = 10%", "refute", "hybrid"], rows)


def t_cost():
    c = csv("phase7_cost")
    c = c[c["model"] == MAIN]
    rows = [[SET_LABEL[r["set"]], f(r["w"], 2), f(r["lam_u"], 2), f(r["lam_v"], 2), r["strategy"], f(r["J"]), f(r["D_over_D0"]), pct(r["E_total"])]
            for _, r in c.iterrows()]
    return md(["Set", "w", "λu", "λv", "strategy", "J", "D/D0", "E (% of N)"], rows)


def t_feedback():
    fb = csv("phase7_feedback")
    fb = fb[(fb["row_type"] == "best_point") & (fb["range"] == "grid")]
    rows = []
    for k in SETS:
        for b in [0.02, 0.05, 0.10]:
            for st in ["block", "refute", "hybrid"]:
                t = fb[(fb["set"] == k) & (fb["budget"] == b) & (fb["strategy"] == st)].set_index("frozen")
                if len(t) == 0:
                    continue
                rows.append([SET_LABEL[k], pct(b, 0) + "%", st, f(t.loc["rho1", "effect_feedback"]), pct(t.loc["rho1", "feedback_share"]),
                             pct(t.loc["dfix", "feedback_share"])])
    return md(["Set", "E", "strategy", "effect 1 − D/D0", "feedback share, ρ = 1 (%)", "feedback share, δ1(t) frozen (%)"], rows)


def t_tipping():
    fits, tip = csv("phase7_kappa_fits").set_index("variant"), csv("phase7_tipping")
    rows = []
    for v in ["main", "kappa5", "kappa12.5", "kappa25", "kappa50", "fb_half"]:
        t = tip[(tip["variant"] == v) & (tip["budget"] == 0.10)].set_index("strategy")
        h, r = t.loc["hybrid"], t.loc["refute"]
        rows.append([v, f(fits.loc[v, "kappa"], 2), f(fits.loc[v, "alpha"], 2), f(fits.loc[v, "dAIC_vs_main"], 1),
                     f(r["D_over_D0"]) if r["reachable"] else "unreachable", f(h["D_over_D0"]) if h["reachable"] else "unreachable",
                     f(h["C_A_final"], 2) if h["reachable"] else "n/a", "yes" if h["reachable"] and bool(h["tipped"]) else "no"])
    return md(["Variant (CH)", "κ", "α", "ΔAIC vs main", "refute D/D0 at 10%", "hybrid D/D0 at 10%", "hybrid final C_A", "tipped"], rows)


def t_pheme():
    p = csv("phase8_pheme")
    p = p[(p["tag"] == "main") & (p["k"] == 50) & (p["s"] == 1.0)]
    rows = []
    for k in SETS:
        for a in ["block", "recruit", "hybrid"]:
            t = p[(p["set"] == k) & (p["action"] == a)].set_index("method")
            rows.append([SET_LABEL[k], a] + [f(t.loc[m, "red_D"]) for m in ["degree", "pagerank", "betweenness"]] +
                        [f"{f(t.loc['random', 'red_D'])} ± {f(t.loc['random', 'red_D_ci95'])}"])
    return md(["Set", "action (k = 50, s = 1)", "degree", "PageRank", "betweenness", "random (mean ± 95% CI)"], rows)


def t_channels():
    c = csv("phase8_channels")
    c = c[(c["tag"] == "main") & (c["k"] == 50) & (c["s"] == 1.0) & (c["action"] == "block")]
    rows = [[SET_LABEL[k], m] + [f(c[(c["set"] == k) & (c["method"] == m)][col].iloc[0]) for col in ["red_friend_only", "red_inject_only", "red_both"]]
            for k in SETS for m in ["degree", "pagerank", "random"]]
    return md(["Set", "method (block, k = 50)", "friend counts only", "source injection only", "both"], rows)


def t_scale():
    p = csv("phase8_pheme")
    p = p[(p["tag"] == "main") & (p["k"] == 50) & (p["action"] == "recruit")]
    rows = []
    for k in SETS:
        for s in [1.0, 0.1, 0.01]:
            t = p[(p["set"] == k) & (p["s"] == s)].set_index("method")
            rows.append([SET_LABEL[k], f(s, 2), pct(t.loc["degree", "effort"], 2), f(t.loc["degree", "red_D"]), f(t.loc["random", "red_D"])])
    return md(["Set", "recruit scale s", "effort, degree (% of N)", "reduction, degree", "reduction, random"], rows)


def t_effort(tag="main", sets=SETS):
    e = csv("phase8_effort")
    e = e[(e["tag"] == tag)]
    rows = []
    for k in sets:
        for lev in [0.0025, 0.005, 0.01, 0.02, 0.05]:
            cells = []
            for a in ["block", "recruit", "hybrid"]:
                v = []
                for m in ["degree", "random"]:
                    r = e[(e["set"] == k) & (e["method"] == m) & (e["action"] == a) & (e["effort_level"] == lev)].iloc[0]
                    v.append(f(r["red_D"]) if r["reachable"] in (True, "True") else "-")
                cells.append(f"{v[0]} [{v[1]}]")
            rows.append([SET_LABEL[k], f"≤ {pct(lev, 2)}%"] + cells)
    return md(["Set", "effort", "block: degree [random]", "recruit: degree [random]", "hybrid: degree [random]"], rows)


def t_alpha():
    a = csv("phase8_alpha_fits").set_index("variant")
    e = csv("phase8_effort")
    rows = []
    for v in ["main", "alpha4.4", "alpha21.5"]:
        cells = []
        for lev in [0.005, 0.01]:
            for act in ["block", "recruit", "hybrid"]:
                r = e[(e["tag"] == v) & (e["set"] == "charliehebdo") & (e["method"] == "degree") & (e["action"] == act) & (e["effort_level"] == lev)].iloc[0]
                cells.append(f(r["red_D"]))
        rows.append([v, f(a.loc[v, "alpha"], 1), f(a.loc[v, "kappa"], 2), f(a.loc[v, "dAIC_vs_main"], 1)] + cells)
    return md(["CH fit", "α", "κ", "ΔAIC", "block ≤ 0.5%", "recruit ≤ 0.5%", "hybrid ≤ 0.5%", "block ≤ 1%", "recruit ≤ 1%", "hybrid ≤ 1%"], rows)


def t_ba_cal():
    c = csv("phase8_ba_calibration")
    rows = []
    for _, r in c.iterrows():
        rows.append([f(r["mult_beta"], 0), f(r["mult_alpha"], 2), f(r["rmse_shape"]), f(r["final_C_I"]), f(r["final_C_A"]),
                     f(r["peak_I_h"], 0), f(r["peak_A_h"], 0), "yes" if r["passes"] else "no", "**selected**" if r["selected"] else ""])
    return md(["β ×", "α ×", "shape RMSE", "final C_I", "final C_A", "I peak (h)", "A peak (h)", "passes", ""], rows)


def t_ba_val():
    v = csv("phase8_ba_validation")
    qs = ["peak I (fraction of N)", "time of peak I (h)", "time of peak A (h)", "final C_I", "final C_A",
          "time C_I reaches 50% of its final value (h)", "corr(C_I(t))", "corr(C_A(t))", "shape RMSE vs data (normalised C_I, C_A)"]
    rows = []
    for q in qs:
        a = v[(v["variant"] == "inject") & (v["quantity"] == q)].iloc[0]
        b = v[(v["variant"] == "plain") & (v["quantity"] == q)].iloc[0]
        rows.append([q, f(a["agent"]), f(b["agent"]), f(a["mean_field"]) if not np.isnan(a["mean_field"]) else "n/a"])
    return md(["Quantity", "agent, source injection (calibrated)", "agent, seeds only", "mean-field main model"], rows)


def t_ba():
    b = csv("phase8_ba")
    b = b[b["variant"] == "inject"]
    rows = []
    for km in ["fitted", "zero"]:
        for a in ["block", "recruit", "hybrid"]:
            for kp in [0.001, 0.005, 0.01, 0.02]:
                t = b[(b["kappa_mode"] == km) & (b["action"] == a) & (b["kpct"] == kp)].set_index("method")
                rows.append(["fitted κ" if km == "fitted" else "κ = 0", a, pct(kp, 1) + "%"] +
                            [f"{pct(t.loc[m, 'red_final'])} ± {pct(t.loc[m, 'red_final_ci95'])}" for m in ["degree", "pagerank", "betweenness", "random"]])
    return md(["κ", "action", "k (% of N)", "degree", "PageRank", "betweenness", "random"], rows)


TABLES = {"data": t_data, "leak": t_leak, "p1": t_p1, "contest": t_contest, "params": t_params, "forecast": t_forecast,
          "recovery": t_recovery, "recovery_mse": t_recovery_mse, "recovery_extra": t_recovery_extra, "recovery_extra_err": t_recovery_extra_err,
          "boot": t_boot, "ablation_main": lambda: t_ablation(MAIN), "ablation_paper": lambda: t_ablation("sairh_plus"), "fig8": t_fig8,
          "fig6": t_fig6, "strategies": t_strategies, "strategies_robust": t_strategies_robust, "cost": t_cost, "feedback": t_feedback,
          "tipping": t_tipping, "pheme": t_pheme, "channels": t_channels, "scale": t_scale, "effort": t_effort, "alpha": t_alpha,
          "ba_cal": t_ba_cal, "ba_val": t_ba_val, "ba": t_ba}


# ================================================================ prose claims: (text as written in REPORT.md, CSV value, format)

def v_model(set_, model, col):
    c = csv("phase5_model_comparison")
    return c[(c["set"] == set_) & (c["model"] == model)][col].iloc[0]


def v_total(model):
    c = csv("phase5_model_comparison")
    return c[c["model"] == model]["AIC"].sum()


def v_param(set_, name, model=MAIN):
    p = csv("phase5_params")
    return p[(p["set"] == set_) & (p["model"] == model)][name].iloc[0]


def v_imp(model, set_, ko):
    t = csv("phase6_importance")
    return t[(t["model"] == model) & (t["set"] == set_) & (t["knockout"] == ko)]["dAIC"].iloc[0]


def v_abl(model, set_, ko, col="d_peak_I_pct"):
    t = csv("phase6_ablation")
    return t[(t["model"] == model) & (t["set"] == set_) & (t["knockout"] == ko)][col].iloc[0]


def v_strat(set_, budget, strategy, col="D_over_D0", model=MAIN, rng="grid"):
    s = csv("phase7_strategies")
    return s[(s["model"] == model) & (s["range"] == rng) & (s["set"] == set_) & (s["budget"] == budget) & (s["strategy"] == strategy)][col].iloc[0]


def v_tip(variant, strategy, col="D_over_D0", budget=0.10):
    t = csv("phase7_tipping")
    return t[(t["variant"] == variant) & (t["budget"] == budget) & (t["strategy"] == strategy)][col].iloc[0]


def v_kfit(variant, col):
    return csv("phase7_kappa_fits").set_index("variant").loc[variant, col]


def v_pheme(set_, action, method, k=50, s=1.0, col="red_D", tag="main"):
    p = csv("phase8_pheme")
    return p[(p["tag"] == tag) & (p["set"] == set_) & (p["action"] == action) & (p["method"] == method) & (p["k"] == k) & (p["s"] == s)][col].iloc[0]


def v_ch(set_, action, method, col, k=50, s=1.0):
    c = csv("phase8_channels")
    return c[(c["tag"] == "main") & (c["set"] == set_) & (c["action"] == action) & (c["method"] == method) & (c["k"] == k) & (c["s"] == s)][col].iloc[0]


def v_eff(set_, action, method, lev, tag="main"):
    e = csv("phase8_effort")
    return e[(e["tag"] == tag) & (e["set"] == set_) & (e["action"] == action) & (e["method"] == method) & (e["effort_level"] == lev)]["red_D"].iloc[0]


def v_ba(action, method, kp, km="fitted", col="red_final"):
    b = csv("phase8_ba")
    return b[(b["variant"] == "inject") & (b["kappa_mode"] == km) & (b["action"] == action) & (b["method"] == method) & (b["kpct"] == kp)][col].iloc[0]


def v_val(q, var="inject", col="agent"):
    v = csv("phase8_ba_validation")
    return v[(v["variant"] == var) & (v["quantity"] == q)][col].iloc[0]


def v_summary(set_, col):
    return csv("phase1_summary").set_index("set").loc[set_, col]


def v_fb(set_, budget, strategy, frozen, col="feedback_share"):
    fb = csv("phase7_feedback")
    return fb[(fb["row_type"] == "best_point") & (fb["range"] == "grid") & (fb["set"] == set_) & (fb["budget"] == budget) &
              (fb["strategy"] == strategy) & (fb["frozen"] == frozen)][col].iloc[0]


def claims():
    """(literal text in REPORT.md, value from the CSVs) pairs; the literal must contain the CSV value in its stated format."""
    CH, SS, ST = SETS
    c = []
    add = lambda text, val, d: c.append((text.format(f(val, d)), val))
    addp = lambda text, val, d: c.append((text.format(pct(val, d)), val))
    # data
    add("N = {}", v_summary(CH, "users_N"), 0)
    add("{} I users and 157", v_summary(CH, "users_I_side"), 0)
    add("7389 I users and {} A users", v_summary(CH, "users_A_side"), 0)
    add("Sydney Siege (N = {}", v_summary(SS, "users_N"), 0)
    add("{} supporters and 266 deniers", v_summary(ST, "users_I_side"), 0)
    add("441 supporters and {} deniers", v_summary(ST, "users_A_side"), 0)
    # model contest
    add("total AIC {}", v_total(MAIN), 1)
    add("(d+) {} and (d)-uniform", v_total("sairh_plus"), 1)
    add("(d)-uniform {}", v_total("sairh_uniform"), 1)
    add("AIC {} on Charlie Hebdo", v_model(CH, "eq26", "AIC"), 1)
    add("Eq. 26 loses by {} AIC", v_model(CH, "eq26", "AIC") - v_model(CH, MAIN, "AIC"), 1)
    add("beats (a+) by {} AIC on Charlie Hebdo", v_model(CH, "const_plus", "AIC") - v_model(CH, MAIN, "AIC"), 1)
    add("and {} on Sydney", v_model(SS, "const_plus", "AIC") - v_model(SS, MAIN, "AIC"), 1)
    add("(a+) beats (a) by {} AIC on Sydney", v_model(SS, "const", "AIC") - v_model(SS, "const_plus", "AIC"), 1)
    add("and by {} on the story", v_model(ST, "const", "AIC") - v_model(ST, "const_plus", "AIC"), 1)
    add("network adds {} AIC on Charlie Hebdo", v_model(CH, "meanfield", "AIC") - v_model(CH, "sairh", "AIC"), 1)
    add("uniform drive beats the ranked one by {} AIC", v_model(CH, "sairh", "AIC") - v_model(CH, "sairh_uniform", "AIC"), 1)
    add("κ = {} on Charlie Hebdo", v_param(CH, "kappa"), 2)
    add("α = {} per hour", v_param(CH, "alpha"), 2)
    # ablation
    add("source injection (ΔAIC {} on Charlie Hebdo", v_imp(MAIN, CH, "no_injection"), 1)
    add("the Bass term on Sydney (ΔAIC {})", v_imp(MAIN, SS, "no_external"), 1)
    add("network P1 on Charlie Hebdo (ΔAIC {})", v_imp(MAIN, CH, "no_network"), 1)
    add("popularity improves the refit by {} AIC", -v_imp("sairh_plus", CH, "no_popularity"), 1)
    add("held-fixed rumour peak by {}%", v_abl(MAIN, CH, "no_injection"), 1)
    # control
    add("hybrid D/D0 = {} at 10% effort on Charlie Hebdo", v_strat(CH, 0.10, "hybrid"), 3)
    add("refute-only {} and block-only", v_strat(CH, 0.10, "refute"), 3)
    add("block-only {} at the same effort", v_strat(CH, 0.10, "block"), 3)
    add("Sydney prefers refute ({} vs hybrid", v_strat(SS, 0.10, "refute"), 3)
    add("vs hybrid {})", v_strat(SS, 0.10, "hybrid"), 3)
    add("the story prefers block ({} vs hybrid", v_strat(ST, 0.10, "block"), 3)
    add("(0.526 vs hybrid {})", v_strat(ST, 0.10, "hybrid"), 3)
    add("final C_A rises to {}", v_strat(CH, 0.10, "hybrid", "C_A_final"), 2)
    add("κ = 5 fit is worse by {} AIC", v_kfit("kappa5", "dAIC_vs_main"), 1)
    add("worse by {} AIC and no strategy tips", v_kfit("fb_half", "dAIC_vs_main"), 1)
    add("hybrid D/D0 {} at 10% with the 0.5 fallback", v_tip("fb_half", "hybrid"), 3)
    addp("{}% (refute) at 10% effort on Charlie Hebdo", v_fb(CH, 0.10, "refute", "dfix"), 1)
    # targeting PHEME
    add("block k = 50 reduces exposure by {} with degree", v_pheme(CH, "block", "degree"), 3)
    add("against {} for random", v_pheme(CH, "block", "random"), 3)
    add("friend-count channel alone gives {}", v_ch(CH, "block", "degree", "red_friend_only"), 3)
    add("injection channel alone {}", v_ch(CH, "block", "degree", "red_inject_only"), 3)
    add("recruit at s = 0.01: {} with degree", v_pheme(CH, "recruit", "degree", s=0.01), 3)
    add("vs {} at random", v_pheme(CH, "recruit", "random", s=0.01), 3)
    add("injection alone {}, friend counts alone", v_ch(CH, "recruit", "degree", "red_inject_only", s=0.01), 3)
    add("friend counts alone {}, both", v_ch(CH, "recruit", "degree", "red_friend_only", s=0.01), 3)
    add("both {} (an interaction)", v_ch(CH, "recruit", "degree", "red_both", s=0.01), 3)
    add("E ≤ 0.50%: block {}", v_eff(CH, "block", "degree", 0.005), 3)
    add("recruit {} and hybrid", v_eff(CH, "recruit", "degree", 0.005), 3)
    add("and hybrid {} (degree)", v_eff(CH, "hybrid", "degree", 0.005), 3)
    # BA
    add("final C_I {} (data 0.895)", v_val("final C_I"), 3)
    add("final C_A {} (data 0.019)", v_val("final C_A"), 3)
    add("shape RMSE {})", v_val("shape RMSE vs data (normalised C_I, C_A)"), 3)
    addp("block the top 2% by degree: {}%", v_ba("block", "degree", 0.02), 1)
    addp("random blocking {}%", v_ba("block", "random", 0.02), 1)
    addp("hybrid {}% vs", v_ba("hybrid", "degree", 0.02), 1)
    addp("vs {}% random", v_ba("hybrid", "random", 0.02), 1)
    addp("(seeds-only variant: final C_A {}%", v_val("final C_A", "plain"), 1)
    add("at most about {}% change", -v_abl("sairh_plus", SS, "no_emotion"), 1)
    rx = csv("phase5_recovery_extra")
    gx = rx.groupby("noise_seed").first()
    add("only {} of the 150 random starts", gx["random_starts_at_floor"].sum(), 0)
    add("of the {} random starts reach the floor", gx["random_starts"].sum(), 0)
    add("θ within {}%", rx[rx["parameter"].isin(["beta", "alpha", "gamma", "eps", "theta"])]["pct_error_oracle"].abs().max(), 1)
    add("κ moves\nas far as {} at the same fit", rx[rx["parameter"] == "kappa"]["oracle_start"].max(), 1)
    return c


# ================================================================ main

BLOCK = re.compile(r"(<!-- table:(\w+) -->\n)(.*?)(<!-- /table:\2 -->)", re.S)


def write():
    text = open(REPORT, encoding="utf-8").read()
    text = BLOCK.sub(lambda m: m.group(1) + TABLES[m.group(2)]() + "\n" + m.group(4), text)
    open(REPORT, "w", encoding="utf-8").write(text)
    print(f"tables written: {len(BLOCK.findall(text))}")


def check():
    text = open(REPORT, encoding="utf-8").read()
    bad = []
    blocks = BLOCK.findall(text)
    for _, name, body, _ in blocks:
        if body != TABLES[name]() + "\n":
            bad.append(f"table {name} differs from the CSVs")
    missing = [n for n in TABLES if f"<!-- table:{n} -->" not in text]
    found = 0
    for lit, val in claims():
        if lit in text:
            found += 1
        else:
            bad.append(f"claim not found (CSV value {val}): '{lit}'")
    prose = BLOCK.sub("", text)
    tokens = re.findall(r"(?<![\w.])-?\d+\.\d+", prose)                    # decimal numbers in the prose
    claimed = " ".join(lit for lit, _ in claims())
    covered = [t for t in tokens if t in claimed]
    print(f"tables: {len(blocks)} blocks regenerated from the CSVs, {len(blocks) - sum('table' in b for b in bad)} identical; "
          f"tables not used in the report: {missing or 'none'}")
    print(f"prose claims: {found} / {len(claims())} found with their CSV value")
    print(f"decimal numbers in the prose: {len(tokens)}, of which {len(covered)} are covered by a claim; uncovered: "
          + ", ".join(sorted(set(tokens) - set(covered))))
    for b in bad:
        print("FAIL", b)
    assert not bad, f"{len(bad)} report check failure(s)"
    print("Report number check passed.")


if __name__ == "__main__":
    write() if "--write" in sys.argv else check()
