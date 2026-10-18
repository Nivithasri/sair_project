"""Phase 9 - presentation/slides.pptx (python-pptx), built from the results CSVs (numbers via src/report_check.py helpers).

Usage:
    python src/slides.py            build presentation/slides.pptx
    python src/slides.py --render   build, render every slide to presentation/render/slide_NN.png with PowerPoint (COM), and check
                                    that no text overflows its box, nothing leaves the slide and no two content shapes overlap
"""
import json
import os
import subprocess
import sys

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

import report_check as rc

ROOT = rc.ROOT
RES = rc.RES
OUT = os.path.join(ROOT, "presentation", "slides.pptx")
RENDER = os.path.join(ROOT, "presentation", "render")
NAVY, GREY, ACCENT, LIGHT = RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x44, 0x44, 0x44), RGBColor(0x2F, 0x6D, 0xB5), RGBColor(0xE9, 0xEE, 0xF5)
W, H = Inches(13.333), Inches(7.5)
L, TOP, R = Inches(0.55), Inches(1.25), Inches(12.78)            # content area
f, pct, CH, SS, ST = rc.f, rc.pct, "charliehebdo", "sydneysiege", "story_three_shootings"
TEXT_SCALE, TABLE_SCALE = 1.25, 1.2                       # all body and table font sizes below are multiplied by these


def md_rows(md):
    """Rows of a markdown table produced by report_check (header first, separator dropped)."""
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    return [[c.strip().replace("**", "") for c in ln.strip("|").split("|")] for i, ln in enumerate(lines) if i != 1]


def new_slide(prs, title, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(L, Inches(0.35), R - L, Inches(0.7))
    tb.name = "title"
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb, p.font.name = Pt(28), True, NAVY, "Calibri"
    bar = s.shapes.add_shape(1, L, Inches(1.05), Inches(1.6), Emu(38000))
    bar.name = "deco_bar"
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    if subtitle:
        st = s.shapes.add_textbox(L + Inches(1.8), Inches(0.93), R - L - Inches(1.8), Inches(0.3))
        st.name = "subtitle"
        q = st.text_frame.paragraphs[0]
        q.text = subtitle
        q.font.size, q.font.italic, q.font.color.rgb, q.font.name = Pt(13), True, GREY, "Calibri"
    return s


def text(s, x, y, w, h, items, size=15, name="body", space=6):
    """Bulleted text box. items: strings (bullets) or (level, string) tuples; a string starting with '##' is a bold sub-heading."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    first = True
    for it in items:
        lvl, t = it if isinstance(it, tuple) else (0, it)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        head = t.startswith("##")
        p.text = (t[2:].strip() if head else ("• " if lvl == 0 else "– ") + t)
        p.level = 0
        p.font.size = Pt(round((size if lvl == 0 else size - 1) * TEXT_SCALE))
        p.font.name, p.font.color.rgb = "Calibri", NAVY if head else GREY
        p.font.bold = head
        p.space_after = Pt(space)
        if lvl:
            p.left_indent = Inches(0.3)
    return tb


def table(s, x, y, w, rows, size=11, col_w=None, name="table", bold_rows=()):
    nr, nc = len(rows), len(rows[0])
    shape = s.shapes.add_table(nr, nc, x, y, w, Inches(0.3) * nr)
    shape.name = name
    t = shape.table
    widths = col_w or [1] * nc
    tot = sum(widths)
    for j, cw in enumerate(widths):
        t.columns[j].width = int(w * cw / tot)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(0.28 * TABLE_SCALE)
        for j, v in enumerate(r):
            cell = t.cell(i, j)
            cell.text = str(v)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            para = cell.text_frame.paragraphs[0]
            para.font.size, para.font.name = Pt(round(size * TABLE_SCALE)), "Calibri"
            para.font.bold = i == 0 or i in bold_rows
            para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else NAVY
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (LIGHT if i in bold_rows else (RGBColor(0xFF, 0xFF, 0xFF) if i % 2 else RGBColor(0xF6, 0xF7, 0xF9)))
    return shape


def picture(s, path, x, y, w, h, name="figure", caption=None):
    """Picture fitted into the box (x, y, w, h), aspect kept; optional caption below it inside the box."""
    cap_h = Inches(0.3) if caption else 0
    iw, ih = Image.open(path).size
    scale = min(w / iw, (h - cap_h) / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    pic = s.shapes.add_picture(path, x + (w - pw) // 2, y, pw, ph)
    pic.name = name
    if caption:
        tb = s.shapes.add_textbox(x, y + ph + Inches(0.02), w, cap_h)
        tb.name = name + "_caption"
        p = tb.text_frame.paragraphs[0]
        p.text = caption
        p.font.size, p.font.italic, p.font.color.rgb, p.font.name = Pt(11), True, GREY, "Calibri"
        p.alignment = PP_ALIGN.CENTER
        tb.text_frame.word_wrap = True
    return pic


def res(name):
    return os.path.join(RES, name)


# ================================================================ slides

def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = text(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(1.4), ["##Whom to Silence, Whom to Amplify"], size=40, name="title")
    tb.text_frame.paragraphs[0].font.color.rgb = NAVY
    text(s, Inches(0.9), Inches(2.75), Inches(11.5), Inches(1.0),
         ["##Centrality-Targeted Hybrid Control of Rumour–Correction Competition on Social Networks"], size=22, name="subtitle")
    bar = s.shapes.add_shape(1, Inches(0.9), Inches(4.05), Inches(2.2), Emu(45000))
    bar.name = "deco_bar"
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    text(s, Inches(0.9), Inches(4.35), Inches(11.5), Inches(2.4),
         ["##A leakage-controlled re-implementation and extension of the SAIR model (Li et al., IEEE Transactions on Big Data, 2026), tested "
          "on the PHEME Twitter rumour dataset: Charlie Hebdo, Sydney Siege and one Ottawa story",
          "##Nivithasri A  ·  Project report and presentation  ·  October 2026"], size=16, name="info", space=14)

    # 2 problem
    s = new_slide(prs, "The problem: rumours and corrections compete", "Two levers, three questions")
    text(s, L, TOP, Inches(6.0), Inches(5.9), [
        "##What happens in a crisis",
        "False or unverified claims spread within minutes, and corrections follow while the rumour is still growing.",
        "Both messages draw from the same pool of users who have not yet taken a side, so they compete for attention.",
        "##What a platform can do",
        "Silence: block rumour accounts or remove their posts (moves spreaders out of the conversation).",
        "Amplify: run a fact-check campaign or recruit credible accounts to broadcast the correction.",
        "Either lever can be aimed at all users or at selected, well-connected accounts.",
    ], size=15)
    text(s, Inches(6.9), TOP, Inches(5.9), Inches(5.9), [
        "##Questions of this project",
        "1. Does the published SAIR model fit real rumour data when it is implemented faithfully and without information leakage, and do its "
        "per-user factors (emotion, popularity, content fit, connectivity, friend power) matter?",
        "2. Given a model that fits, how should blocking and a correction campaign be combined, and at what effort?",
        "3. Whom should the intervention target: hubs by degree, PageRank, betweenness, or random accounts?",
        "##Approach",
        "Re-implementation → honest model contest with validation → ablation → control model SAIR-C → targeting on the real follow graph "
        "and on synthetic scale-free networks.",
    ], size=15)

    # 3 SAIR paper
    s = new_slide(prs, "The base paper: SAIR (Li et al., 2026)", "Malicious vs anti-malicious information on a layered network")
    text(s, L, TOP, Inches(6.2), Inches(5.9), [
        "##Model",
        "Four states: S not yet sided, I spreads the rumour, A spreads the correction, R stopped.",
        "Each user's push toward the rumour: Drif1 = sigmoid((2·P1 − 1)·Mult_eff), with P1 = share of the user's sided friends who support "
        "the rumour. Drif2 = 1 − Drif1.",
        "Mult_eff comes from a regression on emotion similarity, popularity, content fit, influence of higher / lower network layers and "
        "group attraction.",
        "Population drives δ1 = mean Drif1 over susceptible users, δ2 = 1 − δ1, which enter the rate equations (Eq. 26).",
        "##Claims we test",
        "Good fit to Twitter data; every push factor matters (Fig. 7); the strong layer and active groups push harder (Fig. 8); own traits "
        "matter early, external factors later (Fig. 6).",
    ], size=14)
    rows = [["#", "Problem in the paper", "Our fix"], ["C1", "Eq. 26 attaches δ1 to A and δ2 to I", "δ1 drives I, δ2 drives A"],
            ["C2", "Eq. 24 binomial looks like a new quantity", "its mean is Drif, so δ = mean Drif"],
            ["C3", "'Initiative' never defined", "dropped"], ["C4", "regression target unspecified", "first side is I (sided users)"],
            ["C5", "similarity + raw mutual-friend count", "Jaccard of friend lists"], ["C6", "γ used twice", "group weight renamed w_part"]]
    table(s, Inches(7.0), TOP + Inches(0.1), Inches(5.78), rows, size=12, col_w=[0.5, 3, 2.6])
    text(s, Inches(7.0), Inches(4.85), Inches(5.78), Inches(1.2), ["Six corrections were needed before the model could be run at all; all are logged in DECISIONS.md."], size=13)

    # 4 data
    s = new_slide(prs, "Data: PHEME rumour threads", "The only public set with stance labels, follow edges and retweets together")
    d = md_rows(rc.t_data())
    rows = [[r[0], r[1], r[2], r[3], r[4], r[6], r[7]] for r in d]
    rows[0] = ["Set", "N (posters)", "I users", "A users", "neutral", "bin (min)", "window (h)"]
    table(s, L, TOP + Inches(0.05), Inches(6.6), rows, size=13, col_w=[2.2, 1.4, 1.1, 1.1, 1.1, 1.1, 1.3])
    text(s, L, Inches(3.0), Inches(6.6), Inches(4.2), [
        "Only FALSE and UNVERIFIED stories are kept: supporters of true news are not rumour spreaders.",
        "A user's side is fixed by their first support or deny post; users who only ask or comment stay susceptible.",
        "Fitting target: cumulative entry curves C_I(t), C_A(t), trimmed at 99% + 12 h.",
        "N counts posting users only, so shapes and timing are compared, not peak heights.",
        "Follow edges exist only among each thread's participants (a limitation for every network quantity).",
    ], size=14)
    leak = rc.csv("phase2_auc")
    mx = leak["auc_any_direction"].max()
    who = leak.loc[leak["auc_any_direction"].idxmax()]
    text(s, Inches(7.45), TOP, Inches(5.33), Inches(5.9), [
        "##Leakage controls",
        "User features use only the user's own posts written before taking a side, plus the profile; retweeted text is never used.",
        "Friend power counts only timestamped replies and retweets received before the user's entry.",
        "PageRank on follow edges only; groups from profile features, not from shared retweets.",
        f"Tripwire: every single feature must have AUC ≤ 0.95 for the side taken. Highest observed: {f(mx)} "
        f"({who['feature'].replace('_', ' ')}, {rc.SET_LABEL[who['set']]}), explained by pre-entry social proof.",
        "Within threads that contain both sides, the friends' majority P1 predicts the side at chance level: its overall separation is thread "
        "identity.",
    ], size=14)

    # 5 structural flaw
    s = new_slide(prs, "The paper's flaw: Eq. 26 cannot let both sides grow", "A structural result (proof in the report, Section 4.2)")
    text(s, L, TOP, Inches(6.4), Inches(5.9), [
        "##Corrected Eq. 26 with δ2 = 1 − δ1",
        "dI/dt = I·(δ1·S − (1 − δ1))  >  0   ⇔   δ1 > 1 / (1 + S)",
        "dA/dt = A·((1 − δ1)·S − δ1)  >  0   ⇔   δ1 < S / (1 + S)",
        "For S in [0, 1]:  S/(1 + S) ≤ 1/2 ≤ 1/(1 + S), so both conditions can never hold together.",
        "##Consequence",
        "At any moment at most one of rumour and correction can grow. In the data both grow together in the first hours of every set, and "
        "the paper itself describes this 'cooperative' phase.",
        "The model reduces to source pulses followed by decay, and fits worst of all nine candidates.",
    ], size=15)
    c = rc.csv("phase5_model_comparison")
    rows = [["Set", "AIC Eq. 26 (c)", "AIC main model", "Eq. 26 worse by"]]
    for k in rc.SETS:
        a, b = c[(c["set"] == k) & (c["model"] == "eq26")]["AIC"].iloc[0], c[(c["set"] == k) & (c["model"] == rc.MAIN)]["AIC"].iloc[0]
        rows.append([rc.SET_LABEL[k], f(a, 1), f(b, 1), f(a - b, 1)])
    table(s, Inches(7.25), TOP + Inches(0.1), Inches(5.5), rows, size=13, col_w=[2, 1.6, 1.6, 1.6])
    text(s, Inches(7.25), Inches(3.0), Inches(5.5), Inches(3.5), [
        "Lower AIC is better; a difference above about 10 is decisive here.",
        "This is why we keep the paper's mechanism (the friends'-majority drive) but change the equations.",
    ], size=14)

    # 6 SAIR-H+
    s = new_slide(prs, "Our model: SAIR-H+", "SAIR rates driven by the friends' majority, plus Bass external influence and source injection")
    text(s, L, TOP, Inches(6.5), Inches(5.9), [
        "##Equations",
        "dS = −β·δ1·I·S − α·δ2·A·S − (η_I + η_A)·S",
        "dI = β·δ1·I·S + η_I·S − γ·I",
        "dA = α·δ2·A·S + η_A·S − ε·A",
        "dR = γ·I + ε·A",
        "I grows iff β·δ1·S > γ and A grows iff α·δ2·S > ε: independent conditions, so co-growth is possible.",
        "##Mechanisms",
        "Friends'-majority drive δ1 from data friend counts, re-weighted by the model's own curves, so interventions feed back.",
        "Bass (1969) external influence η: users meet rumours and corrections outside the observed population.",
        "Source injection θ: each source tweet moves θ·w_s/Σw of the population into I or A, w_s = log(1 + author followers).",
    ], size=14)
    text(s, Inches(7.3), TOP, Inches(5.45), Inches(5.9), [
        "##Nine models compared",
        "(a) constant rates, (a+) with Bass; (b), (c) Eq. 26; (d) SAIR-H, (d+) with Bass; (d)-uniform and (d+)-uniform with the same "
        "push κ for every user; (e) mean-field.",
        "##Fitting protocol",
        "Least squares on normalised cumulative curves; AIC per set and in total.",
        "Bounded Nelder–Mead in log space, rates in [1e-3, 50] per hour, 20–25 random starts plus cross-model starts.",
        "Euler steps in a numba kernel, bitwise identical to the Python loop; every run checkpointed.",
        "##Main model",
        "(d+)-uniform: best total AIC over the three sets, so one model serves every event in the control experiments.",
    ], size=14)

    # 7 model contest
    s = new_slide(prs, "Model contest", "AIC per set and in total (lower is better)")
    rows = md_rows(rc.t_contest())
    rows = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows]
    table(s, L, TOP + Inches(0.05), Inches(7.6), rows, size=12, col_w=[3.4, 0.5, 1.3, 1.2, 1.3, 1.2], bold_rows=(8,))
    text(s, Inches(8.45), TOP, Inches(4.33), Inches(5.9), [
        f"Main model total AIC {f(rc.v_total(rc.MAIN), 1)}; next best (d+) {f(rc.v_total('sairh_plus'), 1)}.",
        f"The drive adds information: main beats (a+) by {f(rc.v_model(CH, 'const_plus', 'AIC') - rc.v_model(CH, rc.MAIN, 'AIC'), 1)} AIC on Charlie "
        f"Hebdo and {f(rc.v_model(SS, 'const_plus', 'AIC') - rc.v_model(SS, rc.MAIN, 'AIC'), 1)} on Sydney; on the story κ fits to 0.",
        f"Bass is decisive where corrections start before any denying source: (a+) beats (a) by {f(rc.v_model(SS, 'const', 'AIC') - rc.v_model(SS, 'const_plus', 'AIC'), 1)} "
        f"on Sydney.",
        f"The paper's regression ranking adds nothing: the uniform drive beats the ranked one by {f(rc.v_model(CH, 'sairh', 'AIC') - rc.v_model(CH, 'sairh_uniform', 'AIC'), 1)} "
        f"AIC on Charlie Hebdo.",
        "κ, w and α are not identifiable from cumulative curves; their values are never interpreted.",
    ], size=14)

    # 8 validation
    s = new_slide(prs, "Validation: fits, forecasts, recovery", "How far the main model can be trusted")
    picture(s, res("phase5_plot2_cumulative_fit.png"), Inches(6.55), TOP, Inches(6.2), Inches(5.85), caption="Fitted cumulative curves of all models")
    rows = md_rows(rc.t_forecast())
    rows[0] = ["Set", "cut (h)", "(a)", "(a+)", "main", "(d)-unif."]
    table(s, L, TOP + Inches(0.05), Inches(5.8), rows, size=12, col_w=[1.9, 1, 0.9, 0.9, 0.9, 1])
    rec = rc.csv("phase5_recovery")
    floor = (rec[rec["model"] == rc.MAIN].groupby("noise_seed")["fit_mse"].first() <= 1.1e-4).sum()
    boot = rc.csv("phase5_bootstrap_ci")
    kci = boot[(boot["model"] == rc.MAIN) & (boot["set"] == CH) & (boot["parameter"] == "kappa")].iloc[0]
    text(s, L, Inches(2.75), Inches(5.8), Inches(4.4), [
        "Table: held-out pooled RMSE after the cut. The main model forecasts as well as or better than constant rates; no model can forecast "
        "a later wave before its source tweets appear.",
        f"Parameter recovery (Charlie Hebdo, 5 noise seeds × 10 starts): only {floor} of 5 seeds reached the noise floor, so the 8-parameter fit "
        "needs many starts.",
        "Extra starts (30 per seed + one at the truth): from the truth every seed reaches the noise floor and β, α, γ, ε, θ are recovered; "
        "κ is not identifiable. Random starts reach the floor in 2 of 150 runs: the earlier failures were the optimiser.",
        f"Moving-block bootstrap: κ on Charlie Hebdo in [{f(kci['ci_low'], 1)}, {f(kci['ci_high'], 1)}], reaching the bound of 50.",
    ], size=13)

    # 9 ablation
    s = new_slide(prs, "What drives the spreading: ablation", "Refit ΔAIC > 0 means the component is needed")
    rows = md_rows(rc.t_ablation(rc.MAIN))
    rows = [[r[0], r[1], r[2], r[3], r[4]] for r in rows]
    rows[0] = ["Main model: knock-out", "ΔAIC CH", "ΔAIC SS", "ΔAIC story", "peak Δ% CH"]
    table(s, L, TOP + Inches(0.05), Inches(6.6), rows, size=12, col_w=[2.8, 1, 1, 1.1, 1.2])
    rows = md_rows(rc.t_ablation("sairh_plus"))
    rows = [[r[0], r[1], r[2], r[4]] for r in rows if r[0] in ("Knock-out", "emotion (S_emo = 1)", "popularity (Pop = 1)", "content fit (Info_match = 1)",
                                                                 "connectivity (co = 1)", "friend power (fp = 1)")]
    rows[0] = ["Paper-faithful model: factor", "ΔAIC CH", "ΔAIC SS", "peak Δ% CH"]
    table(s, L, Inches(3.15), Inches(6.6), rows, size=12, col_w=[2.8, 1, 1, 1.2])
    text(s, Inches(7.5), TOP, Inches(5.28), Inches(5.9), [
        f"Source injection is essential everywhere (ΔAIC {f(rc.v_imp(rc.MAIN, CH, 'no_injection'), 1)} on Charlie Hebdo; without it the rumour peak "
        f"falls by {f(-rc.v_abl(rc.MAIN, CH, 'no_injection'), 1)}%).",
        f"Bass external influence is decisive on Sydney (ΔAIC {f(rc.v_imp(rc.MAIN, SS, 'no_external'), 1)}) and the story.",
        f"The friends'-majority network matters on Charlie Hebdo (ΔAIC {f(rc.v_imp(rc.MAIN, CH, 'no_network'), 1)}).",
        "None of the paper's per-user factors matters: each moves the rumour peak by about 1% or less, and several refits are better without "
        f"them (popularity: {f(rc.v_imp('sairh_plus', CH, 'no_popularity'), 1)} AIC on Charlie Hebdo).",
        "Reason: almost all users enter by retweet, so only profile text exists before entry; the leak-free features are weak.",
    ], size=14)

    # 10 claims
    s = new_slide(prs, "The paper's claims against our results", "Same model family, leak-free features, real data")
    rows = [["Paper claim", "Our result", "Verdict"],
            ["Eq. 26 fits the data well", "Worst of nine models in every set; cannot produce co-growth", "Does not match"],
            ["Fig. 7: no popularity → push never peaks", "Peak changes by under 1%; refit better without popularity", "Does not match"],
            ["Fig. 7: no emotion → much lower peak", "At most about 0.4% change", "Does not match"],
            ["Fig. 7: no content fit / connectivity / friend power → lower, later peak", "About 1% or less, partly the opposite sign", "Does not match"],
            ["Fig. 8: the strong layer pushes harder", "Yes on Charlie Hebdo, not on Sydney", "Partly"],
            ["Fig. 8: push rises with group activity", "Sign follows each set's regression coefficient", "Partly / does not match"],
            ["Fig. 6: own traits early, external later", "Only two usable windows; own share falls between them", "Partly at most"]]
    table(s, L, TOP + Inches(0.05), R - L, rows, size=14, col_w=[4.3, 5.3, 2.4])
    text(s, L, Inches(5.25), R - L, Inches(1.9), [
        "What the data do support: SAIR-type contact and removal rates, source injection, external influence, and the friends' majority as a "
        "near step-like 'follow the majority' rule on Charlie Hebdo.",
    ], size=15)

    # 11 control
    s = new_slide(prs, "Control: SAIR-C", "Blocking (u: I → B) and a correction campaign (v: S → A), compared at equal effort")
    rows = md_rows(rc.t_strategies())
    rows = [[r[0].replace("Charlie Hebdo", "CH").replace("Sydney Siege", "Sydney").replace("Ottawa story", "Story"), r[1],
             r[2].split(" ")[0], r[3].split(" ")[0], r[4].split(" ")[0]] for r in rows]
    rows[0] = ["Set", "effort", "block D/D0", "refute D/D0", "hybrid D/D0"]
    table(s, L, TOP + Inches(0.05), Inches(6.0), rows, size=12, col_w=[1.1, 0.9, 1.3, 1.3, 1.3])
    text(s, Inches(6.85), TOP, Inches(5.93), Inches(5.9), [
        "##Set-up",
        "Knobs scaled to each event: u = λu·γ, v = λv/T90 (the fitted rates differ by about 300× between sets).",
        "Damage D = ∫I dt relative to no control (D/D0); effort = fraction of N blocked plus fraction reached by the campaign.",
        "Each strategy's best point at exactly equal total effort (2%, 5%, 10% of N).",
        "##Result",
        "At 2–5% effort the strategies tie within a few points.",
        f"At 10%: the hybrid tips Charlie Hebdo (D/D0 {f(rc.v_strat(CH, 0.10, 'hybrid'))}, final C_A {f(rc.v_strat(CH, 0.10, 'hybrid', 'C_A_final'), 2)}), "
        f"refutation wins on Sydney ({f(rc.v_strat(SS, 0.10, 'refute'))}) and blocking on the story ({f(rc.v_strat(ST, 0.10, 'block'))}).",
        "'Hybrid is best' is not a general result; a weighted cost J favours hybrids partly by construction.",
    ], size=14)

    # 12 tipping
    s = new_slide(prs, "Is the tipping robust?", "Charlie Hebdo at 10% effort under refits")
    picture(s, res("phase7_plot5b_pareto.png"), L, TOP, R - L, Inches(3.35), caption="Pareto front: exposure D/D0 against total effort")
    rows = md_rows(rc.t_tipping())
    rows = [[r[0], r[1], r[3], r[5], r[6], r[7]] for r in rows]
    rows[0] = ["CH variant", "κ", "ΔAIC", "hybrid D/D0", "final C_A", "tipped"]
    table(s, L, Inches(4.7), Inches(6.4), rows, size=11, col_w=[1.6, 0.8, 0.9, 1.2, 1.0, 0.8])
    text(s, Inches(7.25), Inches(4.65), Inches(5.53), Inches(2.75), [
        "κ is not identifiable, so it was fixed at 5–50 and the rest refitted: the tipping survives every κ.",
        f"With the fallback P1 = 0.5 for users without sided friends the fit is worse by {f(rc.v_kfit('fb_half', 'dAIC_vs_main'), 1)} AIC and nothing tips.",
        f"Freezing δ1(t) removes up to {pct(rc.v_fb(CH, 0.10, 'refute', 'dfix'), 1)}% of the refute effect: the feedback runs through that fallback.",
    ], size=14)

    # 13 targeting PHEME
    s = new_slide(prs, "Targeting on PHEME: who to block", "k = 50 accounts chosen on the follow graph")
    rows = md_rows(rc.t_pheme())
    rows = [[r[0].replace("Charlie Hebdo", "CH").replace("Sydney Siege", "Sydney").replace("Ottawa story", "Story"), r[1].split(" ")[0], r[2], r[3], r[4], r[5]] for r in rows]
    rows[0] = ["Set", "action", "degree", "PageRank", "betweenness", "random ± CI"]
    table(s, L, TOP + Inches(0.05), Inches(7.0), rows, size=12, col_w=[1, 1, 1, 1, 1.2, 1.5])
    text(s, Inches(7.85), TOP, Inches(4.93), Inches(5.9), [
        "Reduction of rumour exposure 1 − D/D0 (recruit scale s = 1).",
        "Block = accounts that would spread the rumour: they stop counting as rumour friends, and their source tweets lose their injection.",
        f"Charlie Hebdo, block by degree: {f(rc.v_pheme(CH, 'block', 'degree'))}; friend-count channel alone "
        f"{f(rc.v_ch(CH, 'block', 'degree', 'red_friend_only'))}, injection channel alone {f(rc.v_ch(CH, 'block', 'degree', 'red_inject_only'))}.",
        "So hubs win because they are the rumour's source authors, not because of conformity.",
        "At s = 1 every recruit injects like a news source and even random recruiting saturates: the next slide fixes this.",
    ], size=14)

    # 14 effort-fair
    s = new_slide(prs, "Targeting on PHEME: equal effort", "Effort = % of N moved (blocked accounts + injected correction mass)")
    picture(s, res("phase8_plot9_effort.png"), L, TOP, R - L, Inches(3.25), caption="Reduction against effort; solid = degree, dashed = random")
    rows = md_rows(rc.t_effort(sets=[CH]))
    rows = [[r[1], r[2], r[3], r[4]] for r in rows]
    rows[0] = ["CH effort", "block: degree [random]", "recruit: degree [random]", "hybrid: degree [random]"]
    table(s, L, Inches(4.6), Inches(7.0), rows, size=11, col_w=[1.1, 1.9, 1.9, 1.9])
    text(s, Inches(7.85), Inches(4.55), Inches(4.93), Inches(2.85), [
        f"With a small injection (s = 0.01) centrality matters: recruit {f(rc.v_pheme(CH, 'recruit', 'degree', s=0.01))} by degree vs "
        f"{f(rc.v_pheme(CH, 'recruit', 'random', s=0.01))} at random.",
        f"An interaction: injection alone {f(rc.v_ch(CH, 'recruit', 'degree', 'red_inject_only', s=0.01))}, friend counts alone "
        f"{f(rc.v_ch(CH, 'recruit', 'degree', 'red_friend_only', s=0.01))}, both {f(rc.v_ch(CH, 'recruit', 'degree', 'red_both', s=0.01))}.",
        "Recruit beats block at equal effort; hybrid is best from about 0.5–1% of N; same order at both ends of α's CI.",
    ], size=13)

    # 15 BA
    s = new_slide(prs, "Targeting on Barabási–Albert networks", "Agent-based SAIR-H+, 10 graphs × 20 runs, source injection on hubs")
    picture(s, res("phase8_plot8_ba_reduction.png"), L, TOP, R - L, Inches(3.2), caption="Final-size reduction against k; fitted κ (solid) and κ = 0 (dashed)")
    b = rc.csv("phase8_ba")
    b = b[(b["variant"] == "inject") & (b["kappa_mode"] == "fitted")]
    rows = [["action, k", "degree", "PageRank", "betweenness", "random"]]
    for a in ["block", "recruit", "hybrid"]:
        for kp in [0.005, 0.02]:
            t = b[(b["action"] == a) & (b["kpct"] == kp)].set_index("method")
            rows.append([f"{a}, {pct(kp, 1)}%"] + [pct(t.loc[m, "red_final"], 1) + "%" for m in ["degree", "pagerank", "betweenness", "random"]])
    table(s, L, Inches(4.6), Inches(6.4), rows, size=11, col_w=[1.6, 1, 1, 1.1, 1])
    text(s, Inches(7.25), Inches(4.55), Inches(5.53), Inches(2.85), [
        f"Calibrated (β×2, α×0.1) and validated: final C_I {f(rc.v_val('final C_I'))} vs 0.895 in the data, final C_A {f(rc.v_val('final C_A'))} vs 0.019, "
        f"rumour peaks first; shape RMSE {f(rc.v_val('shape RMSE vs data (normalised C_I, C_A)'))} (moderate).",
        "Hubs beat random by a wide margin; the three centralities pick the same hubs.",
        "Blocking works because hubs carry the source injection; conformity matters more for recruiting.",
    ], size=13)

    # 16 limitations
    s = new_slide(prs, "Limitations", "What these results can and cannot support")
    text(s, L, TOP, Inches(6.0), Inches(5.9), [
        "Posters-only population: silent readers are not observed, so peak heights are not comparable with the paper and '% of N' means % of "
        "the modelled population.",
        "The follow graph is collected per thread, so network quantities partly reflect participation; P1's separation is thread identity.",
        "Text features are weak: almost nobody writes before taking a side, so emotion and content fit rest on profile descriptions.",
        "κ (and w) are not identifiable from cumulative curves; random starts rarely reach the optimum of the 8-parameter fit.",
    ], size=15)
    text(s, Inches(6.9), TOP, Inches(5.88), Inches(5.9), [
        "The Charlie Hebdo tipping depends on the global-share fallback for users without sided friends.",
        "The BA agent model is a calibrated analogue: two rates were rescaled to pass validation and the time profile matches only moderately.",
        "Control knobs are constant and act from t = 0; time-varying optimal control is not studied.",
        "Two events and one story, one platform; in both events the corrections come mostly from one story with almost no supporters.",
    ], size=15)

    # 17 conclusions
    s = new_slide(prs, "Conclusions", "Whom to silence, whom to amplify")
    text(s, L, TOP, R - L, Inches(5.9), [
        "The published SAIR equations cannot describe a rumour and its correction growing together; a corrected, simpler structure (SAIR-H+) "
        "fits all three data sets best, and the paper's per-user factors are not supported once leakage is removed.",
        "What drives the spreading in these data is the broadcast of source tweets, outside influence, and on Charlie Hebdo a 'follow the "
        "sided-friends majority' rule.",
        "There is no universal best control: at equal effort the strategies tie at low effort; at higher effort the best mix depends on the "
        "event, and a striking tipping result depends on a modelling choice for users without sided friends.",
        "Whom to silence: the hubs, mainly because they are the rumour's source accounts. Whom to amplify: well-connected correctors, whose "
        "effect comes from shifting their followers' majority together with a small correction mass.",
        "Every number in the report is generated from the result tables and checked automatically; every decision and negative result is logged.",
    ], size=16, space=12)

    # 18 future work
    s = new_slide(prs, "Future work", "Next steps that the limitations point to")
    text(s, L, TOP, R - L, Inches(5.9), [
        "Time-varying optimal control of blocking and refutation (as in Zhong et al., 2026) instead of constant knobs.",
        "A data set with complete follower graphs and exposure (views), to remove the posters-only limitation and make κ identifiable.",
        "Agent-based simulation directly on the real follow graph, and more events and platforms.",
        "Targeting rules that need no global centrality, such as acquaintance immunisation (Cohen et al., 2003).",
        "Combining the effort-fair comparison with a cost model that reflects real platform costs of blocking and of campaigns.",
    ], size=17, space=14)

    prs.core_properties.title = "Whom to Silence, Whom to Amplify"
    prs.core_properties.author = prs.core_properties.last_modified_by = "Nivithasri A"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT}: {len(prs.slides)} slides")
    return len(prs.slides)


# ================================================================ render + layout check (PowerPoint COM)

PS = r"""
$ErrorActionPreference = 'Stop'
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open('{pptx}', $true, $false, $false)
$out = @()
$i = 0
foreach ($s in $pres.Slides) {{
  $i++
  $s.Export(('{render}\slide_{{0:D2}}.png' -f $i), 'PNG', 1600, 900)
  foreach ($sh in $s.Shapes) {{
    $bh = -1
    if ($sh.HasTextFrame -and $sh.TextFrame.HasText) {{ $bh = $sh.TextFrame.TextRange.BoundHeight }}
    $out += [pscustomobject]@{{slide=$i; name=$sh.Name; left=$sh.Left; top=$sh.Top; width=$sh.Width; height=$sh.Height; text_height=$bh; table=[bool]$sh.HasTable}}
  }}
}}
$pres.Close()
$ppt.Quit()
$out | ConvertTo-Json -Depth 3 | Out-File -Encoding utf8 '{render}\layout.json'
"""


def render_check():
    os.makedirs(RENDER, exist_ok=True)
    for fn in os.listdir(RENDER):
        os.remove(os.path.join(RENDER, fn))
    script = PS.format(pptx=OUT, render=RENDER)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], check=True, timeout=600)
    shapes = json.load(open(os.path.join(RENDER, "layout.json"), encoding="utf-8-sig"))
    sw, sh = 960.0, 540.0
    problems = []
    for x in shapes:
        bottom = x["top"] + max(x["height"], x["text_height"])
        if x["text_height"] > x["height"] + 2 and not x["table"]:
            problems.append(f"slide {x['slide']}: text overflows '{x['name']}' ({x['text_height']:.0f} > {x['height']:.0f} pt)")
        if bottom > sh + 0.5 or x["left"] + x["width"] > sw + 0.5 or x["left"] < -0.5 or x["top"] < -0.5:
            problems.append(f"slide {x['slide']}: '{x['name']}' leaves the slide (bottom {bottom:.0f} pt)")
    by_slide = {}
    for x in shapes:
        if not x["name"].startswith("deco"):
            by_slide.setdefault(x["slide"], []).append(x)
    for n, xs in by_slide.items():
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                a, b = xs[i], xs[j]
                ah, bh = max(a["height"], a["text_height"]), max(b["height"], b["text_height"])
                if a["text_height"] >= 0 and not a["table"]:
                    ah = a["text_height"] + 8                                  # occupied height of a text box = its text
                if b["text_height"] >= 0 and not b["table"]:
                    bh = b["text_height"] + 8
                ox = min(a["left"] + a["width"], b["left"] + b["width"]) - max(a["left"], b["left"])
                oy = min(a["top"] + ah, b["top"] + bh) - max(a["top"], b["top"])
                if ox > 2 and oy > 2:
                    problems.append(f"slide {n}: '{a['name']}' overlaps '{b['name']}' ({ox:.0f} × {oy:.0f} pt)")
    n_png = len([fn for fn in os.listdir(RENDER) if fn.endswith(".png")])
    print(f"rendered {n_png} slides to {RENDER}; {len(shapes)} shapes checked")
    for p in problems:
        print("LAYOUT", p)
    print("layout check: " + ("OK, no overflow, nothing off-slide, no overlaps" if not problems else f"{len(problems)} problem(s)"))
    return problems


if __name__ == "__main__":
    n = build()
    if "--render" in sys.argv:
        render_check()
