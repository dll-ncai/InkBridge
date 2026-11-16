#!/usr/bin/env python3
"""InkBridge v12 figures: fig_overview (redrawn, no C1-C3 labels), fig_warm (control renamed), fig_gain, fig_retention, fig_selector.
Result graphs plot paired differences and 95% intervals read from the same cluster files as the v11 tables
(tab:warm, tab:xni, tab:r3a); nothing is recomputed. Each plotted value is checked against the v11 tex table (3 d.p.).
fig_overview reuses the drawing functions and the selected unit of figures_make_v11.py (must sit next to this file).
fig_warm is redrawn with the v12 wording (warm adaptation, source datasets, source-training control, milestone rule) and 8-pt lettering.
fig_qual is the v11 drawing function rerun with 8-pt lettering and wider spacing; the selected units are unchanged.
Lettering: figures set at 0.85 text width (overview, warm, qual) use 8 pt in the figure, about 6.1-6.2 pt on the page (text width 160 mm).
Usage: python make_figs_v12.py SRC_DIR EXTRACT_DIR CHOSEN.json V11_TEX OUT_DIR [ONLY]
  ONLY        optional comma-separated figure names; only these are redrawn (data CSVs are always rewritten)
  SRC_DIR     copies of the cluster exp/RQF files (tables/reviewer_followup_v1/T56_per_seed.csv,
              tables/RQF_R2_interim_latest_contrasts.csv, R3/split_v1/R3_split_paired.csv)
  EXTRACT_DIR fig_extract.npz / fig_extract.json (cluster exp/FIGS/ijdar_v11_figs/extract_v3)
Outputs: OUT_DIR/figs/{fig_overview,fig_warm,fig_qual,fig_gain,fig_retention,fig_selector}.{pdf,png}, OUT_DIR/fig_*_data.csv, OUT_DIR/qa_v12.json"""
import sys, os, re, json, hashlib, inspect
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
import figures_make_v11 as F11          # v11 conventions: sizes 8/7/6, DejaVu Sans, colours, box/arrow/draw_traj/save
try: apply_figure_style(sizes=(8, 7, 6))   # figure-style skill helper when run inside the analysis kernel
except NameError: pass
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.titlelocation": "left", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "legend.frameon": False,
    "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "normal", "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0})
MM, W2, COL, S_ANN, S_TICK = F11.MM, F11.W2, F11.COL, F11.S_ANN, F11.S_TICK
COL_ROUTE = "#CC79A7"
LAB = {"WFT": r"$\mathrm{WFT}_{1\%}$", "WFTPE": r"$\mathrm{WFT}^{\mathrm{PE}}_{1\%}$", "SFT": r"$\mathrm{SFT}_{1\%}$", "FT": r"$\mathrm{FT}_{1\%}$"}
MARGIN = {"aiou": 0.01035, "penup_f1_v2": 0.0141, "strict_order_acc": 0.01065, "direction_acc": 0.01245}   # v11 tex, Sec. Evaluation Protocol
RET_THR = -0.010                                                                                         # v11 tex, retention criterion
MET = [("aiou", "AIoU", "AIoU (shape)"), ("penup_f1_v2", "Pen-up F1", "Pen-up F1"), ("strict_order_acc", "Strict order", "Strict order"),
       ("direction_acc", "Direction", "Direction"), ("kendall_tau", r"Kendall $\tau$", "Kendall $\\tau$\n(not gated)")]
SETS = [("test_casia_zh", "CASIA-zh"), ("test_casia_en", "CASIA-en"), ("test_hpl_tamil", "HPL Tamil"), ("test_iam500", "IAM500"),
        ("german_unipen_de_test", "UNIPEN-de"), ("german_onhw_all", "OnHW")]
BASES = [("E0", r"$f_{\mathrm{ur}}$ (all Urdu)", r"$\fur$ (all Urdu)"), ("LUOft_p10", r"$\mathrm{FT}_{10\%}$", r"$\FT{10\%}$"),
         ("M5", r"$f_{\mathrm{all}}$", r"$\fall$"), ("NOFM_W_FT", LAB["WFT"], r"$\WFT{1\%}$"), ("PEINIT_W_FT", LAB["WFTPE"], r"$\WFTpe{1\%}$")]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def nums(s): return [float(x) for x in re.findall(r"[+-]?\d+\.\d+", s)]

# ---- v11 tex tables (for the 3-d.p. check only; plotted values come from the source CSVs) ---------------------------------
def tex_rows(tex, label):
    a = tex.index("\\label{%s}" % label); b = tex.index("\\end{tabular}", a)
    return [[c.strip() for c in r.split("&")] for r in tex[a:b].split("\\\\")]
def parse_v11(tex):
    T = {}
    rows = tex_rows(tex, "tab:warm")
    for i, r in enumerate(rows):
        key = r[0].split("\n")[-1].strip()
        for m, tname, _ in MET:
            if key == tname and len(r) >= 3:
                ci = rows[i + 1]
                for j, s in ((1, "WFT"), (2, "WFTPE")): T[("tab:warm", s, m)] = (nums(r[j])[0], *nums(ci[j])[:2])
    for r in tex_rows(tex, "tab:xni"):
        if len(r) >= 5:
            s = "WFTPE" if "WFTpe" in r[0] else ("WFT" if "WFT" in r[0] else None)
            if s:
                for j, m in ((2, "aiou"), (4, "direction_acc")): T[("tab:xni", s, r[1], m)] = tuple(nums(r[j])[:3])
    rows = tex_rows(tex, "tab:r3a")
    for i, r in enumerate(rows):
        key = r[0].split("\n")[-1].strip()
        for b, _, tname in BASES:
            if key == tname:
                ci = rows[i + 1]
                for j, m in ((1, "penup_f1_v2"), (2, "strict_order_acc")): T[("tab:r3a", b, m)] = (nums(r[j])[0], *nums(ci[j])[:2])
    return T
def chk(row, t):
    if t is None: return dict(v11_value=np.nan, v11_ci_lo=np.nan, v11_ci_hi=np.nan, in_v11_table=False, matches_v11_3dp="")
    ok = all(abs(a - b) <= 0.0005 + 1e-9 for a, b in zip((row["delta"], row["ci_lo"], row["ci_hi"]), t))
    return dict(v11_value=t[0], v11_ci_lo=t[1], v11_ci_hi=t[2], in_v11_table=True, matches_v11_3dp=bool(ok))

# ---- data ------------------------------------------------------------------------------------------------------------------
def build_data(src, tex, out):
    T = parse_v11(tex)
    f_t56, f_r2, f_r3 = (f"{src}/tables/reviewer_followup_v1/T56_per_seed.csv", f"{src}/tables/RQF_R2_interim_latest_contrasts.csv",
                         f"{src}/R3/split_v1/R3_split_paired.csv")
    t56, r2, r3 = pd.read_csv(f_t56), pd.read_csv(f_r2), pd.read_csv(f_r3)
    r2 = r2[r2.block == "3seed"]
    G = []
    for s, arm in (("WFTPE", "PEINIT_W"), ("WFT", "NOFM_W")):
        q = t56[(t56.table == "T5_p01_vs_FT1") & (t56.arm == arm) & (t56.train_seed.astype(str) == "pooled")]; assert len(q) == 1; q = q.iloc[0]
        for m, tname, _ in MET:
            row = dict(figure="fig_gain", metric=m, series=s, comparator="FT_1pct", test_set="OUHD-L test", n_units=int(q.n_units), n_ft_seeds_arm=3,
                       n_ft_seeds_comparator=3, delta=q[m + "_d"], ci_lo=q[m + "_lo"], ci_hi=q[m + "_hi"], reference_line=-MARGIN[m] if m in MARGIN and m != "aiou" else np.nan,
                       reference_kind="non-inferiority margin" if m in MARGIN and m != "aiou" else ("zero (lower bound must exceed 0)" if m == "aiou" else "none (reported, not gated)"),
                       marker="filled", source_file=os.path.relpath(f_t56, src), source_sha256=sha(f_t56), source_row=f"table=T5_p01_vs_FT1;arm={arm};train_seed=pooled", v11_location="tab:warm")
            row.update(chk(row, T.get(("tab:warm", s, m)))); G.append(row)
    q = r2[(r2.cell == "R2b_STEPS_effect") & (r2.arm == "LUOC40")]; assert len(q) == 1; q = q.iloc[0]
    for m, tname, _ in MET:
        row = dict(figure="fig_gain", metric=m, series="SFT", comparator="FT_1pct", test_set="OUHD-L test", n_units=int(q.n_units), n_ft_seeds_arm=int(q.n_train_seeds_arm),
                   n_ft_seeds_comparator=int(q.n_train_seeds_cmp), delta=q[m + "_d"], ci_lo=q[m + "_lo"], ci_hi=q[m + "_hi"], reference_line=-MARGIN[m] if m in MARGIN and m != "aiou" else np.nan,
                   reference_kind="non-inferiority margin" if m in MARGIN and m != "aiou" else ("zero (lower bound must exceed 0)" if m == "aiou" else "none (reported, not gated)"),
                   marker="open (2 control fine-tuning seeds)", source_file=os.path.relpath(f_r2, src), source_sha256=sha(f_r2), source_row="block=3seed;cell=R2b_STEPS_effect;arm=LUOC40;comparator=pooled_COMP_p01",
                   v11_location="text Sec. Encoder versus extra steps (AIoU only)" if m == "aiou" else "not in v11")
        row.update(chk(row, None)); G.append(row)
    R = []
    for s, arm in (("WFT", "NOFM_W"), ("WFTPE", "PEINIT_W")):
        for key, name in SETS:
            q = r2[(r2.cell == "R2a_X_composed") & (r2.arm == arm) & (r2.set == key)]; assert len(q) == 1; q = q.iloc[0]
            for m in ("aiou", "direction_acc"):
                row = dict(figure="fig_retention", metric=m, series=s, comparator="f_not_ur (LUO_ms6)", test_set=name, n_units=int(q.n_units), n_ft_seeds_arm=int(q.n_train_seeds_arm),
                           n_ft_seeds_comparator=np.nan, delta=q[m + "_d"], ci_lo=q[m + "_lo"], ci_hi=q[m + "_hi"], reference_line=RET_THR if m == "aiou" else -MARGIN[m],
                           reference_kind="retention threshold" if m == "aiou" else "non-inferiority margin (reported, not gated)", marker="filled",
                           source_file=os.path.relpath(f_r2, src), source_sha256=sha(f_r2), source_row=f"block=3seed;cell=R2a_X_composed;arm={arm};set={key}", v11_location="tab:xni")
                row.update(chk(row, T.get(("tab:xni", s, name, m)))); R.append(row)
    S = []
    for b, lab, tname in BASES:
        for m in ("penup_f1_v2", "strict_order_acc"):
            q = r3[(r3.base == b) & (r3.split == "all") & (r3.metric == m)]; assert len(q) == 1; q = q.iloc[0]
            row = dict(figure="fig_selector", metric=m, series=b, comparator="same base without selection", test_set="OUHD-L test", n_units=int(q.n_units), n_ft_seeds_arm=np.nan,
                       n_ft_seeds_comparator=np.nan, delta=q.delta, ci_lo=q.ci_lo, ci_hi=q.ci_hi, reference_line=MARGIN[m], reference_kind="order-gain margin (point estimate beyond, interval above 0)",
                       marker="filled", source_file=os.path.relpath(f_r3, src), source_sha256=sha(f_r3), source_row=f"base={b};split=all;metric={m}", v11_location="tab:r3a")
            row.update(chk(row, T.get(("tab:r3a", b, m)))); S.append(row)
    D = {}
    for name, rows in (("fig_gain", G), ("fig_retention", R), ("fig_selector", S)):
        D[name] = pd.DataFrame(rows); D[name].to_csv(f"{out}/{name}_data.csv", index=False)
    return D

# ---- drawing helpers -------------------------------------------------------------------------------------------------------
def ci(ax, y, r, color, marker, filled=True, ms=3.6):
    ax.plot([r.ci_lo, r.ci_hi], [y, y], color=color, lw=1.2, solid_capstyle="butt", zorder=3)
    for x in (r.ci_lo, r.ci_hi): ax.plot([x, x], [y - 0.085, y + 0.085], color=color, lw=0.9, zorder=3)
    ax.plot(r.delta, y, marker=marker, ms=ms, mfc=color if filled else "white", mec=color, mew=0.9, ls="none", zorder=4)
def bands(ax, n):
    for g in range(0, n, 2): ax.axhspan(-g - 0.5, -g + 0.5, color="0.955", lw=0, zorder=0)
    ax.set_ylim(-n + 0.5, 0.5)
def fmt(v): return f"{v:+.3f}".replace("-", "\u2212")
TEXTWIDTH_MM = 160.0   # sn-jnl iicol: text={160mm,216mm}
INCLUDE = dict(fig_overview=0.85, fig_warm=0.85, fig_qual=0.85, fig_gain=1.0, fig_retention=1.0, fig_selector=0.8)   # \figfile widths in paper_inkbridge_v12.tex
BOXED = []             # (text, patch) pairs: every boxed label must lie inside its box
def tbox(ax, xy, w, h, text="", fs=8, fc="white", ec="0.2", lw=0.7, ls="-", tc="k", weight="normal", z=2):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z); ax.add_patch(p)
    if text: BOXED.append((ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3, linespacing=1.15, fontweight=weight), p))
    return p
def tbar8(fig, rect, fs=8):
    cax = fig.add_axes(rect); g = np.linspace(0, 1, 256)[None]; cax.imshow(g, aspect="auto", cmap=matplotlib.colors.ListedColormap(F11.tcol(g[0])))
    cax.set_xticks([0, 255]); cax.set_xticklabels(["start", "end"], fontsize=fs); cax.set_yticks([]); cax.tick_params(length=0, pad=1.5)
    for sp in cax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.4)
    cax.set_title("pen time", fontsize=fs, loc="center", pad=2); return cax
def qa(fig, name):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    inside = lambda bt, bp: bt.x0 >= bp.x0 and bt.x1 <= bp.x1 and bt.y0 >= bp.y0 and bt.y1 <= bp.y1
    box_overflow = [t.get_text() for t, p in BOXED if t.figure is fig and not inside(t.get_window_extent(r), p.get_window_extent(r))]
    tb = fig.get_tightbbox(r); nat_mm = (tb.width + 0.04) * 25.4; sc = INCLUDE.get(name, 1.0) * TEXTWIDTH_MM / nat_mm
    fsz = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    on_page = dict(natural_width_mm=round(nat_mm, 1), include_width=INCLUDE.get(name, 1.0), scale=round(sc, 3), min_lettering_pt_in_figure=min(fsz),
                   min_lettering_pt_on_page=round(min(fsz) * sc, 2), has_subscripts=any(re.search(r"\$[^$]*[_^]", t.get_text()) for t in fig.findobj(matplotlib.text.Text)))
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    tl = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both")) for ax in fig.axes}
    ov = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    ov += [(t.get_text(), "spine") for t, bt in texts for s, bs in spines if bt.overlaps(bs) and t not in tl[s.axes] and t.axes is s.axes and t.get_text() not in "ab"]
    outside = [t.get_text() for t, b in texts if not (b.x0 >= 0 and b.y0 >= 0)]
    return dict(figure=name, n_texts=len(texts), text_overlaps=ov, outside=outside, box_overflow=box_overflow, on_page=on_page)

# ---- fig_gain --------------------------------------------------------------------------------------------------------------
def fig_gain(D, out):
    d = D["fig_gain"]; fig, ax = plt.subplots(figsize=(W2, 70 * MM)); fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.15)
    ser = [("WFTPE", 0.27, "o", True), ("WFT", 0.0, "s", True), ("SFT", -0.27, "D", False)]
    bands(ax, len(MET)); ax.axvline(0, color="0.35", lw=0.7, zorder=1)
    for g, (m, _, ylab) in enumerate(MET):
        for s, off, mk, filled in ser:
            r = d[(d.metric == m) & (d.series == s)].iloc[0]; ci(ax, -g + off, r, COL[s], mk, filled)
            if m == "aiou" or (m in ("penup_f1_v2", "strict_order_acc") and s in ("WFT", "WFTPE")): ax.text(r.ci_hi + 0.002, -g + off, fmt(r.delta), fontsize=S_ANN, va="center", ha="left")
        if m in MARGIN and m != "aiou": ax.plot([-MARGIN[m]] * 2, [-g - 0.46, -g + 0.46], color="0.2", lw=0.9, ls=(0, (2.5, 1.8)), zorder=2)
    ax.set_yticks([-g for g in range(len(MET))]); ax.set_yticklabels([m[2] for m in MET], fontsize=8)
    ax.set_xlim(-0.03, 0.14); ax.set_xticks(np.arange(-0.02, 0.121, 0.02))
    ax.set_xlabel(r"Paired difference to plain fine-tuning $\mathrm{FT}_{1\%}$ on Urdu (95% interval)   $\rightarrow$ better")
    h = [Line2D([], [], marker="o", color=COL["WFTPE"], ms=4, lw=1.2, label=LAB["WFTPE"] + " warm adaptation, PE-initialised encoder"),
         Line2D([], [], marker="s", color=COL["WFT"], ms=4, lw=1.2, label=LAB["WFT"] + " warm adaptation"),
         Line2D([], [], marker="D", color=COL["SFT"], mfc="white", ms=4, lw=1.2, label=LAB["SFT"] + " control: same extra source training, original encoder"),
         Line2D([], [], color="0.2", lw=0.9, ls=(0, (2.5, 1.8)), label="non-inferiority margin of the metric")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=2.2, borderaxespad=0.6, labelspacing=0.45)
    q = qa(fig, "fig_gain"); F11.save(fig, out + "/figs", "fig_gain"); return q

# ---- fig_retention ---------------------------------------------------------------------------------------------------------
def fig_retention(D, out):
    d = D["fig_retention"]; xa, xb = (-0.02, 0.07), (-0.195, 0.03)
    fig = plt.figure(figsize=(W2, 66 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[xa[1] - xa[0], xb[1] - xb[0]], wspace=0.05, left=0.115, right=0.985, top=0.9, bottom=0.16)   # same length per unit in both panels
    A, B = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]); ser = [("WFTPE", 0.17, "o"), ("WFT", -0.17, "s")]
    for ax, m, xl in ((A, "aiou", xa), (B, "direction_acc", xb)):
        bands(ax, len(SETS)); ax.axvline(0, color="0.35", lw=0.7, zorder=1); ax.set_xlim(*xl)
        for i, (_, name) in enumerate(SETS):
            for s, off, mk in ser:
                r = d[(d.metric == m) & (d.series == s) & (d.test_set == name)].iloc[0]; ci(ax, -i + off, r, COL[s], mk)
                if m == "direction_acc" and name == "UNIPEN-de":
                    ax.text(r.delta, -i + off + (0.13 if off > 0 else -0.13), fmt(r.delta), fontsize=S_ANN, ha="center", va="bottom" if off > 0 else "top")
        ax.set_yticks([-i for i in range(len(SETS))])
    A.axvline(RET_THR, color="0.2", lw=0.9, ls=(0, (2.5, 1.8)), zorder=2); B.axvline(-MARGIN["direction_acc"], color="0.2", lw=0.9, ls=(0, (0.8, 1.6)), zorder=2)
    A.set_yticklabels([n for _, n in SETS], fontsize=8); B.set_yticklabels([])
    A.set_xticks([0, 0.02, 0.04, 0.06]); B.set_xticks(np.arange(-0.18, 0.021, 0.02))
    A.set_title("Shape is retained"); B.set_title("Direction is not retained on UNIPEN-de")
    A.set_xlabel(r"$\Delta$ AIoU"); B.set_xlabel(r"$\Delta$ direction accuracy  (routed composition minus backbone $f_{\neg\mathrm{ur}}$, 95% interval)  $\rightarrow$ better", x=0.47)
    F11.letter(A, "a", dx=-0.04); F11.letter(B, "b", dx=-0.012)
    h = [Line2D([], [], marker="o", color=COL["WFTPE"], ms=4, lw=1.2, label=r"$\mathcal{R}($" + LAB["WFTPE"] + r"$)$ routed, PE-initialised encoder"),
         Line2D([], [], marker="s", color=COL["WFT"], ms=4, lw=1.2, label=r"$\mathcal{R}($" + LAB["WFT"] + r"$)$ routed"),
         Line2D([], [], color="0.2", lw=0.9, ls=(0, (2.5, 1.8)), label="retention threshold for AIoU (\u22120.010)"),
         Line2D([], [], color="0.2", lw=0.9, ls=(0, (0.8, 1.6)), label="non-inferiority margin for direction (\u22120.01245)")]
    B.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=2.2, borderaxespad=0.6, labelspacing=0.45)
    q = qa(fig, "fig_retention"); F11.save(fig, out + "/figs", "fig_retention"); return q

# ---- fig_selector ----------------------------------------------------------------------------------------------------------
def fig_selector(D, out):
    d = D["fig_selector"]; fig, axs = plt.subplots(1, 2, figsize=(W2, 54 * MM), sharey=True); fig.subplots_adjust(left=0.115, right=0.985, top=0.89, bottom=0.3, wspace=0.06)
    for ax, (m, xlab), let in zip(axs, (("penup_f1_v2", r"$\Delta$ pen-up F1"), ("strict_order_acc", r"$\Delta$ strict order")), "ab"):
        bands(ax, len(BASES)); ax.axvline(0, color="0.35", lw=0.7, zorder=1); ax.axvline(MARGIN[m], color="0.2", lw=0.9, ls=(0, (2.5, 1.8)), zorder=2)
        for i, (b, _, _) in enumerate(BASES): ci(ax, -i, d[(d.metric == m) & (d.series == b)].iloc[0], COL["VER"], "o", ms=4)
        ax.set_xlim(-0.003, 0.034); ax.set_xticks(np.arange(0, 0.031, 0.01)); ax.set_xlabel(xlab + r"   $\rightarrow$ better")
        ax.text(MARGIN[m], 1.02, f"selection-gain margin {MARGIN[m]:g}", transform=ax.get_xaxis_transform(), fontsize=S_ANN, ha="center", va="bottom")
        F11.letter(ax, let, dx=-0.02 if let == "a" else 0.03)
    axs[0].set_yticks([-i for i in range(len(BASES))]); axs[0].set_yticklabels([b[1] for b in BASES], fontsize=8)
    fig.text(0.55, 0.02, "Endpoint Consensus Selection ($k$ = 3) minus the same base model without selection, Urdu (95% interval)", ha="center", va="bottom", fontsize=8)
    q = qa(fig, "fig_selector"); F11.save(fig, out + "/figs", "fig_selector"); return q

# ---- fig_overview (8-pt lettering: about 6.2 pt at 0.85 text width) ---------------------------------------------------------
def fig_overview(ex, chosen, out):
    X = dict(np.load(f"{ex}/fig_extract.npz", allow_pickle=False)); J = json.load(open(f"{ex}/fig_extract.json")); F11.CH = json.load(open(chosen))
    st, ro = F11.CH["overview"]; tag = f"{st}__{ro}"; img = X.get(tag + "__cond_f4", X[tag + "__cond"]); G = F11.gt_strokes(X, tag)
    v = J["units"][st][ro]["verifier"]["WFTPE_1"]; samp = [F11.strokes(X[f"{tag}__WFTPE_1__{k}"]) for k in range(3)]
    lim = F11.bbox(G + sum(samp, []), 4); arrow = F11.arrow; FS, Hh, yc = 8, 63, 45; BLUE, GREEN = COL["WFTPE"], COL["VER"]
    fig = plt.figure(figsize=(W2, Hh * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 174); ax.set_ylim(0, Hh); ax.axis("off")
    def sub(x, y, w, h): return fig.add_axes([x / 174, y / Hh, w / 174, h / Hh])
    def inbox(p, x, y, text, **kw): BOXED.append((ax.text(x, y, text, ha="center", va="center", fontsize=FS, zorder=3, linespacing=1.15, **kw), p))
    a = sub(1, yc - 8.5, 17, 17); a.imshow(img, cmap="gray", extent=(0, 64, 64, 0)); F11.frame_ax(a, lim)
    ax.text(9.5, yc + 9.6, "offline\nimage", ha="center", va="bottom", fontsize=FS, linespacing=1.1)
    arrow(ax, (18.5, yc), (21, yc))
    p = tbox(ax, (21, yc - 9), 25.5, 18, fc="#F8E8F1", ec=COL_ROUTE)
    inbox(p, 33.75, yc + 5.6, "Script routing", fontweight="bold"); inbox(p, 33.75, yc - 2.2, "script classifier\nsends the image\nto one model")
    arrow(ax, (46.5, yc + 2.5), (50.5, yc + 5.5), color=COL_ROUTE); arrow(ax, (46.5, yc - 2.5), (50.5, yc - 5.5), color=COL_ROUTE)
    tbox(ax, (50.5, yc + 1), 32.5, 9, "target script:\nwarm-adapted model", fc="0.95", fs=FS)
    tbox(ax, (50.5, yc - 10), 32.5, 9, "other inputs:\nwarm backbone", fc="0.95", fs=FS)
    tbox(ax, (49, yc - 11.5), 35.5, 23, fc="none", ec=BLUE, ls=(0, (3, 2)), lw=0.6)
    ax.text(66.75, yc + 12.7, "image encoder + diffusion trajectory backbone", ha="center", va="bottom", fontsize=FS)
    arrow(ax, (84.5, yc), (86.5, yc))
    for k in range(3):
        a = sub(87 + k * 9.6, yc - 4.5, 9, 9); F11.draw_traj(a, samp[k], lim, lw=0.7, start=False, penup=False)
    ax.text(101.1, yc + 5.8, r"$k$ = 3 samples", ha="center", va="bottom", fontsize=FS)
    arrow(ax, (115.7, yc), (118.2, yc))
    p = tbox(ax, (118.2, yc - 10.5), 34, 21, fc="#E5F5EF", ec=GREEN)
    inbox(p, 135.2, yc + 5.3, "Endpoint Consensus\nSelection", fontweight="bold"); inbox(p, 135.2, yc - 4.4, "keeps the sample\nwhose endpoints agree\nmost (no labels)")
    arrow(ax, (152.2, yc), (155.2, yc))
    a = sub(155.7, yc - 8.5, 17, 17); F11.draw_traj(a, samp[v["sel"]], lim, lw=0.9); F11.frame_ax(a, lim, color=GREEN, lw=1.2)
    ax.text(164.2, yc + 9.6, "online\ntrajectory", ha="center", va="bottom", fontsize=FS, linespacing=1.1)
    # warm adaptation lane (training)
    tbox(ax, (4, 1), 121, 24, fc="#EAF4FB", ec=BLUE, ls=(0, (3, 2)), z=0)
    ax.text(6, 24, "Warm adaptation to a new script (training)", fontsize=FS, va="top", color=BLUE, fontweight="bold")
    tbox(ax, (6, 2.5), 28, 16, "leave-target-out\nbackbone", fs=FS); arrow(ax, (34, 10.5), (37, 10.5), lw=0.6)
    tbox(ax, (37, 2.5), 43, 16, "warm backbone\nnew 224-px encoder\n(optional PE init),\nsource training continued", fs=FS); arrow(ax, (80, 10.5), (83, 10.5), lw=0.6)
    tbox(ax, (83, 2.5), 40, 16, "warm-adapted model\nfine-tuned on a fraction\n" + r"$\rho$ of the target script" + "\n(336 Urdu units at 1%)", fs=FS)
    arrow(ax, (66.75, 25), (66.75, yc - 11.5), color=BLUE, lw=0.8, ls=(0, (3, 2)))
    ax.text(68.55, 29.3, "provides both models", fontsize=FS, color=BLUE, va="center")
    tbar8(fig, [0.775, 0.16, 0.15, 0.04], FS)
    q = qa(fig, "fig_overview"); F11.save(fig, out + "/figs", "fig_overview")
    q.update(unit=[st, ro], selected_sample=int(v["sel"])); return q

# ---- fig_warm (v12 wording, 8-pt lettering) ---------------------------------------------------------------------------------
def fig_warm(out):
    FS, H, h = 8, 57, 9; arrow = F11.arrow
    fig = plt.figure(figsize=(W2, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 174); ax.set_ylim(0, H); ax.axis("off")
    rows = [("FT", r"$\mathrm{FT}_{\rho}$", "plain fine-tune", None),
            ("SFT", r"$\mathrm{SFT}_{\rho}$", "source-training\ncontrol", "original 64-px\nCNN encoder"),
            ("WFT", r"$\mathrm{WFT}_{\rho}$", "warm adaptation", "new 224-px ViT-S/14\nencoder, random init"),
            ("WFTPE", r"$\mathrm{WFT}^{\mathrm{PE}}_{\rho}$", "warm adaptation,\nPE encoder init", "new 224-px ViT-S/14\nencoder, PE init (FM)")]
    ys = [39, 27, 15, 3]
    tbox(ax, (1, 16.5), 21.5, 18, "leave-target-\nout backbone\n(source\ndatasets)", fc="0.93", ec=COL["BB"], fs=FS)
    ax.text(59, H - 0.5, "stage 1: warm stage on the source datasets\n(four 10,000-step milestones, one kept)", ha="center", va="top", fontsize=FS, color="0.3", linespacing=1.15)
    ax.text(97, H - 0.5, "stage 2: target data,\n" + r"$\rho \in \{1\%, 10\%, 30\%\}$", ha="left", va="top", fontsize=FS, color="0.3", linespacing=1.15)
    ax.plot([94, 94], [1, 49.3], color="0.7", lw=0.5, ls=(0, (3, 2)))
    for (k, lab, desc, enc), y in zip(rows, ys):
        c = COL[k]; arrow(ax, (22.8, 25.5), (27, y + h / 2), color="0.4", lw=0.6)
        if enc is None:
            ax.plot([27, 97], [y + h / 2] * 2, color=c, lw=0.8, ls=(0, (2, 2)), zorder=1)
            ax.text(59, y + h / 2 + 0.6, "no warm stage", ha="center", va="bottom", fontsize=FS, color="0.35")
        else:
            tbox(ax, (27, y), 35, h, enc, ec=c, lw=0.9, fs=FS); arrow(ax, (62, y + h / 2), (65, y + h / 2), color=c, lw=0.8)
            tbox(ax, (65, y), 26, h, "continued\nsource training", ec=c, lw=0.9, fs=FS); arrow(ax, (91, y + h / 2), (97, y + h / 2), color=c, lw=0.8)
        tbox(ax, (97, y), 26, h, "fine-tune on\n" + r"$\rho$ of the target", ec=c, lw=0.9, fs=FS)
        arrow(ax, (123, y + h / 2), (126, y + h / 2), color=c, lw=0.8)
        tbox(ax, (126, y), 14, h, lab, fc=c, ec=c, fs=10, tc="white" if k in ("WFTPE", "FT") else "k")
        ax.text(142, y + h / 2, desc, ha="left", va="center", fontsize=FS, color="0.15", linespacing=1.15)
    q = qa(fig, "fig_warm"); F11.save(fig, out + "/figs", "fig_warm")
    q.update(old_label_strings=[t.get_text() for t in fig.findobj(matplotlib.text.Text) if re.search(r"step-matched|non-target|warm fine-tune|40k|\(C[123]\)|RQ[123]", t.get_text())]); return q

# ---- fig_qual (v11 drawing function rerun with 8-pt lettering; units unchanged) -------------------------------------------
QUAL_SUBS = [("Wm, Hm = 174.0, 78.0", "Wm, Hm = 191.0, 96.0"), ("x0s = (6.0, 93.0)", "x0s = (6.0, 100.6)"), ("ytop = Hm - 11.0", "ytop = Hm - 16.5"), ("pitch = 20.6", "pitch = 23.6"),
             ("ci * (p + 0.9)", "ci * (p + 3.0)"), ("1.9 / Hm, 0.1, 0.018]", "4.6 / Hm, 0.1, 0.018]"), ("bg.text(x0s[0] + 21, 2.6,", "bg.text(x0s[0] + 24, 5.4,"),
             ('va="top", fontsize=S_ANN, color="0.25")', 'va="top", fontsize=S_ANN, color="0.25", linespacing=1.15)')]
QUAL_HEADER_WRAP = (" \u00b7 unit 6", "\nUnit 6")   # second block header set on two lines so it stays inside the block
def fig_qual(ex, chosen, out):
    X = dict(np.load(f"{ex}/fig_extract.npz", allow_pickle=False)); J = json.load(open(f"{ex}/fig_extract.json")); CH = json.load(open(chosen))
    code = inspect.getsource(F11.fig_qual)
    for a_, b_ in QUAL_SUBS: assert code.count(a_) == 1, a_; code = code.replace(a_, b_)
    CH["qual_headers"] = [t.replace(*QUAL_HEADER_WRAP) for t in CH["qual_headers"]]
    ns = dict(vars(F11)); ns.update(CH=CH, S_ANN=8, S_TICK=8, tbar=tbar8)
    def save(fig, o, name): ns["_qa"] = qa(fig, name); F11.save(fig, o, name)
    ns["save"] = save; exec(code, ns); shown = ns["fig_qual"](out + "/figs", X, J)
    q = ns["_qa"]; q.update(units=[[u["set"], u["role"], u["unit_id"], int(u["c3_sel"])] for u in shown.values()], substitutions=QUAL_SUBS, headers=CH["qual_headers"]); return q

def main():
    src, ex, chosen, texp, out = sys.argv[1:6]; os.makedirs(out + "/figs", exist_ok=True)
    only = sys.argv[6].split(",") if len(sys.argv) > 6 else None
    D = build_data(src, open(texp).read(), out)
    jobs = dict(fig_overview=lambda: fig_overview(ex, chosen, out), fig_warm=lambda: fig_warm(out), fig_qual=lambda: fig_qual(ex, chosen, out), fig_gain=lambda: fig_gain(D, out),
                fig_retention=lambda: fig_retention(D, out), fig_selector=lambda: fig_selector(D, out))
    Q = {q["figure"]: q for q in json.load(open(out + "/qa_v12.json"))["qa"]} if only and os.path.exists(out + "/qa_v12.json") else {}
    for k, f in jobs.items():
        if only is None or k in only: Q[k] = f()
    Q = [Q[k] for k in jobs if k in Q]
    chkd = {k: dict(rows=len(v), in_v11_table=int(v.in_v11_table.sum()), match_3dp=int((v.matches_v11_3dp == True).sum())) for k, v in D.items()}
    json.dump(dict(qa=Q, table_check=chkd), open(out + "/qa_v12.json", "w"), indent=1, default=str); print(json.dumps(dict(qa=Q, table_check=chkd), default=str))
if __name__ == "__main__": main()
