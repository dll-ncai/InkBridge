#!/usr/bin/env python3
"""RQF R2 tables (facts only + rule flags). Imports the FMDX warm-monitor table code (build_tables_a9.py, unchanged) for loading, per-unit seed means and the paired writer-cluster bootstrap
(B=2000, default_rng(20261103); unit-cluster on german_onhw_all). Training seeds are pooled per unit (mean over training seeds of each seed's sampling-seed mean, unit intersection).
Rules applied verbatim: U-WIN (FMD A4.5 cell U), X-NI (cell X), ENCODER-ATTRIBUTED (PREREG RQF R2b), U-WIN vs plain LUO-FT at the same fraction (R2c). Missing inputs are listed, never filled."""
import sys, glob, json, os, re, hashlib
sys.path.insert(0, "$U/exp/FMD/xs/code/fmdx_monitor_v1")
import numpy as np, pandas as pd
import build_tables_a9 as T
R = "$U"; Q = R + "/exp/RQF"; MMR = R + "/exp/FMD/xs/runs/fmdx_monitor"
MINE = Q + "/score_out/rows"; MON = MMR + "/score_out/rows"; CMPD = MMR + "/score_out_cmp/rows"
T.LUO_DIR = CMPD; T.LUO_TAG = "LUO_ms6_identity"; T.COMP_DIR = CMPD
T.COMP_TAGS = {"20261034": "comp_p01_identity_s20261034_vps_ms3_identity", "20261035": "comp_p01_identity_s20261035_ms3_identity", "20261036": "comp_p01_identity_s20261036_ms4_identity"}
OUT = sys.argv[1] if len(sys.argv) > 1 else Q + "/tables/RQF_R2"
SIX = T.SIX; M = T.METRICS; SEEDS = (20261034, 20261035, 20261036)
MARG = {"aiou": 0.01035, "penup_f1_v2": -0.0141, "strict_order_acc": -0.01065, "direction_acc": -0.01245}
missing = []
def ut(d, tag, s, seeds=(0, 1, 2)):
    paths = T.rows_for(s, tag, seeds=seeds, d=d)
    if not paths: return None
    x = T.unit_table(T.load(paths)); x.attrs["dev"] = "mixed-see-device_log"; return x
def pool(tabs, key):
    tabs = [t for t in tabs if t is not None]
    if not tabs: missing.append(key); return None
    idx = tabs[0].index
    for t in tabs[1:]: idx = idx.intersection(t.index)
    p = sum(t.loc[idx, M] for t in tabs) / len(tabs); p["writer_id"] = tabs[0].loc[idx, "writer_id"]; p["n_seeds"] = min(int(t.n_seeds.min()) for t in tabs)
    p.attrs["dev"] = "mixed"; p.attrs["n_train_seeds"] = len(tabs); return p
def warm(arm, kind, s, frac="p01"):
    tabs = []
    if frac == "p01" and SEEDS[0] in SS: tabs.append(ut(MON, f"{arm}_{'ft' if kind=='ft' else kind}_ms*", s))   # training seed 20261034 = existing monitor FT
    for sd in [x for x in (SEEDS[1:] if frac == "p01" else SEEDS) if x in SS]:
        tabs.append(ut(MINE, f"{arm}_{frac}_s{sd}_{kind}_ms*", s))
    return pool(tabs, f"{arm}/{frac}/{kind}/{s}")
def luo(label, s, frac=None):   # LUO-type FT arms from ctrl_sample / lane_score rows
    tabs = [ut(MINE, f"{label}_{frac}_s{sd}_ms*_identity", s) for sd in SEEDS if sd in SS]
    return pool(tabs, f"{label}/{frac}/{s}")
def comp(s):
    parts = T.pooled_comp(s)
    if not parts: missing.append(f"COMP/{s}"); return None
    p = sum(parts) / len(parts); d = T.load(T.rows_for(s, next(iter(T.COMP_TAGS.values())), d=T.COMP_DIR)); p["writer_id"] = T.unit_table(d)["writer_id"]; p.attrs["dev"] = "A4000"; p.attrs["n_train_seeds"] = len(parts); return p
def luo6(s):
    d = T.load(T.rows_for(s, T.LUO_TAG, d=T.LUO_DIR)); return T.unit_table(d) if d is not None else None
rows = []
VPS_NOTE = "EXPLORATORY (VPS RTX 4090; e0_train.py, torch 2.9.1, backbone ms1)"
SS = (SEEDS[0],); BLK = "seed1"
def lab(cell, arm):   # descriptive note (the evidence_label itself follows the 3-label seed rule in con())
    if arm.startswith("LUOC40VPS") or "_VPS" in cell: return VPS_NOTE
    if cell.startswith(("R2a_var", "R2c_p30", "R2c_p10_X")): return "extra/exploratory cell"
    return "CLUSTER-A"   # R2b confirmatory lane = CLUSTER-A copy 24964380 chain only (<author> 10 Nov)
def evlabel(arm, a, comparator="", b=None, cell=""):
    # SEED1_PLACEHOLDER (block seed1) | CONFIRMATORY_3SEED only if arm AND comparator have all 3 training seeds (a fixed single reference such as LUO ms6 carries no training-seed count) and no VPS row | NOT_ASSESSED
    vps = arm.startswith("LUOC40VPS") or str(comparator).startswith("LUOC40VPS") or "_VPS" in cell
    if BLK == "seed1": return "SEED1_PLACEHOLDER"
    na = a.attrs.get("n_train_seeds") or 1
    nb = b.attrs.get("n_train_seeds") if b is not None else None
    ok = na == 3 and int(a["n_seeds"].min()) >= 3 and (nb is None or nb == 3) and not vps
    return "CONFIRMATORY_3SEED" if ok else "NOT_ASSESSED"
def con(cell, arm, comparator, s, a, b):
    if a is None or b is None: missing.append(f"{cell}:{arm} vs {comparator}:{s}"); return None
    p = T.paired(a, b, s)
    if p is None: return None
    r = {"evidence_label": evlabel(arm, a, comparator, b, cell), "exploratory": bool(arm.startswith("LUOC40VPS") or str(comparator).startswith("LUOC40VPS") or "_VPS" in cell), "block": BLK, "note": lab(cell, arm), "cell": cell, "arm": arm, "comparator": comparator, "set": s, "n_train_seeds_arm": a.attrs.get("n_train_seeds"), "n_train_seeds_cmp": b.attrs.get("n_train_seeds")}
    for m in M: r.update({f"{m}_d": p[m]["delta"], f"{m}_lo": p[m]["ci_lo"], f"{m}_hi": p[m]["ci_hi"]}); r["n_units"] = p[m]["n"]
    ok_ord = (r["penup_f1_v2_lo"] > MARG["penup_f1_v2"]) and (r["strict_order_acc_lo"] > MARG["strict_order_acc"]) and (r["direction_acc_lo"] > MARG["direction_acc"])
    r["U_WIN"] = bool(r["aiou_lo"] > 0 and ok_ord); r["ORDER_NI3"] = bool(ok_ord); r["AIOU_gt0"] = bool(r["aiou_lo"] > 0)
    rows.append(r); return r
WARMS = ["NOFM_W", "PEINIT_W"]
def block():
    WARMS = ["NOFM_W", "PEINIT_W"]
    cp = comp("test_ouhdl")
    # R2a: seeds pooled, U-WIN vs pooled COMP-p01; X-NI composed vs LUO on six sets
    xni = {}
    for arm in WARMS:
        con("R2a_U", arm, "pooled_COMP_p01", "test_ouhdl", warm(arm, "ft", "test_ouhdl"), cp)
        ok = []
        for s in SIX:
            r = con("R2a_X_composed", arm, "LUO_ms6", s, warm(arm, "cmp", s), luo6(s)); ok.append(bool(r and r["aiou_lo"] > -0.010))
        xni[arm] = all(ok) and len(ok) == 6
    # R2a exploratory variance row: second backbone seed + 1 FT seed
    for arm in WARMS:
        t = ut(MINE, f"{arm}2_p01_s{SEEDS[0]}_ft_ms*", "test_ouhdl")
        if t is not None: con("R2a_var_2nd_bb_seed", arm + "2", "pooled_COMP_p01", "test_ouhdl", t, cp)
        else: missing.append(f"{arm}2 FT")
    # R2b: LUOC40
    lc = luo("LUOC40", "test_ouhdl", "p01")
    con("R2b_STEPS_effect", "LUOC40", "pooled_COMP_p01", "test_ouhdl", lc, cp)
    for arm in WARMS:
        r = con("R2b_ENCODER_ATTRIBUTED", arm, "LUOC40", "test_ouhdl", warm(arm, "ft", "test_ouhdl"), lc)
        if r: r["ENCODER_ATTRIBUTED"] = bool(r["AIOU_gt0"] and r["ORDER_NI3"])
    # R2b VPS copy (EXPLORATORY row): LUOC40VPS_* rows pulled from the VPS (renamed on pull)
    lv = luo("LUOC40VPS", "test_ouhdl", "p01")
    con("R2b_STEPS_effect_VPS", "LUOC40VPS", "pooled_COMP_p01", "test_ouhdl", lv, cp)
    con("R2b_VPS_vs_CLUSTER-A_LUOC40", "LUOC40VPS", "LUOC40", "test_ouhdl", lv, lc)
    for arm in WARMS:
        r = con("R2b_ENCODER_ATTRIBUTED_VPS", arm, "LUOC40VPS", "test_ouhdl", warm(arm, "ft", "test_ouhdl"), lv)
        if r: r["ENCODER_ATTRIBUTED"] = bool(r["AIOU_gt0"] and r["ORDER_NI3"])
    # R2c: curve vs plain LUO-FT at the same fraction
    for frac in ("p10", "p30"):
        base = luo("LUOFT", "test_ouhdl", frac) if frac == "p10" else None
        if frac == "p30": missing.append("LUO-FT p30 comparator: existing Route A M5 ft/p30 ms10 (4090 preds, one training seed) - read separately, not pooled here")
        for arm in WARMS + ["LUOC40"]:
            a = warm(arm, "ft", "test_ouhdl", frac) if arm in WARMS else luo(arm, "test_ouhdl", frac)
            if base is not None: con(f"R2c_{frac}_U_vs_LUOFT", arm, f"LUOFT_{frac}", "test_ouhdl", a, base)
            con(f"R2c_{frac}_U_vs_pooled_COMP_p01", arm, "pooled_COMP_p01", "test_ouhdl", a, cp)
            if frac == "p10":   # forgetting read (FT without router) on the six non-Urdu sets vs plain LUO-FT p10
                for s_ in SIX:
                    aa = warm(arm, "ft", s_, frac) if arm in WARMS else luo(arm, s_, frac)
                    bb_ = luo("LUOFT", s_, frac)
                    con("R2c_p10_X_FTnoRouter_vs_LUOFT", arm, "LUOFT_p10", s_, aa, bb_)
    return xni
def r2e():
    # R2e per-writer UNIPEN-de (direction, tau) arm minus LUO ms6
    lu = luo6("german_unipen_de_test")
    def perwriter(tag, a):
        if a is None or lu is None: return
        j = a[M + ["writer_id"]].join(lu[M], rsuffix="_luo", how="inner")
        for w, g in j.groupby("writer_id"):
            pw.append({"evidence_label": evlabel(tag, a), "block": BLK, "arm": tag, "writer": w, "n_units": len(g), "direction_arm": g.direction_acc.mean(), "direction_luo": g.direction_acc_luo.mean(), "direction_d": (g.direction_acc - g.direction_acc_luo).mean(),
                       "tau_arm": g.kendall_tau.mean(), "tau_luo": g.kendall_tau_luo.mean(), "tau_d": (g.kendall_tau - g.kendall_tau_luo).mean()})
    s = "german_unipen_de_test"
    for arm in WARMS: perwriter(arm + "_composed", warm(arm, "cmp", s)); perwriter(arm + "_ft_norouter", warm(arm, "ft", s))
    perwriter("LUOC40_ft", luo("LUOC40", s, "p01"))
    perwriter("LUOC40VPS_ft_EXPLORATORY", luo("LUOC40VPS", s, "p01"))
COMP_TAGS_ALL = dict(T.COMP_TAGS); xni_all = {}; pw = []
for BLK, SS in (("seed1", (SEEDS[0],)), ("3seed", SEEDS)):
    T.COMP_TAGS = {k: v for k, v in COMP_TAGS_ALL.items() if int(k) in SS}
    xni_all[BLK] = block(); r2e()
df = pd.DataFrame(rows); df.to_csv(OUT + "_contrasts.csv", index=False)
pd.DataFrame(pw).to_csv(OUT + "_R2e_unipen_de_per_writer.csv", index=False)
with open(OUT + "_summary.md", "w") as f:
    f.write("# RQF R2 tables (facts only)\nNo multiplicity correction beyond the prereg (Holm over 2 arms for R2a is applied by the reader). Rule flags: U_WIN, ORDER_NI3, ENCODER_ATTRIBUTED as defined above.\n")
    f.write(f"X-NI (six-set composed AIoU CI lo > -0.010, R2a): {json.dumps(xni_all)}\n\n")
    if len(df):
        cols = ["evidence_label", "block", "cell", "arm", "comparator", "set", "n_train_seeds_arm", "n_units", "aiou_d", "aiou_lo", "penup_f1_v2_lo", "strict_order_acc_lo", "direction_acc_lo", "U_WIN"]
        d2 = df[[c for c in cols if c in df]].round(4)
        f.write("| " + " | ".join(d2.columns) + " |\n|" + "---|" * len(d2.columns) + "\n")
        for _, r_ in d2.iterrows(): f.write("| " + " | ".join(str(v) for v in r_.tolist()) + " |\n")
        f.write("\n")
    f.write("## Missing inputs\n" + "\n".join("- " + m for m in sorted(set(missing))) + "\n")
print("contrast rows", len(df), "per-writer rows", len(pw), "missing", len(set(missing)))
