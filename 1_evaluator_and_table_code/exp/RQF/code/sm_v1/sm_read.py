#!/usr/bin/env python3
"""RQF A2.3 one-model arms (SM-R, SM-A): facts-only paired tables + A2.2 rule flags (PREREG_RQ_finalise sha 04766e97..., amendment A2 sha bface42b...).
Loading, per-unit seed means and the paired writer-cluster bootstrap (B=2000, default_rng(20261103); unit bootstrap on german_onhw_all) =
fmdx_monitor_v1/build_tables_a9.py, imported unchanged.  Training seeds pooled per unit (mean of each training seed's sampling-seed mean, unit intersection).
A2.2: target cell test_ouhdl: arm vs plain LUO-FT at the same fraction (p01: COMP-p01 pooled 3 seeds [A4000 copy, with the 4090 copy as check row];
p10: LUO-FT p10 ms10 Route A) -> U_WIN = AIoU lo > 0 AND pen-up lo > -0.0141 AND strict lo > -0.01065 AND direction lo > -0.01245.
Six other sets: arm vs LUO ms6 (Route A preds, 3 sampling seeds; A4000 copy as same-device check row) -> XNI_set = AIoU lo > -0.010; order NI clauses
reported per set.  Joint M5 ms9 reported, not gated.  SOLVES = U_WIN (target) AND XNI on all 6 other sets.  Missing inputs listed, never filled.
FM-ATTRIBUTED (FMD cell F) rows: SM arm on an FM backbone vs the same SM method on NOFM_W and on PEINIT_W (test_ouhdl; AIoU lo > 0 AND order NI)."""
import sys, glob, json, os, time
sys.path.insert(0, "$U/exp/FMD/xs/code/fmdx_monitor_v1")
import numpy as np, pandas as pd
import build_tables_a9 as T
R = "$U"; Q = R + "/exp/RQF"; MINE = Q + "/score_out/rows"; CMPD = R + "/exp/FMD/xs/runs/fmdx_monitor/score_out_cmp/rows"
RA = R + "/exp/D3_ROUTE/routeA/score_out/rows"; SA = R + "/exp/D3_ROUTE/stageA/score_out/rows"; C4090 = R + "/exp/CAND/method_round/COMP/score_out/4090/rows"
OUT = Q + "/tables/RQF_SM" + os.environ.get("SM_OUT_SUFFIX", ""); os.makedirs(os.path.dirname(OUT), exist_ok=True)
SIX = T.SIX; M = T.METRICS; SEEDS = tuple(int(x) for x in os.environ.get("SM_SEEDS", "20261034,20261035,20261036").split(","))   # SM_SEEDS=20261034 -> seed-1-only read (<author>: only seed 1 decides)
MARG = {"aiou": 0.01035, "penup_f1_v2": -0.0141, "strict_order_acc": -0.01065, "direction_acc": -0.01245}
BBS = ["LUO", "NOFM_W", "PEINIT_W", "c-radiov3-h_W", "vit-mae-large_W", "dinov3-vitl16-sat493m_W", "cleandift-sd21_W", "pe-spatial-b16-512_W"]
missing = []; rows = []
def ut(d, tag, s):
    paths = T.rows_for(s, tag, d=d)
    return T.unit_table(T.load(paths)) if paths else None
def pool(tabs, key):
    tabs = [t for t in tabs if t is not None]
    if not tabs: missing.append(key); return None
    idx = tabs[0].index
    for t in tabs[1:]: idx = idx.intersection(t.index)
    p = sum(t.loc[idx, M] for t in tabs) / len(tabs); p["writer_id"] = tabs[0].loc[idx, "writer_id"]; p.attrs["n_train_seeds"] = len(tabs)
    p.attrs["seeds"] = [getattr(t, "attrs", {}).get("seed") for t in tabs]; return p
def arm(meth, bb, frac, s):
    seeds = SEEDS if frac == "p01" else SEEDS[:1]; tabs = []
    for sd in seeds:
        tag = f"{meth}_{bb}_{frac}_s{sd}"
        pat = f"{tag}_ms*_identity" if (meth == "SMR" and bb == "LUO") else f"{tag}_ft_ms*"
        t = ut(MINE, pat, s)
        if t is not None: t.attrs["seed"] = sd
        tabs.append(t)
        if t is None: missing.append(f"{tag}/{s}")
    return pool(tabs, f"{meth}/{bb}/{frac}/{s}")
def comp(s, d, tags):
    parts = []
    for tag in tags:
        t = ut(d, tag, s)
        if t is not None: parts.append(t)
    return pool(parts, f"COMP/{d}/{s}")
COMPA4 = list(T.COMP_TAGS.values())
def con(cell, a_name, cmp_name, s, a, b, extra=None):
    if a is None or b is None: missing.append(f"{cell}:{a_name} vs {cmp_name}:{s}"); return None
    p = T.paired(a, b, s)
    if p is None: return None
    r = {"cell": cell, "arm": a_name, "comparator": cmp_name, "set": s, "n_train_seeds_arm": a.attrs.get("n_train_seeds")}
    for m in M: r.update({f"{m}_d": p[m]["delta"], f"{m}_lo": p[m]["ci_lo"], f"{m}_hi": p[m]["ci_hi"]})
    r["n_units"] = p["aiou"]["n"]; r["n_clusters"] = p["aiou"]["ncl"]
    r["ORDER_NI3"] = bool(r["penup_f1_v2_lo"] > MARG["penup_f1_v2"] and r["strict_order_acc_lo"] > MARG["strict_order_acc"] and r["direction_acc_lo"] > MARG["direction_acc"])
    r["U_WIN"] = bool(r["aiou_lo"] > 0 and r["ORDER_NI3"]); r["XNI_AIOU"] = bool(r["aiou_lo"] > -0.010)
    sds = [x for x in a.attrs.get("seeds", []) if x is not None]
    r["train_seeds_arm"] = ",".join(map(str, sds)) if sds else "n/a"
    r["evidence_label"] = "CONFIRMATORY_3SEED" if len(set(sds)) == 3 else ("SEED1_PLACEHOLDER" if 20261034 in sds else "NOT_ASSESSED")
    if extra: r.update(extra)
    rows.append(r); return r
luo6 = {s: ut(RA, "LUO_ms6", s) for s in SIX}; luo6a4 = {s: ut(CMPD, "LUO_ms6_identity", s) for s in SIX}; m5j = {s: ut(SA, "M5_joint_ms9", s) for s in ["test_ouhdl"] + SIX}
cpA4 = comp("test_ouhdl", CMPD, COMPA4); cp4090 = comp("test_ouhdl", C4090, COMPA4); ftp10 = ut(RA, "FT_p10_ms10", "test_ouhdl")
summ = []
for meth in ("SMR", "SMA"):
    for bb in BBS:
        for frac in ("p01", "p10"):
            name = f"{meth}_{bb}_{frac}"
            a = arm(meth, bb, frac, "test_ouhdl")
            if frac == "p01":
                u = con("A22_target", name, "COMP_p01_pooled[A4000]", "test_ouhdl", a, cpA4); con("A22_target_check", name, "COMP_p01_pooled[4090]", "test_ouhdl", a, cp4090)
            else:
                u = con("A22_target", name, "LUOFT_p10_ms10[RouteA]", "test_ouhdl", a, ftp10)
            con("A22_report_M5", name, "joint_M5_ms9", "test_ouhdl", a, m5j["test_ouhdl"])
            x = []
            for s in SIX:
                aa = arm(meth, bb, frac, s)
                r = con("A22_XNI", name, "LUO_ms6[RouteA]", s, aa, luo6[s]); x.append(r)
                con("A22_XNI_check", name, "LUO_ms6[A4000]", s, aa, luo6a4[s]); con("A22_report_M5", name, "joint_M5_ms9", s, aa, m5j[s])
            xni = all(r is not None and r["XNI_AIOU"] for r in x)
            summ.append(dict(arm=name, U_WIN=None if u is None else u["U_WIN"], XNI_all6=xni if all(r is not None for r in x) else None,
                             SOLVES=(None if u is None or any(r is None for r in x) else bool(u["U_WIN"] and xni)),
                             n_train_seeds=None if a is None else a.attrs.get("n_train_seeds"),
                             evidence_label=("NOT_ASSESSED" if u is None else u["evidence_label"])))
        for bb in BBS[3:]:
            for ctl in ("NOFM_W", "PEINIT_W"):
                con("FMD_cellF", f"{meth}_{bb}_p01", f"{meth}_{ctl}_p01", "test_ouhdl", arm(meth, bb, "p01", "test_ouhdl"), arm(meth, ctl, "p01", "test_ouhdl"))
df = pd.DataFrame(rows); df.to_csv(OUT + "_contrasts.csv", index=False); sm = pd.DataFrame(summ); sm.to_csv(OUT + "_solves.csv", index=False)
with open(OUT + "_summary.md", "w") as f:
    f.write(f"# RQF A2.3 SM-R / SM-A tables (facts only; built {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})\n\n")
    if len(sm): f.write(sm.to_csv(index=False, sep="|") + "\n\n")
    f.write("## Missing inputs\n" + "\n".join("- " + m for m in sorted(set(missing))) + "\n")
json.dump(dict(inputs_sha256={p: T.sha(p) for p in sorted(T.USED)}, n_rows=len(df), built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
          open(OUT + "_provenance.json", "w"), indent=1)
print("contrast rows", len(df), "missing", len(set(missing)))
