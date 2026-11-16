#!/usr/bin/env python3
"""RQF R2d read (facts only). Inputs: lane_score.py rows (uhwr_eval 0.3.1, native protocol) per system in --rows/<SYSTEM>/.
Per unit: mean over sampling seeds (and over the 3 training seeds for pooled FT systems). Paired differences on the unit
intersection; 95 % writer-cluster percentile bootstrap, B = 2,000, numpy default_rng(0) multinomial cluster weights (as FMCMP
fmcmp_score.py); german_onhw_all (2 writers): unit bootstrap.
Rules (prereg section 2 R2d + FMD A4.5 verbatim + amendment A2.2/A2.3):
 U-WIN' (cell T = test_hpl_tamil): LTO_W-FT pooled - LTO-FT pooled: AIoU lo > 0 AND penup_f1_v2 lo > -0.0141 AND strict lo > -0.01065
        AND direction lo > -0.01245.
 X-NI'  (cell X = the 6 other sets incl. OUHD-L): composed LTO_W system (ScriptCNN router script_clf.pt sha 22bb8802, p_hpl_tamil >= 0.5
        -> LTO_W-FT, else LTO_W backbone; per-unit p from exp/D3_ROUTE/router/manifest_<set>.csv) - LTO selected backbone: AIoU lo > -0.010
        on every set. Reported: backbone-only LTO_W - LTO, LTO_W-FT (no router) - LTO-FT, composed LTO-FT - LTO.
 A2.2 (SM-R arms SMR_LTO / SMR_LTO_W, pooled 3 seeds, single checkpoint, no router): target = test_hpl_tamil vs LTO-FT pooled (U-WIN
        clauses); other 6 sets vs LTO: X-NI (AIoU lo > -0.010 every set) + order NI clauses reported; SOLVES = U-WIN AND X-NI.
 Zero-shot (descriptive): LTO mirror_x - LTO identity on test_hpl_tamil."""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--preds", required=True)
ap.add_argument("--router", required=True); ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--B", type=int, default=2000); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
SEEDS = (20261034, 20261035, 20261036); TARGET = "test_hpl_tamil"
OTHER = ["test_ouhdl", "test_casia_zh", "test_casia_en", "test_iam500", "german_unipen_de_test", "german_onhw_all"]
MET = ["aiou", "penup_f1_v2", "strict_order_acc", "direction_acc", "kendall_tau", "tau_cp"]
UWIN = {"aiou": 0.0, "penup_f1_v2": -0.0141, "strict_order_acc": -0.01065, "direction_acc": -0.01245}
XNI = -0.010

def load_sys(name):
    fs = [f for f in glob.glob(os.path.join(a.rows, name, "**", "*.csv"), recursive=True) if not os.path.basename(f).startswith("score_")]
    dfs = []
    for f in fs:
        df = pd.read_csv(f, low_memory=False)
        if "unit_id" in df and "protocol" in df: dfs.append(df)
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True); df = df[df["protocol"] == "native"].copy()
    df["unit_id"] = df["unit_id"].astype(str); df["frame"] = np.where(df["route_mode"].astype(str).str.endswith("mirror_x"), "mirror_x", "identity")
    return df

def per_unit(df, s, frame="identity"):
    d = df[(df["set"] == s) & (df["frame"] == frame)]
    if d.empty: return None
    cols = [c for c in MET if c in d and c != "tau_cp"] + ["n_gt_strokes"]
    g = d.groupby("unit_id", sort=True); agg = g[cols].mean(numeric_only=True); agg["_w"] = g["writer_id"].first().astype(str)
    agg["tau_cp"] = np.where(agg["n_gt_strokes"] >= 2, np.nan_to_num(agg["kendall_tau"].to_numpy(float), nan=0.0), np.nan)
    agg["_nseed"] = g["seed"].nunique() if "seed" in d else np.nan; agg["_files"] = g["preds_file"].nunique() if "preds_file" in d else np.nan
    return agg

def pooled(aggs):
    aggs = [x for x in aggs if x is not None]
    if not aggs: return None
    common = sorted(set.intersection(*[set(x.index) for x in aggs]))
    m = sum(x.loc[common, MET] for x in aggs) / len(aggs); m["_w"] = aggs[0].loc[common, "_w"]; m["_ntrain"] = len(aggs); return m

def devices(name):
    dv = set()
    for j in glob.glob(os.path.join(a.preds, name, "**", "*_preds.json"), recursive=True):
        try: dv.add(str(json.load(open(j)).get("device")))
        except Exception: pass
    return dv

DEV = {}
def dev_of(names): return sorted(set().union(*[DEV.setdefault(n, devices(n)) for n in names]))

def boot_ci(diff, w, s):
    v = np.asarray(diff, float); ok = ~np.isnan(v); v = v[ok]; w = np.asarray(w)[ok]
    if len(v) == 0: return np.nan, np.nan, np.nan, 0, 0
    if s == "german_onhw_all": wix = np.arange(len(v))
    else: _, wix = np.unique(w, return_inverse=True)
    nC = wix.max() + 1; C = np.random.default_rng(0).multinomial(nC, np.full(nC, 1.0 / nC), size=a.B).astype(float)
    S = np.bincount(wix, weights=v, minlength=nC); N = np.bincount(wix, minlength=nC).astype(float)
    num, den = C @ S, C @ N; bm = num[den > 0] / den[den > 0]; lo, hi = np.percentile(bm, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), int(len(v)), int(len(np.unique(w)))

def contrast(rule, cell, s, mname, M, cname, Cc, mdev, cdev):
    out = []
    if M is None or Cc is None:
        return [dict(rule=rule, cell=cell, set=s, method=mname, comparator=cname, metric=m, status="MISSING", evidence_label="NOT_ASSESSED") for m in MET]
    nt = lambda Z: int(Z["_ntrain"].iloc[0]) if "_ntrain" in Z.columns else 3   # backbones: no FT training seed
    lab = "CONFIRMATORY_3SEED" if min(nt(M), nt(Cc)) >= 3 else "SEED1_PLACEHOLDER"
    u = sorted(set(M.index) & set(Cc.index))
    for m in MET:
        d = M.loc[u, m].to_numpy(float) - Cc.loc[u, m].to_numpy(float)
        mean, lo, hi, n, nw = boot_ci(d, M.loc[u, "_w"].to_numpy(), s)
        out.append(dict(rule=rule, cell=cell, set=s, method=mname, comparator=cname, metric=m, n_units=n, n_writers=nw,
                        method_mean=float(np.nanmean(M.loc[u, m])), comp_mean=float(np.nanmean(Cc.loc[u, m])), diff=mean, ci_lo=lo, ci_hi=hi,
                        bootstrap="unit" if s == "german_onhw_all" else "writer-cluster", B=a.B,
                        method_devices=";".join(mdev), comp_devices=";".join(cdev), device_tag="(M)" if set(mdev) != set(cdev) else "", status="OK", evidence_label=lab,
                        n_train_seeds_method=nt(M), n_train_seeds_comp=nt(Cc)))
    return out

def compose(BB, FT, s):
    if BB is None or FT is None: return None
    man = os.path.join(a.router, f"manifest_{s}.csv")
    p = pd.read_csv(man, usecols=["unit_id", "p_hpl_tamil"]).astype({"unit_id": str}).set_index("unit_id")["p_hpl_tamil"]
    u = sorted(set(BB.index) & set(FT.index) & set(p.index)); route = (p.loc[u] >= 0.5).to_numpy()
    C = BB.loc[u, MET].copy(); C.loc[np.array(u)[route], MET] = FT.loc[np.array(u)[route], MET].to_numpy(); C["_w"] = BB.loc[u, "_w"]; C["_ntrain"] = FT["_ntrain"].iloc[0] if "_ntrain" in FT.columns else 3
    return C, int(route.sum()), len(u)

SYS = {n: load_sys(n) for n in ["LTO", "LTO_W"] + [f"{p}_s{k}" for p in ("LTO_FT", "LTO_W_FT", "SMR_LTO", "SMR_LTO_W") for k in SEEDS]}
SETS = [TARGET] + OTHER
def pu(n, s, fr="identity"): return per_unit(SYS[n], s, fr) if SYS.get(n) is not None else None
def pool(p, s): return pooled([pu(f"{p}_s{k}", s) for k in SEEDS])
def fam(p): return [f"{p}_s{k}" for k in SEEDS]
rows, absr, routes = [], [], []
for s in SETS:
    U = {"LTO": pu("LTO", s), "LTO_W": pu("LTO_W", s), "LTO_FT": pool("LTO_FT", s), "LTO_W_FT": pool("LTO_W_FT", s),
         "SMR_LTO": pool("SMR_LTO", s), "SMR_LTO_W": pool("SMR_LTO_W", s)}
    if s == TARGET: U["LTO_mirror_x"] = pu("LTO", s, "mirror_x")
    for k, v in U.items():
        if v is not None:
            ntr = int(v["_ntrain"].iloc[0]) if "_ntrain" in v.columns else 3
            absr.append(dict(set=s, system=k, n_units=len(v), n_train_seeds=ntr, evidence_label="CONFIRMATORY_3SEED" if ntr >= 3 else "SEED1_PLACEHOLDER", **{m: float(np.nanmean(v[m])) for m in MET}))
    dv = {"LTO": dev_of(["LTO"]), "LTO_W": dev_of(["LTO_W"]), "LTO_FT": dev_of(fam("LTO_FT")), "LTO_W_FT": dev_of(fam("LTO_W_FT")),
          "SMR_LTO": dev_of(fam("SMR_LTO")), "SMR_LTO_W": dev_of(fam("SMR_LTO_W"))}
    if s == TARGET:
        rows += contrast("U-WIN'", "T", s, "LTO_W-FT pooled", U["LTO_W_FT"], "LTO-FT pooled", U["LTO_FT"], dv["LTO_W_FT"], dv["LTO_FT"])
        for m in ("SMR_LTO", "SMR_LTO_W"):
            rows += contrast("A2.2 target", "T", s, f"{m} pooled", U[m], "LTO-FT pooled", U["LTO_FT"], dv[m], dv["LTO_FT"])
        rows += contrast("zero-shot (descriptive)", "T", s, "LTO mirror_x", U.get("LTO_mirror_x"), "LTO identity", U["LTO"], dv["LTO"], dv["LTO"])
        rows += contrast("reported", "T", s, "LTO_W backbone", U["LTO_W"], "LTO", U["LTO"], dv["LTO_W"], dv["LTO"])
    else:
        cw = compose(U["LTO_W"], U["LTO_W_FT"], s); cp = compose(U["LTO"], U["LTO_FT"], s)
        if cw: routes.append(dict(set=s, system="LTO_W composed", n_routed_ft=cw[1], n_units=cw[2], evidence_label="CONFIRMATORY_3SEED" if int(cw[0]["_ntrain"].iloc[0]) >= 3 else "SEED1_PLACEHOLDER"))
        rows += contrast("X-NI'", "X", s, "LTO_W composed", cw[0] if cw else None, "LTO", U["LTO"], sorted(set(dv["LTO_W"]) | set(dv["LTO_W_FT"])), dv["LTO"])
        rows += contrast("reported", "X", s, "LTO_W backbone", U["LTO_W"], "LTO", U["LTO"], dv["LTO_W"], dv["LTO"])
        rows += contrast("reported", "X", s, "LTO_W-FT pooled (no router)", U["LTO_W_FT"], "LTO-FT pooled (no router)", U["LTO_FT"], dv["LTO_W_FT"], dv["LTO_FT"])
        rows += contrast("reported", "X", s, "LTO-FT composed", cp[0] if cp else None, "LTO", U["LTO"], sorted(set(dv["LTO"]) | set(dv["LTO_FT"])), dv["LTO"])
        for m in ("SMR_LTO", "SMR_LTO_W"):
            rows += contrast("A2.2 X-NI", "X", s, f"{m} pooled", U[m], "LTO", U["LTO"], dv[m], dv["LTO"])
T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "R2d_tables_long.csv"), index=False)
pd.DataFrame(absr).to_csv(os.path.join(a.out, "R2d_absolute.csv"), index=False); pd.DataFrame(routes).to_csv(os.path.join(a.out, "R2d_routes.csv"), index=False)
def labs(rule, meth, sets):
    r = T[(T.rule == rule) & (T.method == meth) & (T.set.isin(sets))]
    L = set(r["evidence_label"]) if len(r) else {"NOT_ASSESSED"}
    return "NOT_ASSESSED" if "NOT_ASSESSED" in L else ("SEED1_PLACEHOLDER" if "SEED1_PLACEHOLDER" in L else "CONFIRMATORY_3SEED")
def lo(rule, meth, s, m):
    r = T[(T.rule == rule) & (T.method == meth) & (T.set == s) & (T.metric == m)]
    return None if r.empty or r.iloc[0].get("status") != "OK" else float(r.iloc[0]["ci_lo"])
def uwin(rule, meth):
    c = {m: lo(rule, meth, TARGET, m) for m in UWIN}
    return dict(clauses_ci_lo=c, result=None if None in c.values() else all(c[m] > UWIN[m] for m in UWIN), evidence_label=labs(rule, meth, [TARGET]))
def xni(rule, meth):
    c = {s: lo(rule, meth, s, "aiou") for s in OTHER}
    return dict(aiou_ci_lo=c, result=None if None in c.values() else all(v > XNI for v in c.values()), evidence_label=labs(rule, meth, OTHER))
res = {"U-WIN' (LTO_W-FT vs LTO-FT, test_hpl_tamil)": uwin("U-WIN'", "LTO_W-FT pooled"),
       "X-NI' (LTO_W composed vs LTO, 6 sets)": xni("X-NI'", "LTO_W composed")}
for m in ("SMR_LTO", "SMR_LTO_W"):
    u, x = uwin("A2.2 target", f"{m} pooled"), xni("A2.2 X-NI", f"{m} pooled")
    res[f"A2.2 {m}"] = dict(target_uwin=u, xni=x, SOLVES=None if None in (u["result"], x["result"]) else bool(u["result"] and x["result"]))
res["note"] = "result None = not assessed (missing system or set). evidence_label: CONFIRMATORY_3SEED only when all 3 FT training seeds are pooled; SEED1_PLACEHOLDER otherwise (seed 20261034 only); NOT_ASSESSED when a system is missing. Facts only."
json.dump(res, open(os.path.join(a.out, "R2d_rules.json"), "w"), indent=1, default=str)
print(json.dumps({k: (v.get("result", v.get("SOLVES")) if isinstance(v, dict) else v) for k, v in res.items()}))
