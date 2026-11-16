"""Route A (cost of sharing) table builder.  Reuses the D3 Stage A definitions (d3a_tables.py, exp/D3_ROUTE/stageA/results/):
per-unit mean over seeds first, then writer-cluster percentile bootstrap (B=2000, numpy default_rng(0) multinomial writer counts,
ratio of resampled sums); tau_cp = mean over multi-stroke units (n_gt_strokes >= 2) of tau with undefined tau = 0;
paired deltas = per-unit difference on units present in both methods (after the per-unit seed mean).

Inputs (per-unit rows, uhwr_eval 0.3.1 unless stated):
  --rows-routea  exp/D3_ROUTE/routeA/score_out/rows      LUO_ms6, FT_p01_ms4, FT_p10_ms10, FT_p30_ms10, FT_p100_ms9 (VPS RTX 4090, seeds 0-2)
  --rows-stagea  exp/D3_ROUTE/stageA/score_out/rows      E0_ms8, X5_ms9, E27_ms47, M5_joint_ms9 (VPS RTX 4090, seeds 0-2); pred/oracle_route ignored
  --rows-rb4     exp/RB4_baselines/german/rows            e0ms8/x5ms9/jointms9/luoms6 on UNIPEN-de / OnHW (CLUSTER-A A30, seed 0, uhwr_eval 0.1.0-equivalent rows: no penup_f1_v2)
  --rows-ctrl    (optional) E0-from-scratch controls + E0 ms8 on gold, sampled on CLUSTER-A (route_mode e.g. E0scratch_p01_ms10)
python routeA_tables.py --rows-routea R --rows-stagea S --rows-rb4 G --out OUT [--rows-ctrl C] [--boot 2000]
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd

METRICS = ["aiou", "ldtw", "dtw", "kendall_tau", "tau_cp", "direction_acc", "strict_order_acc", "penup_f1_v2"]
LABEL = {"aiou": "AIoU", "ldtw": "LDTW", "dtw": "DTW", "kendall_tau": "tau", "tau_cp": "tau_cp", "direction_acc": "direction",
         "strict_order_acc": "strict order", "penup_f1_v2": "pen-up F1 v2"}
SETS = ["test_ouhdl", "gold_ouhdl", "test_iam500", "test_casia_zh", "test_casia_en", "test_hpl_tamil", "german_unipen_de_test", "german_onhw_all"]
SETLAB = {"test_ouhdl": "OUHD-L Urdu test (4,406)", "gold_ouhdl": "OUHD-L Urdu gold (354)", "test_iam500": "IAM English (500)",
          "test_casia_zh": "CASIA zh (500)", "test_casia_en": "CASIA en (500)", "test_hpl_tamil": "HPL Tamil (500)",
          "german_unipen_de_test": "UNIPEN-de zero-shot (827)", "german_onhw_all": "OnHW German zero-shot (5,925)"}
SPECIALIST = {"test_ouhdl": "E0_ms8", "gold_ouhdl": "E0_ms8", "test_iam500": "X5_ms9", "test_casia_zh": "E27_ms47", "test_casia_en": "X5_ms9"}
FT = ["FT_p01_ms4", "FT_p10_ms10", "FT_p30_ms10", "FT_p100_ms9"]
RB4MAP = {"e0ms8": "E0_ms8", "x5ms9": "X5_ms9", "jointms9": "M5_joint_ms9", "luoms6": "LUO_ms6"}
RB4SET = {"unipen_de_test": "german_unipen_de_test", "onhw_all": "german_onhw_all"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def boot_counts(n_w, B, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_w, np.full(n_w, 1.0 / n_w), size=B).astype(float)

def stat(values, C, wix):
    v = np.asarray(values, float); ok = ~np.isnan(v); n_def = int(ok.sum())
    if n_def == 0: return dict(mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_defined=0)
    m = float(v[ok].mean())
    if C is None: return dict(mean=m, ci_lo=np.nan, ci_hi=np.nan, n_defined=n_def)
    S = np.bincount(wix[ok], weights=v[ok], minlength=C.shape[1]); N = np.bincount(wix[ok], minlength=C.shape[1]).astype(float)
    num, den = C @ S, C @ N; bm = num[den > 0] / den[den > 0]
    lo, hi = np.percentile(bm, [2.5, 97.5]) if len(bm) else (np.nan, np.nan)
    return dict(mean=m, ci_lo=float(lo), ci_hi=float(hi), n_defined=n_def)

def per_item(g):
    cols = [c for c in ["aiou", "ldtw", "dtw", "kendall_tau", "direction_acc", "strict_order_acc", "penup_f1_v2", "n_gt_strokes"] if c in g.columns]
    agg = g.groupby("unit_id", sort=True)[cols].mean(numeric_only=True)
    if "penup_f1_v2" not in agg.columns: agg["penup_f1_v2"] = np.nan
    agg["_writer"] = g.groupby("unit_id", sort=True)["writer_id"].first().astype(str)
    agg["n_seeds"] = g.groupby("unit_id", sort=True)["seed"].nunique()
    multi = (agg["n_gt_strokes"] >= 2).to_numpy()
    t = agg["kendall_tau"].to_numpy(float)
    agg["tau_cp"] = np.where(multi, np.nan_to_num(t, nan=0.0), np.nan)
    return agg

def load_rows(d, source, keep=None, drop=None):
    files = sorted(glob.glob(os.path.join(d, "*__rows031.csv")))
    out = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        rm = str(df["route_mode"].iloc[0])
        if keep is not None and rm not in keep: continue
        if drop is not None and rm in drop: continue
        df["source"] = source; df["rows_file"] = os.path.basename(f); out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def load_rb4(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*_rows.csv"))):
        b = os.path.basename(f).replace("_rows.csv", ""); m, s = b.split("_", 1)
        if m not in RB4MAP: continue
        df = pd.read_csv(f, low_memory=False)
        df["set"] = RB4SET[s]; df["route_mode"] = RB4MAP[m] + "_rb4"; df["seed"] = 0; df["source"] = "RB4 (CLUSTER-A A30, seed 0)"
        df["preds_file"] = m + "_" + s + "_preds.npz"; df["preds_sha256"] = ""; df["rows_file"] = os.path.basename(f); df["error"] = df.get("error", "")
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-routea", required=True); ap.add_argument("--rows-stagea", required=True); ap.add_argument("--rows-rb4", required=True)
    ap.add_argument("--rows-ctrl", default=""); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    parts = [load_rows(a.rows_routea, "RouteA (VPS RTX 4090, seeds 0-2)"),
             load_rows(a.rows_stagea, "StageA (VPS RTX 4090, seeds 0-2)", drop={"pred_route", "oracle_route"}),
             load_rb4(a.rows_rb4)]
    if a.rows_ctrl and os.path.isdir(a.rows_ctrl): parts.append(load_rows(a.rows_ctrl, "controls (CLUSTER-A GPU, seeds 0-2)"))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df["unit_id"] = df["unit_id"].astype(str); df["error"] = df["error"].fillna("")
    print("rows", len(df), "errors", int((df["error"] != "").sum()), flush=True)
    aggs = {}; results = []; seedtab = []
    for (s, m, prot), g in df.groupby(["set", "route_mode", "protocol"], sort=False):
        agg = per_item(g); aggs[(s, m, prot)] = agg
        key = agg["_writer"].to_numpy(); uw, wix = np.unique(key, return_inverse=True); nw = len(uw)
        C = boot_counts(nw, a.boot) if nw > 1 else None
        row = dict(set=s, set_label=SETLAB.get(s, s), method=m, source=g["source"].iloc[0], protocol=prot, n=len(agg), n_writers=nw,
                   n_seeds=int(agg["n_seeds"].max()), cluster="writer" if nw > 1 else "none")
        for met in METRICS:
            r = stat(agg[met].to_numpy(float), C, wix)
            row[met] = r["mean"]; row[met + "_lo"] = r["ci_lo"]; row[met + "_hi"] = r["ci_hi"]; row[met + "_ndef"] = r["n_defined"]
            if met == "tau_cp":
                sm = g.assign(_t=np.where(g["n_gt_strokes"] >= 2, g["kendall_tau"].fillna(0.0), np.nan)).groupby("seed")["_t"].mean()
            elif met in g.columns: sm = g.groupby("seed")[met].mean()
            else: sm = pd.Series(dtype=float)
            row[met + "_seed_sd"] = float(sm.std(ddof=1)) if len(sm) > 1 else np.nan
            for sd, v in sm.items(): seedtab.append(dict(set=s, method=m, protocol=prot, metric=met, seed=int(sd), mean=float(v)))
        row["n_multi_stroke"] = int((agg["n_gt_strokes"] >= 2).sum())
        row["files"] = ";".join(sorted(g["preds_file"].astype(str).unique())); row["preds_sha256"] = ";".join(sorted(g["preds_sha256"].astype(str).unique()))
        row["n_error_rows"] = int((g["error"] != "").sum())
        results.append(row)
    res = pd.DataFrame(results)
    morder = ["E0_ms8", "X5_ms9", "E27_ms47", "M5_joint_ms9", "LUO_ms6"] + FT
    res["_o"] = res["method"].map(lambda m: morder.index(m) if m in morder else (50 if m.endswith("_rb4") else 30))
    res["_s"] = res["set"].map(lambda s: SETS.index(s) if s in SETS else 99)
    res = res.sort_values(["protocol", "_s", "_o", "method"], ascending=[False, True, True, True]).drop(columns=["_o", "_s"])
    res.to_csv(os.path.join(a.out, "results_routeA_sharing_cost.csv"), index=False)
    pd.DataFrame(seedtab).to_csv(os.path.join(a.out, "results_routeA_per_seed.csv"), index=False)
    # contrasts
    def avail(s, prot): return {k[1] for k in aggs if k[0] == s and k[2] == prot}
    prow = []
    for s in SETS:
        for prot in ("native", "standardised"):
            av = avail(s, prot); spec = SPECIALIST.get(s); pairs = []
            if spec and spec in av:
                pairs += [("M5_joint_ms9", spec, "joint - specialist (cost of sharing)"), ("LUO_ms6", spec, "LUO - specialist (cost of sharing, Urdu left out)")]
            pairs += [("LUO_ms6", "M5_joint_ms9", "LUO - joint (effect of Urdu in the joint mix)")]
            for ft in FT:
                pairs += [(ft, "E0_ms8", f"{ft} - E0_ms8 (FT curve vs E0 full data)"), (ft, "M5_joint_ms9", f"{ft} - joint"), (ft, "LUO_ms6", f"{ft} - LUO (Urdu FT effect)")]
                ctrls = sorted(m for m in av if m.startswith("E0scratch_") and m.split("_")[1] == ft.split("_")[1])
                for cm in ctrls: pairs += [(ft, cm, f"{ft} - {cm} (FT vs from-scratch at same fraction)")]
                prev = FT[FT.index(ft) - 1] if FT.index(ft) > 0 else None
                if prev: pairs += [(ft, prev, f"{ft} - {prev} (curve step)")]
            for cm in sorted(m for m in av if m.startswith("E0scratch_")):
                pairs += [(cm, "E0_ms8", f"{cm} - E0_ms8")]
            if s.startswith("german"):
                pairs += [("X5_ms9_rb4", "M5_joint_ms9_rb4", "X5 - joint (RB4 rows, A30 seed 0)"), ("E0_ms8_rb4", "M5_joint_ms9_rb4", "E0 - joint (RB4 rows, A30 seed 0)"),
                          ("LUO_ms6_rb4", "M5_joint_ms9_rb4", "LUO - joint (RB4 rows, A30 seed 0)"), ("LUO_ms6", "LUO_ms6_rb4", "LUO 4090 3-seed - LUO RB4 A30 seed0 (device check)"),
                          ("M5_joint_ms9", "M5_joint_ms9_rb4", "joint 4090 3-seed - joint RB4 A30 seed0 (device check)")]
            for A_, B_, lab in pairs:
                if A_ not in av or B_ not in av: continue
                a_, b_ = aggs[(s, A_, prot)], aggs[(s, B_, prot)]
                for met in METRICS:
                    j = a_[[met, "_writer"]].join(b_[[met]], how="inner", rsuffix="_ref").dropna(subset=[met, met + "_ref"])
                    if len(j) == 0: continue
                    d = (j[met] - j[met + "_ref"]).to_numpy(float)
                    uw, wix = np.unique(j["_writer"].to_numpy(), return_inverse=True)
                    rw = stat(d, boot_counts(len(uw), a.boot) if len(uw) > 1 else None, wix)
                    ru = stat(d, boot_counts(len(d), a.boot), np.arange(len(d)))
                    prow.append(dict(set=s, set_label=SETLAB[s], protocol=prot, contrast=lab, method=A_, comparator=B_, metric=met, metric_label=LABEL[met],
                                     delta=rw["mean"], ci_lo_writer=rw["ci_lo"], ci_hi_writer=rw["ci_hi"], n_writers=len(uw),
                                     ci_lo_unit=ru["ci_lo"], ci_hi_unit=ru["ci_hi"], n_pairs=len(d),
                                     cluster_note=("writer-cluster CI degenerate (<=2 writers); use unit-cluster CI" if len(uw) <= 2 else "writer-cluster"),
                                     frac_pos=float((d > 0).mean()), frac_neg=float((d < 0).mean()),
                                     ci_excludes_zero_writer=bool((rw["ci_lo"] > 0) or (rw["ci_hi"] < 0)) if len(uw) > 2 else bool((ru["ci_lo"] > 0) or (ru["ci_hi"] < 0))))
    paired = pd.DataFrame(prow); paired.to_csv(os.path.join(a.out, "paired_routeA.csv"), index=False)
    # bookkeeping
    ded = df[df["protocol"] == "native"].groupby(["set", "route_mode", "seed", "source"]).agg(n=("unit_id", "size")).reset_index()
    ded.to_csv(os.path.join(a.out, "inputs_routeA.csv"), index=False)
    rows_files = sorted(set(df["rows_file"]))
    json.dump(dict(built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
                   boot=a.boot, rows_routea=a.rows_routea, rows_stagea=a.rows_stagea, rows_rb4=a.rows_rb4, rows_ctrl=a.rows_ctrl,
                   n_rows=int(len(df)), rows_files=rows_files, code_sha256=sha(os.path.abspath(__file__)),
                   pandas=pd.__version__, numpy=np.__version__, seconds=round(time.time() - t0, 1)),
              open(os.path.join(a.out, "PROVENANCE_routeA_tables.json"), "w"), indent=1)
    print(res[res.protocol == "native"][["set", "method", "n", "aiou", "kendall_tau", "ldtw"]].to_string(), flush=True)

if __name__ == "__main__":
    main()
