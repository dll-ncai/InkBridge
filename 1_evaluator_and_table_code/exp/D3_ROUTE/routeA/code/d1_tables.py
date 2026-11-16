"""D1 (token-routed joint diffusion) table builder - derived from d2_tables.py (same statistics). Loads the D1 score-chain rows
(<set>__<method>__s<seed>_rows.csv, method = <set>__ms<ms>__<oracle|unknown|predicted> or <set>__<e0_ms8|x5_ms9|e27_ms47|m5joint_ms9>),
renames sets to the Stage A names and methods to D1_ms<ms>_<mode> / E0_ms8 / X5_ms9 / E27_ms47 / M5_joint_ms9.
Original: D2 (adapter-MoE) table builder: Stage 1 non-regression on the frozen scripts, Stage 2 Urdu fraction curve vs LUO-FT / E0-from-scratch /
joint / E0 ms8, zero-shot German, gate-weight matrix.  Statistics = routeA_tables.py / d3a_tables.py (per-unit seed mean, writer-cluster
percentile bootstrap B=2000, tau_cp with undefined tau = 0 on multi-stroke units, paired per-unit deltas).
python d2_tables.py --rows LABEL=DIR[:SUFFIX] [--rows ...] --gate NAME=RUN_DIR [...] --out OUT [--boot 2000]
  SUFFIX is appended to every route_mode loaded from that DIR (used to keep the Stage A 4090 comparators apart from the CLUSTER-A ones).
"""
import argparse, glob, hashlib, json, os, re, time
import numpy as np, pandas as pd

METRICS = ["aiou", "ldtw", "dtw", "kendall_tau", "tau_cp", "direction_acc", "strict_order_acc", "penup_f1_v2"]
LABEL = {"aiou": "AIoU", "ldtw": "LDTW", "dtw": "DTW", "kendall_tau": "tau", "tau_cp": "tau_cp", "direction_acc": "direction",
         "strict_order_acc": "strict order", "penup_f1_v2": "pen-up F1 v2"}
SETS = ["val_casia_zh", "val_casia_en", "val_hpl_tamil", "val_iam", "val_ouhdl", "test_casia_zh", "test_casia_en", "test_hpl_tamil", "test_iam500",
        "test_ouhdl", "gold_ouhdl", "german_unipen_de_test", "german_onhw_all"]
SPEC = {"test_ouhdl": "E0_ms8", "gold_ouhdl": "E0_ms8", "test_iam500": "X5_ms9", "test_casia_zh": "E27_ms47", "test_casia_en": "X5_ms9", "val_ouhdl": "E0_ms8"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def boot_counts(n_w, B, seed=0):
    return np.random.default_rng(seed).multinomial(n_w, np.full(n_w, 1.0 / n_w), size=B).astype(float)
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
    t = agg["kendall_tau"].to_numpy(float); multi = (agg["n_gt_strokes"] >= 2).to_numpy()
    agg["tau_cp"] = np.where(multi, np.nan_to_num(t, nan=0.0), np.nan)
    return agg
D1SET = {"ouhdl_test": "test_ouhdl", "casia_zh500": "test_casia_zh", "casia_en500": "test_casia_en", "hpl_tamil500": "test_hpl_tamil", "iam500": "test_iam500",
         "unipen_de_test": "german_unipen_de_test", "onhw_all": "german_onhw_all"}
D1CMP = {"e0_ms8": "E0_ms8", "x5_ms9": "X5_ms9", "e27_ms47": "E27_ms47", "m5joint_ms9": "M5_joint_ms9"}
def load(spec):
    """spec = label=dir[:suffix]; dir holds D1-chain rows (<set>__<method>__s<seed>_rows.csv) or lane rows (*__rows031.csv)."""
    label, rest = spec.split("=", 1); d, suf = (rest.split(":", 1) + [""])[:2]
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*_rows.csv")) + glob.glob(os.path.join(d, "*__rows031.csv"))):
        b = os.path.basename(f); df = pd.read_csv(f, low_memory=False)
        if b.endswith("__rows031.csv"):
            if str(df["route_mode"].iloc[0]) in ("pred_route", "oracle_route"): continue
        else:
            m = re.fullmatch(r"(.+?)__(.+)__s(\d)_rows\.csv", b)
            if not m: continue
            s_raw, meth, sd = m.group(1), m.group(2), int(m.group(3))
            if s_raw == "val":   # val__<script>__ms<ms>__<mode>__s<seed>
                sc, meth = meth.split("__", 1); s_raw = "val_" + sc
            mm = re.fullmatch(r"ms(\d+)__(oracle|unknown|predicted)", meth)
            rm = f"D1_ms{int(mm.group(1))}_{mm.group(2)}" if mm else D1CMP.get(meth, meth)
            df["set"] = D1SET.get(s_raw, s_raw); df["route_mode"] = rm; df["seed"] = sd
            df["preds_file"] = b.replace("_rows.csv", "_preds.npz"); df["preds_sha256"] = ""
            if "error" not in df.columns: df["error"] = ""
        df["route_mode"] = df["route_mode"].astype(str) + suf; df["source"] = label; df["rows_file"] = b; out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

SPECD1 = {"test_ouhdl": "E0_ms8", "test_iam500": "X5_ms9", "test_casia_zh": "E27_ms47", "test_casia_en": "X5_ms9"}
def comparators_for(m, av, s=None):
    cands = []
    def anyof(pat): return sorted(x for x in av if re.fullmatch(pat, x))
    mm = re.fullmatch(r"D1_ms(\d+)_(oracle|unknown|predicted)(_\w+)?", m)
    if mm:
        ms, mode = mm.group(1), mm.group(2)
        cands += anyof(r"M5_joint_ms9(_\w+)?")
        if s in SPECD1: cands += anyof(SPECD1[s] + r"(_\w+)?")
        if mode == "oracle": cands += anyof(rf"D1_ms{ms}_unknown") + anyof(rf"D1_ms{ms}_predicted")
        if mode == "predicted": cands += anyof(rf"D1_ms{ms}_unknown")
        if mode == "oracle" and not (s or "").startswith("val"): cands += [x for x in anyof(r"D1_ms\d+_oracle") if x != m]   # script-best vs global milestone (test sets)
    return [c_ for c_ in cands if c_ != m and c_ in av]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", action="append", required=True); ap.add_argument("--gate", action="append", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    df = pd.concat([x for x in (load(s) for s in a.rows) if len(x)], ignore_index=True)
    df["unit_id"] = df["unit_id"].astype(str); df["error"] = df["error"].fillna("")
    print("rows", len(df), "errors", int((df["error"] != "").sum()), flush=True)
    aggs = {}; results = []; seedtab = []
    for (s, m, prot), g in df.groupby(["set", "route_mode", "protocol"], sort=False):
        agg = per_item(g); aggs[(s, m, prot)] = agg
        uw, wix = np.unique(agg["_writer"].to_numpy(), return_inverse=True); nw = len(uw); C = boot_counts(nw, a.boot) if nw > 1 else None
        row = dict(set=s, method=m, source=g["source"].iloc[0], protocol=prot, n=len(agg), n_writers=nw, n_seeds=int(agg["n_seeds"].max()), cluster="writer" if nw > 1 else "none")
        for met in METRICS:
            r = stat(agg[met].to_numpy(float), C, wix)
            row[met] = r["mean"]; row[met + "_lo"] = r["ci_lo"]; row[met + "_hi"] = r["ci_hi"]; row[met + "_ndef"] = r["n_defined"]
            if met == "tau_cp": sm = g.assign(_t=np.where(g["n_gt_strokes"] >= 2, g["kendall_tau"].fillna(0.0), np.nan)).groupby("seed")["_t"].mean()
            elif met in g.columns: sm = g.groupby("seed")[met].mean()
            else: sm = pd.Series(dtype=float)
            row[met + "_seed_sd"] = float(sm.std(ddof=1)) if len(sm) > 1 else np.nan
            for sd, v in sm.items(): seedtab.append(dict(set=s, method=m, protocol=prot, metric=met, seed=int(sd), mean=float(v)))
        row["n_multi_stroke"] = int((agg["n_gt_strokes"] >= 2).sum()); row["files"] = ";".join(sorted(g["preds_file"].astype(str).unique()))
        row["preds_sha256"] = ";".join(sorted(g["preds_sha256"].astype(str).unique())); row["n_error_rows"] = int((g["error"] != "").sum())
        results.append(row)
    res = pd.DataFrame(results); res["_s"] = res["set"].map(lambda s: SETS.index(s) if s in SETS else 99)
    res = res.sort_values(["protocol", "_s", "method"], ascending=[False, True, True]).drop(columns=["_s"])
    res.to_csv(os.path.join(a.out, "results_D1.csv"), index=False); pd.DataFrame(seedtab).to_csv(os.path.join(a.out, "results_D1_per_seed.csv"), index=False)
    prow = []
    for (s, prot) in sorted({(k[0], k[2]) for k in aggs}):
        av = {k[1] for k in aggs if k[0] == s and k[2] == prot}
        for A_ in sorted(av):
            for B_ in comparators_for(A_, av, s):
                a_, b_ = aggs[(s, A_, prot)], aggs[(s, B_, prot)]
                for met in METRICS:
                    j = a_[[met, "_writer"]].join(b_[[met]], how="inner", rsuffix="_ref").dropna(subset=[met, met + "_ref"])
                    if len(j) == 0: continue
                    d = (j[met] - j[met + "_ref"]).to_numpy(float); uw, wix = np.unique(j["_writer"].to_numpy(), return_inverse=True)
                    rw = stat(d, boot_counts(len(uw), a.boot) if len(uw) > 1 else None, wix); ru = stat(d, boot_counts(len(d), a.boot), np.arange(len(d)))
                    lo, hi = (rw["ci_lo"], rw["ci_hi"]) if len(uw) > 2 else (ru["ci_lo"], ru["ci_hi"])
                    prow.append(dict(set=s, protocol=prot, contrast=f"{A_} - {B_}", method=A_, comparator=B_, metric=met, metric_label=LABEL[met], delta=rw["mean"],
                                     ci_lo_writer=rw["ci_lo"], ci_hi_writer=rw["ci_hi"], n_writers=len(uw), ci_lo_unit=ru["ci_lo"], ci_hi_unit=ru["ci_hi"], n_pairs=len(d),
                                     cluster_note=("unit-cluster CI (<=2 writers)" if len(uw) <= 2 else "writer-cluster"), frac_pos=float((d > 0).mean()), frac_neg=float((d < 0).mean()),
                                     ci_excludes_zero=bool(lo > 0 or hi < 0), non_inferior_0p010=bool(lo > -0.010) if met not in ("ldtw", "dtw") else bool(hi < 0.010)))
    pd.DataFrame(prow).to_csv(os.path.join(a.out, "paired_D1.csv"), index=False)
    # gate matrices
    grow = []
    for spec in a.gate:
        name, rd = spec.split("=", 1); fs = sorted(glob.glob(os.path.join(rd, "gate_weights", "step_*.json")))
        if not fs: continue
        g = json.load(open(fs[-1]))
        for sc, v in g["per_script"].items():
            for e in g["experts"]:
                grow.append(dict(run=name, step=g["step"], script=sc, n=v["n"], expert=e, mean_gate=v["mean_gate"][e], argmax_share=v["argmax_hist"].get(e, 0) / max(v["n"], 1), val_den=v.get("val_den")))
    pd.DataFrame(grow).to_csv(os.path.join(a.out, "gate_matrix_D1_unused.csv"), index=False)
    inputs = df[df["protocol"] == "native"].groupby(["set", "route_mode", "seed", "source"]).agg(n=("unit_id", "size")).reset_index()
    inputs.to_csv(os.path.join(a.out, "inputs_D1.csv"), index=False)
    json.dump(dict(built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), boot=a.boot, rows=a.rows, gate=a.gate,
                   n_rows=int(len(df)), rows_files=sorted(set(df["rows_file"])), code_sha256=sha(os.path.abspath(__file__)), seconds=round(time.time() - t0, 1)),
              open(os.path.join(a.out, "PROVENANCE_D1_tables.json"), "w"), indent=1)
    print(res[res.protocol == "native"][["set", "method", "n", "aiou", "kendall_tau", "ldtw"]].to_string(), flush=True)

if __name__ == "__main__":
    main()
