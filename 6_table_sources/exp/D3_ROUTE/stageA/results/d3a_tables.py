"""D3 Stage A table builder.  Inputs: per-unit rows from d3a_score.py (uhwr_eval 0.3.1), router manifests, preds index.
Definitions follow shared/table_builder/code_7d405fd6/build_tables.py: per-item mean over seeds first, then writer-cluster
percentile bootstrap (B=2000, numpy default_rng(0) multinomial writer counts, ratio of resampled sums); tau_cp = mean over
multi-stroke units (n_gt_strokes >= 2) of tau with undefined tau = 0; paired deltas = per-item difference on items in both.
python d3a_tables.py --rows ROWS_DIR --router ROUTER_DIR --out OUT [--boot 2000]
"""
import argparse, glob, hashlib, json, os, re, sys
import numpy as np, pandas as pd

METRICS = ["aiou", "ldtw", "dtw", "kendall_tau", "tau_cp", "direction_acc", "strict_order_acc", "penup_f1_v2"]
LABEL = {"aiou": "AIoU", "ldtw": "LDTW", "dtw": "DTW", "kendall_tau": "tau", "tau_cp": "tau_cp", "direction_acc": "direction",
         "strict_order_acc": "strict order", "penup_f1_v2": "pen-up F1 v2"}
SHAPE = ["aiou", "ldtw", "dtw", "penup_f1_v2"]; ORDER = ["kendall_tau", "tau_cp", "direction_acc", "strict_order_acc"]
SETS = ["test_ouhdl", "test_iam500", "test_casia_zh", "test_casia_en", "test_hpl_tamil", "german_unipen_de_test", "german_onhw_all"]
SETLAB = {"test_ouhdl": "OUHD-L Urdu test (4,406)", "test_iam500": "IAM English (500)", "test_casia_zh": "CASIA zh (500)",
          "test_casia_en": "CASIA en (500)", "test_hpl_tamil": "HPL Tamil (500)", "german_unipen_de_test": "UNIPEN-de zero-shot (827)",
          "german_onhw_all": "OnHW German zero-shot (5,925)"}
HOME = {"test_ouhdl": "E0_ms8", "test_iam500": "X5_ms9", "test_casia_zh": "E27_ms47", "test_casia_en": "X5_ms9",
        "test_hpl_tamil": "M5_joint_ms9", "german_unipen_de_test": "M5_joint_ms9", "german_onhw_all": "M5_joint_ms9"}
METHOD = {"pred_route": "routed_predicted", "oracle_route": "routed_oracle", "M5_joint_ms9": "joint_M5"}

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
    """g: rows of one (set, method, protocol) over seeds -> per-unit mean over seeds + derived tau_cp."""
    cols = [c for c in ["aiou", "ldtw", "dtw", "kendall_tau", "direction_acc", "strict_order_acc", "penup_f1_v2", "n_gt_strokes"] if c in g.columns]
    agg = g.groupby("unit_id", sort=True)[cols].mean(numeric_only=True)
    agg["_writer"] = g.groupby("unit_id", sort=True)["writer_id"].first().astype(str)
    agg["n_seeds"] = g.groupby("unit_id", sort=True)["seed"].nunique()
    multi = (agg["n_gt_strokes"] >= 2).to_numpy()
    t = agg["kendall_tau"].to_numpy(float)
    agg["tau_cp"] = np.where(multi, np.nan_to_num(t, nan=0.0), np.nan)
    return agg

def clusters(agg, B, by="writer"):
    key = agg["_writer"].to_numpy() if by == "writer" else agg.index.to_numpy().astype(str)
    uw, wix = np.unique(key, return_inverse=True)
    return (boot_counts(len(uw), B) if len(uw) > 1 else None), wix, len(uw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--router", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.rows, "*__rows031.csv")))
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df["unit_id"] = df["unit_id"].astype(str); df["error"] = df["error"].fillna("")
    print("rows", len(df), "files", len(files), "errors", int((df["error"] != "").sum()))
    # method label: routed / joint / specialist-<name>
    df["method"] = df["route_mode"].map(METHOD).fillna("specialist_" + df["route_mode"])
    aggs = {}; results = []; seedtab = []
    for (s, m, prot), g in df.groupby(["set", "method", "protocol"], sort=False):
        agg = per_item(g); aggs[(s, m, prot)] = agg
        C, wix, nw = clusters(agg, a.boot)
        row = dict(set=s, set_label=SETLAB.get(s, s), method=m, protocol=prot, n=len(agg), n_writers=nw,
                   n_seeds=int(agg["n_seeds"].max()), cluster="writer" if nw > 1 else "none")
        for met in METRICS:
            r = stat(agg[met].to_numpy(float), C, wix)
            row[met] = r["mean"]; row[met + "_lo"] = r["ci_lo"]; row[met + "_hi"] = r["ci_hi"]; row[met + "_ndef"] = r["n_defined"]
            # seed spread of the set mean
            sm = g.groupby("seed")[met].mean() if met != "tau_cp" else g.assign(_t=np.where(g["n_gt_strokes"] >= 2, g["kendall_tau"].fillna(0.0), np.nan)).groupby("seed")["_t"].mean()
            row[met + "_seed_sd"] = float(sm.std(ddof=1)) if len(sm) > 1 else np.nan
            for sd, v in sm.items(): seedtab.append(dict(set=s, method=m, protocol=prot, metric=met, seed=int(sd), mean=float(v)))
        row["n_multi_stroke"] = int((agg["n_gt_strokes"] >= 2).sum())
        row["files"] = ";".join(sorted(g["preds_file"].unique())); row["preds_sha256"] = ";".join(sorted(g["preds_sha256"].unique()))
        row["n_error_rows"] = int((g["error"] != "").sum())
        results.append(row)
    res = pd.DataFrame(results)
    order = {m: i for i, m in enumerate(["routed_predicted", "routed_oracle", "joint_M5"])}
    res["_o"] = res["method"].map(lambda m: order.get(m, 9)); res["_s"] = res["set"].map(lambda s: SETS.index(s) if s in SETS else 99)
    res = res.sort_values(["protocol", "_s", "_o", "method"], ascending=[False, True, True, True]).drop(columns=["_o", "_s"])
    res.to_csv(os.path.join(a.out, "results_D3A_crossscript.csv"), index=False)
    pd.DataFrame(seedtab).to_csv(os.path.join(a.out, "results_D3A_per_seed.csv"), index=False)
    # paired deltas
    pairs = [("routed_predicted", "joint_M5", "routed_predicted - joint_M5"), ("routed_oracle", "joint_M5", "routed_oracle - joint_M5"),
             ("routed_predicted", "routed_oracle", "routed_predicted - routed_oracle (cost of mis-routing)")]
    prow = []
    for s in SETS:
        for prot in ("native", "standardised"):
            extra = [(m, "joint_M5", f"{m} - joint_M5") for m in sorted({k[1] for k in aggs if k[0] == s and k[1].startswith("specialist_")})]
            for A_, B_, lab in pairs + extra:
                if (s, A_, prot) not in aggs or (s, B_, prot) not in aggs: continue
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
                                     cluster_note=("writer-cluster CI degenerate (2 writers); use unit-cluster CI" if len(uw) <= 2 else "writer-cluster"),
                                     frac_pos=float((d > 0).mean()), frac_neg=float((d < 0).mean())))
    paired = pd.DataFrame(prow); paired.to_csv(os.path.join(a.out, "paired_D3A.csv"), index=False)
    # routing accuracy
    rr = []
    for s in SETS:
        mf = pd.read_csv(os.path.join(a.router, f"manifest_{s}.csv")); mf["unit_id"] = mf["unit_id"].astype(str)
        # route actually used in the sampled pred_route file (seed 0) must equal the manifest
        used = df[(df["set"] == s) & (df["route_mode"] == "pred_route") & (df["seed"] == 0) & (df["protocol"] == "native")].set_index("unit_id")["specialist"]
        agree = float((mf.set_index("unit_id")["pred_specialist"].reindex(used.index) == used).mean())
        shares = mf["pred_specialist"].value_counts(normalize=True)
        row = dict(set=s, set_label=SETLAB[s], n=len(mf), home_specialist=HOME[s], oracle_specialist=mf["oracle_specialist"].iloc[0],
                   router_accuracy_vs_oracle=float((mf["pred_specialist"] == mf["oracle_specialist"]).mean()),
                   script_accuracy=float((mf["pred_script"] == mf["true_script"]).mean()) if not s.startswith("german") else np.nan,
                   family_accuracy=float((mf["pred_family"] == np.where(mf["true_script"] == "ouhdl", "RTL", "LTR")).mean()) if not s.startswith("german") else np.nan,
                   manifest_vs_sampled_route_agreement=agree, n_misrouted=int((mf["pred_specialist"] != mf["oracle_specialist"]).sum()))
        for sp in ["E0_ms8", "E27_ms47", "X5_ms9", "M5_joint_ms9"]:
            row["share_" + sp] = float(shares.get(sp, 0.0)); row["n_" + sp] = int((mf["pred_specialist"] == sp).sum())
        rr.append(row)
    routing = pd.DataFrame(rr); routing.to_csv(os.path.join(a.out, "routing_accuracy_D3A.csv"), index=False)
    # mis-routed subset: paired on the misrouted units only (native AIoU / tau), pred vs oracle
    mis = []
    for s in SETS:
        mf = pd.read_csv(os.path.join(a.router, f"manifest_{s}.csv")); mf["unit_id"] = mf["unit_id"].astype(str)
        ids = set(mf.loc[mf["pred_specialist"] != mf["oracle_specialist"], "unit_id"])
        if not ids or (s, "routed_predicted", "native") not in aggs: continue
        a_, b_ = aggs[(s, "routed_predicted", "native")], aggs[(s, "routed_oracle", "native")]
        for met in ["aiou", "kendall_tau", "direction_acc", "strict_order_acc"]:
            j = a_[[met]].join(b_[[met]], how="inner", rsuffix="_ref"); j = j[j.index.isin(ids)].dropna()
            if len(j) == 0: continue
            d = (j[met] - j[met + "_ref"]).to_numpy(float)
            mis.append(dict(set=s, metric=met, n_misrouted_scored=len(j), mean_pred=float(j[met].mean()), mean_oracle=float(j[met + "_ref"].mean()), delta=float(d.mean())))
    pd.DataFrame(mis).to_csv(os.path.join(a.out, "misrouted_units_D3A.csv"), index=False)
    # dedup / identity bookkeeping
    ded = df[df["protocol"] == "native"].groupby(["set", "route_mode", "seed"]).agg(n=("unit_id", "size"), n_dedup=("dedup_from", lambda x: int((x.fillna("") != "").sum())),
                                                                              dedup_from=("dedup_from", lambda x: ";".join(sorted(set(x.fillna("")) - {""})))).reset_index()
    ded.to_csv(os.path.join(a.out, "dedup_identity_D3A.csv"), index=False)
    print(res[(res.protocol == "native")][["set", "method", "n", "aiou", "kendall_tau"]].to_string())

if __name__ == "__main__":
    main()
