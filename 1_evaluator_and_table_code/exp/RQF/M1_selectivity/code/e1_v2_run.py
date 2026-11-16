"""E1-v2: protocol sensitivity with the redesigned pen-up and timing components (uhwr_eval 0.3.0).

python e1_v2_run.py --bundle B --out OUT [--datasets ouhdl casia tamil] [--workers N] [--limit K] [--boot 2000]

Perturbations: identical to E1 v1 (uhwr_exp.e1_run.perturb / FACTORS, seed 20261016, per-unit rng
[seed, crc32(uid)]), so every perturbed ink is the same as in v1.  The 7 perturbation types form the 5 factors of the
plan: shape (shape_jitter, shape_elastic), order, direction, pen-up (penup_merge, penup_split), timing; 5 strengths
each, plus the identity copy.  Every uhwr_eval 0.3.0 metric is scored (0.2.0 metrics unchanged + penup_f1_v2 +
timing_profile_tv_v2), native and standardised protocols, LPIPS on.
Units: e_bundle_v1/e1_units.pkl (the v1 units: 500 OUHD-L ligatures with timestamps, 500 CASIA units without
timestamps, 500 HPL Tamil units with timestamps).  IAM components are NOT in the bundle (they are an output of X3).
Sensitivity matrix (e1_v2_sensitivity.csv): per dataset x protocol x metric x perturbation: mean at strengths 0..5,
delta5 = mean(s5) - mean(s0) with a cluster bootstrap 95% CI (cluster = writer/file as in the bundle), Spearman rho
of metric vs strength over unit x strength rows, and selectivity = |delta5| / max over perturbations of |delta5|.
"""
import argparse, json, os, pickle, time, zlib
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("UHWR_TORCH_THREADS", "1")


def run_unit(args):
    ds, u, seed = args
    from uhwr_exp.e1_run import perturb, FACTORS
    from uhwr_eval import score
    rows = []
    canvas = (64, 64)
    gt, gt_t = u["ink"], u["t"]
    rng = np.random.default_rng([seed, zlib.crc32(str(u["uid"]).encode())])
    jobs = [("identity", 0, 0, gt, gt_t, 0)]
    for f, ks in FACTORS.items():
        for si, k in enumerate(ks, 1):
            ink, t, eff = perturb(gt, gt_t, f, k, rng)
            jobs.append((f, si, k, ink, t, eff))
    for f, si, k, ink, t, eff in jobs:
        for prot in ("native", "standardised"):
            try:
                r = score(gt, ink, canvas, protocol=prot, gt_times=gt_t, pred_times=t, with_lpips=True); err = ""
            except Exception as e:
                r, err = {}, f"{type(e).__name__}: {e}"
            rows.append(dict(dataset=ds, uid=u["uid"], cluster=u["cluster"], n_strokes=len(gt), factor=f, strength=si,
                             param=k, eff_k=eff, protocol=prot, has_time=gt_t is not None, error=err, **r))
    return rows


FAMILY = {"shape_jitter": "shape", "shape_elastic": "shape", "order": "order", "direction": "direction",
          "penup_merge": "penup", "penup_split": "penup", "timing": "timing"}


def sensitivity(df, B):
    import pandas as pd
    from scipy.stats import spearmanr
    from uhwr_eval import METRICS
    mets = [m for m in METRICS if m in df.columns and m != "stroke_count_err"]
    out = []
    for (ds, prot), g in df.groupby(["dataset", "protocol"]):
        ident = g[g.factor == "identity"].set_index("uid")
        for f in FAMILY:
            gf = g[g.factor == f]
            if gf.empty: continue
            for m in mets:
                row = dict(dataset=ds, protocol=prot, metric=m, perturbation=f, factor=FAMILY[f], metric_direction=METRICS[m][0])
                v0 = ident[m].astype(float)
                row["mean_s0"] = float(v0.mean())
                for s in range(1, 6):
                    row[f"mean_s{s}"] = float(gf[gf.strength == s][m].astype(float).mean())
                s5 = gf[gf.strength == 5].set_index("uid")[m].astype(float)
                d = (s5 - v0.reindex(s5.index)).dropna()
                cl = ident.cluster.reindex(d.index).to_numpy()
                row["delta5"] = float(d.mean()) if len(d) else np.nan
                row["n_units_delta5"] = int(len(d))
                if len(d):
                    uc, inv = np.unique(cl, return_inverse=True)
                    sm = np.bincount(inv, weights=d.to_numpy(), minlength=len(uc)); n = np.bincount(inv, minlength=len(uc)).astype(float)
                    idx = np.random.default_rng(0).integers(0, len(uc), size=(B, len(uc)))
                    bm = sm[idx].sum(1) / np.maximum(n[idx].sum(1), 1e-12)
                    row["delta5_ci95_lo"], row["delta5_ci95_hi"] = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
                xs = np.concatenate([np.zeros(len(v0)), gf.strength.to_numpy(float)])
                ys = np.concatenate([v0.to_numpy(), gf[m].astype(float).to_numpy()])
                ok = np.isfinite(ys)
                row["spearman_rho"] = float(spearmanr(xs[ok], ys[ok]).statistic) if ok.sum() > 3 and np.nanstd(ys[ok]) > 0 else np.nan
                row["n_valid"] = int(ok.sum())
                out.append(row)
    S = pd.DataFrame(out)
    S["abs_delta5"] = S.delta5.abs()
    S["selectivity"] = S.abs_delta5 / S.groupby(["dataset", "protocol", "metric"]).abs_delta5.transform("max")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*", default=["ouhdl", "casia", "tamil"])
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261016); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import uhwr_eval
    assert uhwr_eval.__version__ == "0.3.0", uhwr_eval.__version__
    E = pickle.load(open(os.path.join(a.bundle, "e1_units.pkl"), "rb"))
    tasks = [(ds, u, a.seed) for ds in a.datasets if ds in E for u in (E[ds][: a.limit] if a.limit else E[ds])]
    missing = [ds for ds in a.datasets if ds not in E]
    import multiprocessing as mp, pandas as pd
    t0 = time.time(); rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_unit, tasks, chunksize=4)):
            rows += r
            if i % 100 == 0: print(f"{i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "e1_v2_scores_per_unit.csv.gz"), index=False)
    S = sensitivity(df, a.boot)
    S.to_csv(os.path.join(a.out, "e1_v2_sensitivity.csv"), index=False)
    json.dump(dict(uhwr_eval=uhwr_eval.__version__, datasets=a.datasets, missing_datasets=missing, n_tasks=len(tasks), n_rows=len(df),
                   n_err=int((df.error != "").sum()), seconds=time.time() - t0, seed=a.seed, boot=a.boot,
                   units_per_dataset=df.groupby("dataset").uid.nunique().to_dict()),
              open(os.path.join(a.out, "e1_v2_run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    k = S[(S.protocol == "native") & S.metric.isin(["penup_f1", "penup_f1_v2", "timing_profile_tv", "timing_profile_tv_v2"])]
    print(k.pivot_table(index=["dataset", "metric"], columns="perturbation", values="delta5").round(3).to_string())
    print("done", len(df), "rows", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
