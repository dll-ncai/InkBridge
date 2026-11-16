"""Writer-cluster summary of e0_score.py row files: per (split, input, protocol, metric) mean over units with a defined
value, 95% writer-cluster bootstrap CI (B=2000, seed 0; resampler identical to exp/M1b/code_v4_scorefix/m1b_readout_wc.py),
n_units / n_writers / n_lines / n_defined / errors / empty predictions.  Order and stroke metrics are protocol-insensitive
and are reported for the native protocol only (as e0_score.py summary).  Ligature-level test (4,406) and gold (354) units
are separate rows; they are never pooled.
python pennet_summary_wc.py --rows test_rendered_ms8_rows.csv ... --out-prefix summary_PENNET_writercluster
(file names must start with <split>_<input>_)
"""
import argparse
import os

import numpy as np
import pandas as pd

METRICS = ["aiou", "ldtw", "dtw", "strict_order_acc", "kendall_tau", "direction_acc", "penup_f1", "penup_f1_v2",
           "stroke_count_abs_err", "stroke_match_recall", "loop_orient_acc", "chamfer", "hausdorff", "frechet", "rmse"]
ORDER = {"strict_order_acc", "kendall_tau", "direction_acc", "stroke_count_abs_err", "stroke_match_recall", "loop_orient_acc"}
B, SEED = 2000, 0


def wboot(v, writers):
    rng = np.random.default_rng(SEED)
    ok = ~np.isnan(v)
    v, writers = v[ok], writers[ok]
    if len(v) == 0:
        return (np.nan, np.nan)
    u = np.unique(writers)
    grp = {k: np.flatnonzero(writers == k) for k in u}
    est = []
    for _ in range(B):
        idx = np.concatenate([grp[k] for k in rng.choice(u, len(u), replace=True)])
        est.append(v[idx].mean())
    return tuple(np.percentile(est, [2.5, 97.5]))


ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--out-prefix", required=True)
a = ap.parse_args()
recs = []
for f in a.rows:
    base = os.path.basename(f)
    split, inp = base.split("_")[0], base.split("_")[1]
    d = pd.read_csv(f)
    for prot, g in d.groupby("protocol", sort=False):
        wr = g["writer_id"].values
        for m in METRICS:
            if m not in g.columns or (prot == "standardised" and m in ORDER):
                continue
            v = g[m].astype(float).values
            lo, hi = wboot(v, wr)
            recs.append(dict(split=split, input=inp, method=str(g["method"].iloc[0]), protocol=prot, metric=m,
                             mean=float(np.nanmean(v)) if (~np.isnan(v)).any() else np.nan, ci95_lo=float(lo), ci95_hi=float(hi),
                             n_units=int(g.unit_id.nunique()), n_writers=int(g.writer_id.nunique()),
                             n_lines=int(g.sample_id.nunique()), n_defined=int((~np.isnan(v)).sum()),
                             n_errors=int((g["error"].fillna("") != "").sum()),
                             n_empty_pred=int(g["empty_pred"].fillna(0).sum()) if "empty_pred" in g else -1,
                             ci="writer-cluster bootstrap, B=2000, seed 0", rows_file=base))
df = pd.DataFrame(recs)
df.to_csv(a.out_prefix + ".csv", index=False)
with open(a.out_prefix + ".md", "w") as fh:
    fh.write("# PEN-Net re-implementation, ligature level: mean [95% writer-cluster bootstrap CI, B=2000, seed 0]\n\n")
    for (split, prot), g in df.groupby(["split", "protocol"], sort=False):
        fh.write(f"## {split} units / {prot} protocol\n\n")
        inputs = list(dict.fromkeys(g.input))
        fh.write("| metric | " + " | ".join(f"{i} input (n_defined)" for i in inputs) + " |\n|---|" + "---|" * len(inputs) + "\n")
        for m in dict.fromkeys(g.metric):
            cells = []
            for i in inputs:
                r = g[(g.metric == m) & (g.input == i)]
                cells.append("" if r.empty else f"{r['mean'].iloc[0]:.4f} [{r.ci95_lo.iloc[0]:.4f}, {r.ci95_hi.iloc[0]:.4f}] ({int(r.n_defined.iloc[0])})")
            fh.write(f"| {m} | " + " | ".join(cells) + " |\n")
        nu = g.groupby("input").n_units.first().to_dict()
        nw = g.groupby("input").n_writers.first().to_dict()
        fh.write(f"\nunits {nu}; writers {nw}\n\n")
print(a.out_prefix + ".csv", len(df))
