"""M5 summary table: mean + 95% writer-cluster bootstrap CI (uhwr_baselines.aggregate.boot_ci, B=2000, seed 0,
clusters = writer_id column of the e0_score.py rows) per (row label, eval set, input, protocol, metric).
Order/stroke metrics are reported once (native protocol), as in e0_score.py summary.

python m5_summary.py --row "label|eval_set|path/to/rows.csv" [--row ...] --out summary.csv
Missing row files are listed in the output with status=missing (not silently dropped).
"""
import argparse
import os

import numpy as np
import pandas as pd

from uhwr_baselines.aggregate import boot_ci

METRICS = ["aiou", "ldtw", "dtw", "strict_order_acc", "kendall_tau", "direction_acc", "penup_f1",
           "stroke_count_abs_err", "stroke_match_recall", "loop_orient_acc", "chamfer", "hausdorff", "frechet", "rmse"]
ORDER = {"strict_order_acc", "kendall_tau", "direction_acc", "stroke_count_abs_err", "stroke_match_recall",
         "loop_orient_acc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for spec in a.row:
        label, es, path = spec.split("|", 2)
        if not os.path.exists(path):
            res.append(dict(label=label, eval_set=es, status="missing", rows_file=path)); continue
        df = pd.read_csv(path)
        for (inp, prot), g in df.groupby(["input", "protocol"], sort=False):
            for m in METRICS:
                if m not in g or (prot == "standardised" and m in ORDER):
                    continue
                v = g[m].astype(float)
                lo, hi = boot_ci(v, g["writer_id"].to_numpy())
                res.append(dict(label=label, eval_set=es, method=str(g["method"].iloc[0]), input=inp, protocol=prot, metric=m,
                                mean=float(v.mean()), ci95_lo=lo, ci95_hi=hi, ci="writer-cluster bootstrap B=2000 seed 0",
                                n_valid=int(v.notna().sum()), n_units=int(g.unit_id.nunique()),
                                n_writers=int(g.writer_id.nunique()),
                                n_errors=int((g["error"].fillna("") != "").sum()) if "error" in g else 0,
                                n_empty_pred=int(g["empty_pred"].fillna(0).sum()) if "empty_pred" in g else 0,
                                status="ok", rows_file=path))
    out = pd.DataFrame(res); out.to_csv(a.out, index=False)
    ok = out[out.get("status", "ok") == "ok"] if len(out) else out
    if len(ok):
        p = ok[(ok.protocol == "native") & ok.metric.isin(["aiou", "ldtw", "kendall_tau", "direction_acc"])]
        print(p.pivot_table(index=["label", "eval_set", "input"], columns="metric", values="mean").round(4).to_string())
    print(a.out, len(out))


if __name__ == "__main__":
    main()
