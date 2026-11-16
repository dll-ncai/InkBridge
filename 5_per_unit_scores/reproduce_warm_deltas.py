#!/usr/bin/env python3
"""Recompute the pooled paired differences of the warm-adaptation comparison
(paper section "Warm Adaptation on the Target") from the shipped per-unit scores
and compare them with the table file T56_per_seed.csv.

Usage (from the repository root):
    python 5_per_unit_scores/reproduce_warm_deltas.py

Method (same as 5_per_unit_scores/README_per_unit.md, section "Check"):
per-unit value = mean over the sampling seeds of one checkpoint tag; an arm is the
mean of the per-unit tables of its three training-seed tags; delta = mean over
units of (arm - pooled comparator). Intervals are not recomputed.
Requires numpy and pandas only. Exit code 1 if any difference exceeds --tol.
"""
import argparse, os, re, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_rows import load
METRICS = ["aiou", "penup_f1_v2", "strict_order_acc", "direction_acc", "kendall_tau", "ldtw"]

def pooled(t, d, pattern):
    """Mean over training-seed tags of the per-unit means over sampling seeds."""
    s = t[t["dir"].str.endswith(d) & t["tag"].str.match(pattern)]
    tags = sorted(s["tag"].unique())
    per_tag = [s[s["tag"] == g].groupby("unit_id")[METRICS].mean() for g in tags]
    return sum(per_tag) / len(per_tag), tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--out", default=None, help="optional CSV path for the comparison table")
    a = ap.parse_args()
    root = os.path.dirname(HERE)
    t = load(os.path.join(HERE, "tables_5_6_8_10_11_12"), "test_ouhdl")
    ref = pd.read_csv(os.path.join(root, "6_table_sources/exp/RQF/tables/reviewer_followup_v1/T56_per_seed.csv"))
    ref = ref[ref["train_seed"].astype(str) == "pooled"].set_index(["table", "arm"])
    MON, CMP, RQF = "fmdx_monitor/score_out/rows", "fmdx_monitor/score_out_cmp/rows", "exp/RQF/score_out/rows"
    comp01, tags_c01 = pooled(t, CMP, r"comp_p01_identity_s\d+(_vps)?_ms\d+_identity$")
    comp10, tags_c10 = pooled(t, RQF, r"LUOFT_p10_s\d+_ms\d+_identity$")
    rows = []
    for arm in ["NOFM_W", "PEINIT_W"]:
        s1 = t[(t["dir"].str.endswith(MON) & t["tag"].str.match(arm + r"_ft_ms\d+$")) |
               (t["dir"].str.endswith(RQF) & t["tag"].str.match(arm + r"_p01_s\d+_ft_ms\d+$"))]
        tags1 = sorted(s1["tag"].unique())
        a01 = sum(s1[s1["tag"] == g].groupby("unit_id")[METRICS].mean() for g in tags1) / len(tags1)
        a10, tags10 = pooled(t, RQF, arm + r"_p10_s\d+_ft_ms\d+$")
        for table, x, c, tags in [("T5_p01_vs_FT1", a01, comp01, tags1), ("T6_p10_vs_FT10", a10, comp10, tags10)]:
            u = x.index.intersection(c.index)
            d = (x.loc[u] - c.loc[u]).mean()
            for m in METRICS:
                r = float(ref.loc[(table, arm), m + "_d"])
                rows.append(dict(table=table, arm=arm, n_train_seeds=len(tags), n_units=len(u), metric=m,
                                 delta_from_per_unit_rows=float(d[m]), delta_in_T56_per_seed=r, abs_diff=abs(float(d[m]) - r)))
    out = pd.DataFrame(rows)
    if a.out: out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False, float_format=lambda v: "%.6g" % v))
    worst = out["abs_diff"].max()
    print("rows: %d   largest absolute difference: %.3g   tolerance: %g" % (len(out), worst, a.tol))
    print("comparator tags (1%%): %d, (10%%): %d" % (len(tags_c01), len(tags_c10)))
    sys.exit(0 if worst <= a.tol else 1)

if __name__ == "__main__":
    main()
