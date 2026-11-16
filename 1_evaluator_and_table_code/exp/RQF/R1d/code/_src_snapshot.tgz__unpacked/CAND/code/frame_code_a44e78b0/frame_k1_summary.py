"""K1 frame kill-check readout from e0_score.py rows (native protocol): per-frame means with writer-cluster bootstrap
95% CIs (uhwr_baselines.aggregate.boot_ci, B=2000, seed 0, clusters = sample_id = line / writer key), per-unit
seed-averaged paired deltas vs identity with cluster-bootstrap CIs, the kill rule of
uhwr_method_candidates_2026-10-27.md section 4, and the loop-dominated-stroke subset (units with n_loop_eligible > 0)
for the chirality claim (mirror_x vs vt loop_orient_acc).

python frame_k1_summary.py --rows "preds/val250_ouhdl__LUO_ms6_*__seed*_rows.csv" --set val250_ouhdl --out k1_val250.json
   (one --rows glob per set; file names <set>__<label>_<frame>__seed<s>_rows.csv; frame parsed from the label suffix)
Kill rule (identity reference AIoU 0.7117 from the pack): best frame direction >= 0.70 and tau >= 0.65 and
AIoU >= identity_AIoU - 0.02.  This script does not decide anything; it writes numbers and the rule's verdict.
"""
import argparse
import glob
import json
import re

import numpy as np
import pandas as pd

from uhwr_baselines.aggregate import boot_ci

FRAMES = ("identity", "mirror_x", "vt")
METRICS = ["aiou", "ldtw", "dtw", "kendall_tau", "direction_acc", "strict_order_acc", "penup_f1", "loop_orient_acc", "stroke_count_abs_err"]
KILL = {"direction_min": 0.70, "tau_min": 0.65, "aiou_drop_max": 0.02}


def load(globs):
    recs = []
    for g in globs:
        for f in glob.glob(g):
            m = re.search(r"__(.+)_(identity|mirror_x|vt)__seed(\d+)_rows", f)
            if not m:
                continue
            d = pd.read_csv(f)
            d = d[d.protocol == "native"].assign(label=m.group(1), frame=m.group(2), seed=int(m.group(3)), _file=f)
            recs.append(d)
    if not recs:
        raise SystemExit("no rows files matched")
    return pd.concat(recs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True); ap.add_argument("--set", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--identity-ref-aiou", type=float, default=None, help="override the identity reference (default: the identity frame rows)")
    a = ap.parse_args()
    df = load(a.rows)
    out = {"set": a.set, "frames": {}, "seeds": sorted(df.seed.unique().tolist()), "n_rows_files": int(df._file.nunique()), "kill_rule": KILL}
    # per-frame: seed-averaged per-unit values, then cluster bootstrap over sample_id
    per_unit = df.groupby(["frame", "unit_id", "sample_id"])[METRICS + ["n_loop_eligible", "n_dir_eligible"]].mean().reset_index()
    for fr in FRAMES:
        g = per_unit[per_unit.frame == fr]
        if g.empty:
            continue
        rec = {"n_units": int(len(g)), "n_clusters": int(g.sample_id.nunique()), "n_seeds": int(df[df.frame == fr].seed.nunique())}
        for m in METRICS:
            lo, hi = boot_ci(g[m], g.sample_id)
            rec[m] = {"mean": float(g[m].mean()), "ci95": [lo, hi], "n_defined": int(g[m].notna().sum())}
        loops = g[g.n_loop_eligible > 0]
        lo, hi = boot_ci(loops["loop_orient_acc"], loops.sample_id) if len(loops) else (np.nan, np.nan)
        rec["loop_dominated_subset"] = {"n_units": int(len(loops)), "loop_orient_acc_mean": float(loops.loop_orient_acc.mean()) if len(loops) else None, "ci95": [lo, hi],
                                        "aiou_mean": float(loops.aiou.mean()) if len(loops) else None, "direction_mean": float(loops.direction_acc.mean()) if len(loops) else None}
        out["frames"][fr] = rec
    # paired deltas vs identity
    if "identity" in out["frames"]:
        base = per_unit[per_unit.frame == "identity"].set_index("unit_id")
        out["paired_vs_identity"] = {}
        for fr in ("mirror_x", "vt"):
            g = per_unit[per_unit.frame == fr].set_index("unit_id")
            common = base.index.intersection(g.index)
            rec = {"n_units": int(len(common))}
            for m in METRICS:
                d = (g.loc[common, m] - base.loc[common, m])
                lo, hi = boot_ci(d, base.loc[common, "sample_id"])
                rec[m] = {"delta_mean": float(d.mean()), "ci95": [lo, hi], "ci_excludes_0": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
            out["paired_vs_identity"][fr] = rec
    # kill rule
    ref = a.identity_ref_aiou if a.identity_ref_aiou is not None else out["frames"].get("identity", {}).get("aiou", {}).get("mean")
    verdict = {}
    for fr in ("mirror_x", "vt"):
        if fr not in out["frames"]:
            continue
        r = out["frames"][fr]
        ok_dir = r["direction_acc"]["mean"] >= KILL["direction_min"]; ok_tau = r["kendall_tau"]["mean"] >= KILL["tau_min"]
        ok_aiou = ref is None or r["aiou"]["mean"] >= ref - KILL["aiou_drop_max"]
        verdict[fr] = {"direction_ok": bool(ok_dir), "tau_ok": bool(ok_tau), "aiou_ok": bool(ok_aiou), "passes": bool(ok_dir and ok_tau and ok_aiou)}
    passing = [f for f, v in verdict.items() if v["passes"]]
    best = None
    if passing:
        best = max(passing, key=lambda f: (out["frames"][f]["kendall_tau"]["mean"], out["frames"][f]["direction_acc"]["mean"]))
    out["kill_check"] = {"identity_ref_aiou": ref, "per_frame": verdict, "any_frame_passes": bool(passing), "best_frame_by_tau_then_direction": best,
                         "note": "exploratory readout; the pre-registered decision rule lives in prereg_candidates_2026-10-28.md"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({fr: {m: round(out["frames"][fr][m]["mean"], 4) for m in ("aiou", "kendall_tau", "direction_acc", "loop_orient_acc")} for fr in out["frames"]}))
    print(json.dumps(out["kill_check"]))


if __name__ == "__main__":
    main()
