"""Aggregate per-sample scores -> results_rows (long format) with mean, SD, bootstrap 95% CI.

Bootstrap: percentile, B = 2000, seed 0.  Line units: resample lines.  Ligature units:
cluster bootstrap over lines (resample lines with replacement, keep all their units), so the
CI reflects the 253 (or subset) independent lines rather than the correlated units.
Means are over samples with a defined value (NaN excluded; n_valid reported).
"""
import numpy as np
import pandas as pd

METRIC_NAMES = {
    "dtw": ("DTW", "\u2193", 1), "ldtw": ("LDTW", "\u2193", 1), "aiou": ("AIoU", "\u2191", 1),
    "lpips": ("LPIPS", "\u2193", 1), "rmse": ("RMSE", "\u2193", 1),
    "chamfer": ("Chamfer", "\u2193", 1), "hausdorff": ("Hausdorff", "\u2193", 1),
    "frechet": ("Discrete Frechet", "\u2193", 1), "penup_f1": ("Pen-up F1", "\u2191", 1),
    "stroke_count_abs_err": ("Stroke-count abs error", "\u2193", 1),
    "stroke_match_recall": ("Stroke match recall", "\u2191", 1),
    "stroke_match_precision": ("Stroke match precision", "\u2191", 1),
    "strict_order_acc": ("Strict order accuracy (%)", "\u2191", 100),
    "kendall_tau": ("Kendall tau", "\u2191", 1),
    "direction_acc": ("Direction accuracy (%)", "\u2191", 100),
    "loop_orient_acc": ("Loop orientation agreement (%)", "\u2191", 100),
}
TAGS = {"dtw": "established", "ldtw": "established", "aiou": "established", "lpips": "established",
        "rmse": "established", "chamfer": "established", "hausdorff": "established",
        "frechet": "established", "stroke_count_abs_err": "established"}
ORDER_METRICS = {"stroke_count_abs_err", "stroke_match_recall", "stroke_match_precision",
                 "strict_order_acc", "kendall_tau", "direction_acc", "loop_orient_acc"}


def boot_ci(values, clusters, B=2000, seed=0):
    v = pd.Series(values).astype(float)
    ok = v.notna().to_numpy()
    v, c = v[ok].to_numpy(), np.asarray(clusters)[ok]
    if len(v) == 0:
        return np.nan, np.nan
    uc, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uc))
    cnts = np.bincount(inv, minlength=len(uc)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uc), size=(B, len(uc)))
    means = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-12)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(df: pd.DataFrame, method_info: dict, extra_notes=None):
    out = []
    keys = ["unit", "method", "input", "protocol"]
    for (unit, method, inp, prot), g in df.groupby(keys):
        info = method_info[method]
        n_total = g["unit_id"].nunique()
        n_err = int((g.get("error", "") != "").sum()) if "error" in g else 0
        n_empty = int(g["empty_pred"].fillna(0).sum()) if "empty_pred" in g else 0
        for m, (name, arrow, mult) in METRIC_NAMES.items():
            if prot == "standardised" and m in ORDER_METRICS:
                continue  # protocol-insensitive by construction; reported once (native)
            vals = g[m] * mult
            nv = int(vals.notna().sum())
            if nv == 0:
                continue
            lo, hi = boot_ci(vals, g["sample_id"])
            sd = float(vals.std(ddof=1)) if nv > 1 else float("nan")
            samp = ("native (GT as recorded at 500 Hz pen-down samples; prediction as emitted)"
                    if prot == "native" else "standardised (per-stroke arc-length resampling, 1 px step)")
            if m in ORDER_METRICS:
                samp = "protocol-insensitive (order/stroke metrics)"
            if m == "rmse":
                samp = ("fixed-length K=320 (ligature) / K=2048 (line) after "
                        + ("native" if prot == "native" else "standardised") + " input")
            res = "64x64 (ligature)" if unit == "ligature" else "line height 64 px (ink bbox 56 px)"
            notes = (f"mean over n={nv} {unit}s with defined value (of {n_total}); SD={sd:.4g}; "
                     f"bootstrap 95% CI [{lo:.4g}, {hi:.4g}] (B=2000, cluster=line); "
                     f"empty predictions={n_empty}; scoring errors={n_err}; metric tag="
                     f"{TAGS.get(m, 'proposed')}; {info.get('notes', '')}")
            if extra_notes and (unit, method, inp) in extra_notes:
                notes += " " + extra_notes[(unit, method, inp)]
            out.append(dict(
                method=info["name"], family=info["family"], paper=info["paper"],
                venue_year=info["venue_year"], paper_id=info["paper_id"],
                dataset="OUHD-L v1.0 (Zenodo 10.5281/zenodo.20642162)", script="Urdu (Nastaliq)",
                unit=unit, split=info.get("split", "test (Zenodo writer-disjoint, 253 lines, 32 writers)"),
                seen_unseen="unseen writers", input_type=info["input"][inp],
                resolution=res, sampling=samp, metric=name, direction=arrow,
                value=round(float(vals.mean()), 4), provenance="our-run", verified="yes",
                source_loc="ouhdl_baseline_results.csv (per-sample) / ouhdl_runs_methods.md",
                code_public=info["code_public"], notes=notes,
                _n=nv, _sd=sd, _lo=lo, _hi=hi, _protocol=prot, _input=inp, _method_id=method))
    return pd.DataFrame(out)
