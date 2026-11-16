"""A1.2 leak table. leak(metric, set) = max over off-factor perturbations |delta5| / max over own-factor perturbations |delta5| (protocol native; standardised also written).
Own factor: aiou->shape, kendall_tau/strict_order_acc->order, direction_acc->direction, penup_f1_v2->penup. PASS = leak <= 0.10 with own change > 0. Off-factor perturbations whose delta5 is undefined (NaN, e.g. tau after merge when no unit has a valid pair) are skipped as in the E1 v1 rule and listed in off_undefined."""
import argparse, os
import numpy as np, pandas as pd
FAM = dict(shape_jitter="shape", shape_elastic="shape", order="order", direction="direction", penup_merge="penup", penup_split="penup", timing="timing")
OWN = {"aiou": "shape", "kendall_tau": "order", "strict_order_acc": "order", "direction_acc": "direction", "penup_f1_v2": "penup"}
THR = 0.10
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sens", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    S = pd.read_csv(a.sens); rows = []
    for (ds, prot, m), g in S[S.metric.isin(OWN)].groupby(["dataset", "protocol", "metric"]):
        d = g.set_index("perturbation").delta5
        own = [p for p in FAM if FAM[p] == OWN[m] and p in d.index]; off = [p for p in d.index if FAM[p] != OWN[m]]
        ov = d[own].abs(); fv = d[off].abs()
        o = ov.max(); lk = fv.max() / o if o and np.isfinite(o) and o > 0 else np.nan
        n_nan_off = int(fv.isna().sum())
        rows.append(dict(dataset=ds, protocol=prot, metric=m, own_factor=OWN[m], own_perturbation=ov.idxmax() if ov.notna().any() else "", own_change=o,
                         off_perturbation=fv.idxmax() if fv.notna().any() else "", off_change=fv.max(), leak=lk, n_off_nan=n_nan_off,
                         result=("PASS" if (np.isfinite(lk) and lk <= THR) else ("NA" if not np.isfinite(lk) else "FAIL")), off_undefined=";".join(fv.index[fv.isna()])))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "m1_leak_long.csv"), index=False)
    for prot in ("native", "standardised"):
        t = T[T.protocol == prot]
        t.pivot(index="metric", columns="dataset", values="leak").round(3).to_csv(os.path.join(a.out, f"m1_leak_wide_{prot}.csv"))
        t.pivot(index="metric", columns="dataset", values="result").to_csv(os.path.join(a.out, f"m1_passfail_{prot}.csv"))
    print(T[T.protocol == "native"].pivot(index="metric", columns="dataset", values="leak").round(3).to_string())
    print(T[T.protocol == "native"].pivot(index="metric", columns="dataset", values="result").to_string())
if __name__ == "__main__": main()
