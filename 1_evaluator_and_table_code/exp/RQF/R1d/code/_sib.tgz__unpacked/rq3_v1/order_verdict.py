"""ORDER loop verdicts for one read (prereg ORDER_LOOP_PREREG_2026-11-04.md s5 + ORDER_LOOP_RULES.md composite).
Inputs: --spec (loop_tables spec), --paired (loop_tables day_paired.csv of the same spec), --out DIR, --prov JSON (extra provenance).
ORDER-WIN (per contrast with comparator COMP pool3, from day_paired.csv = frozen loop_tables.py numbers):
  (1) some of {penup_f1_v2, strict_order_acc, direction_acc} has delta >= margin and ci_lo > 0;
  (2) every other order metric has ci_lo > -margin;  (3) aiou ci_lo > -0.01035.
Composite C = dPU/0.0141 + dSO/0.01065 + dDIR/0.01245 (dX = mean over paired units of A - R; direction over units where defined),
CI = loop_tables.boot_idx (writer-cluster, B=2000, rng 20261029, 95 % percentile) on the units paired for penup/strict, shared indices.
Outputs: OUT/verdicts.csv (one row per contrast), OUT/PROVENANCE.json."""
import argparse, json, hashlib, os, sys, datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_tables import load_cell, unit_table, boot_idx
MARG = {"penup_f1_v2": 0.0141, "strict_order_acc": 0.01065, "direction_acc": 0.01245}
AIOU_M = 0.01035
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def composite(um, uc):
    j = um[list(MARG) + ["writer_id"]].join(uc[list(MARG)], rsuffix="_c", how="inner")
    j = j.dropna(subset=["penup_f1_v2", "penup_f1_v2_c", "strict_order_acc", "strict_order_acc_c"])
    D = {m: (j[m] - j[m + "_c"]).to_numpy() for m in MARG}
    f = lambda I: sum(np.nanmean(D[m][I]) / MARG[m] for m in MARG)
    est = f(np.arange(len(j))); bs = np.array([f(I) for I in boot_idx(j.writer_id.to_numpy())])
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(j)
def order_win(P, mth, cmp_):
    q = P[(P.method == mth) & (P.comparator == cmp_)].set_index("metric")
    p1 = [m for m in MARG if q.loc[m, "delta"] >= MARG[m] and q.loc[m, "ci_lo"] > 0]
    ok = False
    for m in p1:
        if all(q.loc[o, "ci_lo"] > -MARG[o] for o in MARG if o != m): ok = True
    p3 = bool(q.loc["aiou", "ci_lo"] > -AIOU_M)
    return bool(ok and p3), ",".join(p1), p3, q
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--paired", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--comp", default="COMP_pool3"); ap.add_argument("--prov", default="")
    a = ap.parse_args(); S = json.load(open(a.spec)); P = pd.read_csv(a.paired); os.makedirs(a.out, exist_ok=True)
    U = {}
    def ut(n):
        if n not in U: U[n] = unit_table(load_cell(S["cells"][n]))
        return U[n]
    rows = []
    for ent in S["contrasts"]:
        mth, cmp_ = ent[0], ent[1]; est, lo, hi, n = composite(ut(mth), ut(cmp_))
        r = dict(method=mth, comparator=cmp_, C=est, C_lo=lo, C_hi=hi, n_units=n)
        win, p1, p3, q = order_win(P, mth, cmp_)
        for m in list(MARG) + ["aiou", "tau_cp", "kendall_tau", "ep3", "stroke_count_err"]:
            if m in q.index: r[f"d_{m}"] = q.loc[m, "delta"]; r[f"lo_{m}"] = q.loc[m, "ci_lo"]; r[f"hi_{m}"] = q.loc[m, "ci_hi"]
        r["order_win"] = win if cmp_ == a.comp else ""; r["part1_metrics"] = p1; r["aiou_noninf"] = p3
        rows.append(r)
    V = pd.DataFrame(rows); V.to_csv(f"{a.out}/verdicts.csv", index=False)
    prov = dict(utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), spec=a.spec, spec_sha256=sha(a.spec),
                paired_csv=a.paired, paired_sha256=sha(a.paired), code={"order_verdict.py": sha(__file__)},
                inputs={n: {k: dict(path=f, sha256=sha(f)) for k, f in c["files"].items()} for n, c in S["cells"].items()},
                host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))
    if a.prov: prov.update(json.load(open(a.prov)))
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".csv"): prov.setdefault("outputs", {})[f] = sha(os.path.join(a.out, f))
    json.dump(prov, open(f"{a.out}/PROVENANCE.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(V[["method", "comparator", "C", "C_lo", "C_hi", "d_penup_f1_v2", "lo_penup_f1_v2", "d_strict_order_acc", "lo_strict_order_acc",
             "d_direction_acc", "lo_direction_acc", "d_aiou", "lo_aiou", "order_win"]].round(4).to_string())
if __name__ == "__main__": main()
