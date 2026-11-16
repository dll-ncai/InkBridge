"""ORDER Synthesis, amendment A1 sealed read of COMP_rep3 (one call per set). Uses loop_tables.py (load_cell, unit_table, boot_idx:
writer-cluster bootstrap B=2000, rng 20261029) and order_verdict.composite, both imported unchanged. All contrasts 'own' (all 9 keys each side).
Cells (prereg s5/s7 + A1): ORDER-WIN of COMP_rep3 vs COMP_pool3 with part (1) Holm across {penup_f1_v2, strict_order_acc, direction_acc}:
  two-sided bootstrap p_m = min(1, 2*min(mean(bs<=0), mean(bs>=0))); sorted ascending, metric i (1-based) tested at percentile CI level
  1 - 0.05/(4-i) (98.33 / 97.5 / 95 %), step-down (stop at first non-rejection); part (1) holds if a Holm-rejected metric has
  delta >= margin and its Holm-level CI lo > 0. Parts (2) other two order metrics 95 % CI lo > -margin; (3) aiou 95 % CI lo > -0.01035.
Usage: python sealed_read.py --spec SPEC.json --out DIR --label SET"""
import argparse, hashlib, json, os, sys, datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("ORDER_CODE", "$U/exp/ORDER/code"))
from loop_tables import load_cell, unit_table, boot_idx, MET
from order_verdict import composite, MARG, AIOU_M
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
LV = {1: 1 - 0.05 / 3, 2: 1 - 0.05 / 2, 3: 0.95}
def pboot(um, uc, m):
    j = um[[m, "writer_id"]].join(uc[[m]], rsuffix="_c", how="inner").dropna()
    d = (j[m] - j[m + "_c"]).to_numpy(); bs = np.array([d[i].mean() for i in boot_idx(j.writer_id.to_numpy())]); return d.mean(), bs, len(j)
def ci(bs, lvl): q = (1 - lvl) / 2 * 100; return np.percentile(bs, q), np.percentile(bs, 100 - q)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", required=True)
    a = ap.parse_args(); S = json.load(open(a.spec)); os.makedirs(a.out, exist_ok=True)
    U = {n: unit_table(load_cell(c)) for n, c in S["cells"].items()}
    rows, cells = [], {}
    for mth, cmp_ in S["contrasts"]:
        um, uc = U[mth], U[cmp_]; R = {}
        for m in MET:
            est, bs, n = pboot(um, uc, m); lo, hi = ci(bs, 0.95)
            p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
            R[m] = dict(delta=est, ci_lo=lo, ci_hi=hi, n=n, p_boot=p, bs=bs)
            rows.append(dict(set=a.label, method=mth, comparator=cmp_, metric=m, delta=est, ci_lo=lo, ci_hi=hi, n=n, p_boot=p))
        C, Clo, Chi, Cn = composite(um, uc)
        rows.append(dict(set=a.label, method=mth, comparator=cmp_, metric="composite_C", delta=C, ci_lo=Clo, ci_hi=Chi, n=Cn, p_boot=np.nan))
        order = sorted(MARG, key=lambda m: R[m]["p_boot"]); holm = []; stop = False
        for i, m in enumerate(order, 1):
            lo_h, hi_h = ci(R[m]["bs"], LV[i]); rej = (not stop) and lo_h > 0
            if not rej: stop = True
            holm.append(dict(metric=m, rank=i, level=round(LV[i] * 100, 2), ci_lo=lo_h, ci_hi=hi_h, rejected=bool(rej), meets_margin=bool(R[m]["delta"] >= MARG[m])))
        p1 = [h["metric"] for h in holm if h["rejected"] and h["meets_margin"]]
        p2 = {m: bool(any(all(R[o]["ci_lo"] > -MARG[o] for o in MARG if o != w) for w in [m])) for m in p1}
        p3 = bool(R["aiou"]["ci_lo"] > -AIOU_M)
        win = bool(any(p2.values()) and p3)
        cells[f"{mth}__vs__{cmp_}"] = dict(order_win=win if cmp_ == "COMP_pool3" else None, part1_metrics=p1, part2_by_part1_metric=p2, part3_aiou_noninf=p3,
                                         holm=holm, composite=dict(C=C, lo=Clo, hi=Chi, n=Cn),
                                         metrics={m: {k: v for k, v in R[m].items() if k != "bs"} for m in MET})
    P = pd.DataFrame(rows); P.to_csv(f"{a.out}/paired_{a.label}.csv", index=False)
    json.dump(dict(set=a.label, utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), cells=cells, spec=a.spec, spec_sha256=sha(a.spec),
                   code={"sealed_read.py": sha(__file__), "loop_tables.py": sha(sys.modules["loop_tables"].__file__), "order_verdict.py": sha(sys.modules["order_verdict"].__file__)},
                   inputs={n: {k: dict(path=f, sha256=sha(f)) for k, f in c["files"].items()} for n, c in S["cells"].items()},
                   host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID")), open(f"{a.out}/cells_{a.label}.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250); print(P[P.metric.isin(list(MARG) + ["aiou", "composite_C"])].round(4).to_string())
    for k, v in cells.items(): print(k, "ORDER-WIN", v["order_win"], "p1", v["part1_metrics"], "holm", [(h["metric"], h["level"], round(h["ci_lo"], 4), h["rejected"]) for h in v["holm"]])
if __name__ == "__main__": main()
