#!/usr/bin/env python3
"""R3 follow-up (11 Nov 2026, reviewer D3): does endpoint-agreement selection change pairwise stroke order, or only segmentation?
Reuses the frozen R3 specs (read_all3/spec_<base>__test_ouhdl.json) and loop_tables (load_cell, unit_table, boot_idx: writer-cluster B=2000, rng 20261029).
Per unit: chosen candidate identified by exact match of the rep3 row (aiou, ldtw, n_pred_strokes) to one pool row.
Split: 'same_count' = all 3 candidates have the same predicted stroke count (selection cannot change stroke count);
'diff_count' = candidates differ. Paired delta rep3 - pool3 (seed-mean) within each split. Descriptive follow-up, not a pre-registered read."""
import json, os, sys, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("ORDER_CODE", "$U/exp/RQF/code/rq3_v1"))
from loop_tables import load_cell, unit_table, boot_idx
R = "$U/exp/RQF/R3"; OUT = R + "/split_v1"; os.makedirs(OUT, exist_ok=True)
MET = ["strict_order_acc", "penup_f1_v2", "kendall_tau", "tau_cp", "aiou", "direction_acc"]
def pb(um, uc, m):
    j = um[[m, "writer_id"]].join(uc[[m]], rsuffix="_c", how="inner").dropna()
    if len(j) < 20: return np.nan, np.nan, np.nan, len(j)
    d = (j[m] - j[m + "_c"]).to_numpy(); bs = np.array([d[i].mean() for i in boot_idx(j.writer_id.to_numpy())])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(j)
rows, meta, prov = [], [], {}
for b in ["E0", "LUOft_p10", "M5", "NOFM_W_FT", "PEINIT_W_FT"]:
    sp = f"{R}/read_all3/spec_{b}__test_ouhdl.json"; S = json.load(open(sp)); prov[sp] = hashlib.sha256(open(sp, "rb").read()).hexdigest()
    P = load_cell(S["cells"]["COMP_pool3"]); Q = load_cell(S["cells"]["COMP_rep3"])
    q = Q.groupby("unit_id").first()
    cnt = P.pivot_table(index="unit_id", columns="seed_k", values="n_pred_strokes", aggfunc="first")
    nun = cnt.nunique(axis=1)
    Pm = P.set_index("unit_id")
    def chosen(u):
        r = q.loc[u]; c = Pm.loc[[u]]
        ok = np.isclose(c.aiou, r.aiou, equal_nan=True) & np.isclose(c.ldtw, r.ldtw, equal_nan=True) & (c.n_pred_strokes == r.n_pred_strokes)
        return int(ok.sum())
    nmatch = pd.Series({u: chosen(u) for u in q.index})
    UQ, UP = unit_table(Q), unit_table(P)
    gt = q.n_gt_strokes; pool_ok = (P.assign(ok=P.n_pred_strokes == P.n_gt_strokes).groupby("unit_id").ok.mean())
    rep_ok = (q.n_pred_strokes == gt).astype(float)
    for split, ids in (("all", q.index), ("same_count", nun.index[nun == 1]), ("diff_count", nun.index[nun > 1])):
        ids = [i for i in ids if i in UQ.index and i in UP.index]
        for m in MET:
            d, lo, hi, n = pb(UQ.loc[ids], UP.loc[ids], m)
            rows.append(dict(base=b, split=split, n_units=len(ids), metric=m, delta=d, ci_lo=lo, ci_hi=hi, n=n))
        meta.append(dict(base=b, split=split, n_units=len(ids), share=len(ids) / len(q),
                         rep_count_eq_gt=rep_ok.loc[ids].mean(), pool_count_eq_gt=pool_ok.loc[ids].mean()))
    meta.append(dict(base=b, split="match_check", n_units=len(q), share=float((nmatch >= 1).mean()), rep_count_eq_gt=np.nan, pool_count_eq_gt=np.nan))
pd.DataFrame(rows).to_csv(OUT + "/R3_split_paired.csv", index=False)
pd.DataFrame(meta).to_csv(OUT + "/R3_split_meta.csv", index=False)
json.dump(dict(inputs=prov, script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()), open(OUT + "/PROVENANCE_R3_split.json", "w"), indent=1)
print("done", len(rows))
