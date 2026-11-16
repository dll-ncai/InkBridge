#!/usr/bin/env python3
"""Reviewer follow-up (15 Nov 2026), descriptive, not pre-registered.
Part A (Tables 5/6): per-training-seed paired delta of each warm arm (one training seed, its sampling seeds 0-2 averaged)
  minus the pooled 3-seed comparator used in the paper (Table 5: pooled COMP-p01 = FT1%; Table 6: pooled LUOFT p10 = FT10%).
  Reuses build_tables_rqf.py helpers unchanged (exec of its definitions, before its main loop); pooled rows recomputed as a check.
Part B (Tables 11/12): Kendall tau (and strict order / pen-up F1 as a check) rep3 - pool3 for every R3 spec in read_all3,
  writer-cluster bootstrap from loop_tables (B=2000, rng 20261029), same pb() as rq3_split.py."""
import sys, os, json, glob, hashlib
import numpy as np, pandas as pd
RF_OUT = sys.argv[1]; os.makedirs(RF_OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
prov = {"script_sha256": sha(__file__), "inputs": {}}
BT = "$U/exp/RQF/code/rqf_v1/build_tables_rqf.py"; prov["inputs"][BT] = sha(BT)
s = open(BT).read(); s = s[:s.index("COMP_TAGS_ALL = dict")]
sys.argv = [BT, RF_OUT + "/unused"]; exec(compile(s, BT, "exec"), globals())
prov["inputs"]["build_tables_a9.py"] = sha(T.__file__)
A = []
def rec(table, arm, seed, a, b):
    if a is None or b is None: A.append(dict(table=table, arm=arm, train_seed=seed, missing=True)); return
    p = T.paired(a, b, "test_ouhdl")
    r = dict(table=table, arm=arm, train_seed=seed, n_units=p["aiou"]["n"], n_samp_seeds_min=int(a["n_seeds"].min()), missing=False)
    for m in M: r.update({f"{m}_d": p[m]["delta"], f"{m}_lo": p[m]["ci_lo"], f"{m}_hi": p[m]["ci_hi"]})
    A.append(r)
SS = SEEDS; cp = comp("test_ouhdl"); f10 = luo("LUOFT", "test_ouhdl", "p10")
for arm in WARMS:
    for sd in ("pooled",) + SEEDS:
        SS = SEEDS if sd == "pooled" else (sd,)
        rec("T5_p01_vs_FT1", arm, sd, warm(arm, "ft", "test_ouhdl"), cp)
        rec("T6_p10_vs_FT10", arm, sd, warm(arm, "ft", "test_ouhdl", "p10"), f10)
pd.DataFrame(A).to_csv(RF_OUT + "/T56_per_seed.csv", index=False)
# Part C: step-matched control (SFT = LUOC40) at 1/10/30 %, 30 % arms vs FT1 %, and Bonferroni(=conservative Holm, 2 arms) lower bounds
def paired_q(a, b, q):
    j = a[M + ["writer_id"]].join(b[M], rsuffix="_c", how="inner"); cl = j.writer_id.to_numpy()
    D = np.column_stack([(j[m] - j[m + "_c"]).to_numpy() for m in M]); V = ~np.isnan(D); S = np.where(V, D, 0.0)
    bs = np.array([S[i].sum(0) / np.maximum(V[i].sum(0), 1) for i in T.boot_idx(cl)])
    return dict(zip(M, np.percentile(bs, q, axis=0)))
C = []
def recC(cell, arm, a, b, cmpn):
    if a is None or b is None: C.append(dict(cell=cell, arm=arm, comparator=cmpn, missing=True)); return
    p = T.paired(a, b, "test_ouhdl")
    r = dict(cell=cell, arm=arm, comparator=cmpn, n_train_seeds_arm=a.attrs.get("n_train_seeds"), n_train_seeds_cmp=b.attrs.get("n_train_seeds"), n_units=p["aiou"]["n"], missing=False)
    for m in M: r.update({f"{m}_d": p[m]["delta"], f"{m}_lo": p[m]["ci_lo"], f"{m}_hi": p[m]["ci_hi"], f"{m}_n": p[m]["n"]})
    if cell.startswith("T5_holm"):
        for m, v in paired_q(a, b, 1.25).items(): r[f"{m}_lo_bonf2"] = v
    C.append(r)
SS = SEEDS; missing.clear()
for frac in ("p01", "p10", "p30"):
    sft = luo("LUOC40", "test_ouhdl", frac)
    for arm in WARMS:
        recC(f"SFT_{frac}", arm, warm(arm, "ft", "test_ouhdl", frac), sft, f"LUOC40_{frac}")
    recC(f"SFT_{frac}_vs_FT1", "LUOC40", sft, cp, "pooled_COMP_p01")
for arm in WARMS:
    recC("p30_vs_FT1", arm, warm(arm, "ft", "test_ouhdl", "p30"), cp, "pooled_COMP_p01")
    recC("T5_holm", arm, warm(arm, "ft", "test_ouhdl"), cp, "pooled_COMP_p01")
pd.DataFrame(C).to_csv(RF_OUT + "/stepmatched_p30_holm.csv", index=False)
json.dump(sorted(set(missing)), open(RF_OUT + "/missing_inputs.json", "w"), indent=1)

# Part B
sys.path.insert(0, "$U/exp/RQF/code/rq3_v1")
from loop_tables import load_cell, unit_table, boot_idx
prov["inputs"]["loop_tables.py"] = sha("$U/exp/RQF/code/rq3_v1/loop_tables.py")
def pb(um, uc, m):
    j = um[[m, "writer_id"]].join(uc[[m]], rsuffix="_c", how="inner").dropna()
    if len(j) < 20: return np.nan, np.nan, np.nan, len(j)
    d = (j[m] - j[m + "_c"]).to_numpy(); bs = np.array([d[i].mean() for i in boot_idx(j.writer_id.to_numpy())])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(j)
B = []
for sp in sorted(glob.glob("$U/exp/RQF/R3/read_all3/spec_*.json")):
    prov["inputs"][sp] = sha(sp); S = json.load(open(sp))
    base, st = os.path.basename(sp)[5:-5].split("__")
    UP, UQ = unit_table(load_cell(S["cells"]["COMP_pool3"])), unit_table(load_cell(S["cells"]["COMP_rep3"]))
    for m in ("kendall_tau", "strict_order_acc", "penup_f1_v2"):
        d, lo, hi, n = pb(UQ, UP, m); B.append(dict(base=base, set=st, metric=m, delta=d, ci_lo=lo, ci_hi=hi, n=n))
pd.DataFrame(B).to_csv(RF_OUT + "/R3_tau_all_cells.csv", index=False)
json.dump(prov, open(RF_OUT + "/PROVENANCE_reviewer_followup.json", "w"), indent=1)
print("done", len(A), len(B))
