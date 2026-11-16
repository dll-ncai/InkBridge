#!/usr/bin/env python3
"""R1d read for one base (PREREG_RQ_finalise 0 + A2.5). Rows: idv3 (identity-k3-verifier, comparator), r1d, mirv3, oracle_frame.
oracle_frame (fixed before any row exists): per unit, of the idv3 and mirv3 outputs take the one with higher GT direction_acc
(NaN = lowest); equal -> higher aiou; equal -> identity. Paired deltas vs idv3 on the unit intersection, native protocol, uhwr_eval 0.3.1
rows; CI = loop_tables.boot_idx writer-cluster percentile bootstrap B=2000 (rng 20261029), 95 %; german_onhw_all: unit bootstrap.
PASS(base) = test_ouhdl: direction CI lo > 0 AND AIoU CI lo > -0.01035 (r1d vs idv3); AND each of the 6 other sets: AIoU CI lo > -0.010,
pen-up lo > -0.0141, strict lo > -0.01065, direction lo > -0.01245. Sets missing -> PASS 'not assessed'.
Usage: r1d_read.py PLAN.json BASE"""
import glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from loop_tables import load_cell, unit_table, boot_idx, MET
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
P = json.load(open(sys.argv[1])); base = sys.argv[2]; root = P["root"]; out = f"{root}/read/{base}"; os.makedirs(out, exist_ok=True)
NI = {"penup_f1_v2": -0.0141, "strict_order_acc": -0.01065, "direction_acc": -0.01245}
MS = ["aiou", "penup_f1_v2", "strict_order_acc", "direction_acc", "kendall_tau", "tau_cp", "ldtw"]
def rowfile(s, tag):
    f = f"{root}/score_out/rows/{s}__{base}_{tag}__seed0__rows031.csv"; return f if os.path.exists(f) else None
def paired(um, uc, m, unit_boot):
    j = um[[m, "writer_id"]].join(uc[[m]], rsuffix="_c", how="inner").dropna()
    if len(j) == 0: return dict(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
    d = (j[m] - j[m + "_c"]).to_numpy(); w = np.arange(len(j)) if unit_boot else j.writer_id.to_numpy()
    bs = np.array([d[i].mean() for i in boot_idx(w, 2000)])
    return dict(delta=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n=len(j))
rows, absr, inputs, verdict = [], [], {}, {}
for s in P["bases"][base]["sets"]:
    fs = {t: rowfile(s, t) for t in ("idv3", "r1d", "mirv3")}
    if any(v is None for v in fs.values()): verdict[s] = "not assessed (rows missing)"; continue
    inputs[s] = {t: dict(path=f, sha256=sha(f)) for t, f in fs.items()}
    U = {t: unit_table(load_cell(dict(files={"0": f}))) for t, f in fs.items()}
    a, b = U["idv3"], U["mirv3"].reindex(U["idv3"].index)
    da, db = a.direction_acc.fillna(-np.inf), b.direction_acc.fillna(-np.inf)
    pick = (db > da) | ((db == da) & (b.aiou > a.aiou))
    O = a.copy(); cols = [c for c in MET if c in a.columns]; O.loc[pick, cols] = b.loc[pick, cols]; U["oracle_frame"] = O
    ch = pd.read_csv(f"{root}/apply/{base}/{s}__{base}_choices.csv")
    ub = s == "german_onhw_all"
    for t, u in U.items():
        absr.append(dict(base=base, set=s, row=t, n_units=len(u), frac_mirror=(float(ch.frame.mean()) if t == "r1d" else float(pick.mean()) if t == "oracle_frame" else None),
                         **{m: float(u[m].mean()) for m in MS}))
        if t == "idv3": continue
        for m in MS:
            rows.append(dict(base=base, set=s, method=t, comparator="idv3", metric=m, bootstrap="unit" if ub else "writer_cluster", **paired(u, U["idv3"], m, ub)))
T = pd.DataFrame(rows); A_ = pd.DataFrame(absr)
T.to_csv(f"{out}/R1d_paired_{base}.csv", index=False); A_.to_csv(f"{out}/R1d_absolute_{base}.csv", index=False)
def q(s, m):
    r = T[(T.set == s) & (T.method == "r1d") & (T.metric == m)]; return float(r.ci_lo.iloc[0]) if len(r) else np.nan
sets = list(P["bases"][base]["sets"]); others = [s for s in sets if s != "test_ouhdl"]; clauses = {}
if "test_ouhdl" in inputs:
    clauses["test_ouhdl"] = dict(direction_lo_gt_0=q("test_ouhdl", "direction_acc") > 0, aiou_lo_gt_m01035=q("test_ouhdl", "aiou") > -0.01035)
for s in others:
    if s in inputs:
        clauses[s] = dict(aiou_lo_gt_m010=q(s, "aiou") > -0.010, **{f"{m}_lo_gt_{v}": q(s, m) > v for m, v in NI.items()})
assessed = all(s in inputs for s in sets)
PASS = (all(all(c.values()) for c in clauses.values()) if assessed else "not assessed")
V = dict(base=base, rule="PREREG_RQ_finalise A2.5 R1d PASS", PASS=PASS, clauses={s: {k: bool(v) for k, v in c.items()} for s, c in clauses.items()},
         missing=verdict, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(V, open(f"{out}/R1d_verdict_{base}.json", "w"), indent=1)
prov = dict(kind="rqf_r1d_read", base=base, plan=sys.argv[1], plan_sha256=sha(sys.argv[1]), inputs=inputs,
            code_sha256={f: sha(os.path.join(HERE, f)) for f in ("r1d_read.py", "loop_tables.py")}, agent=P["agent"], lead_reason=P["lead_reason"],
            prereg=P["prereg"], host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
            outputs={f: sha(f"{out}/{f}") for f in os.listdir(out) if f.endswith((".csv", ".json")) and not f.startswith("PROVENANCE")})
json.dump(prov, open(f"{out}/PROVENANCE_read_{base}.json", "w"), indent=1)
print(json.dumps(dict(base=base, PASS=PASS, n_sets=len(inputs))), flush=True)
