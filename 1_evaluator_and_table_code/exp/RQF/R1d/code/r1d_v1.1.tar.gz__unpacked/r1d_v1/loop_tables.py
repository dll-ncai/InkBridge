"""[2026-10-30: contrasts may carry a third element "own" = each cell uses all of its own seed keys (pooled comparator)]
Loop track daily read tables (uhwr method round). Inputs: lane_score.py rows031 CSVs (uhwr_eval 0.3.1, protocol native) and
loop_endpoint.py per-unit endpoint CSVs. Spec: JSON {cells: {name: {device, files: {seed: rows_csv}, endpoint: {seed: csv}}},
contrasts: [[method, comparator], ...]}.
Per unit, metrics are averaged over the cell's seeds (seed-mean); a contrast pairs units present in both cells and uses the
seeds each cell has (method seeds 0-2 vs comparator seeds 0-2, or seed 0 vs seed 0 when one side has only seed 0 is NOT
mixed: the contrast uses the intersection of seed sets). Writer-cluster bootstrap B=2000 (rng 20261029), 95 % percentile CI.
Metrics: aiou, penup_f1_v2, strict_order_acc, direction_acc, kendall_tau (units defined in both), tau_cp (per seed: units with
n_gt_strokes >= 2, undefined tau = 0; then seed-mean), ldtw, stroke_count_err (signed, pred - GT), ep3 (matched-stroke
endpoint error <= 3 px; ratio of sums), stroke buckets (GT 1/2/3/4+) for penup_f1_v2 and strict_order_acc."""
import argparse, json, hashlib, os
import numpy as np, pandas as pd
MET = ["aiou", "penup_f1_v2", "strict_order_acc", "direction_acc", "kendall_tau", "tau_cp", "ldtw", "stroke_count_err"]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def load_cell(c, seeds=None):
    fr = []
    for s, f in c["files"].items():
        if seeds is not None and int(s) not in seeds: continue
        d = pd.read_csv(f); d = d[d.protocol == "native"].copy(); d["seed_k"] = int(s)
        d["tau_cp"] = np.where(d.n_gt_strokes >= 2, d.kendall_tau.fillna(0.0), np.nan)
        if c.get("endpoint") and str(s) in c["endpoint"]:
            e = pd.read_csv(c["endpoint"][str(s)])[["unit_id", "ep_n", "ep_within3"]]; d = d.merge(e, on="unit_id", how="left")
        fr.append(d)
    return pd.concat(fr, ignore_index=True)
def unit_table(d):
    cols = MET + [c for c in ("ep_n", "ep_within3") if c in d]
    g = d.groupby("unit_id")
    u = g[cols].mean()
    u["writer_id"] = g["writer_id"].first(); u["n_gt_strokes"] = g["n_gt_strokes"].first()
    return u
def boot_idx(writers, B=2000, seed=20261029):
    w = np.asarray(writers); uw = np.unique(w); rng = np.random.default_rng(seed)
    pos = {k: np.where(w == k)[0] for k in uw}
    for _ in range(B):
        pick = rng.choice(uw, len(uw), replace=True)
        yield np.concatenate([pos[k] for k in pick])
def paired(um, uc, metric, B=2000):
    j = um[[metric, "writer_id"]].join(uc[[metric]], rsuffix="_c", how="inner").dropna()
    if len(j) == 0: return dict(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
    dlt = (j[metric] - j[metric + "_c"]).to_numpy(); est = dlt.mean()
    bs = np.array([dlt[i].mean() for i in boot_idx(j.writer_id.to_numpy(), B)])
    return dict(delta=est, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n=len(j))
def paired_ratio(um, uc, B=2000):
    j = um[["ep_n", "ep_within3", "writer_id"]].join(uc[["ep_n", "ep_within3"]], rsuffix="_c", how="inner").dropna()
    if len(j) == 0: return dict(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
    a, an, b, bn = (j[c].to_numpy() for c in ("ep_within3", "ep_n", "ep_within3_c", "ep_n_c"))
    f = lambda i: a[i].sum() / an[i].sum() - b[i].sum() / bn[i].sum()
    est = f(np.arange(len(j))); bs = np.array([f(i) for i in boot_idx(j.writer_id.to_numpy(), B)])
    return dict(delta=est, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n=len(j))
def bucket(n): return np.where(n >= 4, "4+", n.astype(int).astype(str))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    S = json.load(open(a.spec)); os.makedirs(a.out, exist_ok=True)
    inputs = {n: {k: dict(path=f, sha256=sha(f)) for k, f in c["files"].items()} for n, c in S["cells"].items()}
    absr, parr, bkt = [], [], []
    for n, c in S["cells"].items():
        d = load_cell(c)
        for s, ds in d.groupby("seed_k"):
            row = dict(cell=n, device=c["device"], seed=s, n_units=ds.unit_id.nunique())
            for m in MET: row[m] = ds[m].mean()
            row["tau_defined_frac"] = ds.kendall_tau.notna().mean()
            if "ep_n" in ds: row["ep3"] = ds.ep_within3.sum() / ds.ep_n.sum()
            absr.append(row)
        u = unit_table(d); u["bucket"] = bucket(u.n_gt_strokes.to_numpy())
        for b, ub in u.groupby("bucket"):
            bkt.append(dict(cell=n, device=c["device"], kind="absolute", bucket=b, n_units=len(ub), penup_f1_v2=ub.penup_f1_v2.mean(), strict_order_acc=ub.strict_order_acc.mean()))
    A = pd.DataFrame(absr)
    seedmean = A.groupby(["cell", "device"]).mean(numeric_only=True).reset_index().assign(seed="mean")
    A = pd.concat([A, seedmean], ignore_index=True)
    for ent in S["contrasts"]:
        mth, cmp_ = ent[0], ent[1]; own = len(ent) > 2 and ent[2] == "own"
        cm, cc = S["cells"][mth], S["cells"][cmp_]
        if own:   # each cell uses all its own seed keys (e.g. comparator pooled over training seeds); per-unit means are paired
            dm, dcmp = load_cell(cm), load_cell(cc); seeds = ["own"]
        else:
            seeds = sorted(set(map(int, cm["files"])) & set(map(int, cc["files"])))
            dm, dcmp = load_cell(cm, seeds), load_cell(cc, seeds)
        um, uc = unit_table(dm), unit_table(dcmp)
        for m in MET:
            parr.append(dict(method=mth, comparator=cmp_, dev_method=cm["device"], dev_comparator=cc["device"], seeds=" ".join(map(str, seeds)), metric=m, **paired(um, uc, m)))
        if "ep_n" in um and "ep_n" in uc:
            parr.append(dict(method=mth, comparator=cmp_, dev_method=cm["device"], dev_comparator=cc["device"], seeds=" ".join(map(str, seeds)), metric="ep3", **paired_ratio(um, uc)))
        um["bucket"] = bucket(um.n_gt_strokes.to_numpy())
        for b in ("1", "2", "3", "4+"):
            ub = um[um.bucket == b]
            for m in ("penup_f1_v2", "strict_order_acc"):
                bkt.append(dict(cell=mth, comparator=cmp_, device=cm["device"], kind="paired", bucket=b, metric=m, seeds=" ".join(map(str, seeds)), **paired(ub, uc, m)))
    for m1, c1, m2, c2 in S.get("did", []):   # per-unit (m1 - c1) - (m2 - c2): cross-device comparison through same-device anchors
        cells = [S["cells"][x] for x in (m1, c1, m2, c2)]
        seeds = sorted(set.intersection(*[set(map(int, c["files"])) for c in cells]))
        U4 = [unit_table(load_cell(c, seeds)) for c in cells]
        for m in MET:
            j = U4[0][[m, "writer_id"]].join(U4[1][[m]], rsuffix="_c1", how="inner").join(U4[2][[m]], rsuffix="_m2", how="inner").join(U4[3][[m]], rsuffix="_c2", how="inner").dropna()
            dd = ((j[m] - j[m + "_c1"]) - (j[m + "_m2"] - j[m + "_c2"])).to_numpy()
            if len(dd) == 0: continue
            bs = np.array([dd[i].mean() for i in boot_idx(j.writer_id.to_numpy())])
            parr.append(dict(method=f"({m1} - {c1})", comparator=f"({m2} - {c2})", dev_method=cells[0]["device"], dev_comparator=cells[2]["device"],
                             seeds=" ".join(map(str, seeds)), metric=m, delta=dd.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n=len(dd)))
    P = pd.DataFrame(parr); K = pd.DataFrame(bkt)
    A.to_csv(f"{a.out}/day_absolute.csv", index=False); P.to_csv(f"{a.out}/day_paired.csv", index=False); K.to_csv(f"{a.out}/day_buckets.csv", index=False)
    json.dump(dict(spec=S, inputs=inputs, code_sha256=sha(__file__)), open(f"{a.out}/day_tables_inputs.json", "w"), indent=1)
    print(P.to_string(max_rows=400)); print(A[A.seed == "mean"].to_string())
if __name__ == "__main__": main()
