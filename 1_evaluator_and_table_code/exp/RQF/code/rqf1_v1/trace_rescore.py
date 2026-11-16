"""RQF R1c TRACE: rescore the saved TRACE predictions (exp/TRACE/out/full/predictions_TRACE_ouhdl.pkl, deterministic, no sampling seed) on the 253 OUHD-L
test lines with uhwr_eval 0.3.1 (all metrics incl. strict_order_acc, penup_f1_v2). GT = L4 eval-space GT (unit_features.to_eval_space(strokes,'line')), exactly as
L4_TRACE l4_trace.py _work020.  Step 'triples' (uhwr_eval 0.2.0 env, builds GT) -> pickle; step 'score' (uhwr_eval 0.3.1 env) -> rows031-style CSVs.
python trace_rescore.py triples --out DIR ; python trace_rescore.py score --out DIR"""
import argparse, os, pickle, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
U = "$U"
ARMS = {"uhwr_f4": "identity", "uhwr_f4_mirror": "mirror_x", "trace_render": "identity", "trace_render_mirror": "mirror_x"}
def triples(a):
    import common as C, unit_features as UF, uhwr_eval as E
    assert E.__version__ == "0.2.0", E.__version__
    tp = pickle.load(open(f"{U}/exp/TRACE/out/full/predictions_TRACE_ouhdl.pkl", "rb"))
    _, lines = C.load_v3(f"{U}/data/ouhdl_ligatures_v3", "test")
    keep = []
    for sid, ln in lines.items():
        gt, canvas, _ = UF.to_eval_space([s[:, :2] for s in ln["raw"]], "line")
        keep.append(dict(sample_id=sid, writer_id=f"ouhdl:{ln['writer_id']}", gt=gt, canvas=canvas, preds={arm: tp.get((sid, arm)) for arm in ARMS}))
    os.makedirs(a.out, exist_ok=True); pickle.dump(keep, open(os.path.join(a.out, "trace_triples.pkl"), "wb"), protocol=4)
    print(len(keep), "lines", sum(1 for k in keep for v in k["preds"].values() if v is None), "missing preds")
def work(t):
    k, arm = t; import uhwr_eval as E
    pr = k["preds"][arm]; rows = []
    for proto in ("native", "standardised"):
        base = dict(set="test253", route_mode=f"TRACE_{arm.replace('_mirror','')}_{ARMS[arm]}", seed=0, specialist=arm, unit_id=k["sample_id"], sample_id=k["sample_id"], writer_id=k["writer_id"], protocol=proto)
        if pr is None: rows.append(dict(base, error="prediction missing")); continue
        try:
            r = E.score(k["gt"], [np.asarray(s, float) for s in pr], k["canvas"], protocol=proto, with_lpips=False, with_frechet=False)
            rows.append(dict(base, error="", **{kk: v for kk, v in r.items() if np.isscalar(v)}))
        except Exception as e: rows.append(dict(base, error=f"{type(e).__name__}: {e}"))
    return arm, rows
def score(a):
    from multiprocessing import get_context
    import uhwr_eval as E
    assert E.__version__ == "0.3.1", E.__version__
    keep = pickle.load(open(os.path.join(a.out, "trace_triples.pkl"), "rb"))
    tasks = [(k, arm) for k in keep for arm in ARMS]; out = {arm: [] for arm in ARMS}
    with get_context("spawn").Pool(a.workers) as pool:
        for arm, rows in pool.imap_unordered(work, tasks, chunksize=4): out[arm] += rows
    os.makedirs(os.path.join(a.out, "rows"), exist_ok=True)
    for arm, rows in out.items():
        df = pd.DataFrame(rows); df["evaluator"] = "uhwr_eval 0.3.1"
        df.to_csv(os.path.join(a.out, "rows", f"test253__TRACE_{arm.replace('_mirror','')}_{ARMS[arm]}__seed0__rows031.csv"), index=False)
        print(arm, len(df), int((df.error.fillna("") != "").sum()), "errors")
if __name__ == "__main__":  # guard added 12 Nov 2026 (repair): spawn workers re-imported this file and re-ran score() -> nested pools, hang
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["triples", "score"]); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); {"triples": triples, "score": score}[a.cmd](a)
