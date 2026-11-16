"""Lane per-unit scorer (generalisation of routeA_score.py for the VPS lane_sample.py sidecars: route_mode falls back to
"<label>_ms<milestone>", gt path to npz_clusterA).  Route A per-unit scorer (copy of D3 Stage A d3a_score.py; differences: sidecar gt path falls back to "npz",
gt sha256 looked up from plans/routing_waterfall/staging_sha256_clusterA.txt when the sidecar has no source_sha256_expected).
Original docstring: D3 Stage A per-unit scorer (uhwr_eval 0.3.1), same call as the table builder / rescore_031.py:
score(gt, pred, (64, 64), protocol, with_lpips=False), native + standardised, E0 adapter decode/unpack_ink.
NOTE: e0/code_4cd40f57 contains a vendored uhwr_eval 0.1.0; it is appended LAST to sys.path so that the 0.3.1 on PYTHONPATH wins
(rescore_031.py / e0_score.py put the E0 dir first, which shadows 0.3.1 - their rows lack penup_f1_v2).

python d3a_score.py --preds-dir DIR --out OUT --e0code E0CODE --procs 16 [--limit N] [--only GLOB]
Per prediction file -> OUT/rows/<stem>__rows031.csv  (columns: set, route_mode, seed, specialist, unit_id, sample_id,
writer_id, protocol, <uhwr_eval metrics>, error, evaluator, preds_file, preds_sha256, gt_npz, gt_sha256, dedup_from)
Identical prediction arrays (same gt npz + unit + bytes) are scored once and copied (dedup_from = the file scored).
"""
import argparse, glob, hashlib, json, os, re, sys, time
from multiprocessing import get_context
import numpy as np, pandas as pd

E0CODE = None; GT = {}
STAGE_SHA = {}
_sl = "$U/plans/routing_waterfall/staging_sha256_clusterA.txt"
if os.path.exists(_sl):
    for _line in open(_sl):
        _p = _line.split()
        if len(_p) == 2: STAGE_SHA[_p[1]] = _p[0]

UE_FILE = None
def _init(e0code):
    """uhwr_eval 0.3.1 must come from PYTHONPATH; the E0 code dir carries a vendored 0.1.0 copy, so it goes LAST on sys.path."""
    global E0CODE, UE_FILE
    E0CODE = e0code; os.environ.setdefault("OMP_NUM_THREADS", "1")
    import uhwr_eval
    assert uhwr_eval.__version__ == "0.3.1", (uhwr_eval.__version__, uhwr_eval.__file__)
    UE_FILE = uhwr_eval.__file__
    sys.path.append(e0code)

def _gt(path):
    if path not in GT:
        GT.clear()
        z = np.load(path, allow_pickle=False)
        GT[path] = {k: z[k] for k in ["gt_points", "gt_stroke_id", "gt_offsets", "unit_ids", "sample_ids", "writer_ids"] if k in z.files}
    return GT[path]

def _work(t):
    gt_path, i, arr = t
    import e0_adapter as A
    from uhwr_eval import score
    Z = _gt(gt_path)
    gt = A.unpack_ink(Z["gt_points"], Z["gt_stroke_id"], Z["gt_offsets"], i)
    try:
        pr = A.decode(arr); err0 = ""
    except Exception as e:
        pr, err0 = [], f"predict {type(e).__name__}: {e}"
    out = []
    for prot in ("native", "standardised"):
        try:
            out.append({"protocol": prot, **score(gt, pr, (64, 64), protocol=prot, with_lpips=False), "error": err0, "uhwr_eval_file": UE_FILE})
        except Exception as e:
            out.append({"protocol": prot, "error": (err0 + " | " if err0 else "") + f"{type(e).__name__}: {e}"})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--e0code", required=True)
    ap.add_argument("--procs", type=int, default=16); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--only", default="*_preds.npz")
    a = ap.parse_args()
    import uhwr_eval
    assert uhwr_eval.__version__ == "0.3.1", (uhwr_eval.__version__, uhwr_eval.__file__)
    assert "penup_f1_v2" in uhwr_eval.METRICS
    sys.path.append(a.e0code)
    print("uhwr_eval", uhwr_eval.__version__, uhwr_eval.__file__, flush=True)
    os.makedirs(os.path.join(a.out, "rows"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.preds_dir, a.only)))
    print(len(files), "prediction files", flush=True)
    pool = get_context("spawn").Pool(a.procs, initializer=_init, initargs=(a.e0code,))
    cache = {}   # (gt_path, gt_index, sha1(pred bytes)) -> (rows, source file)
    summary = []
    for f in files:
        t0 = time.time(); stem = os.path.basename(f).replace("_preds.npz", "")
        outp = os.path.join(a.out, "rows", stem + "__rows031.csv")
        if os.path.exists(outp):
            print("exists, skip", stem, flush=True); continue
        side = json.load(open(f.replace("_preds.npz", "_preds.json")))
        gt_path = side.get("source_npz_clusterA") or side.get("npz_clusterA") or side["npz"]
        if "route_mode" not in side: side["route_mode"] = f"{side['label']}_ms{side['milestone']}"
        gt_sha = side.get("source_sha256_expected") or STAGE_SHA.get(os.path.relpath(gt_path, "$U"), "")
        z = np.load(f, allow_pickle=False); P = z["preds"]; pu = [str(u) for u in z["unit_ids"]]
        spec = [str(s) for s in z["specialist"]] if "specialist" in z.files else [side["route_mode"]] * len(pu)
        G = np.load(gt_path, allow_pickle=False)
        gu = [str(u) for u in G["unit_ids"]]; pos = {u: i for i, u in enumerate(gu)}
        gs = [str(s) for s in G["sample_ids"]]; gw = [int(w) for w in G["writer_ids"]]
        keys = [(gt_path, pos[u], hashlib.sha1(np.ascontiguousarray(P[k]).tobytes()).hexdigest()) for k, u in enumerate(pu)]
        if a.limit: keys = keys[: a.limit]
        todo = [k for k, key in enumerate(keys) if key not in cache]
        res = pool.imap(_work, [(keys[k][0], keys[k][1], P[k]) for k in todo], chunksize=8)
        for k, rr in zip(todo, res):
            cache[keys[k]] = (rr, stem)
        rows = []
        for k, key in enumerate(keys):
            rr, src = cache[key]; i = key[1]
            for r in rr:
                rows.append(dict(set=side["set"], route_mode=side["route_mode"], seed=int(side["seed"]), specialist=spec[k],
                                 unit_id=pu[k], sample_id=gs[i], writer_id=gw[i], **r,
                                 dedup_from="" if src == stem else src))
        df = pd.DataFrame(rows)
        df["evaluator"] = "uhwr_eval 0.3.1"; df["preds_file"] = os.path.basename(f); df["preds_sha256"] = side.get("preds_sha256", "")
        df["gt_npz"] = gt_path; df["gt_sha256"] = gt_sha
        df.to_csv(outp, index=False)
        s = dict(file=os.path.basename(f), n_units=len(keys), n_scored=len(todo), n_dedup=len(keys) - len(todo),
                 n_error=int((df["error"].fillna("") != "").sum()), seconds=round(time.time() - t0, 1))
        summary.append(s); print(s, flush=True)
        pd.DataFrame(summary).to_csv(os.path.join(a.out, "score_summary_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", a.only) + ".csv"), index=False)
    pool.close(); pool.join()
    json.dump(dict(uhwr_eval=uhwr_eval.__version__, uhwr_eval_file=uhwr_eval.__file__, e0code=a.e0code, procs=a.procs,
                   n_files=len(files), only=a.only, slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename),
              open(os.path.join(a.out, "score_env_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", a.only) + ".json"), "w"), indent=1)

if __name__ == "__main__":
    main()
