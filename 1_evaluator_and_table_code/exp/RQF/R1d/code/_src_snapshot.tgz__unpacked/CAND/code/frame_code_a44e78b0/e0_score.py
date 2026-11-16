"""Score predictions on E0 evaluation arrays with uhwr_eval (native + standardised protocols).

python e0_score.py e0   --npz e0_test.npz --preds preds.npz --method e0_... --input rendered --out rows.csv
python e0_score.py geo  --npz e0_test.npz --input rendered|dataset --out rows.csv    (geometric dots-last
        baseline run here on the f=4 inputs stored in the npz, exactly as uhwr_baselines/run_ouhdl.py)
python e0_score.py summary --rows a.csv b.csv ... --out summary.csv

LPIPS is not computed (with_lpips=False); every other uhwr_eval metric is.
Summary: mean over units with defined value; 95% CI = cluster bootstrap over lines (B=2000, seed 0,
uhwr_baselines.aggregate.boot_ci); order metrics are reported for the native protocol only.
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

import e0_adapter as A

Z = None
P = None
ARGS = None


def _init(npz, preds, args):
    global Z, P, ARGS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    z = np.load(npz, allow_pickle=False)
    keys = ["gt_points", "gt_stroke_id", "gt_offsets", "unit_ids", "sample_ids", "writer_ids"]
    if args["kind"] == "geo":
        keys.append("cond_rendered_f4" if args["input"] == "rendered" else "cond_dataset_f4")
    Z = {k: z[k] for k in keys}
    P = None if preds is None else {"preds": np.load(preds, allow_pickle=False)["preds"]}
    ARGS = args


def _pred(i, k):
    if ARGS["kind"] == "e0":
        return A.decode(P["preds"][k])
    from uhwr_baselines import geometric as G
    img = Z["cond_rendered_f4"][i] if ARGS["input"] == "rendered" else Z["cond_dataset_f4"][i]
    return [np.asarray(s, float) / 4.0 for s in G.recover(img, dots="last", unit_height=56 * 4)]


def _work(ik):
    i, k = ik
    from uhwr_eval import score
    gt = A.unpack_ink(Z["gt_points"], Z["gt_stroke_id"], Z["gt_offsets"], i)
    t0 = time.time()
    try:
        pr = _pred(i, k)
        err0 = ""
    except Exception as e:
        pr, err0 = [], f"predict {type(e).__name__}: {e}"
    dt = time.time() - t0
    base = dict(unit_id=str(Z["unit_ids"][i]), sample_id=str(Z["sample_ids"][i]), writer_id=int(Z["writer_ids"][i]),
                method=ARGS["method"], input=ARGS["input"], runtime_s=dt)
    rows = []
    for prot in ("native", "standardised"):
        try:
            r = score(gt, pr, (64, 64), protocol=prot, with_lpips=False)
            rows.append({**base, "protocol": prot, **r, "error": err0})
        except Exception as e:
            rows.append({**base, "protocol": prot, "error": (err0 + " | " if err0 else "") + f"{type(e).__name__}: {e}"})
    return rows


def run(kind, npz, preds, method, inp, out, procs):
    args = dict(kind=kind, method=method, input=inp)
    zu = [str(u) for u in np.load(npz)["unit_ids"]]
    if preds is not None:
        pu = [str(u) for u in np.load(preds)["unit_ids"]]
        pos = {u: k for k, u in enumerate(zu)}
        todo = [(pos[u], k) for k, u in enumerate(pu)]          # (GT index, prediction index)
    else:
        todo = [(k, k) for k in range(len(zu))]
    with get_context("spawn").Pool(procs, initializer=_init, initargs=(npz, preds, args)) as pool:
        rows = [r for rr in pool.imap(_work, todo, chunksize=16) for r in rr]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, len(rows), "rows")


SUMMARY_METRICS = ["aiou", "ldtw", "dtw", "strict_order_acc", "kendall_tau", "direction_acc", "penup_f1",
                   "stroke_count_abs_err", "stroke_match_recall", "loop_orient_acc", "chamfer", "hausdorff",
                   "frechet", "rmse"]
ORDER = {"strict_order_acc", "kendall_tau", "direction_acc", "stroke_count_abs_err", "stroke_match_recall",
         "loop_orient_acc"}


def summary(paths, out):
    from uhwr_baselines.aggregate import boot_ci
    df = pd.concat([pd.read_csv(p).assign(_file=os.path.basename(p)) for p in paths])
    res = []
    for (meth, inp, prot, f), g in df.groupby(["method", "input", "protocol", "_file"], sort=False):
        for m in SUMMARY_METRICS:
            if prot == "standardised" and m in ORDER:
                continue
            v = g[m].astype(float)
            lo, hi = boot_ci(v, g["sample_id"])
            res.append(dict(method=meth, input=inp, protocol=prot,
                            protocol_note="order/stroke metric: protocol-insensitive, reported once" if m in ORDER else "",
                            metric=m, mean=float(v.mean()), ci95_lo=lo, ci95_hi=hi, n_valid=int(v.notna().sum()),
                            n_units=int(g.unit_id.nunique()), n_lines=int(g.sample_id.nunique()),
                            n_errors=int((g["error"].fillna("") != "").sum()), n_empty_pred=int(g["empty_pred"].fillna(0).sum()),
                            rows_file=f))
    pd.DataFrame(res).to_csv(out, index=False)
    print(out, len(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["e0", "geo", "summary"])
    ap.add_argument("--npz"); ap.add_argument("--preds"); ap.add_argument("--method", default="geo_rtl")
    ap.add_argument("--input", default="rendered"); ap.add_argument("--out", required=True)
    ap.add_argument("--rows", nargs="*"); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.kind == "summary":
        summary(a.rows, a.out)
    else:
        run(a.kind, a.npz, a.preds if a.kind == "e0" else None, a.method, a.input, a.out, a.procs)


if __name__ == "__main__":
    main()
