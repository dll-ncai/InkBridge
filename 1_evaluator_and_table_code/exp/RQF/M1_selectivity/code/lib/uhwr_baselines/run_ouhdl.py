"""Run geometric + Momna baselines on the OUHD-L test split and score with uhwr_eval.

python run_ouhdl.py UNITS_PKL OUHDL_ROOT OUT_DIR [n_workers]
Writes OUT_DIR/rows_<sample>.jsonl (per-sample scores) and OUT_DIR/pred_<sample>.pkl.
"""
import json
import os
import pickle
import sys
import time
import traceback

os.environ.setdefault("UHWR_TORCH_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "uhwr_eval_pkg")]

import numpy as np  # noqa: E402

from uhwr_eval import score  # noqa: E402
from uhwr_baselines import geometric as G  # noqa: E402
from uhwr_baselines import inputs as I  # noqa: E402

LINES = None
MOMNA = None
ROOT = None


def _init(pkl, root):
    global LINES, MOMNA, ROOT
    import torch
    torch.set_num_threads(1)
    LINES = {d["sample_id"]: d for d in pickle.load(open(pkl, "rb"))}
    ROOT = root
    from uhwr_baselines.momna import MomnaStages123
    MOMNA = MomnaStages123()


def _score_rows(base, gt, pred, canvas):
    rows = []
    for prot in ("native", "standardised"):
        try:
            r = score(gt, pred, canvas, protocol=prot)
            rows.append({**base, "protocol": prot, **r, "error": ""})
        except Exception as e:  # never silently drop: record the failure
            rows.append({**base, "protocol": prot, "error": f"{type(e).__name__}: {e}"})
    return rows


def run_line(sid):
    d = LINES[sid]
    L = d["line"]
    rows, preds = [], {}
    img_ds = I.dataset_image(ROOT, d["img"])
    aff_ds = I.dataset_line(L, img_ds)
    meta = dict(sample_id=sid, writer_id=d["writer_id"])
    # ---------------- line level
    img_r4, aff_r4 = I.rendered_line(L, 4.0)
    img_r1, aff_r1 = I.rendered_line(L, 1.0)
    jobs = []
    for cond, img, aff, uh in (("rendered", img_r4, aff_r4, 56 * 4), ("dataset", img_ds, aff_ds, img_ds.shape[0])):
        for var, dots in (("geo_rtl", "last"), ("geo_rtl_local", "after_body")):
            t0 = time.time()
            p = I.to_eval(G.recover(img, dots=dots, unit_height=uh), aff)
            jobs.append((var, cond, p, time.time() - t0))
    for cond, img, aff in (("rendered", img_r1, aff_r1), ("dataset", img_ds, aff_ds)):
        t0 = time.time()
        out = MOMNA.recover(img)
        dt = time.time() - t0
        jobs.append(("momna_s123", cond, I.to_eval(out["emitted"], aff), dt))
        jobs.append(("momna_s123_subpaths", cond, I.to_eval(out["subpaths"], aff), dt))
    for var, cond, p, dt in jobs:
        preds[("line", var, cond, sid)] = p
        base = dict(meta, unit="line", unit_id=sid, method=var, input=cond, canvas_w=L.canvas[1],
                    runtime_s=dt)
        rows += _score_rows(base, L.ink, p, L.canvas)
    # ---------------- ligature level (geometric only)
    for u in L.units:
        for cond in ("rendered", "dataset"):
            if cond == "rendered":
                img, aff = I.rendered_unit(u, 4.0)
            else:
                img, aff = I.dataset_unit(L, u, img_ds, 4.0)
            t0 = time.time()
            p = I.to_eval(G.recover(img, dots="last", unit_height=56 * 4), aff)
            dt = time.time() - t0
            preds[("ligature", "geo_rtl", cond, u["unit_id"])] = p
            base = dict(meta, unit="ligature", unit_id=u["unit_id"], method="geo_rtl", input=cond,
                        canvas_w=64, runtime_s=dt)
            rows += _score_rows(base, u["ink"], p, u["canvas"])
    return sid, rows, preds


def main():
    pkl, root, out = sys.argv[1:4]
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(out, exist_ok=True)
    sids = [d["sample_id"] for d in pickle.load(open(pkl, "rb"))]
    todo = [s for s in sids if not os.path.exists(os.path.join(out, f"rows_{s}.jsonl"))]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(nw, initializer=_init, initargs=(pkl, root)) as pool:
        for k, res in enumerate(pool.imap_unordered(_safe, todo)):
            sid, rows, preds, err = res
            if err:
                with open(os.path.join(out, "errors.txt"), "a") as f:
                    f.write(f"{sid}\t{err}\n")
                continue
            pickle.dump(preds, open(os.path.join(out, f"pred_{sid}.pkl"), "wb"))
            with open(os.path.join(out, f"rows_{sid}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            if k % 10 == 0:
                print(f"{k+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)


def _safe(sid):
    try:
        s, r, p = run_line(sid)
        return s, r, p, ""
    except Exception:
        return sid, [], {}, traceback.format_exc().replace("\n", " | ")


if __name__ == "__main__":
    main()
