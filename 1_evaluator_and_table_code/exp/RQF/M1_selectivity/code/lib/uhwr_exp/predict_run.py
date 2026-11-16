"""Recover trajectories for every OUHD-L test line and v3 ligature under every E2 input condition.

python -m uhwr_exp.predict_run --bundle bundle --out out/pred [--workers N] [--limit K] [--conds ...]

Input conditions (E2 'input renderer' axis):
  r_w{w}_h{H}  our uniform-width anti-aliased renderer (uhwr_eval.render_image) at canvas height
               H in {64, 128, 224, 256} px with stroke width round(w * H / 64) px, w in {1,2,3,4}
               (i.e. w px per 64 px of canvas height).  Ligature canvas H x H (unit ink in the
               central 56/64); line canvas H x (W * H / 64).
  dataset      the OUHD-L img/ PNG (ligatures: crop + mask of the unit's own strokes, resampled
               to 256 x 256 as in uhwr_baselines.inputs.dataset_unit, f = 4; lines: as shipped).
The earlier baseline runs used r_w2_h256 (geometric, f = 4) and r_w2_h64 (Momna, f = 1).

Methods: geo_rtl (dots last), geo_rtl_local (dots after nearest body) at line and ligature
level (uhwr_baselines.geometric, unit_height = 56 * f); momna_s123 (emitted) and
momna_s123_subpaths at line level (uhwr_baselines.momna, unchanged repo pipeline).
Writes out/pred/pred_<sample_id>.pkl = {(level, method, cond, id): ink in eval px} and
out/pred/runtime_<sample_id>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("UHWR_TORCH_THREADS", "1")

import numpy as np  # noqa: E402

WIDTHS = (1, 2, 3, 4)
HEIGHTS = (64, 128, 224, 256)
CONDS = [f"r_w{w}_h{h}" for w in WIDTHS for h in HEIGHTS] + ["dataset"]

G = {}


def _init(bundle, with_momna):
    import torch
    torch.set_num_threads(1)
    from uhwr_exp.common import load_lines
    G["lines"] = load_lines(bundle)
    G["bundle"] = bundle
    import pandas as pd
    G["idx"] = pd.read_csv(os.path.join(bundle, "ligatures_v3_test.csv"))
    if with_momna:
        from uhwr_baselines.momna import MomnaStages123
        G["momna"] = MomnaStages123()


def _parse(c):
    w, h = c[3:].split("_h")
    return int(w), int(h)


def render(ink, canvas, cond):
    from uhwr_eval import render_image
    w, H = _parse(cond)
    f = H / 64.0
    Hc, Wc = int(round(canvas[0] * f)), int(round(canvas[1] * f))
    th = max(1, int(round(w * f)))
    return render_image([s * f for s in ink], (Hc, Wc), th, antialias=True), f


def units_for_line(sid, line):
    import ast
    from uhwr_eval import apply_transform, fit_transform_unit
    out = []
    for r in G["idx"][G["idx"].sample_id == sid].itertuples(index=False):
        order = sorted(ast.literal_eval(r.stroke_idx))
        sl = [line.ink[i] for i in order]
        s, tx, ty = fit_transform_unit(sl, 64, 4)
        out.append(dict(unit_id=r.unit_id, stroke_idx=order, T=(s, tx, ty),
                        ink=apply_transform(sl, s, tx, ty), canvas=(64, 64)))
    return out


def run_line(args):
    sid, conds, levels = args
    try:
        from uhwr_baselines import geometric as GEO
        from uhwr_baselines import inputs as I
        from uhwr_exp.common import line_sample
        import cv2
        d = G["lines"][sid]
        L = line_sample(d, sid)
        img_ds = cv2.imread(os.path.join(G["bundle"], d["img"]), cv2.IMREAD_GRAYSCALE)
        preds, rt = {}, {}
        if "line" in levels:
            for c in conds:
                if c == "dataset":
                    img, aff, uh = img_ds, I.dataset_line(L, img_ds), img_ds.shape[0]
                else:
                    img, f = render(L.ink, L.canvas, c)
                    aff, uh = (np.array([1 / f, 1 / f]), np.zeros(2)), 56 * f
                for var, dots in (("geo_rtl", "last"), ("geo_rtl_local", "after_body")):
                    t0 = time.time()
                    preds[("line", var, c, sid)] = I.to_eval(GEO.recover(img, dots=dots, unit_height=uh), aff)
                    rt[f"line|{var}|{c}"] = time.time() - t0
                if "momna" in G:
                    t0 = time.time()
                    o = G["momna"].recover(img)
                    preds[("line", "momna_s123", c, sid)] = I.to_eval(o["emitted"], aff)
                    preds[("line", "momna_s123_subpaths", c, sid)] = I.to_eval(o["subpaths"], aff)
                    rt[f"line|momna|{c}"] = time.time() - t0
        if "ligature" in levels:
            for u in units_for_line(sid, L):
                for c in conds:
                    if c == "dataset":
                        img, aff = I.dataset_unit(L, u, img_ds, 4.0)
                        uh = 56 * 4
                    else:
                        img, f = render(u["ink"], u["canvas"], c)
                        aff, uh = (np.array([1 / f, 1 / f]), np.zeros(2)), 56 * f
                    for var, dots in (("geo_rtl", "last"), ("geo_rtl_local", "after_body")):
                        t0 = time.time()
                        preds[("ligature", var, c, u["unit_id"])] = I.to_eval(
                            GEO.recover(img, dots=dots, unit_height=uh), aff)
                        rt[f"ligature|{var}|{c}"] = rt.get(f"ligature|{var}|{c}", 0) + time.time() - t0
        return sid, preds, rt, ""
    except Exception:
        return sid, {}, {}, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conds", nargs="*", default=CONDS)
    ap.add_argument("--levels", nargs="*", default=["line", "ligature"])
    ap.add_argument("--no-momna", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import pickle as pk
    sids = sorted(pk.load(open(os.path.join(a.bundle, "lines_test.pkl"), "rb")))
    if a.limit:
        sids = sids[: a.limit]
    todo = [s for s in sids if not os.path.exists(os.path.join(a.out, f"pred_{s}.pkl"))]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                      initargs=(a.bundle, not a.no_momna)) as pool:
        for k, (sid, preds, rt, err) in enumerate(pool.imap_unordered(
                run_line, [(s, a.conds, a.levels) for s in todo])):
            if err:
                with open(os.path.join(a.out, "errors.txt"), "a") as f:
                    f.write(f"{sid}\t{err!r}\n")
                continue
            pickle.dump(preds, open(os.path.join(a.out, f"pred_{sid}.pkl"), "wb"))
            json.dump(rt, open(os.path.join(a.out, f"runtime_{sid}.json"), "w"))
            if k % 10 == 0:
                print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
