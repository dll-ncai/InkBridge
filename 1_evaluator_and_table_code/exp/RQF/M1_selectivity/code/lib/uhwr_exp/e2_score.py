"""E2: score the recovered trajectories under input-renderer, resampling and AIoU-setting sweeps.

python -m uhwr_exp.e2_score --bundle bundle --pred out/pred --out out/e2 [--workers N] [--limit K]

Sweeps (one factor varied at a time, others at the reference setting):
  renderer   all 17 input conditions of predict_run (16 r_w*_h* + dataset), metrics on native inks
  resampling reference input condition only; both GT and prediction transformed identically:
             native | step1 (1 px arc length, = 'standardised') | N50 | N100 | N320 (whole ink to
             N points, allocated to strokes by arc length, >= 2 per stroke with length > 0) |
             down2 (every 2nd point, last point kept; PEN-Net style 2x downsampling)
  aiou       reference input condition, native inks: AIoU with GT thickness {1,2,3,4} x max
             dilations {0,2,4,8} (Nagamatsu adaptive_iou; prediction always drawn at 1 px)
Reference input condition: r_w2_h256 for geo_rtl / geo_rtl_local (the earlier f = 4 runs) and
r_w2_h64 for Momna (the earlier f = 1 runs).
Ligature level also includes momna_s123_part: Momna's line output (emitted strokes) assigned to
v3 units by majority vote of each predicted stroke's points over the unit of the nearest GT point
(an oracle partition, analogous to the oracle unit crops given to the geometric baseline).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("UHWR_TORCH_THREADS", "1")

import numpy as np  # noqa: E402

REF = {"geo_rtl": "r_w2_h256", "geo_rtl_local": "r_w2_h256", "momna_s123": "r_w2_h64",
       "momna_s123_subpaths": "r_w2_h64", "momna_s123_part": "r_w2_h64"}
RESAMPLE = ["native", "step1", "N50", "N100", "N320", "down2"]
AIOU_TH = (1, 2, 3, 4)
AIOU_DIL = (0, 2, 4, 8)
G = {}


def resample(ink, mode):
    from uhwr_eval import resample_ink_step
    from uhwr_eval.strokes import clean_ink, resample_stroke_step, stroke_length
    ink = clean_ink(ink)
    if mode == "native":
        return ink
    if mode == "step1":
        return resample_ink_step(ink, 1.0)
    if mode == "down2":
        return [np.vstack([s[::2], s[-1:]]) if (len(s) - 1) % 2 else s[::2].copy() for s in ink]
    N = int(mode[1:])
    Ls = np.array([stroke_length(s) for s in ink])
    tot = Ls.sum()
    out = []
    for s, L in zip(ink, Ls):
        if L <= 1e-9:
            out.append(s[:1].copy())
            continue
        n = max(2, int(round(N * L / tot)))
        out.append(resample_stroke_step(s, L / (n - 1) * (1 + 1e-9)))
    return out


def partition(line_pred, line, units):
    """Assign predicted line strokes to v3 units (oracle: nearest GT point's unit, majority)."""
    from scipy.spatial import cKDTree
    from uhwr_eval import apply_transform
    P, lab = [], []
    for k, u in enumerate(units):
        for i in u["stroke_idx"]:
            P.append(line.ink[i]); lab.append(np.full(len(line.ink[i]), k))
    tree, lab = cKDTree(np.vstack(P)), np.concatenate(lab)
    per = {k: [] for k in range(len(units))}
    for s in line_pred:
        if len(s) == 0:
            continue
        _, j = tree.query(s)
        k = int(np.bincount(lab[j]).argmax())
        per[k].append(s)
    return {units[k]["unit_id"]: apply_transform(v, *units[k]["T"]) for k, v in per.items()}


def _init(bundle, pred):
    import pandas as pd
    from uhwr_exp.common import load_lines
    G["lines"] = load_lines(bundle)
    G["idx"] = pd.read_csv(os.path.join(bundle, "ligatures_v3_test.csv"))
    G["pred"] = pred


def score_safe(gt, pr, canvas, **kw):
    from uhwr_eval import score
    try:
        return score(gt, pr, canvas, **kw), ""
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"


def aiou_grid(gt, pr, canvas):
    from uhwr_eval import _nagamatsu_metrics as NM
    from uhwr_eval import to_nagamatsu
    from uhwr_eval.strokes import clean_ink
    gt, pr = clean_ink(gt), clean_ink(pr)
    out = {}
    for th in AIOU_TH:
        for dil in AIOU_DIL:
            if not pr:
                out[(th, dil)] = 0.0
                continue
            out[(th, dil)] = float(NM.adaptive_iou(to_nagamatsu(gt, canvas), to_nagamatsu(pr, canvas),
                                                   image_shape=canvas, ground_truth_thickness=th,
                                                   max_dilation_iterations=dil))
    return out


def run_line(sid):
    from uhwr_exp.common import line_sample
    from uhwr_exp.predict_run import CONDS, G as PG, units_for_line
    PG["idx"] = G["idx"]
    t0 = time.time()
    d = G["lines"][sid]
    L = line_sample(d, sid)
    units = units_for_line(sid, L)
    preds = pickle.load(open(os.path.join(G["pred"], f"pred_{sid}.pkl"), "rb"))
    # add Momna ligature partition for every condition
    for c in CONDS:
        key = ("line", "momna_s123", c, sid)
        if key in preds:
            for uid, ink in partition(preds[key], L, units).items():
                preds[("ligature", "momna_s123_part", c, uid)] = ink
    gts = {("line", sid): (L.ink, L.canvas)}
    for u in units:
        gts[("ligature", u["unit_id"])] = (u["ink"], u["canvas"])
    rows = []
    base = dict(sample_id=sid, writer_id=d["writer_id"])
    for (lev, meth, cond, uid), pr in preds.items():
        gt, canvas = gts[(lev, uid)]
        b = dict(base, unit=lev, unit_id=uid, method=meth, input=cond)
        r, e = score_safe(gt, pr, canvas, protocol="native")
        rows.append(dict(b, sweep="renderer", setting=cond, error=e, **r))
        if cond != REF[meth]:
            continue
        rows.append(dict(b, sweep="resampling", setting="native", error=e, **r))
        for mode in RESAMPLE[1:]:
            r, e = score_safe(resample(gt, mode), resample(pr, mode), canvas, protocol="native")
            rows.append(dict(b, sweep="resampling", setting=mode, error=e, **r))
        try:
            for (th, dil), v in aiou_grid(gt, pr, canvas).items():
                rows.append(dict(b, sweep="aiou", setting=f"gt{th}_dil{dil}", error="", aiou=v))
        except Exception as ex:
            rows.append(dict(b, sweep="aiou", setting="all", error=f"{type(ex).__name__}: {ex}"))
    return sid, rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "rows"), exist_ok=True)
    sids = sorted(f[5:-4] for f in os.listdir(a.pred) if f.startswith("pred_"))
    if a.limit:
        sids = sids[: a.limit]
    todo = [s for s in sids if not os.path.exists(os.path.join(a.out, "rows", f"{s}.pkl"))]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.bundle, a.pred),
                                      maxtasksperchild=20) as pool:
        for k, (sid, rows, dt) in enumerate(pool.imap_unordered(run_line, todo)):
            pickle.dump(rows, open(os.path.join(a.out, "rows", f"{sid}.pkl"), "wb"))
            if k % 10 == 0:
                print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f}s (line {dt:.0f}s)", flush=True)
    import pandas as pd
    rows = []
    for s in sids:
        rows += pickle.load(open(os.path.join(a.out, "rows", f"{s}.pkl"), "rb"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "e2_scores_per_unit.csv.gz"), index=False)
    print("done", len(df), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
