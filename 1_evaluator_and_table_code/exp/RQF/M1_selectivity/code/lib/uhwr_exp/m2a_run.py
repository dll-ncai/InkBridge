"""M2a: test-time render-and-compare refinement with a Gaussian-splat soft rasteriser.

python -m uhwr_exp.m2a_run --bundle bundle --pred out/pred --out out/m2a [--workers N] [--n 300]

Sample: 300 v3 test ligatures (numpy default_rng(20261016), units with >= 1 stroke of >= 2
points), plus 50 further units (disjoint) used only to calibrate the rasteriser (sigma, alpha)
by minimising the render loss of the GT against the input image.
Initial prediction: geo_rtl on the reference input r_w2_h256 (from predict_run).
Target: that same input image (256 x 256, 8 px strokes), area-downsampled to 128 x 128 and
converted to ink darkness d = 1 - I / 255.
Rasteriser: prediction strokes resampled to 1 px steps on the 64 px eval canvas (points =
optimisation variables, stroke membership and point order fixed, so no new pen-ups); each point
is a Gaussian splat on the 128 px canvas; soft union 1 - prod_k (1 - alpha exp(-|p-q_k|^2 / 2
sigma^2)).  Loss = mean (R - d)^2 + lam_anchor * mean |q - q0|^2 (eval px^2; the order anchor:
keeps points near the initial trajectory, whose order/direction come from the baseline)
+ 0.01 * mean |second difference|^2 (smoothness).  Adam, lr 0.05 eval px, 200 steps.
lam_anchor in {0, 0.001, 0.01, 0.1}.
Also reported: render loss of GT copies with reversed stroke order, all strokes reversed and
both, against the same target (the loss is identical by construction: it cannot see order).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

LAMS = (0.0, 0.001, 0.01, 0.1)
STEPS = 200
LR = 0.05
G = {}


def target_of(u):
    import cv2
    from uhwr_exp.predict_run import render
    img, f = render(u["ink"], u["canvas"], "r_w2_h256")
    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    return 1.0 - small.astype(np.float32) / 255.0


def raster(pts, sigma, alpha, H=128):
    import torch
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32) + 0.5,
                            torch.arange(H, dtype=torch.float32) + 0.5, indexing="ij")
    P = pts * (H / 64.0)
    d2 = (xs.reshape(-1, 1) - P[:, 0]) ** 2 + (ys.reshape(-1, 1) - P[:, 1]) ** 2
    a = alpha * torch.exp(-d2 / (2 * sigma ** 2))
    return (1 - torch.exp(torch.log1p(-a.clamp(max=0.999)).sum(1))).reshape(H, H)


def dense(ink, step=1.0):
    from uhwr_eval import resample_ink_step
    return resample_ink_step(ink, step)


def render_loss(ink, tgt, sigma, alpha):
    import torch
    pts = torch.tensor(np.vstack(dense(ink)), dtype=torch.float32)
    with torch.no_grad():
        return float(((raster(pts, sigma, alpha) - torch.tensor(tgt)) ** 2).mean())


def refine(ink, tgt, sigma, alpha, lam):
    import torch
    strokes = dense(ink)
    lens = [len(s) for s in strokes]
    q0 = torch.tensor(np.vstack(strokes), dtype=torch.float32)
    q = q0.clone().requires_grad_(True)
    T = torch.tensor(tgt)
    opt = torch.optim.Adam([q], lr=LR)
    cut = np.cumsum([0] + lens)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((raster(q, sigma, alpha) - T) ** 2).mean()
        if lam > 0:
            loss = loss + lam * ((q - q0) ** 2).sum(1).mean()
        sm = []
        for a, b in zip(cut[:-1], cut[1:]):
            if b - a >= 3:
                sm.append(((q[a + 2:b] - 2 * q[a + 1:b - 1] + q[a:b - 2]) ** 2).sum(1))
        if sm:
            loss = loss + 0.01 * torch.cat(sm).mean()
        loss.backward()
        opt.step()
    qn = q.detach().numpy().astype(float)
    with torch.no_grad():
        fl = float(((raster(q, sigma, alpha) - T) ** 2).mean())
    return [qn[a:b] for a, b in zip(cut[:-1], cut[1:])], fl, float(((q.detach() - q0) ** 2).sum(1).sqrt().mean())


def _init(bundle, pred):
    import torch
    torch.set_num_threads(1)
    from uhwr_exp.common import load_lines, line_sample
    import pandas as pd
    G["lines"] = load_lines(bundle)
    G["idx"] = pd.read_csv(os.path.join(bundle, "ligatures_v3_test.csv"))
    G["pred"] = pred
    G["ls"] = line_sample


def get_unit(uid):
    from uhwr_exp.predict_run import G as PG, units_for_line
    PG["idx"] = G["idx"]
    sid = uid.rsplit("_L", 1)[0]
    L = G["ls"](G["lines"][sid], sid)
    u = [x for x in units_for_line(sid, L) if x["unit_id"] == uid][0]
    P = pickle.load(open(os.path.join(G["pred"], f"pred_{sid}.pkl"), "rb"))
    return u, P[("ligature", "geo_rtl", "r_w2_h256", uid)], sid


def calib_one(args):
    uid, grid = args
    u, _, _ = get_unit(uid)
    tgt = target_of(u)
    return {f"{s}_{a}": render_loss(u["ink"], tgt, s, a) for s, a in grid}


def run_one(args):
    uid, sigma, alpha = args
    from uhwr_eval import score
    t0 = time.time()
    u, pr, sid = get_unit(uid)
    tgt = target_of(u)
    gt = u["ink"]
    rows = []
    sc = lambda ink: score(gt, ink, (64, 64), protocol="native", with_lpips=False)
    base = dict(unit_id=uid, sample_id=sid, n_gt_strokes=len(gt))
    r0 = sc(pr)
    rows.append(dict(base, variant="initial", lam=np.nan, render_loss=render_loss(pr, tgt, sigma, alpha),
                     mean_disp=0.0, **r0))
    for lam in LAMS:
        ink, fl, disp = refine(pr, tgt, sigma, alpha, lam)
        rows.append(dict(base, variant="refined", lam=lam, render_loss=fl, mean_disp=disp, **sc(ink)))
    # order blindness check on GT copies
    gl = []
    for name, ink in (("gt", gt), ("gt_order_reversed", gt[::-1]),
                      ("gt_all_strokes_reversed", [s[::-1] for s in gt]),
                      ("gt_order_and_direction_reversed", [s[::-1] for s in gt[::-1]])):
        gl.append(dict(base, variant=name, lam=np.nan, render_loss=render_loss(ink, tgt, sigma, alpha),
                       mean_disp=0.0, **sc(ink)))
    return rows + gl, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_calib", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import pandas as pd
    idx = pd.read_csv(os.path.join(a.bundle, "ligatures_v3_test.csv"))
    have = {f[5:-4] for f in os.listdir(a.pred) if f.startswith("pred_")}
    idx = idx[idx.sample_id.isin(have)].reset_index(drop=True)
    rng = np.random.default_rng(20261016)
    perm = rng.permutation(len(idx))
    sel = [idx.unit_id[i] for i in perm[: a.n]]
    cal = [idx.unit_id[i] for i in perm[a.n: a.n + a.n_calib]]
    grid = [(s, al) for s in (1.0, 1.5, 2.0, 2.5) for al in (0.3, 0.6, 0.9)]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.bundle, a.pred)) as pool:
        C = pool.map(calib_one, [(u, grid) for u in cal])
        cm = {k: float(np.mean([c[k] for c in C])) for k in C[0]}
        best = min(cm, key=cm.get)
        sigma, alpha = map(float, best.split("_"))
        json.dump(dict(grid_mean_gt_render_loss=cm, chosen=dict(sigma=sigma, alpha=alpha),
                       calib_units=cal, sample_units=sel), open(os.path.join(a.out, "m2a_calibration.json"), "w"), indent=1)
        print("calibrated", sigma, alpha, f"{time.time() - t0:.0f}s", flush=True)
        rows = []
        for k, (r, dt) in enumerate(pool.imap_unordered(run_one, [(u, sigma, alpha) for u in sel])):
            rows += r
            if k % 20 == 0:
                print(f"{k + 1}/{len(sel)} {time.time() - t0:.0f}s (unit {dt:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "m2a_scores_per_unit.csv.gz"), index=False)
    print("done", len(rows), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
