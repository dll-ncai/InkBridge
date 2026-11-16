"""E1: controlled perturbation sweep of GT units; every metric scored (uhwr_eval 0.2.0).

python -m uhwr_exp.e1_run --bundle bundle --out out/e1 [--workers N] [--limit K]

Factors (5 strengths each; strength 0 = identical copy, scored once per unit):
  shape_jitter   i.i.d. Gaussian noise on every point, sigma = 0.25/0.5/1/1.5/2 px (64 px canvas)
  shape_elastic  smooth random displacement field (white noise, Gaussian-filtered sigma 8 px,
                 scaled to max |d| = 0.5/1/2/3/4 px), bilinear at every point
  order          k = 1..5 random transpositions, each chosen to strictly increase the number of
                 inversions (so disorder grows monotonically; stops when fully reversed);
                 each stroke keeps its own duration/profile and the sequence is retimed
  direction      k = 1..5 strokes reversed (among strokes whose end points are >= 2 px apart);
                 points AND times are reversed, so the speed at every location is preserved
  penup_merge    k = 1..5 random adjacent pairs joined into one stroke (straight connector);
                 the hover gap time becomes connector time
  penup_split    k = 1..5 strokes (>= 4 points) cut at a uniform interior point (no ink lost)
  timing         global pen-down time warp g(u) = u + a (sin(2 pi f u + phi) - sin phi) / (2 pi f),
                 a = 0.15/0.3/0.45/0.6/0.75, f in {1,2,3}, phi ~ U(0, 2pi); path unchanged
k is capped at what the unit allows; the effective k is recorded (eff_k).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import zlib

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("UHWR_TORCH_THREADS", "1")

FACTORS = {
    "shape_jitter": [0.25, 0.5, 1.0, 1.5, 2.0],
    "shape_elastic": [0.5, 1.0, 2.0, 3.0, 4.0],
    "order": [1, 2, 3, 4, 5],
    "direction": [1, 2, 3, 4, 5],
    "penup_merge": [1, 2, 3, 4, 5],
    "penup_split": [1, 2, 3, 4, 5],
    "timing": [0.15, 0.3, 0.45, 0.6, 0.75],
}


def _pd_times(t):
    """Per-stroke times -> continuous pen-down clock (hover gaps removed), per stroke."""
    out, off = [], 0.0
    for s in t:
        s = np.asarray(s, float) - s[0]
        out.append(s + off)
        off = out[-1][-1] + 1e-3
    return out


def retime(ink, t_strokes):
    """Reassemble timestamps after the stroke sequence changed: each stroke keeps its own
    relative time profile; strokes follow each other with 1 ms gaps."""
    out, off = [], 0.0
    for s in t_strokes:
        s = np.asarray(s, float) - np.asarray(s, float).min()
        out.append(s + off)
        off = out[-1].max() + 1e-3
    return out


def perturb(ink, t, factor, k, rng):
    ink = [np.asarray(s, float).copy() for s in ink]
    t = None if t is None else [np.asarray(s, float).copy() for s in t]
    n = len(ink)
    eff = 0
    if factor == "shape_jitter":
        ink = [s + rng.normal(0, k, s.shape) for s in ink]
        eff = k
    elif factor == "shape_elastic":
        from scipy.ndimage import gaussian_filter, map_coordinates
        H = 64
        dx = gaussian_filter(rng.normal(size=(H, H)), 8, mode="reflect")
        dy = gaussian_filter(rng.normal(size=(H, H)), 8, mode="reflect")
        m = np.sqrt(dx ** 2 + dy ** 2).max()
        dx, dy = dx * k / m, dy * k / m
        out = []
        for s in ink:
            c = np.clip(s, 0, H - 1)
            ddx = map_coordinates(dx, [c[:, 1], c[:, 0]], order=1, mode="nearest")
            ddy = map_coordinates(dy, [c[:, 1], c[:, 0]], order=1, mode="nearest")
            out.append(s + np.c_[ddx, ddy])
        ink = out
        eff = k
    elif factor == "order":
        perm = list(range(n))
        for _ in range(k):
            cand = [(i, j) for i in range(n) for j in range(i + 1, n) if perm[i] < perm[j]]
            if not cand:
                break
            i, j = cand[rng.integers(len(cand))]
            perm[i], perm[j] = perm[j], perm[i]
            eff += 1
        ink = [ink[p] for p in perm]
        if t is not None:
            t = retime(ink, [t[p] for p in perm])
    elif factor == "direction":
        el = [i for i, s in enumerate(ink) if len(s) >= 2 and np.linalg.norm(s[-1] - s[0]) >= 2.0]
        pick = rng.choice(el, min(k, len(el)), replace=False) if el else []
        for i in pick:
            ink[i] = ink[i][::-1].copy()
            if t is not None:
                ti = t[i]
                t[i] = (ti[0] + ti[-1] - ti)[::-1].copy()  # reversed traversal, same location speeds
        eff = len(pick)
    elif factor == "penup_merge":
        for _ in range(k):
            if len(ink) < 2:
                break
            i = int(rng.integers(len(ink) - 1))
            ink[i:i + 2] = [np.vstack([ink[i], ink[i + 1]])]
            if t is not None:
                t[i:i + 2] = [np.concatenate([t[i], t[i + 1]])]
            eff += 1
    elif factor == "penup_split":
        for _ in range(k):
            el = [i for i, s in enumerate(ink) if len(s) >= 4]
            if not el:
                break
            i = el[int(rng.integers(len(el)))]
            c = int(rng.integers(2, len(ink[i]) - 1))
            ink[i:i + 1] = [ink[i][:c], ink[i][c:]]
            if t is not None:
                t[i:i + 1] = [t[i][:c], t[i][c:]]
            eff += 1
    elif factor == "timing":
        if t is not None:
            pdt = _pd_times(t)
            T = pdt[-1][-1] if pdt[-1][-1] > 0 else 1.0
            f = int(rng.integers(1, 4))
            phi = rng.uniform(0, 2 * np.pi)
            g = lambda u: u + k * (np.sin(2 * np.pi * f * u + phi) - np.sin(phi)) / (2 * np.pi * f)
            t = [T * g(s / T) for s in pdt]
            eff = k
    else:
        raise ValueError(factor)
    return ink, t, eff


def run_unit(args):
    ds, u, seed = args
    from uhwr_eval import score
    rows = []
    canvas = (64, 64)
    gt, gt_t = u["ink"], u["t"]
    rng = np.random.default_rng([seed, zlib.crc32(str(u["uid"]).encode())])
    jobs = [("identity", 0, 0, gt, gt_t, 0)]
    for f, ks in FACTORS.items():
        for si, k in enumerate(ks, 1):
            ink, t, eff = perturb(gt, gt_t, f, k, rng)
            jobs.append((f, si, k, ink, t, eff))
    for f, si, k, ink, t, eff in jobs:
        for prot in ("native", "standardised"):
            try:
                r = score(gt, ink, canvas, protocol=prot, gt_times=gt_t, pred_times=t,
                          with_lpips=True)
                err = ""
            except Exception as e:  # never drop silently
                r, err = {}, f"{type(e).__name__}: {e}"
            rows.append(dict(dataset=ds, uid=u["uid"], cluster=u["cluster"], n_strokes=len(gt),
                             factor=f, strength=si, param=k, eff_k=eff, protocol=prot,
                             has_time=gt_t is not None, error=err, **r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261016)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    E = pickle.load(open(os.path.join(a.bundle, "e1_units.pkl"), "rb"))
    tasks = []
    for ds, units in E.items():
        for u in (units[: a.limit] if a.limit else units):
            tasks.append((ds, u, a.seed))
    import multiprocessing as mp
    import pandas as pd
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_unit, tasks, chunksize=4)):
            rows += r
            if i % 100 == 0:
                print(f"{i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "e1_scores_per_unit.csv.gz"), index=False)
    json.dump(dict(n_tasks=len(tasks), n_rows=len(df), n_err=int((df.error != "").sum()),
                   seconds=time.time() - t0), open(os.path.join(a.out, "e1_run_info.json"), "w"))
    print("done", len(df), "rows", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
