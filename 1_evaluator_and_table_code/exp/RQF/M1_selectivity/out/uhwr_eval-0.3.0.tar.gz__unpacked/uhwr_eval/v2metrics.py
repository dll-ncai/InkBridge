"""uhwr_eval 0.3.0 additions (E1-v2 redesign of the pen-up and timing components; experiments_to_run 2026-10-18 A/E1-v2).

Both metrics are NEW names; every 0.2.0 metric (incl. penup_f1, timing_profile_tv) is unchanged.

penup_f1_v2  (proposed, higher is better, [0, 1])
    Orientation- and order-invariant pen-up F1.  Each stroke is represented by its UNORDERED end-point pair
    {first point, last point} (= its pen-down and pen-up locations).  Cost between GT stroke i and predicted stroke j =
    min over the two orientations of max(|g_a - p_a|, |g_b - p_b|).  One-to-one Hungarian assignment; a pair is a true
    positive when its cost <= tau (3 px).  F1 = 2 TP / (n_gt + n_pred).
    Motivation: 0.2.0 penup_f1 matches only the LAST point of each stroke, so reversing a stroke moves its 'pen-up' to
    the other end and is scored as a pen-up error (E1 v1: penup_f1 reacted more to direction than to merge/split).
    v2 is invariant to stroke direction and order by construction, and still drops for merges (two GT pairs -> one
    unmatched pair) and splits (one GT pair -> two pairs with one wrong end each).
timing_profile_tv_v2  (proposed, lower is better, [0, 1])
    As timing_profile_tv (16 arc-length bins, TV distance between dwell-time distributions per matched stroke pair,
    weighted by GT stroke duration, pen-up time ignored, same stroke matching), but the PREDICTED stroke is binned by
    the PROJECTED arc position of each predicted segment midpoint on the matched GT stroke polyline (nearest point on
    the GT polyline, arc length normalised by the GT stroke length), instead of by the prediction's own normalised arc
    length.  The GT stroke is binned by its own arc length as before.  No orientation step is needed (projection is
    orientation-free; per-segment time is |dt|).  Motivation: with own-arc binning, shape jitter lengthens the predicted
    path non-uniformly and shifts dwell mass between bins (timing leaked shape error in E1 v1).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

K_BINS = 16


def penup_endpoint_f1(gt, pr, tau: float = 3.0) -> float:
    if not len(gt) or not len(pr):
        return 0.0
    G0 = np.array([np.asarray(s, float)[0] for s in gt]); G1 = np.array([np.asarray(s, float)[-1] for s in gt])
    P0 = np.array([np.asarray(s, float)[0] for s in pr]); P1 = np.array([np.asarray(s, float)[-1] for s in pr])
    d = lambda A, B: np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    C = np.minimum(np.maximum(d(G0, P0), d(G1, P1)), np.maximum(d(G0, P1), d(G1, P0)))
    r, c = linear_sum_assignment(C)
    tp = int((C[r, c] <= tau).sum())
    return 2.0 * tp / (len(G0) + len(P0))


def _project_arc(P: np.ndarray, G: np.ndarray, chunk: int = 4096) -> Optional[np.ndarray]:
    """Normalised arc position in [0, 1] of the nearest point on polyline G for every point of P."""
    seg = G[1:] - G[:-1]
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    tot = cum[-1]
    if len(G) < 2 or tot <= 1e-9:
        return None
    d2 = (seg ** 2).sum(1)
    ok = d2 > 1e-18
    d2s = np.where(ok, d2, 1.0)
    out = np.empty(len(P))
    for a in range(0, len(P), chunk):
        Q = P[a:a + chunk]
        rel = Q[:, None, :] - G[None, :-1, :]
        t = np.clip((rel * seg[None]).sum(2) / d2s, 0.0, 1.0)
        t = np.where(ok[None], t, 0.0)
        proj = G[None, :-1, :] + t[..., None] * seg[None]
        dist = ((Q[:, None, :] - proj) ** 2).sum(2)
        k = dist.argmin(1)
        out[a:a + chunk] = (cum[k] + t[np.arange(len(Q)), k] * L[k]) / tot
    return out


def _gt_hist(g, tg, K=K_BINS):
    seg = np.linalg.norm(np.diff(g, axis=0), axis=1); L = seg.sum(); dt = np.abs(np.diff(tg))
    if L <= 1e-9 or dt.sum() <= 0:
        return None
    mid = (np.cumsum(seg) - 0.5 * seg) / L
    h = np.bincount(np.clip((mid * K).astype(int), 0, K - 1), weights=dt, minlength=K)
    return h / h.sum()


def _pred_hist_projected(p, tp, g, K=K_BINS):
    if len(p) < 2:
        return None
    dt = np.abs(np.diff(tp))
    if dt.sum() <= 0:
        return None
    s = _project_arc(0.5 * (p[1:] + p[:-1]), g)
    if s is None:
        return None
    h = np.bincount(np.clip((s * K).astype(int), 0, K - 1), weights=dt, minlength=K)
    return h / h.sum()


def timing_profile_tv_v2(gt, pr, gt_t, pr_t, matches) -> float:
    if gt_t is None or pr_t is None or not matches:
        return float("nan")
    tv, w = [], []
    for i, j, _ in matches:
        g, p = np.asarray(gt[i], float), np.asarray(pr[j], float)
        tg, tp = np.asarray(gt_t[i], float), np.asarray(pr_t[j], float)
        if len(tg) != len(g) or len(tp) != len(p):
            raise ValueError("timestamps must align with stroke points")
        hg, hp = _gt_hist(g, tg), _pred_hist_projected(p, tp, g)
        if hg is None or hp is None:
            continue
        tv.append(0.5 * float(np.abs(hg - hp).sum())); w.append(abs(tg[-1] - tg[0]))
    if tv and sum(w) > 0:
        return float(np.average(tv, weights=w))
    return float("nan")
