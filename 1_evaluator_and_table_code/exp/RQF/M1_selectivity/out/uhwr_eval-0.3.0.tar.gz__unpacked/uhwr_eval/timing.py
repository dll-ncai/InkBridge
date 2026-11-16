"""Timing error for uhwr_eval (added in 0.2.0; S4 of the preliminary-experiment plan).

Both metrics compare WHEN the pen was where, independently of shape, stroke order and stroke
direction.  They use the same one-to-one stroke matching as the order metrics
(``metrics.match_strokes``: Hungarian on 5 px stroke-mask IoU, accept IoU >= 0.5) and ignore
pen-up (hover) durations, so they need only per-point timestamps of pen-down samples.

timing_profile_tv  (proposed, lower is better, range [0, 1])
    For every matched stroke pair the predicted stroke is first oriented to the GT stroke
    (reversed if its start is nearer the GT end than the GT start, the same test as
    ``direction_acc``).  Each stroke's dwell-time distribution over normalised arc length is
    then the fraction of the stroke's duration spent in each of K = 16 equal arc-length bins
    (time of a segment = |difference| of its endpoint timestamps, assigned to the bin of the
    segment midpoint).  The score is the total-variation distance 0.5 * sum |h_gt - h_pred|,
    averaged over matched pairs weighted by GT stroke duration.  Orienting first makes the
    metric blind to a reversal that keeps the speed at every location (time reversed with the
    path); it does respond to a reversal that keeps the time stamps in sequence order.
timing_duration_tv (proposed, lower is better, range [0, 1])
    Total-variation distance between the GT and predicted shares of total pen-down time,
    over matched stroke pairs (each side renormalised over its matched strokes).  Needs >= 2
    matched pairs.

Both are NaN when either ink has no timestamps, when no strokes match, or when the matched
GT strokes have zero total duration.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

K_BINS = 16


def _dwell_hist(xy: np.ndarray, t: np.ndarray, K: int = K_BINS) -> Optional[np.ndarray]:
    if len(xy) < 2:
        return None
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    L = seg.sum()
    dt = np.abs(np.diff(t))  # time spent on each segment, whichever way it is traversed
    D = dt.sum()
    if L <= 1e-9 or D <= 0:
        return None
    mid = (np.cumsum(seg) - 0.5 * seg) / L
    b = np.clip((mid * K).astype(int), 0, K - 1)
    h = np.bincount(b, weights=dt, minlength=K)
    s = h.sum()
    return h / s if s > 0 else None


def timing_metrics(gt: Sequence[np.ndarray], pr: Sequence[np.ndarray],
                   gt_t: Optional[Sequence[np.ndarray]], pr_t: Optional[Sequence[np.ndarray]],
                   matches) -> Dict[str, float]:
    """``gt``/``pr``: inks (lists of (n, 2)); ``gt_t``/``pr_t``: per-stroke timestamp arrays
    (same lengths) or None; ``matches``: list of (gt_idx, pred_idx, iou)."""
    out = {"timing_profile_tv": float("nan"), "timing_duration_tv": float("nan")}
    if gt_t is None or pr_t is None or not matches:
        return out
    tv, w, dg, dp = [], [], [], []
    for i, j, _ in matches:
        g, p = np.asarray(gt[i], float), np.asarray(pr[j], float)
        tg, tp = np.asarray(gt_t[i], float), np.asarray(pr_t[j], float)
        if len(tg) != len(g) or len(tp) != len(p):
            raise ValueError("timestamps must align with stroke points")
        dg.append(abs(tg[-1] - tg[0]))
        dp.append(abs(tp[-1] - tp[0]))
        if np.linalg.norm(p[0] - g[0]) > np.linalg.norm(p[0] - g[-1]):
            p, tp = p[::-1], tp[::-1]  # orient to GT; dwell uses |dt| per segment
        hg, hp = _dwell_hist(g, tg), _dwell_hist(p, tp)
        if hg is None or hp is None:
            continue
        tv.append(0.5 * float(np.abs(hg - hp).sum()))
        w.append(dg[-1])
    if tv and sum(w) > 0:
        out["timing_profile_tv"] = float(np.average(tv, weights=w))
    dg, dp = np.array(dg), np.array(dp)
    if len(dg) >= 2 and dg.sum() > 0 and dp.sum() > 0:
        out["timing_duration_tv"] = 0.5 * float(np.abs(dg / dg.sum() - dp / dp.sum()).sum())
    return out
