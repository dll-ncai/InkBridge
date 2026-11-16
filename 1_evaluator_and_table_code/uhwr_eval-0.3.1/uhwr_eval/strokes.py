"""Shared stroke format and conversions for uhwr_eval.

Shared format
-------------
An *ink* is a Python list of strokes; each stroke is a float64 array of shape
(n_i, 2) holding (x, y) in **evaluation pixel space** (x to the right, y down,
origin at the top-left of the evaluation canvas).  The canvas is given
separately as ``(H, W)``.  Strokes are in drawing order and each stroke's
points are in pen order.  No NaN rows, no pen-up sentinel rows: a pen-up is
implicit between consecutive strokes.

Evaluation canvases used in the uhwr baselines phase
  * ligature / connected-component units: (64, 64), isotropic fit into a
    56 x 56 box centred on the canvas (4 px margin).
  * line units: height 64, isotropic scale so that the line's ink bounding box
    height is 56 px (4 px top/bottom margin); width = ceil(56 * aspect) + 8.

Nagamatsu encoding
------------------
``to_nagamatsu`` converts to the (N, 4) array [x, y, pen, eos] used by
``src/utils/metrics.py`` of hirokinaga/Handwriting-Trajectory-Recovery-with-
Diffusion-Models: x, y in [-1, 1] (x_n = 2 x / W - 1, so that the repo's
``_to_image_coordinates`` maps back exactly to our pixel units), pen = -1 on
the LAST point of each stroke (+1 elsewhere), eos = -1 on the final point of
the sequence (+1 elsewhere).  This is the convention the code uses (the paper
text states the opposite).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

Ink = List[np.ndarray]


def clean_ink(ink: Sequence[np.ndarray]) -> Ink:
    """Drop non-finite points and empty strokes; return float64 copies."""
    out = []
    for s in ink:
        s = np.asarray(s, dtype=np.float64).reshape(-1, 2)
        s = s[np.isfinite(s).all(axis=1)]
        if len(s):
            out.append(s.copy())
    return out


def from_nan_separated(xy: np.ndarray) -> Ink:
    """Split an (N, 2) array with NaN pen-up rows (Bezier online export) into strokes."""
    xy = np.asarray(xy, dtype=np.float64)
    bad = ~np.isfinite(xy).all(axis=1)
    strokes, cur = [], []
    for p, b in zip(xy, bad):
        if b:
            if cur:
                strokes.append(np.array(cur))
            cur = []
        else:
            cur.append(p)
    if cur:
        strokes.append(np.array(cur))
    return strokes


def from_pen_flags(xyp: np.ndarray, end_flag_value: float = 1.0) -> Ink:
    """Split (N, 3) [x, y, pen] where pen == end_flag_value marks the last point of a stroke
    (Momna's token format uses pen_state = 1 on the last point)."""
    xyp = np.asarray(xyp, dtype=np.float64)
    strokes, start = [], 0
    for i in range(len(xyp)):
        if xyp[i, 2] == end_flag_value or i == len(xyp) - 1:
            strokes.append(xyp[start:i + 1, :2].copy())
            start = i + 1
    return clean_ink(strokes)


def to_nagamatsu(ink: Sequence[np.ndarray], canvas: Tuple[int, int]) -> np.ndarray:
    """Encode an ink as Nagamatsu's (N, 4) [x, y, pen, eos] float32 array (see module doc)."""
    ink = clean_ink(ink)
    if not ink:
        raise ValueError("empty ink")
    H, W = canvas
    rows = []
    for s in ink:
        p = np.ones((len(s), 1))
        p[-1] = -1
        xn = 2.0 * s[:, :1] / W - 1.0
        yn = 2.0 * s[:, 1:2] / H - 1.0
        rows.append(np.hstack([xn, yn, p]))
    X = np.vstack(rows)
    e = np.ones((len(X), 1))
    e[-1] = -1
    return np.hstack([X, e]).astype(np.float32)


def from_normalized(ink_n: Sequence[np.ndarray], canvas: Tuple[int, int]) -> Ink:
    """Map strokes given in [-1, 1] (Nagamatsu space) to evaluation pixel space."""
    H, W = canvas
    return [np.c_[(np.asarray(s)[:, 0] + 1.0) / 2.0 * W, (np.asarray(s)[:, 1] + 1.0) / 2.0 * H]
            for s in ink_n]


def ink_bbox(ink: Sequence[np.ndarray]) -> Tuple[float, float, float, float]:
    P = np.vstack(ink)
    return float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())


def fit_transform_unit(ink: Sequence[np.ndarray], size: int = 64, margin: int = 4):
    """Isotropic transform that fits the ink bbox into a (size-2*margin)^2 box centred on a
    size x size canvas.  Returns (scale, tx, ty) with p' = scale * p + (tx, ty)."""
    x0, y0, x1, y1 = ink_bbox(ink)
    box = size - 2 * margin
    span = max(x1 - x0, y1 - y0, 1e-6)
    s = box / span
    tx = size / 2.0 - s * (x0 + x1) / 2.0
    ty = size / 2.0 - s * (y0 + y1) / 2.0
    return s, tx, ty


def fit_transform_line(ink: Sequence[np.ndarray], height: int = 64, margin: int = 4):
    """Isotropic transform mapping the ink bbox height to (height - 2*margin) px.
    Returns (scale, tx, ty, canvas) with canvas = (height, width)."""
    x0, y0, x1, y1 = ink_bbox(ink)
    s = (height - 2 * margin) / max(y1 - y0, 1e-6)
    W = int(math.ceil(s * (x1 - x0))) + 2 * margin
    tx = margin - s * x0
    ty = margin - s * y0
    return s, tx, ty, (height, W)


def apply_transform(ink: Sequence[np.ndarray], s: float, tx: float, ty: float) -> Ink:
    return [np.c_[s * a[:, 0] + tx, s * a[:, 1] + ty] for a in ink]


def stroke_length(s: np.ndarray) -> float:
    if len(s) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())


def resample_stroke_step(s: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Uniform arc-length resampling with spacing ~step px; endpoints kept.
    Zero-length strokes collapse to one point."""
    s = np.asarray(s, dtype=np.float64)
    if len(s) < 2:
        return s.copy()
    d = np.linalg.norm(np.diff(s, axis=0), axis=1)
    keep = np.r_[True, d > 1e-9]
    s = s[keep]
    if len(s) < 2:
        return s[:1].copy()
    d = np.linalg.norm(np.diff(s, axis=0), axis=1)
    c = np.r_[0.0, np.cumsum(d)]
    L = c[-1]
    n = max(2, int(math.ceil(L / step)) + 1)
    t = np.linspace(0.0, L, n)
    return np.c_[np.interp(t, c, s[:, 0]), np.interp(t, c, s[:, 1])]


def resample_ink_step(ink: Sequence[np.ndarray], step: float = 1.0) -> Ink:
    """'Standardised' protocol: per-stroke fixed arc-length resampling (default 1 px)."""
    return [resample_stroke_step(s, step) for s in clean_ink(ink)]


def resample_ink_fixed_count(ink: Sequence[np.ndarray], K: int) -> np.ndarray:
    """Resample the whole ink (pen-down path, strokes concatenated in drawing order) to
    exactly K points spaced uniformly in cumulative pen-down arc length.  Pen-up jumps
    carry no arc length.  Returns (K, 2)."""
    ink = clean_ink(ink)
    pts, cum, offset = [], [], 0.0
    for s in ink:
        d = np.r_[0.0, np.linalg.norm(np.diff(s, axis=0), axis=1)] if len(s) > 1 else np.zeros(1)
        c = offset + np.cumsum(d)
        pts.append(s)
        cum.append(c)
        offset = c[-1] + 1e-9  # tiny gap so the jump is a step, not a ramp
    P = np.vstack(pts)
    C = np.concatenate(cum)
    if C[-1] <= 0:
        return np.repeat(P[:1], K, axis=0)
    t = np.linspace(0.0, C[-1], K)
    idx = np.searchsorted(C, t, side="right") - 1
    idx = np.clip(idx, 0, len(C) - 2)
    w = (t - C[idx]) / np.maximum(C[idx + 1] - C[idx], 1e-12)
    w = np.clip(w, 0.0, 1.0)
    out = P[idx] * (1 - w)[:, None] + P[idx + 1] * w[:, None]
    # points that fall exactly on a jump are snapped to the next stroke start
    return out
