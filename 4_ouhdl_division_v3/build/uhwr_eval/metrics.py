"""uhwr_eval metrics.

DTW, LDTW and AIoU are computed by the UNCHANGED Nagamatsu ``metrics.py`` (vendored as
``_nagamatsu_metrics.py``; sha256 recorded in ``NAGAMATSU_SHA256``).  Everything else is
implemented here.  Every metric is registered in ``METRICS`` with a direction and a tag:

* ``established`` - metric with a prior definition in the literature that we follow.
* ``proposed``    - our operationalisation (definition documented in ``METRICS``).

Use :func:`score` to evaluate one (ground truth, prediction) pair under a protocol:

* ``native``       - both inks as given (GT as recorded by the tablet; prediction as emitted).
* ``standardised`` - both inks resampled per stroke at a fixed arc-length step
                     (default 1 px in evaluation space) before scoring.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import kendalltau

from . import _nagamatsu_metrics as NM
from .render import render_image, render_mask
from .strokes import (clean_ink, resample_ink_fixed_count, resample_ink_step, stroke_length,
                      to_nagamatsu)

NAGAMATSU_SHA256 = "7e2d330558f2ddf828cbd618ee80b36cada1818820315523ad826d2f3bbbb0c4"
NAGAMATSU_COMMIT = "7c5cf62392e36cdc9f86445b4acb16c1e9198c68"

# name -> (direction, tag, definition)
METRICS: Dict[str, Tuple[str, str, str]] = {
    "dtw": ("\u2193", "established", "Nagamatsu metrics.py dtw_and_ldtw (tslearn dtw_path: sqrt of summed squared distances along the optimal path), pixel units of the evaluation canvas"),
    "ldtw": ("\u2193", "established", "Nagamatsu metrics.py: DTW / alignment-path length (PEN-Net, Chen et al. ACCV 2022)"),
    "aiou": ("\u2191", "established", "Nagamatsu metrics.py adaptive_iou: GT drawn at 2 px, prediction at 1 px, best IoU over 0-4 3x3 dilations; strokes with <2 points are not drawn"),
    "lpips": ("\u2193", "established", "LPIPS (Zhang et al. 2018, AlexNet, lpips v0.1) between uniform-width renderings (2 px, anti-aliased, black on white) of GT and prediction on the evaluation canvas"),
    "chamfer": ("\u2193", "established", "symmetric mean Chamfer distance 0.5*(mean_a min_b d + mean_b min_a d) between point sets, px"),
    "hausdorff": ("\u2193", "established", "symmetric Hausdorff distance between point sets, px"),
    "frechet": ("\u2193", "established", "discrete Frechet distance (Eiter & Mannila 1994) between the concatenated point sequences in drawing order (pen-ups ignored), px"),
    "rmse": ("\u2193", "established", "point-wise RMSE after resampling each whole ink to K points uniformly in pen-down arc length (K=320 ligature, K=2048 line), px"),
    "penup_f1": ("\u2191", "proposed", "F1 of pen-up locations: last point of every stroke; one-to-one Hungarian matching on Euclidean distance with tolerance tau (3 px)"),
    "stroke_count_abs_err": ("\u2193", "established", "|n_pred - n_gt| strokes"),
    "stroke_count_err": ("n/a", "established", "n_pred - n_gt (signed)"),
    "stroke_match_recall": ("\u2191", "proposed", "fraction of GT strokes matched one-to-one (Hungarian on stroke-mask IoU, masks 5 px wide, accept IoU >= 0.5)"),
    "stroke_match_precision": ("\u2191", "proposed", "fraction of predicted strokes matched under the same rule"),
    "strict_order_acc": ("\u2191", "proposed", "1 if n_pred == n_gt, every GT stroke is matched (IoU >= 0.5, one-to-one; splits/merges therefore fail) and matched predicted indices are in GT order; else 0. Name follows Wang et al. 2026; matching rule is ours"),
    "kendall_tau": ("\u2191", "proposed", "Kendall tau-b between GT index and predicted index over matched strokes (needs >= 2 matches)"),
    "direction_acc": ("\u2191", "proposed", "over matched strokes whose GT start-end distance >= 2 px: fraction where predicted start is nearer GT start than GT end"),
    "loop_orient_acc": ("\u2191", "proposed", "over matched strokes whose GT is curved/looped (|signed area| >= 4 px^2 and 4*pi*|A|/L^2 >= 0.1): fraction with sign(signed area) of prediction == GT (CW vs CCW)"),
}

PROTOCOLS = ("native", "standardised")


def _sha_ok() -> bool:
    p = os.path.join(os.path.dirname(__file__), "_nagamatsu_metrics.py")
    return hashlib.sha256(open(p, "rb").read()).hexdigest() == NAGAMATSU_SHA256


# --------------------------------------------------------------------------- helpers
@njit(cache=True)
def _frechet_rows(A, B):
    n, m = A.shape[0], B.shape[0]
    prev = np.empty(m)
    cur = np.empty(m)
    for i in range(n):
        for j in range(m):
            d = math.sqrt((A[i, 0] - B[j, 0]) ** 2 + (A[i, 1] - B[j, 1]) ** 2)
            if i == 0 and j == 0:
                cur[j] = d
            elif i == 0:
                cur[j] = max(cur[j - 1], d)
            elif j == 0:
                cur[j] = max(prev[j], d)
            else:
                cur[j] = max(min(prev[j], prev[j - 1], cur[j - 1]), d)
        for j in range(m):
            prev[j] = cur[j]
    return prev[m - 1]


def discrete_frechet(A: np.ndarray, B: np.ndarray) -> float:
    return float(_frechet_rows(np.ascontiguousarray(A, dtype=np.float64),
                               np.ascontiguousarray(B, dtype=np.float64)))


def chamfer_hausdorff(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    da, _ = cKDTree(B).query(A)
    db, _ = cKDTree(A).query(B)
    return 0.5 * (float(da.mean()) + float(db.mean())), float(max(da.max(), db.max()))


def signed_area(s: np.ndarray) -> float:
    """Shoelace signed area of the stroke closed by its chord (image coords, y down:
    positive = clockwise on screen)."""
    if len(s) < 3:
        return 0.0
    x, y = s[:, 0], s[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


_LPIPS = None


def _lpips_model():
    global _LPIPS
    if _LPIPS is None:
        import lpips  # noqa: WPS433
        import torch
        torch.set_num_threads(max(1, int(os.environ.get("UHWR_TORCH_THREADS", "4"))))
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).eval()
    return _LPIPS


def lpips_distance(gt: Sequence[np.ndarray], pr: Sequence[np.ndarray], canvas) -> float:
    import torch
    H, W = canvas
    a = render_image(gt, canvas, 2).astype(np.float32) / 127.5 - 1.0
    b = render_image(pr, canvas, 2).astype(np.float32) / 127.5 - 1.0
    # LPIPS/AlexNet needs >= ~32 px per side; pad small canvases with background
    ph, pw = max(0, 32 - H), max(0, 32 - W)
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), constant_values=1.0)
        b = np.pad(b, ((0, ph), (0, pw)), constant_values=1.0)
    ta = torch.from_numpy(a)[None, None].repeat(1, 3, 1, 1)
    tb = torch.from_numpy(b)[None, None].repeat(1, 3, 1, 1)
    with torch.no_grad():
        return float(_lpips_model()(ta, tb).item())


def match_strokes(gt: Sequence[np.ndarray], pr: Sequence[np.ndarray], canvas,
                  thickness: int = 5, iou_min: float = 0.5):
    """One-to-one stroke matching by Hungarian assignment on mask IoU; pairs with
    IoU < iou_min are discarded.  Returns list of (gt_idx, pred_idx, iou)."""
    gm = [render_mask([s], canvas, thickness).astype(bool) for s in gt]
    pm = [render_mask([s], canvas, thickness).astype(bool) for s in pr]
    # bounding-box prefilter keeps line-level matching cheap
    def bb(s):
        return s[:, 0].min() - thickness, s[:, 1].min() - thickness, s[:, 0].max() + thickness, s[:, 1].max() + thickness
    gb = [bb(s) for s in gt]
    pb = [bb(s) for s in pr]
    M = np.zeros((len(gt), len(pr)))
    for i in range(len(gt)):
        for j in range(len(pr)):
            if gb[i][0] > pb[j][2] or pb[j][0] > gb[i][2] or gb[i][1] > pb[j][3] or pb[j][1] > gb[i][3]:
                continue
            inter = np.logical_and(gm[i], pm[j]).sum()
            if inter == 0:
                continue
            M[i, j] = inter / np.logical_or(gm[i], pm[j]).sum()
    if M.size == 0:
        return []
    r, c = linear_sum_assignment(-M)
    return [(int(i), int(j), float(M[i, j])) for i, j in zip(r, c) if M[i, j] >= iou_min]


def penup_f1(gt, pr, tau: float = 3.0) -> float:
    G = np.array([s[-1] for s in gt])
    P = np.array([s[-1] for s in pr])
    D = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=2)
    r, c = linear_sum_assignment(D)
    tp = int((D[r, c] <= tau).sum())
    return 2.0 * tp / (len(G) + len(P))


# --------------------------------------------------------------------------- main API
def order_metrics(gt, pr, canvas, iou_min=0.5, thickness=5) -> Dict[str, float]:
    m = match_strokes(gt, pr, canvas, thickness, iou_min)
    ng, npr = len(gt), len(pr)
    out = {
        "stroke_count_err": float(npr - ng),
        "stroke_count_abs_err": float(abs(npr - ng)),
        "stroke_match_recall": len(m) / ng if ng else float("nan"),
        "stroke_match_precision": len(m) / npr if npr else float("nan"),
    }
    m_sorted = sorted(m, key=lambda t: t[0])
    pred_seq = [j for _, j, _ in m_sorted]
    in_order = all(pred_seq[k] < pred_seq[k + 1] for k in range(len(pred_seq) - 1))
    out["strict_order_acc"] = float(npr == ng and len(m) == ng and in_order)
    if len(m) >= 2:
        tau = kendalltau([i for i, _, _ in m_sorted], pred_seq).statistic
        out["kendall_tau"] = float(tau) if np.isfinite(tau) else float("nan")
    else:
        out["kendall_tau"] = float("nan")
    dir_ok, dir_n, loop_ok, loop_n = 0, 0, 0, 0
    for i, j, _ in m:
        g, p = gt[i], pr[j]
        if np.linalg.norm(g[-1] - g[0]) >= 2.0:
            dir_n += 1
            dir_ok += int(np.linalg.norm(p[0] - g[0]) < np.linalg.norm(p[0] - g[-1]))
        A = signed_area(g)
        L = stroke_length(g)
        if abs(A) >= 4.0 and L > 0 and 4 * math.pi * abs(A) / L ** 2 >= 0.1:
            loop_n += 1
            loop_ok += int(np.sign(signed_area(p)) == np.sign(A))
    out["direction_acc"] = dir_ok / dir_n if dir_n else float("nan")
    out["loop_orient_acc"] = loop_ok / loop_n if loop_n else float("nan")
    out["n_dir_eligible"] = float(dir_n)
    out["n_loop_eligible"] = float(loop_n)
    return out


def score(gt: Sequence[np.ndarray], pred: Sequence[np.ndarray], canvas: Tuple[int, int],
          protocol: str = "native", step: float = 1.0, rmse_K: Optional[int] = None,
          with_lpips: bool = True, with_frechet: bool = True) -> Dict[str, float]:
    """Score one prediction against GT.  Both inks in evaluation pixel space on ``canvas``.

    Returns a dict of metric -> value (NaN where undefined) plus bookkeeping fields
    ``n_gt_pts``, ``n_pred_pts``, ``n_gt_strokes``, ``n_pred_strokes``, ``empty_pred``.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    gt = clean_ink(gt)
    pr = clean_ink(pred)
    if protocol == "standardised":
        gt = resample_ink_step(gt, step)
        pr = resample_ink_step(pr, step)
    H, W = canvas
    if rmse_K is None:
        rmse_K = 320 if (H <= 64 and W <= 64) else 2048
    res: Dict[str, float] = {k: float("nan") for k in METRICS}
    res.update(n_gt_pts=float(sum(map(len, gt))), n_pred_pts=float(sum(map(len, pr))),
               n_gt_strokes=float(len(gt)), n_pred_strokes=float(len(pr)),
               empty_pred=float(len(pr) == 0))
    if not gt:
        raise ValueError("empty ground truth")
    if not pr:
        res.update(stroke_count_err=float(-len(gt)), stroke_count_abs_err=float(len(gt)),
                   stroke_match_recall=0.0, strict_order_acc=0.0, penup_f1=0.0, aiou=0.0)
        return res
    G = to_nagamatsu(gt, canvas)
    P = to_nagamatsu(pr, canvas)
    nm = NM.trajectory_metrics(G, P, image_shape=(H, W))
    res.update(dtw=float(nm["dtw"]), ldtw=float(nm["ldtw"]), aiou=float(nm["aiou"]))
    A, B = np.vstack(gt), np.vstack(pr)
    res["chamfer"], res["hausdorff"] = chamfer_hausdorff(A, B)
    if with_frechet:
        res["frechet"] = discrete_frechet(A, B)
    ra, rb = resample_ink_fixed_count(gt, rmse_K), resample_ink_fixed_count(pr, rmse_K)
    res["rmse"] = float(np.sqrt(((ra - rb) ** 2).sum(axis=1).mean()))
    res["penup_f1"] = penup_f1(gt, pr)
    res.update(order_metrics(gt, pr, canvas))
    if with_lpips:
        res["lpips"] = lpips_distance(gt, pr, canvas)
    return res


def metric_table():
    """List of dicts describing the registered metrics (for documentation)."""
    return [dict(metric=k, direction=v[0], tag=v[1], definition=v[2]) for k, v in METRICS.items()]
