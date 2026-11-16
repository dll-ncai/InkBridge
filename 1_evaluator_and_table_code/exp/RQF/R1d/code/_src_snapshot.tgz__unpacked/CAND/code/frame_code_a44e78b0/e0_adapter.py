"""E0 adapter: OUHD-L ligature units v3.0  <->  Nagamatsu diffusion repo training format.

Nagamatsu repo (hirokinaga/Handwriting-Trajectory-Recovery-with-Diffusion-Models @ 7c5cf623)
expects per sample (src/utils/casia_grid.py PotDataset.__getitem__ + collate_fn):
  img    : float tensor (1, 64, 64) in [0, 1], WHITE ink (1) on BLACK background (0)
  coords : float tensor (n, 4) = [x, y, pen, eos], x/y in [-1, 1],
           pen = -1 on the LAST point of each stroke (+1 elsewhere),
           eos = -1 on the final point of the sequence (+1 elsewhere)
  collate_fn(sequence_length=320) truncates to 320 points or pads by repeating the final row.

This adapter (all decisions documented; no model code is changed):
  * Unit ink -> 64 x 64 evaluation canvas with uhwr_eval.fit_transform_unit (isotropic, bbox
    fitted into 56 x 56, centred, 4 px margin) -- the same canvas every uhwr ligature result uses.
    y stays DOWN (image row = y), x is not mirrored (RTL writing kept as recorded).
  * Target sequence: every stroke resampled at a uniform arc-length step (uhwr_eval
    resample_ink_step) of 1.0 px of the evaluation canvas; if the unit then has > 320 points the
    step is increased (x1.05 per try) until it fits, so no unit is truncated.  Single-point /
    zero-length strokes stay one point.  Encoded with uhwr_eval.to_nagamatsu
    (x_n = 2 x / 64 - 1, same pen/eos convention as the repo), so predictions map back to
    evaluation pixels exactly with uhwr_eval.from_normalized.
    (Deviation from the repo's CASIA path, which uses raw pen points and min-max to [-1, 1]
    without margin: raw OUHD-L units are 500 Hz and ~43% have > 320 points, so raw points would
    be truncated.)
  * Condition image 'rendered': uhwr_eval.render_image(native ink, (64, 64), thickness=2,
    antialias=True) -- the project's fixed renderer (2 px per 64 px, anti-aliased), identical to
    uhwr_baselines.inputs.rendered_unit(u, f=1).  Stored black-on-white uint8; the dataset class
    inverts it to the repo's white-on-black float.  No width / pressure / noise augmentation
    (the repo's CASIA training draws random thickness 1-3; E0 deliberately does not).
  * Condition image 'dataset' (test/gold only): the OUHD-L img/ PNG cropped and masked by
    uhwr_baselines.inputs.dataset_unit at f = 4 (256 x 256), then cv2.INTER_AREA to 64 x 64.
"""
from __future__ import annotations

import numpy as np

from uhwr_eval import (apply_transform, fit_transform_unit, from_normalized, render_image,
                       resample_ink_step, to_nagamatsu)

CANVAS = (64, 64)
SEQ_LEN = 320
BASE_STEP = 1.0
RENDER_THICKNESS = 2


def unit_eval_ink(strokes_xy):
    """Raw tablet strokes (list of (n,2)) -> (ink on the 64x64 evaluation canvas, (s, tx, ty))."""
    st = [np.asarray(s, np.float64)[:, :2] for s in strokes_xy]
    T = fit_transform_unit(st, CANVAS[0], 4)
    return apply_transform(st, *T), T


def resample_to_fit(ink, max_len=SEQ_LEN, base_step=BASE_STEP):
    """Per-stroke uniform arc-length resampling; step grows until the unit fits max_len points."""
    step = base_step
    while True:
        r = resample_ink_step(ink, step)
        n = sum(map(len, r))
        if n <= max_len:
            return r, step
        if len(r) > max_len:
            raise ValueError(f"unit has {len(r)} strokes > {max_len}")
        step *= 1.05


def encode(ink, max_len=SEQ_LEN, base_step=BASE_STEP):
    """Evaluation-canvas ink -> ((n,4) float32 [x, y, pen, eos], resampled ink, step)."""
    r, step = resample_to_fit(ink, max_len, base_step)
    return to_nagamatsu(r, CANVAS), r, step


def decode(arr):
    """(L,4) model output in Nagamatsu space -> ink (list of (n,2)) on the evaluation canvas.
    Trim after the first eos < 0 (inclusive; same rule as the repo's trim_at_end_flag); split
    strokes after every point with pen < 0."""
    a = np.asarray(arr, np.float64)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(a.shape)
    e = np.flatnonzero(a[:, 3] < 0) if a.shape[1] > 3 else np.array([], int)
    if e.size:
        a = a[: e[0] + 1]
    cut = np.flatnonzero(a[:, 2] < 0) + 1
    strokes = [s[:, :2] for s in np.split(a, cut) if len(s)]
    return from_normalized(strokes, CANVAS)


def render_condition(ink):
    """Fixed project renderer at 64x64: black ink on white, uint8."""
    return render_image(ink, CANVAS, RENDER_THICKNESS, antialias=True)


def pack_inks(inks):
    """List of inks -> (points (P,2) float64, stroke_id (P,) int32 local per unit, unit_offsets)."""
    pts, sid, offs = [], [], [0]
    for ink in inks:
        for j, s in enumerate(ink):
            pts.append(np.asarray(s, np.float64)); sid.append(np.full(len(s), j, np.int32))
        offs.append(offs[-1] + sum(map(len, ink)))
    return np.vstack(pts), np.concatenate(sid), np.array(offs, np.int64)


def unpack_ink(points, stroke_id, offsets, i):
    P, S = points[offsets[i]:offsets[i + 1]], stroke_id[offsets[i]:offsets[i + 1]]
    cut = np.flatnonzero(np.diff(S)) + 1
    return [p for p in np.split(P, cut)]
