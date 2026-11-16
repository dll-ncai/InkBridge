"""Input conditions for OUHD-L units.

'rendered'  : our uniform-width (PEN-Net-style) anti-aliased rendering of the GT strokes,
              2 px per 64 px of evaluation-canvas height, drawn at f x the evaluation canvas.
'dataset'   : the image shipped by OUHD-L (``img/*.png``).  Verified to be a rendering of the
              same online strokes (tight min-max crop, ~3-4 px uniform width, isotropic scale,
              >= 99% of pen-down samples on ink, width uncorrelated with pressure); it is NOT a
              scan of the physical paper.  For ligature units the image region is cropped and
              all ink farther than ``mask_px`` from the unit's own strokes is erased (oracle
              unit segmentation, the same oracle used for the rendered unit images).

Every input comes with an affine map back to the evaluation canvas so that predictions in
input-image pixels can be scored.  Maps are axis-aligned: p_eval = A * p_img + b.
"""
from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

from uhwr_eval import render_image, render_mask
from .ouhdl import LineSample

Affine = Tuple[np.ndarray, np.ndarray]  # (A diag (2,), b (2,))


def to_eval(strokes, aff: Affine):
    A, b = aff
    return [np.asarray(s, float) * A + b for s in strokes]


def rendered_line(line: LineSample, f: float = 1.0):
    H, W = line.canvas
    img = render_image([s * f for s in line.ink], (int(round(H * f)), int(round(W * f))),
                       max(1, int(round(2 * f))), antialias=True)
    return img, (np.array([1 / f, 1 / f]), np.zeros(2))


def rendered_unit(unit: dict, f: float = 4.0):
    H, W = unit["canvas"]
    img = render_image([s * f for s in unit["ink"]], (int(H * f), int(W * f)),
                       max(1, int(round(2 * f))), antialias=True)
    return img, (np.array([1 / f, 1 / f]), np.zeros(2))


def dataset_image(root: str, rel: str) -> np.ndarray:
    return cv2.imread(os.path.join(root, rel), cv2.IMREAD_GRAYSCALE)


def img_to_raw_affine(line: LineSample, img_shape) -> Affine:
    """OUHD-L img/ is a tight min-max crop of the raw coordinates: x_img = (x-x0)*(W-1)/(x1-x0)."""
    Hi, Wi = img_shape
    P = np.vstack(line.raw)
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    A = np.array([(x1 - x0) / (Wi - 1), (y1 - y0) / (Hi - 1)])
    return A, np.array([x0, y0])


def compose(a1: Affine, a2: Affine) -> Affine:
    """Apply a1 then a2."""
    return a1[0] * a2[0], a1[1] * a2[0] + a2[1]


def dataset_line(line: LineSample, img: np.ndarray) -> Affine:
    s, tx, ty = line.T
    return compose(img_to_raw_affine(line, img.shape), (np.array([s, s]), np.array([tx, ty])))


def dataset_unit(line: LineSample, unit: dict, img: np.ndarray, f: float = 4.0, mask_px: int = 5):
    """Crop+mask the dataset image for one unit and resample it onto the unit canvas x f.
    Returns (image, affine image->unit-eval)."""
    img2line = dataset_line(line, img)
    s, tx, ty = unit["T"]
    img2unit = compose(img2line, (np.array([s, s]), np.array([tx, ty])))
    # mask: unit strokes drawn in image px with a generous width
    line2img = (1 / img2line[0], -img2line[1] / img2line[0])
    strokes_img = to_eval([line.ink[i] for i in unit["stroke_idx"]], line2img)
    m = render_mask(strokes_img, img.shape, thickness=2 * mask_px + 1)
    clean = np.where(m > 0, img, 255).astype(np.uint8)
    A, b = img2unit
    M = np.array([[A[0] * f, 0, b[0] * f], [0, A[1] * f, b[1] * f]], np.float32)
    H, W = unit["canvas"]
    out = cv2.warpAffine(clean, M, (int(W * f), int(H * f)), flags=cv2.INTER_LINEAR,
                         borderValue=255)
    return out, (np.array([1 / f, 1 / f]), np.zeros(2))
