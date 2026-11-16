"""Rasterisation helpers (uniform-width, PEN-Net-style rendering)."""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

_SHIFT = 4  # sub-pixel precision bits for cv2 drawing
_F = 1 << _SHIFT


def render_mask(ink: Sequence[np.ndarray], canvas: Tuple[int, int], thickness: int = 2,
                antialias: bool = False) -> np.ndarray:
    """Binary (uint8 0/1) mask of the ink drawn with uniform width.  Single-point strokes
    (dots) are drawn as filled discs of diameter ``thickness`` so they are not lost."""
    H, W = canvas
    m = np.zeros((H, W), np.uint8)
    lt = cv2.LINE_AA if antialias else cv2.LINE_8
    for s in ink:
        s = np.asarray(s)
        if len(s) == 0:
            continue
        p = np.round(s * _F).astype(np.int32)
        if len(s) == 1 or np.ptp(s, axis=0).max() < 1e-6:
            cv2.circle(m, tuple(int(v) for v in p[0]), int(max(1, thickness // 2) * _F), 255,
                       -1, lt, _SHIFT)
        else:
            cv2.polylines(m, [p.reshape(-1, 1, 2)], False, 255, int(thickness), lt, _SHIFT)
    return m if antialias else (m > 0).astype(np.uint8)


def render_image(ink: Sequence[np.ndarray], canvas: Tuple[int, int], thickness: int = 2,
                 antialias: bool = True) -> np.ndarray:
    """Grey-level uint8 image, black ink on white background (the 'rendered' input
    condition and the LPIPS input)."""
    if antialias:
        m = render_mask(ink, canvas, thickness, antialias=True)
        return (255 - m).astype(np.uint8)
    m = render_mask(ink, canvas, thickness)
    return np.where(m > 0, 0, 255).astype(np.uint8)
