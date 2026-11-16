"""OUHD-L adapter: raw Wacom CSV -> shared stroke format, line and ligature units, inputs.

Data: OUHD-L v1.0 (Zenodo 10.5281/zenodo.20642162), core archive; Zenodo writer-disjoint
split manifests train.csv / val.csv / test.csv (1,911 / 239 / 253 lines).

Strokes: maximal runs of consecutive rows with ``pen_down == 1`` (hover rows, pen_down == 0,
are dropped).  Coordinates: ``X cood.``, ``Y cood.`` (tablet units, y down).  No NaN rows.

Units
-----
line      : all strokes of the line, in writing order, mapped to the line evaluation canvas
            (height 64; isotropic scale so the ink bbox height is 56 px; 4 px margins).
ligature  : connected-component units.  In line-canvas space every stroke is drawn as a
            2 px mask; strokes whose masks touch/overlap (8-connectivity) are merged
            (union-find) into components.  A component is a *diacritic* if both bbox sides
            are < 0.18 x 56 px (~10 px) and it is not the only component; otherwise a
            *main body*.  Each diacritic is attached to the main body whose x-extent contains
            the diacritic's x-centre (nearest ink among those, if several); if none, to the
            main body with the nearest ink point.  Unit strokes keep their original temporal
            order.  Units are ordered right-to-left by main-body max x.  Each unit is
            normalised to a 64 x 64 canvas (isotropic, ink bbox fitted into 56 x 56, centred).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from uhwr_eval import apply_transform, fit_transform_line, fit_transform_unit, render_mask
from uhwr_eval.strokes import ink_bbox

DOT_FRAC = 0.18
LINE_H, LINE_MARGIN = 64, 4
UNIT_SIZE, UNIT_MARGIN = 64, 4


def load_manifest(root: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, f"{split}.csv"))
    # sample_id = <series>_<board>_<line>; board ids repeat across line series, so the
    # series (Data_<series>) is required for uniqueness
    df["sample_id"] = df["csv"].map(
        lambda p: p.split("/")[-3].replace("Data_", "") + "_" + os.path.splitext(os.path.basename(p))[0].replace("csv_", ""))
    assert df["sample_id"].is_unique
    return df


def load_raw_strokes(csv_path: str) -> List[np.ndarray]:
    d = pd.read_csv(csv_path, usecols=["X cood.", "Y cood.", "pen_down"])
    pdn = d["pen_down"].to_numpy() == 1
    xy = d[["X cood.", "Y cood."]].to_numpy(np.float64)
    edges = np.diff(np.r_[0, pdn.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    strokes = [xy[s:e] for s, e in zip(starts, ends)]
    strokes = [s[np.isfinite(s).all(1)] for s in strokes]
    return [s for s in strokes if len(s)]


@dataclass
class LineSample:
    sample_id: str
    raw: List[np.ndarray]                 # tablet units
    ink: List[np.ndarray]                 # line evaluation canvas px
    canvas: Tuple[int, int]
    T: Tuple[float, float, float]         # (s, tx, ty): raw -> line canvas
    units: List[Dict] = field(default_factory=list)


def make_line(sample_id: str, raw: List[np.ndarray]) -> LineSample:
    s, tx, ty, canvas = fit_transform_line(raw, LINE_H, LINE_MARGIN)
    return LineSample(sample_id, raw, apply_transform(raw, s, tx, ty), canvas, (s, tx, ty))


def _components(ink, canvas, thickness=2):
    n = len(ink)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    masks, boxes = [], []
    for s in ink:
        x0, y0, x1, y1 = ink_bbox([s])
        boxes.append((x0 - 3, y0 - 3, x1 + 3, y1 + 3))
        masks.append(render_mask([s], canvas, thickness).astype(bool))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a[0] > b[2] or b[0] > a[2] or a[1] > b[3] or b[1] > a[3]:
                continue
            # touch under 8-connectivity: dilate one mask by 1 px
            if np.logical_and(cv2.dilate(masks[i].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0,
                              masks[j]).any():
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def make_units(line: LineSample, dot_frac: float = DOT_FRAC) -> List[Dict]:
    ink, canvas = line.ink, line.canvas
    comps = _components(ink, canvas)
    ref_h = LINE_H - 2 * LINE_MARGIN
    info = []
    for c in comps:
        x0, y0, x1, y1 = ink_bbox([ink[i] for i in c])
        dot = (x1 - x0) < dot_frac * ref_h and (y1 - y0) < dot_frac * ref_h
        info.append(dict(strokes=c, bbox=(x0, y0, x1, y1), dot=dot))
    mains = [q for q in info if not q["dot"]]
    if not mains:  # degenerate: treat everything as one unit
        mains, dots = [dict(strokes=sorted(sum([q["strokes"] for q in info], [])),
                            bbox=ink_bbox(ink), dot=False)], []
    else:
        dots = [q for q in info if q["dot"]]
    trees = [cKDTree(np.vstack([ink[i] for i in m["strokes"]])) for m in mains]
    for m in mains:
        m["members"] = list(m["strokes"])
    for d in dots:
        pts = np.vstack([ink[i] for i in d["strokes"]])
        xc = 0.5 * (d["bbox"][0] + d["bbox"][2])
        cand = [k for k, m in enumerate(mains) if m["bbox"][0] <= xc <= m["bbox"][2]]
        pool = cand if cand else range(len(mains))
        dist = {k: float(trees[k].query(pts)[0].min()) for k in pool}
        k = min(dist, key=dist.get)
        mains[k]["members"].extend(d["strokes"])
    mains.sort(key=lambda m: -m["bbox"][2])
    units = []
    for u, m in enumerate(mains):
        idx = sorted(m["members"])  # temporal order
        strokes_line = [ink[i] for i in idx]
        s, tx, ty = fit_transform_unit(strokes_line, UNIT_SIZE, UNIT_MARGIN)
        units.append(dict(unit_id=f"{line.sample_id}_u{u:03d}", stroke_idx=idx,
                          n_diacritic_strokes=len(idx) - len(m["strokes"]),
                          line_bbox=ink_bbox(strokes_line), T=(s, tx, ty),
                          ink=apply_transform(strokes_line, s, tx, ty),
                          canvas=(UNIT_SIZE, UNIT_SIZE)))
    line.units = units
    return units


def render_input(ink, canvas, f: float = 1.0, width_per_64: float = 2.0) -> np.ndarray:
    """'Rendered' input condition: uniform-width (2 px per 64 px of canvas height),
    anti-aliased, black-on-white rendering of the GT strokes at f x the evaluation canvas."""
    from uhwr_eval import render_image
    H, W = canvas
    Hc, Wc = int(round(H * f)), int(round(W * f))
    th = max(1, int(round(width_per_64 * f * (H / 64.0) if H <= 64 else width_per_64 * f)))
    return render_image([s * f for s in ink], (Hc, Wc), th, antialias=True)
