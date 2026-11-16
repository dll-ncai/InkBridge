"""Shared helpers for the uhwr preliminary experiments E1, E2, M1a, M2a, E3a.

Data access goes through bundle files built by ``build_inputs.py`` (see PROVENANCE_e_inputs.json):
  lines_test.pkl : {sample_id: dict(raw=[(n,2) tablet xy], t=[(n,) ms], writer_id, text, img)}
                   pen-down strokes of every OUHD-L test line (runs of pen_down == 1, as in
                   uhwr_baselines.ouhdl.load_raw_strokes) plus the per-point Time column.
  img/<sample_id>.png : OUHD-L dataset image of the line (copied from Dataset/.../img/).
  ligatures_v3_test.csv : v3 unit index (stroke_idx = indices into the line's pen-down strokes).
"""
from __future__ import annotations

import ast
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from uhwr_baselines import ouhdl as O
from uhwr_eval import apply_transform, fit_transform_unit

SEED = 20261016


def load_lines(bundle: str) -> Dict[str, dict]:
    return pickle.load(open(os.path.join(bundle, "lines_test.pkl"), "rb"))


def line_sample(d: dict, sid: str) -> O.LineSample:
    return O.make_line(sid, d["raw"])


def v3_units(bundle: str, lines: Dict[str, dict]):
    """v3 test units in the same dict layout as uhwr_baselines.ouhdl.make_units (unit canvas
    64 x 64, ink fitted into 56 x 56), plus body/mark flags and per-point times (ms).
    Returns (units, {sample_id: LineSample})."""
    idx = pd.read_csv(os.path.join(bundle, "ligatures_v3_test.csv"))
    L = {}
    out = []
    for r in idx.itertuples(index=False):
        if r.sample_id not in L:
            L[r.sample_id] = line_sample(lines[r.sample_id], r.sample_id)
        line = L[r.sample_id]
        order = sorted(ast.literal_eval(r.stroke_idx))  # writing order within the unit
        body = set(ast.literal_eval(r.body_stroke_idx))
        strokes_line = [line.ink[i] for i in order]
        s, tx, ty = fit_transform_unit(strokes_line, O.UNIT_SIZE, O.UNIT_MARGIN)
        out.append(dict(unit_id=r.unit_id, sample_id=r.sample_id, writer_id=r.writer_id,
                        stroke_idx=order, is_body=[i in body for i in order],
                        T=(s, tx, ty), line_scale=line.T[0],
                        ink=apply_transform(strokes_line, s, tx, ty),
                        t=[np.asarray(lines[r.sample_id]["t"][i], float) for i in order],
                        canvas=(O.UNIT_SIZE, O.UNIT_SIZE), aligned_text=r.aligned_text))
    return out, L


def boot_ci(values, clusters, B=2000, seed=0):
    from uhwr_baselines.aggregate import boot_ci as _b
    return _b(values, clusters, B=B, seed=seed)
