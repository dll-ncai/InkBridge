"""Loader for the OUHD-L ligature dataset (division v3.0).

    from ouhdl_ligatures import load_split
    idx, units = load_split('/path/to/ouhdl_ligatures_v3', 'train')
    strokes = units('1000_0014_0_L000')   # list of strokes; each (n, 4) int array: x, y, time, pressure
    for uid, strokes in units:            # iterate in index order

Coordinates are raw tablet units (y down); strokes are in recording (writing) order.
`idx` is the per-unit table (pandas DataFrame) from ligatures_v3_<split>.csv.
"""
import os
import numpy as np
import pandas as pd


class Units:
    def __init__(self, npz):
        z = np.load(npz, allow_pickle=False)
        self.points, self.offsets, self.ids = z['points'], z['unit_offsets'], z['unit_ids']
        self.pos = {str(u): i for i, u in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    def get(self, i):
        P = self.points[self.offsets[i]:self.offsets[i + 1]]
        cut = np.flatnonzero(np.diff(P[:, 4])) + 1
        return [s[:, :4] for s in np.split(P, cut)]

    def __call__(self, unit_id):
        return self.get(self.pos[unit_id])

    def __iter__(self):
        for i, u in enumerate(self.ids):
            yield str(u), self.get(i)


def load_split(root, split):
    idx = pd.read_csv(os.path.join(root, f'ligatures_v3_{split}.csv'))
    units = Units(os.path.join(root, f'ligatures_v3_{split}.npz'))
    assert list(idx.unit_id) == [str(u) for u in units.ids], 'index and arrays out of sync'
    return idx, units
