"""Torch dataset for E0: reads e0_<split>.npz (build_e0_data.py) and returns samples in the exact
format of the Nagamatsu repo's PotDataset.__getitem__, so Trainer1D / Tester1D / collate_fn work
unchanged:  {"img": (1,64,64) float in [0,1], white ink on black,
             "coords": (n,4) float [x, y, pen, eos], "tag_char": unit_id}.

Canonical frames (2026-10-28, e0_frames.py): `frame` in {identity, mirror_x, vt} is applied in __getitem__
to BOTH the stored uint8 condition (before the black/white inversion) and the stored sequence.  The E0 arrays
store the condition pre-rendered (e0_adapter.render_condition of the native ink at build time), so the
transform acts on the pixels, not on a re-render; e0_frames documents why the pixel map and the sequence map
are consistent with the renderer.  `frame_scripts` restricts the frame to units whose script tag is in the
set (tags from the npz key `scripts` when present, else `script_tag` for every unit); None = every unit.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

import e0_frames as F


class OuhdlE0Dataset(Dataset):
    def __init__(self, npz_path, condition="rendered", indices=None, frame="identity", frame_scripts=None,
                 script_tag=None):
        z = np.load(npz_path, allow_pickle=False)
        key = f"cond_{condition}"
        if key not in z.files:
            raise KeyError(f"{npz_path} has no {key}; available: {[f for f in z.files if f.startswith('cond_')]}")
        self.cond = z[key]
        if self.cond.shape[1:] != (64, 64):
            raise ValueError(f"{key} is {self.cond.shape[1:]}, expected 64x64")
        self.seq, self.off = z["seq"], z["seq_offsets"]
        self.unit_ids = z["unit_ids"].astype(str)
        n = len(self.unit_ids)
        if "scripts" in z.files:
            self.scripts = z["scripts"].astype(str)
        else:
            self.scripts = np.full(n, "" if script_tag is None else str(script_tag))
        self.idx = np.arange(n) if indices is None else np.asarray(indices)
        self.frame = F.check_frame(frame)
        self.frame_scripts = None if frame_scripts is None else set(frame_scripts)
        # per-unit frame (identity for units outside frame_scripts); None mask == frame everywhere
        if self.frame == "identity" or self.frame_scripts is None:
            self.framed = np.full(n, self.frame != "identity", bool)
        else:
            self.framed = np.isin(self.scripts, list(self.frame_scripts))

    def __len__(self):
        return len(self.idx)

    def unit_frame(self, i):
        """Frame actually applied to item i (identity if the unit is outside frame_scripts)."""
        j = int(self.idx[int(i)])
        return self.frame if self.framed[j] else "identity"

    def __getitem__(self, i):
        j = int(self.idx[int(i)])
        cond = self.cond[j]
        coords = self.seq[self.off[j]:self.off[j + 1]]
        if self.framed[j]:
            cond = F.frame_cond(cond, self.frame)
            coords = F.frame_seq(coords, self.frame)
        img = (255 - cond.astype(np.float32)) / 255.0      # black-on-white -> white-on-black
        return {"img": torch.from_numpy(img).unsqueeze(0),
                "coords": torch.from_numpy(np.ascontiguousarray(coords, np.float32)),
                "tag_char": self.unit_ids[j]}
