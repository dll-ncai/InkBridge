"""Torch dataset for E0: reads e0_<split>.npz (build_e0_data.py) and returns samples in the exact
format of the Nagamatsu repo's PotDataset.__getitem__, so Trainer1D / Tester1D / collate_fn work
unchanged:  {"img": (1,64,64) float in [0,1], white ink on black,
             "coords": (n,4) float [x, y, pen, eos], "tag_char": unit_id}.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class OuhdlE0Dataset(Dataset):
    def __init__(self, npz_path, condition="rendered", indices=None):
        z = np.load(npz_path, allow_pickle=False)
        key = f"cond_{condition}"
        if key not in z.files:
            raise KeyError(f"{npz_path} has no {key}; available: {[f for f in z.files if f.startswith('cond_')]}")
        self.cond = z[key]
        if self.cond.shape[1:] != (64, 64):
            raise ValueError(f"{key} is {self.cond.shape[1:]}, expected 64x64")
        self.seq, self.off = z["seq"], z["seq_offsets"]
        self.unit_ids = z["unit_ids"].astype(str)
        self.idx = np.arange(len(self.unit_ids)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[int(i)])
        img = (255 - self.cond[j].astype(np.float32)) / 255.0      # black-on-white -> white-on-black
        coords = self.seq[self.off[j]:self.off[j + 1]]
        return {"img": torch.from_numpy(img).unsqueeze(0),
                "coords": torch.from_numpy(np.ascontiguousarray(coords, np.float32)),
                "tag_char": self.unit_ids[j]}
