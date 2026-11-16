"""FMDX backbone data: E0-format npz (seq, seq_offsets, unit_ids, scripts) + a 224-px render array in the same unit order
(uint8 (N,224,224) black-on-white .npy, memory-mapped; written by prep_renders.py with a .meta.json).

Train/held-out split = the Trainer1D rule of the LUO run (train_fraction 0.99, torch.utils.data.random_split with
torch.Generator().manual_seed(split_seed=0)), so the backbone trains on exactly the 79,200 units LUO ms6 trained on and
the 800 held-out units give the validation loss.

Batches: global batch B per optimiser micro-step, split over the ranks (Trainer1D split_batches semantics: B=128 on
2 ranks = 64 per rank). Order: source k, epoch e permutation = numpy default_rng([seed_now, k, e]).permutation(train positions);
rank r takes rows [k*B + r*B/W, k*B + (r+1)*B/W) of the permutation. Optional second source (pseudo-labels): each global
batch then holds round(frac*B) units drawn the same way from the second source.
REPA: target_row[global position] = row in the target array or -1 (no target -> denoising loss only for that unit).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from utils.casia_grid import collate_fn


def trainer1d_split(n, train_fraction=0.99, split_seed=0):
    """Indices of the Trainer1D random_split (train, held-out), as lists of int positions."""
    n_tr = int(train_fraction * n)
    tr, va = torch.utils.data.random_split(range(n), [n_tr, n - n_tr], generator=torch.Generator().manual_seed(split_seed))
    return np.asarray(list(tr.indices), np.int64), np.asarray(list(va.indices), np.int64)


class RenderSource:
    """One E0-format npz + its 224 render array (same order)."""

    def __init__(self, npz, render_npy, name, frame="identity", max_units=None):
        z = np.load(npz, allow_pickle=False)
        self.name = name
        self.unit_ids = z["unit_ids"].astype(str)
        self.scripts = z["scripts"].astype(str) if "scripts" in z.files else np.array([name] * len(self.unit_ids))
        self.seq = np.ascontiguousarray(z["seq"], np.float32)
        self.off = z["seq_offsets"].astype(np.int64)
        self.render_npy = str(render_npy)
        self._img = None
        need = len(self.unit_ids) if max_units is None else min(int(max_units), len(self.unit_ids))   # smoke: a prefix render file is allowed
        if self.img.shape[0] < need or (max_units is None and self.img.shape[0] != len(self.unit_ids)) \
                or self.img.shape[1:] != (224, 224) or self.img.dtype != np.uint8:
            raise SystemExit(f"render array {render_npy} {self.img.shape} {self.img.dtype} does not match {npz} ({len(self.unit_ids)} units)")
        if frame != "identity":
            raise SystemExit("only the identity frame is implemented in fmdx_backbone_v1 (LUO and COMP-p01 use identity)")
        self.n = len(self.unit_ids) if max_units is None else min(int(max_units), len(self.unit_ids))

    @property
    def img(self):
        """memory-mapped render array, opened lazily per process (spawn workers on Windows must not pickle the array)."""
        if self._img is None:
            self._img = np.load(self.render_npy, mmap_mode="r")
        return self._img

    def __getstate__(self):
        d = dict(self.__dict__); d["_img"] = None
        return d

    def item(self, i):
        img = (255.0 - np.asarray(self.img[i], np.float32)) / 255.0        # black-on-white -> white-on-black [0,1]
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(self.seq[self.off[i]:self.off[i + 1]])


class MixDataset(Dataset):
    """Index = (source k, position i) encoded as k * 10**9 + i."""

    def __init__(self, sources, target_rows=None):
        self.sources = sources
        self.target_rows = target_rows   # dict source_k -> int64 array (n_source,) of target rows or -1

    def __len__(self):
        return sum(s.n for s in self.sources)

    def __getitem__(self, key):
        k, i = divmod(int(key), 10 ** 9)
        img, coords = self.sources[k].item(i)
        tr = -1
        if self.target_rows is not None and k in self.target_rows:
            tr = int(self.target_rows[k][i])
        return {"img": img, "coords": coords, "tag_char": self.sources[k].unit_ids[i], "index": int(i), "source": k, "target_row": tr}


def collate(batch, sequence_length=320):
    out = collate_fn(batch, sequence_length=sequence_length)
    out["index"] = torch.tensor([b["index"] for b in batch])
    out["source"] = torch.tensor([b["source"] for b in batch])
    out["target_row"] = torch.tensor([b["target_row"] for b in batch])
    return out


class RankBatchSampler(Sampler):
    """Infinite per-rank batch sampler (see module docstring). positions: list (per source) of int64 arrays."""

    def __init__(self, positions, global_batch, rank, world, seed, fracs=None):
        assert global_batch % world == 0, (global_batch, world)
        self.pos = [np.asarray(p, np.int64) for p in positions]
        self.B, self.rank, self.world, self.seed = int(global_batch), int(rank), int(world), int(seed)
        fr = fracs or [1.0]
        counts = [int(round(f * self.B)) for f in fr]
        counts[0] = self.B - sum(counts[1:])
        assert all(c >= 0 for c in counts) and sum(counts) == self.B, counts
        self.counts = counts

    def __iter__(self):
        per = self.B // self.world
        streams = []
        for k, p in enumerate(self.pos):
            streams.append(self._stream(k, p))
        nb = 0
        while True:
            nb += 1
            glob = []
            for k, c in enumerate(self.counts):
                glob += [k * 10 ** 9 + int(next(streams[k])) for _ in range(c)]
            if len(self.counts) > 1:      # interleave sources deterministically so every rank gets the same mix
                rng = np.random.default_rng([self.seed, 7, nb])
                glob = list(np.asarray(glob)[rng.permutation(len(glob))])
            yield glob[self.rank * per:(self.rank + 1) * per]

    def _stream(self, k, p):
        e = 0
        while True:
            perm = np.random.default_rng([self.seed, k, e]).permutation(p)
            for x in perm:
                yield x
            e += 1

    def __len__(self):
        return 2 ** 31
