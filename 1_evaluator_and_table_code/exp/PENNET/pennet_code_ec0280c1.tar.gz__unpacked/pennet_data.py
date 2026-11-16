"""PEN-Net <-> E0 arrays (data/ouhdl_e0_v3/e0_<split>.npz, built by experiments/e0/build_e0_data.py).

Input image: the same 64x64 condition image E0 uses (cond_rendered for train/val/test/gold; cond_dataset for test/gold),
stored black ink on white uint8.  PEN-Net's released loader feeds white ink on black, scaled (v - 127) / 127
(casia.py: img_tensor = (img - 127.) / 127.), so the image is inverted and scaled that way.

Target: the E0 target sequence (seq, Nagamatsu space x_n = 2x/64 - 1, <=320 points, 1-px arc-length resampling on the
64x64 evaluation canvas) converted to PEN-Net's 5-d rows [x, y, s_down, s_up, s_eos]:
  x, y      = absolute evaluation-canvas pixels, (x_n + 1) * 32  -> [0, 64)                       (paper sec. 5.2)
  s_up = 1  on the last point of every stroke (Nagamatsu pen < 0), s_down = 1 elsewhere              (sketch-rnn [34])
  one EOS row [x_last, y_last, 0, 0, 1] appended after the last point                               (choice R5)
Batches are padded with rows [0, 0, 0, 0, 1] (released collate_fn_).
"""
import numpy as np
import torch

CANVAS = 64.0
MAX_POINTS = 320
MAX_ROWS = MAX_POINTS + 1


def load_split(npz_path, condition="rendered"):
    z = np.load(npz_path, allow_pickle=False)
    key = f"cond_{condition}"
    if key not in z.files:
        raise KeyError(f"{npz_path} has no {key}")
    cond = z[key]
    assert cond.shape[1:] == (64, 64), cond.shape
    return dict(cond=cond, seq=z["seq"], off=z["seq_offsets"], unit_ids=z["unit_ids"].astype(str),
                writer_ids=z["writer_ids"], sample_ids=z["sample_ids"].astype(str))


def image_tensor(cond_u8):
    """(N,64,64) uint8 black-on-white -> (N,1,64,64) float32 white-on-black, (v - 127) / 127."""
    v = 255.0 - cond_u8.astype(np.float32)
    return torch.from_numpy(((v - 127.0) / 127.0)[:, None])


def nagamatsu_to_rows(arr):
    """(n,4) [x_n, y_n, pen, eos] -> (n+1, 5) PEN-Net rows incl. the EOS row."""
    a = np.asarray(arr, np.float64)
    n = len(a)
    rows = np.zeros((n + 1, 5), np.float32)
    rows[:n, 0] = (a[:, 0] + 1.0) * (CANVAS / 2)
    rows[:n, 1] = (a[:, 1] + 1.0) * (CANVAS / 2)
    up = a[:, 2] < 0
    rows[:n, 3] = up
    rows[:n, 2] = ~up
    rows[n, :2] = rows[n - 1, :2]
    rows[n, 4] = 1.0
    return rows


def build_label_tensor(d):
    """All units of a split -> (labels (N, MAX_ROWS, 5) float32 padded with EOS rows, lengths (N,) incl. EOS row)."""
    off = d["off"]
    N = len(off) - 1
    lab = np.zeros((N, MAX_ROWS, 5), np.float32)
    lab[:, :, 4] = 1.0
    lens = np.zeros(N, np.int64)
    for k in range(N):
        r = nagamatsu_to_rows(d["seq"][off[k]:off[k + 1]])
        if len(r) > MAX_ROWS:
            raise ValueError(f"unit {k} has {len(r) - 1} points > {MAX_POINTS}")
        lab[k, :len(r)] = r
        lens[k] = len(r)
    return torch.from_numpy(lab), torch.from_numpy(lens)


def rows_to_nagamatsu(rows, length, L=MAX_POINTS):
    """Decoded PEN-Net rows (T,5) with emitted length (incl. EOS row if any) -> ((L,4) Nagamatsu array, flags).
    Points = rows before the EOS row (or all emitted rows if no EOS, truncated to L).  The last point gets pen=-1 and
    eos=-1; points whose arg-max state is pen-up get pen=-1.  Rows after the end repeat the final row (E0/Nagamatsu
    padding convention), so experiments/e0/e0_adapter.decode() recovers exactly these strokes."""
    r = np.asarray(rows, np.float64)[:int(length)]
    st = np.argmax(r[:, 2:5], axis=1) if len(r) else np.zeros(0, int)
    has_eos = bool(len(st)) and st[-1] == 2
    pts = r[:-1] if has_eos else r
    flags = dict(no_eos=not has_eos, empty=False, truncated=False)
    if len(pts) == 0:                       # EOS emitted first: keep the EOS row position as one point
        pts = r[:1]
        flags["empty"] = True
        st_p = np.array([1])
    else:
        st_p = np.argmax(pts[:, 2:5], axis=1)
    if len(pts) > L:
        pts, st_p = pts[:L], st_p[:L]
        flags["truncated"] = True
    n = len(pts)
    out = np.ones((L, 4), np.float32)
    out[:n, 0] = pts[:, 0] / (CANVAS / 2) - 1.0
    out[:n, 1] = pts[:, 1] / (CANVAS / 2) - 1.0
    out[:n, 2] = np.where(st_p == 1, -1.0, 1.0)
    out[n - 1, 2] = -1.0
    out[n - 1, 3] = -1.0
    out[n:] = out[n - 1]
    return out, flags
