"""Canonical-frame transforms for the E0 pipeline (C1 "direction-canonical transfer", uhwr 2026-10-28).

A frame is a rigid transform of one training/evaluation unit that is applied CONSISTENTLY to
  (a) the stored 64x64 condition image  (uint8, black ink 0 on white 255, as in e0_<split>.npz cond_*), and
  (b) the stored target sequence         ((n, 4) float32 [x, y, pen, eos] in Nagamatsu space, x_n = 2x/W - 1),
and whose exact inverse is applied to sampled trajectories BEFORE uhwr_eval scoring, so every metric is
computed in the original (unmirrored) frame against the original GT.

Frames
  identity  : nothing.
  mirror_x  : x -> W - x on the trajectory (x_n -> -x_n), columns c -> W - c on the condition.  Nothing else.
  vt        : y -> H - y on the trajectory (y_n -> -y_n), rows r -> H - r on the condition, PLUS per-stroke
              time reversal: the point order inside every pen-down stroke is reversed; the stroke slices keep
              their position in the sequence, so the pen-up flag stays on the (new) last point of each stroke,
              eos stays on the last point of the sequence and the inter-stroke order is unchanged.
              (= the unique element of the reflection x time-reversal group that flips horizontal progression
              without flipping chirality; see uhwr_method_candidates_2026-10-27.md section 2.)

Pixel convention (decided by measurement, see test_e0_frames.py / smoke_report.json)
  uhwr_eval.render_image draws with cv2 at sub-pixel shift 4, i.e. a trajectory point at continuous y = r
  lands on the CENTRE of pixel row r.  to_nagamatsu maps continuous [0, 64] onto [-1, 1].  The only reflection
  that is bitwise exact in float32 is a sign flip (y_n -> -y_n, i.e. y -> 64 - y about the canvas centre 32.0
  where fit_transform_unit centres the ink), and the pixel map consistent with it is r -> 64 - r, which sends
  row 64 (non-existent) to row 0: we therefore use the SHIFTED flip  out[0] = background, out[r] = img[64 - r]
  for r = 1..63  (np.flipud rolled by one row).  It is an exact involution whenever row 0 (resp. column 0) is
  background, which holds for every rendered condition in the E0 arrays (4 px margin, 2 px stroke; verified on
  all splits in the tests).  The plain np.flipud / y -> 63 - y alternative would be geometrically consistent
  too, but (i) y_n -> -y_n - 1/32 is not bitwise invertible in float32 for |y_n| < 1/32, and (ii) it shifts the
  ink centre by one pixel relative to every identity-frame training unit.  Decision recorded here on purpose.

All three frames are involutions, so inverse == forward; the inverse_* names exist for readability at the
call site.  Functions never modify their inputs.
"""
from __future__ import annotations

import numpy as np

FRAMES = ("identity", "mirror_x", "vt")
CANVAS = (64, 64)
BACKGROUND = 255  # stored condition images are black-on-white uint8


def check_frame(frame: str) -> str:
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; expected one of {FRAMES}")
    return frame


# ----------------------------------------------------------------------------------------------------------------
# condition images (uint8, black-on-white; (H, W) or (N, H, W))
# ----------------------------------------------------------------------------------------------------------------
def _shifted_flip(img: np.ndarray, axis: int, background) -> np.ndarray:
    """out[0] = background, out[i] = img[n - i] for i = 1..n-1 along `axis` (exact involution iff img[0] is
    background along that axis).  Works on the last two axes of an (..., H, W) array; axis is -2 (rows) or -1."""
    out = np.flip(img, axis=axis)
    out = np.roll(out, 1, axis=axis)
    sl = [slice(None)] * out.ndim
    sl[axis] = 0
    out[tuple(sl)] = background
    return out


def frame_cond(img: np.ndarray, frame: str, background=BACKGROUND) -> np.ndarray:
    """Apply `frame` to a stored condition image (uint8 black-on-white, (H, W) or (N, H, W)).  Returns a copy."""
    check_frame(frame)
    img = np.asarray(img)
    if frame == "identity":
        return img.copy()
    if frame == "mirror_x":
        return _shifted_flip(img, -1, background)
    return _shifted_flip(img, -2, background)  # vt: rows


inverse_frame_cond = frame_cond  # involution


def edge_clean(img: np.ndarray, frame: str, background=BACKGROUND) -> np.ndarray:
    """Per-image bool: the pixel row/column that the shifted flip discards is pure background (so the frame is
    lossless on this image).  identity -> all True."""
    img = np.asarray(img)
    if frame == "identity":
        return np.ones(img.shape[:-2], bool)
    if frame == "mirror_x":
        return (img[..., :, 0] == background).all(-1)
    return (img[..., 0, :] == background).all(-1)


# ----------------------------------------------------------------------------------------------------------------
# sequences in Nagamatsu space  ((n, 4) float32 [x, y, pen, eos]; pen < 0 on the last point of a stroke,
# eos < 0 on the last point of the sequence)
# ----------------------------------------------------------------------------------------------------------------
def stroke_slices(seq: np.ndarray, valid_len: int | None = None):
    """[(start, stop), ...] of the pen-down strokes in seq[:valid_len], split AFTER every point with pen < 0
    (same rule as e0_adapter.decode); a trailing partial stroke (no pen-up before the end) is included."""
    a = np.asarray(seq)
    n = len(a) if valid_len is None else int(valid_len)
    cut = np.flatnonzero(a[:n, 2] < 0) + 1
    starts = np.concatenate([[0], cut])
    stops = np.concatenate([cut, [n]])
    return [(int(s), int(e)) for s, e in zip(starts, stops) if e > s]


def valid_length(seq: np.ndarray) -> int:
    """Length up to and including the first point with eos < 0 (e0_adapter.decode rule); len(seq) if none."""
    a = np.asarray(seq)
    if a.shape[1] < 4:
        return len(a)
    e = np.flatnonzero(a[:, 3] < 0)
    return int(e[0]) + 1 if e.size else len(a)


def frame_seq(seq: np.ndarray, frame: str, respect_eos: bool = True) -> np.ndarray:
    """Apply `frame` to a Nagamatsu-space sequence.  Bitwise exact in float32 (sign flips + permutations only).

    respect_eos=True (default) treats everything after the first eos < 0 as padding: those rows get the
    coordinate sign flip but are not re-ordered (they are never decoded).  For stored GT the eos is on the last
    point, so this makes no difference.  Returns a new array with the dtype of the input."""
    check_frame(frame)
    a = np.array(seq, copy=True)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"expected (n, >=3) [x, y, pen(, eos)], got {a.shape}")
    if frame == "identity":
        return a
    if frame == "mirror_x":
        a[:, 0] = -a[:, 0]
        return a
    # vt: vertical sign flip + per-stroke time reversal
    a[:, 1] = -a[:, 1]
    n = valid_length(a) if (respect_eos and a.shape[1] >= 4) else len(a)
    for s, e in stroke_slices(a, n):
        a[s:e, :2] = a[s:e, :2][::-1]
    return a


inverse_frame_seq = frame_seq  # involution


def frame_preds(preds: np.ndarray, frame: str) -> np.ndarray:
    """Apply the (inverse) frame to a batch of sampled trajectories (N, L, 4) in Nagamatsu space, unit by unit,
    with the eos/pen thresholds of e0_adapter.decode.  This is what goes into *_preds.npz before e0_score.py."""
    check_frame(frame)
    p = np.asarray(preds)
    if p.ndim != 3:
        raise ValueError(f"expected (N, L, 4), got {p.shape}")
    if frame == "identity":
        return p.copy()
    return np.stack([frame_seq(u, frame, respect_eos=True) for u in p]).astype(p.dtype, copy=False)


inverse_frame_preds = frame_preds  # involution


# ----------------------------------------------------------------------------------------------------------------
# inks in evaluation-pixel space (list of (n, 2) arrays), for checks and for scoring code that works on inks
# ----------------------------------------------------------------------------------------------------------------
def frame_ink(ink, frame: str, canvas=CANVAS):
    """Same transform on a pixel-space ink: mirror_x x -> W - x; vt y -> H - y and each stroke reversed."""
    check_frame(frame)
    H, W = canvas
    out = []
    for s in ink:
        s = np.array(s, dtype=np.float64, copy=True)
        if frame == "mirror_x":
            s[:, 0] = W - s[:, 0]
        elif frame == "vt":
            s[:, 1] = H - s[:, 1]
            s = s[::-1].copy()
        out.append(s)
    return out


inverse_frame_ink = frame_ink  # involution


def describe(frame: str) -> dict:
    """Machine-readable description for sidecars."""
    check_frame(frame)
    return {
        "frame": frame,
        "condition_map": {"identity": "none", "mirror_x": "col c -> W - c (c=1..W-1), col 0 := background",
                          "vt": "row r -> H - r (r=1..H-1), row 0 := background"}[frame],
        "trajectory_map": {"identity": "none", "mirror_x": "x_n -> -x_n", "vt": "y_n -> -y_n; point order reversed inside each pen-down stroke; pen/eos columns and stroke order unchanged"}[frame],
        "inverse": "same transform (involution), applied to sampled (N, L, 4) arrays before uhwr_eval",
        "pixel_convention": "cv2 shift-4 rendering: continuous y = r is the centre of pixel row r; flips are about the canvas centre 32.0",
        "module": "e0_frames.py",
    }
