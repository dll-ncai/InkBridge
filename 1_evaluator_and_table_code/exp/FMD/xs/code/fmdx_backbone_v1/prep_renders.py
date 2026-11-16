"""FMDX backbone data prep: 224-px renders (fmd_harness 1.0.1 render.render224 = uhwr_eval.render_image, 64-px eval-canvas
geometry x 3.5, thickness 7, anti-aliased, black ink on white, uint8) with a meta provenance file per output.

  seqset  E0-format npz WITHOUT gt ink (m5_luo_train.npz: seq, seq_offsets, cond_rendered): ink = e0_adapter.decode(seq)
          (the resampled training trajectory, 1 px arc-length steps or the grown step, on the 64-px canvas). Fidelity
          check per unit: render64(decoded ink) vs the stored cond_rendered (rendered by m5_build from the raw ink):
          byte-identical count, max |diff|, ink-pixel IoU (< 128). Output: <out>.npy (N,224,224) uint8 in npz order.
  gtset   E0-format npz WITH gt ink (gt_points / gt_stroke_id / gt_offsets; m5_val_<s>.npz, test sets): ink = gt ink
          (harness rule); check64 = render64(gt ink) == cond_rendered byte for byte (count). Output: <out>.npz
          (unit_ids, img224, cond64) and <out>.npy (img224 only).
  npz2npy existing harness render npz (img224 + unit_ids) -> .npy in the order of a given E0 npz (unit order checked).
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from fmd_harness import common as C
from fmd_harness import render as RH
import e0_adapter as A

_Z = {}


def _init_seq(npz):
    z = np.load(npz, allow_pickle=False)
    _Z.update(seq=z["seq"], off=z["seq_offsets"], cond=z["cond_rendered"])


def _work_seq(i):
    s = _Z["seq"][_Z["off"][i]:_Z["off"][i + 1]]
    ink = A.decode(s)
    im = RH.render224(ink)
    r64 = RH.render64(ink)
    c = _Z["cond"][i]
    d = int(np.abs(r64.astype(int) - c.astype(int)).max())
    a, b = r64 < 128, c < 128
    iou = float((a & b).sum() / max(1, (a | b).sum()))
    return i, im, d, iou


def seqset(npz, out, procs, max_units=None):
    z = np.load(npz, allow_pickle=False)
    n = len(z["unit_ids"]) if max_units is None else min(max_units, len(z["unit_ids"]))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name[:-4] + ".part.npy")
    M = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, 224, 224))
    d = np.zeros(n, np.int32); iou = np.zeros(n, np.float64)
    with get_context("fork").Pool(procs, initializer=_init_seq, initargs=(str(npz),)) as pool:
        for i, im, di, io in pool.imap_unordered(_work_seq, range(n), chunksize=64):
            M[i] = im; d[i] = di; iou[i] = io
    M.flush(); del M
    os.replace(tmp, out)
    X = np.load(out, mmap_mode="r")
    sc = z["scripts"].astype(str)[:n] if "scripts" in z.files else np.array(["?"] * n)
    per_script = {}
    for s in np.unique(sc):
        k = sc == s
        per_script[str(s)] = dict(n=int(k.sum()), identical64=int((d[k] == 0).sum()), maxabs64_p50=float(np.median(d[k])),
                                  maxabs64_max=int(d[k].max()), ink_iou64_mean=float(iou[k].mean()), ink_iou64_p01=float(np.quantile(iou[k], 0.01)),
                                  ink_iou64_min=float(iou[k].min()))
    res = dict(n_units=int(n), unit_ids_sha256=C.sha256_bytes("\n".join(map(str, z["unit_ids"][:n])).encode()),
               img224_sha256=C.sha256_array(np.asarray(X)), identical64=int((d == 0).sum()), ink_iou64_mean=float(iou.mean()),
               ink_iou64_min=float(iou.min()), per_script=per_script)
    np.save(out.with_name(out.name[:-4] + "__unit_ids.npy"), z["unit_ids"][:n])
    C.write_meta(out, kind="fmdx_render224_seqset", inputs={"e0_npz": npz},
                 params=dict(canvas=224, scale_from_64=C.SCALE_224, thickness=C.THICK224, antialias=True, renderer="uhwr_eval.render_image (fmd_harness render.render224)",
                             polarity="black ink (0) on white (255)", ink_source="e0_adapter.decode(seq) (npz has no gt ink)", order="npz order", max_units=max_units),
                 extra=dict(result=res, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
    print(json.dumps({k: v for k, v in res.items()}, default=str))


def gtset(npz, out, check64=True):
    z = np.load(npz, allow_pickle=False)
    n = len(z["unit_ids"])
    imgs, same, mx = [], 0, 0
    for i in range(n):
        ink = C.unpack_ink(z["gt_points"], z["gt_stroke_id"], z["gt_offsets"], i) if hasattr(C, "unpack_ink") else A.unpack_ink(z["gt_points"], z["gt_stroke_id"], z["gt_offsets"], i)
        imgs.append(RH.render224(ink))
        if check64:
            dd = int(np.abs(RH.render64(ink).astype(int) - z["cond_rendered"][i].astype(int)).max())
            same += int(dd == 0); mx = max(mx, dd)
    img = np.stack(imgs).astype(np.uint8)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, unit_ids=z["unit_ids"], img224=img, cond64=z["cond_rendered"])
    np.save(out.with_suffix(".npy"), img)
    res = dict(n_units=n, check64_identical=same if check64 else None, check64_max_abs_diff=mx if check64 else None,
               unit_ids_sha256=C.sha256_bytes("\n".join(map(str, z["unit_ids"])).encode()), img224_sha256=C.sha256_array(img))
    params = dict(canvas=224, scale_from_64=C.SCALE_224, thickness=C.THICK224, antialias=True, renderer="uhwr_eval.render_image (fmd_harness render.render224)",
                  polarity="black ink (0) on white (255)", ink_source="gt ink (gt_points/gt_stroke_id/gt_offsets)", order="npz order")
    extra = dict(result=res, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__)))
    C.write_meta(out, kind="fmdx_render224_gtset", inputs={"e0_npz": npz}, params=params, extra=extra)
    C.write_meta(out.with_suffix(".npy"), kind="fmdx_render224_gtset_npy", inputs={"e0_npz": npz, "render_npz": out}, params=params, extra=extra)
    print(json.dumps(res))


def npz2npy(render_npz, order_npz, out):
    R = np.load(render_npz, allow_pickle=False); Z = np.load(order_npz, allow_pickle=False)
    ru, zu = R["unit_ids"].astype(str), Z["unit_ids"].astype(str)
    if not np.array_equal(ru, zu):
        raise SystemExit("render npz unit order differs from the E0 npz")
    if "cond_rendered" in Z.files and "cond64" in R.files and not np.array_equal(R["cond64"], Z["cond_rendered"]):
        raise SystemExit("cond64 differs from cond_rendered")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, R["img224"].astype(np.uint8))
    res = dict(n_units=int(len(ru)), img224_sha256=C.sha256_array(R["img224"]), unit_order_equal=True)
    C.write_meta(out, kind="fmdx_render224_npy", inputs={"render_npz": render_npz, "order_npz": order_npz},
                 params=dict(note="img224 copied unchanged from the harness render npz"),
                 extra=dict(result=res, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("seqset", "gtset", "npz2npy"))
    ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--render-npz"); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--max-units", type=int)
    a = ap.parse_args()
    if a.kind == "seqset":
        seqset(a.npz, a.out, a.procs, a.max_units)
    elif a.kind == "gtset":
        gtset(a.npz, a.out)
    else:
        npz2npy(a.render_npz, a.npz, a.out)


if __name__ == "__main__":
    main()
