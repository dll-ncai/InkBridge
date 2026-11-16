"""FMDX InkSight pseudo-label data (prereg A4.2, arms INKSIGHT-WARM / INKSIGHT-SCRATCH).

manifest  pseudo_pool_<source>.npz (prep_gt_ink.py pool) -> one PNG per unit of the 224-px render (black ink on white,
          RGB) under <dir>/png/<source>/<k:05d>.png + manifest.json [{id, png}] (id = '<source>|<row>') in source order.
build     InkSight outputs (out_*.jsonl of the unchanged runner, strokes in render pixels: the 224 render is 224x224, so
          scale_and_pad uses ratio 1 and no padding) -> strokes / 3.5 = 64-px eval-canvas ink -> KEEP if the re-render
          (project renderer uhwr_eval.render_image 64 px, thickness 2, anti-aliased = e0_adapter.render_condition) has
          IoU >= 0.80 with the binarised input render at the same 64-px geometry (the unit's cond64 = render64 of its GT
          ink, the 224 input / 3.5; ink = pixel < 128). Kept units ->
          E0-format pseudo_train.npz (unit_ids '<source>|<unit id>', scripts = source, writer_ids, cond_rendered = cond64,
          seq = e0_adapter.encode(pseudo ink), seq_offsets, step) + pseudo_train_render224.npy (the 224 input render, same
          order) + meta with per-source counts (drawn, InkSight output, empty, kept). No Urdu and no German unit enters.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from fmd_harness import common as C
import e0_adapter as A

SOURCES = ("casia_zh", "casia_en", "hpl_tamil", "iam")
IOU_MIN = 0.80


def manifest(pool_dir, out_dir):
    from PIL import Image
    out_dir = Path(out_dir); man = []
    for s in SOURCES:
        z = np.load(Path(pool_dir) / f"pseudo_pool_{s}.npz", allow_pickle=False)
        d = out_dir / "png" / s; d.mkdir(parents=True, exist_ok=True)
        for k, im in enumerate(z["img224"]):
            p = d / f"{k:05d}.png"
            if not p.exists():
                Image.fromarray(np.stack([im] * 3, -1)).save(p)
            man.append(dict(id=f"{s}|{k}", png=str(p)))
    json.dump(man, open(out_dir / "manifest.json", "w"))
    C.write_meta(out_dir / "manifest.json", kind="fmdx_pseudo_manifest", inputs={f"pool_{s}": Path(pool_dir) / f"pseudo_pool_{s}.npz" for s in SOURCES},
                 params=dict(png="224-px render, RGB replicate, black ink on white", id="<source>|<row in pseudo_pool_<source>.npz>"),
                 extra=dict(result=dict(n=len(man)), script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
    print(len(man))


def build(pool_dir, out_glob, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for f in sorted(glob.glob(out_glob)):
        for l in open(f):
            r = json.loads(l); res[r["id"]] = r
    recs, imgs, counts = [], [], {}
    for s in SOURCES:
        z = np.load(Path(pool_dir) / f"pseudo_pool_{s}.npz", allow_pickle=False)
        c = dict(drawn=int(len(z["unit_ids"])), inksight_output=0, empty=0, encode_error=0, iou_below=0, kept=0)
        for k, u in enumerate(z["unit_ids"].astype(str)):
            r = res.get(f"{s}|{k}")
            if r is None:
                continue
            c["inksight_output"] += 1
            ink = [np.asarray(st, np.float64) / C.SCALE_224 for st in r["strokes"] if len(st)]
            if not ink:
                c["empty"] += 1; continue
            rr = A.render_condition(ink)
            a, b = rr < 128, z["cond64"][k] < 128
            iou = float((a & b).sum() / max(1, (a | b).sum()))
            if iou < IOU_MIN:
                c["iou_below"] += 1; continue
            try:
                seq, _, step = A.encode(ink)
            except Exception:
                c["encode_error"] += 1; continue
            recs.append(dict(uid=u if u.startswith(s + "|") else f"{s}|{u}", script=s, cond=z["cond64"][k], seq=seq.astype(np.float32), step=step, iou=iou))
            imgs.append(z["img224"][k]); c["kept"] += 1
        counts[s] = c
    seqs = [r["seq"] for r in recs]
    np.savez(out_dir / "pseudo_train.npz", unit_ids=np.array([r["uid"] for r in recs]), sample_ids=np.array([r["script"] for r in recs]),
             writer_ids=np.zeros(len(recs), np.int64), scripts=np.array([r["script"] for r in recs]),
             cond_rendered=np.stack([r["cond"] for r in recs]).astype(np.uint8), seq=np.vstack(seqs).astype(np.float32),
             seq_offsets=np.r_[0, np.cumsum([len(x) for x in seqs])].astype(np.int64), step=np.array([r["step"] for r in recs], np.float32),
             iou=np.array([r["iou"] for r in recs], np.float32))
    np.save(out_dir / "pseudo_train_render224.npy", np.stack(imgs).astype(np.uint8))
    info = dict(counts=counts, n_kept=len(recs), iou_min=IOU_MIN, n_inksight_records=len(res))
    for name in ("pseudo_train.npz", "pseudo_train_render224.npy"):
        C.write_meta(out_dir / name, kind="fmdx_pseudo_train", inputs={**{f"pool_{s}": Path(pool_dir) / f"pseudo_pool_{s}.npz" for s in SOURCES}},
                     params=dict(rule="prereg A4.2: keep if IoU(render64 thickness 2 of the InkSight ink / 3.5, binarised cond64) >= 0.80",
                                 inksight_outputs=out_glob, model="InkSight Small-p (hf/InkSight-Small-p rev 8514f237), runner prompt mode default"),
                     extra=dict(result=info, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
    print(json.dumps(info))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("manifest", "build"))
    ap.add_argument("--pool-dir", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--out-glob")
    a = ap.parse_args()
    manifest(a.pool_dir, a.out_dir) if a.kind == "manifest" else build(a.pool_dir, a.out_glob, a.out_dir)


if __name__ == "__main__":
    main()
