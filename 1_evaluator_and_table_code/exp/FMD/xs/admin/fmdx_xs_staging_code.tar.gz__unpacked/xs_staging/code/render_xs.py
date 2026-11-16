#!/usr/bin/env python3
"""XS staging: 224-px renders of the cross-script screen sets with fmd_harness.render (f=3.5, thickness 7), 64-px byte-identity check
against each source file's stored cond_rendered, per-dataset meta provenance (fmd_harness.common.write_meta).  Track XS staging (<agent>)."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "$U/exp/FMD/code/fmd_harness_v1")
from fmd_harness import common as C, render as RN
import uhwr_eval
R = Path("$U"); M5 = R/"data/m5_multiscript_v1"; OUT = Path(R/"exp/FMD/xs/data"); OUT.mkdir(parents=True, exist_ok=True)
RET = R/"exp/CAND/method_round/Retention/units"
SETS = [("casia_zh", M5/"m5_val_casia_zh.npz", "val250", RET/"units_val250_casia_zh.txt"),
        ("casia_en", M5/"m5_val_casia_en.npz", "val250", RET/"units_val250_casia_en.txt"),
        ("hpl_tamil", M5/"m5_val_hpl_tamil.npz", "val250", RET/"units_val250_hpl_tamil.txt"),
        ("iam", M5/"m5_val_iam.npz", "val250", RET/"units_val250_iam.txt"),
        ("unipen_de", R/"data/d2_v2/unipen_de/e0_train.npz", "train_standin_all", RET/"units_unipen_de_train.txt")]
summary = {}
for name, src, role, unitlist in SETS:
    z = np.load(src, allow_pickle=False); Z = {k: z[k] for k in z.files}
    n = len(Z["unit_ids"]); pos = np.arange(n)
    imgs, same, mx = [], 0, 0
    for p in pos:
        ink = C.unpack_ink(Z["gt_points"], Z["gt_stroke_id"], Z["gt_offsets"], int(p))
        imgs.append(RN.render224(ink))
        d = int(np.abs(RN.render64(ink).astype(int) - Z["cond_rendered"][p].astype(int)).max())
        same += int(d == 0); mx = max(mx, d)
    img = np.stack(imgs).astype(np.uint8)
    # determinism: second render of 20 units
    det = all((RN.render224(C.unpack_ink(Z["gt_points"], Z["gt_stroke_id"], Z["gt_offsets"], int(p))) == img[p]).all() for p in pos[:20])
    out = OUT/f"xs_render224_{name}_{role}.npz"
    np.savez_compressed(out, unit_ids=Z["unit_ids"], sample_ids=Z["sample_ids"], writer_ids=Z["writer_ids"], positions=pos.astype(np.int64), img224=img, cond64=Z["cond_rendered"])
    ul = [l.strip() for l in open(unitlist) if l.strip()]
    uid = set(map(str, Z["unit_ids"]))
    info = dict(set=name, role=role, n_units=int(n), n_writers=int(len(np.unique(Z["writer_ids"]))), n_sample_id_groups=int(len(np.unique(Z["sample_ids"]))), writer_key_note="n_writers = unique writer_ids (cluster key for the writer bootstrap); sample_ids = writer key (M5 sets) or source file (unipen_de)", n_identical_64=same, n_not_identical_64=n-same, max_abs_diff_64=mx,
                deterministic_rerender_first20=bool(det), img224_shape=list(img.shape), img224_sha256=C.sha256_array(img),
                unit_ids_sha256=C.sha256_bytes("\n".join(map(str, Z["unit_ids"])).encode()),
                retention_unitlist=str(unitlist), retention_unitlist_sha256=C.sha256_file(unitlist), retention_units_n=len(ul), retention_units_found_in_source=int(sum(u in uid for u in ul)),
                uhwr_eval_version=uhwr_eval.__version__, uhwr_eval_file=uhwr_eval.__file__)
    C.write_meta(out, kind="xs_render224", inputs={"source_npz": src, "retention_unit_list": unitlist},
                 params=dict(canvas=224, scale_from_64=3.5, thickness=7, antialias=True, renderer="fmd_harness.render.render224 -> uhwr_eval.render_image",
                             polarity="black ink (0) on white (255)", unit_selection=("all 250 units of the M5 val250 set (seed 20261016 sample, see data/m5_multiscript_v1/DATA_CARD.md)" if role=="val250" else "ALL 3,927 units of d2_v2/unipen_de/e0_train.npz (writer-disjoint train partition used as val stand-in; German is zero-shot, no val split; Retention track sampled 234 of these)"),
                             check64="render64(ink) vs stored cond_rendered, max abs diff and count identical"), extra=dict(result=info))
    summary[name] = info
    print(name, {k: info[k] for k in ["n_units","n_writers","n_identical_64","max_abs_diff_64","deterministic_rerender_first20","retention_units_found_in_source","img224_sha256"]}, flush=True)
json.dump(summary, open(OUT/"xs_render224_summary.json", "w"), indent=1)
