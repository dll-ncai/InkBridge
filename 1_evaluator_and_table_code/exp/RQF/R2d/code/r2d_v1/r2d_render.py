#!/usr/bin/env python3
"""RQF R2d 224-px training renders, FMDX prereg A4.2 rule (exp/FMD/xs/code/fmdx_backbone_v1/prep_gt_ink.py, imported unchanged):
rebuilt GT ink where render64(ink) == stored cond_rendered byte for byte, else the decoded seq.
  lto       m5_lto_train.npz: casia_zh / casia_en / iam rows are COPIED from luo_train_render224.npy (+ __source.npy) by unit id
            (these units are byte-identical in both npz files, checked again here on cond_rendered); ouhdl rows: GT ink from
            data/ouhdl_e0_v3/e0_train.npz (unit 'ouhdl|<id>') through prep_gt_ink._rend_luo (same rule, same renderer).
  tamilp01  tamil_p01.npz through prep_gt_ink.luo (Tamil raw UNIPEN files -> m5_build.encode_raw ink).
Run with PYTHONPATH = FMDX backbone path (fmd_harness + fmdx_backbone_v1 + e0 code)."""
import argparse, json, os, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
sys.path.insert(0, "$U/exp/FMD/xs/code/fmdx_backbone_v1")
import prep_gt_ink as PG
C, A = PG.C, PG.A
R = Path("$U"); DB = R / "exp/FMD/xs/data/backbone"; OUTR = R / "data/ouhdl_e0_v3/e0_train.npz"

def lto(npz, out, procs):
    z = np.load(npz, allow_pickle=False); uids = z["unit_ids"].astype(str); sc = z["scripts"].astype(str); cond = z["cond_rendered"]
    lu = np.load(DB / "luo_train_render224__unit_ids.npy").astype(str); L = np.load(DB / "luo_train_render224.npy", mmap_mode="r")
    Ls = np.load(DB / "luo_train_render224__source.npy"); pos = {u: j for j, u in enumerate(lu)}
    Z = np.load(R / "data/m5_multiscript_v1/m5_luo_train.npz", allow_pickle=False); Lc = Z["cond_rendered"]
    out = Path(out); tmp = out.with_name(out.name[:-4] + ".part.npy")
    M = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(uids), 224, 224))
    src = np.zeros(len(uids), np.int8); maxd = np.full(len(uids), -1, np.int32); copied = cond_eq = 0
    ou = []
    for k, (u, s) in enumerate(zip(uids, sc)):
        if s == "ouhdl": ou.append(k); continue
        j = pos[u]; M[k] = L[j]; src[k] = Ls[j]; copied += 1; cond_eq += int(np.array_equal(cond[k], Lc[j]))
    assert cond_eq == copied, (cond_eq, copied)
    E = dict(np.load(OUTR, allow_pickle=False)); has_gt = "gt_points" in E; ep = {f"ouhdl|{x}": j for j, x in enumerate(E["unit_ids"].astype(str))}
    def ink(u):
        j = ep.get(u)
        return None if (j is None or not has_gt) else A.unpack_ink(E["gt_points"], E["gt_stroke_id"], E["gt_offsets"], j)
    with get_context("fork").Pool(procs, initializer=PG._init_seq, initargs=(str(npz),)) as pool:
        for k, im, s_, d in pool.imap_unordered(PG._rend_luo, [(k, ink(uids[k])) for k in ou], chunksize=64):
            M[k] = im; src[k] = s_; maxd[k] = d
    M.flush(); del M; os.replace(tmp, out)
    np.save(out.with_name(out.name[:-4] + "__unit_ids.npy"), z["unit_ids"]); np.save(out.with_name(out.name[:-4] + "__source.npy"), src)
    X = np.load(out, mmap_mode="r"); ko = np.array(ou)
    res = dict(n_units=int(len(uids)), copied_from_luo_render=copied, copied_cond_identical=cond_eq, ouhdl_units=len(ou), ouhdl_e0_train_has_gt=has_gt,
               ouhdl_rebuilt_gt_identical64=int((src[ko] == 1).sum()), ouhdl_decoded_seq=int((src[ko] == 0).sum()),
               ouhdl_rebuilt_not_identical=int((maxd[ko] > 0).sum()), rebuilt_gt_identical64_all=int(src.sum()),
               img224_sha256=C.sha256_array(np.asarray(X)), source_sha256=C.sha256_array(src), unit_ids_sha256=C.sha256_bytes("\n".join(uids).encode()))
    C.write_meta(out, kind="rqf_r2d_render224_lto_train", inputs={"e0_npz": npz, "luo_render": DB / "luo_train_render224.npy", "luo_render_source": DB / "luo_train_render224__source.npy",
                 "luo_npz": R / "data/m5_multiscript_v1/m5_luo_train.npz", "ouhdl_e0_train": OUTR, "prep_gt_ink": Path(PG.__file__)},
                 params=dict(rule="prereg A4.2 (FMDX): rebuilt GT ink kept where render64(ink)==cond_rendered byte for byte, else decoded seq; non-Urdu rows copied from the LUO render (same units)",
                             canvas=224, thickness=C.THICK224, renderer="fmd_harness render.render224", order="npz order"),
                 extra=dict(result=res, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
    print(json.dumps(res))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=("lto", "tamilp01")); ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=16); a = ap.parse_args()
    lto(a.npz, a.out, a.procs) if a.kind == "lto" else PG.luo(a.npz, a.out, a.procs)
