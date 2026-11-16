"""FMDX backbone training renders from rebuilt GT ink (prereg A4.2) and the InkSight pseudo-label image pool (A4.2).

luo   For every m5_luo_train.npz unit, rebuild its 64-px eval-canvas ink from the source with the M5 build transforms
      (exp/M5/code_e33511dd/m5_build.py, imported unchanged): casia_zh/casia_en = CASIA .pot item k of writer file w
      (unit id '<script>|<w>_<k:05d>') -> encode_raw (clean_ink -> fit_transform_unit 64/4); hpl_tamil = HPL UNIPEN file
      usr_<u>/<f>.txt -> encode_raw; iam = gt ink stored in data/d2_v2/iam/e0_train.npz (unit id 'iam|<id>').
      KEEP the rebuilt ink where render64(ink) == stored cond_rendered byte for byte; otherwise the unit's render comes
      from the decoded seq (e0_adapter.decode, rendered in the same pass). Output: luo_train_render224.npy (N,224,224) uint8 in npz order
      (the trainer's render array) + __source.npy (1 = rebuilt GT ink, 0 = decoded seq) + meta with per-script counts.
pool  Unlabelled LTR images for the InkSight pseudo-label arms: per source, the TRAIN pool units that are not in
      m5_luo_train and belong to train writers (CASIA: files of non-val writers, script by label rule; Tamil: non-val
      users; IAM: data/d2_v2/iam/e0_train.npz), sorted by unit id, 20,000 drawn with a fresh
      numpy default_rng(20261103) per source (.choice(n, 20000, replace=False), sorted). Ink rebuilt as above; output
      pseudo_pool_<source>.npz (unit_ids, writer keys, img224 render, cond64 render, gt arrays kept ONLY for audit and
      never read by the trainer) + meta. GT trajectories are discarded for training (A4.2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from fmd_harness import common as C
from fmd_harness import render as RH

M5C = C.ROOT / "exp/M5/code_e33511dd"
sys.path.insert(0, str(M5C))
import m5_build as MB          # noqa: E402  (unchanged M5 build code: encode_raw, readers, constants)
import e0_adapter as A         # noqa: E402

DATA = C.ROOT / "data"
CAS_DIR = DATA / "casia/CASIA/train"
POT_READER = C.ROOT / "exp/D2/ext/read_pot.py"
TAM_DIR = DATA / "hpl_tamil/extracted/hpl-tamil-iwfhr06-train-online-0.2/new"
TAM_READER = DATA / "hpl_tamil/read_unipen.py"
IAM_TRAIN = DATA / "d2_v2/iam/e0_train.npz"
STATS = DATA / "m5_multiscript_v1/build_stats_M5.json"
POOL_SEED, POOL_N = 20261103, 20000


def _init():
    MB._casia_init(str(POT_READER)); MB._tamil_init(str(TAM_READER))


def casia_file(item):
    """(path, {k: unit_id}) -> [(unit_id, ink or None, err)]; item None = scan only."""
    path, wanted = item
    out = []
    for k, (lab, strokes) in enumerate(MB.RP.iter_pot(path)):
        if k in wanted:
            try:
                out.append((wanted[k], MB.encode_raw(strokes)["ink"], ""))
            except Exception as e:  # noqa: BLE001
                out.append((wanted[k], None, repr(e)))
    return out


def casia_scan(path):
    return MB.casia_scan(path)


def tamil_file(item):
    path, uid = item
    try:
        _, strokes = MB.RU.read_unipen(path, label=-1)
        return [(uid, MB.encode_raw([s[:, :2] for s in strokes])["ink"], "")]
    except Exception as e:  # noqa: BLE001
        return [(uid, None, repr(e))]


def render_pair(ink):
    return RH.render224(ink), RH.render64(ink)


def rebuild(units, procs):
    """units: list of unit ids '<script>|...'. Returns {uid: ink or None}, {uid: err}."""
    ink, err = {}, {}
    cas = {}
    tam = []
    iam = []
    for u in units:
        s, rest = u.split("|", 1)
        if s in ("casia_zh", "casia_en"):
            w, k = rest.rsplit("_", 1)
            cas.setdefault(str(CAS_DIR / f"{w}.pot"), {})[int(k)] = u
        elif s == "hpl_tamil":
            usr, fn = rest.split("_", 2)[0] + "_" + rest.split("_", 2)[1], rest.split("_", 2)[2]
            tam.append((str(TAM_DIR / usr / f"{fn}.txt"), u))
        elif s == "iam":
            iam.append(u)
        else:
            raise ValueError(u)
    with get_context("spawn").Pool(procs, initializer=_init) as pool:
        for res in pool.imap_unordered(casia_file, list(cas.items()), chunksize=1):
            for u, i, e in res:
                ink[u] = i; err[u] = e
        for res in pool.imap_unordered(tamil_file, tam, chunksize=64):
            for u, i, e in res:
                ink[u] = i; err[u] = e
    if iam:
        z = dict(np.load(IAM_TRAIN, allow_pickle=False))
        pos = {f"iam|{x}": j for j, x in enumerate(z["unit_ids"].astype(str))}
        for u in iam:
            j = pos.get(u)
            if j is None:
                ink[u] = None; err[u] = "not in d2_v2/iam/e0_train.npz"; continue
            ink[u] = A.unpack_ink(z["gt_points"], z["gt_stroke_id"], z["gt_offsets"], j); err[u] = ""
    return ink, err


def _rend(args):
    u, i = args
    if i is None:
        return u, None, None
    a, b = render_pair(i)
    return u, a, b


_S = {}


def _init_seq(npz):
    z = np.load(npz, allow_pickle=False)
    _S.update(seq=z["seq"], off=z["seq_offsets"], cond=z["cond_rendered"])


def _rend_luo(args):
    """(k, ink) -> (k, img224, source 1|0, maxabs64 of the rebuilt ink or -1). Fallback = decoded seq."""
    k, i = args
    d = -1
    if i is not None:
        a, b = render_pair(i)
        d = int(np.abs(b.astype(int) - _S["cond"][k].astype(int)).max())
        if d == 0:
            return k, a, 1, d
    s = _S["seq"][_S["off"][k]:_S["off"][k + 1]]
    return k, RH.render224(A.decode(s)), 0, d


def luo(npz, out, procs):
    z = np.load(npz, allow_pickle=False)
    uids = z["unit_ids"].astype(str); sc = z["scripts"].astype(str)
    ink, err = rebuild(list(uids), procs)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name[:-4] + ".part.npy")
    M = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(uids), 224, 224))
    src = np.zeros(len(uids), np.int8); maxd = np.full(len(uids), -1, np.int32)
    with get_context("fork").Pool(procs, initializer=_init_seq, initargs=(str(npz),)) as pool:
        for k, im, s_, d in pool.imap_unordered(_rend_luo, [(k, ink[u]) for k, u in enumerate(uids)], chunksize=64):
            M[k] = im; src[k] = s_; maxd[k] = d
    M.flush(); del M
    os.replace(tmp, out)
    np.save(out.with_name(out.name[:-4] + "__unit_ids.npy"), z["unit_ids"])
    np.save(out.with_name(out.name[:-4] + "__source.npy"), src)
    per = {}
    for s in np.unique(sc):
        k = sc == s
        per[str(s)] = dict(n=int(k.sum()), rebuilt_gt_identical64=int((src[k] == 1).sum()), decoded_seq=int((src[k] == 0).sum()),
                           rebuild_failed=int(sum(1 for u in uids[k] if ink[u] is None)),
                           rebuilt_not_identical=int((maxd[k] > 0).sum()), maxabs64_of_rebuilt_max=int(maxd[k].max()))
    X = np.load(out, mmap_mode="r")
    res = dict(n_units=int(len(uids)), rebuilt_gt_identical64=int(src.sum()), decoded_seq=int((src == 0).sum()), per_script=per,
               img224_sha256=C.sha256_array(np.asarray(X)), source_sha256=C.sha256_array(src),
               unit_ids_sha256=C.sha256_bytes("\n".join(uids).encode()),
               first_errors=[(u, e) for u, e in err.items() if e][:10])
    C.write_meta(out, kind="fmdx_render224_luo_train", inputs={"e0_npz": npz, "m5_build": M5C / "m5_build.py",
                 "pot_reader": POT_READER, "tamil_reader": TAM_READER, "iam_train_npz": IAM_TRAIN, "build_stats": STATS},
                 params=dict(rule="prereg A4.2: rebuilt GT ink kept where render64(ink) == cond_rendered byte for byte, else the decoded seq (e0_adapter.decode)",
                             canvas=224, thickness=C.THICK224, renderer="fmd_harness render.render224", polarity="black ink on white", order="npz order",
                             source_array="__source.npy: 1 = rebuilt GT ink, 0 = decoded seq"),
                 extra=dict(result=res, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
    print(json.dumps({k: v for k, v in res.items() if k != "first_errors"}))
    print("first_errors", res["first_errors"])


def pool(out_dir, procs, n=POOL_N):
    st = json.loads(STATS.read_text())
    luo_ids = set(np.load(C.ROOT / "data/m5_multiscript_v1/m5_luo_train.npz", allow_pickle=False)["unit_ids"].astype(str))
    val_w, val_u = set(st["casia_val_writers"]), set(st["tamil_val_users"])
    files = sorted(CAS_DIR.glob("*.pot"), key=lambda p: int(p.stem))
    with get_context("spawn").Pool(procs, initializer=_init) as pl:
        scan = dict(pl.map(casia_scan, [str(f) for f in files], chunksize=1))
    pools = {"casia_zh": [], "casia_en": []}
    for f in files:
        w = f.stem
        if w in val_w:
            continue
        for s, k, _ in scan[str(f)]:
            u = f"{s}|{w}_{k:05d}"
            if u not in luo_ids:
                pools[s].append(u)
    users = sorted(d for d in os.listdir(TAM_DIR) if d.startswith("usr_") and d not in val_u)
    pools["hpl_tamil"] = [f"hpl_tamil|{u}_{p.stem}" for u in users for p in sorted((TAM_DIR / u).glob("*.txt"))
                          if f"hpl_tamil|{u}_{p.stem}" not in luo_ids]
    iz = np.load(IAM_TRAIN, allow_pickle=False)
    pools["iam"] = [f"iam|{x}" for x in iz["unit_ids"].astype(str) if f"iam|{x}" not in luo_ids]
    summary = {}
    for s in ("casia_zh", "casia_en", "hpl_tamil", "iam"):
        P = sorted(pools[s])
        rng = np.random.default_rng(POOL_SEED)
        pick = np.sort(rng.choice(len(P), min(n, len(P)), replace=False))
        sel = [P[j] for j in pick]
        ink, err = rebuild(sel, procs)
        keep = [u for u in sel if ink[u] is not None]
        with get_context("fork").Pool(procs) as pl:
            rr = dict((u, (a, b)) for u, a, b in pl.imap(_rend, [(u, ink[u]) for u in keep], chunksize=64))
        img = np.stack([rr[u][0] for u in keep]).astype(np.uint8); c64 = np.stack([rr[u][1] for u in keep]).astype(np.uint8)
        P_, S_, O_ = A.pack_inks([ink[u] for u in keep])
        op = Path(out_dir) / f"pseudo_pool_{s}.npz"
        np.savez_compressed(op, unit_ids=np.array(keep), img224=img, cond64=c64, audit_gt_points=P_, audit_gt_stroke_id=S_, audit_gt_offsets=O_)
        res = dict(source=s, pool_units=len(P), drawn=len(sel), rebuilt=len(keep), rebuild_failed=len(sel) - len(keep),
                   unit_ids_sha256=C.sha256_bytes("\n".join(keep).encode()), img224_sha256=C.sha256_array(img))
        C.write_meta(op, kind="fmdx_pseudo_pool", inputs={"m5_luo_train": C.ROOT / "data/m5_multiscript_v1/m5_luo_train.npz", "build_stats": STATS,
                     "m5_build": M5C / "m5_build.py"}, params=dict(rule="prereg A4.2 InkSight pseudo-label pool", seed=POOL_SEED, n=n,
                     draw="fresh numpy default_rng(20261103) per source; choice(len(sorted pool), n, replace=False), sorted",
                     note="audit_gt_* arrays are kept for audit only; the pseudo-label arm never reads them"),
                     extra=dict(result=res, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
        summary[s] = res
        print(json.dumps(res), flush=True)
    C.write_json_atomic(Path(out_dir) / "pseudo_pool_summary.json", summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("luo", "pool"))
    ap.add_argument("--npz"); ap.add_argument("--decoded-npy"); ap.add_argument("--out"); ap.add_argument("--out-dir")
    ap.add_argument("--procs", type=int, default=16); ap.add_argument("--n", type=int, default=POOL_N)
    a = ap.parse_args()
    if a.kind == "luo":
        luo(a.npz, a.out, a.procs)
    else:
        pool(a.out_dir, a.procs, a.n)


if __name__ == "__main__":
    main()
