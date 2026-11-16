"""M5 / X6 data build: m5_multiscript_v1 = equal-size E0-format training sets for five scripts.

python m5_build.py --data $U/data --x4a $U/exp/X4a/data \
    --out <dir> [--procs 32] [--n-train 20000] [--n-val 250] [--limit-files N]  (limit = smoke)

Per script (all E0 format: 64x64 canvas, fit_transform_unit 56 px box + 4 px margin, y down, per-stroke 1 px
arc-length resampling with the step grown x1.05 until <= 320 points (e0_adapter.encode), condition 'rendered' =
e0_adapter.render_condition = uhwr_eval.render_image 2 px anti-aliased 64x64, black on white uint8):
  casia_zh  CASIA OLHWDB1.0+1.1 train .pot (576 writers), single CJK characters (U+4E00-9FFF)
  casia_en  same files, single ASCII letters a-z A-Z (the D2 'casia_en' rule)
  hpl_tamil HPL IWFHR06 train online (new/usr_<id>/<cls>t<trial>.txt), writer = usr_<id>
            (CASIA / Tamil ink: uhwr_eval.clean_ink then fit_transform_unit, i.e. D2 UF.to_eval_space 'isolated')
  iam       data/d2_v2/iam (X3 component units, already E0 format): train = e0_train, val = e0_val1
  ouhdl     data/ouhdl_e0_v3 (the E0 arrays): train = e0_train, val = e0_val
Val writers: CASIA 24 of the 576 train writers and Tamil 5 % of the train users are held out (seeded); their units
form the val pools (writer-disjoint from the train sample and from the test sets).  IAM / OUHD-L use their
official val splits.  Train sample = n_train units uniformly at random (without replacement) from the script's
train pool; val sample = n_val units from the val pool, stored in random order (first k = a random k-subset).
Outputs (all in --out):
  m5_joint_train.npz   5 x n_train units (casia_zh, casia_en, hpl_tamil, iam, ouhdl)
  m5_luo_train.npz     4 x n_train units (ouhdl left out)
  m5_val_<script>.npz  n_val units each, with gt ink (selection)
  m5_test_iam500.npz   500 IAM testset_f units with a known writer (seed 2026), sample_ids = writer key
  m5_ft_ouhdl_p01/p10/p30.npz  nested fixed-seed subsets of OUHD-L E0 train (1 / 10 / 30 %); 100 % = e0_train.npz
  ft_subsets_M5.json, seqlen_check_M5.csv, build_stats_M5.json, members_<script>_{train,val}.csv
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

import e0_adapter as A
from uhwr_eval import apply_transform, clean_ink, fit_transform_unit, resample_ink_step

SEED = 20261016
SCRIPTS = ["casia_zh", "casia_en", "hpl_tamil", "iam", "ouhdl"]
LUO = ["casia_zh", "casia_en", "hpl_tamil", "iam"]
N_CASIA_VAL_WRITERS = 24
TAMIL_VAL_FRAC = 0.05
FT_FRACS = {"p01": 0.01, "p10": 0.10, "p30": 0.30}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _imp(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def is_cjk(lab):
    return len(lab) == 1 and 0x4E00 <= ord(lab) <= 0x9FFF


def is_letter(lab):
    return len(lab) == 1 and lab.isascii() and lab.isalpha()


def n_points_1px(ink):
    return int(sum(len(s) for s in resample_ink_step(ink, 1.0)))


def encode_raw(strokes):
    """raw device strokes -> dict(ink, cond, seq, step, n1px, n_strokes) or raises."""
    st = [np.asarray(s, np.float64)[:, :2] for s in strokes if len(s)]
    ink = clean_ink(st)
    if not ink:
        raise ValueError("empty ink")
    T = fit_transform_unit(ink, 64, 4)
    ink = apply_transform(ink, *T)
    seq, _, step = A.encode(ink)
    return dict(ink=ink, cond=A.render_condition(ink), seq=seq.astype(np.float32), step=float(step),
                n1px=n_points_1px(ink), n_strokes=len(ink))


# ------------------------------------------------------------------ CASIA (.pot)
RP = None


def _casia_init(reader):
    global RP
    RP = _imp(reader, "read_pot")


def casia_scan(path):
    out = []
    for k, (lab, strokes) in enumerate(RP.iter_pot(path)):
        if is_cjk(lab):
            out.append(("casia_zh", k, lab))
        elif is_letter(lab):
            out.append(("casia_en", k, lab))
    return path, out


def casia_extract(item):
    path, wanted = item          # wanted: {k: (script, role)}
    res = []
    for k, (lab, strokes) in enumerate(RP.iter_pot(path)):
        if k in wanted:
            script, role = wanted[k]
            w = os.path.basename(path)[:-4]
            try:
                e = encode_raw(strokes); err = ""
            except Exception as ex:   # noqa: BLE001
                e, err = None, f"{type(ex).__name__}: {ex}"
            res.append(dict(script=script, role=role, unit_id=f"{script}|{w}_{k:05d}", writer=f"{script}:{w}",
                            label=lab, enc=e, error=err))
    return res


# ------------------------------------------------------------------ Tamil (UNIPEN)
RU = None


def _tamil_init(reader):
    global RU
    RU = _imp(reader, "read_unipen")


def tamil_extract(item):
    path, role = item
    usr = os.path.basename(os.path.dirname(path)); fn = os.path.basename(path)[:-4]
    try:
        _, strokes = RU.read_unipen(path, label=-1)
        e = encode_raw([s[:, :2] for s in strokes]); err = ""
    except Exception as ex:   # noqa: BLE001
        e, err = None, f"{type(ex).__name__}: {ex}"
    cls = fn.split("t")[0]
    return [dict(script="hpl_tamil", role=role, unit_id=f"hpl_tamil|{usr}_{fn}", writer=f"hpl_tamil:{usr}",
                 label=cls, enc=e, error=err)]


# ------------------------------------------------------------------ E0-format arrays (IAM, OUHD-L)
def from_e0_npz(path, idx, script, role, writer_key, keep_gt):
    z = dict(np.load(path, allow_pickle=False))     # load every array once (NpzFile re-reads on each access)
    off = z["seq_offsets"]; has_gt = "gt_points" in z
    if has_gt:
        P, S, G = z["gt_points"], z["gt_stroke_id"], z["gt_offsets"]
    recs = []
    for i in idx:
        i = int(i)
        seq = z["seq"][off[i]:off[i + 1]]
        step = float(z["step"][i])
        if has_gt:
            ink = A.unpack_ink(P, S, G, i); n1, n1_exact = n_points_1px(ink), True
        else:
            ink = None
            if step <= 1.0 + 1e-6:
                n1, n1_exact = int(len(seq)), True
            else:   # resampled at step > 1: estimate the 1 px count from the decoded sequence
                n1, n1_exact = n_points_1px(A.decode(seq)), False
        n_str = int((seq[:, 2] < 0).sum())
        recs.append(dict(script=script, role=role, unit_id=f"{script}|{z['unit_ids'][i]}",
                         writer=f"{script}:{writer_key(z, i)}", label="",
                         enc=dict(ink=ink if keep_gt else None, cond=z["cond_rendered"][i], seq=seq.astype(np.float32),
                                  step=step, n1px=n1, n1px_exact=n1_exact, n_strokes=n_str), error=""))
    return recs


# ------------------------------------------------------------------ writers / packing
def pack(recs, with_gt):
    recs = [r for r in recs if r["enc"] is not None]
    wkeys = sorted({r["writer"] for r in recs}); wmap = {w: k for k, w in enumerate(wkeys)}
    seqs = [r["enc"]["seq"] for r in recs]
    d = dict(unit_ids=np.array([r["unit_id"] for r in recs]),
             sample_ids=np.array([r["writer"] for r in recs]),        # writer key -> e0_score clusters = writers
             writer_ids=np.array([wmap[r["writer"]] for r in recs], np.int64),
             scripts=np.array([r["script"] for r in recs]),
             cond_rendered=np.stack([r["enc"]["cond"] for r in recs]).astype(np.uint8),
             seq=np.vstack(seqs).astype(np.float32),
             seq_offsets=np.r_[0, np.cumsum([len(s) for s in seqs])].astype(np.int64),
             step=np.array([r["enc"]["step"] for r in recs], np.float32))
    if with_gt:
        P, S, off = A.pack_inks([r["enc"]["ink"] for r in recs])
        d.update(gt_points=P, gt_stroke_id=S, gt_offsets=off)
    return d


def pct(v, qs=(50, 90, 99, 100)):
    v = np.asarray(v, float)
    return {str(q): float(np.percentile(v, q)) for q in qs} if len(v) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--x4a", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=32); ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=250); ap.add_argument("--limit-files", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); stats = dict(params=vars(a), seed=SEED, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    inputs = {}
    recs = {s: {"train": [], "val": []} for s in SCRIPTS}
    pools = {}

    # ---------------- CASIA
    cas_dir = os.path.join(a.data, "casia/CASIA/train")
    reader = os.path.join(os.path.dirname(a.x4a), "..", "D2", "ext", "read_pot.py")
    reader = os.path.normpath(reader)
    files = sorted(glob.glob(os.path.join(cas_dir, "*.pot")), key=lambda p: int(os.path.basename(p)[:-4]))
    if a.limit_files:
        files = files[:a.limit_files]
    inputs["casia_reader"] = dict(path=reader, sha256=sha(reader))
    inputs["casia_train_files"] = dict(dir=cas_dir, n_files=len(files),
                                       sha256_of_sorted_file_sha256s=hashlib.sha256("".join(sha(f) for f in files).encode()).hexdigest())
    writers = [os.path.basename(f)[:-4] for f in files]
    rng = np.random.default_rng(SEED)
    nvw = min(N_CASIA_VAL_WRITERS, max(1, len(writers) // 4))
    val_w = set(rng.choice(writers, nvw, replace=False).tolist())
    with get_context("spawn").Pool(a.procs, initializer=_casia_init, initargs=(reader,)) as pool:
        scan = dict(pool.map(casia_scan, files, chunksize=1))
    wanted = {f: {} for f in files}
    for k_s, script in enumerate(["casia_zh", "casia_en"]):
        for role in ("train", "val"):
            pool_items = [(f, k) for f in files for (s, k, _) in scan[f] if s == script
                          and ((os.path.basename(f)[:-4] in val_w) == (role == "val"))]
            n = a.n_train if role == "train" else a.n_val
            r = np.random.default_rng([SEED, 1 + k_s, 0 if role == "train" else 1])
            pick = r.choice(len(pool_items), min(n, len(pool_items)), replace=False)
            pools[f"{script}_{role}"] = dict(pool_units=len(pool_items), picked=int(len(pick)),
                                             pool_writers=len({f for f, _ in pool_items}))
            for j in pick:
                f, k = pool_items[j]; wanted[f][k] = (script, role)
    with get_context("spawn").Pool(a.procs, initializer=_casia_init, initargs=(reader,)) as pool:
        out = [r for rr in pool.map(casia_extract, [(f, wanted[f]) for f in files if wanted[f]], chunksize=1) for r in rr]
    for r in out:
        recs[r["script"]][r["role"]].append(r)
    stats["casia_val_writers"] = sorted(val_w, key=int)
    print("casia done", {k: len(v["train"]) for k, v in recs.items()}, round(time.time() - t0), "s", flush=True)

    # ---------------- Tamil
    tdir = os.path.join(a.data, "hpl_tamil/extracted/hpl-tamil-iwfhr06-train-online-0.2/new")
    treader = os.path.join(a.data, "hpl_tamil/read_unipen.py")
    inputs["tamil_reader"] = dict(path=treader, sha256=sha(treader))
    users = sorted(d for d in os.listdir(tdir) if d.startswith("usr_"))
    tfiles = {u: sorted(glob.glob(os.path.join(tdir, u, "*.txt"))) for u in users}
    if a.limit_files:
        users = users[:max(4, a.limit_files)]
    inputs["tamil_train_dir"] = dict(dir=tdir, n_users=len(users), n_files=sum(len(tfiles[u]) for u in users))
    r = np.random.default_rng([SEED, 3])
    tval = set(r.choice(users, max(1, int(round(TAMIL_VAL_FRAC * len(users)))), replace=False).tolist())
    items = []
    for role in ("train", "val"):
        pool_items = [p for u in users if (u in tval) == (role == "val") for p in tfiles[u]]
        n = a.n_train if role == "train" else a.n_val
        rr = np.random.default_rng([SEED, 3, 0 if role == "train" else 1])
        pick = rr.choice(len(pool_items), min(n, len(pool_items)), replace=False)
        pools[f"hpl_tamil_{role}"] = dict(pool_units=len(pool_items), picked=int(len(pick)),
                                          pool_writers=len({os.path.basename(os.path.dirname(p)) for p in pool_items}))
        items += [(pool_items[j], role) for j in pick]
    with get_context("spawn").Pool(a.procs, initializer=_tamil_init, initargs=(treader,)) as pool:
        out = [x for xx in pool.map(tamil_extract, items, chunksize=64) for x in xx]
    for x in out:
        recs["hpl_tamil"][x["role"]].append(x)
    stats["tamil_val_users"] = sorted(tval)
    print("tamil done", len(recs["hpl_tamil"]["train"]), round(time.time() - t0), "s", flush=True)

    # ---------------- IAM (X3 E0 arrays) and OUHD-L (E0 arrays)
    iam = os.path.join(a.data, "d2_v2/iam"); ou = os.path.join(a.data, "ouhdl_e0_v3")
    src = {"iam": {"train": os.path.join(iam, "e0_train.npz"), "val": os.path.join(iam, "e0_val1.npz")},
           "ouhdl": {"train": os.path.join(ou, "e0_train.npz"), "val": os.path.join(ou, "e0_val.npz")}}
    wk = {"iam": lambda z, i: (str(z["writer_ids"][i]) if int(z["writer_ids"][i]) >= 0 else f"nowriter_{z['sample_ids'][i]}"),
          "ouhdl": lambda z, i: str(z["writer_ids"][i])}
    for k_s, script in enumerate(["iam", "ouhdl"]):
        for role in ("train", "val"):
            p = src[script][role]; inputs[f"{script}_{role}_npz"] = dict(path=p, sha256=sha(p))
            n_all = len(np.load(p)["unit_ids"])
            n = a.n_train if role == "train" else a.n_val
            if a.limit_files:
                n = min(n, 200)
            r = np.random.default_rng([SEED, 4 + k_s, 0 if role == "train" else 1])
            pick = r.choice(n_all, min(n, n_all), replace=False)
            pools[f"{script}_{role}"] = dict(pool_units=int(n_all), picked=int(len(pick)))
            recs[script][role] = from_e0_npz(p, pick, script, role, wk[script], keep_gt=(role == "val"))
    print("iam/ouhdl done", round(time.time() - t0), "s", flush=True)

    # ---------------- errors, members, seq-length check
    rows, errs = [], {}
    for s in SCRIPTS:
        for role in ("train", "val"):
            L = recs[s][role]
            errs[f"{s}_{role}"] = [r["error"] for r in L if r["enc"] is None][:5] + [f"n_errors={sum(r['enc'] is None for r in L)}"]
            ok = [r for r in L if r["enc"] is not None]
            pd.DataFrame([dict(unit_id=r["unit_id"], writer=r["writer"], label=r["label"], n_strokes=r["enc"]["n_strokes"],
                               n_points_1px=r["enc"]["n1px"], n1px_exact=r["enc"].get("n1px_exact", True),
                               n_seq=len(r["enc"]["seq"]), step=r["enc"]["step"]) for r in ok]
                         ).to_csv(os.path.join(a.out, f"members_{s}_{role}.csv"), index=False)
            n1 = np.array([r["enc"]["n1px"] for r in ok]); nseq = np.array([len(r["enc"]["seq"]) for r in ok])
            stp = np.array([r["enc"]["step"] for r in ok]); nst = np.array([r["enc"]["n_strokes"] for r in ok])
            row = dict(script=s, role=role, n_units=len(ok), n_writers=len({r["writer"] for r in ok}),
                       n_errors=sum(r["enc"] is None for r in L),
                       n1px_exact_units=int(sum(r["enc"].get("n1px_exact", True) for r in ok)))
            for nm, v in (("n_points_1px", n1), ("n_seq", nseq), ("step", stp), ("n_strokes", nst)):
                for q, x in pct(v, (50, 90, 99, 100)).items():
                    row[f"{nm}_p{q}"] = x
                row[f"{nm}_mean"] = float(v.mean()) if len(v) else np.nan
            row["frac_n1px_gt_320"] = float((n1 > 320).mean()) if len(n1) else np.nan
            row["frac_step_gt_1"] = float((stp > 1 + 1e-6).mean()) if len(stp) else np.nan
            row["mean_seq_fill_of_320"] = float(nseq.mean() / 320) if len(nseq) else np.nan
            rows.append(row)
    seqlen = pd.DataFrame(rows); seqlen.to_csv(os.path.join(a.out, "seqlen_check_M5.csv"), index=False)
    print(seqlen[["script", "role", "n_units", "n_points_1px_p50", "n_points_1px_p90", "n_points_1px_p100",
                  "frac_n1px_gt_320", "n_seq_p50", "step_p90"]].to_string(), flush=True)

    # ---------------- write arrays
    files_out = {}

    def save(name, d):
        f = os.path.join(a.out, name); np.savez(f, **d)    # uncompressed: fast mmap-free loading in training
        files_out[name] = dict(sha256=sha(f), bytes=os.path.getsize(f), n_units=int(len(d["unit_ids"])))
        print("wrote", name, files_out[name], flush=True)

    save("m5_joint_train.npz", pack([r for s in SCRIPTS for r in recs[s]["train"]], with_gt=False))
    save("m5_luo_train.npz", pack([r for s in LUO for r in recs[s]["train"]], with_gt=False))
    for s in SCRIPTS:
        save(f"m5_val_{s}.npz", pack(recs[s]["val"], with_gt=True))

    # IAM test 500 (known writer), seed 2026
    pt = os.path.join(iam, "e0_test.npz"); inputs["iam_test_npz"] = dict(path=pt, sha256=sha(pt))
    zt = np.load(pt); okw = np.flatnonzero(zt["writer_ids"] >= 0)
    pick = np.sort(np.random.default_rng(2026).choice(okw, min(500, len(okw)), replace=False))
    tr = from_e0_npz(pt, pick, "iam", "test", lambda z, i: str(z["writer_ids"][i]), keep_gt=True)
    save("m5_test_iam500.npz", pack(tr, with_gt=True))

    # FT subsets of OUHD-L E0 train (nested; one permutation)
    pz = src["ouhdl"]["train"]; z = dict(np.load(pz)); N = len(z["unit_ids"])
    perm = np.random.default_rng([SEED, 9]).permutation(N)
    ft = dict(source=pz, source_sha256=inputs["ouhdl_train_npz"]["sha256"], n_source=int(N), rule="nested prefixes of one permutation, numpy default_rng([20261016, 9]).permutation(N); k = floor(frac*N + 0.5)", subsets={})
    for tag, frac in FT_FRACS.items():
        k = int(np.floor(frac * N + 0.5)); sel = np.sort(perm[:k])
        off = z["seq_offsets"]; seqs = [z["seq"][off[i]:off[i + 1]] for i in sel]
        d = dict(unit_ids=z["unit_ids"][sel], sample_ids=z["sample_ids"][sel], writer_ids=z["writer_ids"][sel],
                 cond_rendered=z["cond_rendered"][sel], seq=np.vstack(seqs), step=z["step"][sel],
                 seq_offsets=np.r_[0, np.cumsum([len(s) for s in seqs])].astype(np.int64))
        save(f"m5_ft_ouhdl_{tag}.npz", d)
        ft["subsets"][tag] = dict(frac=frac, n_units=k, n_writers=int(len(np.unique(z["writer_ids"][sel]))),
                                  n_lines=int(len(np.unique(z["sample_ids"][sel]))), file=f"m5_ft_ouhdl_{tag}.npz")
    ft["subsets"]["p100"] = dict(frac=1.0, n_units=int(N), n_writers=int(len(np.unique(z["writer_ids"]))),
                                 n_lines=int(len(np.unique(z["sample_ids"]))), file=pz)
    json.dump(ft, open(os.path.join(a.out, "ft_subsets_M5.json"), "w"), indent=1)

    stats.update(inputs=inputs, pools=pools, errors=errs, files=files_out, ft_subsets=ft["subsets"],
                 seconds=round(time.time() - t0, 1), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(stats, open(os.path.join(a.out, "build_stats_M5.json"), "w"), indent=1, default=str)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
