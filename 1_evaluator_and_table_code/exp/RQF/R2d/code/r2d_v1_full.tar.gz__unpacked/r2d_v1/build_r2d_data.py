#!/usr/bin/env python3
"""RQF R2d data splits (prereg PREREG_RQ_finalise_2026-11-07 section 2 R2d). Run with PYTHONPATH = M5 recipe path
($REPO/src:$MC:$C0) so Tamil units are encoded by the unchanged M5 build code (exp/M5/code_e33511dd/m5_build.py).
  lto    m5_lto_train.npz = m5_joint_train.npz minus every hpl_tamil unit (casia_zh, casia_en, iam, ouhdl; Urdu INCLUDED),
         stored order kept; writer_ids re-indexed exactly as m5_build.pack (sorted writer keys). Check: every non-Urdu unit
         is byte-identical (seq, cond_rendered, step) to the same unit in m5_luo_train.npz.
  tamil  tamil_p01.npz: 1 % of the HPL Tamil M5 train pool (IWFHR06 train, the 111 non-val users of build_stats_M5.json),
         writer-stratified, seed 20261107: K = floor(0.01 N + 0.5); writer quota 0.01 n_w, floors + largest remainders
         (ties broken by numpy default_rng([20261107, 1]).permutation over writers); files per writer drawn with
         default_rng([20261107, 2]).choice(n_w, q_w, replace=False) in sorted-writer order. Packed like the M5 train arrays.
         tamil_val1000.npz: 1,000 units of the M5 Tamil val pool (the 6 held-out val users, writer-disjoint from train and
         from the IWFHR06 test), default_rng([20261107, 3]).choice(n, 1000, replace=False), stored in draw order, with gt ink.
         Check: units shared with m5_joint_train (p01) / m5_val_hpl_tamil (val) are byte-identical to the M5 encodings.
Each output gets PROVENANCE_<name>.json (inputs + sha256, rule, seed, counts, checks, content sha256)."""
import argparse, glob, hashlib, json, os, platform, subprocess, sys, tarfile, time
from multiprocessing import get_context
import numpy as np
R = "$U"; D = f"{R}/data/m5_multiscript_v1"; MC = f"{R}/exp/M5/code_e33511dd"
SEED = 20261107
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def content_sha(d):
    h = hashlib.sha256()
    for k in sorted(d):
        a = np.ascontiguousarray(d[k]); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()
def utc(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def env():
    import uhwr_eval
    return dict(python=platform.python_version(), numpy=np.__version__, host=platform.node(), slurm=os.environ.get("SLURM_JOB_ID"),
                site=os.environ.get("R2D_SITE", "clusterA"), uhwr_eval_file=uhwr_eval.__file__, pythonpath=os.environ.get("PYTHONPATH"))
def save(out, name, d, prov):
    os.makedirs(out, exist_ok=True); f = os.path.join(out, name); tmp = f[:-4] + ".part.npz"
    np.savez(tmp, **d); os.replace(tmp, f)
    prov["output"] = dict(path=f, sha256=sha(f), bytes=os.path.getsize(f), content_sha256=content_sha(d),
                          unit_ids_sha256=hashlib.sha256("\n".join(map(str, d["unit_ids"])).encode()).hexdigest(),
                          n_units=int(len(d["unit_ids"])), n_writers=int(len(np.unique(d["sample_ids"]))), keys=sorted(d))
    prov.update(created_utc=utc(), env=env(), script=dict(path=os.path.abspath(__file__), sha256=sha(__file__)),
                prereg="exp/RQF/plans/PREREG_RQ_finalise_2026-11-07.md sha256 04766e973badf7ff469d468e25bab17cebabf38f2c9537d731a4d91ea960c624, section 2 R2d")
    json.dump(prov, open(os.path.join(out, f"PROVENANCE_{name[:-4]}.json"), "w"), indent=1)
    print(name, json.dumps({k: prov["output"][k] for k in ("n_units", "n_writers", "content_sha256")}), flush=True)
def sl(z, i): o = z["seq_offsets"]; return z["seq"][o[i]:o[i + 1]]

def build_lto(out):
    pj, pl = f"{D}/m5_joint_train.npz", f"{D}/m5_luo_train.npz"
    z = dict(np.load(pj, allow_pickle=False)); sc = z["scripts"].astype(str); keep = np.flatnonzero(sc != "hpl_tamil")
    seqs = [sl(z, i) for i in keep]; wr = z["sample_ids"][keep].astype(str); wk = sorted(set(wr)); wm = {w: k for k, w in enumerate(wk)}
    d = dict(unit_ids=z["unit_ids"][keep], sample_ids=z["sample_ids"][keep], writer_ids=np.array([wm[w] for w in wr], np.int64),
             scripts=z["scripts"][keep], cond_rendered=z["cond_rendered"][keep], seq=np.vstack(seqs).astype(np.float32),
             seq_offsets=np.r_[0, np.cumsum([len(s) for s in seqs])].astype(np.int64), step=z["step"][keep])
    assert set(d) == set(z), (sorted(d), sorted(z))
    L = dict(np.load(pl, allow_pickle=False)); pos = {u: j for j, u in enumerate(L["unit_ids"].astype(str))}
    n_chk = n_same = 0; miss = []
    for k, i in enumerate(keep):
        if sc[i] == "ouhdl": continue
        u = str(z["unit_ids"][i]); j = pos.get(u); n_chk += 1
        if j is None: miss.append(u); continue
        n_same += int(np.array_equal(sl(z, i), sl(L, j)) and np.array_equal(z["cond_rendered"][i], L["cond_rendered"][j]) and z["step"][i] == L["step"][j])
    per = {s: int((sc[keep] == s).sum()) for s in np.unique(sc[keep])}
    assert n_same == n_chk and not miss and sum(per.values()) == 80000, (n_same, n_chk, len(miss), per)
    prov = dict(split="m5_lto_train (leave-Tamil-out backbone train set, RQF R2d)", transform="m5_joint_train minus scripts=='hpl_tamil'; stored order kept; writer_ids re-indexed over sorted writer keys (m5_build.pack rule)",
                seed=None, inputs=dict(m5_joint_train=dict(path=pj, sha256=sha(pj)), m5_luo_train_for_check=dict(path=pl, sha256=sha(pl))),
                counts=dict(per_script=per, n_writers_per_script={s: int(len(np.unique(wr[sc[keep] == s]))) for s in per}),
                checks=dict(non_urdu_units_identical_to_luo=f"{n_same}/{n_chk}", n_removed_hpl_tamil=int((sc == "hpl_tamil").sum())))
    save(out, "m5_lto_train.npz", d, prov)

def tamil_items():
    sys.path.insert(0, MC); import m5_build as MB
    st = json.load(open(f"{D}/build_stats_M5.json")); val_users = set(st["tamil_val_users"])
    tar = f"{R}/data/stage/tamil_train_online.tar"; tmp = f"/tmp/r2d_tamil_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    if not glob.glob(f"{tmp}/**/usr_*", recursive=True):
        os.makedirs(tmp, exist_ok=True); subprocess.run(["tar", "-xf", tar, "-C", tmp, "--exclude=._*"], check=True)
    cand = [p for p in glob.glob(f"{tmp}/**/new", recursive=True) if os.path.isdir(p)]; assert len(cand) == 1, cand
    tdir = cand[0]; users = sorted(x for x in os.listdir(tdir) if x.startswith("usr_"))
    tf = {u: sorted(glob.glob(os.path.join(tdir, u, "*.txt"))) for u in users}
    assert val_users <= set(users)
    tr = [u for u in users if u not in val_users]; va = [u for u in users if u in val_users]
    ntr, nva = sum(len(tf[u]) for u in tr), sum(len(tf[u]) for u in va)
    assert (ntr, nva) == (st["pools"]["hpl_tamil_train"]["pool_units"], st["pools"]["hpl_tamil_val"]["pool_units"]), (ntr, nva)
    reader = f"{R}/data/hpl_tamil/read_unipen.py"
    inputs = dict(tamil_train_tar=dict(path=tar, sha256=sha(tar)), tamil_reader=dict(path=reader, sha256=sha(reader)),
                  m5_build=dict(path=f"{MC}/m5_build.py", sha256=sha(f"{MC}/m5_build.py")),
                  build_stats_M5=dict(path=f"{D}/build_stats_M5.json", sha256=sha(f"{D}/build_stats_M5.json")), extracted_dir_in_job=tdir)
    return MB, tf, tr, va, reader, inputs, tmp

def extract(MB, reader, paths, role, procs):
    with get_context("spawn").Pool(procs, initializer=MB._tamil_init, initargs=(reader,)) as pool:
        recs = [x for xx in pool.map(MB.tamil_extract, [(p, role) for p in paths], chunksize=8) for x in xx]
    return recs

def build_tamil(out, procs):
    MB, tf, tr, va, reader, inputs, tmp = tamil_items()
    # ---- p01, writer-stratified
    n_w = np.array([len(tf[u]) for u in tr]); N = int(n_w.sum()); K = int(np.floor(0.01 * N + 0.5))
    q = 0.01 * n_w; base = np.floor(q).astype(int); rem = q - base; extra = K - int(base.sum())
    tie = np.random.default_rng([SEED, 1]).permutation(len(tr))
    order = sorted(range(len(tr)), key=lambda i: (-round(rem[i], 12), tie[i])); quota = base.copy(); quota[order[:extra]] += 1
    rng = np.random.default_rng([SEED, 2]); paths = []
    for u, qq in zip(tr, quota):
        if qq: paths += [tf[u][j] for j in sorted(rng.choice(len(tf[u]), int(qq), replace=False))]
    recs = extract(MB, reader, paths, "train", procs); err = [(r["unit_id"], r["error"]) for r in recs if r["enc"] is None]
    d = MB.pack(recs, with_gt=False)
    J = dict(np.load(f"{D}/m5_joint_train.npz", allow_pickle=False)); pos = {u: j for j, u in enumerate(J["unit_ids"].astype(str))}
    sh = [(i, pos[u]) for i, u in enumerate(d["unit_ids"].astype(str)) if u in pos]
    same = sum(int(np.array_equal(sl(d, i), sl(J, j)) and np.array_equal(d["cond_rendered"][i], J["cond_rendered"][j])) for i, j in sh)
    assert same == len(sh), (same, len(sh))
    prov = dict(split="tamil_p01 (HPL Tamil 1 % FT subset, RQF R2d)", seed=SEED, inputs=inputs,
                transform="m5_build.tamil_extract (read_unipen -> encode_raw: clean_ink, fit_transform_unit 64-px eval canvas, 1 px resample / grown step) -> m5_build.pack(with_gt=False)",
                rule=f"writer-stratified 1 % of the M5 Tamil train pool (non-val users): K=floor(0.01*N+0.5); per-writer quota floor(0.01 n_w) + largest remainders (ties: default_rng([{SEED},1]).permutation); files: default_rng([{SEED},2]).choice per writer in sorted user order",
                counts=dict(pool_units=N, pool_writers=len(tr), K=K, n_selected_files=len(paths), n_encode_errors=len(err), writers_with_units=int((quota > 0).sum()),
                            quota_per_writer={u: int(qq) for u, qq in zip(tr, quota)}),
                checks=dict(units_shared_with_m5_joint_train=len(sh), identical_to_m5_encoding=f"{same}/{len(sh)}", encode_errors=err[:10],
                            writer_disjoint_from_val=bool(not set(tr) & set(va))))
    save(out, "tamil_p01.npz", d, prov)
    # ---- val1000 from the val pool
    vp = [p for u in va for p in tf[u]]; pick = np.random.default_rng([SEED, 3]).choice(len(vp), 1000, replace=False)
    recs = extract(MB, reader, [vp[i] for i in pick], "val", procs); err = [(r["unit_id"], r["error"]) for r in recs if r["enc"] is None]
    d = MB.pack(recs, with_gt=True)
    V = dict(np.load(f"{D}/m5_val_hpl_tamil.npz", allow_pickle=False)); pos = {u: j for j, u in enumerate(V["unit_ids"].astype(str))}
    sh = [(i, pos[u]) for i, u in enumerate(d["unit_ids"].astype(str)) if u in pos]
    same = sum(int(np.array_equal(sl(d, i), sl(V, j)) and np.array_equal(d["cond_rendered"][i], V["cond_rendered"][j])) for i, j in sh)
    assert same == len(sh), (same, len(sh))
    prov = dict(split="tamil_val1000 (HPL Tamil selection split for the R2d fine-tunes)", seed=SEED, inputs=inputs,
                transform="m5_build.tamil_extract -> m5_build.pack(with_gt=True); stored in draw order",
                rule=f"1,000 of the {len(vp)} units of the 6 M5 Tamil val users (build_stats_M5.json tamil_val_users): default_rng([{SEED},3]).choice(n,1000,replace=False)",
                counts=dict(pool_units=len(vp), pool_writers=len(va), val_users=va, n_encode_errors=len(err)),
                checks=dict(units_shared_with_m5_val_hpl_tamil=len(sh), identical_to_m5_encoding=f"{same}/{len(sh)}", encode_errors=err[:10],
                            note="val users are writer-disjoint from the M5 Tamil train pool and from the IWFHR06 test writers; never used to train or select LTO / LTO_W"))
    save(out, "tamil_val1000.npz", d, prov)
    subprocess.run(["bash", "-c", f"chmod -R u+w {tmp}; rm -rf {tmp}"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("lto", "tamil")); ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    build_lto(a.out) if a.mode == "lto" else build_tamil(a.out, a.procs)
