"""Build the compact input bundle for E1/E2/M1a/M2a/E3a (runs locally, reads ~/Work/uhwr/data).

python -m uhwr_exp.build_inputs --data ~/Work/uhwr/data --momna ext/momna --out bundle

Writes into --out:
  test.csv                OUHD-L test split manifest (copied)
  ligatures_v3_test.csv   v3 unit index (copied)
  lines_test.pkl          {sample_id: {raw, t, p, writer_id, text, img}}: pen-down strokes of
                          each test line (runs of pen_down == 1; X/Y cood., Time ms, Pressure)
  img/<sample_id>.png     OUHD-L dataset line images (copied)
  e1_units.pkl            {ouhdl|casia|tamil: [unit dicts]}: 500 random GT units per script,
                          ink on the 64 x 64 unit canvas, per-point times (s) or None
  momna_gt_tokens.pkl     {sample_id: (tokens (T,3) float32, text)}: Momna recogniser input
                          built from the raw CSV by the repo's own OnlineRecogDataset
                          (STROKE_SIMPLIFY default on, max_points 1024) = verify_results.py path
  momna_gt_tokens_raw.pkl the same with simplification off (STROKE_SIMPLIFY=0), the input
                          format of recognizer_realonly_baseline.pt
  PROVENANCE_e_inputs.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from uhwr_eval import apply_transform, fit_transform_unit
from uhwr_exp.common import SEED

N_E1 = 500


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def line_strokes(csv_path):
    d = pd.read_csv(csv_path, usecols=["X cood.", "Y cood.", "pen_down", "Time", "Pressure"])
    pdn = d["pen_down"].to_numpy() == 1
    xy = d[["X cood.", "Y cood."]].to_numpy(np.float64)
    t = d["Time"].to_numpy(np.float64)
    p = d["Pressure"].to_numpy(np.float64)
    e = np.diff(np.r_[0, pdn.astype(int), 0])
    st, en = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    raw, tt, pp = [], [], []
    for s, q in zip(st, en):
        ok = np.isfinite(xy[s:q]).all(1)
        if ok.any():
            raw.append(xy[s:q][ok]); tt.append(t[s:q][ok]); pp.append(p[s:q][ok])
    return raw, tt, pp


def unit_from(strokes, times):
    s, tx, ty = fit_transform_unit(strokes, 64, 4)
    return apply_transform(strokes, s, tx, ty), times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--momna", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = os.path.expanduser(a.data)
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    prov = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seed=SEED,
                command=" ".join(sys.argv), inputs={}, outputs={}, notes=[])
    root = os.path.join(D, "ouhdl")
    v3 = os.path.join(D, "ouhdl_ligatures_v3")
    for src, dst in ((os.path.join(root, "test.csv"), "test.csv"),
                     (os.path.join(v3, "ligatures_v3_test.csv"), "ligatures_v3_test.csv")):
        shutil.copy(src, os.path.join(a.out, dst))
        prov["inputs"][src] = sha(src)
    man = pd.read_csv(os.path.join(root, "test.csv"))
    man["sample_id"] = man["csv"].map(lambda p: p.split("/")[-3].replace("Data_", "") + "_" +
                                      os.path.splitext(os.path.basename(p))[0].replace("csv_", ""))
    lines = {}
    csv_sha = hashlib.sha256()
    for r in man.itertuples(index=False):
        raw, tt, pp = line_strokes(os.path.join(root, r.csv))
        csv_sha.update(open(os.path.join(root, r.csv), "rb").read())
        shutil.copy(os.path.join(root, r.img), os.path.join(a.out, "img", f"{r.sample_id}.png"))
        lines[r.sample_id] = dict(raw=raw, t=tt, p=pp, writer_id=int(r.writer_id), text=r.line,
                                  img=f"img/{r.sample_id}.png")
    prov["inputs"]["ouhdl test line CSVs (253, concatenated in manifest order)"] = csv_sha.hexdigest()
    pickle.dump(lines, open(os.path.join(a.out, "lines_test.pkl"), "wb"))

    # ---- check v3 units against the v3 npz points (stroke_idx -> line strokes)
    sys.path.insert(0, v3)
    import ouhdl_ligatures as OL
    idx, U = OL.load_split(v3, "test")
    prov["inputs"][os.path.join(v3, "ligatures_v3_test.npz")] = sha(os.path.join(v3, "ligatures_v3_test.npz"))
    import ast
    bad = 0
    for k, r in enumerate(idx.itertuples(index=False)):
        st = U.get(k)
        mine = [lines[r.sample_id]["raw"][i] for i in sorted(ast.literal_eval(r.stroke_idx))]
        mt = [lines[r.sample_id]["t"][i] for i in sorted(ast.literal_eval(r.stroke_idx))]
        if len(st) != len(mine) or any(not (np.array_equal(s[:, :2], m) and np.array_equal(s[:, 2], tm))
                                       for s, m, tm in zip(st, mine, mt)):
            bad += 1
    prov["notes"].append(f"v3 units rebuilt from line CSV strokes via stroke_idx: {len(idx) - bad}/{len(idx)} identical to ligatures_v3_test.npz (x, y, time)")
    assert bad == 0, bad

    # ---- E1 samples
    rng = np.random.default_rng(SEED)
    e1 = {}
    pick = rng.choice(len(idx), N_E1, replace=False)
    e1["ouhdl"] = []
    for k in sorted(pick):
        r = idx.iloc[k]
        st = U.get(int(k))
        ink, _ = unit_from([s[:, :2].astype(float) for s in st], None)
        e1["ouhdl"].append(dict(uid=r.unit_id, cluster=r.sample_id, ink=ink,
                                t=[s[:, 2].astype(float) / 1000.0 for s in st]))
    # CASIA OLHWDB1.1 test (writer-disjoint test half; 60 files); sample 500 characters
    sys.path.insert(0, os.path.join(D, "casia"))
    from read_pot import iter_pot
    pots = sorted(glob.glob(os.path.join(D, "casia", "extracted", "Pot1.1Test", "*.pot")))
    allc = []
    for p in pots:
        for j, (lab, st) in enumerate(iter_pot(p)):
            allc.append((os.path.basename(p), j, lab))
    prov["inputs"]["CASIA OLHWDB1.1 Pot1.1Test (*.pot)"] = {os.path.basename(p): sha(p) for p in pots}
    ck = sorted(rng.choice(len(allc), N_E1, replace=False).tolist())
    want = {}
    for k in ck:
        want.setdefault(allc[k][0], set()).add(allc[k][1])
    e1["casia"] = []
    for p in pots:
        b = os.path.basename(p)
        if b not in want:
            continue
        for j, (lab, st) in enumerate(iter_pot(p)):
            if j in want[b]:
                ink, _ = unit_from([np.asarray(s, float) for s in st], None)
                e1["casia"].append(dict(uid=f"{b}#{j}", cluster=b, label=lab, ink=ink, t=None))
    # HPL Tamil IWFHR06 test online: T channel unreliable -> time = point index / 120 Hz
    sys.path.insert(0, os.path.join(D, "hpl_tamil"))
    from read_unipen import read_unipen
    tdir = os.path.join(D, "hpl_tamil", "extracted", "hpl-tamil-iwfhr06-test-online", "hpl-tamil-iwfhr06-test-online")
    tf = sorted(glob.glob(os.path.join(tdir, "*.txt")))
    tk = sorted(rng.choice(len(tf), N_E1, replace=False).tolist())
    e1["tamil"] = []
    th = hashlib.sha256()
    for k in tk:
        th.update(open(tf[k], "rb").read())
        _, st = read_unipen(tf[k], label=-1)
        st = [s for s in st if len(s)]
        ink, _ = unit_from([s[:, :2] for s in st], None)
        n0 = 0
        times = []
        for s in st:  # pen-down sample index at 120 points/s (pen-up time not known)
            times.append((n0 + np.arange(len(s))) / 120.0); n0 += len(s)
        e1["tamil"].append(dict(uid=os.path.basename(tf[k]), cluster=os.path.basename(tf[k]), ink=ink, t=times))
    prov["inputs"]["HPL Tamil IWFHR06 test online, 500 sampled files (concatenated in sample order)"] = th.hexdigest()
    prov["notes"].append("E1 samples: rng = numpy default_rng(20261016); OUHD-L 500 of 4,406 v3 test units; CASIA 500 of all characters in the 60 OLHWDB1.1 test .pot files; Tamil 500 of 26,926 IWFHR06 test files. CASIA has no timestamps (t=None). Tamil T channel is unreliable per the dataset docs, so time = pen-down sample index / 120 Hz.")
    pickle.dump(e1, open(os.path.join(a.out, "e1_units.pkl"), "wb"))

    # ---- Momna recogniser GT tokens via the repo's own dataset class
    os.environ["STROKE_DATA_ROOT"] = root
    M = os.path.abspath(a.momna)
    for sub in ["", "common", "stage1_unet_segmentation", "stage2_trajectory_recovery",
                "stage3_order_predictor", "stage4_recognizer", "stage5_two_stage_training"]:
        sys.path.insert(0, os.path.join(M, sub))
    import config
    from vocab_loader import load_vocab
    from recog_datasets import OnlineRecogDataset, APPLY_SIMPLIFICATION
    c2i = load_vocab(config.RECOG_VOCAB_JSON)
    tmp = os.path.join(a.out, "_momna_manifest.csv")
    man.to_csv(tmp, index=False)
    ds = OnlineRecogDataset(tmp, char_to_idx=c2i, max_points=1024)
    toks = {}
    for k, di in enumerate(ds.indices):
        x, y, n, _ = ds[k]
        toks[man.iloc[di]["sample_id"]] = (x.numpy(), man.iloc[di]["line"])
    import recog_datasets as RD
    RD.APPLY_SIMPLIFICATION = False  # = STROKE_SIMPLIFY=0, the realonly_baseline preprocessing
    toks_raw = {}
    for k, di in enumerate(ds.indices):
        x, y, n, _ = ds[k]
        toks_raw[man.iloc[di]["sample_id"]] = (x.numpy(), man.iloc[di]["line"])
    os.remove(tmp)
    pickle.dump(toks, open(os.path.join(a.out, "momna_gt_tokens.pkl"), "wb"))
    pickle.dump(toks_raw, open(os.path.join(a.out, "momna_gt_tokens_raw.pkl"), "wb"))
    prov["notes"].append(f"momna_gt_tokens: {len(toks)}/{len(man)} lines kept by OnlineRecogDataset (lines whose text encodes to empty are dropped); APPLY_SIMPLIFICATION={APPLY_SIMPLIFICATION}")
    prov["momna_repo"] = dict(url="https://github.com/dll-<site>/Synthetic-Stroke-Trajectory",
                              commit=subprocess.run(["git", "-C", M, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "see code provenance")
    for f in sorted(os.listdir(a.out)):
        p = os.path.join(a.out, f)
        if os.path.isfile(p):
            prov["outputs"][f] = sha(p)
    prov["outputs"]["img/ (253 PNG, concatenated sorted)"] = hashlib.sha256(b"".join(
        open(p, "rb").read() for p in sorted(glob.glob(os.path.join(a.out, "img", "*.png"))))).hexdigest()
    prov["environment"] = dict(python=sys.version, platform=platform.platform(),
                               numpy=np.__version__, pandas=pd.__version__)
    json.dump(prov, open(os.path.join(a.out, "PROVENANCE_e_inputs.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v if k != "inputs" else len(v)) for k, v in prov.items() if k != "outputs"}, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
