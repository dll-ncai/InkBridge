#!/usr/bin/env python3
"""FMDX warm monitor: A4.4 composed preds. Per unit: p_rtl (ScriptCNN router exp/D3_ROUTE/router/script_clf.pt, sha256 22bb8802..., outputs stored per unit in
exp/D3_ROUTE/router/manifest_<set>.csv column p_rtl) >= 0.5 -> the arm's Urdu-FT preds, else the arm's selected backbone preds.
Inputs: --bb-dir/--ft-dir with <set>__<ARM>_<bb|ft>_ms<m>__seed<k>_preds.{npz,json} (bb_sample.py outputs). Output: <set>__<ARM>_cmp_ms<ft_m>__seed<k>_preds.{npz,json}
(preds, unit_ids, specialist = 'ft'/'bb' per unit) + routes_<set>__<ARM>.csv. Original files are only read."""
import argparse, csv, glob, hashlib, json, os, re, sys, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--bb-dir", required=True); ap.add_argument("--ft-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--manifest-dir", default="$U/exp/D3_ROUTE/router"); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--sets", default="test_casia_zh,test_casia_en,test_hpl_tamil,test_iam500,german_unipen_de_test,german_onhw_all")
a = ap.parse_args()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
os.makedirs(a.out, exist_ok=True); done = 0
for s in a.sets.split(","):
    man = os.path.join(a.manifest_dir, f"manifest_{s}.csv")
    rows = list(csv.DictReader(open(man))); prtl = {r["unit_id"]: float(r["p_rtl"]) for r in rows}
    for seed in (0, 1, 2):
        bf = glob.glob(f"{a.bb_dir}/{s}__{a.arm}_bb_ms*__seed{seed}_preds.npz"); ff = glob.glob(f"{a.ft_dir}/{s}__{a.arm}_ft_ms*__seed{seed}_preds.npz")
        if len(bf) != 1 or len(ff) != 1: print("skip", s, seed, len(bf), len(ff)); continue
        B = np.load(bf[0], allow_pickle=False); F = np.load(ff[0], allow_pickle=False)
        ub = B["unit_ids"].astype(str); uf = F["unit_ids"].astype(str)
        assert (ub == uf).all(), f"unit order differs {s} seed {seed}"
        missing = [u for u in ub if u not in prtl]; assert not missing, (s, missing[:3])
        p = np.array([prtl[u] for u in ub]); use_ft = p >= a.thr
        P = np.where(use_ft[:, None, None], F["preds"], B["preds"]).astype(np.float32)
        ms_b = int(re.search(r"_bb_ms(\d+)__", bf[0]).group(1)); ms_f = int(re.search(r"_ft_ms(\d+)__", ff[0]).group(1))
        stem = f"{s}__{a.arm}_cmp_ms{ms_f}__seed{seed}"; op = os.path.join(a.out, stem + "_preds.npz")
        spec = np.where(use_ft, "ft", "bb")
        bm = json.load(open(bf[0][:-4] + ".json")); fm = json.load(open(ff[0][:-4] + ".json"))
        meta = dict(kind="fmdx_composed_A4.4", arm=a.arm, set=s, seed=seed, milestone=ms_f, backbone_milestone=ms_b, thr=a.thr, router_ckpt_sha256_expected="22bb8802f2da46fc37972eec430a1439ad830905712f22f6d923c38829db45aa",
                    manifest=man, manifest_sha256=sha(man), n_units=int(len(ub)), n_ft=int(use_ft.sum()), n_bb=int((~use_ft).sum()),
                    bb_preds=os.path.basename(bf[0]), bb_preds_sha256=sha(bf[0]), ft_preds=os.path.basename(ff[0]), ft_preds_sha256=sha(ff[0]),
                    gt_npz_sha256=bm.get("gt_npz_sha256"), device_bb=bm.get("device"), device_ft=fm.get("device"), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        tmp = op[:-4] + ".part.npz"; np.savez_compressed(tmp, preds=P, unit_ids=ub, specialist=spec, meta=json.dumps(meta)); os.replace(tmp, op)
        json.dump(dict(meta, preds_file=os.path.basename(op), preds_sha256=sha(op)), open(op[:-4] + ".json", "w"), indent=1)
        if seed == 0:
            with open(os.path.join(a.out, f"routes_{s}__{a.arm}.csv"), "w") as f:
                f.write("unit_id,p_rtl,route\n"); [f.write(f"{u},{pp:.6f},{'ft' if r else 'bb'}\n") for u, pp, r in zip(ub, p, use_ft)]
        done += 1; print(stem, "n_ft", int(use_ft.sum()), "of", len(ub))
print("composed files", done)
