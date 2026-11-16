"""M1 (A1.2) E1 units for the 7 RQF test sets, built from the exact GT npz files the RQF scorer uses.
Unit format = e_bundle_v1/e1_units.pkl: dict(uid, cluster, ink=list of (n,2) strokes on the 64x64 canvas, t=list of per-stroke times in s or None).
Sets (key -> npz): ouhdl, casia_zh, casia_en, tamil, iam500, unipen_de, onhw.
Sampling: sets with > 500 units: numpy default_rng(20261016).choice(n,500,replace=False), sorted; else all units.
Ink = gt_points split by gt_stroke_id (order of first appearance), already on the unit canvas (asserted within [0,64]).
Times: UNIPEN-de and OnHW only (raw_points col t, ms -> s, one-to-one with gt_points, asserted); other sets have no timestamps (t=None).
Cluster = writer (writer_ids; >=0 known) else uid; OnHW: unit (RQF unit bootstrap, 2 writers).
python build_m1_units.py --out DIR
"""
import argparse, hashlib, json, os, pickle, time
import numpy as np
U = "$U"
SETS = {
 "ouhdl": (f"{U}/data/ouhdl_e0_v3/e0_test.npz", None),
 "casia_zh": (f"{U}/exp/X4a/data/x4a_casia_zh.npz", None),
 "casia_en": (f"{U}/exp/X4a/data/x4a_casia_en.npz", None),
 "tamil": (f"{U}/exp/X4a/data/x4a_hpl_tamil.npz", None),
 "iam500": (f"{U}/data/m5_multiscript_v1/m5_test_iam500.npz", None),
 "unipen_de": (f"{U}/data/d2_v2/unipen_de/e0_test.npz", f"{U}/data/d2_v2/unipen_de/units_test.npz"),
 "onhw": (f"{U}/data/d2_v2/onhw/e0_all.npz", f"{U}/data/d2_v2/onhw/units_all.npz"),
}
SEED, N = 20261016, 500

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    E, info = {}, dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seed=SEED, n_target=N, sets={}, inputs={})
    for key, (gtp, rawp) in SETS.items():
        g = np.load(gtp, allow_pickle=False)
        n = len(g["unit_ids"]); pick = list(range(n)) if n <= N else sorted(np.random.default_rng(SEED).choice(n, N, replace=False).tolist())
        P, SID, OFF = g["gt_points"], g["gt_stroke_id"], g["gt_offsets"]
        if rawp:
            z = np.load(rawp, allow_pickle=False); assert (z["unit_ids"].astype(str) == g["unit_ids"].astype(str)).all()
            assert z["raw_points"].shape[0] == P.shape[0] and (z["raw_offsets"] == OFF).all(); assert list(z["raw_columns"][:3]) == ["x", "y", "t"]
            RT = z["raw_points"][:, 2].astype(float) / 1000.0
        units, nstr, steps = [], [], []
        for k in pick:
            q, s = P[OFF[k]:OFF[k + 1]], SID[OFF[k]:OFF[k + 1]]
            ink, tt = [], []
            for i in dict.fromkeys(int(x) for x in s):
                m = s == i; ink.append(q[m].astype(float))
                if rawp: tt.append(RT[OFF[k]:OFF[k + 1]][m])
            assert min(x.min() for x in ink) >= 0 and max(x.max() for x in ink) <= 64, (key, k)
            t = None
            if rawp:
                cat = np.concatenate(tt)
                if np.isfinite(cat).all(): t = tt; d = np.diff(cat); steps += list(d[d > 0])
            w = int(g["writer_ids"][k]); uid = str(g["unit_ids"][k])
            cl = uid if (key == "onhw" or w < 0) else f"{key}:w{w}"
            units.append(dict(uid=uid, cluster=cl, ink=ink, t=t)); nstr.append(len(ink))
        assert len({u["uid"] for u in units}) == len(units)
        E[key] = units
        info["sets"][key] = dict(n_pool=int(n), n=len(units), n_clusters=len({u["cluster"] for u in units}), n_units_with_times=int(sum(u["t"] is not None for u in units)),
                                 median_positive_time_step_s=float(np.median(steps)) if steps else None, n_strokes_median=float(np.median(nstr)), gt_npz=gtp, raw_npz=rawp)
        info["inputs"][gtp] = sha(gtp)
        if rawp: info["inputs"][rawp] = sha(rawp)
    pickle.dump(E, open(os.path.join(a.out, "e1_units.pkl"), "wb"))
    info["code_sha256"] = sha(__file__); info["output_sha256"] = sha(os.path.join(a.out, "e1_units.pkl"))
    json.dump(info, open(os.path.join(a.out, "m1_units_info.json"), "w"), indent=1); print(json.dumps(info)[:3000])

if __name__ == "__main__": main()
