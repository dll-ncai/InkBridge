#!/usr/bin/env python3
"""FMDX warm monitor: make lane_score.py-readable score inputs from bb_sample.py outputs WITHOUT touching the originals.
For every <stem>_preds.npz (+ .json written by bb_sample.py) in --src: symlink the npz into --dst and write a sidecar json with the keys
lane_score.py needs (set, route_mode, seed, milestone, npz_clusterA, source_sha256_expected) plus the original sidecar under 'bb_sample_meta'.
Stem convention (vps_sample_cmd.py): <set>__<tag>__seed<k>. --gt-map set=path,...  maps set -> CLUSTER-A gt npz."""
import argparse, json, os, re, hashlib, sys
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--gt-map", required=True)
a = ap.parse_args()
gm = dict(x.split("=", 1) for x in a.gt_map.split(","))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
os.makedirs(a.dst, exist_ok=True); n = 0
for f in sorted(os.listdir(a.src)):
    if not f.endswith("_preds.npz"): continue
    m = re.match(r"(.+?)__(.+)__seed(\d+)_preds\.npz$", f)
    if not m: continue
    st, tag, sd = m.group(1), m.group(2), int(m.group(3))
    js = os.path.join(a.src, f[:-4] + ".json")
    if not os.path.exists(js) or st not in gm: continue
    meta = json.load(open(js)); d = os.path.join(a.dst, f)
    if os.path.lexists(d): os.remove(d)
    os.symlink(os.path.abspath(os.path.join(a.src, f)), d)
    side = dict(set=st, label=tag, route_mode=tag, specialist=tag, seed=sd, milestone=meta.get("milestone"), npz_clusterA=gm[st],
                source_sha256_expected=meta.get("gt_npz_sha256"), preds_sha256=meta.get("preds_sha256"), bb_sample_meta=meta)
    json.dump(side, open(os.path.join(a.dst, f[:-4] + ".json"), "w"), indent=1); n += 1
print("adapted", n)
