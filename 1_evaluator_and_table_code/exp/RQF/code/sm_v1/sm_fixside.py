#!/usr/bin/env python3
"""Relayed VPS Route-A preds (c1_sample) -> CLUSTER-A-scorable copy: preds_clusterA/ with the npz symlinked and the sidecar json
'npz' path mapped $VPS_ROOT\\uhwr\\stage\\<rel> -> $U/<rel> (original kept as npz_vps). Usage: sm_fixside.py SRC DST"""
import json, os, sys, glob
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True); n = 0
for j in glob.glob(os.path.join(src, "*.json")):
    d = json.load(open(j)); p = d.get("npz", "")
    if p.startswith("C:"):
        rel = p.split("\\stage\\", 1)[1].replace("\\", "/"); d["npz_vps"] = p; d["npz"] = "$U/" + rel
        assert os.path.exists(d["npz"]), d["npz"]
    json.dump(d, open(os.path.join(dst, os.path.basename(j)), "w"), indent=1); n += 1
    z = j[:-5] + ".npz"
    if os.path.exists(z) and not os.path.exists(os.path.join(dst, os.path.basename(z))): os.symlink(os.path.abspath(z), os.path.join(dst, os.path.basename(z)))
print("fixed", n)
