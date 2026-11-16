#!/usr/bin/env python3
"""CLUSTER-B: download PUBLIC FM weights directly from HF at the revision pinned on CLUSTER-A, verify every downloaded file against the CLUSTER-A per-file sha256 (admin/clusterB_expected.json).
Used because the VPS hub path CLUSTER-A->VPS->CLUSTER-B is blocked for CLUSTER-B (VPS authorized_keys 'from=' restriction).  Track XS staging."""
import os, sys, json, hashlib, time
sys.path.insert(0, "$U/exp/FMD/xs/pylibs_clusterB")
R = "$U"; os.environ["HF_HOME"] = R + "/exp/FMD/xs/weights/.hf_home"
from huggingface_hub import snapshot_download
exp = json.load(open(R + "/exp/FMD/xs/admin/clusterB_expected.json")); res = []
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
for e in exp:
    t0 = time.time(); out = f"{R}/{e['local_dir']}"; r = dict(name=e["name"], repo=e["repo"], revision=e["revision"])
    try:
        snapshot_download(e["repo"], revision=e["revision"], local_dir=out, allow_patterns=e["allow_patterns"], max_workers=8)
        ok = bad = 0; bads = []
        for dp, dn, fn in os.walk(out):
            dn[:] = [d for d in dn if d != ".cache"]
            for f in fn:
                p = os.path.join(dp, f); rel = os.path.relpath(p, out)
                if rel not in e["files"]: bads.append("UNEXPECTED " + rel); bad += 1; continue
                if sha(p) == e["files"][rel]: ok += 1
                else: bad += 1; bads.append("SHA_MISMATCH " + rel)
        r.update(status="ok" if bad == 0 else "mismatch", files_ok=ok, files_bad=bad, bad=bads[:10], seconds=round(time.time() - t0))
    except Exception as ex:
        r.update(status="error", error=type(ex).__name__ + ": " + str(ex)[:400])
    res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open(R + "/exp/FMD/xs/admin/clusterB_stage_result.json", "w"), indent=1)
