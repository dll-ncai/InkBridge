#!/usr/bin/env python3
"""Build the CLUSTER-B mirror file lists (paths relative to $U).  usage: mirror_lists.py <tag> <spec...>   spec = dir:/path | file:/path | w:<weightdir>[:excl-glob,...]"""
import sys, os, fnmatch, json, hashlib
R = "$U"; tag = sys.argv[1]; files = []
def add_dir(rel, excl=()):
    for dp, dn, fn in os.walk(f"{R}/{rel}"):
        dn[:] = [d for d in dn if d not in (".cache", "__pycache__", ".git", "scratch")]
        for f in fn:
            p = os.path.join(dp, f); r = os.path.relpath(p, R)
            if f.startswith("slurm-") or f.endswith(".pyc") or any(fnmatch.fnmatch(f, e) for e in excl): continue
            files.append(r) if os.path.isfile(p) and not os.path.islink(p) else None
for s in sys.argv[2:]:
    k, _, v = s.partition(":")
    if k == "dir": add_dir(v)
    elif k == "file": files.append(v)
    elif k == "w":
        name, _, ex = v.partition(":"); add_dir(f"exp/FMD/xs/weights/{name}" if os.path.isdir(f"{R}/exp/FMD/xs/weights/{name}") else f"exp/FMD/weights/{name}", ex.split(",") if ex else ())
files = sorted(set(files))
out = f"{R}/exp/FMD/xs/admin/mirror"; os.makedirs(out, exist_ok=True)
open(f"{out}/mirror_{tag}.list", "w").write("\n".join(files) + "\n")
print(tag, len(files), "files", round(sum(os.path.getsize(f"{R}/{f}") for f in files)/1e9, 3), "GB")
