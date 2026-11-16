#!/usr/bin/env python3
"""Check downloaded checkpoint files against checkpoints/checkpoint_manifest.csv.

Usage (from the repository root, after downloading files into the folder given by the
column public_file, e.g. checkpoints/script_router/router_reported.pt):
    python checkpoints/verify_checkpoints.py              # every file that is present
    python checkpoints/verify_checkpoints.py --root /path/to/download --require-all

For each manifest row whose file exists under --root, the size in bytes and the SHA-256
checksum are compared with the manifest. Exit code 1 if a file differs, or if
--require-all is given and a file is missing. Standard library only.
"""
import argparse, csv, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "checkpoint_manifest.csv"))
    ap.add_argument("--root", default=os.path.dirname(HERE), help="folder that contains checkpoints/<family>/<file>")
    ap.add_argument("--require-all", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest, newline="")))
    ok = bad = missing = 0
    for r in rows:
        p = os.path.join(a.root, r["public_file"])
        if not os.path.exists(p):
            missing += 1
            continue
        size = os.path.getsize(p)
        if size != int(r["bytes"]):
            bad += 1; print("SIZE DIFFERS  %s  (%d, manifest %s)" % (r["public_file"], size, r["bytes"])); continue
        if sha256(p) != r["sha256"]:
            bad += 1; print("SHA256 DIFFERS  %s" % r["public_file"]); continue
        ok += 1; print("ok  %s" % r["public_file"])
    print("manifest rows: %d   verified: %d   different: %d   not present: %d" % (len(rows), ok, bad, missing))
    sys.exit(1 if bad or (a.require_all and missing) else 0)

if __name__ == "__main__":
    main()
