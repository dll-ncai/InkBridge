#!/usr/bin/env python3
"""Concatenate per-base R1d reads -> read/R1d_summary_{paired,absolute}.csv, R1d_summary_verdicts.json, PROVENANCE_R1d_summary.json."""
import json, os, glob, hashlib, sys
import pandas as pd
RD = "$U/exp/RQF/R1d"; o = RD + "/read"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for k in ("paired", "absolute"):
    fs = sorted(glob.glob(f"{o}/*/R1d_{k}_*.csv"))
    if fs: pd.concat([pd.read_csv(f) for f in fs]).to_csv(f"{o}/R1d_summary_{k}.csv", index=False)
V = {os.path.basename(f)[12:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{o}/*/R1d_verdict_*.json"))}
json.dump(V, open(f"{o}/R1d_summary_verdicts.json", "w"), indent=1)
json.dump(dict(kind="rqf_r1d_summary", inputs={f: sha(f) for f in glob.glob(f"{o}/*/*.csv") + glob.glob(f"{o}/*/R1d_verdict_*.json")},
               outputs={f: sha(f"{o}/{f}") for f in os.listdir(o) if f.startswith("R1d_summary")}, code_sha256=sha(__file__)),
          open(f"{o}/PROVENANCE_R1d_summary.json", "w"), indent=1)
print({b: v["PASS"] for b, v in V.items()})
