#!/usr/bin/env python3
"""Addendum-1 summary table from MANIFEST_v2.json (A1 models only).  Track XS staging."""
import json
X="$U/exp/FMD/xs"
m=json.load(open(X+"/weights/MANIFEST_v2.json"))
L=["# FMDX staging - addendum 1 summary (facts)","",f"Source: weights/MANIFEST_v2.json (created {m['created_utc']}). Token map = H x W at 224-px input unless the input_px column says otherwise; load-test conventions only (not prereg extractors). units/s = A4000, load-test forward, 4 val1000 renders batch-timed (xs_probe.py time).","",
"| local dir | repo id | revision | licence | GB | load status | token map(s) | input px | units/s A4000 |","|---|---|---|---|---|---|---|---|---|"]
n=0
for x in m["models"]:
    if not x.get("addendum1"): continue
    n+=1
    lt=x.get("load_test") or []
    st="; ".join(t["status"] for t in lt) or "no load test"
    tm="; ".join(json.dumps(t["token_map_hw_at_224"]) for t in lt if t["status"]=="ok") or "-"
    px="; ".join(str(t.get("input_px")) for t in lt if t["status"]=="ok" and t.get("input_px")) or "224"
    lic=x["license"] if isinstance(x["license"],str) else ",".join(x["license"] or [])
    u=x.get("units_per_s"); us="; ".join(f"{k}={v}" for k,v in u.items()) if u else "-"
    L.append(f"| {x['local_name']} | {x['repo_id']} | `{x['revision'][:10]}` | {lic} | {x['total_bytes']/1e9:.2f} | {st} | {tm} | {px} | {us} |")
L+=["",f"{n} models staged in addendum 1.","","## Blocked (addendum 1)",""]
for b in m["blocked"]:
    L.append(f"- {b['local_name']} ({b['repo_id']}): {b['error_type']}: {(b['error'] or '')[:400]}")
open(X+"/admin/ADDENDUM1_SUMMARY.md","w").write("\n".join(L)); print(n)
