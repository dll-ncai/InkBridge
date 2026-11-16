#!/usr/bin/env python3
"""FMDX warm monitor end-of-run (10 Nov 18:00Z): per-arm last stage for every non-cancelled WARM/SCRATCH arm (prereg A4.3f 'not assessed').
Reads admin/arm_queue.csv, runs/<arm>_clusterA, monitor preds/score rows, state/warm_state.json, squeue. Writes runs/fmdx_monitor/tables/FMDX_not_assessed_A43f.csv + prints JSON."""
import csv, json, os, glob, subprocess, time
X = "$U/exp/FMD/xs"; MR = X + "/runs/fmdx_monitor"
S = json.load(open(MR + "/state/warm_state.json")); q = list(csv.DictReader(open(X + "/admin/arm_queue.csv", newline="")))
sq = {}
for l in subprocess.run(["squeue", "-u", os.environ.get("USER", "<user>"), "-h", "-o", "%j|%T|%r"], capture_output=True, text=True).stdout.splitlines():
    p = l.split("|"); sq.setdefault(p[0], []).append(p[1] + ("(" + p[2][:30] + ")" if p[1] == "PENDING" else ""))
def step(d):
    try:
        t = open(d + "/train_log.jsonl", "rb"); t.seek(max(0, os.path.getsize(d + "/train_log.jsonl") - 4000)); s = t.read().decode("utf-8", "replace")
        import re; m = re.findall(r'"step": (\d+)', s); return int(m[-1]) if m else None
    except Exception: return None
NEED = {"bb": 18, "ft": 24, "cmp": 18}; out = []
for r in q:
    a = r["arm_id"]
    if r["state"].startswith("CANCELLED") or "BLOCKED" in r["state"] or "HELD" in r["state"]: continue
    st = S["arms"].get(a, {}); rows = st.get("rows", {}); d = f"{X}/runs/{a}_clusterA"
    if st and all(rows.get(k, 0) >= NEED[k] for k in NEED): continue
    tc = os.path.exists(d + "/training_complete.json")
    if not os.path.isdir(d) and not sq.get("fx_" + a): stage = f"training not started on CLUSTER-A (queue state {r['state']})"
    elif not tc: stage = f"training step {step(d)} of {'40000' if r['regime']=='warm' else '100000'} (slurm {','.join(sq.get('fx_'+a, ['not queued']))})"
    elif st.get("ft_ms") is None: stage = f"backbone selected ms{st.get('bb_ms')}; FT not finished (fxm_ft {','.join(sq.get('fxm_ft_'+a, ['not queued']))}); bb rows {rows.get('bb',0)}/18"
    else:
        miss = [f"{k} rows {rows.get(k,0)}/{NEED[k]}" for k in NEED if rows.get(k, 0) < NEED[k]]
        stage = f"FT selected ms{st['ft_ms']}; incomplete: {'; '.join(miss)} (sampling {','.join(sq.get('fxm_as9_'+a+'_bb', [])+sq.get('fxm_as9_'+a+'_ft', [])) or 'none queued'})"
    out.append(dict(arm=a, regime=r["regime"], fm=r["fm"], queue_state=r["state"], label="not assessed by 2026-11-10T18:00Z (A4.3f)", last_stage=stage))
with open(MR + "/tables/FMDX_not_assessed_A43f.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else ["arm"]); w.writeheader(); w.writerows(out)
full = sorted(a for a, st in S["arms"].items() if all(st.get("rows", {}).get(k, 0) >= NEED[k] for k in NEED))
print(json.dumps(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), full=full, n_not_assessed=len(out), tables_job=S.get("tables_job"), tables_for=S.get("tables_built_for"),
                      partial={a: st.get("rows") for a, st in S["arms"].items() if a not in full}, na=[(o["arm"], o["last_stage"][:90]) for o in out if o["regime"] == "warm" or "training step" in o["last_stage"] or "not started" not in o["last_stage"]][:40])))
