#!/usr/bin/env python3
"""FMDX warm monitor: read-only state collector (stdlib). Run on CLUSTER-A login node (or CLUSTER-B login with --runs-dir/--queue).
Reads X/admin/arm_queue.csv, X/admin/priority.json, run dirs <arm>_<lane>, squeue; prints one JSON to stdout.
Never writes into run dirs and never touches jobs."""
import csv, json, os, subprocess, sys, glob, argparse, time
ap = argparse.ArgumentParser()
ap.add_argument("--x", default="$U/exp/FMD/xs")
ap.add_argument("--runs-dir"); ap.add_argument("--lane", default="clusterA")
ap.add_argument("--only-warm", action="store_true", default=True)
a = ap.parse_args()
X = a.x; RUNS = a.runs_dir or X + "/runs"
def rj(p):
    try: return json.load(open(p))
    except Exception: return None
def tail_lines(p, n=3, blk=65536):
    try:
        sz = os.path.getsize(p)
        with open(p, "rb") as f:
            f.seek(max(0, sz - blk)); t = f.read().decode("utf-8", "replace").splitlines()
        return t[-n:]
    except Exception: return []
q = list(csv.DictReader(open(X + "/admin/arm_queue.csv", newline="")))
order = (rj(X + "/admin/priority.json") or {}).get("order", [])
rank = {k: i for i, k in enumerate(order)}
sq = {}
try:
    o = subprocess.run(["squeue", "-u", os.environ.get("USER", "<user>"), "-h", "-o", "%i|%j|%T|%M|%R|%b"], capture_output=True, text=True, timeout=40).stdout
    for l in o.splitlines():
        p = l.split("|")
        if len(p) >= 5 and p[1].startswith("fx_"): sq[p[1][3:]] = dict(job=p[0], state=p[2], elapsed=p[3], reason=p[4], gres=p[5] if len(p) > 5 else "")
except Exception as e:
    sq = {"_err": str(e)}
out = []
for r in q:
    if a.only_warm and r["regime"] != "warm": continue
    if r["state"] in ("CANCELLED_GATE_FAIL",) : continue
    arm = r["arm_id"]; d = f"{RUNS}/{arm}_{a.lane}"
    info = dict(arm=arm, rank=rank.get(arm, 9999), fm=r["fm"], queue_state=r["state"], clusterA_job=r["clusterA_job"], clusterB_job=r["clusterB_job"], vps_cmd=r["vps_cmd"], note=r["note"][:80], slurm=sq.get(arm))
    if os.path.isdir(d):
        tl = tail_lines(d + "/train_log.jsonl", 40)
        step = None
        for l in reversed(tl):
            try:
                j = json.loads(l)
                if j.get("event") == "train": step = j["step"]; sps = j.get("sec_per_step"); break
            except Exception: pass
        ev = d + "/eval"
        info.update(run_dir=d, training_complete=os.path.exists(d + "/training_complete.json"), gate_fail=os.path.exists(d + "/GATE_FAIL"),
                    race_lost=os.path.exists(d + "/RACE_LOST.json"), step=step,
                    n_ms=len(glob.glob(d + "/bb_ms*.pt")),
                    n_val_preds=len(glob.glob(ev + "/val__*_preds.npz")), n_val_rows=len(glob.glob(ev + "/val__*_rows.csv")),
                    best_milestone=(open(ev + "/best_milestone.txt").read().strip() if os.path.exists(ev + "/best_milestone.txt") else None),
                    outputs_sha=os.path.exists(d + "/SHA256SUMS_outputs.txt"),
                    last_hist=[x[:170] for x in tail_lines(d + "/job_history.txt", 2)])
        if step is not None: info["sec_per_step"] = sps
    out.append(info)
out.sort(key=lambda x: x["rank"])
print(json.dumps(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), lane=a.lane, arms=out)))
