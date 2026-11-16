#!/usr/bin/env python3
"""Submit every QUEUED FMDX arm of arms.tsv to CLUSTER-A (prereg A4.3: all arms queued up front) as job 'fx_<arm>' with
nice = 100 + 100 * (initial priority position) and --hold; the controller (gate_controller.py) releases jobs in priority order
within the GPU cap once the arm's inputs exist ('needs'), cancels FAIL FMs, and resolves races. Warm arms first, then scratch.
Records job ids in arm_queue.csv (state SUBMITTED) and one line per arm in arm_log.md and exp/CAND/queue/cand_log.md.
Usage (CLUSTER-A login host): python3 submit_arms.py [--regime warm|scratch|all] [--only ARM,...] [--dry-run] [-- extra sbatch args]
"""
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

X = Path("$U/exp/FMD/xs")
SB = X / "code/fmdx_backbone_v1/jobs/clusterA_arm.sbatch"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="all"); ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    arms = list(csv.DictReader(open(X / "admin/arms.tsv"), delimiter="\t"))
    order = sorted(arms, key=lambda r: (0 if r["regime"] == "warm" else 1, float(r["init_rank"]), r["fm"]))
    qp = X / "admin/arm_queue.csv"
    q = list(csv.DictReader(open(qp))); qcols = list(q[0].keys()); qi = {r["arm_id"]: r for r in q}
    only = set(x for x in a.only.split(",") if x)
    pos = 0
    for r in order:
        pos += 1
        aid = r["arm_id"]
        if r["status"] != "QUEUED" or "clusterA" not in r["lanes"]:
            continue
        if a.regime != "all" and r["regime"] != a.regime:
            continue
        if only and aid not in only:
            continue
        if qi[aid].get("clusterA_job"):
            continue
        nice = 100 + 100 * pos
        cmd = ["sbatch", "--parsable", "-J", f"fx_{aid}", f"--nice={nice}", "--hold"] + a.extra + [str(SB), aid]
        if a.dry_run:
            print(" ".join(cmd)); continue
        jid = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().split(";")[0]
        if not jid.isdigit():
            print("submit failed", aid); continue
        qi[aid].update(state="SUBMITTED", clusterA_job=jid, updated_utc=utc())
        partner = "no CLUSTER-B copy: CLUSTER-B<->VPS hub blocked (VPS key from= ACL, lead 3 Nov)" if "clusterB" in r["lanes"] else "no CLUSTER-B copy: licence lanes CLUSTER-A/VPS only"
        line = f"- {utc()} FMDX arm {aid} ({r['regime']}, fm {r['fm']}): CLUSTER-A job {jid} submitted held, nice {nice}; race partner: {partner}; VPS copy added when the 4090 has no earlier FMDX work\n"
        for f in (X / "admin/arm_log.md", Path("$U/exp/CAND/queue/cand_log.md")):
            with open(f, "a") as fh:
                fh.write(line)
        print(aid, jid, nice)
    if not a.dry_run:
        tmp = qp.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, qcols); w.writeheader(); w.writerows(q)
        os.replace(tmp, qp)


if __name__ == "__main__":
    main()
