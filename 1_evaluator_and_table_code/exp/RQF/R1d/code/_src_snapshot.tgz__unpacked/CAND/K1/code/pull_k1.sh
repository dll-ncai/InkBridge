#!/bin/bash
# CAND K1: pull VPS $VPS_ROOT\uhwr\runs\CAND\K1 (+ runner log/meta of job 500_cand_k1_frames_luo_ms6, jobs.csv) to CLUSTER-A exp/CAND/K1.
# Run on <node> with: setsid nohup bash pull_k1.sh > /dev/null 2>&1 < /dev/null &   (log: exp/CAND/K1/logs/pull_k1.log)
set -u
U=$U; K=$U/exp/CAND/K1; L=$K/logs/pull_k1.log; mkdir -p $K/logs $K/preds $K/vps_runner
SSH="ssh -i $HOME/.ssh/uhwr_staging_vps_ed25519 -o BatchMode=yes -o ConnectTimeout=30 Administrator@<ip>"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(utc) pull start" >> $L
T=$K/.pull_tmp_$$; mkdir -p $T
$SSH "tar -cf - -C $VPS_ROOT/uhwr/runs/CAND/K1 --exclude=*.part.npz preds" | tar -xf - -C $T; rc1=$?
# top-level files separately: the sampler holds LUO_ms6_frame_sample_log.jsonl open while running (bsdtar then aborts)
$SSH "tar -cf - -C $VPS_ROOT/uhwr/runs/CAND/K1 job_history.txt 500_cand_k1_frames_luo_ms6.cmd LUO_ms6_frame_sample_log.jsonl" | tar -xf - -C $T; rc3=$?
$SSH "tar -cf - -C $VPS_ROOT/uhwr/runner logs/500_cand_k1_frames_luo_ms6.log logs/500_cand_k1_frames_luo_ms6.log.err logs/500_cand_k1_frames_luo_ms6.meta.txt jobs.csv" | tar -xf - -C $K/vps_runner; rc2=$?
echo "$(utc) tar rc=$rc1/$rc2/$rc3 files=$(find $T -type f | wc -l)" >> $L
cp -p $T/preds/*_preds.npz $T/preds/*_preds.json $K/preds/ 2>>$L
cp -p $T/*.jsonl $T/job_history.txt $T/*.cmd $K/ 2>>$L
rm -rf $T
# verify every pulled preds file against the sha256 written by the sampler into its sidecar
python3 - <<'PY' >> $L 2>&1
import json, glob, hashlib, os
K = "$U/exp/CAND/K1"; U = "$U"
ok = bad = 0; lines = []
for f in sorted(glob.glob(K + "/preds/*_preds.npz")):
    j = f[:-4] + ".json"
    if not os.path.exists(j): print("pending (no sidecar yet)", os.path.basename(f)); continue
    s = json.load(open(j))
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    if h == s.get("preds_sha256"): ok += 1
    else: bad += 1; print("SHA MISMATCH", f, h, s.get("preds_sha256"))
    lines.append(f"{h}  {os.path.relpath(f, U)}")
    # additive sidecar field for CLUSTER-A scoring (lane_score.py reads npz_clusterA); original VPS sidecar kept as .json.vps_orig
    if "npz_clusterA" not in s:
        os.replace(j, j + ".vps_orig"); s["npz_clusterA"] = os.path.join(U, s["npz_rel"])
        s["npz_clusterA_note"] = "added on CLUSTER-A by exp/CAND/K1/code/pull_k1.sh; original sidecar = .json.vps_orig"
        json.dump(s, open(j, "w"), indent=1)
open(K + "/SHA256SUMS_preds.txt", "w").write("\n".join(lines) + "\n")
print("verify preds ok=%d bad=%d" % (ok, bad))
PY
echo "$(utc) pull done" >> $L
