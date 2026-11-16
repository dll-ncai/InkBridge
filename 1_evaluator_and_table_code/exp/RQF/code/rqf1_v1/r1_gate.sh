#!/bin/bash
# RQF R1 gate (CLUSTER-A, 1 CPU): waits until all 3-seed preds of JOB exist in R1/preds (CLUSTER-A copy) or arrive from the CLUSTER-B copy via VPS
# (runs\RQF1_CLUSTER-B\preds, sha256 checked against the sidecar), then cancels the still-pending CLUSTER-A sampling copy if the CLUSTER-B copy won, submits the scoring array
# (no dependency) and waits for it to end. Idempotent / requeue-safe.
set -uo pipefail; exec 2>&1
JOB=$1; R=$U; O=$R/exp/RQF/R1; Q=$R/exp/RQF/code/rqf1_v1; T=$O/tasks/$JOB.txt
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }; K=$HOME/.ssh/uhwr_staging_vps_ed25519
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $K Administrator@<ip>"
N=$(wc -l < $T); need=$((N*3))
have() { local n=0 pat; while IFS='|' read -r pat j; do n=$((n+$(ls $O/preds/$pat 2>/dev/null | wc -l)+$(ls $O/preds/${pat%.npz}.json 2>/dev/null | wc -l))); done < $T; echo $n; }
SAMP=$(python3 -c "import json;print(json.load(open('$O/logs/R1_jobs.json'))['sampling']['$JOB'])")
ARR=""; [ -f $O/logs/gate_${JOB}.array ] && ARR=$(cat $O/logs/gate_${JOB}.array)
if [ -z "$ARR" ]; then
  while :; do
    [ "$(have)" -ge $((need*2)) ] && break
    case $JOB in R1c_*|R1a_*|R1b_*) mkdir -p $O/clusterBpull; rm -rf $O/clusterBpull/preds
      timeout 900 $SSHV "tar -cf - -C $VPS_ROOT/uhwr/runs/RQF1_CLUSTER-B preds" 2>/dev/null | tar -xf - -C $O/clusterBpull 2>/dev/null
      if [ -d $O/clusterBpull/preds ]; then
        while IFS='|' read -r pat j; do for f in $O/clusterBpull/preds/$pat; do [ -f "$f" ] || continue; b=$(basename $f); js=${b%.npz}.json
          [ -e $O/preds/$b ] && continue; [ -f $O/clusterBpull/preds/$js ] || continue
          ok=$(python3 - "$f" "$O/clusterBpull/preds/$js" <<'PYE'
import hashlib,json,sys
h=hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest(); print("1" if h==json.load(open(sys.argv[2])).get("preds_sha256") else "0")
PYE
)
          if [ "$ok" = 1 ]; then cp -n $f $O/preds/; cp -n $O/clusterBpull/preds/$js $O/preds/; echo "$(utc) $JOB pulled $b from CLUSTER-B via VPS (sha ok)" >> $O/logs/race_R1.txt; else echo "$(utc) $JOB SHA MISMATCH $b" >> $O/logs/race_R1.txt; fi
        done; done < $T
      fi;; esac
    sleep 600
  done
  # which lane won? (CLUSTER-B files carry host != CLUSTER-A node names; use the race log owner)
  if grep -q "$JOB pulled" $O/logs/race_R1.txt 2>/dev/null; then
    st=$(squeue -h -j $SAMP -o %T 2>/dev/null); if [ "$st" = PENDING ]; then scancel $SAMP; echo "$(utc) [rqf-s1] gate $JOB: CLUSTER-B copy won; CANCEL own losing CLUSTER-A copy $SAMP" | tee -a $R/exp/RQF/queue/rqf_log.md $O/logs/race_R1.txt; fi
  fi
  ARR=$(TASKS=$T sbatch --parsable -J rqf1_score_$JOB --array=1-$N --export=ALL,TASKS $Q/score.sbatch); echo $ARR > $O/logs/gate_${JOB}.array
  echo "$(utc) [rqf-s1, <agent>] SUBMIT R1 scoring $JOB job=$ARR (from gate, preds complete)" >> $R/exp/RQF/queue/rqf_log.md
fi
while sacct -j $ARR -n -X -o State 2>/dev/null | grep -Eq 'PENDING|RUNNING|REQUEUED'; do sleep 300; done
echo "$(utc) gate $JOB done; scoring array $ARR finished"
