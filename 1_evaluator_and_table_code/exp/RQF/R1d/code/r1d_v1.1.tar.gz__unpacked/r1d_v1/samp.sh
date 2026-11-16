#!/bin/bash
# R1d CLUSTER-A sampling copy. Race: (1) local atomic mkdir between the two CLUSTER-A copies; (2) cross-cluster lock on the VPS hub vs the CLUSTER-B copy
# (owner 'clusterA'|'clusterB'; first to START wins; VPS unreachable -> CLUSTER-A proceeds, logged). Winner scancels its own CLUSTER-A sibling; a loser exits 0.
# Requeue-safe (owner re-enters; samplers skip finished files).  Usage: samp.sh UNIT KIND ARGS...  (see run_sampler.sh)
set -uo pipefail; exec 2>&1
source $U/exp/RQF/R1d/code/r1d_v1/r1d_env.sh   # takes the VPS lock (v1.1 hook)
UNIT=$1; KIND=$2; shift 2; L=$RD/race/$UNIT; mkdir -p $RD/race
if mkdir $L 2>/dev/null; then echo "$SLURM_JOB_ID $(utc) $(hostname)" > $L/owner
else sleep 3; own=$(cut -d' ' -f1 $L/owner 2>/dev/null); [ "$own" = "$SLURM_JOB_ID" ] || { echo "$(utc) LOSE $UNIT owner=$own"; exit 0; }; fi
for i in $(seq 1 60); do [ -s $RD/race/$UNIT.ids ] && break; sleep 5; done
for j in $(cat $RD/race/$UNIT.ids 2>/dev/null); do
  if [ "$j" != "$SLURM_JOB_ID" ] && [ ! -e $L/cancelled_$j ]; then
    /usr/bin/scancel $j && touch $L/cancelled_$j && r1d_log "CANCEL job=$j (own losing race copy of R1d unit $UNIT; winner $SLURM_JOB_ID started on $(hostname))"; fi
done
GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(utc) start $UNIT kind=$KIND job=$SLURM_JOB_ID host=$(hostname) gpu=[$GPUN] restart=${SLURM_RESTART_COUNT:-0}" >> $RD/logs/samp_history.txt
bash $RC/run_sampler.sh clusterA $KIND "$@"; rc=$?
echo "$(utc) end $UNIT rc=$rc job=$SLURM_JOB_ID gpu=[$GPUN]" >> $RD/logs/samp_history.txt; exit $rc
