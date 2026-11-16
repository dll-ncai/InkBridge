#!/bin/bash
# R2d seed-priority release watcher (<author> 10 Nov: seed 20261034 first). Releases each held seed-2/3 job (20261035/36) once the
# seed-1 job of the same arm has STARTED (sacct RUNNING/COMPLETED); if the seed-1 job ends without starting work (CANCELLED/FAILED/
# TIMEOUT) its siblings stay held and one line is logged. Requeue-safe; exits when nothing is left to release.
L=$U/exp/RQF/queue/rqf_log.md; M=$U/exp/RQF/R2d/admin/release_watch; mkdir -p $M
PAIRS="24970053:24970061,24970069 24970054:24970062,24970070 24970055:24970063,24970071 24970056:24970064,24970072 24970057:24970065,24970073 24970058:24970066,24970074 24970059:24970067,24970075 24970060:24970068,24970076"
while :; do left=0
  for p in $PAIRS; do s1=${p%%:*}; rest=${p#*:}; st=$(sacct -j $s1 -X -n -o State | head -1 | tr -d ' +')
    for j in ${rest//,/ }; do
      squeue -h -j $j -o '%r' 2>/dev/null | grep -q JobHeldUser || continue
      case $st in RUNNING|COMPLETED|COMPLETING) scontrol release $j && echo "$(date -u +%FT%TZ) [R2d release-watcher slurm=${SLURM_JOB_ID:-none}] RELEASE job=$j (seed-1 job $s1 $st)" >> $L;;
        CANCELLED*|FAILED|TIMEOUT|OUT_OF_ME*|NODE_FAIL) [ -f $M/$j ] || { touch $M/$j; echo "$(date -u +%FT%TZ) [R2d release-watcher] KEEP_HELD job=$j (seed-1 job $s1 ended $st)" >> $L; };;
        *) left=1;; esac
    done; done
  [ $left = 0 ] && break; sleep 300; done
echo "$(date -u +%FT%TZ) [R2d release-watcher slurm=${SLURM_JOB_ID:-none}] done" >> $L
