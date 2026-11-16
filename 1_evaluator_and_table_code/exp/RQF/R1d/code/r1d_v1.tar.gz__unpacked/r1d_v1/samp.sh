#!/bin/bash
# R1d sampling unit with a two-copy race (first copy to START wins: it takes an atomic mkdir lock and scancels its own other copy;
# a copy that finds the lock owned by another job exits 0). Requeue-safe: the owner re-enters; the samplers skip finished files.
# Usage: samp.sh UNIT e0 LABEL RUNDIR MS OUTDIR FRAMES SET...      |  samp.sh UNIT bb LABEL CKPT OUTDIR FRAMES SET:RENDER:GT ...
set -uo pipefail; exec 2>&1
source $U/exp/RQF/R1d/code/r1d_v1/r1d_env.sh
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
if [ $KIND = e0 ]; then
  LABEL=$1; RUN=$2; MS=$3; OUT=$4; FR=${5//,/ }; shift 5
  apptainer exec --nv -B $SCRATCH_ROOT $IMG env $ENVB PYTHONPATH=$FCODE/lib:$REPO/src:$FCODE OMP_NUM_THREADS=4 $VE/bin/python $RC/rqf1_sample.py \
    --label $LABEL --run-dir $RUN --milestone $MS --stage $R --out-dir $OUT --sets "$@" --frames $FR --seeds 0 1 2 --batch-size 256 2>&1 | grep -av "sampling loop" | tail -30
  rc=${PIPESTATUS[0]}
else
  LABEL=$1; CK=$2; OUT=$3; FR=${4//,/ }; shift 4; rc=0
  for t in "$@"; do IFS=: read S RN GT <<< "$t"
    apptainer exec --nv -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB BB_DIR=$BB PYTHONPATH=$REPO/src:$BB:$BB/vendor:$HC:$C0:$UE:$TIMMDEPS:$FCODE OMP_NUM_THREADS=4 $VE/bin/python \
      $RC/bb_frame_sample.py --ckpt $CK --label $LABEL --set $S --render $RN --render-key img224 --gt-npz $GT --frames $FR --seeds 0 1 2 --batch-size 250 --out-dir $OUT 2>&1 | grep -av "sampling loop" | tail -8
    r=${PIPESTATUS[0]}; [ $r -ne 0 ] && rc=$r
  done
fi
echo "$(utc) end $UNIT rc=$rc job=$SLURM_JOB_ID gpu=[$GPUN]" >> $RD/logs/samp_history.txt; exit $rc
