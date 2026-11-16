#!/bin/bash
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH -o $U/exp/FMD/xs/runs/fmdx_monitor/logs/sel_%x_%j.out
# FMDX warm monitor: stalled in-job selection (arm.sh block 'milestone selection', copied unchanged in logic: val250 x 4 LUO sets x ms1-4, bb_sample.py seed 0 DDIM-50 eta 0 EMA; e0_score; m5_select LUO rule)
# for an arm whose training is complete but whose controller job is PENDING. Usage: sbatch -J fxm_sel_<ARM> -p llms --gres=gpu:rtx6000:1 sel_job.sh ARM ORIG_JOBID
# Collision guard: exits if ORIG_JOBID is RUNNING; otherwise holds ORIG_JOBID (scontrol hold, reversible) so that only one process writes eval/.
set -u; exec 2>&1
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
ARM=$1; ORIG=$2; LANE=clusterA; RUN=$X/runs/${ARM}_${LANE}; EV=$RUN/eval; mkdir -p $EV
check_code || { hist $RUN "SEL_CODE_SHA_MISMATCH"; exit 3; }
[ -f $RUN/training_complete.json ] || { hist $RUN "SEL_NO_TRAINING_COMPLETE"; exit 5; }
[ -f $EV/best_milestone.txt ] && { hist $RUN "SEL_ALREADY_DONE"; exit 0; }
if squeue -h -j $ORIG -t R 2>/dev/null | grep -q .; then hist $RUN "SEL_ORIG_RUNNING $ORIG: monitor copy exits"; exit 0; fi
scontrol hold $ORIG 2>/dev/null; hist $RUN "SEL_monitor_copy start slurm=${SLURM_JOB_ID:-x}; held controller job $ORIG (scontrol hold; released by the monitor after selection)"
MS=$(ls $RUN/bb_ms*.pt | sed 's/.*bb_ms\([0-9]*\)\.pt/\1/' | sort -n)
for m in $MS; do for s in $VALSETS; do
  o=$EV/val__${s}__ms${m}_preds.npz
  [ -s $o ] || $PY $BB/bb_sample.py --ckpt $RUN/bb_ms${m}.pt --render $DB/val250_${s}_render224.npz --gt-npz $D/m5_val_${s}.npz --subset 250 --subset-seed 0 --seed 0 --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 --out $o >> $EV/sample.log 2>&1 || { hist $RUN "SEL_VAL_SAMPLING_FAILED"; exit 8; }
done; done
CPUS=${SLURM_CPUS_PER_TASK:-8}
for m in $MS; do for s in $VALSETS; do r=$EV/val__${s}__ms${m}_rows.csv
  [ -s $r ] && continue
  $PY $C0/e0_score.py e0 --npz $D/m5_val_${s}.npz --preds $EV/val__${s}__ms${m}_preds.npz --method fmdx_${ARM}_ms${m} --input rendered --out ${r%.csv}.part.csv --procs $CPUS >> $EV/score.log 2>&1 && mv ${r%.csv}.part.csv $r || { hist $RUN "SEL_VAL_SCORING_FAILED $r"; exit 9; }
done; done
BEST=$($PY $BB/vendor/m5_select.py --eval-dir $EV --valsets ${VALSETS// /,} --out $EV/selection.json --criterion-note "FMDX $ARM: LUO rule, val250 x {${VALSETS}}, bb_sample.py --subset 250 --subset-seed 0 --seed 0, DDIM-50 eta 0, EMA (monitor copy of the arm.sh selection block)" | tail -1)
[[ "$BEST" =~ ^[0-9]+$ ]] || { hist $RUN "SEL_SELECTION_FAILED"; exit 10; }
echo $BEST > $EV/best_milestone.txt; hist $RUN "selected milestone $BEST (monitor copy sel_job.sh)"
(cd $RUN && sha256sum bb_ms*.pt eval/selection.json eval/*_preds.npz run_config*.json training_complete.json > SHA256SUMS_outputs.txt)
scontrol release $ORIG 2>/dev/null; hist $RUN "SEL_monitor_copy end; released controller job $ORIG"
