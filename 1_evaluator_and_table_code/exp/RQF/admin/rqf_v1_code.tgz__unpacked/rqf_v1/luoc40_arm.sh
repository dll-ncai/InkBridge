#!/bin/bash
# RQF LUOC40 copy of m5_arm.sh phases 1-2 (+ --init-from LUO ms6 model+EMA, fresh optimiser, step 0; steps/save via env). M5 / X6 arm job: train (E0 recipe) -> checkpoint selection on the arm's val set -> evaluation.  Requeue-safe:
# training auto-resumes from the newest checkpoint in RUNDIR; every sampling/scoring output is skipped if present.
# Env: TAG (code tag), ARM (joint|luo), RUNROOT, STEPS (100000), SAVE (10000), EVAL_MAX (0 = all units), NGPU (2)
set -uo pipefail
: "${TAG:?}" "${ARM:?}" "${RUNROOT:?}"; STEPS=${STEPS:-100000}; SAVE=${SAVE:-10000}; EVAL_MAX=${EVAL_MAX:-0}; NGPU=${NGPU:-2}
VALIDATION_EVERY=${VALIDATION_EVERY:-1000}
source $U/exp/M5/code_${TAG}/m5_lib.sh
gpu_guard; check_code
# RQF local race guard (same-filesystem copies): the first copy to start owns the run dir; later copies exit 0 while the owner is RUNNING.
RDL=$RUNROOT/$ARM; mkdir -p $RDL
if [ ! -f $RDL/training_complete.json ]; then
  ME=${SLURM_JOB_ID:-x}
  if mkdir $RDL/.local_lock 2>/dev/null; then echo $ME > $RDL/.local_owner
  else OWN=$(cat $RDL/.local_owner 2>/dev/null)
    if [ "$OWN" != "$ME" ]; then
      if squeue -h -j "$OWN" -t R,CF,CG 2>/dev/null | grep -q .; then echo "$(date -u +%FT%TZ) RACE_LOST $ME owner=$OWN" >> $RDL/job_history.txt; exit 0; fi
      echo $ME > $RDL/.local_owner
    fi
  fi
fi
RD=$RUNROOT/$ARM; EV=$RD/eval; mkdir -p $EV
R_LUOCK=$U/exp/M5/runs/luo/checkpoints/checkpoint_000006.pt   # LUO ms6 (sha256 ab6b178d...)
log_job $RD "arm_job_start arm=$ARM steps=$STEPS save=$SAVE"
CPUS=${SLURM_CPUS_PER_TASK:-16}
# RQF layout: 128 global x accum 4 = 512 sequences/step (LUO recipe; 2 procs split_batches). 1 GPU < 40 GB: 32 x accum 16 (same 512 sequences per optimiser step)
MEMG0=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d " "); MEMG0=$((MEMG0 / 1024)); BS=128; GA=4
read -r -a GL <<< "$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr ',' ' ')"
NG_AVAIL=${#GL[@]}; [ $NG_AVAIL -lt $NGPU ] && NGPU=$NG_AVAIL
if [ "$ARM" = joint ]; then TRAIN=$D/m5_joint_train.npz; VALSETS=casia_zh,casia_en,hpl_tamil,iam,ouhdl; VALN=200
else TRAIN=$D/m5_luo_train.npz; VALSETS=casia_zh,casia_en,hpl_tamil,iam; VALN=250; fi

# ---------------- phase 1: train (E0 recipe: batch 128 global x grad-accum 4 = 512 per optimiser step, lr 8e-5)
if [ ! -f $RD/training_complete.json ]; then
  GLS=$(IFS=,; echo "${GL[*]:0:$NGPU}")
if [ "$NGPU" = 1 ] && [ $MEMG0 -lt 70 ]; then BS=32; GA=16; fi
log_job $RD "rqf_batch_layout ngpu=$NGPU mem_gb=$MEMG0 batch=$BS accum=$GA (512 sequences per step)"
  train_arm $RD $NGPU $GLS --e0-train-npz $TRAIN --condition rendered --run-dir $RD \
      --batch-size $BS --gradient-accumulate-every $GA --train-steps $STEPS --save-every $SAVE --validation-every $VALIDATION_EVERY \
      --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers 6 \
      --init-from $R_LUOCK --wandb-mode disabled --random-seed 20261016 --sample-count 25 --classes 0 --notes M5_${ARM}_E0recipe \
    || { log_job $RD "TRAINING_FAILED"; exit 1; }
fi
log_job $RD "training_complete $(cat $RD/training_complete.json | tr -d '\n ')"

# ---------------- phase 2: checkpoint selection (val units of the arm's training scripts only)
MS=$(ckpt_ms $RD)
jobs_list=(); for m in $MS; do for s in ${VALSETS//,/ }; do jobs_list+=("$m:$s"); done; done
gpu_stream() {  # gpu_stream GPU_INDEX STRIDE OFFSET
  local k=0
  for js in "${jobs_list[@]}"; do
    if [ $((k % $2)) -eq $3 ]; then m=${js%%:*}; s=${js#*:}
      CUDA_VISIBLE_DEVICES=$1 sample_one $RD $D/m5_val_$s.npz rendered $m $EV/val__${s}__ms${m}_preds.npz --subset $VALN --subset-seed 0 || return 5
    fi; k=$((k + 1))
  done
}
pids=(); for i in $(seq 0 $((NG_AVAIL - 1))); do gpu_stream ${GL[$i]} $NG_AVAIL $i & pids+=($!); done
for p in "${pids[@]}"; do wait $p || { log_job $RD "VAL_SAMPLING_FAILED"; exit 2; }; done
for js in "${jobs_list[@]}"; do m=${js%%:*}; s=${js#*:}
  score_one $D/m5_val_$s.npz $EV/val__${s}__ms${m}_preds.npz m5_${ARM}_ms${m} rendered $EV/val__${s}__ms${m}_rows.csv $CPUS || { log_job $RD "VAL_SCORING_FAILED"; exit 2; }
done
BEST=$($RUNCPU $MC/m5_select.py --eval-dir $EV --valsets $VALSETS --out $EV/selection.json \
       --criterion-note "M5 $ARM: ${VALN} val units per script ($VALSETS), e0_sample --subset $VALN --subset-seed 0" | tail -1)
[[ "$BEST" =~ ^[0-9]+$ ]] || { log_job $RD "SELECTION_FAILED"; exit 2; }
log_job $RD "selected milestone $BEST"

log_job $RD "arm_phase12_end selected=$BEST"
