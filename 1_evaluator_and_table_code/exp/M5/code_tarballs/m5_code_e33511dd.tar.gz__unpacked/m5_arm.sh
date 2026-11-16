#!/bin/bash
# M5 / X6 arm job: train (E0 recipe) -> checkpoint selection on the arm's val set -> evaluation.  Requeue-safe:
# training auto-resumes from the newest checkpoint in RUNDIR; every sampling/scoring output is skipped if present.
# Env: TAG (code tag), ARM (joint|luo), RUNROOT, STEPS (100000), SAVE (10000), EVAL_MAX (0 = all units), NGPU (2)
set -uo pipefail
: "${TAG:?}" "${ARM:?}" "${RUNROOT:?}"; STEPS=${STEPS:-100000}; SAVE=${SAVE:-10000}; EVAL_MAX=${EVAL_MAX:-0}; NGPU=${NGPU:-2}
VALIDATION_EVERY=${VALIDATION_EVERY:-1000}
source $U/exp/M5/code_${TAG}/m5_lib.sh
gpu_guard; check_code
RD=$RUNROOT/$ARM; EV=$RD/eval; mkdir -p $EV
log_job $RD "arm_job_start arm=$ARM steps=$STEPS save=$SAVE"
CPUS=${SLURM_CPUS_PER_TASK:-16}
read -r -a GL <<< "$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr ',' ' ')"
NG_AVAIL=${#GL[@]}; [ $NG_AVAIL -lt $NGPU ] && NGPU=$NG_AVAIL
if [ "$ARM" = joint ]; then TRAIN=$D/m5_joint_train.npz; VALSETS=casia_zh,casia_en,hpl_tamil,iam,ouhdl; VALN=200
else TRAIN=$D/m5_luo_train.npz; VALSETS=casia_zh,casia_en,hpl_tamil,iam; VALN=250; fi

# ---------------- phase 1: train (E0 recipe: batch 128 global x grad-accum 4 = 512 per optimiser step, lr 8e-5)
if [ ! -f $RD/training_complete.json ]; then
  GLS=$(IFS=,; echo "${GL[*]:0:$NGPU}")
  train_arm $RD $NGPU $GLS --e0-train-npz $TRAIN --condition rendered --run-dir $RD \
      --batch-size 128 --gradient-accumulate-every 4 --train-steps $STEPS --save-every $SAVE --validation-every $VALIDATION_EVERY \
      --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers 6 \
      --wandb-mode disabled --random-seed 20261016 --sample-count 25 --classes 0 --notes M5_${ARM}_E0recipe \
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

# ---------------- phase 3: evaluation with the selected checkpoint
EVSETS=("ouhdl_test|$OU/e0_test.npz|rendered,dataset" "ouhdl_gold|$OU/e0_gold.npz|rendered,dataset"
        "casia_zh500|$X4/x4a_casia_zh.npz|rendered" "casia_en500|$X4/x4a_casia_en.npz|rendered"
        "hpl_tamil500|$X4/x4a_hpl_tamil.npz|rendered" "iam500|$D/m5_test_iam500.npz|rendered")
EXTRA=(); [ "$EVAL_MAX" != 0 ] && EXTRA=(--max-units $EVAL_MAX)
ejobs=(); for e in "${EVSETS[@]}"; do IFS='|' read -r nm npz conds <<< "$e"; for cd in ${conds//,/ }; do ejobs+=("$nm|$npz|$cd"); done; done
eval_stream() {
  local k=0
  for e in "${ejobs[@]}"; do
    if [ $((k % $2)) -eq $3 ]; then IFS='|' read -r nm npz cd <<< "$e"
      CUDA_VISIBLE_DEVICES=$1 sample_one $RD $npz $cd $BEST $EV/${nm}__${cd}__ms${BEST}_preds.npz "${EXTRA[@]}" || return 5
    fi; k=$((k + 1))
  done
}
pids=(); for i in $(seq 0 $((NG_AVAIL - 1))); do eval_stream ${GL[$i]} $NG_AVAIL $i & pids+=($!); done
for p in "${pids[@]}"; do wait $p || { log_job $RD "EVAL_SAMPLING_FAILED"; exit 2; }; done
ROWS=()
for e in "${ejobs[@]}"; do IFS='|' read -r nm npz cd <<< "$e"
  score_one $npz $EV/${nm}__${cd}__ms${BEST}_preds.npz m5_${ARM}_ms${BEST}_ddim50 $cd $EV/${nm}__${cd}__ms${BEST}_rows.csv $CPUS || { log_job $RD "EVAL_SCORING_FAILED"; exit 2; }
  ROWS+=(--row "m5_${ARM}_ms${BEST}|$nm|$EV/${nm}__${cd}__ms${BEST}_rows.csv")
done
# comparison rows (existing, read-only): E0 ms8 final eval (Urdu) and X4a zero-shot E0 ms8 (CASIA zh/en, Tamil)
E0EV=$B/runs/e0/e0_v3_rendered_100k_bs512/eval
for S in test gold; do for cd in rendered dataset; do ROWS+=(--row "e0_ms8|ouhdl_$S|$E0EV/${S}_${cd}_ms8_ddim50_rows.csv"); done; done
for s in casia_zh casia_en hpl_tamil; do ROWS+=(--row "e0_ms8_x4a|${s}500|$B/exp/X4a/out/${s}_rendered_ms8_ddim50_rows.csv"); done
$RUNCPU $MC/m5_summary.py "${ROWS[@]}" --out $EV/summary_M5_${ARM}.csv | tail -25 || { log_job $RD "SUMMARY_FAILED"; exit 2; }
$RUNCPU $MC/m5_provenance.py --m5 $M --runroot $RUNROOT || true
log_job $RD "arm_job_end"
