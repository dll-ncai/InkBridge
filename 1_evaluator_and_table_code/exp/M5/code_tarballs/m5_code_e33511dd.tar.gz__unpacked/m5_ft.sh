#!/bin/bash
# M5 / X6 data-efficiency fine-tunes of the leave-Urdu-out (LUO) model on OUHD-L train subsets 1/10/30/100 %.
# Init: LUO val-selected checkpoint (RUNROOT/luo/eval/selection.json), model + EMA weights; fresh Adam, step 0.
# Budget per arm: FT_STEPS optimiser steps at the E0 recipe (batch 128 x grad-accum 4 = 512, lr 8e-5), 1 GPU per arm,
# checkpoint every FT_SAVE steps; selection = E0 rule on E0's 1,000 random val units (e0_val.npz, subset seed 0);
# evaluation = OUHD-L v3 test (4,406) + gold (354), rendered + dataset input.  Two arms run in parallel per GPU stream.
# Env: TAG, RUNROOT, FT_STEPS (10000), FT_SAVE (1000), EVAL_MAX (0), VAL_SUBSET (1000), ARMS ("p01 p10 p30 p100")
set -uo pipefail
: "${TAG:?}" "${RUNROOT:?}"; FT_STEPS=${FT_STEPS:-10000}; FT_SAVE=${FT_SAVE:-1000}; EVAL_MAX=${EVAL_MAX:-0}
VAL_SUBSET=${VAL_SUBSET:-1000}; ARMS=${ARMS:-"p01 p10 p30 p100"}; INIT_RUN=${INIT_RUN:-$RUNROOT/luo}
source $U/exp/M5/code_${TAG}/m5_lib.sh
gpu_guard; check_code
FR=$RUNROOT/ft; mkdir -p $FR
log_job $FR "ft_job_start steps=$FT_STEPS save=$FT_SAVE arms=[$ARMS] init_run=$INIT_RUN"
[ -f $INIT_RUN/eval/selection.json ] || { log_job $FR "NO_LUO_SELECTION"; exit 1; }
BEST_LUO=$(python3 -c "import json;print(json.load(open('$INIT_RUN/eval/selection.json'))['best_milestone'])")
INIT=$INIT_RUN/checkpoints/checkpoint_$(printf %06d $BEST_LUO).pt
[ -s $INIT ] || { log_job $FR "NO_INIT_CKPT $INIT"; exit 1; }
log_job $FR "init_from $INIT (LUO milestone $BEST_LUO)"
CPUS=${SLURM_CPUS_PER_TASK:-16}
read -r -a GL <<< "$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr ',' ' ')"; NG=${#GL[@]}
EXTRA=(); [ "$EVAL_MAX" != 0 ] && EXTRA=(--max-units $EVAL_MAX)
npz_of() { if [ "$1" = p100 ]; then echo $OU/e0_train.npz; else echo $D/m5_ft_ouhdl_$1.npz; fi; }

one_arm() {  # one_arm ARMTAG GPU PROCS
  local a=$1 g=$2 procs=$3 rd=$FR/$1; local ev=$rd/eval; mkdir -p $ev
  if [ ! -f $rd/training_complete.json ]; then
    train_arm $rd 1 $g --e0-train-npz $(npz_of $a) --condition rendered --run-dir $rd --init-from $INIT \
      --batch-size 128 --gradient-accumulate-every 4 --train-steps $FT_STEPS --save-every $FT_SAVE --validation-every $FT_SAVE \
      --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers 4 \
      --wandb-mode disabled --random-seed 20261016 --sample-count 25 --classes 0 --notes M5_ft_${a}_from_luo_ms${BEST_LUO} \
      || { log_job $rd "TRAINING_FAILED"; return 1; }
  fi
  for m in $(ckpt_ms $rd); do
    CUDA_VISIBLE_DEVICES=$g sample_one $rd $OU/e0_val.npz rendered $m $ev/val__ouhdl1000__ms${m}_preds.npz --subset $VAL_SUBSET --subset-seed 0 || return 2
    score_one $OU/e0_val.npz $ev/val__ouhdl1000__ms${m}_preds.npz m5_ft_${a}_ms${m} rendered $ev/val__ouhdl1000__ms${m}_rows.csv $procs || return 2
  done
  local best; best=$($RUNCPU $MC/m5_select.py --eval-dir $ev --valsets ouhdl1000 --out $ev/selection.json \
       --criterion-note "M5 FT $a: E0 rule on e0_val.npz --subset $VAL_SUBSET --subset-seed 0" | tail -1)
  [[ "$best" =~ ^[0-9]+$ ]] || { log_job $rd "SELECTION_FAILED"; return 2; }
  log_job $rd "selected milestone $best"
  for S in test gold; do for cd in rendered dataset; do
    CUDA_VISIBLE_DEVICES=$g sample_one $rd $OU/e0_${S}.npz $cd $best $ev/ouhdl_${S}__${cd}__ms${best}_preds.npz "${EXTRA[@]}" || return 3
    score_one $OU/e0_${S}.npz $ev/ouhdl_${S}__${cd}__ms${best}_preds.npz m5_ft_${a}_ms${best}_ddim50 $cd $ev/ouhdl_${S}__${cd}__ms${best}_rows.csv $procs || return 3
  done; done
  echo $best > $ev/best_milestone.txt
}
stream() {  # stream GPU PROCS ARM...
  local g=$1 p=$2; shift 2
  for a in "$@"; do one_arm $a $g $p || return 1; done
}
read -r -a AR <<< "$ARMS"
PER=$(( CPUS / NG )); [ $PER -lt 1 ] && PER=1
pids=()
for i in $(seq 0 $((NG - 1))); do
  mine=(); for k in "${!AR[@]}"; do [ $((k % NG)) -eq $i ] && mine+=("${AR[$k]}"); done
  [ ${#mine[@]} -gt 0 ] && { stream ${GL[$i]} $PER "${mine[@]}" & pids+=($!); }
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { log_job $FR "FT_ARM_FAILED"; }
# summary: FT arms + LUO zero-shot + E0 ms8 (comparison rows)
ROWS=()
for a in "${AR[@]}"; do b=$(cat $FR/$a/eval/best_milestone.txt 2>/dev/null || echo NA)
  for S in test gold; do for cd in rendered dataset; do ROWS+=(--row "m5_ft_${a}_ms${b}|ouhdl_$S|$FR/$a/eval/ouhdl_${S}__${cd}__ms${b}_rows.csv"); done; done; done
for S in test gold; do for cd in rendered dataset; do
  ROWS+=(--row "m5_luo_ms${BEST_LUO}_zeroshot|ouhdl_$S|$INIT_RUN/eval/ouhdl_${S}__${cd}__ms${BEST_LUO}_rows.csv")
  ROWS+=(--row "e0_ms8|ouhdl_$S|$B/runs/e0/e0_v3_rendered_100k_bs512/eval/${S}_${cd}_ms8_ddim50_rows.csv"); done; done
$RUNCPU $MC/m5_summary.py "${ROWS[@]}" --out $FR/summary_M5_ft.csv | tail -30 || rc=1
$RUNCPU $MC/m5_provenance.py --m5 $M --runroot $RUNROOT || true
log_job $FR "ft_job_end rc=$rc"
exit $rc
