#!/bin/bash
# SM-A one run, end to end on one GPU: sma_train.py (frozen backbone + 1 LoRA r16 on the D2 projection set, target subset only,
# 10,000 steps, ckpt every 1,000) -> val1000 milestone selection (E0 rule: lowest native LDTW, e0_val.npz --subset 1000 --subset-seed 0,
# seed 0, DDIM-50 eta 0, EMA adapter, adapter ON for every val unit) -> test sampling seeds 0-2 on the 7 sets with the D3 router gate
# (adapter on iff p_rtl >= 0.5).  Auto-resumes (training from latest checkpoint; sampling/scoring skip finished files).
# Usage: sma_chain.sh TAG KIND BACKBONE FRAC SEED      KIND = luo | bb
set -uo pipefail; exec 2>&1
source $U/exp/RQF/code/sm_v1/sm_env.sh
TAG=$1; KIND=$2; BBK=$3; FRAC=$4; SEED=$5
RUN=$QR/$TAG; EV=$RUN/eval; OUT=$QP/$TAG/ft; mkdir -p $EV $OUT
STEPS=${SM_SMOKE_STEPS:-10000}; SAVE=${SM_SMOKE_STEPS:-1000}; NMS=$((STEPS / SAVE))
PYS="$WRAP env PYTHONPATH=$SMC:$PYP $VENV/bin/python"
hist $RUN "sma_chain_start tag=$TAG kind=$KIND backbone=$BBK frac=$FRAC seed=$SEED steps=$STEPS"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
awk -v c="$cc" 'BEGIN{exit !(c+0 >= 8.0)}' || { hist $RUN "GPU_CC_LT_8 $cc"; exit 4; }
NPZ=$D/m5_ft_ouhdl_${FRAC}.npz; RA=()
if [ $KIND = bb ]; then if [ $FRAC = p01 ]; then R224=$P01_R224; else R224=$Q/data/render224/${FRAC}train_render224.npy; fi; RA=(--render-npy $R224); fi
[ -s $NPZ ] && [ -s $BBK ] || { hist $RUN "MISSING_INPUT $NPZ $BBK"; exit 6; }
MEMG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '); MEMG=$((MEMG / 1024))
if [ $MEMG -ge 20 ]; then MB=64; AC=4; else MB=32; AC=8; fi    # 256 units per optimiser step (D2 recipe) on every GPU
if [ ! -f $RUN/training_complete.json ]; then
  hist $RUN "train_start layout=${MB}x${AC}"
  $PYS $SMC/sma_train.py --backbone-kind $KIND "${RA[@]}" --stage 1 --run-dir $RUN --backbone $BBK --backbone-weights ema \
    --train-npz $NPZ --scripts ouhdl --script-override ouhdl --n-experts 1 --rank 16 --lora-alpha 16 --lora-targets cross,attn,linear \
    --gate-hidden 128 --gate-feat deepest --lb-coef 0 --batch-size $MB --grad-accum $AC --train-steps $STEPS --save-every $SAVE \
    --val-every $((STEPS * 10)) --learning-rate 1e-4 --keep-last 0 --num-workers 4 --seed $SEED --notes "RQF A2.3 SM-A $TAG" \
    > $RUN/train_${SLURM_JOB_ID:-x}_${SLURM_RESTART_COUNT:-0}.log 2>&1
  rc=$?; hist $RUN "train_exit rc=$rc"; [ -f $RUN/training_complete.json ] || { tail -20 $RUN/train_*_${SLURM_RESTART_COUNT:-0}.log; exit 7; }
fi
VR=(); [ $KIND = bb ] && VR=(--render $R/exp/FMD/data/fmd_render224_val1000.npz)
for m in $(seq 1 $NMS); do o=$EV/val__ouhdl1000__ms${m}_preds.npz; [ -s $o ] && continue
  $PYS $SMC/sma_sample.py --kind $KIND --ckpt $RUN/checkpoints/checkpoint_$(printf %06d $m).pt --backbone $BBK --gt-npz $R/data/ouhdl_e0_v3/e0_val.npz "${VR[@]}" \
    --set-name val1000_ouhdl --route all --subset 1000 --subset-seed 0 --seed 0 --batch-size 64 --out $o >> $EV/sample.log 2>&1 || { hist $RUN "VAL_SAMPLING_FAILED ms$m"; exit 8; }
done
for m in $(seq 1 $NMS); do r=$EV/val__ouhdl1000__ms${m}_rows.csv; [ -s $r ] && continue
  $PY $C0/e0_score.py e0 --npz $R/data/ouhdl_e0_v3/e0_val.npz --preds $EV/val__ouhdl1000__ms${m}_preds.npz --method sma_${TAG}_ms${m} --input rendered \
    --out ${r%.csv}.part.csv --procs ${SLURM_CPUS_PER_TASK:-8} >> $EV/score.log 2>&1 && mv ${r%.csv}.part.csv $r || { hist $RUN "VAL_SCORING_FAILED ms$m"; exit 9; }
done
BEST=$($PY $BB/vendor/m5_select.py --eval-dir $EV --valsets ouhdl1000 --out $EV/selection.json \
  --criterion-note "RQF SM-A $TAG: E0 rule on val1000 (e0_val.npz --subset 1000 --subset-seed 0), sma_sample.py seed 0 route=all, DDIM-50 eta 0, EMA adapter" | tail -1)
[[ "$BEST" =~ ^[0-9]+$ ]] || { hist $RUN SELECTION_FAILED; exit 10; }
echo $BEST > $EV/best_milestone.txt; hist $RUN "selected SM-A milestone $BEST (step $((BEST * SAVE)))"
[ -n "${SM_SMOKE_STEPS:-}" ] && SETS7=${SM_SMOKE_SETS:-$SETS7}
CK=$RUN/checkpoints/checkpoint_$(printf %06d $BEST).pt; GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
for SD in ${SM_SMOKE_SEEDS:-0 1 2}; do for NAME in $SETS7; do
  o=$OUT/${NAME}__${TAG}_ft_ms${BEST}__seed${SD}_preds.npz; [ -f ${o%.npz}.json ] && continue
  RR=(); [ $KIND = bb ] && RR=(--render $(r224_of $NAME))
  $PYS $SMC/sma_sample.py --kind $KIND --ckpt $CK --backbone $BBK --gt-npz $(gt_of $NAME) "${RR[@]}" --set-name $NAME --route $ROUT/manifest_${NAME}.csv \
    --seed $SD --batch-size 64 ${SM_SMOKE_MAXU:+--max-units $SM_SMOKE_MAXU} --out $o >> $OUT/sample.log 2>&1 || { hist $RUN "SAMPLING_FAILED $NAME seed=$SD"; exit 8; }
  printf "%s\t%s\t%s\t%s\t%s\n" "$(basename $o)" "$GPUN" "$(hostname)" "${SLURM_JOB_ID:-x}" "$(utc)" >> $OUT/device_log.tsv
done; done
(cd $RUN && sha256sum checkpoints/*.pt eval/selection.json run_config*.json training_complete.json > SHA256SUMS_outputs.txt)
hist $RUN "sampling_done ms$BEST"; touch $RUN/SAMPLING_DONE
