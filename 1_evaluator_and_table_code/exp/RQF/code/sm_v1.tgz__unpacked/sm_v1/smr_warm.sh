#!/bin/bash
# SM-R on a WARM (224-px encoder) backbone: copy of rqf_v1/warm_ft.sh with two changes: (a) ft_job_smr.sh adds the 50/50 replay mix
# (bb_train.py --pseudo-npz m5_luo_train.npz --pseudo-render-npy luo_train_render224.npy --pseudo-frac 0.5 = fixed 50 % share per batch,
# drawn uniformly by RankBatchSampler, seed = training seed + step), (b) test sampling on all 7 sets at every fraction (A2.3).
# Usage: smr_warm.sh TAG ARM SRC_CKPT SEED FRAC
set -uo pipefail; exec 2>&1
source $U/exp/RQF/code/sm_v1/sm_env.sh
TAG=$1; ARM=$2; SRC=$3; SEED=$4; FRAC=$5; LANE=${RQF_LANE:-clusterA}
RUN=$QR/$TAG; mkdir -p $RUN $QP/$TAG/ft; export RQF_RUN=$RUN
export FT_NPZ=$D/m5_ft_ouhdl_${FRAC}.npz
if [ "$FRAC" = p01 ]; then export FT_R224=$P01_R224; else export FT_R224=$Q/data/render224/${FRAC}train_render224.npy; fi
[ -s $FT_NPZ ] && [ -s $FT_R224 ] && [ -s $SRC ] || { hist $RUN "MISSING_INPUT $FT_NPZ $FT_R224 $SRC"; exit 6; }
bash $SMC/ft_job_smr.sh $ARM $LANE $SRC $SEED || exit $?
[ -f $RUN/eval/best_milestone.txt ] || { ls $RUN/RACE_LOST_* >/dev/null 2>&1 && exit 0; hist $RUN NO_BEST_MILESTONE; exit 7; }
BEST=$(cat $RUN/eval/best_milestone.txt); CK=$RUN/bb_ms${BEST}.pt; OUT=$QP/$TAG/ft; GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
for SD in 0 1 2; do for NAME in $SETS7; do
  o=$OUT/${NAME}__${TAG}_ft_ms${BEST}__seed${SD}_preds.npz; [ -f ${o%.npz}.json ] && continue
  $PY $BB/bb_sample.py --ckpt $CK --render $(r224_of $NAME) --gt-npz $(gt_of $NAME) --seed $SD --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 --out $o >> $OUT/sample.log 2>&1 \
    || { hist $RUN "SAMPLING_FAILED $NAME seed=$SD"; exit 8; }
  printf "%s\t%s\t%s\t%s\t%s\n" "$(basename $o)" "$GPUN" "$(hostname)" "${SLURM_JOB_ID:-x}" "$(utc)" >> $OUT/device_log.tsv
done; done
hist $RUN "sampling_done ms$BEST"; touch $RUN/SAMPLING_DONE
