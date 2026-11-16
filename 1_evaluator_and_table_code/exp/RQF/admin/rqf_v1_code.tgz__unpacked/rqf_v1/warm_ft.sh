#!/bin/bash
# RQF warm-arm p01/p10/p30 FT (FMDX bb_train.py --regime ft, A4.3(e) recipe via ft_job_rqf.sh) -> val1000 selection -> sampling (bb_sample.py, DDIM-50 eta 0 EMA, seeds 0-2).
# Usage: warm_ft.sh TAG ARM SRC_CKPT SEED FRAC     run dir $QR/TAG, preds $QP/TAG/ft. p01: test_ouhdl + 6 non-Urdu sets; p10/p30: test_ouhdl only (declared).
set -uo pipefail; exec 2>&1
source $U/exp/RQF/code/rqf_v1/rqf_env.sh
TAG=$1; ARM=$2; SRC=$3; SEED=$4; FRAC=$5; LANE=${RQF_LANE:-clusterA}
RUN=$QR/$TAG; mkdir -p $RUN $QP/$TAG/ft; export RQF_RUN=$RUN
case "$SRC" in @*) RN=${SRC#@}; BM=$(cat $QR/$RN/eval/best_milestone.txt 2>/dev/null) || { hist $RUN "SRC_RUN_NOT_SELECTED $RN"; exit 5; }; SRC=$QR/$RN/bb_ms${BM}.pt;; esac
export FT_NPZ=$D/m5_ft_ouhdl_${FRAC}.npz
if [ "$FRAC" = p01 ]; then export FT_R224=$P01_R224; else export FT_R224=$Q/data/render224/${FRAC}train_render224.npy; fi
[ -s $FT_NPZ ] && [ -s $FT_R224 ] || { hist $RUN "MISSING_INPUT $FT_NPZ $FT_R224"; exit 6; }
bash $QC/ft_job_rqf.sh $ARM $LANE $SRC $SEED
if [ ! -f $RUN/eval/best_milestone.txt ]; then ls $RUN/RACE_LOST_${SLURM_JOB_ID:-x}.json >/dev/null 2>&1 && exit 0; hist $RUN "NO_BEST_MILESTONE"; exit 7; fi
BEST=$(cat $RUN/eval/best_milestone.txt); CK=$RUN/bb_ms${BEST}.pt; OUT=$QP/$TAG/ft; GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
TBL=$(mktemp /tmp/rqf_sets_XXXXXX)
cat >> $TBL <<EOF
test_ouhdl $R/data/ouhdl_e0_v3/e0_test.npz $R/exp/FMD/data/fmd_render224_test.npz
EOF
if [ "$FRAC" = p01 ] || [ "$FRAC" = p10 ]; then cat >> $TBL <<EOF
test_casia_zh $R/exp/X4a/data/x4a_casia_zh.npz $MD/test_test_casia_zh_render224.npz
test_casia_en $R/exp/X4a/data/x4a_casia_en.npz $MD/test_test_casia_en_render224.npz
test_hpl_tamil $R/exp/X4a/data/x4a_hpl_tamil.npz $MD/test_test_hpl_tamil_render224.npz
test_iam500 $R/data/m5_multiscript_v1/m5_test_iam500.npz $MD/test_test_iam500_render224.npz
german_unipen_de_test $R/data/d2_v2/unipen_de/e0_test.npz $MD/test_german_unipen_de_test_render224.npz
german_onhw_all $R/data/d2_v2/onhw/e0_all.npz $MD/test_german_onhw_all_render224.npz
EOF
fi
for SD in 0 1 2; do
 while read -r NAME GT RD; do
  o=$OUT/${NAME}__${TAG}_ft_ms${BEST}__seed${SD}_preds.npz; [ -f ${o%.npz}.json ] && continue
  $PY $BB/bb_sample.py --ckpt $CK --render $RD --gt-npz $GT --seed $SD --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 --out $o >> $OUT/sample.log 2>&1 \
    || { hist $RUN "SAMPLING_FAILED $NAME seed=$SD"; rm -f $TBL; exit 8; }
  printf "%s\t%s\t%s\t%s\t%s\n" "$(basename $o)" "$GPUN" "$(hostname)" "${SLURM_JOB_ID:-x}" "$(utc)" >> $OUT/device_log.tsv
 done < $TBL
done
rm -f $TBL; hist $RUN "sampling_done ms$BEST"
