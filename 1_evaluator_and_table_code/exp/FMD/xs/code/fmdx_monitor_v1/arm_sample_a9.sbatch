#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=<node>
#SBATCH -o $U/exp/FMD/xs/runs/fmdx_monitor/logs/asamp9_%x_%A_%a.out
# FMDX warm monitor A4000-free sampler (prereg A9, sha256 7a71c715...): identical sampling to arm_sample.sbatch (v1) plus
# (1) per (tag, seed) race lock: first copy (any non-MIG GPU type) to start wins, later copies log RACE_LOST and exit 0; lock released on failure;
# (2) MIG devices rejected; (3) device_log.tsv row per pred file (file, gpu name, node, slurm job, utc).
# Usage: sbatch -J fxm_as9_<ARM>_<KIND> -p <part> --gres=gpu:<type>:1 --array=0-2 arm_sample_a9.sbatch <ARM> <bb|ft> <LANE> <MS>
set -uo pipefail; exec 2>&1
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
ARM=$1; KIND=$2; LANE=$3; MS=$4; SEED=${SLURM_ARRAY_TASK_ID:?}
MR=$X/runs/fmdx_monitor; MD=$X/data/monitor_v1; TAG=${ARM}_${KIND}_ms${MS}; OUT=$MR/preds/$ARM/$KIND; mkdir -p $OUT
if [ "$KIND" = bb ]; then CK=$X/runs/${ARM}_${LANE}/bb_ms${MS}.pt; else CK=$MR/ft/${ARM}_FT_${LANE}/bb_ms${MS}.pt; fi
check_code || { echo "$(utc) CODE_SHA_MISMATCH" >> $OUT/job_history.txt; exit 3; }
[ -f $CK ] || { echo "$(utc) CKPT_MISSING $CK" >> $OUT/job_history.txt; exit 5; }
echo "$(utc) start $TAG seed=$SEED slurm=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) gpu=[$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tr '\n' ';')] ckpt_sha16=$(sha256sum $CK | cut -c1-16)" >> $OUT/job_history.txt

nvidia-smi -L | grep -q MIG && { echo "$(utc) MIG_REJECT seed=$SEED slurm=${SLURM_JOB_ID} host=$(hostname)" >> $OUT/job_history.txt; exit 0; }
LK=$OUT/.lock_${TAG}_seed${SEED}; ME=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SEED}
if mkdir $LK 2>/dev/null; then echo $ME > $LK/owner; elif [ "$(cat $LK/owner 2>/dev/null)" != "$ME" ]; then echo "$(utc) RACE_LOST $TAG seed=$SEED slurm=${SLURM_JOB_ID} owner=$(cat $LK/owner 2>/dev/null)" >> $OUT/job_history.txt; exit 0; fi
GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
TBL=$(mktemp /tmp/fxm_sets_XXXXXX)
if [ "$KIND" = ft ]; then
cat >> $TBL <<EOF
test_ouhdl $R/data/ouhdl_e0_v3/e0_test.npz $R/exp/FMD/data/fmd_render224_test.npz -
val1000_ouhdl $R/data/ouhdl_e0_v3/e0_val.npz $R/exp/FMD/data/fmd_render224_val1000.npz SUB1000
EOF
fi
cat >> $TBL <<EOF
test_casia_zh $R/exp/X4a/data/x4a_casia_zh.npz $MD/test_test_casia_zh_render224.npz -
test_casia_en $R/exp/X4a/data/x4a_casia_en.npz $MD/test_test_casia_en_render224.npz -
test_hpl_tamil $R/exp/X4a/data/x4a_hpl_tamil.npz $MD/test_test_hpl_tamil_render224.npz -
test_iam500 $R/data/m5_multiscript_v1/m5_test_iam500.npz $MD/test_test_iam500_render224.npz -
german_unipen_de_test $R/data/d2_v2/unipen_de/e0_test.npz $MD/test_german_unipen_de_test_render224.npz -
german_onhw_all $R/data/d2_v2/onhw/e0_all.npz $MD/test_german_onhw_all_render224.npz -
EOF
while read -r NAME GT RD SUB; do
  o=$OUT/${NAME}__${TAG}__seed${SEED}_preds.npz
  [ -f ${o%.npz}.json ] && continue
  SUBARG=""; [ "$SUB" = SUB1000 ] && SUBARG="--subset 1000 --subset-seed 0"
  [ -n "${SMOKE_UNITS:-}" ] && SUBARG="--subset ${SMOKE_UNITS} --subset-seed 0"   # smoke test only
  $PY $BB/bb_sample.py --ckpt $CK --render $RD --gt-npz $GT $SUBARG --seed $SEED --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 --out $o >> $OUT/sample_seed${SEED}.log 2>&1 \
    || { echo "$(utc) FAILED $NAME seed=$SEED" >> $OUT/job_history.txt; rm -f $TBL; rm -rf $LK; exit 8; }
  echo "$(utc) done $NAME seed=$SEED" >> $OUT/job_history.txt
  printf "%s\t%s\t%s\t%s\t%s\n" "$(basename $o)" "$GPUN" "$(hostname)" "${SLURM_JOB_ID}" "$(utc)" >> $OUT/device_log.tsv
done < $TBL
rm -f $TBL
echo "$(utc) end $TAG seed=$SEED" >> $OUT/job_history.txt
