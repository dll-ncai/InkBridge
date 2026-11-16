#!/bin/bash
#SBATCH -J fmdx_prep_r224
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/FMD/xs/logs/build/prep_renders_%j.out
# 224-px renders for the backbone (prereg A4.2): val250 x 4 LUO sets (gt ink, check64), p01 train (.npy copy of the
# Student-build harness renders, unit order checked), LUO train (80,000 units: rebuilt GT ink where render64 == cond_rendered,
# else decoded seq). Idempotent.
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
check_code || exit 3
mkdir -p $DB; echo "$(utc) start host $(hostname)"
for s in $VALSETS; do
  [ -f $DB/val250_${s}_render224.npz.meta.json ] || $PY $BB/prep_renders.py gtset --npz $D/m5_val_${s}.npz --out $DB/val250_${s}_render224.npz || exit 4
done
[ -f $P01_R224.meta.json ] || $PY $BB/prep_renders.py npz2npy --npz $P01_NPZ --render-npz $R/exp/FMD/data/student/fmd_render224_p01train.npz --out $P01_R224 || exit 5
[ -f $LUO_R224.meta.json ] || $PY $BB/prep_gt_ink.py luo --npz $LUO_NPZ --out $LUO_R224 --procs ${SLURM_CPUS_PER_TASK:-32} || exit 6
(cd $DB && sha256sum *.npy *.npz > SHA256SUMS_data.txt)
echo "$(utc) end"
