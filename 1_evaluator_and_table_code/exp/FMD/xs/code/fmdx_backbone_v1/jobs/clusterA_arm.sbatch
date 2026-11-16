#!/bin/bash
#SBATCH -p gpuidle
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/FMD/xs/logs/arms/%x_%j.out
# FMDX backbone arm, CLUSTER-A copy. Usage: sbatch -J fx_<ARM> [--nice N] clusterA_arm.sbatch <ARM>
# Default lane = gpuidle, one cc >= 8 GPU (preemptible -> requeue-safe; A4000 nodes need --mem <= 48G). 2-GPU layouts (same
# script, used only when sbatch --test-only gives a start before 2026-11-08, lead 3 Nov):
#   sbatch -J fx_<ARM> -p b200idle --gres=gpu:2 --exclude= --cpus-per-task=16 --mem=200G clusterA_arm.sbatch <ARM>
#   sbatch -J fx_<ARM> -p gpuidle --gres=gpu:a100:2 --exclude= --cpus-per-task=16 --mem=240G clusterA_arm.sbatch <ARM>
set -u
export FMDX_SITE=clusterA
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID arm $1 host $(hostname) partition $SLURM_JOB_PARTITION restart ${SLURM_RESTART_COUNT:-0}"
bash $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/arm.sh "$1" clusterA
rc=$?; echo "$(date -u +%FT%TZ) arm.sh exit $rc"; exit $rc
