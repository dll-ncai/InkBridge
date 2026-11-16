#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/logs/%x_%j.out
# RQF R2a second backbone seed (20261017) of NOFM_W / PEINIT_W (FMDX WARM, 40k steps, 2 GPU). Usage: sbatch -J rqf_bb2_<ARM> -p gpuidle --gres=gpu:<type>:2 --exclude=... bb2_arm.sbatch <ARM2>
set -u; export FMDX_SITE=clusterA
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID arm $1 host $(hostname) partition $SLURM_JOB_PARTITION restart ${SLURM_RESTART_COUNT:-0}"
bash $U/exp/RQF/code/rqf_v1/arm_rqf.sh "$1" clusterA
rc=$?; echo "$(date -u +%FT%TZ) arm_rqf.sh exit $rc"; exit $rc
