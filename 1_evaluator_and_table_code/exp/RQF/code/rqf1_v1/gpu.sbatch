#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/R1/logs/%x_%j.out
# usage: sbatch -J rqf1_<JOB> -p gpuidle --gres=gpu:a4000:1 gpu.sbatch <JOB>
set -u
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID $* host $(hostname) part $SLURM_JOB_PARTITION restart ${SLURM_RESTART_COUNT:-0}"
bash $U/exp/RQF/code/rqf1_v1/r1_job.sh "$@"; rc=$?; echo "$(date -u +%FT%TZ) exit $rc"; exit $rc
