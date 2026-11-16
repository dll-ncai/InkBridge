#!/bin/bash
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/logs/%x_%j.out
# RQF generic wrapper: sbatch -J <name> [-p ... --gres=...] cpu1.sbatch <script in rqf_v1> <args...>
set -u
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID $* host $(hostname) part $SLURM_JOB_PARTITION restart ${SLURM_RESTART_COUNT:-0}"
S=$1; shift; bash $U/exp/RQF/code/rqf_v1/$S "$@"; rc=$?; echo "$(date -u +%FT%TZ) $S exit $rc"; exit $rc
