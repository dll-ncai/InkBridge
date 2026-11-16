#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/logs/%x_%j.out
# RQF A2.3 GPU chain step. Usage: sbatch -J <name> -p gpuidle --gres=gpu:1 --exclude=<cc<8 nodes> sm_gpu.sbatch <script.sh> <args...>
set -u; export FMDX_SITE=clusterA
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID host $(hostname) partition $SLURM_JOB_PARTITION restart ${SLURM_RESTART_COUNT:-0} cmd $*"
S=$1; shift; bash $U/exp/RQF/code/sm_v1/$S "$@"; rc=$?; echo "$(date -u +%FT%TZ) exit $rc"; exit $rc
