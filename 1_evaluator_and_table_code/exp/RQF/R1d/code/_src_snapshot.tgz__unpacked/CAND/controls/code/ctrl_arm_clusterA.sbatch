#!/bin/bash
#SBATCH --job-name=cand_ctrl
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:a30:1
#SBATCH --exclude=gpu[001-003]
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=$U/exp/CAND/controls/logs/%x_%j.out
# CAND control arm on one full A30 (not a MIG slice; <node>-003 excluded). Requeue-safe (frame_arm.sh resumes; finished files skipped).
#   sbatch -J cand_ctrl_luoft_s1 --export=ALL,ARM=ctrl_luoft_p01_s1,FRAC=p01,FRAME=identity,REPLAY=0,TRAIN_SEED=20261017,SEEDS="0 1 2",EXTRA_SETS="..." ctrl_arm_clusterA.sbatch
set -uo pipefail; exec 2>&1
echo "host $(hostname) job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} arm ${ARM:-?} start $(date -Is)"
nvidia-smi --query-gpu=name,compute_cap,memory.total,compute_mode --format=csv,noheader
bash $U/exp/CAND/controls/code/ctrl_arm.sh
