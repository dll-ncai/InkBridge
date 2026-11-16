#!/bin/bash
#SBATCH --job-name=frame_k1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=$U/exp/FRAME/logs/%x_%j.out
# K1 frame kill check on CLUSTER-A (A30; do not name the gpu partition, --gres routes it).  Code: extract frame_code_<TAG>.tar.gz to
# $B/exp/FRAME/code_<TAG>.  Apptainer + PYTHONPATH exactly as experiments/e0/jobs/e0_eval_final.sh, plus $C/lib (uhwr_eval 0.3.1).
set -uo pipefail; exec 2>&1
TAG=5dd59e88
B=$U; IMG=$B/images/torch27.sif; VENV=$B/venvs/e0; REPO=$B/repos/nagamatsu
export CODE=$B/exp/FRAME/code_$TAG STAGE=$B OUT=$B/exp/FRAME/k1 SEEDS="0 1 2" PROCS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p $B/exp/FRAME/logs $OUT
export PYG="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$CODE/lib:$REPO/src:$CODE OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV/bin/python"
export PYC="apptainer exec -B $SCRATCH_ROOT $IMG env PYTHONPATH=$CODE/lib:$REPO/src:$CODE OMP_NUM_THREADS=1 $VENV/bin/python"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)"; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
bash $CODE/jobs/frame_k1.sh
