#!/bin/bash
#SBATCH --job-name=frame_arm
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=$U/exp/FRAME/logs/%x_%j.out
# One C1/C3 arm on CLUSTER-A (A30, 1 GPU per arm as the Route A FT arms).  Usage (all arm knobs via --export):
#   sbatch -J frame_c3_p01 --export=ALL,ARM=c3_p01_identity_replay,FRAC=p01,FRAME=identity,REPLAY=1 frame_arm_clusterA.sbatch
#   sbatch -J frame_c1_p01 --export=ALL,ARM=c1_p01_vt_replay,FRAC=p01,FRAME=vt,REPLAY=1 frame_arm_clusterA.sbatch
#   sbatch -J frame_c1_p01n --export=ALL,ARM=c1_p01_vt_noreplay,FRAC=p01,FRAME=vt,REPLAY=0 frame_arm_clusterA.sbatch
#   optional: QU=0.5 | QU_END=0.5 CUR_START=7000 | GATE_STEP= (disable) | TRAIN_SEED=... | SMOKE_UNITS=8 STEPS=20 SAVE=10 GATE_STEP=20
# Requeue-safe: e0_train.py auto-resumes from the run dir's latest checkpoint; finished sampling/scoring files are skipped.
set -uo pipefail; exec 2>&1
TAG=5dd59e88
B=$U; IMG=$B/images/torch27.sif; VENV=$B/venvs/e0; REPO=$B/repos/nagamatsu
export CODE=$B/exp/FRAME/code_$TAG STAGE=$B RUNROOT=$B/exp/FRAME/runs PROCS=${SLURM_CPUS_PER_TASK:-16}
: "${ARM:?set ARM via --export}" "${FRAC:?}" "${FRAME:?}" "${REPLAY:?}"
mkdir -p $B/exp/FRAME/logs $RUNROOT
export PYG="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$CODE/lib:$REPO/src:$CODE OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled $VENV/bin/python"
export PYC="apptainer exec -B $SCRATCH_ROOT $IMG env PYTHONPATH=$CODE/lib:$REPO/src:$CODE OMP_NUM_THREADS=1 $VENV/bin/python"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} arm $ARM start $(date -Is)"; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
bash $CODE/jobs/frame_arm.sh
