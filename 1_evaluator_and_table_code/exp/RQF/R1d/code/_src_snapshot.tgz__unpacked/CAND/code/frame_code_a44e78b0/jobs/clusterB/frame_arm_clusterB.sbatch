#!/bin/bash
#SBATCH -J frame_arm
#SBATCH -A <account>
#SBATCH -p A100-RP,A100-80GB,H100-RP,H100,H100-PCI,H200-SDS
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U_B/exp/FRAME/logs/%x_%j.out
# One C1/C3 arm on CLUSTER-B <site> (Pyxis/enroot; venvs/e0 and the nagamatsu repo as placed for LB2).  Same --export knobs as the
# CLUSTER-A file: ARM FRAC FRAME REPLAY [QU QU_END CUR_START GATE_STEP TRAIN_SEED SMOKE_UNITS STEPS SAVE].  Data under
# $U_B with the CLUSTER-A-relative layout (data/m5_multiscript_v1, data/ouhdl_e0_v3, data/d2_v2/unipen_de, exp/M5/runs/luo).
# K1 on CLUSTER-B: replace the last line by  bash $CODE/jobs/frame_k1.sh  with OUT=$D/exp/FRAME/k1 exported.
set -uo pipefail; exec 2>&1
TAG=5dd59e88
D=$U_B; IMG=$D/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh; NG=$D/repos/nagamatsu/src
export CODE=$D/exp/FRAME/code_$TAG STAGE=$D RUNROOT=$D/exp/FRAME/runs PROCS=${SLURM_CPUS_PER_TASK:-16}
: "${ARM:?set ARM via --export}" "${FRAC:?}" "${FRAME:?}" "${REPLAY:?}"
mkdir -p $D/exp/FRAME/logs $RUNROOT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CT="srun --ntasks=1 --container-image=$IMG --container-name=frametorch --container-mounts=$SCRATCH_B:$SCRATCH_B --container-workdir=$D/exp/FRAME"
export PYG="$CT env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$CODE/lib:$NG:$CODE OMP_NUM_THREADS=4 PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled $D/venvs/e0/bin/python"
export PYC="$CT env PYTHONPATH=$CODE/lib:$NG:$CODE OMP_NUM_THREADS=1 PYTHONHASHSEED=0 $D/venvs/e0/bin/python"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} partition ${SLURM_JOB_PARTITION:-} arm $ARM start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
$CT nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
bash $CODE/jobs/frame_arm.sh
