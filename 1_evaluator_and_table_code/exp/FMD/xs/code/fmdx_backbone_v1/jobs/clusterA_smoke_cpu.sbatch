#!/bin/bash
#SBATCH -J fmdx_bb_smoke_cpu
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH -o $U/exp/FMD/xs/logs/build/smoke_cpu_%j.out
# CPU-only logic smoke (1 process, no AMP, no DDP); the GPU smoke (clusterA_smoke.sbatch / clusterB_smoke.sbatch) covers DDP + AMP.
set -u
export FMDX_SITE=clusterA CUDA_VISIBLE_DEVICES=
bash $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/smoke.sh $U/exp/FMD/xs/runs/smoke/clusterA_cpu_$SLURM_JOB_ID clusterA ${1:-}
